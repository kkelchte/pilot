#!/usr/bin/python
import os, sys
import numpy as np
import tensorflow as tf
import threading
from os import listdir
from os.path import isfile, join, isdir
import time
import random
from math import floor
import tools
from PIL import Image

import skimage.io as sio
import skimage.transform as sm
import matplotlib.pyplot as plt

import argparse

FLAGS=None

full_set = {}
im_size=(128,128,3)
de_size = (55,74)
collision_num=10 # number of frames before collision labelled with 10

def prepare_data(_FLAGS, size, size_depth=(55,74)):
  global FLAGS, im_size, full_set, de_size, max_key, datasetdir, collision_num
  '''Load lists of tuples refering to images from which random batches can be drawn'''
  FLAGS=_FLAGS
  # stime = time.time()
  # some startup settings
  # np.random.seed(FLAGS.random_seed)
  # tf.set_random_seed(FLAGS.random_seed)
  random.seed(FLAGS.random_seed)

  if FLAGS.data_root == "~/pilot_data": FLAGS.data_root=os.path.join(os.getenv('HOME'),'pilot_data')
  datasetdir = join(FLAGS.data_root, FLAGS.dataset)
  
  train_set = load_set('train')
  val_set=load_set('val')
  test_set=load_set('test')
  full_set={'train':train_set, 'val':val_set, 'test':test_set}

  print("Loaded data: {0} training, {1} validation, {2} test.".format(len(train_set),len(val_set),len(test_set)))

  im_size=size
  try:
    collision_num = int(FLAGS.collision_file.split('.')[0].split('_')[2])
  except:
    collision_num = 5
  
def load_run_info(coord, run_list, set_list, checklist):
  """Load information from run with multiple threads"""
  while not coord.should_stop():
    try:
      run_dir = run_list.pop()
      print run_dir
      # get list of all image numbers available in listdir
      imgs_jpg=listdir(join(run_dir,'RGB'))
      num_imgs=sorted([int(im[0:-4]) for im in imgs_jpg[::FLAGS.subsample]])
      assert len(num_imgs)!=0 , IOError('no images in {0}: {1}'.format(run_dir,len(imgs_jpg)))
      if not isfile(join(run_dir,'RGB','{0:010d}.jpg'.format(num_imgs[-1]))):
        print('ERROR:',run_dir,' imgnum: ',num_imgs[-1])
      # parse control data  
      ctr_file = open(join(run_dir,FLAGS.control_file),'r')
      control_file_list = ctr_file.readlines()[2:]
      # cut last lines to avoid emtpy lines
      while len(control_file_list[-1])<=1 : control_file_list=control_file_list[:-1]
      control_parsed = [(int(ctr.strip().split(' ')[0]),float(ctr.strip().split(' ')[6])) for ctr in control_file_list]
      def sync_control():
        control_list = []
        corresponding_imgs = []
        ctr_ind, ctr_val = control_parsed.pop(0)
        for ni in num_imgs:
          while(ctr_ind < ni):
            try:
              ctr_ind, ctr_val = control_parsed.pop(0)
            except (IndexError): # In case control has no more lines though RGB has still images, stop anyway:
              return corresponding_imgs, control_list
          # clip at -1 and 1
          if abs(ctr_val) > 1: ctr_val = np.sign(ctr_val)
          control_list.append(ctr_val)
          corresponding_imgs.append(ni)
        return corresponding_imgs, control_list
      num_imgs, control_list = sync_control()
      assert len(num_imgs) == len(control_list), "Length of number of images {0} is not equal to number of control {1}".format(len(num_imgs),len(control_list))
      
      # Load images in RAM and preprocess
      imgs=[]
      for num in num_imgs:
        img_file = join(run_dir,'RGB', '{0:010d}.jpg'.format(num))
        img = sio.imread(img_file)
        scale_height = int(np.floor(img.shape[0]/im_size[0]))
        scale_width = int(np.floor(img.shape[1]/im_size[1]))
        img = img[::scale_height,::scale_width]
        img = sm.resize(img,im_size,mode='constant').astype(float)
        assert len(img) != 0, '[data] Loading image failed: {}'.format(img_file)
        imgs.append(img)

      # Create collision list
      collision_list=[]
      if FLAGS.network == 'coll_q_net':
        if not os.path.isfile(join(run_dir,FLAGS.collision_file)):
          print("error, could not find collision file in {}".format(run_dir))
        coll_file = open(join(run_dir,FLAGS.collision_file),'r')
        coll_file_list = coll_file.readlines()
        while len(coll_file_list[-1])<=1 : coll_file_list=coll_file_list[:-1]
        collision_dict={int(coll.strip().split(' ')[0]):int(coll.strip().split(' ')[1]) for coll in coll_file_list}
        collision_list=[collision_dict[num_img] for num_img in num_imgs]
        assert len(num_imgs) == len(collision_list), "Length of number of images {0} is not equal to number of collision labels {1}".format(len(num_imgs),len(collision_list))
     
      # Add scan information if file exist
      scan_list = []
      if os.path.isfile(join(run_dir,'scan.txt')):
        if not os.path.isfile(join(run_dir,'scan.txt')):
          print("error, could not find scan file in {}".format(run_dir))
        scans = open(join(run_dir,'scan.txt'),'r').readlines()[2:]
        assert(len(scans) != 0)
        ranges = np.zeros((len(scans),int(FLAGS.field_of_view/FLAGS.smooth_scan)))
        for si,s in enumerate(scans):
          def check(r):
            """clip at FLAGS.max_depth, set 0's to nan"""
            return min(r,FLAGS.max_depth) if r!=0 else np.nan
          # clip left FOV/2 range from 0:FOV/2 reversed with right FOV/2degree range from the last FOV/2:
          scan=list(reversed([ check(float(r)) for r in s[12:-2].split(',')[0:FLAGS.field_of_view/2]]))+list(reversed([check(float(r)) for r in s[12:-2].split(',')[-FLAGS.field_of_view/2:]]))
          # add some smoothing by averaging over 4 neighboring bins
          raw_ranges=[np.nanmean(scan[i*FLAGS.smooth_scan:i*FLAGS.smooth_scan+FLAGS.smooth_scan]) for i in range(int(len(scan)/FLAGS.smooth_scan))]
          # in case of nan --> set value to -1 as it will be ignored by the loss.
          ranges[si]=[r if not np.isnan(r) else -1 for r in raw_ranges]
        scan_list=np.asarray(ranges)
      # add all data to the dataset
      set_list.append({'name':run_dir, 'num_imgs':num_imgs, 'controls':control_list, 'imgs':imgs, 'collisions':collision_list, 'scans':scan_list})
      
    except IndexError as e:
      coord.request_stop()
    except Exception as e:
      print('Problem in loading data: {0} @ {1}'.format(e, run_dir))
      checklist.append(False)
      coord.request_stop()
    
def load_set(data_type):
  """Load a type (train, val or test) of set in the set_list
  as a tuple: first tuple element the directory of the fligth 
  and the second the number of images taken in that flight
  """
  if not os.path.exists(join(datasetdir, data_type+'_set.txt')):
    print('Datatype {0} not available for dataset {1}.'.format(data_type, datasetdir))
    return []
  f = open(join(datasetdir, data_type+'_set.txt'), 'r')
  run_list = [ l.strip() for l in f.readlines() if len(l) > 2]
  f.close() 
  set_list = []
  checklist = []
  try:
    coord=tf.train.Coordinator()
    threads = [threading.Thread(target=load_run_info, args=(coord, run_list, set_list, checklist)) for i in range(FLAGS.num_threads)]
    for t in threads: t.start()
    coord.join(threads, stop_grace_period_secs=30)
  except RuntimeError as e:
    print("threads are not stopping...",e)
  else:
    if len(checklist) != sum(checklist):
      ok=False
      print('[data]: Failed to read {0}_set.txt from {1} in {2}.'.format(data_type, FLAGS.dataset, FLAGS.data_root))
  return set_list

def generate_batch(data_type):
  """ 
  input:
    data_type: 'train', 'val' or 'test'
  Generator object that gets a random batch when next() is called
  yields: 
  - index: of current batch relative to the data seen in this epoch.
  - ok: boolean that defines if batch was loaded correctly
  - imb: batch of input rgb images
  - trgb: batch of corresponding control targets
  - auxb: batch with auxiliary info
  """
  data_set=full_set[data_type]
  number_of_frames = sum([len(run['num_imgs']) for run in data_set])
  # When there is that much data applied that you can get more than 100 minibatches out
  # stick to 100, otherwise one epoch takes too long and the training is not updated
  # regularly enough.
  max_num_of_batch = {'train':100, 'val':10, 'test':1000}
  number_of_batches = min(int(number_of_frames/FLAGS.batch_size),max_num_of_batch[data_type])
  if number_of_batches == 0:
    print('Only {0} frames to fill {1} batch size, so set batch_size to {2}'.format(number_of_frames, FLAGS.batch_size, number_of_frames))
  b=0
  while b < number_of_batches:
    if b>0 and b%10==0:
      print('batch {0} of {1}'.format(b,number_of_batches))
    ok = True
    
    #Multithreaded implementation
    # sample indices from dataset
    # from which threads can start loading
    batch=[]
    batch_indices = []
    # checklist keeps track for each batch whether all the loaded data was loaded correctly
    checklist = []
    # list of samples to fill batch for threads to know what sample to load
    stime=time.time()
    
    count_tags={0:0,1:0}
    for batch_item in range(FLAGS.batch_size):
      # choose random index over all runs:
      run_ind = random.choice(range(len(data_set)))
      if FLAGS.normalize_data: 
        if FLAGS.network == "coll_q_net":
          # import pdb; pdb.set_trace()
          if count_tags[0] >= FLAGS.batch_size /2:
            frame_ind = random.choice([ i for i in range(len(data_set[run_ind]['collisions'])) if data_set[run_ind]['collisions'][i]==1])
            # print("ensure next frames are of type 1 to balance the 0 and 1 labels")
            # for i in range(len(data_set[run_ind]['num_imgs'])-1)[-8:]:
            #   print data_set[run_ind]['collisions'][i]
            # frame_ind = random.choice(range(len(data_set[run_ind]['num_imgs'])-1)[-(collision_num-1):])
            # assert data_set[run_ind]['collisions'][frame_ind] == 1, "collision is %d instead of 1 \n run: %d , frame: %d" % (data_set[run_ind]['collisions'][frame_ind], run_ind, frame_ind)
          elif count_tags[1] >= FLAGS.batch_size /2:
            while len(data_set[run_ind]['num_imgs']) < collision_num+1: 
              run_ind = random.choice(range(len(data_set)))
            frame_ind = random.choice([ i for i in range(len(data_set[run_ind]['collisions'])) if data_set[run_ind]['collisions'][i]==0])
            
            # print("ensure next frames are of type 0 to balance the 0 and 1 labels")
            # for i in range(len(data_set[run_ind]['num_imgs'])-1)[:-10]:
            #   print data_set[run_ind]['collisions'][i]
            # frame_ind = random.choice(range(len(data_set[run_ind]['num_imgs'])-1)[:-(collision_num)])
            # assert data_set[run_ind]['collisions'][frame_ind] == 0
          else:
            # choose random index over image numbers:
            frame_ind = random.choice(range(len(data_set[run_ind]['num_imgs'])-1))
          count_tags[data_set[run_ind]['collisions'][frame_ind]]+=1
        elif FLAGS.network == "depth_q_net":
          frame_ind = random.choice(range(len(data_set[run_ind]['num_imgs'])-1))
        else:
          raise NotImplementedError("[data.py]: normalization offline not implemented for network {}".format(FLAGS.network))
      else:
        # choose random index over image numbers: (-1 because future depth becomes target label)
        frame_ind = random.choice(range(len(data_set[run_ind]['num_imgs'])-FLAGS.future_steps))
      
      # if FLAGS.n_fc:
      #   frame_ind = random.choice(range(len(data_set[run_ind]['num_imgs'])-FLAGS.n_frames))
      batch_indices.append((run_ind, frame_ind))
    
    for run_ind, frame_ind in batch_indices:
      img = data_set[run_ind]['imgs'][frame_ind]
      scan = []
      try: #load next scan 
        scan = data_set[run_ind]['scans'][frame_ind+FLAGS.future_steps]
      except:
        pass

      ctr = data_set[run_ind]['controls'][frame_ind]
      # clip control avoiding values larger than 1
      ctr=max(min(ctr,FLAGS.action_bound),-FLAGS.action_bound)
      
      if FLAGS.network == 'coll_q_net': 
        col = data_set[run_ind]['collisions'][frame_ind]
        batch.append({'img':img, 'ctr':ctr, 'trgt':col})
      else:
        # append rgb image, control and depth to batch. Use scan if it is loaded, else depth
        batch.append({'img':img, 'ctr':ctr, 'depth': scan})
    b+=1
    ok=True    
    yield b, ok, batch
    
#### FOR TESTING ONLY
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test reading in the offline data.')

  parser.add_argument("--normalize_data", action='store_true', help="Define wether the collision tags 0 or 1 are normalized in a batch.")
  parser.add_argument("--dataset", default="canyon_turtle_scan_pruned", type=str, help="pick the dataset in data_root from which your movies can be found.")
  parser.add_argument("--data_root", default="~/pilot_data",type=str, help="Define the root folder of the different datasets.")
  parser.add_argument("--num_threads", default=4, type=int, help="The number of threads for loading one minibatch.")
  parser.add_argument("--action_bound", default=1, type=float, help="Bound the action space between -b and b")

  parser.add_argument("--network",default='depth_q_net',type=str, help="Define the type of network: depth_q_net, coll_q_net.")
  parser.add_argument("--random_seed", default=123, type=int, help="Set the random seed to get similar examples.")
  parser.add_argument("--batch_size",default=64,type=int,help="Define the size of minibatches.")
  parser.add_argument("--control_file",default='control_info.txt',type=str,help="Control file.")
  parser.add_argument("--depth_directory",default='Depth',type=str,help="Depth directory.")
  
  parser.add_argument("--field_of_view", default=104, type=int, help="The field of view of the camera cuts the depth scan in the range visible for the camera. Value should be even. Normal: 72 (-36:36), Wide-Angle: 120 (-60:60)")
  parser.add_argument("--smooth_scan", default=4, type=int, help="The 360degrees scan has a lot of noise and is therefore smoothed out over 4 neighboring scan readings.")
  parser.add_argument("--max_depth", default=6, type=float, help="clip depth loss with weigths to focus on correct depth range.")
  parser.add_argument("--output_size",default=[1,26],type=int, nargs=2, help="Define the output size of the depth frame: 55x74 [drone], 1x26 [turtle], only used in case of depth_q_net.")

  # parser.add_argument("--collision_file",default='collision_info.txt',type=str,help="define file with collision labels")
  
  FLAGS=parser.parse_args()  

  prepare_data(FLAGS, (128,128,3))

  print 'run_dir: {}'.format(full_set['train'][0]['name'])
  print 'len images: {}'.format(len(full_set['train'][0]['num_imgs']))
  print 'len control: {}'.format(len(full_set['train'][0]['controls']))
  print 'len depth: {}'.format(len(full_set['train'][0]['scans']))
  print 'len collisions: {}'.format(len(full_set['train'][0]['collisions']))
  
  
  start_time=time.time()
  for index, ok, batch in generate_batch('train'):
    inputs = np.array([_['img'] for _ in batch])
    actions = np.array([[_['ctr']] for _ in batch])
    if FLAGS.network == 'depth_q_net':
      targets = np.array([_['depth'] for _ in batch]).reshape((-1,FLAGS.output_size[0],FLAGS.output_size[1]))
    else:
      targets = np.array([_['trgt'] for _ in batch]).reshape((-1,1))
    print '---------------------------inputs---------------------'
    print inputs.shape
    print '---------------------------actions---------------------'
    print actions.shape
    print '---------------------------targets---------------------'
    print targets.shape
    print targets    
    import pdb; pdb.set_trace()
    pass
    # actions=[_['ctr'] for _ in batch]
    # print("avg: {0},var: {1}".format(np.mean(actions), np.var(actions)))

    # if FLAGS.network =='coll_q_net':
    #   print ('rgb value: {0:0.1f}, depth value: {1:0.4f}, control: {2}, collision: {3}'.format(batch[0]['img'][0,0,0], batch[0]['depth'][0,0], batch[0]['ctr'], batch[0]['trgt']))
    # print ('rgb value: {0:0.1f}, depth value: {1:0.4f}, control: {2}'.format(batch[0]['img'][0,0,0], batch[0]['depth'][0,0], batch[0]['ctr']))
    # break

  print('loading time one episode: {}'.format(tools.print_dur(time.time()-start_time)))
  
  start_time=time.time()
  for index, ok, batch in generate_batch('train'):
    pass
  print('loading time one episode: {}'.format(tools.print_dur(time.time()-start_time)))

  start_time=time.time()
  for index, ok, batch in generate_batch('train'):
    pass
  print('loading time one episode: {}'.format(tools.print_dur(time.time()-start_time)))


