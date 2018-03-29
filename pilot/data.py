#!/usr/bin/python
import os
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
# import scipy.io as sio
# import scipy.misc as sm
import skimage.io as sio
import skimage.transform as sm
import matplotlib.pyplot as plt

import argparse

#import skimage
#import skimage.transform
#from skimage import io

# FLAGS = tf.app.flags.FLAGS

FLAGS=None

# ===========================
#   Data Parameters
# ===========================
full_set = {}
im_size=(250,250,3)
de_size = (55,74)
collision_num=10 # number of frames before collision labelled with 10

def load_set(data_type):
  """Load a type (train, val or test) of set in the set_list
  as a tuple: first tuple element the directory of the fligth 
  and the second the number of images taken in that flight
  """
  set_list = []
  if not os.path.exists(join(datasetdir, data_type+'_set.txt')):
    print('Datatype {0} not available for dataset {1}.'.format(data_type, datasetdir))
    return []

  f = open(join(datasetdir, data_type+'_set.txt'), 'r')
  lst_runs = [ l.strip() for l in f.readlines() if len(l) > 2]
  
  for run_dir in lst_runs:
    print(run_dir)
    
    imgs_jpg=listdir(join(run_dir,'RGB'))
    
    # get list of all image numbers available in listdir
    num_imgs=sorted([int(im[0:-4]) for im in imgs_jpg])
    assert len(num_imgs)!=0 , IOError('no images in {0}: {1}'.format(run_dir,len(imgs_jpg)))
    if not isfile(join(run_dir,'RGB','{0:010d}.jpg'.format(num_imgs[-1]))):
      print('ERROR:',run_dir,' imgnum: ',num_imgs[-1])
    # parse control data  
    ctr_file = open(join(run_dir,FLAGS.control_file),'r')
    control_file_list = ctr_file.readlines()
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
    
    # Create collision list
    collision_list=[]
    if FLAGS.network == 'coll_q_net':
      coll_file = open(join(run_dir,FLAGS.collision_file),'r')
      coll_file_list = coll_file.readlines()
      while len(coll_file_list[-1])<=1 : coll_file_list=coll_file_list[:-1]
      collision_dict={int(coll.strip().split(' ')[0]):int(coll.strip().split(' ')[1]) for coll in coll_file_list}
      collision_list=[collision_dict[num_img] for num_img in num_imgs]
      assert len(num_imgs) == len(collision_list), "Length of number of images {0} is not equal to number of collision labels {1}".format(len(num_imgs),len(collision_list))
   
    # Add depth links if files exist
    depth_list = [] 
    try:
      depths_jpg=listdir(join(run_dir,'Depth'))
      if len(depths_jpg)==0: raise OSError('Depth folder is empty') 
    except OSError as e:
      # print('Failed to find Depth directory of: {0}. \n {1}'.format(run_dir, e))
      pass
    else:
      num_depths=sorted([int(de[0:-4]) for de in depths_jpg])
      smallest_depth = num_depths.pop(0)
      for ni in num_imgs: #link the indices of rgb images with the smallest depth bigger than current index
        while(ni > smallest_depth):
          try:
            smallest_depth = num_depths.pop(0)
          except IndexError:
            break
        depth_list.append(smallest_depth)
      num_imgs = num_imgs[:len(depth_list)]
      control_list = control_list[:len(depth_list)]
      assert len(num_imgs) == len(depth_list), "Length of input(imags,control,depth) is not equal"    
    set_list.append({'name':run_dir, 'num_imgs':num_imgs, 'controls':control_list, 'depths':depth_list, 'collisions':collision_list})
  f.close()
  if len(set_list)==0:
    print('[data]: Failed to read {0}_set.txt from {1} in {2}.'.format(data_type, FLAGS.dataset, FLAGS.data_root))
  return set_list

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
  im_size=size
  de_size = size_depth
  # collision_info_3.txt --> collision_num = 3
  try:
    collision_num = int(FLAGS.collision_file.split('.')[0].split('_')[2])
  except:
    collision_num = 10

  
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
    for batch_num in range(FLAGS.batch_size):
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
        frame_ind = random.choice(range(len(data_set[run_ind]['num_imgs'])-1))
      
      # if FLAGS.n_fc:
      #   frame_ind = random.choice(range(len(data_set[run_ind]['num_imgs'])-FLAGS.n_frames))
      batch_indices.append((batch_num, run_ind, frame_ind))
    # print("picking random indices duration: ",time.time()-stime)
    def load_image_and_target(coord, batch_indices, batch, checklist):
      while not coord.should_stop():
        try:
          loc_ind, run_ind, frame_ind = batch_indices.pop()
          def load_rgb_depth_image(run_ind, frame_ind):
            # load image
            img_file = join(data_set[run_ind]['name'],'RGB', '{0:010d}.jpg'.format(data_set[run_ind]['num_imgs'][frame_ind]))
            # print('img_file ',img_file)
            # img = Image.open(img_file)
            img = sio.imread(img_file)
            img = img[::2,::5,:]
            img = sm.resize(img,im_size,mode='constant').astype(float) #.astype(np.float32)
            assert len(img) != 0, '[data] Loading image failed: {}'.format(img_file)
            de = []
            try:
              depth_file = join(data_set[run_ind]['name'],'Depth', '{0:010d}.jpg'.format(data_set[run_ind]['depths'][frame_ind+1]))
            except:
              pass
            else:
              # de = Image.open(depth_file)
              de = sio.imread(depth_file)
              de = de[::6,::8]
              # clip depth image with small values as they are due to image processing
              de = sm.resize(de,de_size,order=1,mode='constant', preserve_range=True)
              de[de<10]=0
              de = de * 1/255. * 5.
            return img, de
          # if FLAGS.n_fc: #concatenate features
          #   ims = []
          #   for frame in range(FLAGS.n_frames):
          #     # target depth (de) is each time overwritten, only last frame is kept
          #     image, de = load_rgb_depth_image(run_ind, frame_ind+frame)
          #     ims.append(image)
          #   im = np.concatenate(ims, axis=2)
          #   ctr = data_set[run_ind]['controls'][frame_ind+FLAGS.n_frames-1]
          # else:
          im, de = load_rgb_depth_image(run_ind, frame_ind)

              
          ctr = data_set[run_ind]['controls'][frame_ind]
          # clip control avoiding values larger than 1
          ctr=max(min(ctr,FLAGS.action_bound),-FLAGS.action_bound)
          # normalize control form -bound:+bound to 0 and 1
          if FLAGS.action_normalization: ctr=(ctr+FLAGS.action_bound)/(2.*FLAGS.action_bound) 
          if FLAGS.network == 'coll_q_net': 
            col = data_set[run_ind]['collisions'][frame_ind]
            batch.append({'img':im, 'ctr':ctr, 'depth':de, 'trgt':col})
          else:
            # append rgb image, control and depth to batch
            batch.append({'img':im, 'ctr':ctr, 'depth':de})
          checklist.append(True)
        except IndexError as e:
          # print(e)
          #print('batch_loaded, wait to stop', e)
          coord.request_stop()
        except Exception as e:
          print('Problem in loading data: ',e)
          checklist.append(False)
          coord.request_stop()
    try:
      coord=tf.train.Coordinator()
      #print(FLAGS.num_threads)
      threads = [threading.Thread(target=load_image_and_target, args=(coord, batch_indices, batch, checklist)) for i in range(FLAGS.num_threads)]
      for t in threads: t.start()
      coord.join(threads, stop_grace_period_secs=5)
    except RuntimeError as e:
      print("threads are not stopping...",e)
    else:
      if len(checklist) != sum(checklist): ok=False
    if ok: b+=1
    yield b, ok, batch
    
#### FOR TESTING ONLY
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test reading in the offline data.')

  parser.add_argument("--normalize_data", action='store_true', help="Define wether the collision tags 0 or 1 are normalized in a batch.")
  parser.add_argument("--dataset", default="canyon_rl_turtle_collision_free", type=str, help="pick the dataset in data_root from which your movies can be found.")
  parser.add_argument("--data_root", default="~/pilot_data",type=str, help="Define the root folder of the different datasets.")
  parser.add_argument("--num_threads", default=4, type=int, help="The number of threads for loading one minibatch.")
  parser.add_argument("--action_bound", default=1, type=float, help="Bound the action space between -b and b")

  parser.add_argument("--network",default='depth_q_net',type=str, help="Define the type of network: depth_q_net, coll_q_net.")
  parser.add_argument("--random_seed", default=123, type=int, help="Set the random seed to get similar examples.")
  parser.add_argument("--batch_size",default=64,type=int,help="Define the size of minibatches.")
  
  # parser.add_argument("--collision_file",default='collision_info.txt',type=str,help="define file with collision labels")
  
  FLAGS=parser.parse_args()  

  prepare_data(FLAGS, (240,320,3))

  print 'run_dir: {}'.format(full_set['train'][0]['name'])
  print 'len images: {}'.format(len(full_set['train'][0]['num_imgs']))
  print 'len control: {}'.format(len(full_set['train'][0]['controls']))
  print 'len depth: {}'.format(len(full_set['train'][0]['depths']))
  print 'len collisions: {}'.format(len(full_set['train'][0]['collisions']))
  
  
  start_time=time.time()
  for index, ok, batch in generate_batch('train'):
    actions=[_['ctr'] for _ in batch]
    print("avg: {0},var: {1}".format(np.mean(actions), np.var(actions)))

    # if FLAGS.network =='coll_q_net':
    #   print ('rgb value: {0:0.1f}, depth value: {1:0.4f}, control: {2}, collision: {3}'.format(batch[0]['img'][0,0,0], batch[0]['depth'][0,0], batch[0]['ctr'], batch[0]['trgt']))
    # print ('rgb value: {0:0.1f}, depth value: {1:0.4f}, control: {2}'.format(batch[0]['img'][0,0,0], batch[0]['depth'][0,0], batch[0]['ctr']))
    # break

  print('loading time one episode: {}'.format(tools.print_dur(time.time()-start_time)))
  
