#!/usr/bin/python
# Block all numpy-scipy incompatibility warnings (could be removed at following scipy update (>1.1))
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

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

# import h5py

FLAGS=None

full_set = {}
im_size=(128,128,3)
de_size = (55,74)

"""
Load data from hdf5 file (--hdf5) or from the dataset folder that contains a train_set.txt val_set.txt and test_set.txt.
In these text file are absolute paths defined towards the data. 

You can read in the full dataset in RAM which allows for very fast training (10s per 100 batches of 64 samples) or you can 
read them in during training (100s per 100batches) (--load_data_in_ram).

The script has following functions:
- prepare_data
- generate_batch
- load_set_in_ram
- load_set
- load_set_hdf5
"""
def prepare_data(_FLAGS, size, size_depth=(55,74)):
  global FLAGS, im_size, full_set, de_size, max_key, datasetdir
  '''Load lists of tuples refering to images from which random batches can be drawn'''
  FLAGS=_FLAGS
  random.seed(FLAGS.random_seed)

  im_size=size
  de_size = size_depth

  if FLAGS.data_root[0] != '/':  # 2. Pilot_data directory for saving data
    FLAGS.data_root=os.environ['HOME']+'/'+FLAGS.data_root
  # if FLAGS.data_root == "~/pilot_data": FLAGS.data_root=os.path.join(os.getenv('HOME'),'pilot_data')
  datasetdir = join(FLAGS.data_root, FLAGS.dataset)
  
  train_set = load_set('train') #if not FLAGS.hdf5 else load_set_hdf5('train')
  val_set=load_set('val') #if not FLAGS.hdf5 else load_set_hdf5('val')
  test_set=load_set('test')
  full_set={'train':train_set, 'val':val_set, 'test':test_set}

  
# def load_set_hdf5(data_type):
#   """Load a type (train, val or test) of set in the set_list
#   as a tuple: first tuple element the directory of the fligth 
#   and the second the number of images taken in that flight
#   """
#   set_list = []
#   if not os.path.exists(join(datasetdir, data_type+'_set.txt')):
#     print('Datatype {0} not available for dataset {1}.'.format(data_type, datasetdir))
#     return []

#   # open text file with list of data directories corresponding to the datatype
#   f = open(join(datasetdir, data_type+'_set.txt'), 'r')
#   runs={} #dictionary with for each parent directory a list of all folders related to this datatype
#   for r in sorted([ l.strip() for l in f.readlines() if len(l) > 2]):
#     if os.path.dirname(r) in runs.keys():
#       runs[os.path.dirname(r)].append(os.path.basename(r))  
#     else :
#       runs[os.path.dirname(r)] = [os.path.basename(r)]     
#   for directory in runs.keys():
#     data_file = h5py.File(directory+'/data.hdf5', 'r')
#     for run in data_file.keys():
#       set_list.append({'name':directory+'/'+str(run), 
#                        'images':data_file[run]['RGB'][:],
#                        'depths':data_file[run]['Depth'][:],
#                        'controls':data_file[run]['control_info'][:]})
#     data_file.close()
#   f.close()
#   if len(set_list)==0:
#     print('[data]: Failed to read {0}_set.txt from {1} in {2}.'.format(data_type, FLAGS.dataset, FLAGS.data_root))
#   return set_list

def load_set(data_type):
  """Load a type (train, val or test) in the set_list
  as a dictionary {name, controls, num_imgs, num_depths}
  with multiple threads. In case of --load_data_in_ram, the imgs and depths are loaded as well.
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

def load_run_info(coord, run_list, set_list, checklist):
  """Load information from run with multiple threads"""
  while not coord.should_stop():
    try:
      run_dir = run_list.pop()
      print run_dir
      # get list of all image numbers available in listdir
      imgs_jpg=[f for f in listdir(join(run_dir,'RGB')) if not f.startswith('.')]
      num_imgs=sorted([int(im[0:-4]) for im in imgs_jpg[::FLAGS.subsample]])
      # print("{}".format(num_imgs))
      assert len(num_imgs)!=0 , IOError('no images in {0}: {1}'.format(run_dir,len(imgs_jpg)))
      if not isfile(join(run_dir,'RGB','{0:010d}.jpg'.format(num_imgs[-1]))):
        print('ERROR:',run_dir,' imgnum: ',num_imgs[-1])
      # parse control data  
      ctr_file = open(join(run_dir,FLAGS.control_file),'r')
      control_file_list = ctr_file.readlines()[2:]
      # cut last lines to avoid emtpy lines
      while len(control_file_list[-1])<=1 : control_file_list=control_file_list[:-1]
      control_parsed = [(int(ctr.strip().split(' ')[0]),float(ctr.strip().split(' ')[6])) for ctr in control_file_list]
      # print("{}".format(control_parsed))
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
      if FLAGS.load_data_in_ram:
        for num in num_imgs:
          img_file = join(run_dir,'RGB', '{0:010d}.jpg'.format(num))
          img = sio.imread(img_file)
          scale_height = int(np.floor(img.shape[0]/im_size[0]))
          scale_width = int(np.floor(img.shape[1]/im_size[1]))
          img = img[::scale_height,::scale_width]
          img = sm.resize(img,im_size,mode='constant').astype(float)
          assert len(img) != 0, '[data] Loading image failed: {}'.format(img_file)
          imgs.append(img)
      # Add depth links if files exist
      depth_list = [] 
      try:
        depths_jpg=listdir(join(run_dir,FLAGS.depth_directory))
        if len(depths_jpg)==0: 
          raise OSError('Depth folder is empty') 
      except OSError as e:
        # print('Failed to find Depth directory of: {0}. \n {1}'.format(run_dir, e))
        pass
      else:
        num_depths=sorted([int(de[0:-4]) for de in depths_jpg[::FLAGS.subsample]])
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
        # print("{}".format(num_imgs))
        # print("{}".format(control_list))
        # print("{}".format(depth_list))
        assert len(num_imgs) == len(depth_list), "Length of input(images,control,depth) is not equal"

      # Load depth in RAM and preprocess
      depths=[]
      if FLAGS.load_data_in_ram:
        for num in depth_list:
          depth_file = join(run_dir,'Depth', '{0:010d}.jpg'.format(num))
          de = sio.imread(depth_file)
          scale_height = int(np.floor(de.shape[0]/de_size[0]))
          scale_width = int(np.floor(de.shape[1]/de_size[1]))
          de = de[::scale_height,::scale_width]
          de = sm.resize(de,de_size,order=1,mode='constant', preserve_range=True)
          de[de<10]=0
          de = de * (1/255. * 5.)
          # clip to minimum and maximum depth
          de = np.minimum(np.maximum(de, FLAGS.min_depth),FLAGS.max_depth)
          assert len(de) != 0, '[data] Loading depth failed: {}'.format(depth_file)
          depths.append(de)

      # add all data to the dataset
      set_list.append({'name':run_dir, 'controls':control_list, 'num_imgs':num_imgs, 'imgs':imgs, 'num_depths':depth_list, 'depths':depths})
      
    except IndexError as e:
      coord.request_stop()
    except Exception as e:
      print('Problem in loading data: {0} @ {1}'.format(e.message, run_dir))
      checklist.append(False)
      coord.request_stop()

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
  # max_num_of_batch = {'train':1, 'val':1, 'test':1000}
  number_of_batches = min(int(number_of_frames/FLAGS.batch_size),max_num_of_batch[data_type])
  if number_of_batches == 0:  
    print('Only {0} frames to fill {1} batch size, so set batch_size to {2}'.format(number_of_frames, FLAGS.batch_size, number_of_frames))
    FLAGS.batch_size = number_of_frames
  
  b=0
  while b < number_of_batches:
    if b>0 and b%10==0:
      print('batch {0} of {1}'.format(b,number_of_batches))
    ok = True
    
    # Sample indices from dataset
    # from which threads can start loading
    batch=[]
    batch_indices = []
    # checklist keeps track for each batch whether all the loaded data was loaded correctly
    checklist = []
    # list of samples to fill batch for threads to know what sample to load
    stime=time.time()
    
    # keep track of the distribution of controls
    count_controls={-1:0, 0:0, 1:0}
    batch_num=0
    while batch_num < FLAGS.batch_size:
      # choose random index over all runs:
      run_ind = random.choice(range(len(data_set)))
      options=range(len(data_set[run_ind]['num_imgs']) if not '_nfc' in FLAGS.network else len(data_set[run_ind]['num_imgs'])-FLAGS.n_frames)
      if FLAGS.normalize_over_actions and count_controls[0] >= FLAGS.batch_size/3.: # if '0' is full
        # take out all frames with 0 in control
        options = [i for i in options if np.abs(data_set[run_ind]['controls'][i]) > 0.3]
      if FLAGS.normalize_over_actions and count_controls[-1] >= FLAGS.batch_size/3.: # if '-1' is full
        # take out all frames with -1 in control
        options = [i for i in options if not (np.abs(data_set[run_ind]['controls'][i]) > 0.3 and np.sign(data_set[run_ind]['controls'][i])==-1)]
      if FLAGS.normalize_over_actions and count_controls[1] >= FLAGS.batch_size/3.: # if '-1' is full
        # take out all frames with -1 in control
        options = [i for i in options if not (np.abs(data_set[run_ind]['controls'][i]) > 0.3 and np.sign(data_set[run_ind]['controls'][i])==+1)]
      if not len(options) == 0: # in case there are still frames left...
        frame_ind = random.choice(options)
        batch_num += 1
        batch_indices.append((batch_num, run_ind, frame_ind))
        ctr = data_set[run_ind]['controls'][frame_ind]
        ctr_index = 0 if np.abs(ctr) < 0.3 else np.sign(ctr)
        count_controls[ctr_index]+=1
    
    # print count_controls
    if not FLAGS.load_data_in_ram:
      # load data multithreaded style into RAM
      def load_image_and_target(coord, batch_indices, batch, checklist):
        while not coord.should_stop():
          try:
            loc_ind, run_ind, frame_ind = batch_indices.pop()
            def load_rgb_depth_image(run_ind, frame_ind):
              # load image
              img_file = join(data_set[run_ind]['name'],'RGB', '{0:010d}.jpg'.format(data_set[run_ind]['num_imgs'][frame_ind]))
              # print('img_file {}'.format(img_file))
              # img = Image.open(img_file)
              img = sio.imread(img_file)
              scale_height = int(np.floor(img.shape[0]/im_size[0]))
              scale_width = int(np.floor(img.shape[1]/im_size[1]))
              img = img[::scale_height,::scale_width]
              img = sm.resize(img,im_size,mode='constant').astype(float) #.astype(np.float32)
              assert len(img) != 0, '[data] Loading image failed: {}'.format(img_file)
              de = []
              try:
                depth_file = join(data_set[run_ind]['name'],'Depth', '{0:010d}.jpg'.format(data_set[run_ind]['num_depths'][frame_ind]))
              except:
                pass
              else:
                # de = Image.open(depth_file)
                de = sio.imread(depth_file)
                scale_height = int(np.floor(de.shape[0]/de_size[0]))
                scale_width = int(np.floor(de.shape[1]/de_size[1]))
                de = de[::scale_height,::scale_width]
                # clip depth image with small values as they are due to image processing
                de = sm.resize(de,de_size,order=1,mode='constant', preserve_range=True)
                de[de<10]=0
                de = de * (1/255. * 5.)
                de = np.minimum(np.maximum(de, FLAGS.min_depth),FLAGS.max_depth)

                if len(de) == 0: print('failed loading depth image: {0} from {1}'.format(data_set[run_ind]['num_depths'][frame_ind], data_set[run_ind]['name']))
              return img, de
            if '_nfc' in FLAGS.network:
              ims = []
              for frame in range(FLAGS.n_frames):
                # target depth (de) is each time overwritten, only last frame is kept
                image, de = load_rgb_depth_image(run_ind, frame_ind+frame)
                ims.append(image)
              im = np.concatenate(ims, axis=2)
              ctr = data_set[run_ind]['controls'][frame_ind+FLAGS.n_frames-1]
            else:
              im, de = load_rgb_depth_image(run_ind, frame_ind)
              ctr = data_set[run_ind]['controls'][frame_ind]

            # clip control avoiding values larger than 1
            ctr=max(min(ctr,FLAGS.action_bound),-FLAGS.action_bound)
              
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
    else:
      # just combine the data in a batch
      for batch_num, run_ind, frame_ind in batch_indices:
        img = data_set[run_ind]['imgs'][frame_ind]
        try:
          depth = data_set[run_ind]['depths'][frame_ind]
        except:
          depth=[]
          # print("[data.py]: Problem loading depth in batch.")
          pass
        ctr = data_set[run_ind]['controls'][frame_ind]
        # clip control avoiding values larger than 1
        ctr=max(min(ctr,FLAGS.action_bound),-FLAGS.action_bound)
        
        # append rgb image, control and depth to batch. Use scan if it is loaded, else depth
        batch.append({'img':img, 'ctr':ctr, 'depth': depth})
        ok=True
    if ok: b+=1
    yield b, ok, batch
    

#### FOR TESTING ONLY
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test reading in the offline data.')

  parser.add_argument("--dataset", default="small", type=str, help="pick the dataset in data_root from which your movies can be found.")
  parser.add_argument("--data_root", default="pilot_data/",type=str, help="Define the root folder of the different datasets.")
  parser.add_argument("--control_file", default="control_info.txt",type=str, help="Define text file with logged control info.")
  parser.add_argument("--num_threads", default=4, type=int, help="The number of threads for loading one minibatch.")
  parser.add_argument("--action_bound", default=1, type=float, help="Bound the action space between -b and b")
  parser.add_argument("--auxiliary_depth", action='store_true', help="Define wether network is trained with auxiliary depth prediction.")
  # parser.add_argument("--n_fc", action='store_true', help="Define wether network uses 3 concatenated consecutive frames.")
  parser.add_argument("--hdf5", action='store_true', help="Define wether dataset is hdf5 type. [not working in singularity]")
  parser.add_argument("--load_data_in_ram", action='store_true', help="Define wether dataset is loaded into RAM.")
  parser.add_argument("--min_depth", default=0.0, type=float, help="clip depth loss with weigths to focus on correct depth range.")
  parser.add_argument("--max_depth", default=5.0, type=float, help="clip depth loss with weigths to focus on correct depth range.")
  
  parser.add_argument("--subsample", default=1, type=int, help="Subsample data over time: e.g. subsample 2 to get from 20fps to 10fps.")
  parser.add_argument("--depth_directory", default='Depth', type=str, help="Define the name of the directory containing the depth images: Depth or Depth_predicted.")

  parser.add_argument("--network",default='mobile',type=str, help="Define the type of network: depth_q_net, coll_q_net.")
  parser.add_argument("--random_seed", default=123, type=int, help="Set the random seed to get similar examples.")
  parser.add_argument("--batch_size",default=64,type=int,help="Define the size of minibatches.")
  parser.add_argument("--normalize_over_actions", action='store_true', help="Try to fill a batch with different actions [-1, 0, 1].")
  
  FLAGS=parser.parse_args()  

  prepare_data(FLAGS, (128,128,3))

  for dt in 'train', 'val', 'test':
    print("------------------------")
    print("Datatype: {}".format(dt))
    print("Number of runs: {}".format(len(full_set[dt])))
    print("Number of images: {}".format(sum([ len(s['num_imgs']) for s in full_set[dt]])))
    print("Number of depths: {}".format(sum([ len(s['num_depths']) for s in full_set[dt]])))
    print("Number of controls: {}".format(sum([ len(s['controls']) for s in full_set[dt]])))
  
  start_time=time.time()
  for index, ok, batch in generate_batch('train'):
    print("Batch: {}".format(index))
    inputs = np.array([_['img'] for _ in batch])
    actions = np.array([[_['ctr']] for _ in batch])
    target_depth = np.array([_['depth'] for _ in batch]).reshape((-1,55,74)) if FLAGS.auxiliary_depth else []
    
    print("batchsize: {}".format(len(batch)))
    print("images size: {}".format(inputs.shape))
    print("actions size: {}".format(actions.shape))
    print("depths size: {}".format(target_depth.shape if FLAGS.auxiliary_depth else 0))

    # import pdb; pdb.set_trace()  
    
  print('loading time one episode: {}'.format(tools.print_dur(time.time()-start_time)))
  
