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
import scipy.io as sio
import scipy.misc as sm
#import skimage
#import skimage.transform
#from skimage import io

FLAGS = tf.app.flags.FLAGS

# ===========================
#   Data Parameters
# ===========================
tf.app.flags.DEFINE_string("dataset", "small","pick the dataset in data_root from which your movies can be found.")
tf.app.flags.DEFINE_string("data_root", "~/pilot_data", "Define the root folder of the different datasets.")
tf.app.flags.DEFINE_integer("num_threads", 4, "The number of threads for loading one minibatch.")

full_set = {}
im_size=(250,250,3)
de_size = (55,74)
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
  lst_runs = [ l.strip() for l in f.readlines() ]
  for run_dir in lst_runs:
    # print(run_dir)
    imgs_jpg=listdir(join(run_dir,'RGB'))
    # get list of all image numbers available in listdir
    num_imgs=sorted([int(im[0:-4]) for im in imgs_jpg])
    assert len(num_imgs)!=0 , IOError('no images in {0}: {1}'.format(run_dir,len(imgs_jpg)))
    if not isfile(join(run_dir,'RGB','{0:010d}.jpg'.format(num_imgs[-1]))):
      print('ERROR:',run_dir,' imgnum: ',num_imgs[-1])
    # parse control data  
    control_file = open(join(run_dir,'control_info.txt'),'r')
    control_file_list = control_file.readlines()
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
    
    # Add depth links if files exist
    depth_list = [] 
    if FLAGS.auxiliary_depth:
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
    set_list.append({'name':run_dir, 'num_imgs':num_imgs, 'controls':control_list, 'depths':depth_list})
  f.close()
  if len(set_list)==0:
    print('[data]: Failed to read {0}_set.txt from {1} in {2}.'.format(data_type, FLAGS.dataset, FLAGS.data_root))
  return set_list

def prepare_data(size, size_depth=(55,74)):
  global im_size, full_set, de_size, max_key, datasetdir
  '''Load lists of tuples refering to images from which random batches can be drawn'''
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
    for batch_num in range(FLAGS.batch_size):
      # choose random index over all runs:
      run_ind = random.choice(range(len(data_set)))
      # choose random index over image numbers:
      frame_ind = random.choice(range(len(data_set[run_ind]['num_imgs'])))
      if FLAGS.n_fc:
        frame_ind = random.choice(range(len(data_set[run_ind]['num_imgs'])-FLAGS.n_frames))
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
            img = Image.open(img_file)
            img = sm.imresize(img,im_size,'nearest').astype(float) #.astype(np.float32)
            assert len(img) != 0, '[data] Loading image failed: {}'.format(img_file)
            de = []
            if FLAGS.auxiliary_depth:
              try:
                depth_file = join(data_set[run_ind]['name'],'Depth', '{0:010d}.jpg'.format(data_set[run_ind]['depths'][frame_ind]))
              except:
                pass
              else:
                de = Image.open(depth_file)
                de = sm.imresize(de,de_size,'nearest')
                de = de * 1/255. * 5.
            return img, de
          if FLAGS.n_fc: #concatenate features
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
  FLAGS.auxiliary_depth = True
  FLAGS.n_fc = False
  FLAGS.n_frames = 3
  FLAGS.random_seed = 123
  FLAGS.batch_size = 32

  prepare_data((240,320,3))

  print 'run_dir: {}'.format(full_set['train'][0]['name'])
  print 'len images: {}'.format(len(full_set['train'][0]['num_imgs']))
  print 'len control: {}'.format(len(full_set['train'][0]['controls']))
  print 'len depth: {}'.format(len(full_set['train'][0]['depths']))
  
  # import pdb; pdb.set_trace()
  
  start_time=time.time()
  for index, ok, batch in generate_batch('train'):
    print ('rgb value: {0:0.1f}, depth value: {1:0.4f}'.format(batch[0]['img'][0,0,0], batch[0]['depth'][0,0]))
    # break

  print('loading time one episode: {}'.format(tools.print_dur(time.time()-start_time)))
  
