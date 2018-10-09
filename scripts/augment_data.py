#!/usr/bin/python

# Block all numpy-scipy incompatibility warnings (could be removed at following scipy update (>1.1))
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import os,sys, time
import argparse
import shutil
import subprocess, shlex
import json

import skimage.io as sio
import matplotlib.pyplot as plt

import ou_noise

class bcolors:
  """ Colors to print in terminal with color!
  """
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

'''
augment_data.py: 
In a factorized control the network should solely focus on the relationship between the obstacle in the foreground and the annotated control. Just to be sure that the factor does not put attention on the background, the background is filled with noise.

The experiments are done with two types of noise:

1. Uniform noise over all three channels
2. OU Noise in x and y direction for each channel

Runs are collected from .txt files

All runs are saved in destination with incrementing index and a set.txt file.

exit code:
2: not enough success runs so shutting down.

'''
print("\n {0} Augment data.py: started.".format(time.strftime("%Y-%m-%d_%I%M")))

# 1. Parse arguments and make settings
parser = argparse.ArgumentParser(description='Clean up dataset collected by a group of recordings that loop over different runs.')
parser.add_argument("--data_root", default="pilot_data/",type=str, help="Define the root folder of the different datasets.")
parser.add_argument("--mother_dir", default='', type=str, help="Define the mother dir in data_root with all runs (over rules endswith).")
parser.add_argument("--txt_set", default='', type=str, help="Define the run dirs with a train/val/test_set.txt.")
parser.add_argument("--destination", default='', type=str, help="Define the name of the final dataset.")
parser.add_argument("--noise_type", default='uni', type=str, help="Define how background of image is filled with noise: uni, ou")
parser.add_argument("--owr", action='store_true', help="If destination already exists, overwrite.")
parser.add_argument("--gray_min",default=170,type=int,help="The minimum gray value that channel should have to be put as background.")
parser.add_argument("--gray_max",default=180,type=int,help="The maximum gray value that channel should have to be put as background.")

FLAGS, others = parser.parse_known_args()

if FLAGS.data_root[0] != '/':  # 2. Pilot_data directory for saving data
  FLAGS.data_root=os.environ['HOME']+'/'+FLAGS.data_root

# get all runs
if len(FLAGS.txt_set) != 0:
  runs=[r.strip() for r in open(FLAGS.data_root+FLAGS.mother_dir+'/'+FLAGS.txt_set,'r').readlines()]
else:
  raise NotImplementedError("loop over different txt files not implemented yet, please provide --txt_set option")

# default destination is mother_dir + '_' + noise_type
if FLAGS.destination == '':
  FLAGS.destination = FLAGS.data_root + FLAGS.mother_dir + '_' + FLAGS.noise_type
elif not FLAGS.destination.startswith('/'):
  FLAGS.destination = FLAGS.data_root + FLAGS.destination

print("\nSettings:")
for f in FLAGS.__dict__: print("{0}: {1}".format( f, FLAGS.__dict__[f]))

# create destination
count_offset=0
if os.path.isdir(FLAGS.destination):
  if FLAGS.owr:
    shutil.rmtree(FLAGS.destination)
  else:
    count_offset = len([d for d in os.listdir(FLAGS.destination) if os.path.isdir(FLAGS.destination+'/'+d)])
    print("Copy with offset as there were already {0} directories in {1}".format(count_offset, FLAGS.destination))
    # raise NameError( 'Destination already exists, overwriting alert: '+ FLAGS.destination )    

if not os.path.isdir(FLAGS.destination): os.makedirs(FLAGS.destination)

new_runs=[]

# for each run 
for i,r in enumerate(runs):
  print("{0}: {1}/{2} : {3}".format(time.strftime("%Y-%m-%d_%I%M"), i, len(runs), r))
  #   1. copy run to destination
  subprocess.call(shlex.split("cp -r {0} {1}/{2:05d}_{3}".format(r, FLAGS.destination, count_offset+i, os.path.basename(r).split('_')[1])))
  #   2. mv RGB to RGB_old
  subprocess.call(shlex.split("mv {1}/{2:05d}_{3}/RGB {1}/{2:05d}_{3}/RGB_old".format(r, FLAGS.destination, count_offset+i, os.path.basename(r).split('_')[1])))
  #   3. create new RGB
  os.makedirs("{0}/{1:05d}_{2}/RGB".format(FLAGS.destination, count_offset+i, os.path.basename(r).split('_')[1]))
  #   4. for img in RGB_old
  images=["{0}/{1:05d}_{2}/RGB_old/{3}".format(FLAGS.destination, count_offset+i, os.path.basename(r).split('_')[1],img) for img in os.listdir("{0}/{1:05d}_{2}/RGB_old".format(FLAGS.destination, count_offset+i, os.path.basename(r).split('_')[1]))]
  for file_name in images:
    # 4.1. read in image
    img=sio.imread(file_name)
    # 4.2. create mask
    mask=np.zeros(img.shape) #initialize all negative
    # go over each channel to filter vector wise
    mask_0=mask[:,:,0]
    mask_1=mask[:,:,1]
    mask_2=mask[:,:,2]
    
    img_0 = img[:,:,0]
    img_1 = img[:,:,1]
    img_2 = img[:,:,2]
    
    for mask_i in [mask_0, mask_1, mask_2]:
      for img_i in [img_0, img_1, img_2]:
        mask_i[img_i<FLAGS.gray_min]=1
        mask_i[img_i>FLAGS.gray_max]=1

    # mask_0[img_0<FLAGS.gray_min]=1
    # mask_0[img_0>FLAGS.gray_max]=1
    # mask_1[img_1<FLAGS.gray_min]=1
    # mask_1[img_1>FLAGS.gray_max]=1
    # mask_2[img_2<FLAGS.gray_min]=1
    # mask_2[img_2>FLAGS.gray_max]=1


    # 4.3. create background and combine
    if FLAGS.noise_type == 'uni':
      background = np.random.randint(0,255+1,size=img.shape)
    elif FLAGS.noise_type == 'ou':
      # theta=0.1
      # sigma=0.1
      theta=np.random.beta(2,2) #min(max(np.random.normal(0.5,1),0),2)
      sigma=np.random.beta(1,3)
    
      # create horizontal noise over the columns repeated over the rows
      ou = ou_noise.OUNoise(3,0,theta,sigma)
      horizontal_noise = []
      for j in range(img.shape[1]):
          horizontal_noise.append(np.asarray(256*ou.noise()+256/2.))

      horizontal_noise = np.repeat(np.expand_dims(np.asarray(horizontal_noise), axis=0),img.shape[0],axis=0).astype(np.uint8)

      # create vertical noise over the rows repeated over the columns
      ou = ou_noise.OUNoise(3,0,theta,sigma)
      vertical_noise = []
      for j in range(img.shape[0]):
          vertical_noise.append(np.asarray(256*ou.noise()+256/2.))

      vertical_noise = np.repeat(np.expand_dims(np.asarray(vertical_noise), axis=1),img.shape[1],axis=1).astype(np.uint8)

      # combine the two 
      background = (horizontal_noise + vertical_noise)/2.

      # ensure it is uint8
      background = background.astype(np.uint8)
    else:
      raise NotImplementedError("Type of noise is unknown:{0}".format(FLAGS.noise_type))
    # 4.4. combine in new image
    inv_mask=np.abs(mask-1)
    combined=np.multiply(mask,img)+np.multiply(inv_mask,background)

    # 4.5. save the image away
    plt.imsave("{0}/{1:05d}_{2}/RGB/{3}".format(FLAGS.destination, count_offset+i, os.path.basename(r).split('_')[1], os.path.basename(file_name)),combined.astype(np.uint8))

  # 5. append runs to set.txt
  new_runs.append("{0}/{1:05d}_{2} \n".format(FLAGS.destination, count_offset+i, os.path.basename(r).split('_')[1]))

  # 6. remove RGB_old
  subprocess.call(shlex.split("rm -r {0}/{1:05d}_{2}/RGB_old".format(FLAGS.destination, count_offset+i, os.path.basename(r).split('_')[1])))

with open("{0}/{1}".format(FLAGS.destination, FLAGS.txt_set),'w') as new_set:
  for l in new_runs: new_set.write(l)

print("\n {0} Augment data.py: finished.".format(time.strftime("%Y-%m-%d_%I%M")))

