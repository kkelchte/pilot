#!/usr/bin/python
""" 
DNN policy trained in simulation supervised fashion offline from a dataset
Author: Klaas Kelchtermans (based on code of Patrick Emami)
"""
# Block all numpy-scipy incompatibility warnings (could be removed at following scipy update (>1.1))
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import sys, os, os.path
import subprocess
import shutil
import time
import signal
import argparse
import random
import torch
import torch.backends.cudnn as cudnn

from model import Model
import offline
import online
import tools
import arguments
# import models.mobile_net as mobile_net



# Use the main method for starting the training procedure and closing it in the end.
def main(_):
  parser = argparse.ArgumentParser(description='Main pilot that can train or evaluate online or offline from a dataset.')
  parser=arguments.add_arguments(parser)
  try:
    FLAGS, others = parser.parse_known_args()
  except:
    sys.exit(2)
  
  np.random.seed(FLAGS.random_seed)
  random.seed(FLAGS.random_seed)
  print("[main.py] Found {0} cuda devices available.".format(torch.cuda.device_count()))
  torch.manual_seed(FLAGS.random_seed)
  torch.cuda.manual_seed(FLAGS.random_seed)
  cudnn.benchmark = False
  cudnn.deterministic = True

  if FLAGS.random_learning_rate:
    FLAGS.learning_rate = 10**np.random.uniform(-2,0)
  
  # Create absolute paths where necessary
  if FLAGS.summary_dir[0] != '/': FLAGS.summary_dir = os.path.join(os.getenv('HOME'),FLAGS.summary_dir)

  #Check log folders and if necessary remove:
  if (FLAGS.log_tag == 'testing' or FLAGS.owr) and not FLAGS.on_policy:
    if os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag):
      shutil.rmtree(FLAGS.summary_dir+FLAGS.log_tag, ignore_errors=False)
  
  if os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag) and os.path.isfile(FLAGS.summary_dir+FLAGS.log_tag+'/my-model') and os.path.isfile(FLAGS.summary_dir+FLAGS.log_tag+'/configuration.xml'):
    print("[main.py]: found previous checkpoint from which training is continued: {0}".format(FLAGS.log_tag))
    FLAGS.load_config = True
    FLAGS.scratch = False
    FLAGS.continue_training = True # ensures that optimizers parameters are loaded as well.
    FLAGS.checkpoint_path = FLAGS.log_tag

  if len(FLAGS.checkpoint_path) != 0 and FLAGS.checkpoint_path[0] != '/': 
    FLAGS.checkpoint_path = os.path.join(FLAGS.summary_dir, FLAGS.checkpoint_path) 
  
  if not os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag): 
    os.makedirs(FLAGS.summary_dir+FLAGS.log_tag)
    
  if FLAGS.load_config:
    FLAGS=tools.load_config(FLAGS, FLAGS.checkpoint_path)
    
  tools.save_config(FLAGS, FLAGS.summary_dir+FLAGS.log_tag)
  model = Model(FLAGS)

  if FLAGS.on_policy: # online training/evaluating
    print('[main] On-policy training.')
    import rosinterface
    rosnode = rosinterface.PilotNode(FLAGS, model, FLAGS.summary_dir+FLAGS.log_tag)
    while True:
        try:
          sys.stdout.flush()
          signal.pause()
        except Exception as e:
          print('! EXCEPTION: ',e)
          # sess.close()
          print('done')
          sys.exit(0)
  elif FLAGS.online:
    print("[main] Online training (off-policy).")
    online.run(FLAGS,model)  
  else:
    print('[main] Offline training.')
    offline.run(FLAGS,model)
  
    
if __name__ == '__main__':
  # execute only if run as the entry point into the program
  main('')