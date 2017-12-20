#!/usr/bin/python
""" 
DNN policy trained in simulation supervised fashion offline from a dataset
Author: Klaas Kelchtermans (based on code of Patrick Emami)
"""
#from lxml import etree as ET
import xml.etree.cElementTree as ET
import tensorflow as tf
import tensorflow.contrib.losses as losses
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim import model_analyzer as ma
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import random_ops

import numpy as np
import sys, os, os.path
import subprocess
import shutil
import time
import signal

# Block all the ugly printing...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


FLAGS = tf.app.flags.FLAGS

# ==========================
#   Training Parameters
# ==========================
tf.app.flags.DEFINE_boolean("testing", False, "In case we're only testing, the model is tested on the test.txt files and not trained.")
tf.app.flags.DEFINE_boolean("offline", True, "Training from an offline dataset.")
# ===========================
#   Utility Parameters
# ===========================
# Print output of ros verbose or not
tf.app.flags.DEFINE_boolean("load_config", False, "Load flags from the configuration file found in the checkpoint path.")
tf.app.flags.DEFINE_boolean("verbose", True, "Print output of ros verbose or not.")
tf.app.flags.DEFINE_string("summary_dir", 'tensorflow/log/', "Choose the directory to which tensorflow should save the summaries.")
tf.app.flags.DEFINE_string("log_tag", 'testing', "Add log_tag to overcome overwriting of other log files.")
tf.app.flags.DEFINE_string("device", '/gpu:0', "Choose to run on gpu or cpu: /cpu:0 or /gpu:0")
tf.app.flags.DEFINE_integer("random_seed", 123, "Set the random seed to get similar examples.")
tf.app.flags.DEFINE_boolean("owr", False, "Overwrite existing logfolder when it is not testing.")
tf.app.flags.DEFINE_float("action_bound", 1.0, "Define between what bounds the actions can go. Default: [-1:1].")
tf.app.flags.DEFINE_boolean("real", False, "Define settings in case of interacting with the real (bebop) drone.")
tf.app.flags.DEFINE_boolean("evaluate", False, "Just evaluate the network without training.")
tf.app.flags.DEFINE_string("network", 'mobile', "Define the type of network.")
tf.app.flags.DEFINE_float("depth_multiplier", 0.25, "Define the depth of the network in case of mobilenet.")

tf.app.flags.DEFINE_boolean("auxiliary_depth", False, "Specify whether a depth map is predicted.")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Start learning rate.")
tf.app.flags.DEFINE_boolean("random_learning_rate", False, "Use sampled learning rate from UL(10**-2, 1)")

tf.app.flags.DEFINE_boolean("plot_depth", False, "Specify whether the depth predictions is saved as images.")
tf.app.flags.DEFINE_boolean("n_fc", False, "In case of True, prelogit features are concatenated before feeding to the fully connected layers.")
tf.app.flags.DEFINE_integer("n_frames", 3, "Specify the amount of frames concatenated in case of n_fc.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Define the size of minibatches.")

tf.app.flags.DEFINE_string("data_format", 'NHWC', "NHWC is the most convenient (way data is saved), though NCHW is faster on GPU.")

from model import Model
import tools
if not FLAGS.offline: import rosinterface
import offline
import models.mobile_net as mobile_net

if not FLAGS.offline: from std_msgs.msg import Empty


# ===========================
#   Save settings
# ===========================
def save_config(logfolder, file_name = "configuration"):
  """
  save all the FLAG values in a config file / xml file
  """
  print("Save configuration to: {}".format(logfolder))
  root = ET.Element("conf")
  flg = ET.SubElement(root, "flags")
  
  flags_dict = FLAGS.__dict__['__flags']
  for f in flags_dict:
    #print f, flags_dict[f]
    ET.SubElement(flg, f, name=f).text = str(flags_dict[f])
  tree = ET.ElementTree(root)
  tree.write(os.path.join(logfolder,file_name+".xml"), encoding="us-ascii", xml_declaration=True, method="xml")

# ===========================
#   Load settings
# ===========================
def load_config(modelfolder, file_name = "configuration"):
  """
  save all the FLAG values in a config file / xml file
  """
  print("Load configuration from: ", modelfolder)
  tree = ET.parse(os.path.join(modelfolder,file_name+".xml"))
  boollist=['n_fc','auxiliary_depth', 'discrete']
  intlist=['n_frames', 'num_outputs']
  floatlist=[]
  stringlist=['network', 'data_format']
  for child in tree.getroot().find('flags'):
    try :
      if child.attrib['name'] in boollist:
        FLAGS.__setattr__(child.attrib['name'], child.text=='True')
        # print 'set:', child.attrib['name'], child.text=='True'
      elif child.attrib['name'] in intlist:
        FLAGS.__setattr__(child.attrib['name'], int(child.text))
        # print 'set:', child.attrib['name'], int(child.text)
      elif child.attrib['name'] in floatlist:
        FLAGS.__setattr__(child.attrib['name'], float(child.text))
        # print 'set:', child.attrib['name'], float(child.text)
      elif child.attrib['name'] in stringlist:
        FLAGS.__setattr__(child.attrib['name'], str(child.text))
        # print 'set:', child.attrib['name'], str(child.text)
    except : 
      print 'couldnt set:', child.attrib['name'], child.text
      pass

# Use the main method for starting the training procedure and closing it in the end.
def main(_):
  # for p in sys.path:
  #   print 'path: {}'.format(p)
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)
  
  if FLAGS.random_learning_rate:
    FLAGS.learning_rate = 10**np.random.uniform(-2,0)
    
  if FLAGS.load_config:
    checkpoint_path = FLAGS.checkpoint_path
    if checkpoint_path[0]!='/': checkpoint_path = os.path.join(os.getenv('HOME'),'tensorflow/log',checkpoint_path)
    if not os.path.isfile(checkpoint_path+'/checkpoint'):
      checkpoint_path = checkpoint_path+'/'+[mpath for mpath in sorted(os.listdir(checkpoint_path)) if os.path.isdir(checkpoint_path+'/'+mpath) and os.path.isfile(checkpoint_path+'/'+mpath+'/checkpoint')][-1]
    load_config(checkpoint_path)
    
  FLAGS.summary_dir = os.path.join(os.getenv('HOME'),FLAGS.summary_dir)
  print("summary dir: {}".format(FLAGS.summary_dir))
  
  #Check log folders and if necessary remove:
  if FLAGS.log_tag == 'testing' or FLAGS.owr:
    if os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag):
      shutil.rmtree(FLAGS.summary_dir+FLAGS.log_tag,ignore_errors=False)
  else :
    if os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag):
      raise NameError( 'Logfolder already exists, overwriting alert: '+ FLAGS.summary_dir+FLAGS.log_tag ) 
  os.makedirs(FLAGS.summary_dir+FLAGS.log_tag)
  save_config(FLAGS.summary_dir+FLAGS.log_tag)

  action_dim = 1 #only turn in yaw from -1:1
  
  config=tf.ConfigProto(allow_soft_placement=True)
  # Keep it at true, in online fashion with singularity (not condor) on qayd (not laptop) resolves this in a Cudnn Error
  config.gpu_options.allow_growth = True
  # config.gpu_options.allow_growth = False
  sess = tf.Session(config=config)
  model = Model(sess, action_dim, bound=FLAGS.action_bound)
  writer = tf.summary.FileWriter(FLAGS.summary_dir+FLAGS.log_tag, sess.graph)
  model.writer = writer
  
  def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    print('saving checkpoints')
    model.save(FLAGS.summary_dir+FLAGS.log_tag)
    sess.close()
    print('done.')
    sys.exit(0)
  signal.signal(signal.SIGINT, signal_handler)
  print('------------Press Ctrl+C to end the learning') 
  
  if FLAGS.offline:
    print('Offline training.')
    offline.run(model)
  else: # online training/evaluating
    print('Online training.')
    rosnode = rosinterface.PilotNode(model, FLAGS.summary_dir+FLAGS.log_tag)
    while True:
        try:
          sys.stdout.flush()
          signal.pause()
        except Exception as e:
          print('! EXCEPTION: ',e)
          sess.close()
          print('done')
          sys.exit(0)
  
    
if __name__ == '__main__':
  tf.app.run() 
