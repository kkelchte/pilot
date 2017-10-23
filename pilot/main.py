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
from model import Model
import tools
import data
import mobile_net
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
tf.app.flags.DEFINE_integer("max_episodes", 80, "The maximum number of episodes (~runs through all the training data.)")
tf.app.flags.DEFINE_boolean("testing", False, "In case we're only testing, the model is tested on the test.txt files and not trained.")

# ===========================
#   Utility Parameters
# ===========================
# Print output of ros verbose or not
tf.app.flags.DEFINE_boolean("verbose", True, "Print output of ros verbose or not.")
# Directory for storing tensorboard summary results
tf.app.flags.DEFINE_string("summary_dir", 'tensorflow/log/', "Choose the directory to which tensorflow should save the summaries.")
# Add log_tag to overcome overwriting other log files
tf.app.flags.DEFINE_string("log_tag", 'testing', "Add log_tag to overcome overwriting of other log files.")
# Choose to run on gpu or cpu
tf.app.flags.DEFINE_string("device", '/gpu:0', "Choose to run on gpu or cpu: /cpu:0 or /gpu:0")
# Set the random seed to get similar examples
tf.app.flags.DEFINE_integer("random_seed", 123, "Set the random seed to get similar examples.")
# Overwrite existing logfolder
tf.app.flags.DEFINE_boolean("owr", True, "Overwrite existing logfolder when it is not testing.")
tf.app.flags.DEFINE_float("action_bound", 1.0, "Define between what bounds the actions can go. Default: [-1:1].")
tf.app.flags.DEFINE_string("network", 'mobile', "Define the type of network.")
tf.app.flags.DEFINE_float("depth_multiplier", 0.25, "Define the depth of the network in case of mobilenet.")

tf.app.flags.DEFINE_boolean("auxiliary_depth", True, "Specify whether a depth map is predicted.")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Start learning rate.")
tf.app.flags.DEFINE_boolean("random_learning_rate", False, "Use sampled learning rate from UL(10**-2, 1)")

tf.app.flags.DEFINE_boolean("plot_depth", False, "Specify whether the depth predictions is saved as images.")
tf.app.flags.DEFINE_boolean("n_fc", True, "In case of True, prelogit features are concatenated before feeding to the fully connected layers.")
tf.app.flags.DEFINE_integer("n_frames", 3, "Specify the amount of frames concatenated in case of n_fc.")

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


# Use the main method for starting the training procedure and closing it in the end.
def main(_):
  # some startup settings
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)
  
  if FLAGS.random_learning_rate:
    FLAGS.learning_rate = 10**np.random.uniform(-2,0)
  

  #Check log folders and if necessary remove:
  summary_dir = os.path.join(os.getenv('HOME'),FLAGS.summary_dir)
  # summary_dir = FLAGS.summary_dir
  print("summary dir: {}".format(summary_dir))
  #Check log folders and if necessary remove:
  if FLAGS.log_tag == 'testing' or FLAGS.owr:
    if os.path.isdir(summary_dir+FLAGS.log_tag):
      shutil.rmtree(summary_dir+FLAGS.log_tag,ignore_errors=False)
  else :
    if os.path.isdir(summary_dir+FLAGS.log_tag):
      raise NameError( 'Logfolder already exists, overwriting alert: '+ summary_dir+FLAGS.log_tag ) 
  os.makedirs(summary_dir+FLAGS.log_tag)
  save_config(summary_dir+FLAGS.log_tag)
    
    
  #define the size of the network input
  if FLAGS.network =='mobile':
    state_dim = [1, mobile_net.mobilenet_v1.default_image_size[FLAGS.depth_multiplier], 
      mobile_net.mobilenet_v1.default_image_size[FLAGS.depth_multiplier], 3]  
  else:
    raise NameError( 'Network is unknown: ', FLAGS.network)

  action_dim = 1 #only turn in yaw from -1:1
  
  print( "Number of State Dimensions:{}".format(state_dim))
  print( "Number of Action Dimensions:{}".format(action_dim))
  print( "Action bound:{}".format(FLAGS.action_bound))
  
  config=tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  model = Model(sess, state_dim, action_dim, bound=FLAGS.action_bound)
  writer = tf.summary.FileWriter(summary_dir+FLAGS.log_tag, sess.graph)
  model.writer = writer
  
  # def signal_handler(signal, frame):
  #   print('You pressed Ctrl+C!')
  #   #save checkpoint?
  #   # print('saving checkpoints')
  #   # model.save(summary_dir+FLAGS.log_tag)
  #   sess.close()
  #   print('done.')
  #   sys.exit(0)
  # signal.signal(signal.SIGINT, signal_handler)
  # print('------------Press Ctrl+C to end the learning') 
  
  def run_episode(data_type, sumvar):
    '''run over batches
    return different losses
    type: 'train', 'val' or 'test'
    '''
    depth_predictions = []
    start_time=time.time()
    data_loading_time = 0
    calculation_time = 0
    start_data_time = time.time()
    tot_loss=[]
    ctr_loss=[]
    dep_loss=[]

    for index, ok, batch in data.generate_batch(data_type):
      data_loading_time+=(time.time()-start_data_time)
      start_calc_time=time.time()
      if ok:
        inputs = np.array([_['img'] for _ in batch])
        targets = np.array([[_['ctr']] for _ in batch])
        try:
          target_depth = np.array([_['depth'] for _ in batch]).reshape((-1,55,74)) if FLAGS.auxiliary_depth else []
          if len(target_depth) == 0 : raise ValueError('No depth in batch.')
        except ValueError: 
          target_depth = [] # In case there is no depth targets available
        if data_type=='train':
          losses = model.backward(inputs, targets, depth_targets=target_depth)
        elif data_type=='val' or data_type=='test':
          losses, aux_results = model.forward(inputs, auxdepth=False, targets=targets, target_depth=target_depth)
        try:
          ctr_loss.append(losses['c'])
          if FLAGS.auxiliary_depth: dep_loss.append(losses['d'])
          tot_loss.append(losses['t'])
        except KeyError:
          pass
        if index == 1 and data_type=='val' and FLAGS.plot_depth: # TO TEST
            depth_predictions = tools.plot_depth(inputs, target_depth, model)
      else:
        print('Failed to run {}.'.format(data_type))
      calculation_time+=(time.time()-start_calc_time)
      start_data_time = time.time()
    if len(tot_loss)!=0: sumvar['loss_'+data_type+'_total']=np.mean(tot_loss) 
    if len(ctr_loss)!=0: sumvar['loss_'+data_type+'_control']=np.mean(ctr_loss)   
    if len(tot_loss)!=0 and FLAGS.auxiliary_depth: sumvar['loss_'+data_type+'_depth']=np.mean(dep_loss)   
    if len(depth_predictions) != 0: sumvar['depth_predictions']=depth_predictions
    print('>>{0} [{1[2]}/{1[1]}_{1[3]:02d}:{1[4]:02d}]: data {2}; calc {3}'.format(data_type.upper(),tuple(time.localtime()[0:5]),
      tools.print_dur(data_loading_time),tools.print_dur(calculation_time)))
    if data_type == 'val' or data_type == 'test':
      print('{}'.format(str(sumvar)))
    sys.stdout.flush()
    return sumvar

  data.prepare_data((state_dim[1], state_dim[2], state_dim[3]))
  ep=0
  while ep<FLAGS.max_episodes-1 and not FLAGS.testing:
    ep+=1

    print('start episode: {}'.format(ep))
    # ----------- train episode
    sumvar = run_episode('train', {})
    
    # ----------- validate episode
    # sumvar = run_episode('val', {})
    sumvar = run_episode('val', sumvar)
    # ----------- write summary
    try:
      model.summarize(sumvar)
    except Exception as e:
      print('failed to summarize {}'.format(e))
    # write checkpoint every x episodes
    if (ep%20==0 and ep!=0):
      print('saved checkpoint')
      model.save(summary_dir+FLAGS.log_tag)
  # ------------ test
  sumvar = run_episode('test', {})  
  # ----------- write summary
  try:
    model.summarize(sumvar)
  except Exception as e:
    print('failed to summarize {}'.format(e))
  # write checkpoint every x episodes
  if ((ep%20==0 and ep!=0) or ep==(FLAGS.max_episodes-1)) and not FLAGS.testing:
    print('saved checkpoint')
    model.save(summary_dir+FLAGS.log_tag)
    
if __name__ == '__main__':
  tf.app.run() 
