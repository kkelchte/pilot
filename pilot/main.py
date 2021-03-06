#!/usr/bin/python
""" 
DNN policy trained in simulation supervised fashion offline from a dataset
Author: Klaas Kelchtermans (based on code of Patrick Emami)
"""
# Block all numpy-scipy incompatibility warnings (could be removed at following scipy update (>1.1))
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

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
import argparse


from model import Model
import tools
import offline
import models.mobile_net as mobile_net


# ===========================
#   Save settings
# ===========================
def save_config(FLAGS, logfolder, file_name = "configuration"):
  """
  save all the FLAG values in a config file / xml file
  """
  print("Save configuration to: {}".format(logfolder))
  root = ET.Element("conf")
  flg = ET.SubElement(root, "flags")
  
  flags_dict=FLAGS.__dict__
  for f in sorted(flags_dict.keys()):
    # print f, flags_dict[f]
    e = ET.SubElement(flg, f, name=f) 
    e.text = str(flags_dict[f])
    e.tail = "\n  "
  tree = ET.ElementTree(root)
  tree.write(os.path.join(logfolder,file_name+".xml"), encoding="us-ascii", xml_declaration=True, method="xml")

# ===========================
#   Load settings
# ===========================
def load_config(FLAGS, modelfolder, file_name = "configuration"):
  """
  save all the FLAG values in a config file / xml file
  """
  print("Load configuration from: ", modelfolder)
  tree = ET.parse(os.path.join(modelfolder,file_name+".xml"))
  boollist=['auxiliary_depth', 'discrete']
  intlist=['n_frames', 'num_outputs']
  floatlist=['depth_multiplier','speed','action_bound']
  stringlist=['network', 'data_format']
  for child in tree.getroot().find('flags'):
    try :
      if child.attrib['name'] in boollist:
        FLAGS.__setattr__(child.attrib['name'], child.text=='True')
        print 'set:', child.attrib['name'], child.text=='True'
      elif child.attrib['name'] in intlist:
        FLAGS.__setattr__(child.attrib['name'], int(child.text))
        print 'set:', child.attrib['name'], int(child.text)
      elif child.attrib['name'] in floatlist:
        FLAGS.__setattr__(child.attrib['name'], float(child.text))
        print 'set:', child.attrib['name'], float(child.text)
      elif child.attrib['name'] in stringlist:
        # Temporary hack to load models from doshico
        # if not FLAGS.network != 'mobile_nfc': 
        FLAGS.__setattr__(child.attrib['name'], str(child.text))
        print 'set:', child.attrib['name'], str(child.text)
      # Temporary hack to load models from doshico
      elif child.attrib['name'] == 'n_fc':
        FLAGS.network='mobile_nfc'
        print 'set: network to mobile_nfc'
    except : 
      print 'couldnt set:', child.attrib['name'], child.text
      pass

  return FLAGS

# Use the main method for starting the training procedure and closing it in the end.
def main(_):
  parser = argparse.ArgumentParser(description='Main pilot that can train or evaluate online or offline from a dataset.')
  
  # ==========================
  #   Training Parameters
  # ==========================
  parser.add_argument("--testing", action='store_true', help="In case we're only testing, the model is tested on the test.txt files and not trained.")
  parser.add_argument("--learning_rate", default=0.1, type=float, help="Start learning rate.")
  parser.add_argument("--batch_size",default=32,type=int,help="Define the size of minibatches.")
  
  # ==========================
  #   Lifelonglearning Parameters
  # ==========================
  parser.add_argument("--lifelonglearning",action='store_true',help="In case there is a previous domain upon which the model was trained, use the lifelonglearning regularization to overcome forgetting.")
  parser.add_argument("--update_importance_weights",action='store_true',help="Update importance weights for all variables for this domain.")
  parser.add_argument("--lll_weight", default=1, type=float, help="Weight the lifelonglearning regularization term in the overall loss.")
  
  # ==========================
  #   Offline Parameters
  # ==========================
  parser.add_argument("--max_episodes",default=600,type=int,help="The maximum number of episodes (~runs through all the training data.)")
  parser.add_argument("--visualize_saliency_of_output",action='store_true',help="Visualize saliency maps of the output.")
  parser.add_argument("--visualize_deep_dream_of_output",action='store_true',help="Visualize gradient ascent maps for different extreme controls.")
  parser.add_argument("--visualize_activations",action='store_true',help="Visualize activation.")
  parser.add_argument("--visualize_control_activation_maps",action='store_true',help="Visualize control activation map.")
  parser.add_argument("--histogram_of_activations",action='store_true',help="Summarize all activations in a histogram.")
  parser.add_argument("--histogram_of_weights",action='store_true',help="Summarize all weights in a histogram.")

  # ===========================
  #   Utility Parameters
  # ===========================
  # Print output of ros verbose or not
  parser.add_argument("--load_config", action='store_true',help="Load flags from the configuration file found in the checkpoint path.")
  parser.add_argument("--verbose", action='store_false', help="Print output of ros verbose or not.")
  parser.add_argument("--summary_dir", default='tensorflow/log/', type=str, help="Choose the directory to which tensorflow should save the summaries.")
  parser.add_argument("-t","--log_tag", default='testing', type=str, help="Add log_tag to overcome overwriting of other log files.")
  parser.add_argument("--device", default='/gpu:0', type=str, help= "Choose to run on gpu or cpu: /cpu:0 or /gpu:0")
  parser.add_argument("--random_seed", default=123, type=int, help="Set the random seed to get similar examples.")
  parser.add_argument("--owr", action='store_true', help="Overwrite existing logfolder when it is not testing.")
  parser.add_argument("--action_bound", default=1.0, type=float, help= "Define between what bounds the actions can go. Default: [-1:1].")
  parser.add_argument("--action_dim", default=1.0, type=float, help= "Define the dimension of the actions: 1dimensional as it only turns in yaw.")
  parser.add_argument("--real", action='store_true', help="Define settings in case of interacting with the real (bebop) drone.")
  parser.add_argument("--evaluate", action='store_true', help="Just evaluate the network without training.")
  parser.add_argument("--random_learning_rate", action='store_true', help="Use sampled learning rate from UL(10**-2, 1)")
  parser.add_argument("--plot_depth", action='store_true', help="Specify whether the depth predictions is saved as images.")

  # ===========================
  #   Data Parameters
  # ===========================
  # parser.add_argument("--hdf5", action='store_true', help="Define wether dataset is hdf5 type.")
  parser.add_argument("--load_data_in_ram", action='store_true', help="Define wether the dataset is preloaded into RAM.")
  parser.add_argument("--dataset", default="small", type=str, help="pick the dataset in data_root from which your movies can be found.")
  parser.add_argument("--data_root", default="pilot_data/",type=str, help="Define the root folder of the different datasets.")
  parser.add_argument("--num_threads", default=4, type=int, help="The number of threads for loading one minibatch.")
  parser.add_argument("--control_file", default='control_info.txt', type=str, help="Define the name of the file with the action labels.")
  parser.add_argument("--depth_directory", default='Depth', type=str, help="Define the name of the directory containing the depth images: Depth or Depth_predicted.")
  parser.add_argument("--subsample", default=1, type=int, help="Subsample data over time: e.g. subsample 2 to get from 20fps to 10fps.")
  parser.add_argument("--normalize_over_actions", action='store_true', help="Try to fill a batch with different actions [-1, 0, 1].")
  
  # ===========================
  #   Model Parameters
  # ===========================
  parser.add_argument("--depth_multiplier",default=0.25,type=float, help= "Define the depth of the network in case of mobilenet.")
  parser.add_argument("--network",default='mobile',type=str, help="Define the type of network (anything in models folder without _net.py): mobile, mobile_nfc, alex, squeeze, ...")
  parser.add_argument("--output_size",default=[55,74],type=int, nargs=2, help="Define the output size of the depth frame: 55x74 [drone], 1x26 [turtle], only used in case of depth_q_net.")
  # parser.add_argument("--n_fc", action='store_true',help="In case of True, prelogit features are concatenated before feeding to the fully connected layers.")
  parser.add_argument("--n_frames",default=3,type=int,help="Specify the amount of frames concatenated in case of n_fc like mobile_nfc.")
  parser.add_argument("--auxiliary_depth", action='store_true',help="Specify whether a depth map is predicted.")
  parser.add_argument("--discrete", action='store_true',help="Specify whether the output action space is discrete.")
  parser.add_argument("--action_quantity",default=3, type=int, help="Define the number of actions in the output layer.")
  
  # INITIALIZATION
  parser.add_argument("--checkpoint_path",default='mobilenet_025', type=str, help="Specify the directory of the checkpoint of the earlier trained model.")
  parser.add_argument("--continue_training",action='store_true', help="Continue training of the prediction layers. If false, initialize the prediction layers randomly.")
  parser.add_argument("--scratch", action='store_true', help="Initialize full network randomly.")

  # TRAINING
  parser.add_argument("--depth_weight", default=1.0, type=float, help="Define the weight applied to the depth values in the loss relative to the control loss.")
  parser.add_argument("--control_weight", default=1.0, type=float, help="Define the weight applied to the control loss.")
  parser.add_argument("--weight_decay",default=0.00004,type=float, help= "Weight decay of inception network")
  parser.add_argument("--init_scale", default=0.0005, type=float, help= "Std of uniform initialization")
  parser.add_argument("--grad_mul_weight", default=0.001, type=float, help="Specify the amount the gradients of prediction layers.")
  parser.add_argument("--dropout_rate", default=0.5, type=float, help="Specify the probability of dropout to keep the activation.")
  # parser.add_argument("--clip_grad", default=0, type=int, help="Specify the max gradient norm: default 0 is no clipping, recommended 4.")
  parser.add_argument("--min_depth", default=0.0, type=float, help="clip depth loss with weigths to focus on correct depth range.")
  parser.add_argument("--max_depth", default=5.0, type=float, help="clip depth loss with weigths to focus on correct depth range.")
  parser.add_argument("--optimizer", default='adadelta', type=str, help="Specify optimizer, options: adam, adadelta, gradientdescent, rmsprop")
  parser.add_argument("--no_batchnorm_learning",action='store_false', help="In case of no batchnorm learning, are the batch normalization params (alphas and betas) not further adjusted.")
  parser.add_argument("--initializer",default='xavier',type=str, help="Define the initializer: xavier or uniform [-init_scale, init_scale]")

  parser.add_argument("--loss",default='mse',type=str, help="Define the loss: mse, huber or absolute")

  parser.add_argument("--max_loss", default=100, type=float, help= "Define the maximum loss before it is clipped.")
  
  parser.add_argument("--clip_loss_to_max",action='store_true', help="Over time, allow only smaller losses by clipping the maximum allowed loss to the lowest maximum loss.")

  # ===========================
  #   Replay Parameters
  # ===========================

  parser.add_argument("--replay_priority", default='no', type=str, help="Define which type of weights should be used when sampling from replay buffer: no, uniform_action, uniform_collision, td_error, state/action/target_variance, random_action")
  parser.add_argument("--prioritized_keeping", action='store_true', help="In case of True, the replay buffer only keeps replay data that is most likely to be sampled.")

  # ===========================
  #   Rosinterface Parameters
  # ===========================
  parser.add_argument("--online", action='store_true', help="Training/evaluating online in simulation.")
  parser.add_argument("--buffer_size", default=1000, type=int, help="Define the number of experiences saved in the buffer.")
  parser.add_argument("--ou_theta", default=0.05, type=float, help= "Theta is the pull back force of the OU Noise.")
  parser.add_argument("--noise", default='ou', type=str, help="Define whether the noise is temporally correlated (ou) or uniformly distributed (uni).")
  parser.add_argument("--sigma_z", default=0.0, type=float, help= "sigma_z is the amount of noise in the z direction.")
  parser.add_argument("--sigma_x", default=0.0, type=float, help= "sigma_x is the amount of noise in the forward speed.")
  parser.add_argument("--sigma_y", default=0.0, type=float, help= "sigma_y is the amount of noise in the y direction.")
  parser.add_argument("--sigma_yaw", default=0.0, type=float, help= "sigma_yaw is the amount of noise added to the steering angle.")
  parser.add_argument("--speed", default=1.3, type=float, help= "Define the forward speed of the quadrotor.")
  parser.add_argument("--alpha",default=0., type=float, help="Policy mixing: choose with a binomial probability of alpha for the experts policy instead of the DNN policy..")
  parser.add_argument("--epsilon",default=0, type=float, help="Apply epsilon-greedy policy for exploration.")
  parser.add_argument("--epsilon_decay", default=0.0, type=float, help="Decay the epsilon exploration over time with a slow decay rate of 1/10.")
  parser.add_argument("--prefill", action='store_true', help="Fill the replay buffer first with random (epsilon 1) flying behavior before training.")
  
  parser.add_argument("--break_and_turn", action='store_true', help="In case the robot should turn at zero forward speed by breaking.")

  parser.add_argument("--off_policy",action='store_true', help="In case the network is off_policy, the control is published on supervised_vel instead of cmd_vel.")
  parser.add_argument("--dont_show_depth",action='store_true', help="Publish the predicted horizontal depth array to topic ./depth_prection so show_depth can visualize this in another node.")

  parser.add_argument("--grad_steps", default=10, type=int, help="Define the number of batches or gradient steps are taken between 2 runs.")

  parser.add_argument("--field_of_view", default=104, type=int, help="The field of view of the camera cuts the depth scan in the range visible for the camera. Value should be even. Normal: 72 (-36:36), Wide-Angle: 120 (-60:60)")
  parser.add_argument("--smooth_scan", default=4, type=int, help="The 360degrees scan has a lot of noise and is therefore smoothed out over 4 neighboring scan readings")


  # FLAGS=parser.parse_args()
  try:
    FLAGS, others = parser.parse_known_args()
  except:
    sys.exit(2)
  # if FLAGS.random_seed == 123: FLAGS.random_seed = (int(time.time()*100)%4000)

  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)
  
  if FLAGS.random_learning_rate:
    FLAGS.learning_rate = 10**np.random.uniform(-2,0)
  
  FLAGS.summary_dir = os.path.join(os.getenv('HOME'),FLAGS.summary_dir)
  print("summary dir: {}".format(FLAGS.summary_dir))
  
  start_ep=0 # used in case a model is found with same log_tag meaning that a job disconnected on condor and model can continue training.
  
  #Check log folders and if necessary remove:
  if FLAGS.log_tag == 'testing' or FLAGS.owr:
    if os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag):
      shutil.rmtree(FLAGS.summary_dir+FLAGS.log_tag,ignore_errors=False)
  elif not FLAGS.continue_training : # in case we are training from scratch/checkpoint and not evaluating
    # check if previous run is there
    found_previous_run=False
    if os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag):
      found_previous_run=True
      previous_run = FLAGS.summary_dir+FLAGS.log_tag
    elif FLAGS.log_tag.split('/')[-1].startswith('2018'): # In case the last part of the log_tag is a date (running on condor)
      run_dir=os.path.dirname(FLAGS.summary_dir+FLAGS.log_tag)
      previous_runs = sorted([run_dir+'/'+d for d in os.listdir(run_dir) if os.path.isdir(run_dir+'/'+d) and d.startswith('2018')])
      found_previous_run = len(previous_runs) >= 1
      if found_previous_run: 
        previous_run = previous_runs[-1] 
    
    if found_previous_run:
      # extract previous run
      checkpoints=[fs for fs in os.listdir(previous_run) if fs.endswith('.meta')]
      if os.path.isfile(previous_run+'/checkpoint') and os.path.isfile(previous_run+'/configuration.xml'): 
        # if a checkpoint is found in current folder, use this folder as checkpoint path.
        #raise NameError( 'Logfolder already exists, overwriting alert: '+ previous_run )
        FLAGS.load_config = True
        FLAGS.scratch = False
        FLAGS.continue_training = True
        FLAGS.checkpoint_path = previous_run[len(FLAGS.summary_dir):] #cut off summary_dir to get previous log_tag
        checkpoint_model=open(previous_run+'/checkpoint').readlines()[0]
        start_ep=int(int(checkpoint_model.split('-')[-1][:-2])/100)
        print("Found model: {0} trained for {1} episodes in {2}".format(FLAGS.log_tag,start_ep, previous_run))
      else:
        shutil.rmtree(previous_run,ignore_errors=False)
  if not os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag): 
    os.makedirs(FLAGS.summary_dir+FLAGS.log_tag)
    
  if FLAGS.load_config:
    checkpoint_path = FLAGS.checkpoint_path
    if checkpoint_path[0]!='/': checkpoint_path = os.path.join(os.getenv('HOME'),'tensorflow/log',checkpoint_path)
    if not os.path.isfile(checkpoint_path+'/checkpoint'):
      checkpoint_path = checkpoint_path+'/'+[mpath for mpath in sorted(os.listdir(checkpoint_path)) if os.path.isdir(checkpoint_path+'/'+mpath) and os.path.isfile(checkpoint_path+'/'+mpath+'/checkpoint')][-1]
    if not os.path.isfile(checkpoint_path+'/configuration.xml'):
      checkpoint_path = checkpoint_path+'/'+[mpath for mpath in sorted(os.listdir(checkpoint_path)) if os.path.isdir(checkpoint_path+'/'+mpath) and os.path.isfile(checkpoint_path+'/'+mpath+'/configuration.xml')][-1]
    FLAGS=load_config(FLAGS, checkpoint_path)
    
  save_config(FLAGS, FLAGS.summary_dir+FLAGS.log_tag)
  config=tf.ConfigProto(allow_soft_placement=True)
  # config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
  # Keep it at true, in online fashion with singularity (not condor) on qayd (not laptop) resolves this in a Cudnn Error
  config.gpu_options.allow_growth = True
  # config.gpu_options.per_process_gpu_memory_fraction = 0.1

  # config.gpu_options.allow_growth = False
  sess = tf.Session(config=config)
  model = Model(FLAGS, sess)
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
  
  if FLAGS.online: # online training/evaluating
    print('Online training.')
    import rosinterface
    rosnode = rosinterface.PilotNode(FLAGS, model, FLAGS.summary_dir+FLAGS.log_tag)
    while True:
        try:
          sys.stdout.flush()
          signal.pause()
        except Exception as e:
          print('! EXCEPTION: ',e)
          sess.close()
          print('done')
          sys.exit(0)
  else:
    print('Offline training.')
    offline.run(FLAGS,model,start_ep)
  
    
if __name__ == '__main__':
  tf.app.run() 
