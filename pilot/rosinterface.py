import rospy
import numpy as np
# import scipy.misc as sm
import skimage.io as sio
import skimage.transform as sm

import sys, time, re, copy, cv2, os
from os import path

from tf import transformations

from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf

bridge = CvBridge()

from replay_buffer import ReplayBuffer
from model import Model
from ou_noise import OUNoise
import tools

from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from nav_msgs.msg import Odometry

import matplotlib.animation as animation
import matplotlib.pyplot as plt


#from PIL import Image

class PilotNode(object):
  """Node to listen to ROS topics like depth, rgb input and supervised control.
  The node also publishes to pilot control and predicted depth for visualization.
  """
  
  def __init__(self, FLAGS, model, logfolder):
    print('initialize pilot node')  
    self.FLAGS=FLAGS
    # Initialize fields
    self.logfolder = logfolder
    f=open(os.path.join(self.logfolder,'tf_log'),'a')
    f.write(self.FLAGS.log_tag)
    f.write('\n')
    f.close()
    self.model = model 
    self.ready=False 
    self.finished=True
    self.training=False
    
    self.last_pose=[] # previous pose, used for accumulative distance
    self.world_name = ''
    self.runs={'train':0, 'test':0} # number of online training run (used for averaging)
    # self.accumlosses = {} # gather losses and info over the run in a dictionary
    self.current_distance=0 # accumulative distance travelled from beginning of run used at evaluation
    self.furthest_point=0 # furthest point reached from spawning point at the beginning of run
    self.average_distances={'train':0, 'test':0} # running average over different runs
    self.target_control = [] # field to keep the latest supervised control
    self.target_depth = [] # field to keep the latest supervised depth
    self.nfc_images =[] #used by n_fc networks for building up concatenated frames
    self.exploration_noise = OUNoise(4, 0, self.FLAGS.ou_theta,1)
    if not self.FLAGS.dont_show_depth: self.depth_pub = rospy.Publisher('/depth_prediction', numpy_msg(Floats), queue_size=1)
    self.action_pub=rospy.Publisher('/nn_vel', Twist, queue_size=1)

    self.model.reset_metrics()

    rospy.Subscriber('/nn_start', Empty, self.ready_callback)
    rospy.Subscriber('/nn_stop', Empty, self.finished_callback)

    # extract imitation loss from supervised velocity
    rospy.Subscriber('/supervised_vel', Twist, self.supervised_callback)
    
    self.start_time = 0
    self.imitation_loss=[]
    self.depth_prediction=[]
    self.depth_loss=[]
    self.driving_duration=-1

    self.skip_frames = 0
    self.img_index = 0
    self.fsm_index = 0

    if rospy.has_param('rgb_image'): 
      image_topic=rospy.get_param('rgb_image')
      if 'compressed' in image_topic:
        rospy.Subscriber(image_topic, CompressedImage, self.compressed_image_callback)
      else:
        rospy.Subscriber(image_topic, Image, self.image_callback)
    if rospy.has_param('depth_image'):
      depth_topic = rospy.get_param('depth_image')
      if 'scan' in depth_topic:
        rospy.Subscriber(depth_topic, LaserScan, self.scan_depth_callback)
      else:
        rospy.Subscriber(depth_topic, Image, self.depth_callback)
    
    self.replay_buffer = ReplayBuffer(self.FLAGS.buffer_size, self.FLAGS.random_seed)
    if self.FLAGS.hard_replay_buffer:
      self.hard_replay_buffer = ReplayBuffer(self.FLAGS.hard_batch_size, self.FLAGS.random_seed)
      
    self.accumloss = 0
    if rospy.has_param('gt_info'):
      rospy.Subscriber(rospy.get_param('gt_info'), Odometry, self.gt_callback)

    # Add some lines to debug delays:
    self.time_im_received=[]
    self.time_ctr_send=[]

    rospy.init_node('pilot', anonymous=True)  
  
  #--------------------------------
  # Callbacks
  #--------------------------------

  def ready_callback(self,msg):
    """ callback function that makes DNN policy starts the ready flag is set on 1 (for 3s)"""
    if not self.ready and self.finished:
      print('Neural control activated.')
      self.ready = True
      self.start_time = rospy.get_time()
      self.finished = False
      self.exploration_noise.reset()
      # choose one speed for this flight
      self.FLAGS.speed=self.FLAGS.speed + (not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_x, self.FLAGS.sigma_x)
      if rospy.has_param('evaluate'):
        self.FLAGS.evaluate = rospy.get_param('evaluate')
        print '--> set evaluate to: {0} with speed {1}'.format(self.FLAGS.evaluate, self.FLAGS.speed)
      if rospy.has_param('skip_frames'):
        self.skip_frames = rospy.get_param('skip_frames')
        print '--> set skip_frames to: {0}'.format(self.skip_frames)
      if rospy.has_param('world_name') :
        self.world_name = rospy.get_param('world_name')
      time.sleep(1) # wait one second, otherwise create_dataset can't follow...
        
  def gt_callback(self, data):
    """Callback function that keeps track of positions for logging"""
    if not self.ready or self.training: return
    current_pos=[data.pose.pose.position.x,
                    data.pose.pose.position.y,
                    data.pose.pose.position.z]
    if len(self.last_pose)!= 0:
        self.current_distance += np.sqrt((self.last_pose[0,3]-current_pos[0])**2+(self.last_pose[1,3]-current_pos[1])**2)
    self.furthest_point=max([self.furthest_point, np.sqrt(current_pos[0]**2+current_pos[1]**2)])

    # Get pose (rotation and translation) [DEPRECATED: USED FOR ODOMETRY]
    quaternion = (data.pose.pose.orientation.x,
      data.pose.pose.orientation.y,
      data.pose.pose.orientation.z,
      data.pose.pose.orientation.w)
    self.last_pose = transformations.quaternion_matrix(quaternion) # orientation of current frame relative to global frame
    self.last_pose[0:3,3]=current_pos

  def process_rgb(self, msg):
    """ Convert RGB serial data to opencv image of correct size"""
    try:
      # Convert your ROS Image message to OpenCV2
      # changed to normal RGB order as i ll use matplotlib and PIL instead of opencv
      img =bridge.imgmsg_to_cv2(msg, 'rgb8') 
    except CvBridgeError as e:
      print(e)
    else:
      img = img[::2,::5,:]
      size = self.model.input_size[1:]
      img = sm.resize(img,size,mode='constant').astype(float)
      return img

  def process_rgb_compressed(self, msg):
    """ Convert RGB serial data to opencv image of correct size"""
    # if not self.ready or self.finished: return []
    try:
      img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
    except CvBridgeError as e:
      print(e)
    else:
      # 308x410 to 128x128
      img = img[::2,::3,:]
      size = self.model.input_size[1:]
      img = sm.resize(img,size,mode='constant').astype(float)
      return img

  def process_depth(self, msg):
    """ Convert depth serial data to opencv image of correct size"""
    # if not self.ready or self.finished: return [] 
    try:
      # Convert your ROS Image message to OpenCV2
      de = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')#gets float of 32FC1 depth image
    except CvBridgeError as e:
      print(e)
    else:
      
      de = de[::6,::8]
      shp=de.shape
      # # assume that when value is not a number it is due to a too large distance (set to 5m)
      # # values can be nan for when they are closer than 0.5m but than the evaluate node should
      # # kill the run anyway.
      de=np.asarray([ e*1.0 if not np.isnan(e) else 5 for e in de.flatten()]).reshape(shp) # clipping nans: dur: 0.010
      size = (55,74)
      # print 'DEPTH: min: ',np.amin(de),' and max: ',np.amax(de)
      
      de = sm.resize(de,size,order=1,mode='constant', preserve_range=True)
      return de

  def process_scan(self, msg):
    """Preprocess serial scan: clip horizontal field of view, clip at 1's and ignore 0's, smooth over 4 bins."""
    # field of view should follow camera: 
    #    wide-angle camera: -60 to 60. 
    #    normal camera: -35 to 35.
    ranges=[1 if r > 1 or r==0 else r for r in msg.ranges]
    # clip left 45degree range from 0:45 reversed with right 45degree range from the last 45:
    ranges=list(reversed(ranges[:self.FLAGS.field_of_view/2]))+list(reversed(ranges[-self.FLAGS.field_of_view/2:]))
    # add some smoothing by averaging over 4 neighboring bins
    ranges = [sum(ranges[i*self.FLAGS.smooth_scan:i*self.FLAGS.smooth_scan+self.FLAGS.smooth_scan])/self.FLAGS.smooth_scan for i in range(int(len(ranges)/self.FLAGS.smooth_scan))]
    # make it a numpy array
    de = np.asarray(ranges).reshape((1,-1))
    # if list(de.shape) != self.model.output_size: # reshape if necessary
    #   de = sm.resize(de,self.model.output_size,order=1,mode='constant', preserve_range=True)
    return de
    
  def compressed_image_callback(self, msg):
    """ Process serial image data with process_rgb and concatenate frames if necessary"""
    im = self.process_rgb_compressed(msg)
    if len(im)!=0: 
      self.process_input(im)
  
  def image_callback(self, msg):
    """ Process serial image data with process_rgb and concatenate frames if necessary"""
    self.time_im_received.append(time.time())
    im = self.process_rgb(msg)
    if len(im)!=0: 
      if 'nfc' in self.FLAGS.network: # when features are concatenated, multiple images should be kept.
        self.nfc_images.append(im)
        if len(self.nfc_images) < self.FLAGS.n_frames: return
        else:
          # concatenate last n-frames
          im = np.concatenate(np.asarray(self.nfc_images[-self.FLAGS.n_frames:]),axis=2)
          self.nfc_images = self.nfc_images[-self.FLAGS.n_frames+1:] # concatenate last n-1-frames
      self.process_input(im)
    
  def depth_callback(self, msg):
    im = self.process_depth(msg)
    if len(im)!=0 and self.FLAGS.auxiliary_depth:
      self.target_depth = im
  
  def scan_depth_callback(self, msg):
    im = self.process_scan(msg)
    if len(im)!=0:
      self.depth = im
      # calculate depth loss on the fly
      if len(self.depth_prediction) != 0:
        # print("pred: {0} trg: {1}".format(self.depth_prediction, self.depth))
        self.depth_loss.append(np.mean((self.depth_prediction - self.depth.flatten())**2))
  
  def supervised_callback(self, data):
    """Get target control from the /supervised_vel node"""
    # print 'received control'

    if not self.ready: return
    self.target_control = [data.linear.x,
      data.linear.y,
      data.linear.z,
      data.angular.x,
      data.angular.y,
      data.angular.z]

  def finished_callback(self,msg):
    """When run is finished:
        sample 10 batches from the replay buffer,
        apply gradient descent on the model,
        write log file and checkpoints away
    """
    if self.ready and not self.finished:
      print('neural control deactivated. @ time: {}'.format(time.time()))

      self.ready=False
      self.finished=True
      if self.start_time!=0: 
        self.driving_duration = rospy.get_time() - self.start_time

      # Update importance weights if driving duration was long enough
      if self.driving_duration > self.FLAGS.minimum_collision_free_duration and self.FLAGS.update_importance_weights:
        print("[rosinterface]: Update importance weights.")
        self.update_importance_weights()

      
      if self.replay_buffer.size()>=self.FLAGS.batch_size and not self.FLAGS.evaluate:
        losses_train, depth_predictions = self.train_model()
        self.save_summary(losses_train, depth_predictions)
      else:
        self.save_summary()

      self.reset_variables()  

  #--------------------------------
  # Extra functions
  #--------------------------------
  
  def process_input(self, im):
    """Process the inputs: images, targets, auxiliary tasks
      Predict control based on the inputs.
      Plot auxiliary predictions.
      Fill replay buffer.
    """
    # skip a number of frames to lower the actual control rate
    # independently of the image frame rate
    if self.skip_frames != 0:
      self.img_index+=1
      if self.img_index % (self.skip_frames+1) != 0:
        return

    aux_depth=[] # variable to keep predicted depth 
    trgt = []
    
    # Evaluate the input in your network
    trgt=np.array([[self.target_control[5]]]) if len(self.target_control) != 0 else []
    trgt_depth = np.array([copy.deepcopy(self.target_depth)]) if len(self.target_depth) !=0 and self.FLAGS.auxiliary_depth else []
    control, aux_results = self.model.forward([im], auxdepth= not self.FLAGS.dont_show_depth, targets=trgt, depth_targets=trgt_depth)
    if (not self.FLAGS.dont_show_depth) and self.FLAGS.auxiliary_depth and len(aux_results)>0: 
      aux_depth = aux_results['d']
    
    ### SEND CONTROL
    control = control[0]
    
    # POLICY MIXING
    if len(trgt) != 0 and not self.FLAGS.evaluate: # policy mixing with self.FLAGS.alpha
      action = trgt if np.random.binomial(1, self.FLAGS.alpha**(self.runs['train']+1)) else control
    else:
      action = control
    
    msg = Twist()

    # Epsilon-Exploration with exponential decay
    epsilon=self.FLAGS.epsilon*np.exp(-self.runs['train']*self.FLAGS.epsilon_decay)
    
    if self.FLAGS.noise == 'ou':
      noise = self.exploration_noise.noise()
      # exploration noise
      if epsilon > 10**-2 and not self.FLAGS.evaluate: 
        action = noise[3]*self.FLAGS.action_bound if np.random.binomial(1, epsilon) else action
      # general distortion
      msg.linear.y = (not self.FLAGS.evaluate)*noise[1]*self.FLAGS.sigma_y
      msg.linear.z = (not self.FLAGS.evaluate)*noise[2]*self.FLAGS.sigma_z
      msg.angular.z = max(-1,min(1,action+(not self.FLAGS.evaluate)*self.FLAGS.sigma_yaw*noise[3]))    
    elif self.FLAGS.noise == 'uni':
      # exploration noise
      if epsilon  > 10**-2 and not self.FLAGS.evaluate: 
        action = np.random.uniform(-self.FLAGS.action_bound, self.FLAGS.action_bound) if np.random.binomial(1, epsilon) else action
      # general distortion
      # msg.linear.x = self.FLAGS.speed + (not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_x, self.FLAGS.sigma_x)
      msg.linear.y = (not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_y, self.FLAGS.sigma_y)
      msg.linear.z = (not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_z, self.FLAGS.sigma_z)
      msg.angular.z = max(-1,min(1,action+(not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_yaw, self.FLAGS.sigma_yaw)))
    else:
      raise IOError( 'Type of noise is unknown: {}'.format(self.FLAGS.noise))
    
    # if np.abs(msg.angular.z) > 0.3: msg.linear.x =  0.
    if np.abs(msg.angular.z) > 0.3 and self.FLAGS.break_and_turn: 
      msg.linear.x = 0. + self.FLAGS.speed*np.random.binomial(1, 0.1)
    else:
      msg.linear.x = self.FLAGS.speed

    self.action_pub.publish(msg)
    self.time_ctr_send.append(time.time())

    ### keep track of imitation loss on the fly
    if len(self.target_control) != 0:
      self.imitation_loss.append((self.target_control[5]-action)**2)

    if not self.FLAGS.dont_show_depth and len(aux_depth) != 0 and not self.finished:
      aux_depth = aux_depth.flatten()
      self.depth_pub.publish(aux_depth)
      aux_depth = []
      
    # ADD EXPERIENCE REPLAY
    if not self.FLAGS.evaluate and trgt != -100 and not self.finished and len(trgt) != 0:
      experience={'state':im,
                  'action':action,
                  'trgt':trgt}
      if self.FLAGS.auxiliary_depth: experience['target_depth']=trgt_depth
      self.replay_buffer.add(experience)
      # print("added experience: {0} vs {1}".format(action, trgt))

  def backward_step_model(self, inputs, targets, aux_info, losses_train, depth_predictions):
    """Apply gradient step with a backward pass
    """
    # in case the batch size is -1 the full replay buffer is send back
    if self.FLAGS.auxiliary_depth: 
      depth_targets=aux_info['target_depth'].reshape(-1,55,74)
    else:
      depth_targets=[]
  
    if len(inputs) != 0 and len(targets) != 0:
      # losses = self.model.backward(inputs,targets,depth_targets)
      losses = self.model.backward(inputs,targets[:].reshape(-1,1),depth_targets)
    
      for k in losses.keys(): 
        try:
          losses_train[k].append(losses[k])
        except:
          losses_train[k]=[losses[k]]
    else:
      print("[rosinterface]: failed to train due to {0} inputs and {1} targets".format(len(inputs), len(targets)))
    # create a depth prediction map of the auxiliary task on the first batch
    # if b==0 and self.FLAGS.plot_depth and self.FLAGS.auxiliary_depth:
    #   depth_predictions = tools.plot_depth(inputs, aux_info['target_depth'].reshape(-1,55,74))
    
    return losses_train, depth_predictions

  def update_importance_weights(self):
    """Update the importance weights on ALL data in the replay buffer
    """
    # get full replay buffer to calculate new importance weights
    inputs, targets, aux_info = self.replay_buffer.get_all_data(self.FLAGS.max_batch_size)

    # Possible extension: if hard replay buffer combine the two datasets...

    if len(inputs) == 0 or len(targets) == 0: 
      print("Nothing in replay buffer, so nothing to update.")
      return
    self.model.update_importance_weights(inputs)
    # update star variables
    if self.FLAGS.lifelonglearning:
      self.model.sess.run([tf.assign(self.model.star_variables[v.name], v) for v in self.model.copied_trainable_variables])
  
  def train_model(self):
    """Sample a batch from the replay buffer and train on it
    """
    # Train model from experience replay:
    # Train the model with batchnormalization out of the image callback loop
    depth_predictions = []
    losses_train = {}
    
    # get full replay buffer to take one gradient step
    if self.FLAGS.buffer_size == -1 and self.FLAGS.batch_size == -1:
      inputs, targets, aux_info = self.replay_buffer.get_all_data(self.FLAGS.max_batch_size)
      if len(inputs) == 0: 
        return losses_train, depth_predictions
      
      # if there is a hard replay buffer that is full, use it for your batch
      if self.FLAGS.hard_replay_buffer and self.hard_replay_buffer.size() != 0 and len(inputs) != 0:
        print("inputs: {}".format(inputs.shape))
        hard_inputs, hard_targets, hard_aux_info = self.hard_replay_buffer.get_all_data(self.FLAGS.hard_batch_size)
        print("hardinputs: {}".format(hard_inputs.shape))
        inputs = np.concatenate([inputs, hard_inputs], axis=0)
        targets = np.concatenate([targets, hard_targets], axis=0)
        # aux_info = np.concatenate([aux_info, hard_aux_info], axis=0)
        print("inputs: {}".format(inputs.shape))
        print("targets: {}".format(targets.shape))

      losses_train, depth_predictions = self.backward_step_model(inputs, targets, [], losses_train, depth_predictions)
      
      # in case there is a hard replay buffer: fill it with the hardest examples over the inputs
      if self.FLAGS.hard_replay_buffer and len(losses_train.keys()) != 0:
        assert(len(list(losses_train['ce'][0]))==len(list(inputs)))
        assert(len(list(losses_train['ce'][0]))==len(list(targets)))
        sorted_inputs=[np.array(x) for _,x in reversed(sorted(zip(list(losses_train['ce'][0]), inputs.tolist())))]
        sorted_targets=[np.array(y) for _,y in reversed(sorted(zip(list(losses_train['ce'][0]), targets.tolist())))]
        # as all data in hard buffer was in batch we can clear the hard buffer totally
        self.hard_replay_buffer.clear()
        # and gradually add it with the hardest experiences till it is full.
        for e in range(min(self.FLAGS.hard_batch_size, len(sorted_inputs))):
          experience={'state':sorted_inputs[e],
                  'action':None,
                  'trgt':sorted_targets[e],
                  'priority':losses_train['ce'][0][e]} 
          self.hard_replay_buffer.add(experience)
    else:
      # go over all data in the replay buffer over different batches
      # for b in range(min(int(self.replay_buffer.size()/self.FLAGS.batch_size), 10)):
      for b in range(min(int(self.replay_buffer.size()/self.FLAGS.batch_size), self.FLAGS.max_gradient_steps)):
        inputs, targets, aux_info = self.replay_buffer.sample_batch(self.FLAGS.batch_size)
        losses_train, depth_predictions = self.backward_step_model(inputs, targets, aux_info, losses_train, depth_predictions)

    return losses_train, depth_predictions

  def reset_variables(self):
    """After each roll out some field variables should be reset.
    They are collected here.
    """    
    # self.accumlosses = {}
    self.current_distance = 0
    self.last_pose = []
    self.nfc_images = []
    self.furthest_point = 0
    self.world_name = ''
    if self.runs['train']%20==0 and not self.FLAGS.evaluate:
      # Save a checkpoint every 20 runs.
      self.model.save(self.logfolder)
      print('model saved [run {0}]'.format(self.runs['train']))
    self.time_im_received=[]
    self.time_ctr_send=[]

    self.model.reset_metrics()
    
    self.start_time=0
    self.imitation_loss=[]
    self.depth_loss=[]
    self.driving_duration=-1
    self.img_index=0    
    self.fsm_index = 0

    if self.FLAGS.empty_buffer: self.replay_buffer.clear()    

  def save_summary(self, losses_train={}, depth_predictions=[]):
    """Collect all field variables and save them for visualization in tensorboard
    """
    # Gather all info to build a proper summary and string of results
    k='train' if not self.FLAGS.evaluate else 'test'
    self.average_distances[k]= self.average_distances[k]-self.average_distances[k]/(self.runs[k]+1)
    self.average_distances[k] = self.average_distances[k]+self.current_distance/(self.runs[k]+1)
    self.runs[k]+=1
    sumvar={}
    result_string='{0}: run {1}'.format(time.strftime('%H:%M'),self.runs[k])
    vals={'current':self.current_distance, 'furthest':self.furthest_point}
    for d in ['current', 'furthest']:
      name='Distance_{0}_{1}'.format(d,'train' if not self.FLAGS.evaluate else 'test')
      if len(self.world_name)!=0: name='{0}_{1}'.format(name,self.world_name)
      sumvar[name]=vals[d]
      result_string='{0}, {1}:{2}'.format(result_string, name, vals[d])
    for k in losses_train.keys():
      name={'total':'Loss_train_total'}
      name['ce']='Loss_train_ce'
      for lll_k in self.model.lll_losses.keys():
        name['lll_'+lll_k]='Loss_train_lll_'+lll_k
      sumvar[name[k]]=np.mean(losses_train[k])
      result_string='{0}, {1}:{2}'.format(result_string, name[k], np.mean(losses_train[k]))
    
    # get all metrics of this episode and add them to var
    results = self.model.get_metrics()
    for k in results.keys(): 
      sumvar[k] = results[k]
      result_string='{0}, {1}:{2}'.format(result_string, k, results[k])
    
    if self.FLAGS.plot_depth and self.FLAGS.auxiliary_depth:
      sumvar["depth_predictions"]=depth_predictions
    # add driving duration (collision free)
    if self.driving_duration != -1: 
      result_string='{0}, driving_duration: {1:0.3f}'.format(result_string, self.driving_duration)
      sumvar['driving_time']=self.driving_duration
    # add imitation loss
    if len(self.imitation_loss)!=0:
      result_string='{0}, imitation_loss: {1:0.3}'.format(result_string, np.mean(self.imitation_loss))
      sumvar['imitation_loss']=np.mean(self.imitation_loss)
    # add depth loss
    if len(self.depth_loss)!=0:
      result_string='{0}, depth_loss: {1:0.3f}, depth_loss_var: {2:0.3f}'.format(result_string, np.mean(self.depth_loss), np.var(self.depth_loss))
      sumvar['depth_loss']=np.mean(self.depth_loss)
    if len(self.time_ctr_send) > 10 and len(self.time_im_received) > 10:
      # calculate control-rates and rgb-rates from differences
      avg_ctr_rate = 1/np.mean([self.time_ctr_send[i+1]-self.time_ctr_send[i] for i in range(len(self.time_ctr_send)-1)])
      std_ctr_delays = np.std([self.time_ctr_send[i+1]-self.time_ctr_send[i] for i in range(len(self.time_ctr_send)-1)])
      avg_im_rate = 1/np.mean([self.time_im_received[i+1]-self.time_im_received[i] for i in range(1,len(self.time_im_received)-1)]) #skip first image delay as network still needs to 'startup'
      std_im_delays = np.std([self.time_ctr_send[i+1]-self.time_ctr_send[i] for i in range(len(self.time_ctr_send)-1)])

      result_string='{0}, control_rate: {1:0.3f}, image_rate: {2:0.3f} , control_delay_std: {1:0.3f}, image_delay_std: {2:0.3f} '.format(result_string, avg_ctr_rate, avg_im_rate, std_ctr_delays, std_im_delays)
    try:
      self.model.summarize(sumvar)
    except Exception as e:
      print('failed to write', e)
      pass
    else:
      print(result_string)
    # ! Note: tf_log is used by evaluate_model train_model and train_and_evaluate_model in simulation_supervised/scripts
    # Script starts next run once this file is updated.
    try:
      f=open(os.path.join(self.logfolder,'tf_log'),'a')
      f.write(result_string)
      f.write('\n')
      f.close()
    except Exception as e:
      print('failed to write txt tf_log {}'.format(e))
      print('retry after sleep 60')
      time.sleep(60)
      f=open(os.path.join(self.logfolder,'tf_log'),'a')
      f.write(result_string)
      f.write('\n')
      f.close()
