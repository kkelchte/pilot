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
    self.start_time=0
    self.world_name = ''
    self.runs={'train':0, 'test':0} # number of online training run (used for averaging)
    self.accumlosses = {} # gather losses and info over the run in a dictionary
    self.current_distance=0 # accumulative distance travelled from beginning of run used at evaluation
    self.furthest_point=0 # furthest point reached from spawning point at the beginning of run
    self.average_distances={'train':0, 'test':0} # running average over different runs
    self.target_control = [] # field to keep the latest supervised control
    self.target_depth = [] # field to keep the latest supervised depth
    self.nfc_images =[] #used by n_fc networks for building up concatenated frames
    self.exploration_noise = OUNoise(4, 0, self.FLAGS.ou_theta,1)
    if not self.FLAGS.dont_show_depth: self.depth_pub = rospy.Publisher('/depth_prediction', numpy_msg(Floats), queue_size=1)
    self.action_pub=rospy.Publisher('/nn_vel', Twist, queue_size=1)

    rospy.Subscriber('/nn_start', Empty, self.ready_callback)
    rospy.Subscriber('/nn_stop', Empty, self.finished_callback)

    # extract imitation loss from supervised velocity
    rospy.Subscriber('/supervised_vel', Twist, self.supervised_callback)
    self.supervised_vel=[]
    self.imitation_loss=[]
    self.depth_prediction=[]
    self.depth_loss=[]
    self.driving_duration=None
    
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
    if not self.FLAGS.real: # initialize the replay buffer
      self.replay_buffer = ReplayBuffer(self.FLAGS, self.FLAGS.random_seed)
      self.accumloss = 0
      if rospy.has_param('gt_info'):
        rospy.Subscriber(rospy.get_param('gt_info'), Odometry, self.gt_callback)

    # Add some lines to debug delays:
    self.time_im_received=[]
    self.time_ctr_send=[]
    self.time_delay=[]

    rospy.init_node('pilot', anonymous=True)  
    
       
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
        print '--> set evaluate to: {}'.format(self.FLAGS.evaluate)
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
  
  def supervised_callback(self, data):
    """Save supervised velocity command to calculate an imitation loss on the fly."""
    self.supervised_vel=[data.linear.x, 
                        data.linear.y, 
                        data.linear.z, 
                        data.angular.x, 
                        data.angular.y,
                        data.angular.z]

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
      size = self.model.output_size #(55,74)
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
    if list(de.shape) != self.model.output_size: # reshape if necessary
      de = sm.resize(de,self.model.output_size,order=1,mode='constant', preserve_range=True)
    return de
    
  def compressed_image_callback(self, msg):
    """ Process serial image data with process_rgb and concatenate frames if necessary"""
    im = self.process_rgb_compressed(msg)
    if len(im)!=0: 
      self.process_input(im)
  
  def image_callback(self, msg):
    """ Process serial image data with process_rgb and concatenate frames if necessary"""
    rec=time.time()
    im = self.process_rgb(msg)
    if len(im)!=0: 
      if self.FLAGS.n_fc: # when features are concatenated, multiple images should be kept.
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

  def process_input(self, im):
    """Process the inputs: images, targets, auxiliary tasks
      Predict control based on the inputs.
      Plot auxiliary predictions.
      Fill replay buffer.
    """
    aux_depth=[] # variable to keep predicted depth 
    trgt = -100.
    inpt=im
    if self.FLAGS.evaluate: ### EVALUATE
      trgt=np.array([[self.target_control[5]]]) if len(self.target_control) != 0 else []
      trgt_depth = np.array([copy.deepcopy(self.target_depth)]) if len(self.target_depth) !=0 and self.FLAGS.auxiliary_depth else []
      control, losses, aux_results = self.model.forward([inpt], auxdepth=self.FLAGS.show_depth,targets=trgt, depth_targets=trgt_depth)
      for k in ['c', 't', 'd']: 
        if k in losses.keys(): 
          try:
            self.accumlosses[k] += losses[k]
          except KeyError:
            self.accumlosses[k] = losses[k]
      if self.FLAGS.show_depth and self.FLAGS.auxiliary_depth and len(aux_results)>0: aux_depth = aux_results['d']
    else: ###TRAINING
      # Get necessary labels, if label is missing wait...
      def check_field(target_name):
        if len (target_name) == 0:
          print('Waiting for {}'.format(target_name))
          return False
        else:
          return True
      if not check_field(self.target_control): 
        return
      else: 
        trgt = self.target_control[5]
      if self.FLAGS.auxiliary_depth:
        if not check_field(self.target_depth): 
          return
        else: 
          trgt_depth = copy.deepcopy(self.target_depth)
      control, losses, aux_results = self.model.forward([inpt], auxdepth=self.FLAGS.show_depth)
      if self.FLAGS.show_depth and self.FLAGS.auxiliary_depth: aux_depth = aux_results['d']
    
    ### SEND CONTROL
    if trgt != -100 and not self.FLAGS.evaluate: # policy mixing with self.FLAGS.alpha
      action = trgt if np.random.binomial(1, self.FLAGS.alpha**(self.runs['train']+1)) else control[0,0]
    else:
      action = control[0,0]
    if self.FLAGS.discrete:
      # print control
      control = self.model.bin_vals[np.argmax(control)]
    msg = Twist()
    if self.FLAGS.noise == 'ou':
      noise = self.exploration_noise.noise()
      msg.linear.x = self.FLAGS.speed 
      msg.linear.y = (not self.FLAGS.evaluate)*noise[1]*self.FLAGS.sigma_y
      msg.linear.z = (not self.FLAGS.evaluate)*noise[2]*self.FLAGS.sigma_z
      msg.angular.z = max(-1,min(1,action+(not self.FLAGS.evaluate)*self.FLAGS.sigma_yaw*noise[3]))
    elif self.FLAGS.noise == 'uni':
      msg.linear.x = self.FLAGS.speed
      # msg.linear.x = self.FLAGS.speed + (not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_x, self.FLAGS.sigma_x)
      msg.linear.y = (not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_y, self.FLAGS.sigma_y)
      msg.linear.z = (not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_z, self.FLAGS.sigma_z)
      msg.angular.z = max(-1,min(1,action+(not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_yaw, self.FLAGS.sigma_yaw)))
    else:
      raise IOError( 'Type of noise is unknown: {}'.format(self.FLAGS.noise))
    self.action_pub.publish(msg)
    
    # write control to log
    f=open(self.logfolder+'/ctr_log','a')
    f.write("{0} {1} {2} {3} {4} {5} \n".format(msg.linear.x,msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z))
    f.close()

    if not self.finished:
      rec=time.time()
      self.time_ctr_send.append(rec)
      delay=self.time_ctr_send[-1]-self.time_im_received[-1]
      self.time_delay.append(delay)  
    
    if self.FLAGS.show_depth and len(aux_depth) != 0 and not self.finished:
      aux_depth = aux_depth.flatten()
      self.depth_pub.publish(aux_depth)
      aux_depth = []
      
    # ADD EXPERIENCE REPLAY
    if not self.FLAGS.evaluate and trgt != -100:
      aux_info = {}
      if self.FLAGS.auxiliary_depth: aux_info['target_depth']=trgt_depth
      self.replay_buffer.add(im,[trgt],aux_info=aux_info)

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
      
      # Train model from experience replay:
      # Train the model with batchnormalization out of the image callback loop
      depth_predictions = []
      losses_train = {}
      if self.replay_buffer.size()>self.FLAGS.batch_size and not self.FLAGS.evaluate:
        for b in range(min(int(self.replay_buffer.size()/self.FLAGS.batch_size), 10)):
          inputs, targets, aux_info = self.replay_buffer.sample_batch(self.FLAGS.batch_size)
          if b==0:
            if self.FLAGS.plot_depth and self.FLAGS.auxiliary_depth:
              depth_predictions = tools.plot_depth(inputs, aux_info['target_depth'].reshape(-1,55,74))
          depth_targets=[]
          if self.FLAGS.auxiliary_depth: 
            depth_targets=aux_info['target_depth'].reshape(-1,55,74)
          losses = self.model.backward(inputs,targets[:].reshape(-1,1),depth_targets)
          for k in losses.keys(): 
            try:
              losses_train[k].append(losses[k])
            except:
              losses_train[k]=[losses[k]]
      
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
        name={'t':'Loss_train_total','c':'Loss_train_control','d':'Loss_train_depth'}
        sumvar[name[k]]=np.mean(losses_train[k])
        result_string='{0}, {1}:{2}'.format(result_string, name[k], np.mean(losses_train[k]))
      for k in self.accumlosses.keys():
        name={'t':'Loss_test_total','c':'Loss_test_control','d':'Loss_test_depth'}
        sumvar[name[k]]=self.accumlosses[k]
        result_string='{0}, {1}:{2}'.format(result_string, name[k], self.accumlosses[k]) 
      if self.FLAGS.plot_depth and self.FLAGS.auxiliary_depth:
        sumvar["depth_predictions"]=depth_predictions
      result_string='{0}, delays: {1:0.3f} | {2:0.3f} | {3:0.3f} | '.format(result_string, np.min(self.time_delay[1:]), np.mean(self.time_delay[1:]), np.max(self.time_delay))
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
      self.accumlosses = {}
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
      self.time_delay=[]
    
      

