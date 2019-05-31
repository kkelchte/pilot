import rospy
import numpy as np
# import scipy.misc as sm
import skimage.io as sio
import skimage.transform as sm

import sys, time, re, copy, cv2, os
from os import path

import collections

from tf import transformations

from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

from replay_buffer import ReplayBuffer
from model import Model
from ou_noise import OUNoise
import tools
import online

import torch

from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from nav_msgs.msg import Odometry

from std_srvs.srv import Empty as Emptyservice
from std_srvs.srv import EmptyRequest # for pausing and unpausing physics engine


# import matplotlib.animation as animation
# import matplotlib.pyplot as plt


#from PIL import Image

class PilotNode(object):
  """Node to listen to ROS topics like depth, rgb input and supervised control.
  The node also publishes to pilot control and predicted depth for visualization.
  """
  
  def __init__(self, FLAGS, model, logfolder):
    print('[rosinterface] initialize pilot node')  
    self.FLAGS=FLAGS
    # Initialize fields
    self.logfolder = logfolder    
    self.model = model
    self.epoch = model.epoch 
    self.ready=False 
    self.finished=True
    self.training=False
    
    self.last_pose=[] # previous pose, used for accumulative distance
    self.world_name = ''
    self.runs={'train':0, 'test':0} # number of online training run (used for averaging)
    # self.accumlosses = {} # gather losses and info over the run in a dictionary
    self.current_distance=0 # accumulative distance travelled from beginning of run used at evaluation
    self.furthest_point=0 # furthest point reached from spawning point at the beginning of run
    self.initial_pos=[] # save starting position of drone  
    self.average_distances={'train':0, 'test':0} # running average over different runs
    self.target_control = [] # field to keep the latest supervised control
    self.target_depth = [] # field to keep the latest supervised depth
    self.nfc_images =[] #used by n_fc networks for building up concatenated frames
    self.exploration_noise = OUNoise(1, 0, self.FLAGS.ou_theta,self.FLAGS.sigma_yaw)
    # if not self.FLAGS.dont_show_depth: self.depth_pub = rospy.Publisher('/depth_prediction', numpy_msg(Floats), queue_size=1)
    
    self.overtake_pub=rospy.Publisher('/overtake', Empty, queue_size=1)
    self.action_pub=rospy.Publisher('/nn_vel', Twist, queue_size=1)
    rospy.Subscriber('/nn_start', Empty, self.ready_callback)
    rospy.Subscriber('/nn_stop', Empty, self.finished_callback)
    # extract imitation loss from supervised velocity or use it for policy mixing
    rospy.Subscriber('/supervised_vel', Twist, self.supervised_callback)
    
    self.start_time = 0
    self.imitation_loss=[]
    self.confidences=[]
    self.depth_prediction=[]
    self.depth_loss=[]
    self.driving_duration=-1

    self.skip_frames = 0
    self.img_index = 0
    self.fsm_index = 0

    if 'LSTM' in self.FLAGS.network:
      self.hidden_states=tools.get_hidden_state([], self.model, astype='numpy')
      
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
    
    # first see if a replaybuffer is within the my-model torch checkpoint.
    self.replay_buffer=tools.load_replaybuffer_from_checkpoint(FLAGS)
    if not self.replay_buffer: #other wise create a new.
      self.replay_buffer=ReplayBuffer(buffer_size=FLAGS.buffer_size, random_seed=FLAGS.random_seed)

    # self.replay_buffer = ReplayBuffer(self.FLAGS.buffer_size, self.FLAGS.random_seed, checkpoint=self.logfolder+'/replaybuffer')
    # if self.FLAGS.hard_replay_buffer:
    #   self.hard_replay_buffer = ReplayBuffer(self.FLAGS.hard_batch_size, self.FLAGS.random_seed, checkpoint=self.logfolder+'/hardbuffer')
      
    self.accumloss = 0
    if rospy.has_param('gt_info'):
      rospy.Subscriber(rospy.get_param('gt_info'), Odometry, self.gt_callback)

    # Recovery cameras
    if rospy.has_param('recovery'):
      if rospy.get_param('recovery') and rospy.has_param('rgb_image_left') and rospy.has_param('rgb_image_right'):
        print("[rosinterface]: using recovery cameras with topics: {0} and {1}".format(rospy.get_param('rgb_image_left'),rospy.get_param('rgb_image_right')))
        rospy.Subscriber(rospy.get_param('rgb_image_left'), Image, self.image_callback, callback_args='left')
        rospy.Subscriber(rospy.get_param('rgb_image_right'), Image, self.image_callback, callback_args='right')
    self.recovery_images={}

    # Add some lines to debug delays:
    self.time_im_received=[]
    self.time_ctr_send=[]

    # initialize ROS node
    rospy.init_node('pilot', anonymous=True)

    # Pausing and unpausing gazebo physics simulator
    if self.FLAGS.pause_simulator:
      self.pause_physics_client=rospy.ServiceProxy('/gazebo/pause_physics',Emptyservice)
      self.unpause_physics_client=rospy.ServiceProxy('/gazebo/unpause_physics',Emptyservice)
    

    # write nn_ready to indicate to run_script initialization is finished.
    f=open(os.path.join(self.logfolder,'nn_ready'),'a')
    f.write("{0}: start {1}".format(time.strftime('%H.%M.%S'),self.FLAGS.log_tag))
    f.write('\n')
    f.close()

  #--------------------------------
  # Callbacks
  #--------------------------------

  def ready_callback(self,msg):
    """ callback function that makes DNN policy starts the ready flag is set on 1 (for 3s)"""
    if not self.ready and self.finished:
      print('Neural control activated.')
      self.ready = True
      self.finished = False
      self.start_time = rospy.get_time()
      self.exploration_noise.reset()
      # choose one speed for this flight
      self.FLAGS.speed=self.FLAGS.speed + (not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_x, self.FLAGS.sigma_x)
      if rospy.has_param('/evaluate'):
        self.FLAGS.evaluate = rospy.get_param('/evaluate')
        print('--> set evaluate to: {0} with speed {1}'.format(self.FLAGS.evaluate, self.FLAGS.speed))
      if rospy.has_param('skip_frames'):
        self.skip_frames = rospy.get_param('skip_frames')
        print('--> set skip_frames to: {0}'.format(self.skip_frames))
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
    if len(self.initial_pos) == 0:
      self.initial_pos=current_pos
    
    self.furthest_point=max([self.furthest_point, np.sqrt((current_pos[0]-self.initial_pos[0])**2+(current_pos[1]-self.initial_pos[1])**2)])

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
      if self.FLAGS.normalized_input:
        input_normalization='normalized'
      elif self.FLAGS.scaled_input:
        input_normalization='scaled'
      elif self.FLAGS.skew_input:
        input_normalization='skewinput'
      else:
        input_normalization='none'
      img=tools.load_rgb(im_object=img, 
                        im_size=self.model.input_size,
                        im_mode='CHW',
                        im_norm=input_normalization,
                        im_means=self.FLAGS.normalize_means,
                        im_stds=self.FLAGS.normalize_stds)
      return img

  def process_rgb_compressed(self, msg):
    """ [DEPRECATED] Convert RGB serial data to opencv image of correct size
    """
    # if not self.ready or self.finished: return []
    try:
      img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
    except CvBridgeError as e:
      print(e)
    else:
      # 308x410 to 128x128
      img = img[::2,::3,:]
      img = sm.resize(img,self.model.input_size,mode='constant').astype(np.float16)
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
  
  def image_callback(self, msg, camera_type='straight'):
    """ Process serial image data with process_rgb and concatenate frames if necessary"""
    # print("[rosinterface] got image of camera type: {0}".format(camera_type))
    self.time_im_received.append(time.time())
    im = self.process_rgb(msg)
    if len(im)!=0: 
      # when features are concatenated, multiple images should be kept.
      if 'nfc' in self.FLAGS.network or '3d' in self.FLAGS.network: 
        self.nfc_images.append(im)
        if len(self.nfc_images) < self.FLAGS.n_frames: return
        else:
          # concatenate last n-frames
          # im = np.concatenate(np.asarray(self.nfc_images[-self.FLAGS.n_frames:]),axis=2)
          if '3d' in self.FLAGS.network:
            im = np.concatenate(self.nfc_images, axis=0)
          elif 'nfc' in self.FLAGS.network:
            im = np.asarray(self.nfc_images)
          self.nfc_images = self.nfc_images[-self.FLAGS.n_frames+1:] 
      if camera_type=='straight':
        self.process_input(im)
      else:
        self.recovery_images[camera_type]=im
    
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
      if os.path.isfile(self.logfolder+'/fsm_log'):
        bump=''
        with open(self.logfolder+'/fsm_log', 'r') as f:
          bump=f.readlines()[-1].strip().split(' ')[0]
        # print("[rosinterface]: found fsm_file: {0}".format(bump))
        if bump == 'BUMP' and not self.FLAGS.evaluate:
          self.replay_buffer.annotate_collision(self.FLAGS.horizon)
        # elif bump == 'success':
        #   pass
        # else:
        #   bump='BUMP'

      # export full buffer
      if not self.FLAGS.evaluate and self.FLAGS.export_buffer:
        self.pause_physics_client(EmptyRequest())
        data_folder=self.FLAGS.data_root+self.FLAGS.log_tag+'/{0:05d}'.format(self.runs['train'])
        self.replay_buffer.export_buffer(data_folder)
        self.unpause_physics_client(EmptyRequest())
      
      # save separate run logging
      self.save_end_run(bump)
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
    # Pause Gazebo
    if self.FLAGS.pause_simulator and self.ready:
      # print("[rosinterface] {0} pause simulator".format(time.strftime('%H:%M')))
      self.pause_physics_client(EmptyRequest())  

    # keep track of pausing duration
    pause_start = time.time()
    
    # skip a number of frames to lower the actual control rate
    # independently of the image frame rate
    if self.skip_frames != 0:
      self.img_index+=1
      if self.img_index % (self.skip_frames+1) != 0:
        return

    aux_depth=[] # variable to keep predicted depth 
    
    # Evaluate the input in your network
    trgt=np.array([[self.target_control[5]]]) if len(self.target_control) != 0 else []
    trgt_depth = np.array([copy.deepcopy(self.target_depth)]) if len(self.target_depth) !=0 and self.FLAGS.auxiliary_depth else []
    
    inputs=np.array([im])
    if 'LSTM' in self.FLAGS.network:
      h_t, c_t = (torch.from_numpy(self.hidden_states[0]),torch.from_numpy(self.hidden_states[1]))
      inputs=(torch.from_numpy(np.expand_dims(inputs, axis=0)).type(torch.FloatTensor).to(self.model.device),(h_t.to(self.model.device),c_t.to(self.model.device)))
    control, losses, self.hidden_states = self.model.predict(inputs)
    
    if 'confidence' in losses.keys():
      self.confidences.append(losses['confidence'])
      print(losses['confidence'])

    ### SEND CONTROL
    if isinstance(control, collections.Iterable):
      control = control[0]
    
    ### IMITATION LOSS
    if len(self.target_control) != 0:
      loss=(self.target_control[5]-control)**2
      self.imitation_loss.append(loss)
      # print("ctr: {0}, trgt: {1}, loss:{2}".format(control, self.target_control[5], loss))

    # POLICY MIXING
    if len(trgt) != 0: # policy mixing with self.FLAGS.alpha
      action = trgt if np.random.binomial(1, self.FLAGS.alpha) else control
      # action = trgt if np.random.binomial(1, self.FLAGS.alpha**(self.runs['train']+1)) else control
    else:
      action = control
    
    msg = Twist()

    # add noise over yaw
    if self.FLAGS.noise == 'ou':
      noise = self.exploration_noise.noise()[0]
    elif self.FLAGS.noise == 'uni':
      noise = np.random.uniform(-self.FLAGS.sigma_yaw,self.FLAGS.sigma_yaw)
    elif self.FLAGS.noise == 'gau':
      noise = np.random.normal(0,self.FLAGS.sigma_yaw)
    else:
      noise = 0
    msg.angular.z = max(-1,min(1,action+noise))    
    
    # # Epsilon-Exploration with exponential decay
    # epsilon=self.FLAGS.epsilon*np.exp(-self.runs['train']*self.FLAGS.epsilon_decay)
    
    # if self.FLAGS.noise == 'ou':
    #   noise = self.exploration_noise.noise()
    #   # exploration noise
    #   # if epsilon > 10**-2 and not self.FLAGS.evaluate: 
    #   if epsilon > 10**-2: 
    #     action = noise[3]*self.FLAGS.action_bound if np.random.binomial(1, epsilon) else action
    #   # general distortion
    #   msg.linear.y = (not self.FLAGS.evaluate)*noise[1]*self.FLAGS.sigma_y
    #   msg.linear.z = (not self.FLAGS.evaluate)*noise[2]*self.FLAGS.sigma_z
    #   msg.angular.z = max(-1,min(1,action+(not self.FLAGS.evaluate)*self.FLAGS.sigma_yaw*noise[3]))    
    # elif self.FLAGS.noise == 'uni':
    #   # exploration noise
    #   if epsilon > 10**-2 and not self.FLAGS.evaluate: 
    #     action = np.random.uniform(-self.FLAGS.action_bound, self.FLAGS.action_bound) if np.random.binomial(1, epsilon) else action
    #   # general distortion
    #   # msg.linear.x = self.FLAGS.speed + (not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_x, self.FLAGS.sigma_x)
    #   msg.linear.y = (not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_y, self.FLAGS.sigma_y)
    #   msg.linear.z = (not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_z, self.FLAGS.sigma_z)
    #   msg.angular.z = max(-1,min(1,action+(not self.FLAGS.evaluate)*np.random.uniform(-self.FLAGS.sigma_yaw, self.FLAGS.sigma_yaw)))
    # else:
    #   raise IOError( 'Type of noise is unknown: {}'.format(self.FLAGS.noise))
    
    if np.abs(msg.angular.z) > 0.3: 
      msg.linear.x =  self.FLAGS.turn_speed
      # if np.abs(msg.angular.z) > 0.3 and self.FLAGS.break_and_turn: 
      #   msg.linear.x = 0. + self.FLAGS.speed*np.random.binomial(1, 0.1)
    else:
      msg.linear.x = self.FLAGS.speed

    self.action_pub.publish(msg)
    self.time_ctr_send.append(time.time())
    
    # call ONLINE METHOD to collect data in buffer and train model
    # if not self.FLAGS.evaluate and not self.finished and len(trgt) != 0 and not self.FLAGS.no_training:
    if not self.finished:
      experience={'state':im,
                  'action':float(action),
                  'speed':msg.linear.x,
                  'collision':0}
      if len(trgt) != 0: experience['trgt']=np.squeeze(trgt) 
      if self.FLAGS.auxiliary_depth: 
        experience['target_depth']=trgt_depth
      online.method(self.model, experience, self.replay_buffer, self.replay_buffer.get_details())
      
      # Train on experiences from recovery cameras
      for k in self.recovery_images.keys():
        if len(self.recovery_images[k])!=0:
          experience={'state':self.recovery_images[k][:],
                  'action':float(action),
                  'speed':msg.linear.x,
                  'collision':0}
          if len(trgt) != 0: experience['trgt']=np.squeeze(trgt)+self.FLAGS.recovery_compensation if k == 'right' else np.squeeze(trgt)-self.FLAGS.recovery_compensation
          del self.recovery_images[k]
          online.method(self.model, experience, self.replay_buffer)
          # self.replay_buffer.add(experience)

      if not self.FLAGS.evaluate:        
        if self.epoch > self.FLAGS.max_episodes:
          self.overtake_pub.publish(Empty())
          self.model.save(self.logfolder, replaybuffer=self.replay_buffer)
          try:
            with open(self.logfolder+'/fsm_log', 'a') as f: f.write('FINISHED\n')
          except:
            print('[rosinterface]: failed to write logfile: {}'.format(self.logfolder+'/fsm_log'))

    # Unpause the simulator
    if self.FLAGS.pause_simulator and self.ready: self.unpause_physics_client(EmptyRequest()) 
  
  def reset_variables(self):
    """After each roll out some field variables should be reset.
    They are collected here.
    """    
    # self.accumlosses = {}
    self.current_distance = 0
    self.last_pose = []
    self.nfc_images = []
    self.furthest_point = 0
    self.initial_pos=[]    
    self.world_name = ''
    if self.runs['train']%5==0 and not self.FLAGS.evaluate:
      # Save a checkpoint every 20 runs.
      self.model.save(self.logfolder)
      print('model saved [run {0}]'.format(self.runs['train']))
    self.time_im_received=[]
    self.time_ctr_send=[]
    
    self.start_time=0
    self.imitation_loss=[]
    self.confidences=[]
    self.depth_loss=[]
    self.driving_duration=-1
    self.img_index=0    
    self.fsm_index = 0

    if 'LSTM' in self.FLAGS.network:
      self.hidden_states=tools.get_hidden_state([], self.model, astype='numpy')
    # if self.FLAGS.empty_buffer: self.replay_buffer.clear()    

  
  def save_end_run(self, result=""):
    """At the end of a run, update nn_ready so run_script knows it can start the next run.
    """
    run_type='train' if not self.FLAGS.evaluate else 'test'
    self.average_distances[run_type]= self.average_distances[run_type]-self.average_distances[run_type]/(self.runs[run_type]+1)
    self.average_distances[run_type] = self.average_distances[run_type]+self.current_distance/(self.runs[run_type]+1)
    self.runs[run_type]+=1

    if self.FLAGS.save_CAM_images: self.FLAGS.save_CAM_images = False
    if self.FLAGS.save_annotated_images: self.FLAGS.save_annotated_images = False

    sumvar={}
    result_string="start_time: {0}, run_number: {1}, run_type: {2}".format(time.strftime('%H.%M.%S'), self.runs[run_type], run_type)

    if len(result)!= 0: 
      result_string='{0}, {2}_result: {1}'.format(result_string, result, run_type)
      result_string='{0}, {2}_success: {1}'.format(result_string, float(result=='success'), run_type)

    vals={'current':self.current_distance, 'furthest':self.furthest_point}
    for d in ['current', 'furthest']:
      name='Distance_{0}_{1}'.format(d,'train' if not self.FLAGS.evaluate else 'test')
      if len(self.world_name)!=0: name='{0}_{1}'.format(name,self.world_name)
      sumvar[name]=vals[d]
      result_string='{0}, {1}:{2:0.3f}'.format(result_string, name, vals[d])
    
    # add driving duration (collision free)
    if self.driving_duration != -1: 
      result_string='{0}, run_driving_duration: {1:0.3f}'.format(result_string, self.driving_duration)
      sumvar['run_driving_time']=self.driving_duration
    # add imitation loss
    if len(self.imitation_loss)!=0:
      result_string='{0}, run_imitation_loss: {1:0.3f}'.format(result_string, np.mean(self.imitation_loss))
      sumvar['run_imitation_loss']=np.mean(self.imitation_loss)
    # add confidence
    if len(self.confidences)!=0:
      result_string='{0}, confidence: {1:0.3f}, confidence_std: {2:0.3f}'.format(result_string, np.mean(self.confidences), np.mean(self.confidences))
      sumvar['confidence']=np.mean(self.confidences)

    
    if len(self.time_ctr_send) > 10 and len(self.time_im_received) > 10:
      # calculate control-rates and rgb-rates from differences
      avg_ctr_rate = 1/np.mean([self.time_ctr_send[i+1]-self.time_ctr_send[i] for i in range(len(self.time_ctr_send)-1)])
      std_ctr_delays = np.std([self.time_ctr_send[i+1]-self.time_ctr_send[i] for i in range(len(self.time_ctr_send)-1)])
      avg_im_rate = 1/np.mean([self.time_im_received[i+1]-self.time_im_received[i] for i in range(1,len(self.time_im_received)-1)]) #skip first image delay as network still needs to 'startup'
      std_im_delays = np.std([self.time_ctr_send[i+1]-self.time_ctr_send[i] for i in range(len(self.time_ctr_send)-1)])

      result_string='{0}, run_rate_control: {1:0.3f}, run_rate_image: {2:0.3f} , run_delay_std_control: {1:0.3f}, run_delay_std_image: {2:0.3f} '.format(result_string, avg_ctr_rate, avg_im_rate, std_ctr_delays, std_im_delays)
    
    if self.FLAGS.save_every_num_epochs == 0:
      self.model.save(self.logfolder)

    # write it to tensorboard, output and nn_ready file.
    try:
      self.model.summarize(sumvar)
    except Exception as e:
      print('failed to write', e)
      pass
    print(result_string)
    try:
      with open(self.logfolder+'/nn_ready','a') as f:
        f.write(result_string+' \n')
    except Exception as e:
      print('failed to write txt {0}/nn_ready {1}'.format(self.logfolder, e))



  # def save_summary(self, losses_train={}, extra_info={}):
  #   """Collect all field variables and save them for visualization in tensorboard
  #   """
  #   # Gather all info to build a proper summary and string of results
  #   sumvar={}
  #   result_string='time: {0}, epoch: {1}'.format(time.strftime('%H.%M.%S'),self.epoch)
  #   for k in sorted(losses_train.keys()):
  #     sumvar[k]=np.mean(losses_train[k])
  #     result_string='{0}, {1}:{2:0.5f}'.format(result_string, k, np.mean(losses_train[k]))
  #   for k in sorted(extra_info.keys()):
  #     sumvar[k]=extra_info[k]
  #     result_string='{0}, {1}:{2:0.2f}'.format(result_string, k, extra_info[k])
  #   try:
  #     self.model.summarize(sumvar)
  #   except Exception as e:
  #     print('failed to write', e)
  #     pass
  #   print(result_string)
  #   # ! Note: nn_log is used by evaluate_model train_model and train_and_evaluate_model in simulation_supervised/scripts
  #   # Script starts next run once this file is updated.
  #   try:
  #     with open(os.path.join(self.logfolder,'nn_log'),'a') as f:
  #       f.write(result_string+' \n')
  #   except Exception as e:
  #     print('failed to write txt {0}/nn_log {1}'.format(self.logfolder, e))
      
  # def train_model(self):
  #   """DEPRECATED Sample a batch from the replay buffer and train on it
  #   """
  #   losses_train = {}

  #   # take multiple gradient steps, each on a fresh sampled batch of data
  #   for grad_step in range(self.FLAGS.gradient_steps):
  #     if (grad_step%100) == 99:
  #       print("[rosinterface]: take step {0} of {1}".format(grad_step, self.FLAGS.gradient_steps))
  #     # sampling grows less than linear with batch size
  #     inputs, targets, actions, collisions = self.replay_buffer.sample_batch(self.FLAGS.batch_size, horizon=self.FLAGS.horizon if self.FLAGS.il_weight != 1 else 0)
  #     # training increases more than linear with batch size
  #     self.epoch, predictions, losses, hidden_states = self.model.train(inputs,targets.reshape([-1,1]), actions.reshape([-1,1]), collisions.reshape([-1,1]))
  #     for k in losses.keys(): tools.save_append(losses_train, k, losses[k])
  #     if self.FLAGS.gradient_steps > 100 and (grad_step%100) == 99:
  #       self.save_summary(losses)

  #   self.replay_buffer.update(self.FLAGS.buffer_update_rule)

  #   return losses_train

  # def update_importance_weights(self):
  #   """DEPRECATED Update the importance weights on ALL data in the replay buffer
  #   """
  #   # get full replay buffer to calculate new importance weights
  #   inputs, targets, aux_info = self.replay_buffer.get_all_data(self.FLAGS.max_batch_size)

  #   # Possible extension: if hard replay buffer combine the two datasets...

  #   if len(inputs) == 0 or len(targets) == 0: 
  #     print("Nothing in replay buffer, so nothing to update.")
  #     return
  #   self.model.update_importance_weights(inputs)
  #   # update star variables
  #   if self.FLAGS.lifelonglearning:
  #     self.model.sess.run([tf.assign(self.model.star_variables[v.name], v) for v in self.model.copied_trainable_variables])

  # def train_model_old(self):
  #   """Sample total recent and hard buffer and loop over it with several gradient steps.
  #   """
  #   # Train model from experience replay:
  #   # Train the model with batchnormalization out of the image callback loop
  #   losses_train = {}

  #   # 1. get the data from the replay buffer shuffled
  #   if self.replay_buffer.size() > self.FLAGS.batch_size * 2:
  #     inputs, targets, actions, collisions = self.replay_buffer.get_all_data_shuffled(horizon=self.FLAGS.horizon if self.FLAGS.il_weight != 1 else 0)
  #   else:
  #     inputs, targets, actions, collisions = self.replay_buffer.get_all_data(horizon=self.FLAGS.horizon if self.FLAGS.il_weight != 1 else 0)
  #   if len(inputs) == 0: return self.epoch, losses_train
    
  #   # 1.b if there is a hard replay buffer that is full, use it for your batch
  #   if self.FLAGS.hard_replay_buffer and self.hard_replay_buffer.size() != 0 and len(inputs) != 0:
  #     hard_inputs, hard_targets, hard_actions, hard_collisions = self.hard_replay_buffer.get_all_data()
  #     # print("hardinputs: {}".format(hard_inputs.shape))
  #     inputs = np.concatenate([inputs, hard_inputs], axis=0)
  #     targets = np.concatenate([targets, hard_targets], axis=0)
  #     actions = np.concatenate([actions, hard_actions], axis=0)
  #     collisions = np.concatenate([collisions, hard_collisions], axis=0)

  #   # 2. Loop over data grad_step times in batches
  #   for grad_step in range(self.FLAGS.gradient_steps):
  #     b=0
  #     while b < len(inputs)-30: #don't train on a batch smaller than 30, it's not worth the delay.
  #       if len(inputs) >= b+self.FLAGS.batch_size:
  #         input_batch = inputs[b:b+self.FLAGS.batch_size]
  #         target_batch = targets[b:b+self.FLAGS.batch_size]
  #         action_batch = actions[b:b+self.FLAGS.batch_size]
  #         collision_batch = collisions[b:b+self.FLAGS.batch_size]
  #       else:  
  #         input_batch = inputs[b:]
  #         target_batch = targets[b:]
  #         action_batch = actions[b:]
  #         collision_batch = collisions[b:]
  #       epoch, losses_train = self.backward_step_model(input_batch, target_batch, action_batch, collision_batch, losses_train)
  #       b+=self.FLAGS.batch_size
      
  #   # 3. in case there is a hard replay buffer: fill it with the hardest examples over the inputs
  #   if self.FLAGS.hard_replay_buffer and len(losses_train.keys()) != 0:
  #     inputs=inputs[:len(losses_train['Loss_train_total'])]
  #     targets=targets[:len(losses_train['Loss_train_total'])]
  #     actions=actions[:len(losses_train['Loss_train_total'])]
  #     collisions=collisions[:len(losses_train['Loss_train_total'])]
      
  #     assert len(losses_train['Loss_train_total'])==len(inputs), "[rosinterface] Failed to update hard buffer due to {0} losses while {1} inputs.".format(len(losses_train['Loss_train_total']),len(inputs))
  #     sorted_inputs=[np.array(x) for _,x in reversed(sorted(zip(list(losses_train['Loss_train_total']), inputs.tolist())))]
  #     sorted_targets=[np.array(y) for _,y in reversed(sorted(zip(list(losses_train['Loss_train_total']), targets.tolist())))]
  #     sorted_actions=[np.array(y) for _,y in reversed(sorted(zip(list(losses_train['Loss_train_total']), actions.tolist())))]
  #     sorted_collisions=[np.array(y) for _,y in reversed(sorted(zip(list(losses_train['Loss_train_total']), collisions.tolist())))]
  #     # as all data in hard buffer was in batch we can clear the hard buffer totally
  #     self.hard_replay_buffer.clear()
  #     # and gradually add it with the hardest experiences till it is full.
  #     for e in range(min(self.FLAGS.hard_batch_size, len(sorted_inputs))):
  #       experience={'state':sorted_inputs[e],
  #               'action':sorted_actions[e],
  #               'collision':sorted_collisions[e],
  #               'trgt':sorted_targets[e],
  #               'priority':losses_train['Loss_train_total'][e]} 
  #       self.hard_replay_buffer.add(experience)
  
  #   return epoch, losses_train

  # def backward_step_model(self, inputs, targets, actions, collisions, losses_train):
  #   """DEPRECATED Apply gradient step with a backward pass
  #   """
  #   # in case the batch size is -1 the full replay buffer is send back 
  #   if len(inputs) != 0 and len(targets) != 0:
  #     epoch, predictions, losses, hidden_states = self.model.train(inputs,targets.reshape([-1,1]), actions.reshape([-1,1]), collisions.reshape([-1,1]))
  #     for k in losses.keys(): tools.save_append(losses_train, k, losses[k])
  #   else:
  #     print("[rosinterface]: failed to train due to {0} inputs and {1} targets".format(len(inputs), len(targets)))    
  #   return epoch, losses_train     
  
