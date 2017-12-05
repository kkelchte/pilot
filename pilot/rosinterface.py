import rospy
import numpy as np
import scipy.misc as sm
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

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from nav_msgs.msg import Odometry



#from PIL import Image

FLAGS = tf.app.flags.FLAGS
# =================================================
tf.app.flags.DEFINE_integer("buffer_size", 2000, "Define the number of experiences saved in the buffer.")
tf.app.flags.DEFINE_float("ou_theta", 0.15, "Theta is the pull back force of the OU Noise.")
tf.app.flags.DEFINE_string("type_of_noise", 'ou', "Define whether the noise is temporally correlated (ou) or uniformly distributed (uni).")
tf.app.flags.DEFINE_float("sigma_z", 0.01, "sigma_z is the amount of noise in the z direction.")
tf.app.flags.DEFINE_float("sigma_x", 0.01, "sigma_x is the amount of noise in the forward speed.")
tf.app.flags.DEFINE_float("sigma_y", 0.01, "sigma_y is the amount of noise in the y direction.")
tf.app.flags.DEFINE_float("sigma_yaw", 0.1, "sigma_yaw is the amount of noise added to the steering angle.")
tf.app.flags.DEFINE_float("speed", 1.3, "Define the forward speed of the quadrotor.")
tf.app.flags.DEFINE_float("alpha",0.,"Policy mixing: choose with a binomial probability of alpha for the experts policy instead of the DNN policy.")

tf.app.flags.DEFINE_boolean("off_policy",False,"In case the network is off_policy, the control is published on supervised_vel instead of cmd_vel.")
tf.app.flags.DEFINE_boolean("show_depth",True,"Publish the predicted horizontal depth array to topic ./depth_prection so show_depth can visualize this in another node.")
# =================================================

class PilotNode(object):
  """Node to listen to ROS topics like depth, rgb input and supervised control.
  The node also publishes to pilot control and predicted depth for visualization.
  """
  
  def __init__(self, model, logfolder):
    print('initialize pilot node')  
    # Initialize fields
    self.logfolder = logfolder
    f=open(os.path.join(self.logfolder,'tf_log'),'a')
    f.write(FLAGS.log_tag)
    f.write('\n')
    f.close()
    self.world_name = ''
    self.runs={'train':0, 'test':0} # number of online training run (used for averaging)
    self.accumlosses = {} # gather losses and info over the run in a dictionary
    self.current_distance=0 # accumulative distance travelled from beginning of run used at evaluation
    self.furthest_point=0 # furthest point reached from spawning point at the beginning of run
    self.average_distances={'train':0, 'test':0} # running average over different runs
    self.last_pose=[] # previous pose, used for accumulative distance
    self.model = model 
    self.ready=False 
    self.finished=True
    self.target_control = [] # field to keep the latest supervised control
    self.target_depth = [] # field to keep the latest supervised depth
    self.nfc_images =[] #used by n_fc networks for building up concatenated frames
    self.exploration_noise = OUNoise(4, 0, FLAGS.ou_theta,1)
    if FLAGS.show_depth: self.depth_pub = rospy.Publisher('/depth_prediction', numpy_msg(Floats), queue_size=1)
    if FLAGS.real or FLAGS.off_policy: # publish on pilot_vel so it can be used by control_mapping when flying in the real world
      self.action_pub=rospy.Publisher('/pilot_vel', Twist, queue_size=1)
    else: # if you fly in simulation, listen to supervised vel to get the target control from the BA expert
      rospy.Subscriber('/supervised_vel', Twist, self.supervised_callback)
      # the control topic is defined in the drone_sim yaml file
      if rospy.has_param('control'): self.action_pub = rospy.Publisher(rospy.get_param('control'), Twist, queue_size=1)
    if rospy.has_param('ready'): rospy.Subscriber(rospy.get_param('ready'), Empty, self.ready_callback)
    if rospy.has_param('finished'): rospy.Subscriber(rospy.get_param('finished'), Empty, self.finished_callback)
    if rospy.has_param('rgb_image'): rospy.Subscriber(rospy.get_param('rgb_image'), Image, self.image_callback)
    if rospy.has_param('depth_image') and FLAGS.auxiliary_depth:
        rospy.Subscriber(rospy.get_param('depth_image'), Image, self.depth_callback)
    if not FLAGS.real: # initialize the replay buffer
      self.replay_buffer = ReplayBuffer(FLAGS.buffer_size, FLAGS.random_seed)
      self.accumloss = 0
      rospy.Subscriber('/ground_truth/state', Odometry, self.gt_callback)

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
      FLAGS.speed=FLAGS.speed + (not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_x, FLAGS.sigma_x)
      if rospy.has_param('evaluate') and not FLAGS.real:
        FLAGS.evaluate = rospy.get_param('evaluate')
        print '--> set evaluate to: {}'.format(FLAGS.evaluate)
      if rospy.has_param('world_name') :
        self.world_name = os.path.basename(rospy.get_param('world_name').split('.')[0])
        if 'sandbox' in self.world_name: self.world_name='sandbox'
    
  def gt_callback(self, data):
    """Callback function that keeps track of positions for logging"""
    if not self.ready: return
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
    if not self.ready or self.finished: return []
    try:
      # Convert your ROS Image message to OpenCV2
      # changed to normal RGB order as i ll use matplotlib and PIL instead of opencv
      im = bridge.imgmsg_to_cv2(msg, 'rgb8') 
    except CvBridgeError as e:
      print(e)
    else:
      size = [self.model.input_size[2],self.model.input_size[3],self.model.input_size[1]] if FLAGS.data_format=='NCHW' else self.model.input_size[1:]
      im = sm.imresize(im,tuple(size),'nearest')
      return im

  def process_depth(self, msg):
    """ Convert depth serial data to opencv image of correct size"""
    if not self.ready or self.finished: return [] 
    try:
      # Convert your ROS Image message to OpenCV2
      im = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')#gets float of 32FC1 depth image
    except CvBridgeError as e:
      print(e)
    else:
      im = im[::8,::8]
      shp=im.shape
      # assume that when value is not a number it is due to a too large distance (set to 5m)
      # values can be nan for when they are closer than 0.5m but than the evaluate node should
      # kill the run anyway.
      im=np.asarray([ e*1.0 if not np.isnan(e) else 5 for e in im.flatten()]).reshape(shp) # clipping nans: dur: 0.010
      # print 'min: ',np.amin(im),' and max: ',np.amax(im)
      # Resize image
      if FLAGS.auxiliary_depth or FLAGS.rl:
        size = self.model.depth_input_size #(55,74)
        im=sm.imresize(im,size,'nearest') # dur: 0.002
        # cv2.imshow('depth', im)
      im = im *1/255.*5. # dur: 0.00004
      return im
    
  def image_callback(self, msg):
    """ Process serial image data with process_rgb and concatenate frames if necessary"""
    rec=time.time()
    # print 'time: {0}, len im: {1}, len ctr: {2}, act: received image.'.format(rec, len(self.time_im_received),len(self.time_ctr_send))
    if self.ready and not self.finished: self.time_im_received.append(rec)

    im = self.process_rgb(msg)
    if len(im)!=0: 
      if FLAGS.n_fc: # when features are concatenated, multiple images should be kept.
        self.nfc_images.append(im)
        if len(self.nfc_images) < FLAGS.n_frames: return
        else:
          # concatenate last n-frames
          im = np.concatenate(np.asarray(self.nfc_images[-FLAGS.n_frames:]),axis=2)
          self.nfc_images = self.nfc_images[-FLAGS.n_frames+1:] # concatenate last n-1-frames
      self.process_input(im)
    
  def depth_callback(self, msg):
    im = self.process_depth(msg)
    if len(im)!=0 and FLAGS.auxiliary_depth:
        self.target_depth = im #(64,) 
    
  def process_input(self, im):
    """Process the inputs: images, targets, auxiliary tasks
      Predict control based on the inputs.
      Plot auxiliary predictions.
      Fill replay buffer.
    """
    aux_depth=[] # variable to keep predicted depth 
    trgt = -100.
    if FLAGS.data_format == 'NCHW': 
      inpt=np.swapaxes(np.swapaxes(im,1,2),0,1) 
    else :
      inpt=im
    if FLAGS.evaluate: ### EVALUATE
      trgt=np.array([[self.target_control[5]]]) if len(self.target_control) != 0 else []
      trgt_depth = np.array([copy.deepcopy(self.target_depth)]) if len(self.target_depth) !=0 and FLAGS.auxiliary_depth else []
      control, losses, aux_results = self.model.forward([inpt], auxdepth=FLAGS.show_depth,targets=trgt, depth_targets=trgt_depth)
      for k in ['c', 't', 'd']: 
        if k in losses.keys(): 
          try:
            self.accumlosses[k] += losses[k]
          except KeyError:
            self.accumlosses[k] = losses[k]
      if FLAGS.show_depth and FLAGS.auxiliary_depth and len(aux_results)>0: aux_depth = aux_results['d']
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
      if FLAGS.auxiliary_depth:
        if not check_field(self.target_depth): 
          return
        else: 
          trgt_depth = copy.deepcopy(self.target_depth)
      control, losses, aux_results = self.model.forward([inpt], auxdepth=FLAGS.show_depth)
      if FLAGS.show_depth and FLAGS.auxiliary_depth: aux_depth = aux_results['d']
    
    ### SEND CONTROL
    if trgt != -100 and not FLAGS.evaluate: # policy mixing with FLAGS.alpha
      action = trgt if np.random.binomial(1, FLAGS.alpha**(self.runs['train']+1)) else control[0,0]
    else:
      action = control[0,0]
    msg = Twist()
    if FLAGS.type_of_noise == 'ou':
      noise = self.exploration_noise.noise()
      msg.linear.x = FLAGS.speed 
      msg.linear.y = (not FLAGS.evaluate)*noise[1]*FLAGS.sigma_y
      msg.linear.z = (not FLAGS.evaluate)*noise[2]*FLAGS.sigma_z
      msg.angular.z = max(-1,min(1,action+(not FLAGS.evaluate)*FLAGS.sigma_yaw*noise[3]))
    elif FLAGS.type_of_noise == 'uni':
      msg.linear.x = FLAGS.speed
      # msg.linear.x = FLAGS.speed + (not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_x, FLAGS.sigma_x)
      msg.linear.y = (not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_y, FLAGS.sigma_y)
      msg.linear.z = (not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_z, FLAGS.sigma_z)
      msg.angular.z = max(-1,min(1,action+(not FLAGS.evaluate)*np.random.uniform(-FLAGS.sigma_yaw, FLAGS.sigma_yaw)))
    else:
      raise IOError( 'Type of noise is unknown: {}'.format(FLAGS.type_of_noise))
    self.action_pub.publish(msg)
    
    rec=time.time()
    self.time_ctr_send.append(rec)
    # time_ind = len(self.time_ctr_send)
    delay=self.time_ctr_send[-1]-self.time_im_received[-1]
    # delay=self.time_ctr_send[time_ind-1]-self.time_im_received[time_ind-1]
    self.time_delay.append(delay)  
    # print 'time: {0}, len im: {1}, len ctr: {2}, act: control, delay: {3}'.format(rec, len(self.time_im_received),len(self.time_ctr_send),delay)
    
    if FLAGS.show_depth and len(aux_depth) != 0 and not self.finished:
      aux_depth = aux_depth.flatten()
      self.depth_pub.publish(aux_depth)
      aux_depth = []
      
    # ADD EXPERIENCE REPLAY
    if not FLAGS.evaluate and trgt != -100:
      aux_info = {}
      if FLAGS.auxiliary_depth: aux_info['target_depth']=trgt_depth
      self.replay_buffer.add(im,[trgt],aux_info=aux_info)

  def supervised_callback(self, data):
    """Get target control from the /supervised_vel node"""
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
      print('neural control deactivated.')
      self.ready=False
      self.finished=True
      # Train model from experience replay:
      # Train the model with batchnormalization out of the image callback loop
      depth_predictions = []
      losses_train = {}
      if self.replay_buffer.size()>FLAGS.batch_size and not FLAGS.evaluate:
        for b in range(min(int(self.replay_buffer.size()/FLAGS.batch_size), 10)):
          inputs, targets, aux_info = self.replay_buffer.sample_batch(FLAGS.batch_size)
          if FLAGS.data_format=="NCHW": inputs = np.swapaxes(np.swapaxes(inputs,2,3),1,2) #make data NCHW instead of NHWC
          if b==0:
            if FLAGS.plot_depth and FLAGS.auxiliary_depth:
              depth_predictions = tools.plot_depth(inputs, aux_info['target_depth'].reshape(-1,55,74))
          depth_targets=[]
          if FLAGS.auxiliary_depth: 
            depth_targets=aux_info['target_depth'].reshape(-1,55,74)
          losses = self.model.backward(inputs,targets[:].reshape(-1,1),depth_targets)
          for k in losses.keys(): 
            try:
              losses_train[k].append(losses[k])
            except:
              losses_train[k]=[losses[k]]
      k='train' if not FLAGS.evaluate else 'test'
      self.average_distances[k]= self.average_distances[k]-self.average_distances[k]/(self.runs[k]+1)
      self.average_distances[k] = self.average_distances[k]+self.current_distance/(self.runs[k]+1)
      self.runs[k]+=1
      sumvar={}
      result_string='{0}: run {1}'.format(time.strftime('%H:%M'),self.runs[k])
      vals={'current':self.current_distance, 'furthest':self.furthest_point}
      for d in ['current', 'furthest']:
        name='Distance_{0}_{1}'.format(d,'train' if not FLAGS.evaluate else 'test')
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
      if FLAGS.plot_depth and FLAGS.auxiliary_depth:
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
      if self.runs['train']%20==0 and not FLAGS.evaluate:
        # Save a checkpoint every 20 runs.
        self.model.save(self.logfolder)
      self.time_im_received=[]
      self.time_ctr_send=[]
      self.time_delay=[]
    
      

