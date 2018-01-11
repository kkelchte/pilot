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

tf.app.flags.DEFINE_boolean("recovery",False,"Recovery cameras 1 left 1 right.")
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
    rospy.Subscriber(rospy.get_param('depth_image'), Image, self.depth_callback)
    if not FLAGS.real: # initialize the replay buffer
      self.replay_buffer = ReplayBuffer(FLAGS.buffer_size, FLAGS.random_seed)
      self.accumloss = 0
      rospy.Subscriber('/ground_truth/state', Odometry, self.gt_callback)

    if FLAGS.recovery:
      rospy.Subscriber(rospy.get_param('depth_image')+'_left', Image, self.depth_left_callback)
      rospy.Subscriber(rospy.get_param('depth_image')+'_right', Image, self.depth_right_callback)
      self.depth_left=[]
      self.depth_right=[]

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

  def process_depth(self, msg):
    """ Convert depth serial data to opencv image of correct size"""
    if not self.ready or self.finished: return [] 
    try:
      # Convert your ROS Image message to OpenCV2
      im = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')#gets float of 32FC1 depth image
    except CvBridgeError as e:
      print(e)
    else:
      im = im[::20,::20]
      # im = im[::8,::8]
      shp=im.shape
      # assume that when value is not a number it is due to a too large distance (set to 5m)
      # values can be nan for when they are closer than 0.5m but than the evaluate node should
      # kill the run anyway.
      im=np.asarray([ e*1.0 if not np.isnan(e) else 5 for e in im.flatten()]).reshape(shp) # clipping nans: dur: 0.010
      # print 'min: ',np.amin(im),' and max: ',np.amax(im)
      # Resize image
      size = self.model.depth_input_size #(55,74)
      # im=sm.imresize(im,size,'nearest') # dur: 0.002
      # im = im *1/255.*5. # dur: 0.00004
      return im
    
  def depth_callback(self, msg):
      im = self.process_depth(msg)
      if len(im) != 0:
        self.process_input(im)
  
  def depth_left_callback(self, msg):
    im = self.process_depth(msg)
    if len(im) != 0:
      self.depth_left=im
  
  def depth_right_callback(self, msg):
    im = self.process_depth(msg)
    if len(im) != 0:
      self.depth_right=im

  def process_input(self, im):
    """Process the inputs: images, targets, auxiliary tasks
      Predict control based on the inputs.
      Plot auxiliary predictions.
      Fill replay buffer.
    """
    ### EXTRACT CONTROL FROM DEPTH IMAGE
    if FLAGS.recovery and len(self.depth_right)!=0 and len(self.depth_left)!=0:
      im=np.asarray(im)
      depth_right=np.asarray(self.depth_right)
      depth_left=np.asarray(self.depth_left)
      left=sum(sum(depth_left))
      right=sum(sum(depth_right))
      straight=sum(sum(im))
      action=np.argmax([right,straight,left])-1
    else:
      im=np.asarray(im)
      left=sum(sum(im[:,:im.shape[1]/2]))
      right=sum(sum(im[:,im.shape[1]/2:]))
      straight=sum(sum(im[:,im.shape[1]/4:im.shape[1]*3/4]))
      action=np.argmax([right,straight,left])-1
    
    print("{0}, left: {1}, middle: {2}, right: {3}".format(action, left, straight, right))
    ### SEND CONTROL
    msg = Twist()
    msg.linear.x = 1.8
    msg.linear.y = 0
    msg.linear.z = 0
    msg.angular.z = action

    self.action_pub.publish(msg)
    
    # write control to log
    f=open(self.logfolder+'/ctr_log','a')
    f.write("{0} {1} {2} {3} {4} {5} \n".format(msg.linear.x,msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z))
    f.close()

    # if not self.finished:
    #   rec=time.time()
    #   self.time_ctr_send.append(rec)
    #   delay=self.time_ctr_send[-1]-self.time_im_received[-1]
    #   self.time_delay.append(delay)  
          
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
      result_string="woopwoop"
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
    
      

