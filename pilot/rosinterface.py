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
    
    self.last_pose=[] # previous pose, used for accumulative distance
    self.depth = [] # field to keep the latest supervised depth
    self.prev_im=[] # in case of depth_q_net experience = (I_(t-1), a_(t-1), d_t)
    self.prev_action=-100 # so keep action and image during 1 step and save it in the next step
    self.prev_prediction=[]
    self.prev_weight=[]
    self.prev_random=False

    self.start_time=0
    self.world_name = ''
    self.runs={'train':0, 'test':0} # number of online training run (used for averaging)
    # self.accumlosses = {} # gather losses and info over the run in a dictionary
    self.current_distance=0 # accumulative distance travelled from beginning of run used at evaluation
    self.furthest_point=0 # furthest point reached from spawning point at the beginning of run
    self.average_distances={'train':0, 'test':0} # running average over different runs
    # self.nfc_images =[] #used by n_fc networks for building up concatenated frames
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
      self.validation_buffer = ReplayBuffer(self.FLAGS, self.FLAGS.random_seed) if self.FLAGS.validate_online else None
      self.accumloss = 0
      if rospy.has_param('gt_info'):
        rospy.Subscriber(rospy.get_param('gt_info'), Odometry, self.gt_callback)

    # Add some lines to debug delays:
    self.time_im_received=[]
    self.time_ctr_send=[]
    self.time_delay=[]


    # # create animation to display outputs:
    # fig=plt.figure()
    # self.outputs=np.asarray([0,0,0]).reshape((-1,1))
    # output_plot=plt.plot(self.outputs)
    # def update(frame_number):
    #   output_plot
    # animation.FuncAnimation(fig, update)
    # plt.show()

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
      img =bridge.imgmsg_to_cv2(msg) 
      # img =bridge.imgmsg_to_cv2(msg, 'rgb8') 
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
      img = sm.resize(img,size,mode='constant').astype(float) #.astype(np.float32)
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
      # de=de*1/5.*255
      # print 'DEPTH: min: ',np.amin(de),' and max: ',np.amax(de)

      de = de[::6,::8]
      # im = im[::8,::8]
      shp=de.shape
      # # assume that when value is not a number it is due to a too large distance (set to 5m)
      # # values can be nan for when they are closer than 0.5m but than the evaluate node should
      # # kill the run anyway.
      de=np.asarray([ e*1.0 if not np.isnan(e) else 5 for e in de.flatten()]).reshape(shp) # clipping nans: dur: 0.010
      size = self.model.output_size #(55,74)
      de = sm.resize(de,size,order=1,mode='constant', preserve_range=True)

      # de[de<0.001]=0      
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
    im = self.process_rgb(msg)
    if len(im)!=0: 
      # if self.FLAGS.n_fc: # when features are concatenated, multiple images should be kept.
      #   self.nfc_images.append(im)
      #   if len(self.nfc_images) < self.FLAGS.n_frames: return
      #   else:
      #     # concatenate last n-frames
      #     im = np.concatenate(np.asarray(self.nfc_images[-self.FLAGS.n_frames:]),axis=2)
      #     self.nfc_images = self.nfc_images[-self.FLAGS.n_frames+1:] # concatenate last n-1-frames
      self.process_input(im)
    
  def depth_callback(self, msg):
    im = self.process_depth(msg)
    if len(im)!=0:
      self.depth = im
  
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
    # btime=time.time()
    
    # save depth to keep images close.
    depth = copy.deepcopy(self.depth)

    ### FORWARD 
    # feed in 3 actions corresponding to right, straight and left.
    actions=np.arange(-self.FLAGS.action_bound, self.FLAGS.action_bound+2./self.FLAGS.action_quantity, 2./(self.FLAGS.action_quantity-1)).reshape((-1,1))
    if self.FLAGS.action_smoothing: actions=np.array([a+np.random.uniform(low=-1./(self.FLAGS.action_quantity-1),high=1./(self.FLAGS.action_quantity-1)) for a in actions]).reshape((-1,1))
  
    # TOBE REMOVED:
    if rospy.has_param('action_amplitude'):
      if self.FLAGS.action_amplitude != rospy.get_param('action_amplitude'):
        self.FLAGS.action_amplitude = rospy.get_param('action_amplitude')
        print('changed action amplitude to: {}'.format(self.FLAGS.action_amplitude))
       
    output, _ = self.model.forward(np.asarray([im]*len(actions)), self.FLAGS.action_amplitude*actions)
    # output=np.asarray([1,0,1]).reshape((-1,1))
    if not self.ready or self.finished: return

    ### EXTRACT CONTROL
    if self.FLAGS.network == 'depth_q_net':
      # take action corresponding to the maximum minimum depth:
      
      # best_output=np.argmax([np.mean(o[o!=0]) for o in output]) # TOBE REMOVED!!

      best_output=np.argmax([np.amin(o[o!=0]) for o in output])
      # keep only depth prediction of action going to be performed
      self.depth_prediction= output[best_output]
      action = float(actions[best_output])
      # print 'action: ',action
    else:
      # take action giving the lowest collision probability
      # if all actions are equally likeli to end with a bump, make straight the default:
      outputs_compared=[output[i]==output[i+1] for i in range(len(output)-1)]
      if sum(outputs_compared) == len(outputs_compared): action = 0
      else: action = float(actions[np.argmin(output)])

    noise_sample = self.exploration_noise.noise()

    random=False
    if self.FLAGS.prefill and self.replay_buffer.size() < self.FLAGS.buffer_size and not self.FLAGS.evaluate:
      # print 'sample random to fill buffer!'
      # action=0
      action=2*np.random.random_sample()-1 if self.FLAGS.noise=='uni' else 0.3*noise_sample[0]
      random=True #needed for random_action replay priority
    # elif not self.FLAGS.evaluate and self.FLAGS.epsilon != 0: #TODO: change back to evaluation dependency when training online with epsilon!!!
    elif self.FLAGS.epsilon != 0: 
        # calculate decaying epsilon
        random_action=2*np.random.random_sample()-1 if self.FLAGS.noise=='uni' else self.FLAGS.action_bound*0.3*noise_sample[0] #make turtlebot less reactive 
        # random_action=2*np.random.random_sample()-1 if self.FLAGS.noise=='uni' else 0.3*noise_sample[0]
        epsilon=min([1, self.FLAGS.epsilon*np.exp(-self.FLAGS.epsilon_decay*(self.runs['train']+1))])
        action = random_action if np.random.binomial(1,epsilon) else action
        random= action==random_action #needed for random_action replay priority
        # print("random: {0}, epsilon: {1}, action:{2}".format(random_action,epsilon,action))
        if epsilon < 0.0000001: epsilon = 0 #avoid taking binomial of too small epsilon.
        # clip action to -1, 0 or 1.
        # action = 0 if np.abs(action) > 0.3 else np.sign(action) #added on 28/06/18
        
    # if len(self.supervised_vel) != 0: # TOBE REMOVED !!!
    #   action = self.supervised_vel[5]

    
    # Adjust speed in real_maze
    speed = self.FLAGS.speed
    if self.world_name ==  'real_maze':
      speed_dict={-1:0.6*self.FLAGS.speed,
                  0:self.FLAGS.speed,
                  1:0.6*self.FLAGS.speed}
      try:
        speed = speed_dict[np.round(action)]
      except:
        pass 
    ### SEND CONTROL (with possibly some noise)

    msg = Twist()
    msg.linear.x = speed
    msg.linear.y = noise_sample[1]*self.FLAGS.sigma_y
    msg.linear.z = noise_sample[2]*self.FLAGS.sigma_z
    msg.angular.z = action
    self.action_pub.publish(msg)
  

    ### keep track of imitation loss on the fly
    if len(self.supervised_vel) != 0:
      self.imitation_loss.append((self.supervised_vel[5]-action)**2)


    rec=time.time()
    
    if self.FLAGS.network == 'depth_q_net' and not self.FLAGS.dont_show_depth and not self.finished:
      self.depth_pub.publish(np.concatenate([np.array([action]),output.flatten()],axis=0))
      # self.depth_pub.publish(output.flatten())
      
    # ADD EXPERIENCE REPLAY
    if ( not self.FLAGS.evaluate or self.FLAGS.validate_online) and not self.finished:
      if self.FLAGS.network=='depth_q_net':
        closest_action_index = np.argmin([np.abs(a-action) for a in actions])
        if len(self.prev_im)!= 0 and self.prev_action!=-100 and len(self.prev_prediction) != 0 and self.prev_weight != -1:
          # weight error according to how close input-action on which depth is predicted and actual applied action
          experience={'state':self.prev_im,
            'action':self.prev_action,
            'trgt':depth}
          if self.FLAGS.replay_priority == 'td_error': experience['error']=self.prev_weight*np.mean((self.prev_prediction-depth)**2)
          elif self.FLAGS.replay_priority == 'random_action': experience['rnd']=self.prev_random
          if self.FLAGS.evaluate: 
            self.validation_buffer.add(experience)
          else:
            self.replay_buffer.add(experience)
        self.prev_im=copy.deepcopy(im)
        self.prev_action=action
        self.prev_prediction=output[closest_action_index]
        self.prev_weight=1-(self.FLAGS.action_quantity-1)*(np.abs(actions[closest_action_index]-action)/2)
        self.prev_random=random
      elif self.FLAGS.network=='coll_q_net':
        experience={'state':im,
          'action':action,
          'trgt':0}
        if self.FLAGS.replay_priority == 'td_error': 
          experience['error']=output[best_output]
        elif self.FLAGS.replay_priority == 'random_action': 
          experience['rnd']=random
        if self.FLAGS.evaluate:
          self.validation_buffer.add(experience)
        else:
          self.replay_buffer.add(experience)

    delay=time.time()-rec
    self.time_delay.append(delay)  

      
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
        driving_duration = rospy.get_time() - self.start_time
      # Train model from experience replay:
      losses_train = {}
      if self.replay_buffer.size()>self.FLAGS.batch_size and not self.FLAGS.evaluate and ((not self.FLAGS.prefill) or (self.FLAGS.prefill and self.replay_buffer.size() == self.FLAGS.buffer_size)):
        # add code for cleaning up buffer: adding target 1 for last frames before collision
        self.replay_buffer.label_collision(self.logfolder)
        self.replay_buffer.preprocess()
        # for b in range(min(int(self.replay_buffer.size()/self.FLAGS.batch_size), 100)): # sample max 10 batches from all experiences gathered.
        for b in range(min(int(self.replay_buffer.size()/self.FLAGS.batch_size), self.FLAGS.grad_steps)): # sample max 10 batches from all experiences gathered.
          states, actions, targets = self.replay_buffer.sample_batch()
          losses = self.model.backward(states,
                                      actions.reshape(-1,1),
                                      targets.reshape(-1,1) if self.FLAGS.network == 'coll_q_net' else targets.reshape(-1,self.model.output_size[0],self.model.output_size[1]))
          for k in losses.keys():
            try:
              losses_train[k].extend(np.asarray([losses[k]]).flatten()) #in order to cope both with integers and lists
            except Exception : # first element of training
              losses_train[k]=list(np.asarray([losses[k]]).flatten())
          if self.FLAGS.replay_priority == 'td_error':
            self.replay_buffer.update_probabilities(states,actions,targets,np.asarray(losses['o']).flatten())
          if self.FLAGS.clip_loss_to_max:
            self.FLAGS.max_loss = np.amax(np.asarray(losses_train['o']).flatten())
      # validate on validation buffer
      losses_test = {}
      if self.FLAGS.validate_online and self.validation_buffer.size()>self.FLAGS.batch_size:
        # add code for cleaning up buffer: adding target 1 for last frames before collision
        self.validation_buffer.label_collision(self.logfolder)
        for b in range(min(int(self.validation_buffer.size()/self.FLAGS.batch_size), self.FLAGS.grad_steps)): # sample max 10 batches from all experiences gathered.
          states, actions, targets = self.validation_buffer.sample_batch()
          _, losses = self.model.forward(states,
                                      actions.reshape(-1,1),
                                      targets.reshape(-1,1) if self.FLAGS.network == 'coll_q_net' else targets.reshape(-1,self.model.output_size[0],self.model.output_size[1]))
          for k in losses.keys():
            try:
              losses_test[k].extend(np.asarray([losses[k]]).flatten()) #in order to cope both with integers and lists
            except Exception : # first batch
              losses_test[k]=list(np.asarray([losses[k]]).flatten())
      
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
      if len(losses_train) != 0:
        for k in losses_train.keys():
          name={'t':'Loss_train_total','o':'Loss_train_output'}
          sumvar[name[k]]=np.mean(losses_train[k])   
          result_string='{0}, {1}:{2}'.format(result_string, name[k], np.mean(losses_train[k]))
          if k=='o':
            sumvar[name[k]+'_min']=np.amin(np.asarray(losses_train[k]).flatten())
            result_string='{0}, {1}:{2}'.format(result_string, name[k]+'_min', np.amin(np.asarray(losses_train[k]).flatten()))
            sumvar[name[k]+'_max']=np.amax(np.asarray(losses_train[k]).flatten())
            result_string='{0}, {1}:{2}'.format(result_string, name[k]+'_max', np.amax(np.asarray(losses_train[k]).flatten()))
            sumvar[name[k]+'_var']=np.var(np.asarray(losses_train[k]).flatten())
            result_string='{0}, {1}:{2}'.format(result_string, name[k]+'_var', np.var(np.asarray(losses_train[k]).flatten()))
      if len(losses_test) != 0:
        for k in losses_test.keys():
          name={'t':'Loss_test_total','o':'Loss_test_output'}
          sumvar[name[k]]=np.mean(losses_test[k])   
          result_string='{0}, {1}:{2}'.format(result_string, name[k], np.mean(losses_test[k]))

      # for k in self.accumlosses.keys():
      # name={'t':'Loss_test_total','o':'Loss_test_output'}
      # sumvar[name[k]]=self.accumlosses[k]
      # result_string='{0}, {1}:{2}'.format(result_string, name[k], self.accumlosses[k])
      if self.replay_buffer.size > 10: 
        buffer_variances=self.replay_buffer.get_variance()
        for i in ['state','action','trgt']:
          sumvar[i+'_variance']=buffer_variances[i]
          result_string='{0}, {1}:{2:0.5e}'.format(result_string, i+'_variance',buffer_variances[i]) 
      if len(self.time_delay) > 2: 
        result_string='{0}, min_delay: {1}, avg_delay: {2}, max_delay: {3}'.format(result_string, np.min(self.time_delay[1:]), np.mean(self.time_delay[1:]), np.max(self.time_delay))
      # add driving duration (collision free)
      if self.start_time!=0: 
        result_string='{0}, driving_duration: {1:0.3f}'.format(result_string, driving_duration)
        sumvar['driving_time']=driving_duration
      # add imitation loss
      if len(self.imitation_loss)!=0:
        result_string='{0}, imitation_loss: {1:0.3}'.format(result_string, np.mean(self.imitation_loss))
        sumvar['imitation_loss']=np.mean(self.imitation_loss)
      # add depth loss
      if len(self.depth_loss)!=0:
        result_string='{0}, depth_loss: {1:0.3f}, depth_loss_var: {2:0.3f}'.format(result_string, np.mean(self.depth_loss), np.var(self.depth_loss))
        sumvar['depth_loss']=np.mean(self.depth_loss)
      try:
        self.model.summarize(sumvar)
      except Exception as e:
        print('failed to write', e)
        pass
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
      # self.accumlosses = {}
      self.current_distance = 0
      self.last_pose = []
      self.furthest_point = 0
      self.world_name = ''
      if self.runs['train']%10==1 and not self.FLAGS.evaluate:
        # Save a checkpoint every 20 runs. (but also the first one)
        self.model.save(self.logfolder)
        print('model saved [run {0}]'.format(self.runs['train']))
      self.time_im_received=[]
      self.time_ctr_send=[]
      self.time_delay=[]
      self.prev_im=[]
      self.prev_action=-100
      self.prev_prediction=[]
      self.prev_weight=[]
      self.prev_random=False
      self.start_time=0
      self.imitation_loss=[]
      self.depth_loss=[]
