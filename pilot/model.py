
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import models.coll_q_net as coll_q_net
import models.depth_q_net as depth_q_net

from tensorflow.contrib.slim import model_analyzer as ma
from tensorflow.python.ops import variables as tf_variables

import numpy as np

import matplotlib.pyplot as plt

"""
Build basic NN model
"""
class Model(object):
 
  def __init__(self, FLAGS, session, prefix='model', device='/gpu:0'):
    '''initialize model
    '''
    self.sess = session
    self.action_dim = FLAGS.action_dim
    self.output_size = FLAGS.output_size
    self.prefix = prefix
    self.device = device
    
    self.FLAGS=FLAGS

    self.lr = self.FLAGS.learning_rate
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    
    #define the input size of the network input
    if self.FLAGS.network == 'depth_q_net' or self.FLAGS.network == 'coll_q_net':
      # Use NCHW instead of NHWC data input because this is faster on GPU.    
      self.input_size = [None, depth_q_net.depth_q_net.default_image_size[self.FLAGS.depth_multiplier], 
        depth_q_net.depth_q_net.default_image_size[self.FLAGS.depth_multiplier], 3]
    else:
      raise NotImplementedError( 'Network is unknown: ', self.FLAGS.network)
    self.define_network()
        
    # Only feature extracting part is initialized from pretrained model
    if not self.FLAGS.continue_training:
      # make sure you exclude the prediction layers of the model
      list_to_exclude = ["global_step", "DpthQnet/q_depth", "CollQnet/q_coll"]
      variables_to_restore = slim.get_variables_to_restore(exclude=list_to_exclude)
      # adjust list in order to map mobilenet_025 weights correctly
      # Map DepthQnet/Conv_var_name/... [without :0] to MobilenetV1/Conv_var_name as saved in checkpoint
      variables_to_restore={'MobilenetV1/'+v.name[9:-2]:v for v in variables_to_restore}
    else: #If continue training
      variables_to_restore = slim.get_variables_to_restore()
    
    # get latest folder out of training directory if there is no checkpoint file
    if self.FLAGS.checkpoint_path[0]!='/':
      self.FLAGS.checkpoint_path = self.FLAGS.summary_dir+self.FLAGS.checkpoint_path
    if not os.path.isfile(self.FLAGS.checkpoint_path+'/checkpoint'):
      self.FLAGS.checkpoint_path = self.FLAGS.checkpoint_path+'/'+[mpath for mpath in sorted(os.listdir(self.FLAGS.checkpoint_path)) if os.path.isdir(self.FLAGS.checkpoint_path+'/'+mpath) and not mpath[-3:]=='val' and os.path.isfile(self.FLAGS.checkpoint_path+'/'+mpath+'/checkpoint')][-1]
    

    if not self.FLAGS.scratch: 
      print('checkpoint: {}'.format(self.FLAGS.checkpoint_path))
      init_assign_op, init_feed_dict = slim.assign_from_checkpoint(tf.train.latest_checkpoint(self.FLAGS.checkpoint_path), variables_to_restore)
    
    # create saver for checkpoints
    self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
    
    # Add the loss function to the graph.
    self.define_loss()

    # Define the training op based on the total loss
    self.define_train()
    
    # Define summaries
    self.build_summaries()
    
    init_all=tf_variables.global_variables_initializer()
    self.sess.run([init_all])
    if not self.FLAGS.scratch:
      self.sess.run([init_assign_op], init_feed_dict)
      print('Successfully loaded model from:{}'.format(self.FLAGS.checkpoint_path))
    else:
      print('Training model from scratch so no initialization.')
  
  def define_network(self):
    '''build the network and set the tensors
    '''
    with tf.device(self.device):
      self.inputs = tf.placeholder(tf.float32, shape = self.input_size)
      args_for_scope={'weight_decay': self.FLAGS.weight_decay,
      'stddev':self.FLAGS.init_scale,
      'initializer':self.FLAGS.initializer,
      'random_seed':self.FLAGS.random_seed}
      if self.FLAGS.network=='coll_q_net':
        self.actions = tf.placeholder(tf.float32, shape = [None, self.action_dim])
        args_for_model={'inputs':self.inputs,
                        'actions':self.actions,
                        'depth_multiplier':self.FLAGS.depth_multiplier,
                        'dropout_keep_prob':self.FLAGS.dropout_keep_prob,
                        'fc2_nodes':self.FLAGS.fc2_nodes,
                        'predict_action':self.FLAGS.predict_action,
                        'upscale_action':self.FLAGS.upscale_action,
                        'add_inverted_action':self.FLAGS.add_inverted_action} 
        with slim.arg_scope(coll_q_net.coll_q_net_arg_scope(is_training=True,**args_for_scope)):
          self.predictions_train, self.endpoints = coll_q_net.coll_q_net(is_training=True,**args_for_model)
        with slim.arg_scope(coll_q_net.coll_q_net_arg_scope(is_training=False, **args_for_scope)):
          self.predictions_eval, self.endpoints_eval = coll_q_net.coll_q_net(is_training=False, reuse = True,**args_for_model)
      elif self.FLAGS.network=='depth_q_net':
        self.actions = tf.placeholder(tf.float32, shape = [None, self.action_dim])
        args_for_model={'inputs':self.inputs,
                        'actions':self.actions,
                        'depth_multiplier':self.FLAGS.depth_multiplier,
                        'dropout_keep_prob':self.FLAGS.dropout_keep_prob,
                        'output_size':self.output_size,
                        'predict_action':self.FLAGS.predict_action,
                        'upscale_action':self.FLAGS.upscale_action,
                        'add_inverted_action':self.FLAGS.add_inverted_action} 
        with slim.arg_scope(depth_q_net.depth_q_net_arg_scope(is_training=True,**args_for_scope)):
          self.predictions_train, self.endpoints = depth_q_net.depth_q_net(is_training=True,**args_for_model)
        with slim.arg_scope(depth_q_net.depth_q_net_arg_scope(is_training=False, **args_for_scope)):
          self.predictions_eval, self.endpoints_eval = depth_q_net.depth_q_net(is_training=False, reuse = True,**args_for_model)
      else:
        raise NameError( '[model] Network is unknown: ', self.FLAGS.network)
  
  def define_loss(self):
    '''tensor for calculating the loss
    '''
    with tf.device(self.device):
      self.targets = tf.placeholder(tf.float32, [None,self.output_size[0],self.output_size[1]] if self.FLAGS.network=='depth_q_net' else [None, 1])
      if self.FLAGS.network=='depth_q_net':
        weights=tf.multiply(tf.cast(tf.greater(self.targets,self.FLAGS.min_depth), tf.float32),tf.cast(tf.less(self.targets,self.FLAGS.max_depth), tf.float32))
        self.weights=-1*tf.nn.pool(tf.expand_dims(-1*weights,3), [2,2], "MAX",padding="SAME")
        if self.FLAGS.loss == 'huber':
          self.loss = tf.reduce_mean(tf.losses.huber_loss(self.targets, self.predictions_train, weights=self.weights[:,:,:,0],reduction=tf.losses.Reduction.NONE,loss_collection=''),axis=[1,2])
        elif self.FLAGS.loss == 'absolute':
          self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.targets, self.predictions_train, weights=self.weights[:,:,:,0],reduction=tf.losses.Reduction.NONE,loss_collection=''),axis=[1,2])
        else: 
          self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.predictions_train, self.targets, weights=self.weights[:,:,:,0],reduction=tf.losses.Reduction.NONE,loss_collection=''),axis=[1,2])
      else:
        if self.FLAGS.loss == 'ce':
          # cross entropy loss:
          # self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.targets, logits=self.predictions_train,  dim=-1)
          # avoid log(0) with tensorflow max (!)
          self.loss = -tf.multiply(self.targets, tf.log(tf.clip_by_value(self.predictions_train,0.001,1)))-tf.multiply((1-self.targets),tf.log(tf.clip_by_value(1-self.predictions_train,0.001,1)))
          # self.loss = -tf.multiply(self.targets, tf.log(self.predictions_train))+tf.multiply((1-self.targets),tf.log(1-self.predictions_train))
        else:
          self.loss = tf.losses.mean_squared_error(self.predictions_train, self.targets, weights= 1.,reduction=tf.losses.Reduction.NONE,loss_collection='')
      self.max_loss = tf.placeholder(tf.float32)
      self.loss = tf.clip_by_value(self.loss, tf.constant(0.,dtype=tf.float32), self.max_loss)
      # self.loss = tf.clip_by_value(self.loss, tf.constant(0.,dtype=tf.float32), tf.constant(self.FLAGS.max_loss,dtype=tf.float32))
      tf.losses.add_loss(tf.reduce_mean(self.loss))
      
      # Add action prediction loss
      if self.FLAGS.predict_action:
        self.action_loss = tf.losses.mean_squared_error(self.endpoints['fc_a'], self.actions)
        tf.losses.add_loss(tf.reduce_mean(self.action_loss))
      self.total_loss = tf.losses.get_total_loss()
      # self.total_loss = self.loss
      
  def define_train(self):
    '''applying gradients to the weights from normal loss function
    '''
    with tf.device(self.device):
      # Specify the optimizer and create the train op:
      optimizer_list={'adam':tf.train.AdamOptimizer(learning_rate=self.lr),
        'adadelta':tf.train.AdadeltaOptimizer(learning_rate=self.lr),
        'gradientdescent':tf.train.GradientDescentOptimizer(learning_rate=self.lr),
        'rmsprop':tf.train.RMSPropOptimizer(learning_rate=self.lr)}
      self.optimizer=optimizer_list[self.FLAGS.optimizer]
      # Create the train_op and scale the gradients by providing a map from variable
      # name (or variable) to a scaling coefficient:
      # Take possible a smaller step (gradient multiplier) for the feature extracting part
      mobile_variables = [v for v in tf.global_variables() if (v.name.find('Adadelta')==-1 and v.name.find('BatchNorm')==-1 and v.name.find('Adam')==-1  and v.name.find('q_depth')==-1 and v.name.find('q_coll')==-1)]
      # self.train_op = slim.learning.create_train_op(self.total_loss, 
      self.train_op = slim.learning.create_train_op(self.total_loss, 
        self.optimizer, 
        global_step=self.global_step, 
        gradient_multipliers={v.name: self.FLAGS.grad_mul_weight for v in mobile_variables}, 
        clip_gradient_norm=self.FLAGS.clip_grad,
        summarize_gradients=True)

  def forward(self, inputs, actions=[], targets=[]):
    '''run forward pass and return action prediction
    inputs=batch of RGB images
    actions=applied action in the batch for each image
    targets=supervised target corresponding to the next depth frame or a tag defining whether there was a collision.
    '''
    feed_dict={self.inputs: inputs}  
    feed_dict[self.actions]=actions
    tensors = [self.predictions_eval]

    if len(targets) != 0: # if target control is available, calculate loss
      tensors.append(self.loss)
      feed_dict[self.targets]=targets
      feed_dict[self.max_loss]=self.FLAGS.max_loss
    
    if self.FLAGS.predict_action: 
      tensors.append(self.action_loss)

    results = self.sess.run(tensors, feed_dict=feed_dict)

    output=results.pop(0)
    losses = {}
    
    if len(targets) != 0:
      losses['o']=results.pop(0) # output loss
    
    if self.FLAGS.predict_action: 
      losses['a'] = results.pop(0)

    return output, losses

  def backward(self, inputs, actions=[], targets=[]):
    '''run backward pass and return losses
    '''
    tensors = [self.train_op]
    feed_dict = {self.inputs: inputs}
    feed_dict[self.actions]=actions
    tensors.append(self.loss)
    feed_dict[self.targets]=targets
    feed_dict[self.max_loss]=self.FLAGS.max_loss
    tensors.append(self.total_loss)
    
    if self.FLAGS.predict_action: 
      tensors.append(self.action_loss)
    #DEBUG
    # tensors.append(self.weights)

    results = self.sess.run(tensors, feed_dict=feed_dict)
    losses={}
    _ = results.pop(0) # train_op
    losses['o']=results.pop(0) # control loss or Q-loss 
    losses['t'] = results.pop(0) # total loss
    if self.FLAGS.predict_action: losses['a'] = results.pop(0)
    return losses

  def save(self, logfolder):
    '''save a checkpoint'''
    self.saver.save(self.sess, logfolder+'/my-model', global_step=tf.train.global_step(self.sess, self.global_step))
    
  def add_summary_var(self, name):
    var_name = tf.Variable(0., name=name)
    self.summary_vars[name]=var_name
    self.summary_ops[name] = tf.summary.scalar(name, var_name)
    
  def build_summaries(self): 
    self.summary_vars = {}
    self.summary_ops = {}
    for t in ['train', 'test', 'val']:
      for l in ['total', 'output','action']:
        name='Loss_{0}_{1}'.format(t,l)
        self.add_summary_var(name)
        if l == 'output':
          for n in ['min','max','var']:
            name='Loss_{0}_{1}_{2}'.format(t,l,n)
            self.add_summary_var(name)
    for d in ['current','furthest']:
      for t in ['train', 'test']:
        for w in ['','sandbox','forest','canyon','esat_corridor_v1', 'esat_corridor_v2','maze','real_maze']:
          name = 'Distance_{0}_{1}'.format(d,t)
          if len(w)!=0: name='{0}_{1}'.format(name,w)
          self.add_summary_var(name)
    for i in ['state','action','trgt']:
      name = i+'_variance'
      self.add_summary_var(name)
    self.add_summary_var('driving_time')      
    self.add_summary_var('imitation_loss')      
    self.add_summary_var('depth_loss')      
    if self.FLAGS.plot_depth:
      name="depth_predictions"
      dep_images = tf.placeholder(tf.uint8, [1, 400, 400, 3])
      # dep_images = tf.placeholder(tf.float32, [1, 400, 400, 3])
      self.summary_vars[name]=dep_images
      self.summary_ops[name]=tf.summary.image(name, dep_images)
    
  def summarize(self, sumvars):
    '''write summary vars with ops'''
    if self.writer:
      feed_dict={self.summary_vars[key]:sumvars[key] for key in sumvars.keys()}
      sum_op = tf.summary.merge([self.summary_ops[key] for key in sumvars.keys()])
      summary_str = self.sess.run(sum_op, feed_dict=feed_dict)
      self.writer.add_summary(summary_str,  tf.train.global_step(self.sess, self.global_step))
      self.writer.flush()
