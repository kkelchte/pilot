
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import models.mobile_net as mobile_net
import models.depth_q_net as depth_q_net

from tensorflow.contrib.slim import model_analyzer as ma
from tensorflow.python.ops import variables as tf_variables

import numpy as np



FLAGS = tf.app.flags.FLAGS

# INITIALIZATION
tf.app.flags.DEFINE_string("checkpoint_path", 'mobile_net', "Specify the directory of the checkpoint of the earlier trained model.")
tf.app.flags.DEFINE_boolean("continue_training", False, "Continue training of the prediction layers. If false, initialize the prediction layers randomly. The default value should remain True.")
tf.app.flags.DEFINE_boolean("scratch", False, "Initialize full network randomly.")

# TRAINING
tf.app.flags.DEFINE_float("weight_decay", 0.00004, "Weight decay of inception network")
tf.app.flags.DEFINE_float("init_scale", 0.0005, "Std of uniform initialization")
tf.app.flags.DEFINE_float("depth_weight", 1, "Define the weight applied to the depth values in the loss relative to the control loss.")
tf.app.flags.DEFINE_float("control_weight", 1, "Define the weight applied to the control loss.")
tf.app.flags.DEFINE_float("grad_mul_weight", 1, "Specify the amount the gradients of prediction layers.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Specify the probability of dropout to keep the activation.")
tf.app.flags.DEFINE_integer("clip_grad", 0, "Specify the max gradient norm: default 0 is no clipping, recommended 4.")
tf.app.flags.DEFINE_string("optimizer", 'adadelta', "Specify optimizer, options: adam, adadelta, gradientdescent, rmsprop")
tf.app.flags.DEFINE_boolean("discrete", False, "Define the output of the network as discrete control values or continuous.")
tf.app.flags.DEFINE_integer("num_outputs", 9, "Specify the number of discrete outputs.")
tf.app.flags.DEFINE_string("initializer", 'xavier', "Define the initializer: xavier or uniform [-init_scale, init_scale]")

"""
Build basic NN model
"""
class Model(object):
 
  def __init__(self, session, action_dim, prefix='model', device='/gpu:0', bound=1):
    '''initialize model
    '''
    self.sess = session
    self.action_dim = action_dim
    self.bound=bound
    self.prefix = prefix
    self.device = device

    self.lr = FLAGS.learning_rate
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    #define the input size of the network input
    if FLAGS.network =='mobile':
      self.input_size = [None, mobile_net.mobilenet_v1.default_image_size[FLAGS.depth_multiplier], 
        mobile_net.mobilenet_v1.default_image_size[FLAGS.depth_multiplier], 3]
    else:
      raise NotImplementedError( 'Network is unknown: ', FLAGS.network)
    self.define_network()
        
    # Only feature extracting part is initialized from pretrained model
    if not FLAGS.continue_training:
      # make sure you exclude the prediction layers of the model
      list_to_exclude = ["global_step"]
      list_to_exclude.append("MobilenetV1/control")
      list_to_exclude.append("MobilenetV1/aux_depth")
      list_to_exclude.append("concatenated_feature")
      list_to_exclude.append("control")
      list_to_exclude.append("MobilenetV1/q_depth")
    else: #If continue training
      list_to_exclude = []
    variables_to_restore = slim.get_variables_to_restore(exclude=list_to_exclude)
    

    # get latest folder out of training directory if there is no checkpoint file
    if FLAGS.checkpoint_path[0]!='/':
      FLAGS.checkpoint_path = FLAGS.summary_dir+FLAGS.checkpoint_path
    if not os.path.isfile(FLAGS.checkpoint_path+'/checkpoint'):
      FLAGS.checkpoint_path = FLAGS.checkpoint_path+'/'+[mpath for mpath in sorted(os.listdir(FLAGS.checkpoint_path)) if os.path.isdir(FLAGS.checkpoint_path+'/'+mpath) and not mpath[-3:]=='val' and os.path.isfile(FLAGS.checkpoint_path+'/'+mpath+'/checkpoint')][-1]
    
    if not FLAGS.scratch: 
      print('checkpoint: {}'.format(FLAGS.checkpoint_path))
      init_assign_op, init_feed_dict = slim.assign_from_checkpoint(tf.train.latest_checkpoint(FLAGS.checkpoint_path), variables_to_restore)
    
    # Add the loss function to the graph.
    self.define_loss()

    # Define the training op based on the total loss
    self.define_train()
    
    # Define summaries
    self.build_summaries()
    
    init_all=tf_variables.global_variables_initializer()
    self.sess.run([init_all])
    if not FLAGS.scratch:
      self.sess.run([init_assign_op], init_feed_dict)
      print('Successfully loaded model from:{}'.format(FLAGS.checkpoint_path))
    else:
      print('Training model from scratch so no initialization.')
  
  def define_network(self):
    '''build the network and set the tensors
    '''
    with tf.device(self.device):
      self.inputs = tf.placeholder(tf.float32, shape = self.input_size)
      args_for_scope={'weight_decay': FLAGS.weight_decay,
      'stddev':FLAGS.init_scale}
      if FLAGS.network=='mobile':
        args_for_model={'inputs':self.inputs, 
                      'num_classes':self.action_dim,
                      'scope': self.prefix} 
        with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training=True,**args_for_scope)):
          self.outputs, self.endpoints = mobile_net.mobilenet_v1(is_training=True,**args_for_model)
        with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training=False, **args_for_scope)):
          self.controls, _ = mobile_net.mobilenet_v1(is_training=False, reuse = True,**args_for_model)
      else:
        raise NameError( '[model] Network is unknown: ', FLAGS.network)
      if self.bound!=1 and self.bound!=0:
        self.outputs = tf.multiply(self.outputs, self.bound) # Scale output to -bound to bound

  def define_loss(self):
    '''tensor for calculating the loss
    '''
    with tf.device(self.device):
      self.targets = tf.placeholder(tf.float32, [None, self.action_dim])
      self.loss = tf.losses.mean_squared_error(self.outputs, self.targets, weights=FLAGS.control_weight)
      self.total_loss = tf.losses.get_total_loss()
      
  def define_train(self):
    '''applying gradients to the weights from normal loss function
    '''
    with tf.device(self.device):
      # Specify the optimizer and create the train op:
      optimizer_list={'adam':tf.train.AdamOptimizer(learning_rate=self.lr),
        'adadelta':tf.train.AdadeltaOptimizer(learning_rate=self.lr),
        'gradientdescent':tf.train.GradientDescentOptimizer(learning_rate=self.lr),
        'rmsprop':tf.train.RMSPropOptimizer(learning_rate=self.lr)}
      self.optimizer=optimizer_list[FLAGS.optimizer]
      # Create the train_op and scale the gradients by providing a map from variable
      # name (or variable) to a scaling coefficient:
      gradient_multipliers = {}
      # Take possible a smaller step (gradient multiplier) for the feature extracting part
      mobile_variables = [v for v in tf.global_variables() if (v.name.find('Adadelta')==-1 and v.name.find('BatchNorm')==-1 and v.name.find('Adam')==-1  and v.name.find('aux_depth')==-1  and v.name.find('control')==-1)]
      for v in mobile_variables: gradient_multipliers[v.name]=FLAGS.grad_mul_weight
      self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer, global_step=self.global_step, gradient_multipliers=gradient_multipliers, clip_gradient_norm=FLAGS.clip_grad)

  def forward(self, inputs):
    '''run forward pass and return action prediction
    inputs=batch of RGB images
    targets = supervised target control
    '''
    feed_dict={self.inputs: inputs}  
    tensors = [self.controls]
    results = self.sess.run(tensors, feed_dict=feed_dict)
    output=results.pop(0)
    return output

  def backward(self, inputs, targets=[]):
    '''run backward pass and return losses
    '''
    tensors = [self.train_op]
    feed_dict = {self.inputs: inputs}
    feed_dict[self.targets]=targets
    tensors.append(self.loss)
    tensors.append(self.total_loss)
    results = self.sess.run(tensors, feed_dict=feed_dict)
    losses={}
    _ = results.pop(0) # train_op
    losses['c']=results.pop(0) # control loss or Q-loss 
    losses['t'] = results.pop(0) # total loss
    return losses

  
  
