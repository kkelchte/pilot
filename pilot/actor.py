import tensorflow as tf
import models.mobile_net as mobile_net
from tensorflow.python.ops import variables as tf_variables
import tensorflow.contrib.slim as slim
import os, sys
import numpy as np

from tensorflow.python.tools import inspect_checkpoint as chkp

FLAGS = tf.app.flags.FLAGS
# Apply batch normalization on batch state input
tf.app.flags.DEFINE_float("actor_learning_rate", 0.5, "Start learning rate.")
tf.app.flags.DEFINE_string("actor_network", 'mobile', "Define the type of network: mobile.")
tf.app.flags.DEFINE_float("actor_tau", 0.001, "Update rate from target to real actor network")
tf.app.flags.DEFINE_float("actor_weight_decay", 0.00004, "Weight decay of inception network")

"""
Build basic actor model inherited from model.
"""
class Actor():
  """Actor Model: inherited model mapping state input to action output"""
  
  def __init__(self, 
              session, 
              output_size,
              prefix='actor', 
              device='/gpu:0'):

    self.tau = FLAGS.actor_tau
    self.sess = session
    self.action_dim = output_size
    self.prefix = prefix
    self.device = device

    self.lr = FLAGS.actor_learning_rate

    with tf.variable_scope(self.prefix):
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
    
    #define the input size of the network input
    if FLAGS.actor_network =='mobile':
      self.input_size = [None, mobile_net.mobilenet_v1.default_image_size[FLAGS.depth_multiplier], 
        mobile_net.mobilenet_v1.default_image_size[FLAGS.depth_multiplier], 3]
    else:
      raise NotImplementedError( 'Network is unknown: ', FLAGS.actor_network)
    
    # get latest folder out of training directory if there is no checkpoint file
    if FLAGS.checkpoint_path[0]!='/':
      FLAGS.checkpoint_path = FLAGS.summary_dir+FLAGS.checkpoint_path
    if not os.path.isfile(FLAGS.checkpoint_path+'/checkpoint'):
      FLAGS.checkpoint_path = FLAGS.checkpoint_path+'/'+[mpath for mpath in sorted(os.listdir(FLAGS.checkpoint_path)) if os.path.isdir(FLAGS.checkpoint_path+'/'+mpath) and not mpath[-3:]=='val' and os.path.isfile(FLAGS.checkpoint_path+'/'+mpath+'/checkpoint')][-1]
    
    ###### MAIN NETWORK

    # define actor network and save placeholders & tensors
    self.inputs, self.outputs = self.define_network()

    # Only feature extracting part is initialized from pretrained model
    if not FLAGS.continue_training:
      # make sure you exclude the prediction layers of the model
      list_to_exclude = ["actor/global_step","actor/control","actor/aux_depth"]
      variables_to_restore = slim.get_variables_to_restore(exclude=list_to_exclude)
      # Map actor/Conv_var_name/... [!! without :0 !!] to MobilenetV1/Conv_var_name as saved in checkpoint
      variables_to_restore={'MobilenetV1/'+v.name[6:-2]:v for  v in variables_to_restore}
      # for k in variables_to_restore.keys(): print("{0}: {1}".format(k, variables_to_restore[k]))
    else: #If continue training
      list_to_exclude = []
      variables_to_restore = slim.get_variables_to_restore(exclude=list_to_exclude)
      # for v in variables_to_restore: print v.name

    # chkp.print_tensors_in_checkpoint_file(tf.train.latest_checkpoint(FLAGS.checkpoint_path), tensor_name='MobilenetV1/Conv2d_1_depthwise', all_tensors=True)
    if not FLAGS.scratch: 
      print('checkpoint: {}'.format(FLAGS.checkpoint_path))
      init_assign_op, init_feed_dict = slim.assign_from_checkpoint(tf.train.latest_checkpoint(FLAGS.checkpoint_path), variables_to_restore)
    
    ###### TARGET NETWORK

    # define target networks
    self.target_inputs, self.target_outputs = self.define_network(pref="_target")
    
    # Only feature extracting part is initialized from pretrained model
    if not FLAGS.continue_training:
      # make sure you exclude the prediction layers of the model
      list_to_exclude = ["actor/global_step","actor_target/control","actor_target/aux_depth","actor/"]
      variables_to_restore = slim.get_variables_to_restore(exclude=list_to_exclude)
      # Map actor/Conv_var_name/... [!! without :0 !!] to MobilenetV1/Conv_var_name as saved in checkpoint
      variables_to_restore={'MobilenetV1/'+v.name[13:-2]:v for  v in variables_to_restore}
      # for k in variables_to_restore.keys(): print("{0}: {1}".format(k, variables_to_restore[k]))
    else: #If continue training
      list_to_exclude = []
      variables_to_restore = slim.get_variables_to_restore(exclude=list_to_exclude)
      # for v in variables_to_restore: print v.name
    
    # chkp.print_tensors_in_checkpoint_file(tf.train.latest_checkpoint(FLAGS.checkpoint_path), tensor_name='MobilenetV1/Conv2d_1_depthwise', all_tensors=True)
    if not FLAGS.scratch: 
      print('checkpoint: {}'.format(FLAGS.checkpoint_path))
      init_assign_op_target, init_feed_dict_target = slim.assign_from_checkpoint(tf.train.latest_checkpoint(FLAGS.checkpoint_path), variables_to_restore)

    ###### SOFT UPDATE TARGET->MAIN
    
    self.network_params = [v for v in slim.get_variables_to_restore(include=['actor/'],exclude=['actor/global_step'])]
    self.target_network_params = [v for v in slim.get_variables_to_restore(include=['actor_target/'],exclude=['actor/'])]
    # parameter used for critic network to know with tf.trainable_variables which vars are of the critic network
    self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    # for v in self.network_params : print v.name
    # print '---------------------------------'
    # for v in self.target_network_params: print v.name

    # Op for periodically updating target network with online network weights
    self.update_target_network_params = [ self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)+tf.multiply(self.target_network_params[i], 1.-self.tau)) for i in range(len(self.target_network_params))]

    # Define the training op based on the total loss
    self.define_train()
    
    init_all=tf_variables.global_variables_initializer()
    self.sess.run([init_all])
    if not FLAGS.scratch:
      self.sess.run([init_assign_op], init_feed_dict)
      self.sess.run([init_assign_op_target], init_feed_dict_target)
      print('Successfully loaded model from:{}'.format(FLAGS.checkpoint_path))
    else:
      print('Training model from scratch so no initialization.')
    
  def define_network(self, pref=""):
    '''build the network and set the tensors
    '''
    with tf.device(self.device):
      inputs = tf.placeholder(tf.float32, shape = self.input_size)
      args_for_scope={'weight_decay': FLAGS.actor_weight_decay,
                      'stddev':FLAGS.init_scale}
      if FLAGS.actor_network=='mobile':
        args_for_model={'inputs':inputs, 
                      'num_classes':self.action_dim,
                      'scope': self.prefix+pref} 
        with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training = not FLAGS.evaluate,**args_for_scope)):
          outputs, endpoints = mobile_net.mobilenet_v1(is_training=not FLAGS.evaluate,**args_for_model)
      else:
        raise NameError( '[model] Network is unknown: ', FLAGS.actor_network)
    return inputs, outputs
  
  def define_train(self):
    '''Optimize the actor by applying the weighted gradients
    '''
    # This gradient will be provided by the critic network: dq/da
    self.critic_gradients = tf.placeholder(tf.float32, [None, self.action_dim])
    
    # Combine the gradients here: dq/ds = dq/da * da/ds 
    # critic gradients are negative ? --> in order to maximize critic?
    self.actor_gradients = tf.gradients(self.outputs, self.network_params, -self.critic_gradients)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train = optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))

  def update_target_network(self):
    self.sess.run(self.update_target_network_params)

  def get_num_trainable_vars(self):
    return self.num_trainable_vars
  
  def forward_target(self, inputs):
    return self.sess.run(self.target_outputs, feed_dict={self.target_inputs: inputs , self.is_training: False})
  
  def backward(self, inputs, gradients):
    '''run backward pass applying gradients calculated
    dq/ds = dq/da * da/ds with dq/da ~ gradients coming from the critic network
    '''
    self.sess.run(self.train, feed_dict={self.inputs: inputs, self.critic_gradients: gradients, self.is_training: True})
  
  
