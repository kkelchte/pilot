
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import mobile_net

from tensorflow.contrib.slim import model_analyzer as ma
from tensorflow.python.ops import variables as tf_variables

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import numpy as np
import math



FLAGS = tf.app.flags.FLAGS

# INITIALIZATION
tf.app.flags.DEFINE_string("checkpoint_path", 'mobilenet_025', "Specify the directory of the checkpoint of the earlier trained model.")
tf.app.flags.DEFINE_boolean("continue_training", False, "Continue training of the prediction layers. If false, initialize the prediction layers randomly.")
tf.app.flags.DEFINE_boolean("scratch", False, "Initialize full network randomly.")

# TRAINING
tf.app.flags.DEFINE_float("weight_decay", 0.00004, "Weight decay of inception network")
tf.app.flags.DEFINE_float("init_scale", 0.0005, "Std of uniform initialization")
tf.app.flags.DEFINE_float("depth_weight", 1, "Define the weight applied to the depth values in the loss relative to the control loss.")
tf.app.flags.DEFINE_float("control_weight", 1, "Define the weight applied to the control loss.")
tf.app.flags.DEFINE_float("grad_mul_weight", 0.001, "Specify the amount the gradients of prediction layers.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Specify the probability of dropout to keep the activation.")
tf.app.flags.DEFINE_integer("clip_grad", 0, "Specify the max gradient norm: default 0 is no clipping, recommended 4.")
tf.app.flags.DEFINE_string("optimizer", 'adadelta', "Specify optimizer, options: adam, adadelta, gradientdescent, rmsprop")
# tf.app.flags.DEFINE_string("no_batchnorm_learning", True, "In case of no batchnorm learning, are the batch normalization params (alphas and betas) not further adjusted.")
# tf.app.flags.DEFINE_boolean("grad_mul", True, "Specify whether the weights of the prediction layers should be learned faster.")

"""
Build basic NN model
"""
class Model(object):
 
  def __init__(self,  session, input_size, output_size, prefix='model', device='/gpu:0', bound=1, depth_input_size=(55,74)):
    '''initialize model
    '''
    # np.random.seed(FLAGS.random_seed)
    # tf.set_random_seed(FLAGS.random_seed)
    
    self.sess = session
    self.output_size = output_size
    self.bound=bound
    self.input_size = input_size
    self.input_size[0] = None
    self.prefix = prefix
    self.device = device

    self.lr = FLAGS.learning_rate
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.define_network()
    
    
    # Only feature extracting part is initialized from pretrained model
    if not FLAGS.continue_training:
      # make sure you exclude the prediction layers of the model
      list_to_exclude = ["global_step"]
      list_to_exclude.append("MobilenetV1/control")
      list_to_exclude.append("MobilenetV1/aux_depth")
      list_to_exclude.append("concatenated_feature")
      list_to_exclude.append("control")
    else: #If continue training
      list_to_exclude = []
    variables_to_restore = slim.get_variables_to_restore(exclude=list_to_exclude)
      
    # get latest folder out of training directory if there is no checkpoint file
    if FLAGS.checkpoint_path[0]!='/':
      FLAGS.checkpoint_path = os.path.join(os.getenv('HOME'),'tensorflow/log',FLAGS.checkpoint_path)
    if not os.path.isfile(FLAGS.checkpoint_path+'/checkpoint'):
      FLAGS.checkpoint_path = FLAGS.checkpoint_path+'/'+[mpath for mpath in sorted(os.listdir(FLAGS.checkpoint_path)) if os.path.isdir(FLAGS.checkpoint_path+'/'+mpath) and not mpath[-3:]=='val' and os.path.isfile(FLAGS.checkpoint_path+'/'+mpath+'/checkpoint')][-1]
    
    print('checkpoint: {}'.format(FLAGS.checkpoint_path))
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(tf.train.latest_checkpoint(FLAGS.checkpoint_path), variables_to_restore)
    
    # create saver for checkpoints
    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=5)
    
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
  
  def define_network(self):
    '''build the network and set the tensors
    '''
    with tf.device(self.device):
      self.inputs = tf.placeholder(tf.float32, shape = self.input_size)
      if FLAGS.network=='mobile': 
        if FLAGS.n_fc: # concatenate consecutive features from a shared feature extracting CNN network
          self.inputs = tf.placeholder(tf.float32, shape = (self.input_size[0],self.input_size[1],self.input_size[2],FLAGS.n_frames*self.input_size[3]))
          with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training= True, weight_decay=FLAGS.weight_decay,
                               stddev=FLAGS.init_scale)):
            self.outputs, self.aux_depth, self.endpoints = mobile_net.mobilenet_n(self.inputs, num_classes=self.output_size, is_training=True)
          with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training= False, weight_decay=FLAGS.weight_decay,
                               stddev=FLAGS.init_scale)):
            self.controls, self.pred_depth, _ = mobile_net.mobilenet_n(self.inputs, num_classes=self.output_size, is_training=False)
        else: # Use only 1 frame to create a feature
          with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training=True, weight_decay=FLAGS.weight_decay,
                           stddev=FLAGS.init_scale)):
            self.outputs, self.endpoints = mobile_net.mobilenet_v1(self.inputs, num_classes=self.output_size, 
              is_training=True, dropout_keep_prob=FLAGS.dropout_keep_prob, depth_multiplier=FLAGS.depth_multiplier)
            self.aux_depth = self.endpoints['aux_depth_reshaped']
          with slim.arg_scope(mobile_net.mobilenet_v1_arg_scope(is_training=False, weight_decay=FLAGS.weight_decay,
                           stddev=FLAGS.init_scale)):
            self.controls, _ = mobile_net.mobilenet_v1(self.inputs, num_classes=self.output_size, 
              is_training=False, reuse = True, depth_multiplier=FLAGS.depth_multiplier)
            self.pred_depth = _['aux_depth_reshaped']
      else:
        raise NameError( '[model] Network is unknown: ', FLAGS.network)
      if(self.bound!=1 or self.bound!=0):
        self.outputs = tf.multiply(self.outputs, self.bound) # Scale output to -bound to bound

  def define_loss(self):
    '''tensor for calculating the loss
    '''
    with tf.device(self.device):
      self.targets = tf.placeholder(tf.float32, [None, self.output_size])
      self.loss = tf.losses.mean_squared_error(self.outputs, self.targets, weights=FLAGS.control_weight)
      if FLAGS.auxiliary_depth:
        self.depth_targets = tf.placeholder(tf.float32, [None,55,74])
        weights = FLAGS.depth_weight*tf.cast(tf.greater(self.depth_targets, 0), tf.float32) # put loss weight on zero where depth is negative.        
        self.depth_loss = tf.losses.huber_loss(self.aux_depth,self.depth_targets,weights=weights)
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

  def forward(self, inputs, auxdepth=False, targets=[], target_depth=[]):
    '''run forward pass and return action prediction
    inputs=batch of RGB iamges
    auxdepth = variable defining wether auxiliary depth should be predicted
    targets = supervised target control
    target_depth = supervised target depth
    '''
    tensors = [] 
    feed_dict={self.inputs: inputs}
    if auxdepth: # predict auxiliary depth
      tensors.append(self.pred_depth)
    
    if len(targets) != 0: # if target control is available, calculate loss
      tensors.append(self.loss)
      feed_dict[self.targets] = targets
      if not FLAGS.auxiliary_depth: tensors.append(self.total_loss)
    
    if len(target_depth) != 0 and FLAGS.auxiliary_depth:# if target depth is available, calculate loss
      tensors.append(self.depth_loss)
      feed_dict[self.depth_targets] = target_depth
      if len(targets) != 0: tensors.append(self.total_loss)

    results = self.sess.run(tensors, feed_dict=feed_dict)
    losses = {}
    aux_results = []
    
    if auxdepth: 
      aux_results.append(results.pop(0))
    
    if len(targets) != 0:
      losses['c']=results.pop(0) # control loss
      if not FLAGS.auxiliary_depth: losses['t']=results.pop(0)
    
    if len(target_depth) != 0:
      if FLAGS.auxiliary_depth: 
        losses['d']=results.pop(0) # depth loss
        if len(targets) != 0: losses['t']=results.pop(0)

    return losses, aux_results

  def backward(self, inputs, targets=[], depth_targets=[]):
    '''run forward pass and return action prediction
    '''
    tensors = [self.train_op]
    feed_dict = {self.inputs: inputs}
    tensors.append(self.loss)
    feed_dict[self.targets]=targets
    if FLAGS.auxiliary_depth:
      feed_dict[self.depth_targets] = depth_targets
      tensors.append(self.depth_loss)
    tensors.append(self.total_loss)
    
    results = self.sess.run(tensors, feed_dict=feed_dict)
    losses={}
    _ = results.pop(0) # train_op
    losses['c']=results.pop(0) # control loss
    if FLAGS.auxiliary_depth: losses['d']=results.pop(0)
    losses['t'] = results.pop(0) # total loss
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
    self.add_summary_var("loss_train_total")
    self.add_summary_var("loss_train_control")
    self.add_summary_var("loss_train_depth")
    self.add_summary_var("loss_val_total")
    self.add_summary_var("loss_val_control")
    self.add_summary_var("loss_val_depth")
    self.add_summary_var("loss_test_total")
    self.add_summary_var("loss_test_control")
    self.add_summary_var("loss_test_depth")
    
    if FLAGS.auxiliary_depth and FLAGS.plot_depth:
      name="depth_predictions"
      dep_images = tf.placeholder(tf.float32, [None, 500, 500, 3])
      self.summary_vars[name]=dep_images
      self.summary_ops[name]=tf.summary.image(name, dep_images, max_outputs=4)
    
  def summarize(self, sumvars):
    '''write summary vars with ops'''
    if self.writer:
      feed_dict={self.summary_vars[key]:sumvars[key] for key in sumvars.keys()}
      sum_op = tf.summary.merge([self.summary_ops[key] for key in sumvars.keys()])
      summary_str = self.sess.run(sum_op, feed_dict=feed_dict)
      self.writer.add_summary(summary_str,  tf.train.global_step(self.sess, self.global_step))
      self.writer.flush()
    
  
  
