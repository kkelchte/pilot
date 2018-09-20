
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim

from models import *
# import models.mobile_net as mobile_net
# import models.mobile_nfc_net as mobile_nfc_net
# import models.depth_q_net as depth_q_net
# import models.alex_net as alex_net


from tensorflow.contrib.slim import model_analyzer as ma
from tensorflow.python.ops import variables as tf_variables

import numpy as np

"""
Build basic NN model
"""
class Model(object):
 
  def __init__(self, FLAGS, session, prefix='model'):
    '''initialize model
    '''
    self.sess = session
    self.action_dim = FLAGS.action_dim
    # self.output_size = FLAGS.output_size
    self.prefix = prefix
    self.device = FLAGS.device

    # set in offline.py coming from data.py and used for training offline.
    self.factor_offsets={}

    self.FLAGS=FLAGS

    self.lr = self.FLAGS.learning_rate
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    
    #define the input size of the network input
    if self.FLAGS.network == 'mobile':
      self.input_size = [mobile_net.default_image_size[FLAGS.depth_multiplier], 
          mobile_net.default_image_size[FLAGS.depth_multiplier], 3]
    elif self.FLAGS.network =='mobile_nfc':
      self.input_size = [mobile_nfc_net.default_image_size[FLAGS.depth_multiplier], 
          mobile_nfc_net.default_image_size[FLAGS.depth_multiplier], 3*self.FLAGS.n_frames]
    elif sum([self.FLAGS.network.startswith(name) for name in ['alex','squeeze','tiny']]):
      versions={'alex': alex_net,
                'alex_v1': alex_net_v1,
                'alex_v2': alex_net_v2,
                'alex_v3': alex_net_v3,
                'alex_v4': alex_net_v4,
                'squeeze': squeeze_net,
                'squeeze_v1': squeeze_net_v1,
                'squeeze_v2': squeeze_net_v2,
                'squeeze_v3': squeeze_net_v3,
                'tiny':tiny_net,
                'tiny_CAM':tiny_CAM_net}
      self.input_size = versions[self.FLAGS.network].default_image_size
    else:
      raise NotImplementedError( 'Network is unknown: ', self.FLAGS.network)
    
    self.input_size=[None]+self.input_size
    self.output_size=int(self.FLAGS.n_factors*(self.action_dim if not self.FLAGS.discrete else self.action_dim * self.FLAGS.action_quantity))
    
    # define a network for training and for evaluation
    self.inputs= tf.placeholder(tf.float32, shape = self.input_size, name = 'Inputs')
    self.endpoints={}
    for mode in ['train', 'eval']:
      self.define_network(mode)
      params=0
      for t in tf.trainable_variables():
        if len(t.shape) == 4:
          params+=t.shape[0]*t.shape[1]*t.shape[3]
        elif len(t.shape) == 3:
          params+=t.shape[1]*t.shape[2]
        elif len(t.shape) == 1:
          params+=t.shape[0]        
        # print t.name
      print("total number of parameters: {0}".format(params))
      # import pdb; pdb.set_trace()
        
    if self.FLAGS.discrete:
      self.define_discrete_bins(FLAGS.action_bound, FLAGS.action_quantity)
      self.add_discrete_control_layers(self.endpoints['train']) #add probability output with softmax
      self.add_discrete_control_layers(self.endpoints['eval'])

    # add control_layers to parse from the outputs the correct control
    self.add_control_layer(self.endpoints['train'])
    self.add_control_layer(self.endpoints['eval'])

    # Only feature extracting part is initialized from pretrained model
    if not self.FLAGS.continue_training:
      # make sure you exclude the prediction layers of the model
      list_to_exclude = ["global_step"]
      list_to_exclude.append("MobilenetV1/control")
      list_to_exclude.append("MobilenetV1/aux_depth")
      list_to_exclude.append("H_fc_control")
      # list_to_exclude.append("outputs")
      list_to_exclude.append("MobilenetV1/q_depth")
      list_to_exclude.append("discriminator/gates")
    else: #If continue training
      list_to_exclude = []
    variables_to_restore = slim.get_variables_to_restore(exclude=list_to_exclude)
    
    # get latest folder out of training directory if there is no checkpoint file
    if self.FLAGS.checkpoint_path[0]!='/':
      self.FLAGS.checkpoint_path = self.FLAGS.summary_dir+self.FLAGS.checkpoint_path
    if not os.path.isfile(self.FLAGS.checkpoint_path+'/checkpoint'):
      try:
        self.FLAGS.checkpoint_path = self.FLAGS.checkpoint_path+'/'+[mpath for mpath in sorted(os.listdir(self.FLAGS.checkpoint_path)) if os.path.isdir(self.FLAGS.checkpoint_path+'/'+mpath) and not mpath[-3:]=='val' and os.path.isfile(self.FLAGS.checkpoint_path+'/'+mpath+'/checkpoint')][-1]
      except:
        pass  

    if not self.FLAGS.scratch: 
      print('checkpoint: {}'.format(self.FLAGS.checkpoint_path))
      try:
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(tf.train.latest_checkpoint(self.FLAGS.checkpoint_path), variables_to_restore)
      except Exception as e:
        print("Failed to initialize network {0} with checkpoint {1} so training from scratch: {2}".format(FLAGS.network, FLAGS.checkpoint_path, e.message))
        FLAGS.scratch = True

    # create saver for checkpoints
    self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
    
    # Add the loss and metric functions to the graph for both endpoints of train and eval.
    # self.targets = tf.placeholder(tf.int32, [None, self.output_size]) if FLAGS.discrete else tf.placeholder(tf.float32, [None, self.action_dim])
    # self.targets = tf.placeholder(tf.int32 if FLAGS.discrete else tf.float32, [None, self.output_size])
    self.targets = tf.placeholder(tf.float32, [None, self.output_size])
    self.weights = tf.placeholder(tf.float32, [None, self.output_size])
    # the discriminator is not strongly fixed to learn to predict the correct expert
    # the loss takes both the discriminator as the network outputs into account
    self.gate_targets = tf.placeholder(tf.float32, [None, self.FLAGS.action_quantity]) 
    # self.gate_targets = tf.placeholder(tf.float32, [None, self.FLAGS.n_factors])

    self.depth_targets = tf.placeholder(tf.float32, [None,55,74])
        
    self.define_loss(self.endpoints['train'])
    self.define_metrics(self.endpoints['eval'])

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
  
  def define_network(self, mode):
    '''build the network and set the tensors
    '''
    with tf.device(self.device):
      if self.FLAGS.network.startswith('mobile'):
        args={'inputs':self.inputs,
              'weight_decay': self.FLAGS.weight_decay,
              'stddev':self.FLAGS.init_scale,
              'initializer':self.FLAGS.initializer,
              'random_seed':self.FLAGS.random_seed,
              'num_outputs':self.output_size,
              'dropout_keep_prob':1-self.FLAGS.dropout_rate,
              'depth_multiplier':self.FLAGS.depth_multiplier}
        if not '_nfc' in self.FLAGS.network:
          self.endpoints[mode] = mobile_net.mobilenet(is_training=mode=='train', #in case of training do dropout
                                                      reuse = mode=='eval', #reuse in case of evaluation network
                                                      **args)
        else:
          args['n_frames']=self.FLAGS.n_frames
          self.endpoints[mode] = mobile_nfc_net.mobilenet_nfc(is_training=mode=='train',
                                                            reuse = mode=='eval',
                                                            **args)
      elif self.FLAGS.network.startswith('alex'):
        args={'inputs':self.inputs,
              'num_outputs':self.output_size,
              'verbose':True,
              'dropout_rate':self.FLAGS.dropout_rate if mode == 'train' else 0,
              'reuse':None if mode == 'train' else True,
              'is_training': mode == 'train'}
        versions={'alex': alex_net,
            'alex_v1': alex_net_v1,
            'alex_v2': alex_net_v2,
            'alex_v3': alex_net_v3,
            'alex_v4': alex_net_v4}
        self.endpoints[mode] = versions[self.FLAGS.network].alexnet(**args)
      elif self.FLAGS.network.startswith('squeeze'):
        args={'inputs':self.inputs,
              'num_outputs':self.output_size,
              'verbose':True,
              'dropout_rate':self.FLAGS.dropout_rate if mode == 'train' else 0,
              'reuse':None if mode == 'train' else True,
              'is_training': mode == 'train'}
        versions={'squeeze': squeeze_net,
            'squeeze_v1': squeeze_net_v1,
            'squeeze_v2': squeeze_net_v2,
            'squeeze_v3': squeeze_net_v3}
        self.endpoints[mode] = versions[self.FLAGS.network].squeezenet(**args)
      elif self.FLAGS.network.startswith('tiny'):
        args={'inputs':self.inputs,
              'num_outputs':self.output_size,
              'verbose':True,
              'dropout_rate':self.FLAGS.dropout_rate if mode == 'train' else 0,
              'reuse':None if mode == 'train' else True,
              'is_training': mode == 'train'}
        versions={'tiny': tiny_net,
                  'tiny_CAM': tiny_CAM_net,}
        self.endpoints[mode] = versions[self.FLAGS.network].tinynet(**args)
      else:
        raise NameError( '[model] Network is unknown: ', self.FLAGS.network)
      self.define_gating_function(mode)

  def define_gating_function(self, mode):
    """Add layers to the network to train a discriminator based on the inner feature representation.
    This gating function can be based on all intermediate filter activations or only on the final feature.
    The tensors are added to the endpoints with name [gate]
    """
    if self.FLAGS.discriminator_input == 'feature': #tiny net is too small to extract real features from
      self.feature_names={'mobile':'AvgPool_1a',
                        'alex':'fc_7',
                        'squeeze':'10_conv',
                        'tiny':'conv_2'}
      feature_sizes={'mobile':[1,1],
                        'alex':[1,1],
                        'squeeze':[13,13],
                        'tiny':[20,20]}
      gating_inputs=self.endpoints[mode][self.feature_names[self.FLAGS.network.split('_')[0] if not 'CAM' in self.FLAGS.network else 'avg_pool']]
      end_point='gates'
      with tf.variable_scope('discriminator'):
        if len(gating_inputs.shape) == 4:
          if gating_inputs.shape[1] != 1:
            gating_inputs=tf.expand_dims(tf.layers.flatten(gating_inputs),axis=1)
          else:
            gating_inputs=tf.squeeze(gating_inputs,[1])
        ep=tf.layers.conv1d(gating_inputs,
                             filters=self.FLAGS.n_factors,
                             kernel_size=1,
                             # kernel_size=feature_sizes[self.FLAGS.network.split('_')[0]],
                             strides=1,
                             padding='valid',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name=end_point,
                             reuse=mode=='eval')
      if len(ep.shape) == 4:
        self.endpoints[mode][end_point]=tf.squeeze(ep,[1,2],name=end_point+'_squeeze')
      elif len(ep.shape) == 3:
        self.endpoints[mode][end_point]=tf.squeeze(ep,[1],name=end_point+'_squeeze')

    elif self.FLAGS.discriminator_input == 'activations':
      raise NotImplementedError
    else: # self.FLAGS.discriminator_input == 'image'
      gating_inputs=self.inputs
      with tf.variable_scope('discriminator'):
        end_point='gates_conv_1'
        ep=tf.layers.conv2d(gating_inputs,
                            filters=10,
                            kernel_size=[6,6],
                            strides=3,
                            padding='valid',
                            activation=tf.nn.relu,
                            use_bias=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name=end_point,
                            reuse=mode=='eval')
        self.endpoints[mode][end_point]=ep
        print("shape gates_conv_1: {}".format(ep.shape))

        end_point='gates_conv_2'
        ep=tf.layers.conv2d(ep,
                            filters=20,
                            kernel_size=[3,3],
                            strides=2,
                            padding='valid',
                            activation=tf.nn.relu,
                            use_bias=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name=end_point,
                            reuse=mode=='eval')
        self.endpoints[mode][end_point]=ep                    
        print("shape gates_conv_2: {}".format(ep.shape))
        end_point='gates'
        ep=tf.layers.conv2d(ep,
                             filters=self.FLAGS.n_factors,
                             kernel_size=[20,20],
                             strides=1,
                             padding='valid',
                             activation=None,
                             use_bias=False,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name=end_point,
                             reuse=mode=='eval')
        self.endpoints[mode][end_point]=tf.squeeze(ep,[1,2],name=end_point+'_squeeze')
    
  def define_discrete_bins(self, action_bound, action_quantity):
    '''
    Calculate the boundaries of the different bins for discretizing the targets.
    Define a list form [-bound, +bound] with action_quantity steps and keep the boundaries in a field.
    Returns the boundaries as well as the values.
    Note that this all is abount control bins unrelated to different factors or offsets.
    '''
    # the width of each bin over the range defined by action_bound
    bin_width=2*action_bound/(action_quantity-1.)
    # Define the corresponding float values for each index [0:action_quantity]
    self.control_values=[-action_bound+n*bin_width for n in range(action_quantity)]
    b=round(-action_bound+bin_width/2,4)
    self.boundaries=[]
    while b < action_bound:
      self.boundaries.append(b)
      b=round(b+bin_width,4)
    assert len(self.boundaries) == action_quantity-1  
    print("[model.py]: Divided {0} discrete actions over {1} with boundaries {2}.".format(action_quantity, self.control_values, self.boundaries))
    # add hash table to convert discrete bin to float with tensors
    keys = tf.constant(np.arange(action_quantity), dtype=tf.int32)
    values = tf.constant(np.array(self.control_values),  dtype=tf.int32)
    default_value = tf.constant(0,  dtype=tf.int32)
    with tf.variable_scope('table'):
      self.control_values_from_tensors = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_value,
          name='control_values')
    self.control_values_from_tensors.init.run(session=self.sess)

  def continuous_to_discrete(self, continuous_value):
    """
    Change a continuous value from between [-bound:bound]
    to the corresponding discrete bin_index [0:action_quantity-1]
    Example: -0.35 --> 0 or +1.1 --> 2
    Can also handle numpy arrays which it reshapes
    """
    if isinstance(continuous_value, np.ndarray):
      shape=continuous_value.shape
      discrete_values=[]
      for c in continuous_value:
        discrete=0
        for b in self.boundaries:
            if b < c:
                discrete+=1
            else:
                break
        discrete_values.append(discrete)
      return np.array(discrete_values).reshape(shape)
    elif isinstance(continuous_value, tf.Tensor):
      raise(NotImplementedError)
    else:
      discrete=0
      for b in self.boundaries:
        if b > continuous_value:
          return discrete
        else:
          discrete+=1
      return discrete

  def discrete_to_continuous(self, discrete_value, name=None):
    """
    Changes a discrete value to the continuous value (center of the corresponding bin)
    [0,1,2] --> [-1,0,1]
    """
    if isinstance(discrete_value, tf.Tensor):
      return self.control_values_from_tensors.lookup(discrete_value,name=name)
    elif hasattr(discrete_value, '__contains__'):
      return [self.control_values[int(v)] for v in discrete_value]
    else:
      return self.control_values[int(discrete_value)]

  def one_hot(self, index):
    """
    Index should be an integer 32.
    Change from index to one_hot encoding:
    0 --> 1,0,0
    1 --> 0,1,0
    2 --> 0,0,1
    """
    one_hot=np.zeros((len(self.control_values)))
    one_hot[index]=1
    return one_hot

  def one_hot_to_control_digits(self, one_hots, name=None):
    """
    Changes an array of one_hots over different factors to control indices [0,1,2].
    Only tensor version is used in code and only for targets.
    """
    if isinstance(one_hots, tf.Tensor):
      digits=tf.floormod(tf.argmax(one_hots,axis=-1, name=name, output_type=tf.int32), tf.constant(self.FLAGS.action_quantity, dtype=tf.int32))
      return digits
    else:
      raise NotImplementedError

  def adjust_targets(self, targets, factors=[]):
    """
    Create new targets of shape [batch_size, num_outputs].
    In the continuous case the target control is placed in the bin according to the factor of that sample.
    If discrete we work with an offset when changing from continuous to discrete.
    It returns the new targets.

    In case there are no factors given, the targets are filled in for the first factor.
    """
    new_targets=np.zeros((targets.shape[0],self.output_size))+(999 if self.FLAGS.single_loss_training else 0)
    if len(factors)==0: factors=[self.factor_offsets.keys()[0]]*len(targets)
    if not self.FLAGS.discrete:
      for i,t in enumerate(targets):
        new_targets[i,self.factor_offsets[factors[i]]]=t
    else:
      for i,t in enumerate(targets):
        new_targets[i,self.factor_offsets[factors[i]]:self.factor_offsets[factors[i]]+self.FLAGS.action_quantity]=self.one_hot(self.continuous_to_discrete(t))
    return new_targets

  def adjust_weights(self, targets, factors):
    """
    Adjust the weights over the loss (MSE)
    """
    # initialize all with non-expert weight
    weights = np.zeros((len(targets), self.output_size)) + self.FLAGS.non_expert_weight
    # except of course for the expert outputs extracted from factors
    for i,t in enumerate(targets): #iterate over samples in batch
      weights[i,self.factor_offsets[factors[i]]:self.factor_offsets[factors[i]]+(1 if not self.FLAGS.discrete else self.FLAGS.action_quantity)]=1
    return weights  

  def add_control_layer(self, endpoints):
    """Each expert part has one output node in continuous case or different in discrete case.
    From the outputs of all the experts one final output should be extract with a control.
    In the continuous case this is the mean over the output of all experts.
    In the discrete case, the maximum is taken from the three discrete bins and the index is kept: 0,1,2.
    """
    if not self.FLAGS.discrete:
      # The outputs of the gating functions are used as weights of the outputs of the different experts
      # The weighted sum of these n experts corresponds to the final control
      end_point='weighted_sum'
      ws=tf.multiply(endpoints['gates'],tf.reshape(endpoints['outputs'],(-1,self.FLAGS.n_factors)))
      endpoints[end_point]=ws
      end_point='control'
      endpoints[end_point] = ws
    else:    
      #The weights are the predictions of the discriminator with shape(NxF) weighting each factor output with a likelihood.
      # make weights of shape NxFxA by tiling: NxF --> NxFx1 --> NxFxA
      end_point='weights_reshaped'
      w=tf.tile(tf.expand_dims(endpoints['gates'],axis=-1),tf.constant([1,1,self.FLAGS.action_quantity]))
      endpoints[end_point]=w
      #The outputs are the outputs of the network with shape Nx(AF) --> NxFxA --> sum over F to get NxA.
      end_point='weighted_sum'
      ws=tf.reduce_sum(tf.multiply(w,tf.reshape(endpoints['outputs'],(-1,self.FLAGS.n_factors,self.FLAGS.action_quantity))),axis=-2)
      endpoints[end_point]=ws
      # take the max over the actions to get digits
      end_point='digit'
      digit = tf.argmax(ws, axis=-1,name=end_point, output_type=tf.int32)
      endpoints[end_point]=digit
      end_point='control' # change local control digit to continuous control float
      endpoints[end_point] = self.discrete_to_continuous(digit, name=end_point)

  def add_discrete_control_layers(self, endpoints):
    """
    The output of the network defined in the previous function returns logits before activations.
    In case of discrete controls these logits have to be transformed to probabilities and possibly control indices.
    The operations are defined as tensors and added to the train and eval endpoints.
    """
    # for train and eval endpoints
    end_point='probs'
    probs = tf.nn.softmax(endpoints['outputs'], axis=-1, name='softmax')
    endpoints[end_point] = probs
    # end_point='digit'
    # digit = tf.argmax(probs, axis=-1, name='argmax', output_type=tf.int32)
    # endpoints[end_point] = digit
    # end_point='control'
    # endpoints[end_point] = self.discrete_to_continuous(digit, name=end_point)

  def get_gate_targets(self, targets):
    """
    Creates from the list of factors ['factor1', 'factor0', 'factor5',...]
    a onehot label for training the discriminator of NxF parsed from factor_offsets.
    [0 1 0 0 0 ..,
     1 0 0 0 0 ..,
     0 0 0 0 0 ..]
    """
    # gate_labels = np.zeros((len(factors), self.FLAGS.n_factors))
    # for i,f in enumerate(factors): 
    #   gate_labels[i,int(self.factor_offsets[f]/(self.FLAGS.action_quantity if self.FLAGS.discrete else 1))]=1
    """New implementation:
    targets should be just discretized controls of shape NxA 
    """
    gate_labels = []
    for t in targets:
      gate_labels.append(self.one_hot(self.continuous_to_discrete(t)))
    return np.asarray(gate_labels)

  def define_loss(self, endpoints):
    '''tensors for calculating the loss are added in LOSS collection
    total_loss includes regularization loss defined in tf.layers
    '''
    with tf.device(self.device):
      if not self.FLAGS.discrete:
        self.loss = tf.losses.mean_squared_error(self.targets, endpoints['outputs'], weights=self.weights)
      else:
        if self.FLAGS.loss == 'ce':
          loss = -tf.multiply(tf.multiply(self.targets,self.weights), tf.log(endpoints['probs']+0.001))
          loss = loss-tf.multiply(tf.multiply((1-self.targets),self.weights), tf.log(1-endpoints['probs']))
          self.loss = tf.reduce_mean(loss)
          tf.losses.add_loss(tf.reduce_mean(self.loss))        
        elif self.FLAGS.loss == 'smce':
          self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.targets, logits=endpoints['outputs'])
        else:
          self.loss = tf.losses.mean_squared_error(self.targets, endpoints['outputs'], weights=self.weights)
      
      # self.discriminator_loss = tf.losses.mean_squared_error(self.gate_targets, endpoints['gates'])
      self.discriminator_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.gate_targets, logits=endpoints['weighted_sum'])
      # tf.losses.add_loss(self.discriminator_loss)

      if self.FLAGS.auxiliary_depth:
        weights = self.FLAGS.depth_weight*tf.cast(tf.greater(self.depth_targets, 0), tf.float32) # put loss weight on zero where depth is negative or zero.        
        self.depth_loss = tf.losses.huber_loss(self.depth_targets,endpoints['aux_depth_reshaped'],weights=weights)
        tf.losses.add_loss(self.depth_loss)
      self.total_loss = tf.losses.get_total_loss()

  def define_metrics(self, endpoints):
    '''tensors to evaluate the performance. Uses only the endpoints of the 'eval' copy of the network
    '''
    with tf.device(self.device):
      with tf.variable_scope('metrics'):
        self.mse={}
        self.mse_depth={}
        self.max_expert_accuracy={}
        self.accuracy={}
        self.gating_accuracy={}
        for mode in ['train', 'val']: #following modes offline training
          self.gating_accuracy[mode] = tf.metrics.accuracy(tf.argmax(self.gate_targets, axis=-1, output_type=tf.int32), tf.argmax(endpoints['weighted_sum'],axis=-1, output_type=tf.int32), name='gating_accuracy_'+mode)
          
          # keep running variables for both validation and training data
          if self.FLAGS.discrete: # accuracy is measured on local control digits [0,1,2]
            self.mse[mode] = tf.metrics.mean_squared_error(self.one_hot_to_control_digits(self.targets), endpoints['digit'], name="mse_"+mode)
            self.max_expert_accuracy[mode] = tf.metrics.accuracy(self.one_hot_to_control_digits(self.targets), self.one_hot_to_control_digits(endpoints['outputs']), name='max_expert_accuracy_'+mode)
            self.accuracy[mode] = tf.metrics.accuracy(self.one_hot_to_control_digits(self.targets), endpoints['digit'], name='accuracy_'+mode)
          else:
            self.mse[mode] = tf.metrics.mean_squared_error(self.targets, endpoints['outputs'], weights=self.weights, name="mse_"+mode)
          if self.FLAGS.auxiliary_depth:
            self.mse_depth[mode] = tf.metrics.mean_squared_error(self.depth_targets, endpoints['aux_depth_reshaped'], weights=self.weights, name="mse_depth_"+mode)
    # keep metric variables in a field so they are easily updated
    self.metric_variables=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")

  def reset_metrics(self):
    """ Sets all local variables of metrics back to zero
    """
    # initialize both train and val metric variables
    self.sess.run(tf.variables_initializer(var_list=self.metric_variables))
    
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
      # gradient_multipliers={}
      # mobile_variables = [v for v in tf.global_variables() if (v.name.find('Adadelta')==-1 and v.name.find('BatchNorm')==-1 and v.name.find('Adam')==-1  and v.name.find('control')==-1 and v.name.find('aux_depth')==-1)]
      # gradient_multipliers = {v.name: self.FLAGS.grad_mul_weight for v in mobile_variables}

      # if self.FLAGS.no_batchnorm_learning:
      #   batchnorm_variables = [v for v in tf.global_variables() if v.name.find('BatchNorm')!=-1]
      #   gradient_multipliers = {v.name: 0 for v in mobile_variables}
      
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        self.train_op = self.optimizer.minimize(self.total_loss,
                                                global_step=self.global_step)

      # discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
      #                                       "discriminator")
      # self.train_op = self.optimizer.minimize(self.discriminator_loss, var_list=discriminator_vars, global_step=self.global_step)

      # print discriminator_vars
      # import pdb; pdb.set_trace()
      # self.train_op = slim.learning.create_train_op(self.total_loss, 
      #   self.optimizer, 
      #   global_step=self.global_step, 
      #   gradient_multipliers=gradient_multipliers, 
      #   clip_gradient_norm=self.FLAGS.clip_grad)

  def forward(self, inputs, auxdepth=False, targets=[], factors=[], depth_targets=[]):
    '''run forward pass and return action prediction
    inputs=batch of RGB images
    auxdepth = variable defining wether auxiliary depth should be predicted
    targets = supervised target control
    depth_targets = supervised target depth
    '''
    feed_dict={self.inputs: inputs}  

    tensors = [self.endpoints['eval']['control']]
    
    if auxdepth: # predict auxiliary depth
      tensors.append(self.endpoints['eval']['aux_depth_reshaped'])
    
    if len(targets) != 0 and len(factors) != 0: # if target control is available, calculate loss
      feed_dict[self.gate_targets] = self.get_gate_targets(targets)
      tensors.append(self.mse['val']) # 1 is required to get the update operation
      tensors.append(self.gating_accuracy['val']) # 1 is required to get the update operation
      
      # tensors.append(self.mse['val'][1]) # 1 is required to get the update operation
      new_targets = self.adjust_targets(targets,factors)
      feed_dict[self.targets]=new_targets  

      weights = self.adjust_weights(targets, factors)
      feed_dict[self.weights] = weights

      if self.FLAGS.discrete: 
        tensors.append(self.max_expert_accuracy['val'])
        tensors.append(self.accuracy['val'])
      
    if len(depth_targets) != 0 and self.FLAGS.auxiliary_depth:# if target depth is available, calculate loss
      tensors.append(self.mse_depth['val'])
      # tensors.append(self.mse_depth['val'][1])
      feed_dict[self.depth_targets] = depth_targets

    if len(targets) != 0 and len(factors) == 0: # evaluate on test data without factors but with control targets
      new_targets = self.adjust_targets(targets)
      feed_dict[self.targets]=new_targets  
      if self.FLAGS.discrete: tensors.append(self.accuracy['val'])

    results = self.sess.run(tensors, feed_dict=feed_dict)

    output=results.pop(0)
    # metrics = {}
    aux_results = {}   

    if auxdepth: 
      aux_results['d']=results.pop(0)

    # print "target: ", targets
    # print "output: ", self.sess.run(self.endpoints['eval']['outputs'], feed_dict=feed_dict)
    # print "loss: ", self.sess.run(self.loss, feed_dict=feed_dict)
    # print "total loss: ",self.sess.run(self.total_loss, feed_dict=feed_dict)
    # print "metrics: "
    # for v in self.metric_variables:
    #   print("{0} : {1}".format(v.name, self.sess.run(v)))
    
    
    # if len(targets) != 0:
    #   metrics['control']=results.pop(0) # MSE of control prediction
    #   if self.FLAGS.discrete: metrics['accuracy']=results.pop(0) # accuracy of control prediction
    
    # if len(depth_targets) != 0 and self.FLAGS.auxiliary_depth:
    #   metrics['depth']=results.pop(0) # depth loss

    return output, aux_results

  def backward(self, inputs, targets, factors, depth_targets=[], sumvar=None):
    '''run backward pass and return losses
    inputs: batch of images
    targets: batch of control labels
    depth_targets: batch of depth labels
    sumvar: summary variables dictionary to which values can be added.
    '''
    tensors = [self.train_op]
    feed_dict = {self.inputs: inputs}

    new_targets = self.adjust_targets(targets, factors)
    feed_dict[self.targets] = new_targets

    gate_targets=self.get_gate_targets(targets)
    feed_dict[self.gate_targets] = gate_targets
    
    weights = self.adjust_weights(targets, factors)

    feed_dict[self.weights] = weights

    
    # append loss
    # tensors.append(self.discriminator_loss)
    tensors.append(self.total_loss)

    # append visualizations
    if self.FLAGS.histogram_of_activations and isinstance(sumvar,dict):
      for e in sorted(self.endpoints['eval'].keys()):
        tensors.append(self.endpoints['eval'][e])
    
    if self.FLAGS.histogram_of_weights and isinstance(sumvar,dict):
      for v in tf.trainable_variables():
        tensors.append(v.value())
    
    # append updates for metrics
    # tensors.append(self.mse['train'][1])
    tensors.append(self.mse['train'])

    tensors.append(self.gating_accuracy['train'])
    
    if self.FLAGS.discrete: 
      tensors.append(self.max_expert_accuracy['train'])
      tensors.append(self.accuracy['train'])

    if self.FLAGS.auxiliary_depth:
      feed_dict[self.depth_targets] = depth_targets
      tensors.append(self.depth_mse['train'])

    results = self.sess.run(tensors, feed_dict=feed_dict)
    
    _ = results.pop(0) # train_op
    
    losses={'total':results.pop(0)}

        
    if self.FLAGS.histogram_of_activations and isinstance(sumvar,dict):
      for e in sorted(self.endpoints['eval'].keys()):
        res = results.pop(0)
        try:
          sumvar['activations_'+e].extend(res)
        except:
          sumvar['activations_'+e]=res

    if self.FLAGS.histogram_of_weights and isinstance(sumvar,dict):
      for v in tf.trainable_variables():
        res = results.pop(0)
        try:
          sumvar['weights_'+v.name].extend(res)
        except:
          sumvar['weights_'+v.name]=res

    return losses
  
  def get_metrics(self):
    """
    This function calls the the final results of the different metrics
    and returns them.
    """
    tensors=[]
    results={}
    for mode in ['train', 'val']:  
      tensors.append(self.mse[mode][0])
      tensors.append(self.gating_accuracy[mode][0])
      if self.FLAGS.discrete:
        tensors.append(self.max_expert_accuracy[mode][0])
        tensors.append(self.accuracy[mode][0])
      if self.FLAGS.auxiliary_depth:
        tensors.append(self.mse_depth[mode][0])
    output=self.sess.run(tensors)
    for mode in ['train', 'val']:
      results['mse_'+mode]=output.pop(0)
      results['gating_accuracy_'+mode]=output.pop(0)
      if self.FLAGS.discrete:
        results['max_expert_accuracy_'+mode]=output.pop(0)
        results['accuracy_'+mode]=output.pop(0)
      if self.FLAGS.auxiliary_depth:
        results['mse_depth_'+mode]=output.pop(0)
    return results

  def save(self, logfolder):
    '''save a checkpoint'''
    self.saver.save(self.sess, logfolder+'/my-model', global_step=tf.train.global_step(self.sess, self.global_step))
    
  def add_summary_var(self, name):
    var_name = tf.Variable(0., trainable=False, name=name)
    self.summary_vars[name]=var_name
    self.summary_ops[name] = tf.summary.scalar(name, var_name)
    
  def build_summaries(self): 
    self.summary_vars = {}
    self.summary_ops = {}
    self.add_summary_var('Loss_train_total')

    for t in ['train', 'val']:
      for l in ['mse', 'gating_accuracy', 'accuracy', 'max_expert_accuracy','mse_depth']:
        name='{0}_{1}'.format(l,t)
        self.add_summary_var(name)
    for d in ['current','furthest']:
      for t in ['train', 'test']:
        for w in ['','corridor','sandbox','forest','canyon','esatv1', 'esatv2']:
          name = 'Distance_{0}_{1}'.format(d,t)
          if len(w)!=0: name='{0}_{1}'.format(name,w)
          self.add_summary_var(name)
    self.add_summary_var('driving_time')      
    self.add_summary_var('imitation_loss')      
    self.add_summary_var('depth_loss')
    if self.FLAGS.auxiliary_depth and self.FLAGS.plot_depth:
      name="depth_predictions"
      dep_images = tf.placeholder(tf.uint8, [1, 400, 400, 3])
      # dep_images = tf.placeholder(tf.float32, [1, 400, 400, 3])
      self.summary_vars[name]=dep_images
      self.summary_ops[name]=tf.summary.image(name, dep_images)
    if self.FLAGS.histogram_of_activations:
      for e in self.endpoints['eval'].keys():
        data=tf.placeholder(tf.float32, self.endpoints['eval'][e].shape)
        self.summary_vars['activations_'+e] = data 
        self.summary_ops['activations_'+e] = tf.summary.histogram('activations_'+e, data)
    if self.FLAGS.histogram_of_weights:
      for v in tf.trainable_variables():
        data=tf.placeholder(tf.float32)
        self.summary_vars['weights_'+v.name] = data
        self.summary_ops['weights_'+v.name] = tf.summary.histogram('weights_'+v.name.split(':')[0], v)

  def summarize(self, sumvars):
    '''write summary vars with ops'''
    if self.writer:
      feed_dict={self.summary_vars[key]:sumvars[key] for key in sumvars.keys()}
      sum_op = tf.summary.merge([self.summary_ops[key] for key in sumvars.keys()])
      summary_str = self.sess.run(sum_op, feed_dict=feed_dict)
      self.writer.add_summary(summary_str,  tf.train.global_step(self.sess, self.global_step))
      self.writer.flush()
