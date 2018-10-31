
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
                'tiny_v1':tiny_net_v1,
                'tiny_v2':tiny_net_v2,
                'tiny_v3':tiny_net_v3,
                'tiny_CAM':tiny_CAM_net}
      self.input_size = versions[self.FLAGS.network].default_image_size
    else:
      raise NotImplementedError( 'Network is unknown: ', self.FLAGS.network)
    
    self.input_size=[None]+self.input_size
    self.output_size=int(self.action_dim if not self.FLAGS.discrete else self.action_dim * self.FLAGS.action_quantity)

    # define a network for training and for evaluation
    self.inputs= tf.placeholder(tf.float32, shape = self.input_size, name = 'Inputs')
    self.endpoints={}
    for mode in ['train', 'eval']:
      self.define_network(mode)
      params=sum([reduce(lambda x,y: x*y, v.get_shape().as_list()) for v in tf.trainable_variables()])
      print("total number of parameters: {0}".format(params))
  
    if self.FLAGS.discrete:
      self.define_discrete_bins(FLAGS.action_bound, FLAGS.action_quantity)
      self.add_discrete_control_layers(self.endpoints['train'])
      self.add_discrete_control_layers(self.endpoints['eval'])
    
    # Only feature extracting part is initialized from pretrained model
    if not self.FLAGS.continue_training:
      # make sure you exclude the prediction layers of the model
      list_to_exclude = ["global_step"]
      list_to_exclude.append("MobilenetV1/control")
      list_to_exclude.append("MobilenetV1/aux_depth")
      list_to_exclude.append("H_fc_control")
      list_to_exclude.append("outputs")
      list_to_exclude.append("MobilenetV1/q_depth")
      list_to_exclude.append("Omega")
      print("[model.py]: only load feature extracting part in network.")
    else: #If continue training
      print("[model.py]: continue training of total network.")
      # list_to_exclude = ["Omega"]
      list_to_exclude = []
      # In case of lifelonglearning and continue learning: 
      # add variables for importance weights of previous domain and keep optimal variables for previous domain
    if self.FLAGS.lifelonglearning or self.FLAGS.update_importance_weights:
      self.define_importance_weights(self.endpoints['train'])

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
    self.saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)
    
    # Add the loss and metric functions to the graph for both endpoints of train and eval.
    self.targets = tf.placeholder(tf.int32, [None, self.action_dim]) if FLAGS.discrete else tf.placeholder(tf.float32, [None, self.action_dim])
    self.depth_targets = tf.placeholder(tf.float32, [None,55,74])
        
    self.define_metrics(self.endpoints['eval'])

    if self.FLAGS.continue_training and self.FLAGS.lifelonglearning:
      self.define_star_variables(self.endpoints['train'])

    # Define the training op based on the total loss
    self.define_loss(self.endpoints['train'])
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

    if self.FLAGS.continue_training and self.FLAGS.lifelonglearning:
      # print info on loaded importance weights
      for v in tf.trainable_variables():
        weights=self.sess.run(self.importance_weights[v.name])
        weights=weights.flatten()
        # print("{0}: {1} ({2}) min: {3} max: {4}".format(v.name, np.mean(weights), np.var(weights), np.amin(weights), np.amax(weights)))
        print("| {0} | {1} | {2} | {3} | ".format(v.name, 
                                                  np.percentile(weights,1),
                                                  np.percentile(weights,50),
                                                  np.percentile(weights,100)))

      # assign star_variables after initialization
      self.sess.run([tf.assign(self.star_variables[v.name], v) for v in tf.trainable_variables()])

  
  def define_network(self, mode):
    '''build the network and set the tensors
    '''
    with tf.device(self.device):
      if self.FLAGS.network.startswith('mobile'):
        args={'inputs':self.inputs,
              'auxiliary_depth': self.FLAGS.auxiliary_depth,
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
                  'tiny_v1': tiny_net_v1,
                  'tiny_v2': tiny_net_v2,
                  'tiny_v3': tiny_net_v3,
                  'tiny_CAM': tiny_CAM_net}
        self.endpoints[mode] = versions[self.FLAGS.network].tinynet(**args)
      else:
        raise NameError( '[model] Network is unknown: ', self.FLAGS.network)

  def define_discrete_bins(self, action_bound, action_quantity):
    '''
    Calculate the boundaries of the different bins for discretizing the targets.
    Define a list form [-bound, +bound] with action_quantity steps and keep the boundaries in a field.
    Returns the boundaries as well as the values
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
    keys = tf.constant(np.arange(action_quantity,dtype=np.int64), dtype=tf.int64)
    values = tf.constant(np.array(self.control_values),  dtype=tf.float32)
    default_value = tf.constant(0,  dtype=tf.float32)
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
    end_point='digit'
    digit = tf.argmax(probs, axis=-1, name='argmax', output_type=tf.int32)
    endpoints[end_point] = digit
    end_point='control'
    endpoints[end_point] = self.discrete_to_continuous(tf.cast(digit, tf.int64))
  
  def define_star_variables(self, endpoints):
    '''Define star variables for previous domains used in combination with the importance weights for a lifelong regularization term in the loss.
    In case copy is True the last domain creates new variables and copy the current weights of the model in these variables to introduce a new set of start variables corresponding to the last domain.
    Star variables is a dictionary keeping a list of variables for each domain.
    '''
    self.star_variables={}
    for v in tf.trainable_variables():
      # self.star_variables[v.name]=self.sess.run(v)
      # self.star_variables[v.name]= self.var_list[v].eval()
      self.star_variables[v.name]=tf.get_variable('Star/'+v.name.split(':')[0],
                                                      initializer=np.zeros(v.get_shape().as_list()).astype(np.float32),
                                                      dtype=tf.float32,
                                                      trainable=False)

  def define_importance_weights(self, endpoints):
    '''Define an important weight ~ omegas for each trainable variable
    domain corresponds to the dataset
    '''
    self.importance_weights={}
    for v in tf.trainable_variables():
      self.importance_weights[v.name]=tf.get_variable('Omega/'+v.name.split(':')[0],
                                                      initializer=np.zeros(v.get_shape().as_list()).astype(np.float32),
                                                      dtype=tf.float32,
                                                      trainable=False)
    
  def update_importance_weights(self, inputs):
    '''Take one episode of data and keep track of gradients.
    Calculate a importance value between 0 and 1 for each weight.
    '''
    # self.sess.run([tf.assign(self.importance_weights[v.name], np.ones((v.shape.as_list()))) for i,v in enumerate(tf.trainable_variables())])
    # return

    # Add model variables for importance_weights for current dataset
    gradients=[]
    # batchsize = 200
    batchsize = 1
    #number of batches of 200
    N=int(inputs.shape[0]/batchsize)
    print('[model.py]: update importance weights on {0} batches of batchsize.'.format(N))
    for i in range(N):
      # print i
      batch_inputs=inputs[batchsize*i:batchsize*(i+1)]
      # get gradients for a batch of images

      results = self.sess.run(tf.gradients(tf.square(tf.norm(self.endpoints['train']['outputs'])), tf.trainable_variables()), feed_dict={self.inputs: batch_inputs})

      # print results
      # import pdb; pdb.set_trace()
      # sum up over the absolute values
      try:
        gradients = [gradients[j]+np.abs(results[j]) for j in range(len(results))]
      except:
        gradients = [np.abs(results[j]) for j in range(len(results))]
    #gradients are summed up by tf.gradients so average over the batchsize samples
    # divide to get the mean absolute value
    gradients = [g/(batchsize*N) for g in gradients]

    new_weights=[gradients[i] + self.sess.run(self.importance_weights[v.name]) for i,v in enumerate(tf.trainable_variables())]

    # for i,v in enumerate(tf.trainable_variables()):
    #   print("{0} {1} {2}".format(v.name,self.importance_weights[v.name].shape, gradients[i].shape))
    # import pdb; pdb.set_trace()    

    # assign these values to the importance weight variables
    self.sess.run([tf.assign(self.importance_weights[v.name], new_weights[i]) for i,v in enumerate(tf.trainable_variables())])
    
    # with open(self.FLAGS.summary_dir+self.FLAGS.log_tag+"/omegas",'w') as f:
    #   for i,v in enumerate(tf.trainable_variables()):
    #     f.write("{0}: {1}\n".format(v.name, new_weights[i]))

    with open(self.FLAGS.summary_dir+self.FLAGS.log_tag+"/omegas_w",'w') as f:
      for i,v in enumerate(tf.trainable_variables()):
        f.write("{0}: {1}\n".format(v.name, self.sess.run(self.importance_weights[v.name])))

    # copy importance weights in dictionary to save as numpy file
    copied_weights={}
    for i,v in enumerate(tf.trainable_variables()):
      copied_weights[v.name]=self.sess.run(self.importance_weights[v.name])
    np.save(self.FLAGS.summary_dir+self.FLAGS.log_tag+"/omegas", copied_weights)

    for v in tf.trainable_variables():
        weights=self.sess.run(self.importance_weights[v.name])
        weights=weights.flatten()
        # print("{0}: {1} ({2}) min: {3} max: {4}".format(v.name, np.mean(weights), np.var(weights), np.amin(weights), np.amax(weights)))
        print("| {0} | {1} | {2} | {3} | ".format(v.name, 
                                                  np.percentile(weights,1),
                                                  np.percentile(weights,50),
                                                  np.percentile(weights,100)))


    # import pdb; pdb.set_trace()    

  def define_metrics(self, endpoints):
    '''tensors to evaluate the performance. Uses only the endpoints of the 'eval' copy of the network
    '''
    with tf.device(self.device):
      with tf.variable_scope('metrics'):
        self.mse={}
        self.mse_depth={}
        self.accuracy={}
        for mode in ['train', 'val']: #following modes offline training
          # keep running variables for both validation and training data
          self.mse[mode] = tf.metrics.mean_squared_error(self.targets, endpoints['digit'] if self.FLAGS.discrete else endpoints['outputs'], name="mse_"+mode)
          if self.FLAGS.auxiliary_depth:
            self.mse_depth[mode] = tf.metrics.mean_squared_error(self.depth_targets, endpoints['aux_depth_reshaped'], name="mse_depth_"+mode)
          # add an accuracy metric
          if self.FLAGS.discrete: 
            self.accuracy[mode] = tf.metrics.accuracy(self.targets, endpoints['digit'], name='accuracy_'+mode)
    # keep metric variables in a field so they are easily updated
    self.metric_variables=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")

  def reset_metrics(self):
    """ Sets all local variables of metrics back to zero
    """
    # initialize both train and val metric variables
    self.sess.run(tf.variables_initializer(var_list=self.metric_variables))

  def define_loss(self, endpoints):
    '''tensors for calculating the loss are added in LOSS collection
    total_loss includes regularization loss defined in tf.layers
    '''
    with tf.device(self.device):
      if not self.FLAGS.discrete:
        self.loss = tf.losses.mean_squared_error(self.targets, endpoints['outputs'], weights=self.FLAGS.control_weight)
      else:
        # targets should be class indices like [0,1,2, ... action_dim] for classes [-1,0,1]
        one_hot=tf.squeeze(tf.one_hot(self.targets, depth=self.FLAGS.action_quantity, on_value=1., off_value=0., axis=-1),[1])
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=endpoints['outputs'], weights=self.FLAGS.control_weight)
      
      #add regularization loss for lifelonglearning
      self.lll_losses={}
      if self.FLAGS.lifelonglearning:
        for v in tf.trainable_variables():
          # self.lll_losses[v.name.split(':')[0]]=self.FLAGS.lll_weight * tf.reduce_sum(tf.multiply(self.importance_weights[v.name],tf.abs(tf.subtract(v,self.star_variables[v.name]))))
          self.lll_losses[v.name.split(':')[0]]=self.FLAGS.lll_weight * tf.reduce_sum(tf.multiply(self.importance_weights[v.name],tf.square(tf.subtract(v,self.star_variables[v.name]))))
          tf.losses.add_loss(self.lll_losses[v.name.split(':')[0]])

       # tf.losses.add_loss(self.loss)
      if self.FLAGS.auxiliary_depth:
        weights = self.FLAGS.depth_weight*tf.cast(tf.greater(self.depth_targets, 0), tf.float32) # put loss weight on zero where depth is negative or zero.        
        self.depth_loss = tf.losses.huber_loss(self.depth_targets,endpoints['aux_depth_reshaped'],weights=weights)
        tf.losses.add_loss(self.depth_loss)
      
      # self.total_loss = self.loss + self.FLAGS.lll_weight * self.lll_loss
      # self.total_loss = self.FLAGS.lll_weight * self.lll_loss
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
      

      # self.train_op = slim.learning.create_train_op(self.total_loss, 
      #   self.optimizer, 
      #   global_step=self.global_step, 
      #   gradient_multipliers=gradient_multipliers, 
      #   clip_gradient_norm=self.FLAGS.clip_grad)

  def forward(self, inputs, auxdepth=False, targets=[], depth_targets=[]):
    '''run forward pass and return action prediction
    inputs=batch of RGB images
    auxdepth = variable defining wether auxiliary depth should be predicted
    targets = supervised target control
    depth_targets = supervised target depth
    '''
    feed_dict={self.inputs: inputs}  

    tensors = [self.endpoints['eval']['control']] if self.FLAGS.discrete else [self.endpoints['eval']['outputs']]
    
    if auxdepth: # predict auxiliary depth
      tensors.append(self.endpoints['eval']['aux_depth_reshaped'])
    
    if len(targets) != 0: # if target control is available, calculate loss
      tensors.append(self.mse['val']) # 1 is required to get the update operation
      # tensors.append(self.mse['val'][1]) # 1 is required to get the update operation
      feed_dict[self.targets]=targets if not self.FLAGS.discrete else self.continuous_to_discrete(targets)
      # if self.FLAGS.discrete: tensors.append(self.accuracy['val'][1])
      if self.FLAGS.discrete: tensors.append(self.accuracy['val'])
      
    if len(depth_targets) != 0 and self.FLAGS.auxiliary_depth:# if target depth is available, calculate loss
      tensors.append(self.mse_depth['val'])
      # tensors.append(self.mse_depth['val'][1])
      feed_dict[self.depth_targets] = depth_targets

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

  def backward(self, inputs, targets=[], depth_targets=[], sumvar=None):
    '''run backward pass and return losses
    inputs: batch of images
    targets: batch of control labels
    depth_targets: batch of depth labels
    sumvar: summary variables dictionary to which values can be added.
    '''
    tensors = [self.train_op]
    feed_dict = {self.inputs: inputs}
    feed_dict[self.targets]= self.continuous_to_discrete(targets) if self.FLAGS.discrete else targets
    
    # append loss
    tensors.append(self.total_loss)
    tensors.append(self.loss)
    if self.FLAGS.lifelonglearning:
      tensors.extend([self.lll_losses[k] for k in self.lll_losses.keys()])

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
    
    if self.FLAGS.discrete: 
      tensors.append(self.accuracy['train'])

    if self.FLAGS.auxiliary_depth:
      feed_dict[self.depth_targets] = depth_targets
      tensors.append(self.depth_mse['train'])

    results = self.sess.run(tensors, feed_dict=feed_dict)
    
    _ = results.pop(0) # train_op
    
    losses={'total':results.pop(0)}
    losses['ce']=results.pop(0)

    if self.FLAGS.lifelonglearning:
      for k in self.lll_losses.keys(): losses['lll_'+k]=results.pop(0)

        
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
      if self.FLAGS.discrete:
        tensors.append(self.accuracy[mode][0])
      if self.FLAGS.auxiliary_depth:
        tensors.append(self.mse_depth[mode][0])
    output=self.sess.run(tensors)
    for mode in ['train', 'val']:
      results['mse_'+mode]=output.pop(0)
      if self.FLAGS.discrete:
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
    self.add_summary_var('Loss_train_ce')
    
    if self.FLAGS.lifelonglearning:
      for k in self.lll_losses.keys():
        self.add_summary_var('Loss_train_lll_'+k)

    for t in ['train', 'val']:
      for l in ['mse', 'accuracy', 'mse_depth']:
        name='{0}_{1}'.format(l,t)
        self.add_summary_var(name)
    for d in ['current','furthest']:
      for t in ['train', 'test']:
        for w in ['','corridor','sandbox','forest','canyon','esatv1', 'esatv2','osb_yellow_barrel','osb_carton_box','osb_yellow_barrel_blue']:
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
