
import os

from models import *

# import tensorflow as tf
# import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim import model_analyzer as ma
# from tensorflow.python.ops import variables as tf_variables

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

"""
Build basic NN model
"""
class Model(object):
 
  def __init__(self, FLAGS, prefix='model'):
    '''initialize model
    '''
    # INITIALIZE FIELDS
    self.action_dim = FLAGS.action_dim
    self.prefix = prefix
    self.device = FLAGS.device
    self.FLAGS=FLAGS
    self.output_size=int(self.action_dim if not self.FLAGS.discrete else self.action_dim * self.FLAGS.action_quantity)
    self.epoch=0


    if self.FLAGS.discrete:
      self.define_discrete_bins(FLAGS.action_bound, FLAGS.action_quantity)
      # target: continuous value (-0.3) --> bin (2) --> label (0 0 1 0 0 0 0 0 0 0)

    # DEFINE NETWORK
    # define a network for training and for evaluation
    self.net = eval(self.FLAGS.network).Net(self.output_size)
    
    # load on GPU
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # self.device = torch.device( "cpu")
    self.net.to(self.device)

    # to get a complexity idea, count number of weights in the network
    count=0
    for p in self.net.parameters():
      w=1
      for d in p.size(): w*=d
      count+=w
    print("Total number of trainable parameters: {}".format(count))

    # DEFINE LOSS
    self.criterion = eval("nn.{0}Loss()".format(self.FLAGS.loss))
    
    # DEFINE OPTIMIZER
    self.optimizer = eval("optim.{0}(self.net.parameters(), lr={1})".format(self.FLAGS.optimizer, self.FLAGS.learning_rate))

    # DEFINE MAS REGULARIZER
    # if self.FLAGS.continue_training and self.FLAGS.lifelonglearning:
    #   self.define_star_variables(self.endpoints['train'])

    # DEFINE SUMMARIES
    # self.build_summaries()

    # import pdb; pdb.set_trace()


  def initialize_network(self):
    """Initialize all parameters of the network conform the FLAGS configuration
    """
    if self.FLAGS.scratch:
      # scratch
      for p in self.net.parameters():
        try:
          # nn.init.constant_(p, 0.001)
          nn.init.xavier_uniform_(p)
        except:
          nn.init.normal_(p, 0, 0.1)
    elif self.FLAGS.continue_training:
      # load model checkpoint in its whole
      #try:
      # torch.load(PATH, strict=True)
      pass
    else: # Try to load as much as possible
      # torch.load(PATH, strict=False)
      pass

  def save(self, logfolder):
    '''save a checkpoint'''
    torch.save({
      'epoch': self.epoch,
      'optimizer_state_dict': self.optimizer.state_dict(),
      'model_state_dict': self.net.state_dict()}, logfolder+'/my-model')
  
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

  def continuous_to_bins(self, continuous_value):
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
    else:
      discrete=0
      for b in self.boundaries:
        if b > continuous_value:
          return discrete
        else:
          discrete+=1
      return discrete

  def bins_to_continuous(self, discrete_value, name=None):
    """
    Changes a discrete value to the continuous value (center of the corresponding bin)
    [0,1,2] --> [-1,0,1]
    """
    if hasattr(discrete_value, '__contains__'):
      return [self.control_values[int(v)] for v in discrete_value]
    else:
      return self.control_values[int(discrete_value)]
 
  def predict(self, inputs, targets=[]):
    '''run forward pass and return prediction with loss if target is given
    inputs=batch of RGB images (Batch x Channel x Height x Width)
    targets = supervised target control (Batch x Action dim)
    '''
    assert (len(inputs.shape) == 4 and list(inputs.shape[1:]) == self.net.default_image_size), "inputs shape: {0} instead of {1}".format(inputs.shape, self.net.default_image_size)
    inputs=torch.from_numpy(inputs).type(torch.FloatTensor).to(self.device)
    predictions = self.net.forward(inputs)

    losses={}
    if len(targets) != 0: 
      assert (len(targets.shape) == 2 and targets.shape[0] ==inputs.shape[0]), "targets shape: {0} instead of {1}".format(targets.shape, inputs.shape[0])
      if self.FLAGS.discrete:
        # TODO: add discretization code ...
        targets = self.continuous_to_bins(targets)
        targets = torch.zeros(len(targets),self.FLAGS.action_quantity).scatter_(1,torch.from_numpy(targets).unsqueeze(1),1.)
      targets=torch.from_numpy(targets).type(torch.FloatTensor).to(self.device)
      losses[self.FLAGS.loss] = self.criterion(predictions, targets).cpu().detach().numpy()
  
    if self.FLAGS.discrete:
      predictions = bins_to_continuous(torch.max(predictions, 1))

    return predictions.cpu().detach().numpy(), losses
    

  def train(self, inputs, targets):
    '''take backward pass from loss and apply gradient step
    inputs: batch of images
    targets: batch of control labels
    '''
    # Ensure correct shapes at the input
    assert (len(inputs.shape) == 4 and list(inputs.shape[1:]) == self.net.default_image_size), "inputs shape: {0} instead of {1}".format(inputs.shape, self.net.default_image_size)
    assert (len(targets.shape) == 2 and targets.shape[0] ==inputs.shape[0]), "targets shape: {0} instead of {1}".format(targets.shape, inputs.shape[0])
    # Ensure gradient buffers are zero
    self.optimizer.zero_grad()

    # TODO: test if model parameter gradient buffers are empty as well.
    losses={'total':0}
    inputs=torch.from_numpy(inputs).type(torch.FloatTensor).to(self.device)
    predictions = self.net.forward(inputs)
    targets=torch.from_numpy(targets).type(torch.FloatTensor).to(self.device)
    losses[self.FLAGS.loss]=self.criterion(predictions, targets)
    losses['total']+=losses[self.FLAGS.loss] 
    #TODO: add regularizers...
    losses['total'].backward() # fill gradient buffers with the gradient according to this loss
    self.optimizer.step() # apply what is in the gradient buffers to the parameters
    self.epoch+=1

    # ensure losses are of type numpy
    for k in losses: losses[k]=losses[k].cpu().detach().numpy()
    return self.epoch, predictions.cpu().detach().numpy(), losses
    
  # CONTINUAL LEARNING
  def define_star_variables(self, endpoints):
    '''Define star variables for previous domains used in combination with the importance weights for a lifelong regularization term in the loss.
    In case copy is True the last domain creates new variables and copy the current weights of the model in these variables to introduce a new set of start variables corresponding to the last domain.
    Star variables is a dictionary keeping a list of variables for each domain.
    '''
    # self.star_variables={}
    # for v in tf.trainable_variables():
    #   # self.star_variables[v.name]=self.sess.run(v)
    #   # self.star_variables[v.name]= self.var_list[v].eval()
    #   self.star_variables[v.name]=tf.get_variable('Star/'+v.name.split(':')[0],
    #                                                   initializer=np.zeros(v.get_shape().as_list()).astype(np.float32),
    #                                                   dtype=tf.float32,
    #                                                   trainable=False)

  def define_importance_weights(self, endpoints):
    '''Define an important weight ~ omegas for each trainable variable
    domain corresponds to the dataset
    '''
    # self.importance_weights={}
    # for v in tf.trainable_variables():
    #   self.importance_weights[v.name]=tf.get_variable('Omega/'+v.name.split(':')[0],
    #                                                   initializer=np.zeros(v.get_shape().as_list()).astype(np.float32),
    #                                                   dtype=tf.float32,
    #                                                   trainable=False)
    
  def update_importance_weights(self, inputs):
    '''Take one episode of data and keep track of gradients.
    Calculate a importance value between 0 and 1 for each weight.
    '''
    # # self.sess.run([tf.assign(self.importance_weights[v.name], np.ones((v.shape.as_list()))) for i,v in enumerate(tf.trainable_variables())])
    # # return

    # # Add model variables for importance_weights for current dataset
    # gradients=[]
    # # batchsize = 200
    # batchsize = 1
    # #number of batches of 200
    # N=int(inputs.shape[0]/batchsize)
    # print('[model.py]: update importance weights on {0} batches of batchsize.'.format(N))
    # for i in range(N):
    #   # print i
    #   batch_inputs=inputs[batchsize*i:batchsize*(i+1)]
    #   # get gradients for a batch of images

    #   results = self.sess.run(tf.gradients(tf.square(tf.norm(self.endpoints['train']['outputs'])), tf.trainable_variables()), feed_dict={self.inputs: batch_inputs})

    #   # print results
    #   # import pdb; pdb.set_trace()
    #   # sum up over the absolute values
    #   try:
    #     gradients = [gradients[j]+np.abs(results[j]) for j in range(len(results))]
    #   except:
    #     gradients = [np.abs(results[j]) for j in range(len(results))]
    # #gradients are summed up by tf.gradients so average over the batchsize samples
    # # divide to get the mean absolute value
    # gradients = [g/(batchsize*N) for g in gradients]

    # new_weights=[gradients[i] + self.sess.run(self.importance_weights[v.name]) for i,v in enumerate(tf.trainable_variables())]

    # # for i,v in enumerate(tf.trainable_variables()):
    # #   print("{0} {1} {2}".format(v.name,self.importance_weights[v.name].shape, gradients[i].shape))
    # # import pdb; pdb.set_trace()    

    # # assign these values to the importance weight variables
    # self.sess.run([tf.assign(self.importance_weights[v.name], new_weights[i]) for i,v in enumerate(tf.trainable_variables())])
    
    # # with open(self.FLAGS.summary_dir+self.FLAGS.log_tag+"/omegas",'w') as f:
    # #   for i,v in enumerate(tf.trainable_variables()):
    # #     f.write("{0}: {1}\n".format(v.name, new_weights[i]))

    # with open(self.FLAGS.summary_dir+self.FLAGS.log_tag+"/omegas_w",'w') as f:
    #   for i,v in enumerate(tf.trainable_variables()):
    #     f.write("{0}: {1}\n".format(v.name, self.sess.run(self.importance_weights[v.name])))

    # # copy importance weights in dictionary to save as numpy file
    # copied_weights={}
    # for i,v in enumerate(tf.trainable_variables()):
    #   copied_weights[v.name]=self.sess.run(self.importance_weights[v.name])
    # np.save(self.FLAGS.summary_dir+self.FLAGS.log_tag+"/omegas", copied_weights)

    # for v in tf.trainable_variables():
    #     weights=self.sess.run(self.importance_weights[v.name])
    #     weights=weights.flatten()
    #     # print("{0}: {1} ({2}) min: {3} max: {4}".format(v.name, np.mean(weights), np.var(weights), np.amin(weights), np.amax(weights)))
    #     print("| {0} | {1} | {2} | {3} | ".format(v.name, 
    #                                               np.percentile(weights,1),
    #                                               np.percentile(weights,50),
    #                                               np.percentile(weights,100)))


    # import pdb; pdb.set_trace()    

  