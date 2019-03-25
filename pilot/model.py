import os, sys

from models import *
# from torchvision.models import *

import tensorflow as tf
# import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim import model_analyzer as ma
# from tensorflow.python.ops import variables as tf_variables

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import time

"""
Build basic NN model

Exit code: 2: failed to initialize checkpoint
"""
class Model(object):
 
  def __init__(self, FLAGS, prefix='model'):
    '''initialize model
    '''
    # INITIALIZE FIELDS
    self.FLAGS=FLAGS
    self.action_dim = FLAGS.action_dim
    self.prefix = prefix
    self.device = torch.device("cuda:0" if torch.cuda.is_available() and 'gpu' in FLAGS.device else "cpu")
    self.epoch=0
    self.output_size=int(self.action_dim if not self.FLAGS.discrete else self.action_dim * self.FLAGS.action_quantity)
    
    if self.FLAGS.discrete:
      self.define_discrete_bins(FLAGS.action_bound, FLAGS.action_quantity)
      # target: continuous value (-0.3) --> bin (2) --> label (0 0 1 0 0 0 0 0 0 0)

    # DEFINE NETWORK
    # define a network for training and for evaluation
    try:
      self.net = eval(self.FLAGS.network).Net(self.output_size, self.FLAGS.pretrained)
    except Exception as e:
      print(e)
      print("[model] Failed to load model {0}.".format(self.FLAGS.network))
      sys.exit(2)
    if self.FLAGS.pretrained:
      print("[model] loaded imagenet pretrained weights.")
    self.input_size=self.net.default_image_size
    # load on GPU
    # self.device = torch.device( "cpu")
    self.net.to(self.device)
    # to get a complexity idea, count number of weights in the network
    count=0
    for p in self.net.parameters():
      w=1
      for d in p.size(): w*=d
      count+=w
    print("[model] Total number of trainable parameters: {}".format(count))

    # DEFINE LOSS
    self.criterion = eval("nn.{0}Loss(reduction='none')".format(self.FLAGS.loss))
    # self.collision_criterion=nn.MSELoss()
    self.softmax = torch.nn.Softmax(dim=1) #assumes [Batch x Outputs]
    # DEFINE OPTIMIZER
    self.optimizer = eval("optim.{0}(self.net.parameters(), lr={1})".format(self.FLAGS.optimizer, self.FLAGS.learning_rate))

    # DEFINE MAS REGULARIZER
    # if self.FLAGS.continue_training and self.FLAGS.lifelonglearning:
    #   self.define_star_variables(self.endpoints['train'])

    # DEFINE SUMMARIES WITH TENSORBOARD
    if self.FLAGS.tensorboard:
      self.graph = tf.Graph()
      self.summary_vars = {}
      self.summary_ops = {}
      self.writer = None
      config=tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      self.sess=tf.Session(graph=self.graph, config=config)
      

    if not self.FLAGS.pretrained or self.FLAGS.continue_training: 
      self.initialize_network()


  def initialize_network(self):
    """Initialize all parameters of the network conform the FLAGS configuration
    """
    if self.FLAGS.checkpoint_path == '':
      for p in self.net.parameters():
        try:
          # nn.init.constant_(p, 0.001)
          nn.init.xavier_uniform_(p)
        except:
          nn.init.normal_(p, 0, 0.1)
      print("[model]: initialized model from scratch")
    else:
      # load model checkpoint in its whole
      checkpoint=torch.load(self.FLAGS.checkpoint_path+'/my-model')
      try:
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=self.FLAGS.continue_training)
      except Exception as e:
        print("[model]: FAILED to load model {1} from {0} into {2}, {3}.".format(self.FLAGS.checkpoint_path, 
                                                                          checkpoint['network'],
                                                                          self.FLAGS.network,
                                                                          e.message))
        if self.FLAGS.continue_training: print("\t put continue_training FALSE to avoid strict matching.")
        sys.exit(2)
      else:
        self.epoch = checkpoint['epoch']
        print("[model]: loaded model from {0} at epoch: {1}".format(self.FLAGS.checkpoint_path, self.epoch))
      if  not 'scratch' in self.FLAGS.checkpoint_path and checkpoint['optimizer'] == self.FLAGS.optimizer:
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("[model]: loaded optimizer parameters from {0}".format(self.FLAGS.checkpoint_path))
        
  def save(self, logfolder, save_optimizer=True):
    '''save a checkpoint'''
    if save_optimizer:
      torch.save({
        'epoch': self.epoch,
        'network': self.FLAGS.network,
        'optimizer': self.FLAGS.optimizer,
        'optimizer_state_dict': self.optimizer.state_dict(),
        'model_state_dict': self.net.state_dict()}, logfolder+'/my-model')
    else:
      torch.save({
        'epoch': self.epoch,
        'network': self.FLAGS.network,
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
      # TODO SPEED UP BY CREATING VECTOR AND FILLING IT, or multithreading???
      discrete_values=[]
      for c in continuous_value.flatten():
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
 
  def discretize(self, continuous):
    """With the use of previous methods, discretize numpy array of shape [nBatch, 1]
        into target values one-hot encoded Torch Tensors of shape [nBatch, action_quantity] in case of a normal loss
        or into target bins in case of a cross entropy loss.
    """
    bins = self.continuous_to_bins(continuous)
    if self.FLAGS.loss == 'CrossEntropy': # return bins in (N) shape
      bins = torch.from_numpy(bins).squeeze().type(torch.LongTensor)
      if len(bins.shape) == 0: bins.unsqueeze_(-1)
      return bins
      # return torch.from_numpy(bins).type(torch.LongTensor) 
    else: # normal loss like MSE : return one-hot vecotr
      return torch.zeros(len(bins),self.FLAGS.action_quantity).scatter_(1,torch.from_numpy(bins),1.).type(torch.FloatTensor)
      
  def predict(self, inputs, targets=[], lstm_info=()):
    '''run forward pass and return prediction with loss if target is given
    inputs=batch of RGB images (Batch x Channel x Height x Width)
    targets = supervised target control (Batch x Action dim)
    '''
    # assert (len(inputs.shape) == 4 and list(inputs.shape[1:]) == self.input_size), "inputs shape: {0} instead of {1}".format(inputs.shape, self.input_size)
    if not isinstance(inputs, tuple):
      inputs=torch.from_numpy(inputs).type(torch.FloatTensor).to(self.device)
    
    predictions = self.net.forward(inputs, train=False)
    hidden_states=()
    if isinstance(predictions,tuple):
      h_t, c_t=predictions[1]
      predictions=predictions[0].view(inputs[0].size()[0]*inputs[0].size()[1],self.FLAGS.action_quantity)
      hidden_states=(h_t.cpu().detach().numpy(),
                    c_t.cpu().detach().numpy())
    losses={}
    if len(targets) != 0: 
      # assert (len(targets.shape) == 2 and targets.shape[0] ==inputs.shape[0]), "targets shape: {0} instead of {1}".format(targets.shape, inputs.shape[0])
      targets = self.discretize(targets) if self.FLAGS.discrete else torch.from_numpy(targets).type(torch.FloatTensor)
      if len(targets.shape) == 2:
        # Assumption: pytorch view (X,Y,3) --> (X*Y,3) arranges in the same way as pytorch (X,Y).flatten
        targets=targets.flatten()
      # import pdb; pdb.set_trace()
      losses['imitation_learning'] = np.mean(self.criterion(predictions, targets.to(self.device)).cpu().detach().numpy())
      # get accuracy and append to loss: don't change this line to above, as accuracy is calculated on cpu() in numpy floats
      if self.FLAGS.discrete: losses['accuracy'] = (torch.argmax(predictions.data,1).cpu()==targets).sum().item()/float(len(targets))

    predictions=predictions.cpu().detach().numpy()
    

    if self.FLAGS.discrete: predictions = self.bins_to_continuous(np.argmax(predictions, 1))

    return predictions, losses, hidden_states

  def train(self, inputs, targets, actions=[], collisions=[], lstm_info=()):
    '''take backward pass from loss and apply gradient step
    inputs: batch of images
    targets: batch of control labels
    '''
    # Ensure correct shapes at the input
    # assert (len(inputs.shape) == 4 and list(inputs.shape[1:]) == self.input_size), "inputs shape: {0} instead of {1}".format(inputs.shape, self.input_size)
    # assert (len(targets.shape) == 2 and targets.shape[0] ==inputs.shape[0]), "targets shape: {0} instead of {1}".format(targets.shape, inputs.shape[0])

    # Ensure gradient buffers are zero
    self.optimizer.zero_grad()

    losses={'total':0}
    if not isinstance(inputs, tuple):
      inputs=torch.from_numpy(inputs).type(torch.FloatTensor).to(self.device)
    predictions = self.net.forward(inputs, train=True)
    hidden_states=()
    if isinstance(predictions,tuple):
      h_t, c_t=predictions[1]
      predictions=predictions[0].view(inputs[0].size()[0]*inputs[0].size()[1],self.FLAGS.action_quantity)
      hidden_states=(h_t.cpu().detach().numpy(),
                    c_t.cpu().detach().numpy())

    # import pdb; pdb.set_trace()
    targets = self.discretize(targets) if self.FLAGS.discrete else torch.from_numpy(targets).type(torch.FloatTensor)
    if len(targets.shape) == 2:
      # Assumption: pytorch view (X,Y,3) --> (X*Y,3) arranges in the same way as pytorch (X,Y).flatten
      targets=targets.flatten()

    # cross entropy is sometimes numerically unstable...
    # import pdb; pdb.set_trace()
    losses['imitation_learning']=self.criterion(predictions, targets.to(self.device))
    
    losses['total']+=self.FLAGS.il_weight*losses['imitation_learning']
    
    if len(actions) == len(collisions) == len(inputs) and self.FLAGS.il_weight != 1:
      # 1. from logits to probabilities with softmax for each output in batch
      probabilities=self.softmax(predictions)
      # 2. probabilities of corresponding actions
      actions=self.continuous_to_bins(actions)
      action_probabilities=torch.stack([probabilities[i, actions[i]] for i in range(len(probabilities))])
      # 3. add MSE Loss for specific output of corresponding action with 1-collision probability
      no_collisions=1-collisions 
      # manual cross-entropy implementation sum(-target_probs*log(predicted_probs))
      log_y=torch.log((action_probabilities+10**-8).type(torch.FloatTensor).to(self.device))
      p=torch.from_numpy(no_collisions).type(torch.FloatTensor).to(self.device)
      log_y_1=torch.log((1-action_probabilities+10**-8).type(torch.FloatTensor).to(self.device))
      
      # losses['Loss_train_reinforcement_learning']=torch.mean(-p*log_y-(1-p)*log_y_1)
      losses['reinforcement_learning']=-p*log_y-(1-p)*log_y_1
      
      # 4. add loss to Loss_train_total loss with corresponding weight.
      losses['total']+=(1-self.FLAGS.il_weight)*losses['reinforcement_learning'] 
    
    # stime=time.time()
    mean_loss=torch.mean(losses['total'])
    mean_loss.backward() # fill gradient buffers with the gradient according to this loss
    # print("backward time: ", time.time()-stime)
    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.FLAGS.clip)

    self.optimizer.step() # apply what is in the gradient buffers to the parameters
    self.epoch+=1

    predictions_list=predictions.cpu().detach().numpy()
    if self.FLAGS.discrete: predictions_list = self.bins_to_continuous(np.argmax(predictions_list, 1))

    # ensure losses are of type numpy
    for k in losses: losses[k]=losses[k].cpu().detach().numpy()
    
    # get accuracy and append to loss: don't change this line to above, as accuracy is calculated on cpu() in numpy floats
    if self.FLAGS.discrete:
      if not self.FLAGS.loss == 'CrossEntropy': 
        targets=np.argmax(targets,1) # For MSE loss is targets one hot encoded
      losses['accuracy'] = (torch.argmax(predictions.data,1).cpu()==targets).sum().item()/float(len(targets))
    
    return self.epoch, predictions_list, losses, hidden_states
     
  def add_summary_var(self, name):
    '''given the name of the new variable
    add an actual variable to  summary_vars
    and add an operation to update the variable to summary_ops
    '''
    with self.graph.as_default():
      with self.graph.device('/cpu:0'):
        var_name = tf.Variable(0., trainable=False, name=name)
        self.summary_vars[name]=var_name
        self.summary_ops[name] = tf.summary.scalar(name, var_name)
  
  def summarize(self, sumvars):
    '''write summary sumvars by defining variables to
    summary_vars and calling summary_ops in a merge'''
    try:
      # Ensure that for each variable key a summary var is defined in the summary_vars and summary_ops fields
      for k in sumvars.keys(): 
        if k not in self.summary_vars.keys(): #if k does not have variable and operation yet, add it
          self.add_summary_var(k)
      feed_dict={self.summary_vars[key]:sumvars[key] for key in sumvars.keys()}
      with self.graph.as_default():
        with self.graph.device('/cpu:0'):
          sum_op = tf.summary.merge([self.summary_ops[key] for key in sumvars.keys()])
      
        summary_str = self.sess.run(sum_op, feed_dict=feed_dict)
        if self.writer == None:  self.writer = tf.summary.FileWriter(self.FLAGS.summary_dir+self.FLAGS.log_tag, self.graph)
        self.writer.add_summary(summary_str, self.epoch)
        self.writer.flush()
    except Exception as e:
      if self.FLAGS.tensorboard: print("[model] failed to summarize: {}".format(e.message))
      pass

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

  