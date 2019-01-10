""" 
Data structure for implementing experience replay
Author: Patrick Emami
"""
import time
from collections import deque
import random
import numpy as np
import tensorflow as tf

import argparse


class ReplayBuffer(object):

    def __init__(self, buffer_size=-1, random_seed=123):
      """
      The right side of the deque contains the most recent experiences 
      """
      self.buffer_size = buffer_size if buffer_size != -1 else 1000000
      self.count = 0
      self.buffer = deque()
      self.num_steps = 1 #num
      self.probs = None

    def add(self, experience):
      # add experience dictionary to buffer
      if self.count < self.buffer_size: 
        self.count += 1
      else:
        # Get rid of oldest buffer/run once it is smaller than number of steps
        # if len(self.buffer[0])<=self.num_steps:
        #   self.count-=len(self.buffer.pop(0))
        #   self.count+=1
        # else:
        self.buffer.popleft()

      self.buffer.append(experience)

    def remove(self):
      self.buffer.popleft()
      self.count-=1

    def size(self):      
      return self.count
    
    def softmax(self, x):
      e_x = np.exp(x-np.max(x))
      return e_x/e_x.sum()

    def get_all_data(self, max_batch_size):
      # fill in a batch of size batch_size
      # return an array of inputs, targets and auxiliary information
      
      input_batch = np.array([_['state'] for _ in self.buffer])
      target_batch = np.array([_['trgt'] for _ in self.buffer])
      aux_batch = {}
      # for k in batch[0].keys():
      #   aux_batch[k]=np.array([_[k] for _ in batch])

      if input_batch.shape[0] > max_batch_size: input_batch=input_batch[:max_batch_size]
      if target_batch.shape[0] > max_batch_size: target_batch=target_batch[:max_batch_size]

      return input_batch, target_batch, aux_batch

    def get_all_data_shuffled(self, max_batch_size=-1):
      # fill in a batch of size batch_size
      # return an array of inputs, targets and auxiliary information
      
      shuffled_buffer = self.buffer
      input_batch = np.array([_['state'] for _ in shuffled_buffer])
      target_batch = np.array([_['trgt'] for _ in shuffled_buffer])
      aux_batch = {}
      
      if max_batch_size != -1 and input_batch.shape[0] > max_batch_size: input_batch=input_batch[:max_batch_size]
      if max_batch_size != -1 and target_batch.shape[0] > max_batch_size: target_batch=target_batch[:max_batch_size]

      return input_batch, target_batch, aux_batch

    def sample_batch(self, batch_size):
      # fill in a batch of size batch_size
      # return an array of inputs, targets and auxiliary information
      assert batch_size <= self.count, IOError('batchsize ',batch_size,' is bigger than buffer size: ',self.count)
      batch=random.sample(self.buffer, batch_size)

      input_batch = np.array([_['state'] for _ in batch])
      target_batch = np.array([_['trgt'] for _ in batch])
      aux_batch = {}
      # for k in batch[0].keys():
      #   aux_batch[k]=np.array([_[k] for _ in batch])

      return input_batch, target_batch, aux_batch

    def clear(self):
        self.buffer.clear()
        # self.buffer = []
        self.count = 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test replay buffer:')

  parser.add_argument("--replay_priority", default='no', type=str, help="Define which type of weights should be used when sampling from replay buffer: no, uniform_action, uniform_collision, td_error, recency, min_variance")
  parser.add_argument("--network",default='mobile',type=str, help="Define the type of network: depth_q_net, coll_q_net.")
  parser.add_argument("--buffer_size", default=100, type=int, help="Define the number of experiences saved in the buffer.")
  parser.add_argument("--batch_size",default=10,type=int,help="Define the size of minibatches.")
  parser.add_argument("--action_bound", default=1.0, type=float, help= "Define between what bounds the actions can go. Default: [-1:1].")
  
  FLAGS=parser.parse_args()  

 
  print "FLAGS.replay_priority: ",FLAGS.replay_priority

  # sample episode of data acquisition
  buffer=ReplayBuffer(FLAGS)
  for i in range(30):
    buffer.add({'state':np.zeros((1,1))+i,
                'action':np.random.choice([-1,0,1],p=[0.1,0.1,0.8]),
                'trgt':0,
                'error':1})
  for i in range(10):
    buffer.add({'state':np.zeros((1,1))+100+i,
                'action':np.random.choice([-1,0,1],p=[0.1,0.1,0.8]),
                'trgt':0,
                'error':10})
  
  print("\n content of the buffer: \n")
  buffer.to_string()


  prop_zero=[]
  for i in range(10):
    stime=time.time()
    state, action, trgt = buffer.sample_batch()
    # print("states: {0}".format(state))
    prop_zero.append(state)
    
    # print("trgt 0: {0}, trgt 1: {1}".format(len(trgt[trgt==0]),len(trgt[trgt==1])))
    # prop_zero.append(float(len(trgt[trgt==0]))/FLAGS.batch_size)
    
    # prop_zero.append(float(len(action[action==0]))/FLAGS.batch_size)
    # print("time to sample: {0:f}s, proportion of labels -1: {1}, proportion of labels 0: {2}, proportion of labels 1: {3}".format(time.time()-stime, 
                                                                                                    # float(len(action[action==-1]))/FLAGS.batch_size, 
                                                                                                    # float(len(action[action==0]))/FLAGS.batch_size, 
                                                                                                    # float(len(action[action==1]))/FLAGS.batch_size))
# print("avg prop 0: {0} var prop 0: {1}".format(np.mean(prop_zero), np.var(prop_zero)))