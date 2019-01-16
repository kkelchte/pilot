""" 
Data structure for implementing experience replay
Author: Patrick Emami
"""
import time
from collections import deque
import random
import numpy as np
# import tensorflow as tf

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

    def get_all_data(self, max_batch_size=-1, data_buffer=None):
      # fill in a batch of size batch_size
      # return an array of inputs, targets and auxiliary information
      if data_buffer == None: data_buffer=self.buffer      
      input_batch = np.array([_['state'] for _ in data_buffer])
      target_batch = np.array([_['trgt'] for _ in data_buffer])
      action_batch = np.array([_['action'] for _ in data_buffer])
      collision_batch = np.array([_['collision'] for _ in data_buffer])
      
      if max_batch_size != -1 and input_batch.shape[0] > max_batch_size: 
        input_batch=input_batch[:max_batch_size]
        target_batch=target_batch[:max_batch_size]
        action_batch=action_batch[:max_batch_size]
        collision_batch=collision_batch[:max_batch_size]

      return input_batch, target_batch, action_batch, collision_batch

    def get_all_data_shuffled(self, max_batch_size=-1, horizon=0):
      """ fill in a batch of size batch_size
      don't return the last horizon of labels as they might still get changed if a bump occurs at the next step.
      return an array of inputs, targets and auxiliary information
      """
      shuffled_indices =np.arange(len(self.buffer)-horizon)
      np.random.shuffle(shuffled_indices)
      shuffled_buffer = [self.buffer[i] for i in shuffled_indices]
      return self.get_all_data(max_batch_size, shuffled_buffer)


    def sample_batch(self, batch_size):
      # fill in a batch of size batch_size
      # return an array of inputs, targets and auxiliary information
      assert batch_size <= self.count, IOError('batchsize ',batch_size,' is bigger than buffer size: ',self.count)
      batch=random.sample(self.buffer, batch_size)

      input_batch = np.array([_['state'] for _ in batch])
      target_batch = np.array([_['trgt'] for _ in batch])
      action_batch = np.array([_['action'] for _ in batch])
      collision_batch = np.array([_['collision'] for _ in batch])
      
      return input_batch, target_batch, action_batch, collision_batch

    def annotate_collision(self, horizon):
      """Annotate the experiences over the last horizon with a 1 for collision.
      """
      last_experiences=[self.buffer.pop() for i in range(horizon)]
      for e in last_experiences: e['collision']=1
      self.buffer.extend(reversed(last_experiences))

    def clear(self):
      self.buffer.clear()
      # self.buffer = []
      self.count = 0

    def get_details(self,keys="all"):
      """Return dictionary details on current state of the replay buffer:
      num_experiences
      relative_left
      relative_right
      relative_straight
      relative_collision
      extension:
      feature representation from scan
      """
      if keys=='all':
        count_total, count_right, count_left, count_straight, count_collision, count_imitation_error = 0,0,0,0,0,0
      for e in self.buffer:
        count_total+=1
        if 'trgt' in e.keys():
          if np.abs(e['trgt']) < 0.5:
            count_straight+=1
          elif np.sign(e['trgt']) > 0:
            count_left+=1
          else:
            count_right+=1
          if e['trgt'] != e['action']: count_imitation_error+=1
        if 'collision' in e.keys():
          if e['collision'] == 1: count_collision += 1

      return {'buffer_count_total': count_total,
              'buffer_relative_right': float(count_right)/count_total,
              'buffer_relative_left': float(count_left)/count_total,
              'buffer_relative_straight': float(count_straight)/count_total,
              'buffer_relative_collision': float(count_collision)/count_total,
              'buffer_relative_imitation_error': float(count_imitation_error)/count_total}

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
  mybuffer=ReplayBuffer(FLAGS)
  for i in range(30):
    mybuffer.add({'state':np.zeros((1,1))+i,
                'action':np.random.choice([-1,0,1],p=[0.1,0.1,0.8]),
                'trgt':np.random.choice([-1,0,1],p=[0.1,0.1,0.8]),
                'collision':1})
  for i in range(10):
    mybuffer.add({'state':np.zeros((1,1))+100+i,
                'action':np.random.choice([-1,0,1],p=[0.1,0.1,0.8]),
                'trgt':np.random.choice([-1,0,1],p=[0.1,0.1,0.8]),
                'collision':0})
  
  print("\n content of the mybuffer: \n")
  # mybuffer.to_string()

  print mybuffer.get_details()

  # prop_zero=[]
  # for i in range(10):
  #   stime=time.time()
  #   state, action, trgt = buffer.sample_batch()
  #   # print("states: {0}".format(state))
  #   prop_zero.append(state)
    
  # print("trgt 0: {0}, trgt 1: {1}".format(len(trgt[trgt==0]),len(trgt[trgt==1])))
  # prop_zero.append(float(len(trgt[trgt==0]))/FLAGS.batch_size)
  
  # prop_zero.append(float(len(action[action==0]))/FLAGS.batch_size)
  # print("time to sample: {0:f}s, proportion of labels -1: {1}, proportion of labels 0: {2}, proportion of labels 1: {3}".format(time.time()-stime, 
                                                                                                    # float(len(action[action==-1]))/FLAGS.batch_size, 
                                                                                                    # float(len(action[action==0]))/FLAGS.batch_size, 
                                                                                                    # float(len(action[action==1]))/FLAGS.batch_size))
# print("avg prop 0: {0} var prop 0: {1}".format(np.mean(prop_zero), np.var(prop_zero)))