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

# FLAGS = tf.app.flags.FLAGS


class ReplayBuffer(object):

    def __init__(self, FLAGS, buffer_size, random_seed=123):
      """
      The right side of the deque contains the most recent experiences 
      """
      self.FLAGS=FLAGS
      self.buffer_size = buffer_size
      self.count = 0
      self.buffer = deque()
      self.num_steps = 1 #num

      
    def add(self, experience):
      if self.count < self.buffer_size: 
        self.count += 1
      else:
        self.buffer.popleft()
      self.buffer.append(experience)
    
    def size(self):
      return self.count
    
    def sample_batch(self, batch_size):
      assert batch_size < self.count, IOError('batchsize ',batch_size,' is bigger than buffer size: ',self.count)
      
      if self.FLAGS.normalized_replay:
        if self.FLAGS.network == "coll_q_net" :
          probs=[]
          N={0:0, 1:0}
          for e in self.buffer: N[e['trgt']]+=1
          for e in self.buffer: probs.append(1/(2.0*N[e['trgt']]))
          # print("Current number of trgt 0: {0} and 1: {1}.".format(N[0], N[1]))
          # print("Probs: {}".format(probs))

        if self.FLAGS.network == "depth_q_net":
          probs=[]
          N={-1:0, 0:0, 1:0}
          for e in self.buffer:
            if np.abs(e['action']) > 0.3: N[np.sign(e['action'])]+=1
            else: N[0]+=1
          for e in self.buffer: 
            if np.abs(e['action']) > 0.3: probs.append(1/(3.0*N[np.sign(e['action'])]))
            else: probs.append(1/(3.0*N[0]))
          # print("Current number of action -1: {0}, 0: {1} and 1: {2}".format(N[-1], N[0], N[1]))
          # print("Probs: {}".format(probs))

        # ensure that probs sum to one by adjusting the last
        if sum(probs)!=1: probs[-1]=1-sum(probs[:-1])

      batch=np.random.choice(self.buffer, batch_size, p=probs if self.FLAGS.normalized_replay else None)      
      
      # batch=random.sample(self.buffer, batch_size)      
      state_batch = np.array([_['state'] for _ in batch])
      action_batch = np.array([_['action'] for _ in batch])
      trgt_batch = np.array([_['trgt'] for _ in batch])
      
      return state_batch, action_batch, trgt_batch

    def label_collision(self):
      #label the last n experiences with target 1 
      # as collision appeared in the next 10 steps
      n=10
      # from t_end till t_end-n
      last_experiences=[self.buffer.pop() for i in range(n)]
      for e in last_experiences: e['trgt']=1
      self.buffer.extend(reversed(last_experiences))

    def clear(self):
        self.buffer.clear()
        self.count = 0


    def to_string(self):
      for e in self.buffer: print("action: {0}, target: {1}, state: {2}".format(e['action'],e['trgt'],e['state'][0]))
      # print self.buffer


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test replay buffer for coll_q_net:')

  parser.add_argument("--network",default='coll_q_net',type=str, help="Define the type of network: depth_q_net, coll_q_net.")
  parser.add_argument("--normalized_replay", action='store_false', help="Make labels / actions equally likely for the coll / depth q net.")


  FLAGS=parser.parse_args()  
  # FLAGS.normalized_replay=False
  print FLAGS.normalized_replay

  FLAGS.network='coll_q_net'
  # FLAGS.network='depth_q_net'
  # sample episode
  buffer=ReplayBuffer(FLAGS, 100)
  for i in range(30):
    buffer.add({'state':np.zeros((3,3))+i,
                'action':np.random.choice([-1,0,1],p=[0.1,0.8,0.1]),
                'trgt':0})
  buffer.label_collision()
  
  N={-1:0, 0:0, 1:0}
  for e in buffer.buffer:
    if np.abs(e['action']) > 0.3: N[np.sign(e['action'])]+=1
    else: N[0]+=1
  print("Current number of action -1: {0}, 0: {1} and 1: {2}".format(N[-1], N[0], N[1]))

  N={0:0, 1:0}
  for e in buffer.buffer:
    if e['trgt']==0: N[0]+=1
    else: N[1]+=1
  print("Current number of target 0: {0}, 1: {1}".format(N[0], N[1]))
    

  print("\n content of the buffer: \n")
  buffer.to_string()
  
  # prop_zero=[]
  # sample_size=10
  # for i in range(10):
  #   stime=time.time()
  #   state, action, trgt = buffer.sample_batch(sample_size)
  #   prop_zero.append(float(len(action[action==0]))/sample_size)
  #   print("time to sample: {0:f}s, proportion of labels -1: {1}, proportion of labels 0: {2}, proportion of labels 1: {3}".format(time.time()-stime, 
  #                                                                                                   float(len(action[action==-1]))/sample_size, 
  #                                                                                                   float(len(action[action==0]))/sample_size, 
  #                                                                                                   float(len(action[action==1]))/sample_size))

  # print("avg prop 0: {0} var prop 0: {1}".format(np.mean(prop_zero), np.var(prop_zero)))

  print("\n sample batch normalized \n")

  prop_zero=[]
  sample_size=10
  for i in range(10):
    stime=time.time()
    state, action, trgt = buffer.sample_batch(sample_size)
    print("trgt 0: {0}, trgt 1: {1}".format(len(trgt[trgt==0]),len(trgt[trgt==1])))
    # print("prp labels 0: {0}/10 prop labels 1 {1}/10".format(len(trgt[trgt == 0])), len(trgt[trgt==1]))
    # prop_zero.append(float(len(action[action==0]))/sample_size)
    # print("time to sample: {0:f}s, proportion of labels -1: {1}, proportion of labels 0: {2}, proportion of labels 1: {3}".format(time.time()-stime, 
    #                                                                                                 float(len(action[action==-1]))/sample_size, 
                                                                                                    # float(len(action[action==0]))/sample_size, 
                                                                                                    # float(len(action[action==1]))/sample_size))
  # print("avg prop 0: {0} var prop 0: {1}".format(np.mean(prop_zero), np.var(prop_zero)))
  
  # print("\n sampled batch normalized \n")
  # for i in range(6):
  #   print("action: {0}, target: {1}, state: {2}".format(action[i], trgt[i], state[i][0]))
