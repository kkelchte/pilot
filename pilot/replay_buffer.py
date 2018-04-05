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

    def __init__(self, FLAGS, random_seed=123):
      """
      The right side of the deque contains the most recent experiences 
      """
      self.FLAGS=FLAGS
      self.buffer_size = FLAGS.buffer_size
      self.count = 0
      self.buffer = deque()
      self.num_steps = 1 #num
      self.probs = None

    def add(self, experience):
      if self.count < self.buffer_size: 
        self.count += 1
        self.buffer.append(experience)
      else:
        if self.FLAGS.prioritized_keeping and self.probs:
          last=np.argmin(self.probs)
          del self.probs[last]
          del self.buffer[last]
          self.buffer.append(experience)
          self.preprocess()
        else:
          self.buffer.popleft()
          self.buffer.append(experience)
    
    def size(self):
      
      return self.count
    
    def sample_batch(self):
      assert self.FLAGS.batch_size < self.count, IOError('batchsize ',self.FLAGS.batch_size,' is bigger than buffer size: ',self.count)
      
      batch=np.random.choice(self.buffer, self.FLAGS.batch_size, p=np.squeeze(self.probs) if self.probs else None)      
        
      state_batch = np.array([_['state'] for _ in batch])
      action_batch = np.array([max(min(_['action'],self.FLAGS.action_bound),-self.FLAGS.action_bound) for _ in batch])
      trgt_batch = np.array([_['trgt'] for _ in batch])
      
      return state_batch, action_batch, trgt_batch

    def prioritize_with_uniform_action(self):
      """Provide probability weights with sampling ensuring that every type of action -1,0,1 is uniformly sampled
      """
      probs=[]
      N={-1:0, 0:0, 1:0}
      for e in self.buffer:
        if np.abs(e['action']) > 0.3: N[np.sign(e['action'])]+=1
        else: N[0]+=1
      for e in self.buffer: 
        if np.abs(e['action']) > 0.3: probs.append(1/(3.0*N[np.sign(e['action'])]))
        else: probs.append(1/(3.0*N[0]))
      # ensure that probs sum to one by adjusting the last
      if sum(probs)!=1: probs[-1]=1-sum(probs[:-1])
      self.probs = probs[:]

    def prioritize_with_uniform_collision(self):
      """Provide probability weights with sampling ensuring that every type of collision 0,1 is uniformly sampled
      """
      if self.FLAGS.network != 'coll_q_net': 
        print('[replay_buffer]: prioritize_with_uniform_collision is not possible for {}.'.format(self.FLAGS.network))
        return
      probs=[]
      N={0:0, 1:0}
      for e in self.buffer: N[e['trgt']]+=1
      for e in self.buffer: probs.append(1/(2.0*N[e['trgt']]))
      # ensure that probs sum to one by adjusting the last
      if sum(probs)!=1: probs[-1]=1-sum(probs[:-1])
      self.probs = probs[:]
 
    def prioritize_with_td_error(self):
      """Provide probability weights according to TD-error
      """
      try:
        total=sum([e['error'] for e in self.buffer])
      except:
        print('[replay_buffer]: failed to extract TD error from replay buffer, continue without priority replay.')
      else:
        probs=[e['error']/(total+0.01) for e in self.buffer]
        # ensure that probs sum to one by adjusting the last
        if sum(probs)!=1: probs[-1]=1-sum(probs[:-1])
        self.probs = probs[:]

    def prioritize_with_variance(self,variance_source='state'):
      """Provide probability weights according to state/action/target variance
      """
      mean_image=np.mean([e[variance_source] for e in self.buffer])
      variances=[np.mean((e[variance_source]-mean_image)**2) for e in self.buffer]
      total_variance=sum(variances)
      self.probs=[v/total_variance for v in variances]
      # ensure that probs sum to one by adjusting the last
      if sum(self.probs)!=1: self.probs[-1]=1-sum(self.probs[:-1])
        
    def prioritize_with_random_actions(self):
      """Pick actions from replay buffer if they were selected randomly
      """
      # stime=time.time()
      total_random=sum([e['rnd'] for e in self.buffer])
      # print('total_random: {}'.format(total_random))
      if total_random == 0: total_random=self.count #everything is equally likely if no experiences were random
      self.probs=[float(e['rnd'])/(total_random+0.0) for e in self.buffer]
      if sum(self.probs)!=1: self.probs[-1]=1-sum(self.probs[:-1])
      # print("duration: {}".format(time.time()-stime))

    def preprocess(self):
      '''Calculate priorities according to type of prioritized replay
      '''
      # Calculate priorities according to type of prioritized replay
      if self.FLAGS.replay_priority == 'no': 
        return
      elif self.FLAGS.replay_priority == 'uniform_action': 
        self.prioritize_with_uniform_action()
      elif self.FLAGS.replay_priority == 'uniform_collision': 
        self.prioritize_with_uniform_collision()
      elif self.FLAGS.replay_priority == 'td_error': 
        self.prioritize_with_td_error()
      elif self.FLAGS.replay_priority == 'action_variance':
        self.prioritize_with_variance('action')
      elif self.FLAGS.replay_priority == 'state_variance':
        self.prioritize_with_variance('state')
      elif self.FLAGS.replay_priority == 'trgt_variance':
        self.prioritize_with_variance('trgt')
      elif self.FLAGS.replay_priority == 'random_action':
        self.prioritize_with_random_actions()
      else:
        raise NotImplementedError( '[ReplayBuffer] Type of priority is not implemented: ', self.FLAGS.replay_priority)

    def label_collision(self,logfolder=''):
      #label the last n experiences with target 1 
      # as collision appeared in the next 7 steps
      if self.FLAGS.network == 'coll_q_net':
        try:
          f=open(logfolder+'/log','r')
        except:
          pass
        else:
          lines=f.readlines()
          f.close()
          if "bump" in lines[-1] and self.count!=0:
            print('label last n frames with collision') 
        n=7
        # from t_end till t_end-n
        last_experiences=[self.buffer.pop() for i in range(n)]
        for e in last_experiences: e['trgt']=1
        if self.FLAGS.replay_priority == 'td_error':
          for e in last_experiences: 
            e['error']=1-e['error']
        self.buffer.extend(reversed(last_experiences))

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def to_string(self):
      for i,e in enumerate(self.buffer): 
        msg=""
        for k in e.keys(): msg="{0} {1}: {2}".format(msg, k, np.asarray(e[k]).flatten()[0])
        if self.probs: msg="{0}, p: {1}".format(msg,self.probs[i])
        print msg
     
    def get_variance(self):
      images=np.asarray([e['state'] for e in self.buffer])
      mean_state_variance=np.mean(np.var(images,axis=0))
      # print 'mean state variance: '+str(mean_state_variance)
      actions=np.asarray([e['action'] for e in self.buffer])
      action_variance=np.var(actions,axis=0)
      # print 'action variance: ',action_variance
      targets=np.asarray([e['trgt'] for e in self.buffer])
      mean_trgt_variance=np.mean(np.var(targets,axis=0))
      # print 'mean trgt variance: '+str(mean_state_variance)
      return {'state':mean_state_variance, 'action':action_variance, 'trgt':mean_trgt_variance}  


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test replay buffer for coll_q_net:')

  parser.add_argument("--replay_priority", default='no', type=str, help="Define which type of weights should be used when sampling from replay buffer: no, uniform_action, uniform_collision, td_error, recency, min_variance")
  parser.add_argument("--network",default='coll_q_net',type=str, help="Define the type of network: depth_q_net, coll_q_net.")
  parser.add_argument("--buffer_size", default=20, type=int, help="Define the number of experiences saved in the buffer.")
  parser.add_argument("--batch_size",default=10,type=int,help="Define the size of minibatches.")
  parser.add_argument("--action_bound", default=1.0, type=float, help= "Define between what bounds the actions can go. Default: [-1:1].")
  
  FLAGS=parser.parse_args()  

  # FLAGS.network='coll_q_net'
  FLAGS.network='depth_q_net'

  FLAGS.replay_priority="random_action"
  FLAGS.prioritized_keeping=True

  print "FLAGS.replay_priority: ",FLAGS.replay_priority

  # sample episode of data acquisition
  buffer=ReplayBuffer(FLAGS)
  for i in range(20):
    action=np.random.choice([-1,0,1],p=[0.1,0.8,0.1])
    buffer.add({'state':np.zeros((1,1))+100+i,
                'action':action,
                'trgt':0,
                'error':10,
                'rnd':action==1})
  buffer.preprocess()
  print("\n content of the buffer: \n")
  buffer.to_string()

  # print buffer.get_variance()
  
  # buffer.label_collision()
  
  prop_zero=[]
  for i in range(10):
    stime=time.time()
    state, action, trgt = buffer.sample_batch()
    print("state: {1}, actions: {0}".format(np.asarray(action).flatten()[0],np.asarray(state).flatten()[0]))
    prop_zero.append(state)

    # print("trgt 0: {0}, trgt 1: {1}".format(len(trgt[trgt==0]),len(trgt[trgt==1])))
    # prop_zero.append(float(len(trgt[trgt==0]))/FLAGS.batch_size)
    
    # prop_zero.append(float(len(action[action==0]))/FLAGS.batch_size)
    # print("time to sample: {0:f}s, proportion of labels -1: {1}, proportion of labels 0: {2}, proportion of labels 1: {3}".format(time.time()-stime,
     # float(len(action[action==-1]))/FLAGS.batch_size,float(len(action[action==0]))/FLAGS.batch_size,float(len(action[action==1]))/FLAGS.batch_size))

  print("avg prop 0: {0} var prop 0: {1}".format(np.mean(prop_zero), np.var(prop_zero)))

  for i in range(50):
    action=np.random.choice([-1,0,1],p=[0.1,0.8,0.1])
    buffer.add({'state':np.zeros((1,1))+1000+i,
                'action':action,
                'trgt':0,
                'error':10,
                'rnd':action==1})
  print("\n content of the buffer: \n")
  buffer.to_string()
  
  prop_zero=[]
  for i in range(10):
    stime=time.time()
    state, action, trgt = buffer.sample_batch()
    print("state: {1}, actions: {0}".format(np.asarray(action).flatten()[0],np.asarray(state).flatten()[0]))
    prop_zero.append(state)
    
    # print("trgt 0: {0}, trgt 1: {1}".format(len(trgt[trgt==0]),len(trgt[trgt==1])))
    # prop_zero.append(float(len(trgt[trgt==0]))/FLAGS.batch_size)
    
    # prop_zero.append(float(len(action[action==0]))/FLAGS.batch_size)
    # print("time to sample: {0:f}s, proportion of labels -1: {1}, proportion of labels 0: {2}, proportion of labels 1: {3}".format(time.time()-stime, 
                                                                                                    # float(len(action[action==-1]))/FLAGS.batch_size, 
                                                                                                    # float(len(action[action==0]))/FLAGS.batch_size, 
                                                                                                    # float(len(action[action==1]))/FLAGS.batch_size))
  print("avg prop 0: {0} var prop 0: {1}".format(np.mean(prop_zero), np.var(prop_zero)))
  
  # print("\n sampled batch normalized \n")
  # for i in range(6):
  #   print("action: {0}, target: {1}, state: {2}".format(action[i], trgt[i], state[i][0]))
