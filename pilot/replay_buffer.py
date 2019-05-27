""" 
Data structure for implementing experience replay
Author: Patrick Emami
"""
import time, os, shutil
from collections import deque
import random
import numpy as np
# import tensorflow as tf

import pickle

import skimage.io as sio

class ReplayBuffer(object):

    def __init__(self, buffer_size=-1, random_seed=123, action_normalization=False):
      """
      The right side of the deque contains the most recent experiences 
      """
      self.buffer_size = buffer_size if buffer_size != -1 else 1000000
      self.buffer = []
      self.action_normalization=action_normalization
      if self.action_normalization:
        self.action_count={}
      # self.buffer = deque()

    def add(self, experience):
      # add experience dictionary to buffer
      self.buffer.append(experience)
      if self.action_normalization and 'trgt' in experience.keys():
        if str(experience['trgt']) in self.action_count.keys():
          self.action_count[str(experience['trgt'])]+=1
        else:
          self.action_count[str(experience['trgt'])]=1
        print [str(k)+': '+str(self.action_count[k]) for k in self.action_count.keys()]

      # if buffer is full get rid of last experience
      if self.buffer_size != -1 and self.size() > self.buffer_size:
        if self.action_normalization and 'trgt' in experience.keys():
          # [KK] action normalization for discrete case:
          #   - discard not 0 but lowest sample with action equal to majority action.
          for i,e in enumerate(self.buffer):
            if str(e['trgt']) == max(self.action_count):
              self.buffer.pop(i)
              self.action_count[max(self.action_count)]-=1
              break
        else:
          self.buffer.pop(0)
        
    def remove(self):
      # self.buffer.popleft()
      self.buffer.pop(0)  

    def size(self):
      return len(self.buffer)
    
    def softmax(self, x):
      e_x = np.exp(x-np.max(x))
      return e_x/e_x.sum()

    def get_all_data(self, max_batch_size=-1, data_buffer=None, horizon=0):
      # fill in a batch of size batch_size
      # return an array of inputs, targets and auxiliary information
      if data_buffer == None: data_buffer=self.buffer
      # if horizon != 0:       data_buffer=data_buffer[:-horizon] if horizon != 0 else data_buffer      
      result={}
      for key in ['state', 'trgt', 'action', 'collision']:
        try:
          data=np.array([_[key] for i,_ in enumerate(data_buffer) if horizon == 0 or i < len(data_buffer)-horizon])
        except:
          pass
        else:
          if max_batch_size != -1 and data.shape[0] > max_batch_size:
            data=data[:max_batch_size]
          result[key]=data
      return result

    def get_all_data_shuffled(self, max_batch_size=-1, horizon=0):
      """ fill in a batch of size batch_size
      don't return the last horizon of labels as they might still get changed if a bump occurs at the next step.
      return an array of inputs, targets and auxiliary information
      """
      shuffled_indices =np.arange(len(self.buffer)-horizon)
      np.random.shuffle(shuffled_indices)
      shuffled_buffer = [self.buffer[i] for i in shuffled_indices]
      return self.get_all_data(max_batch_size, shuffled_buffer)

    def sample_batch(self, batch_size, horizon=0):
      """
      sample in a batch of size batch_size
      in case batch size can take full buffer, return full buffer
      return an array of inputs, targets and auxiliary information
      """
      batch_size=min(self.size()-horizon, batch_size)
    
      # sample from population unique instances, so doubles won't be there
      # if batch size can take the full buffer, it will
      batch=random.sample(self.buffer if horizon == 0 else self.buffer[:-horizon], batch_size)      
      result={}
      for key in ['state', 'trgt', 'action', 'collision']:
        try:
          data=np.array([_[key] for _ in batch])
        except:
          pass
        else:
          result[key]=data
      return result

    def update(self, update_rule='nothing', losses=[], train_every_N_steps=1):
      """Update after each training step the replay buffer according to an update rule
      nothing: 'dont do anything to update'
      empty: 'empty the buffer'
      hard: 'fill in ratio of buffer with hard samples and remove the rest'
      """
      if update_rule== 'nothing':
        pass
      elif update_rule == 'empty':
        self.clear()
        return
      elif update_rule == 'hard':
        assert len(losses) == self.size(), "{0} != {1}".format(len(losses), self.size())
        new_buffer=[e for _,e in reversed(sorted(zip(losses.tolist(), self.buffer), key=lambda f:f[0]))]
        self.buffer=new_buffer
      else:
        raise NotImplementedError('[replay_buffer]: update rule {0} is unknown.'.format(update_rule))
      if train_every_N_steps != 1:
        self.buffer=self.buffer[train_every_N_steps-1:]

    def annotate_collision(self, horizon):
      """Annotate the experiences over the last horizon with a 1 for collision.
      (!) Note that this is not compatible with recovery camera's
      for this an extension is required translating the horizon in time period (ms)
      and using the 'time' stamps of the experiences to annotate the experiences within this horizon period.
      --> shorter and easier hack is to multiply horizon with 3 for recovery from main arguments.
      """
      last_experiences=[]
      try:
        last_experiences=[self.buffer.pop() for i in range(horizon)]
      except:
        pass
      for e in last_experiences: e['collision']=1
      self.buffer.extend(reversed(last_experiences))
      
    def clear(self):
      # self.buffer.clear()
      self.buffer = []
      # self.count = 0

    def export_buffer(self, data_folder):
      """export buffer to file system as dataset:
      /.../log/${data_folder}/RGB/0000000x.png
      /.../log/${data_folder}/Depth/0000000x.png
      /.../log/${data_folder}/control_info.txt
      todo: Add data_folder incrementation over different runs...
      todo: add gt_listener position overview function
      """
      stime=time.time()
      if not data_folder.startswith('/'):
        data_folder=os.environ['HOME']+'/'+data_folder
      # loop over experiences and concatenate files and save images
      if os.path.isdir(data_folder): 
        print("[replaybuffer] delete existing data folder: {0}.".format(data_folder))
        shutil.rmtree(data_folder)
      os.makedirs("{0}/RGB".format(data_folder))
      for index, e in enumerate(self.buffer):
        if 'state' in e.keys():
          img=e['state'][:]
          if img.shape[0]==3:
            img=np.swapaxes(np.swapaxes(e['state'],0,1),1,2)
          sio.imsave("{0}/RGB/{1:010d}.png".format(data_folder, index),img)
        if 'trgt' in e.keys():
          with open("{0}/control_info.txt".format(data_folder),'a') as f:
            f.write("{0:010d} {1} 0 0 0 0 {2}\n".format(index, e['speed'] if 'speed' in e.keys() else 0.8, e['trgt']))
        if 'collision' in e.keys():
          with open("{0}/collision_info.txt".format(data_folder),'a') as f:
            f.write("{0:010d} {1}\n".format(index, e['collision']))
        if 'action' in e.keys():
          with open("{0}/action_info.txt".format(data_folder),'a') as f:
            f.write("{0:010d} {1} 0 0 0 0 {2}\n".format(index, e['speed'] if 'speed' in e.keys() else 0.8, e['action']))
      print("[replaybuffer] saving batch duration: {0:0.2f}s in folder {1}".format(time.time()-stime, data_folder))
      return

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
      if self.size()==0: return

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

  # mybuffer = ReplayBuffer(100, 512)

  # for i in range(30):
  #   mybuffer.add({'state':np.zeros((1,1))+i,
  #               'action':np.random.choice([-1,0,1],p=[0.1,0.1,0.8]),
  #               'trgt':np.random.choice([-1,0,1],p=[0.1,0.1,0.8]),
  #               'collision':1})
  # for i in range(10):
  #   mybuffer.add({'state':np.zeros((1,1))+100+i,
  #               'action':np.random.choice([-1,0,1],p=[0.1,0.1,0.8]),
  #               'trgt':np.random.choice([-1,0,1],p=[0.1,0.1,0.8]),
  #               'collision':0})

  # print mybuffer.get_details()

  # # mybuffer.export_buffer("{0}/tmp_data".format(os.environ['HOME']))
  # mybuffer.save_checkpoint('/esat/opal/kkelchte/docker_home/tensorflow/log/test_replaybuffer/replaybuffer')


  mybuffer = ReplayBuffer(100, 512, checkpoint='/esat/opal/kkelchte/docker_home/tensorflow/log/test_train_online_condor/seed_0/replaybuffer')
  print mybuffer.get_details()


