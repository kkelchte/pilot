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

    def __init__(self, buffer_size=-1, random_seed=123, checkpoint=None):
      """
      The right side of the deque contains the most recent experiences 
      """
      self.buffer_size = buffer_size if buffer_size != -1 else 1000000
      self.count = 0
      self.buffer = deque()
      self.num_steps = 1 #num
      self.probs = None
      if checkpoint:
        self.load_checkpoint(checkpoint)

    def load_checkpoint(self, filename):
      """
      Open pickle file defined by checkpoint string, 
      load experiences in buffer,
      and close pickle file.
      """
      if os.path.isfile(filename):
        print("[replaybuffer] loading checkpoint")
        checkpoint=open(filename,'rb')
        data=pickle.load(checkpoint)
        checkpoint.close()
        for e in data: self.add(e)
      else:
        print("[replaybuffer] could not find replaybuffer checkpoint in filename {0}".format(filename))

    def save_checkpoint(self, filename):
      """
      Export queue as pickle binary file to load next time.
      """
      try:
        checkpoint=open(filename,'wb')
        pickle.dump(self.buffer, checkpoint)
        checkpoint.close()
      except Exception as e:
        print("[replaybuffer] failed to save replaybuffer: {0}".format(e.message))

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

      # print("[replaybuffer] added experience: {0}".format([experience[k] for k in ['action','trgt']]))
      self.buffer.append(experience)

    def remove(self):
      self.buffer.popleft()
      self.count-=1

    def size(self):      
      return self.count
    
    def softmax(self, x):
      e_x = np.exp(x-np.max(x))
      return e_x/e_x.sum()

    def get_all_data(self, max_batch_size=-1, data_buffer=None, horizon=0):
      # fill in a batch of size batch_size
      # return an array of inputs, targets and auxiliary information
      if data_buffer == None: data_buffer=self.buffer
      data_buffer=data_buffer[:-horizon] if horizon != 0 else data_buffer      
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
      
      input_batch = np.array([_['state'] for _ in batch])
      target_batch = np.array([_['trgt'] for _ in batch])
      action_batch = np.array([_['action'] for _ in batch])
      collision_batch = np.array([_['collision'] for _ in batch])
      
      return input_batch, target_batch, action_batch, collision_batch

    def update(self, update_rule='nothing', losses=[], hard_ratio=0):
      """Update after each training step the replay buffer according to an update rule
      nothing: 'dont do anything to update'
      empty: 'empty the buffer'
      hard: 'fill in ratio of buffer with hard samples and remove the rest'
      """
      if update_rule== 'nothing':
        return
      elif update_rule == 'empty':
        self.clear()
        return
      elif update_rule == 'hard':
        raise NotImplementedError('[replay_buffer]: hard update rule has not been implemented yet.')
      else:
        raise NotImplementedError('[replay_buffer]: update rule is unknown.')

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
      self.buffer.clear()
      # self.buffer = []
      self.count = 0

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


  mybuffer = ReplayBuffer(100, 512, checkpoint='/esat/opal/kkelchte/docker_home/tensorflow/log/test_replaybuffer/replaybuffer')
  print mybuffer.get_details()


