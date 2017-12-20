import time
import numpy as np
import tools
import sys
import data

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("max_episodes", 80, "The maximum number of episodes (~runs through all the training data.)")

"""
This module scripts the procedure of running over episodes of training, validation and testing offline.
The data is collected from the data module.
"""


def run_episode(data_type, sumvar, model):
  '''run over batches
  return different losses
  type: 'train', 'val' or 'test'
  '''
  depth_predictions = []
  start_time=time.time()
  data_loading_time = 0
  calculation_time = 0
  start_data_time = time.time()
  tot_loss=[]
  ctr_loss=[]
  dep_loss=[]
  for index, ok, batch in data.generate_batch(data_type):
    data_loading_time+=(time.time()-start_data_time)
    start_calc_time=time.time()
    if ok:
      inputs = np.array([_['img'] for _ in batch])
      if FLAGS.data_format == "NCHW": 
        inputs = np.swapaxes(np.swapaxes(inputs,2,3),1,2)
      targets = np.array([[_['ctr']] for _ in batch])
      try:
        target_depth = np.array([_['depth'] for _ in batch]).reshape((-1,55,74)) if FLAGS.auxiliary_depth else []
        if len(target_depth) == 0 : raise ValueError('No depth in batch.')
      except ValueError: 
        target_depth = [] # In case there is no depth targets available
      if data_type=='train':
        losses = model.backward(inputs, targets, depth_targets=target_depth)
      elif data_type=='val' or data_type=='test':
        _, losses, aux_results = model.forward(inputs, auxdepth=False, targets=targets, depth_targets=target_depth)
      try:
        ctr_loss.append(losses['c'])
        if FLAGS.auxiliary_depth: dep_loss.append(losses['d'])
        tot_loss.append(losses['t'])
      except KeyError:
        pass
      if index == 1 and data_type=='val' and FLAGS.plot_depth: 
          depth_predictions = tools.plot_depth(inputs, target_depth, model)
    else:
      print('Failed to run {}.'.format(data_type))
    calculation_time+=(time.time()-start_calc_time)
    start_data_time = time.time()
  if len(tot_loss)!=0: sumvar['Loss_'+data_type+'_total']=np.mean(tot_loss) 
  if len(ctr_loss)!=0: sumvar['Loss_'+data_type+'_control']=np.mean(ctr_loss)   
  if len(tot_loss)!=0 and FLAGS.auxiliary_depth: sumvar['Loss_'+data_type+'_depth']=np.mean(dep_loss)   
  if len(depth_predictions) != 0: sumvar['depth_predictions']=depth_predictions
  print('>>{0} [{1[2]}/{1[1]}_{1[3]:02d}:{1[4]:02d}]: data {2}; calc {3}'.format(data_type.upper(),tuple(time.localtime()[0:5]),
    tools.print_dur(data_loading_time),tools.print_dur(calculation_time)))
  if data_type == 'val' or data_type == 'test':
    print('{}'.format(str([sumvar[k] for k in sumvar if k != 'depth_predictions'])))

  sys.stdout.flush()
  return sumvar

def run(model):
  if FLAGS.data_format=="NCHW":
    data.prepare_data((model.input_size[2], model.input_size[3], model.input_size[1]))
  else:
    data.prepare_data((model.input_size[1], model.input_size[2], model.input_size[3]))
  
  ep=0
  while ep<FLAGS.max_episodes-1 and not FLAGS.testing:
    ep+=1

    print('start episode: {}'.format(ep))
    # ----------- train episode
    sumvar = run_episode('train', {}, model)
    
    # ----------- validate episode
    #sumvar = run_episode('val', {}, model)
    sumvar = run_episode('val', sumvar, model)
    
    #import pdb; pdb.set_trace()

    # ----------- write summary
    try:
      model.summarize(sumvar)
    except Exception as e:
      print('failed to summarize {}'.format(e))
    # write checkpoint every x episodes
    if (ep%20==0 and ep!=0):
      print('saved checkpoint')
      model.save(FLAGS.summary_dir+FLAGS.log_tag)
  # ------------ test
  sumvar = run_episode('test', {}, model)  
  # ----------- write summary
  try:
    model.summarize(sumvar)
  except Exception as e:
    print('failed to summarize {}'.format(e))
  # write checkpoint every x episodes
  if ((ep%20==0 and ep!=0) or ep==(FLAGS.max_episodes-1)) and not FLAGS.testing:
    print('saved checkpoint')
    model.save(FLAGS.summary_dir+FLAGS.log_tag)
