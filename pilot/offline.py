import time
import numpy as np
import tools
import sys
import data

import tensorflow as tf

FLAGS = None
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
  output_loss=[]
  for index, ok, batch in data.generate_batch(data_type):
    data_loading_time+=(time.time()-start_data_time)
    start_calc_time=time.time()
    if ok:

      inputs = np.array([_['img'] for _ in batch])
      actions = np.array([[_['ctr']] for _ in batch])
      if FLAGS.network == 'depth_q_net':
        targets = np.array([_['depth'] for _ in batch]).reshape((-1,FLAGS.output_size[0],FLAGS.output_size[1]))
      else:
        targets = np.array([_['trgt'] for _ in batch]).reshape((-1,1))
      if data_type=='train':
        losses = model.backward(inputs, actions=actions, targets=targets)
      elif data_type=='val' or data_type=='test':
        _, losses = model.forward(inputs, actions=actions, targets=targets)
      try:
        output_loss.extend(losses['o']) # for each element in the batch
        tot_loss.append(losses['t'])
      except KeyError:
        pass
      if index == 1 and data_type=='val' and FLAGS.plot_depth: 
          depth_predictions = tools.plot_depth(inputs, target, model)
    else:
      print('Failed to run {}.'.format(data_type))
    calculation_time+=(time.time()-start_calc_time)
    start_data_time = time.time()
  if len(tot_loss)!=0: sumvar['Loss_'+data_type+'_total']=np.mean(tot_loss) 
  if len(output_loss)!=0: 
    sumvar['Loss_'+data_type+'_output']=np.mean(output_loss)   
    sumvar['Loss_'+data_type+'_output_min']=np.min(output_loss)   
    sumvar['Loss_'+data_type+'_output_max']=np.max(output_loss)   
    sumvar['Loss_'+data_type+'_output_var']=np.var(output_loss)   
  if len(depth_predictions) != 0: sumvar['depth_predictions']=depth_predictions
  print('>>{4}: {0} [{1[2]}/{1[1]}_{1[3]:02d}:{1[4]:02d}]: data {2}; calc {3}'.format(data_type.upper(),tuple(time.localtime()[0:5]),
    tools.print_dur(data_loading_time),tools.print_dur(calculation_time), FLAGS.log_tag))
  if data_type == 'val' or data_type == 'test':
    # print('{}'.format(str([k+" : "+sumvar[k] for k in sumvar if k != 'depth_predictions'])))
    msg=str(["{0} : {1}".format(k,sumvar[k]) for k in sumvar.keys() if k != 'depth_predictions'])
    print(msg)
    f=open(FLAGS.summary_dir+FLAGS.log_tag+"/tf_log",'a')
    f.write(msg+'\n')
    f.close()
  sys.stdout.flush()
  return sumvar

def run(_FLAGS, model, start_ep=0):
  global FLAGS

  FLAGS=_FLAGS
  start_time=time.time()
  data.prepare_data(FLAGS, (model.input_size[1], model.input_size[2], model.input_size[3]))
  print("data loading time: {0:0.0f}".format(time.time()-start_time))
  ep=start_ep
  while ep<FLAGS.max_episodes-1 and not FLAGS.testing:
    ep+=1

    print('start episode: {}'.format(ep))
    # ----------- train episode
    sumvar = run_episode('train', {}, model)

    # ----------- validate episode
    #sumvar = run_episode('val', {}, model)

    sumvar = run_episode('val', sumvar, model)
    
    # ----------- write summary
    try:
      model.summarize(sumvar)
    except Exception as e:
      print('failed to summarize {}'.format(e))
    # write checkpoint every x episodes
    if (ep%20==0 and ep!=0) or ep==FLAGS.max_episodes-1:
      print('saved checkpoint')
      model.save(FLAGS.summary_dir+FLAGS.log_tag)
  # ------------ test
  sumvar = run_episode('test', {}, model)  
  # ----------- write summary
  try:
    model.summarize(sumvar)
  except Exception as e:
    print('failed to summarize {}'.format(e))
