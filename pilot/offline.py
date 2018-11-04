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


def run_episode(mode, sumvar, model, update_importance_weights=False):
  '''run over batches
  return different losses
  type: 'train', 'val' or 'test'
  '''
  depth_predictions = []
  start_time=time.time()
  data_loading_time = 0
  calculation_time = 0
  start_data_time = time.time()
  # results = {}
  # results['control'] = []
  results={'total': []}
  results['ce'] = []
  for k in model.lll_losses.keys():
    results['lll_'+k]=[]
  # results['accuracy'] = []
  # if FLAGS.auxiliary_depth: results['depth'] = []
  
  all_inputs=[]

  for index, ok, batch in data.generate_batch(mode):
    data_loading_time+=(time.time()-start_data_time)
    start_calc_time=time.time()
    if ok:
      inputs = np.array([_['img'] for _ in batch])
      targets = np.array([[_['ctr']] for _ in batch])
      if update_importance_weights and len(all_inputs) < 2000 :
        try:
          all_inputs=np.concatenate([all_inputs,inputs], axis=0)
        except:
          all_inputs=inputs[:]
      # print("targets: {}".format(targets))
      # try:
      target_depth = np.array([_['depth'] for _ in batch]).reshape((-1,55,74)) if FLAGS.auxiliary_depth else []
      if len(target_depth) == 0 and FLAGS.auxiliary_depth: raise ValueError('No depth in batch.')
      if mode=='train':
        # model.backward(inputs, targets=targets, depth_targets=target_depth, sumvar=sumvar)
        losses = model.backward(inputs, targets=targets, depth_targets=target_depth, sumvar=sumvar)
        for k in losses.keys():
          results[k].append(losses[k])
      elif mode=='val' or mode=='test':
        _, aux_results = model.forward(inputs, auxdepth=False, targets=targets, depth_targets=target_depth)
      if index == 1 and mode=='val' and FLAGS.plot_depth: 
          depth_predictions = tools.plot_depth(inputs, target_depth, model)
    else:
      print('Failed to run {}.'.format(mode))
    calculation_time+=(time.time()-start_calc_time)
    start_data_time = time.time()

  if update_importance_weights:
    model.update_importance_weights(all_inputs)
    
  for k in results.keys():
    if len(results[k])!=0: sumvar['Loss_'+mode+'_'+k]=np.mean(results[k]) 
  if len(depth_predictions) != 0: sumvar['depth_predictions']=depth_predictions
  print('>>{0} [{1[2]}/{1[1]}_{1[3]:02d}:{1[4]:02d}]: data {2}; calc {3}'.format(mode.upper(),tuple(time.localtime()[0:5]),
    tools.print_dur(data_loading_time),tools.print_dur(calculation_time)))
  return sumvar

def run(_FLAGS, model, start_ep=0):
  global FLAGS

  FLAGS=_FLAGS
  start_time=time.time()
  data.prepare_data(FLAGS, (model.input_size[1], model.input_size[2], model.input_size[3]))
  print("data loading time: {0:0.0f}".format(time.time()-start_time))
  ep=start_ep
  
  model.reset_metrics()    
  while ep<FLAGS.max_episodes-1 and not FLAGS.testing:
    ep+=1

    print('\n {0} : start episode: {1}'.format(FLAGS.log_tag, ep))
    # reset running metric variables
    model.reset_metrics()    

    # ----------- train episode: update importance weights on training data
    # sumvar = run_episode('train', {}, model, ep==FLAGS.max_episodes-1 and FLAGS.update_importance_weights)    
    sumvar = run_episode('train', {}, model)    
    
    # ----------- validate episode
    # sumvar = run_episode('val', {}, model)
    sumvar = run_episode('val', sumvar, model, ep==FLAGS.max_episodes-1 and FLAGS.update_importance_weights)

    # get all metrics of this episode and add them to var
    results = model.get_metrics()
    for k in results.keys():
      sumvar[k] = results[k]
    # print end of episode
    tags_not_to_print=['depth_predictions']+['activations_'+e for e in model.endpoints['eval'].keys()]+['weights_'+v.name for v in tf.trainable_variables()]
    msg="run : {0}".format(ep)
    for k in sumvar.keys(): msg="{0}, {1} : {2}".format(msg, k, sumvar[k]) if k not in tags_not_to_print else msg
    print(msg)
    f=open(FLAGS.summary_dir+FLAGS.log_tag+"/tf_log",'a')
    f.write(msg+'\n')
    f.close()
    sys.stdout.flush()

    # ----------- write summary
    try:
      model.summarize(sumvar)
    except Exception as e:
      print('failed to summarize {}'.format(e))
    # write checkpoint every x episodes
    if (ep%20==0 and ep!=0) or ep==FLAGS.max_episodes-1:
      print('saved checkpoint')
      model.save(FLAGS.summary_dir+FLAGS.log_tag)

    # import pdb; pdb.set_trace()


  if FLAGS.max_episodes != 0:
    # ------------ test
    model.reset_metrics()    
    sumvar = run_episode('test', {}, model)  
    # sumvar = run_episode('test', {}, model, FLAGS.update_importance_weights)  
    # ----------- write summary
    results = model.get_metrics()
    for k in results.keys(): 
      if results[k] != 0: sumvar[k] = results[k]
    tags_not_to_print=['depth_predictions']+['activations_'+e for e in model.endpoints['eval'].keys()]+['weights_'+v.name for v in tf.trainable_variables()]
    msg="run : {0}".format(ep)
    for k in sumvar.keys(): msg="{0}, {1} : {2}".format(msg, k, sumvar[k]) if k not in tags_not_to_print else msg
    print(msg)
    f=open(FLAGS.summary_dir+FLAGS.log_tag+"/tf_log",'a')
    f.write(msg+'\n')
    f.close()
    sys.stdout.flush()
    print('saved checkpoint')
    model.save(FLAGS.summary_dir+FLAGS.log_tag)

    try:
      model.summarize(sumvar)
    except Exception as e:
      print('failed to summarize {}'.format(e))    

  if FLAGS.visualize_saliency_of_output:
    tools.visualize_saliency_of_output(FLAGS, model)

  if FLAGS.visualize_deep_dream_of_output:
    tools.deep_dream_of_extreme_control(FLAGS, model)

  if FLAGS.visualize_activations:
    tools.visualize_activations(FLAGS,model) 

  if FLAGS.visualize_control_activation_maps and 'CAM' in FLAGS.network:
    tools.visualize_control_activation_maps(FLAGS,model) 

