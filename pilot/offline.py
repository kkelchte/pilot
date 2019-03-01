import time
import numpy as np
import tools
import sys
import data

# import tensorflow as tf

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
  start_time=time.time()
  data_loading_time = 0
  calculation_time = 0
  start_data_time = time.time()
  results={}
  all_inputs=[]
  for index, ok, batch in data.generate_batch(mode):
    data_loading_time+=(time.time()-start_data_time)
    start_calc_time=time.time()
    if ok:
      inputs = np.array([_['img'] for _ in batch])
      targets = np.array([[_['ctr']] for _ in batch])
      if update_importance_weights: all_inputs.append(inputs)
      if mode=='train':
        step, predictions, losses = model.train(inputs, targets)
        for k in losses.keys(): tools.save_append(results, k, losses[k])
      elif mode=='val' or mode=='test':
        predictions, losses = model.predict(inputs, targets)
        for k in losses.keys(): tools.save_append(results, k, losses[k])
    else:
      print('Failed to run {}.'.format(mode))
    calculation_time+=(time.time()-start_calc_time)
    start_data_time = time.time()

  if update_importance_weights:
    model.update_importance_weights(np.asarray(all_inputs))

  for k in results.keys():
    if len(results[k])!=0: sumvar['Loss_'+mode+'_'+k]=np.mean(results[k]) 
  print('>>{0} [{1[2]}/{1[1]}_{1[3]:02d}:{1[4]:02d}]: data {2}; calc {3}'.format(mode.upper(),tuple(time.localtime()[0:5]),
    tools.print_dur(data_loading_time),tools.print_dur(calculation_time)))
  return sumvar

def run(_FLAGS, model, start_ep=0):
  global FLAGS
  
  FLAGS=_FLAGS
  start_time=time.time()
  data.prepare_data(FLAGS, model.input_size)
  print("data loading time: {0:0.0f}".format(time.time()-start_time))
  ep=start_ep
  
  while ep<FLAGS.max_episodes-1 and not FLAGS.testing:
    ep+=1

    print('\n {0} : start episode: {1}'.format(FLAGS.log_tag, ep))

    # ----------- train episode: update importance weights on training data
    # sumvar = run_episode('train', {}, model, ep==FLAGS.max_episodes-1 and FLAGS.update_importance_weights)    
    sumvar = run_episode('train', {}, model)    
    
    # ----------- validate episode
    # sumvar = run_episode('val', {}, model)
    sumvar = run_episode('val', sumvar, model, ep==FLAGS.max_episodes-1 and FLAGS.update_importance_weights)

    # get all metrics of this episode and add them to var
    # print end of episode
    tags_not_to_print=[]
    msg="run : {0}".format(ep)
    for k in sumvar.keys(): msg="{0}, {1} : {2}".format(msg, k, sumvar[k]) if k not in tags_not_to_print else msg
    print(msg)
    f=open(FLAGS.summary_dir+FLAGS.log_tag+"/tf_log",'a')
    f.write(msg+'\n')
    f.close()
    sys.stdout.flush()
    model.summarize(sumvar)

    # write checkpoint every x episodes
    if (ep%20==0 and ep!=0) or ep==FLAGS.max_episodes-1:
      print('saved checkpoint')
      model.save(FLAGS.summary_dir+FLAGS.log_tag)

  if FLAGS.max_episodes != 0:
    # ------------ test
    sumvar = run_episode('test', {}, model)  
    # ----------- write summary
    tags_not_to_print=[]
    msg="final test run : {0}".format(ep)
    for k in sumvar.keys(): msg="{0}, {1} : {2}".format(msg, k, sumvar[k]) if k not in tags_not_to_print else msg
    print(msg)
    f=open(FLAGS.summary_dir+FLAGS.log_tag+"/tf_log",'a')
    f.write(msg+'\n')
    f.close()
    sys.stdout.flush()
    print('saved checkpoint')
    model.save(FLAGS.summary_dir+FLAGS.log_tag)

  # if FLAGS.visualize_saliency_of_output:
  #   tools.visualize_saliency_of_output(FLAGS, model)

  # if FLAGS.visualize_deep_dream_of_output:
  #   tools.deep_dream_of_extreme_control(FLAGS, model)

  # if FLAGS.visualize_activations:
  #   tools.visualize_activations(FLAGS,model) 

  # if FLAGS.visualize_control_activation_maps and 'CAM' in FLAGS.network:
  #   tools.visualize_control_activation_maps(FLAGS,model) 

