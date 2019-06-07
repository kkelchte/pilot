import time, copy
import numpy as np
import tools
import sys
import data
import torch

# import tensorflow as tf

FLAGS = None
epoch = 0

"""
This module scripts the procedure of running over episodes of training, validation and testing offline.
The data is collected from the data module.
"""

def run_episode(mode, sumvar, model):
  global epoch
  '''run over batches
  return different losses
  type: 'train', 'validation' or 'test'
  '''
  start_time=time.time()
  data_loading_time = 0
  calculation_time = 0
  start_data_time = time.time()
  results={}
  all_inputs=[]
  
  hidden_states=()

  # manual set zero grads in case of gradient accumulation with slidingtbptt
  if FLAGS.accum_grads:
    model.optimizer.zero_grad()

  for index, ok, batch in data.generate_batch(mode):
    data_loading_time+=(time.time()-start_data_time)
    start_calc_time=time.time()
    if ok:
      if len(batch[0]['img'].shape) > 3 and 'LSTM' in FLAGS.network:
        if len(hidden_states) != 0 and FLAGS.sliding_tbptt:
          h_t, c_t = (torch.from_numpy(hidden_states[0]),torch.from_numpy(hidden_states[1]))
          # print h_t, c_t
        else:
          stime=time.time()
          # for each sample in batch h_t is LxH --> has to become LxBxH
          hs, cs = [], []
          for _ in batch:
            h, c = tools.get_hidden_state(_['prev_imgs'], model)
            hs.append(torch.squeeze(h))
            cs.append(torch.squeeze(c))
          h_t=torch.stack(hs, dim=1)
          c_t=torch.stack(cs, dim=1)
          print("[offline] hiddenstate duration: {0:0.2f}".format(time.time()-stime))
        # assert(inputs.shape[0] == h_t.size()[1])
        inputs = np.array([_['img'] for _ in batch])
        if inputs.shape[0] != h_t.size()[1]:
          print("offline.py: h_t {0} and inputs {1} dont fit: ".format(h_t.size(),inputs.shape))
        inputs=(torch.from_numpy(inputs).type(torch.FloatTensor).to(model.device),(h_t.to(model.device),c_t.to(model.device)))
      else:
        inputs = np.array([_['img'] for _ in batch])
      targets = np.array([_['ctr'] for _ in batch])
      
      if mode=='train':
        epoch, predictions, losses, hidden_states = model.train(inputs, targets)
        for k in losses.keys(): tools.save_append(results, k, losses[k])
      elif mode=='validation' or mode=='test':
        predictions, losses, hidden_states = model.predict(inputs, targets)
        for k in losses.keys(): tools.save_append(results, k, losses[k])
    else:
      print('Failed to run {}.'.format(mode))
    calculation_time+=(time.time()-start_calc_time)
    start_data_time = time.time()

  if FLAGS.accum_grads:
    model.optimizer.step()
    model.epoch+=1


  for k in results.keys():
    if len(results[k])!=0: sumvar[mode+'_'+k]=np.mean(results[k]) 
  print('>>{0} [{1[2]}/{1[1]}_{1[3]:02d}:{1[4]:02d}]: data {2}; calc {3}'.format(mode.upper(),tuple(time.localtime()[0:5]),
    tools.print_dur(data_loading_time),tools.print_dur(calculation_time)))
  return sumvar

def run(_FLAGS, model):
  global FLAGS, epoch
  
  FLAGS=_FLAGS
  start_time=time.time()

  if FLAGS.create_scratch_checkpoint:
    model.save(FLAGS.summary_dir+FLAGS.log_tag, save_optimizer=False)
    sys.exit(0)

  data.prepare_data(FLAGS, model.input_size)
  print("data loading time: {0:0.0f}".format(time.time()-start_time))
  epoch=model.epoch
 
  while epoch<FLAGS.max_episodes and not FLAGS.testing:
    print('\n {0} : start episode: {1}'.format(FLAGS.log_tag, epoch))

    debug=False
    if debug: print("ALLERT ONLY VALIDATION")
    # ----------- train episode: update importance weights on training data
    # sumvar = run_episode('train', {}, model, ep==FLAGS.max_episodes-1 and FLAGS.update_importance_weights)    
    if not debug: sumvar = run_episode('train', {}, model)    
    
    # if FBPTT: don't validate after each training step:
    if 'LSTM' in FLAGS.network and (FLAGS.time_length==-1 or FLAGS.accum_grads) and model.epoch%100 != 0:
      continue

    # ----------- validate episode
    # validate with SBPTT at 1
    if 'LSTM' in FLAGS.network:
      time_length=FLAGS.time_length
      sliding_step_size=FLAGS.sliding_step_size
      sliding_tbptt=FLAGS.sliding_tbptt
      FLAGS.time_length=1
      FLAGS.sliding_step_size=1
      FLAGS.sliding_tbptt=True
    if debug: sumvar = run_episode('validation', {}, model)
    # sumvar = run_episode('validation', sumvar, model, ep==FLAGS.max_episodes-1 and FLAGS.update_importance_weights)
    if not debug: sumvar = run_episode('validation', sumvar, model)
    if 'LSTM' in FLAGS.network:
      FLAGS.time_length=time_length
      FLAGS.sliding_step_size=sliding_step_size
      FLAGS.sliding_tbptt=sliding_tbptt
    
    # get all metrics of this episode and add them to var
    # print end of episode
    tags_not_to_print=[]
    msg="run : {0}".format(epoch)
    for k in sumvar.keys(): msg="{0}, {1} : {2}".format(msg, k, sumvar[k]) if k not in tags_not_to_print else msg
    print(msg)
    f=open(FLAGS.summary_dir+FLAGS.log_tag+"/tf_log",'a')
    f.write(msg+'\n')
    f.close()
    sys.stdout.flush()
    model.summarize(sumvar)

    # write checkpoint every x episodes
    if (epoch%2000==0 and epoch!=0) or FLAGS.max_episodes-epoch <= 100:
      print('[offline]: save checkpoint')
      model.save(FLAGS.summary_dir+FLAGS.log_tag)

  if FLAGS.max_episodes != 0:
    # ------------ test
    if 'LSTM' in FLAGS.network:
      time_length=FLAGS.time_length
      sliding_step_size=FLAGS.sliding_step_size
      sliding_tbptt=FLAGS.sliding_tbptt
      FLAGS.time_length=1
      FLAGS.sliding_step_size=1
      FLAGS.sliding_tbptt=True
    sumvar = run_episode('test', {}, model)  
    if 'LSTM' in FLAGS.network:
      FLAGS.time_length=time_length
      FLAGS.sliding_step_size=sliding_step_size
      FLAGS.sliding_tbptt=sliding_tbptt
    # ----------- write summary
    tags_not_to_print=[]
    msg="final test run : {0}".format(epoch)
    for k in sumvar.keys(): msg="{0}, {1} : {2}".format(msg, k, sumvar[k]) if k not in tags_not_to_print else msg
    print(msg)
    f=open(FLAGS.summary_dir+FLAGS.log_tag+"/tf_log",'a')
    f.write(msg+'\n')
    f.close()
    sys.stdout.flush()
    print('saved checkpoint')
    model.save(FLAGS.summary_dir+FLAGS.log_tag)

  if FLAGS.visualize_saliency_of_output:
    # Select hard images from last test set
    for index, ok, batch in data.generate_batch('test'):
      inputs = np.array([_['img'] for _ in batch])
      targets = np.array([_['ctr'] for _ in batch])
      predictions, losses = model.predict(inputs, targets)
      # --> loss is already mean ==> have to keep raw losses.
      hard_input=[np.array(x) for _,x in reversed(sorted(zip(list(losses['imitation_learning']), inputs.tolist())))][0]
      tools.visualize_saliency_of_output(FLAGS, model, hard_input)
      break

  if FLAGS.calculate_importance_weights:
    importance_weights=tools.calculate_importance_weights(model, data.get_all_inputs('validation'), level='neuron')
    import pickle
    with open(FLAGS.summary_dir+FLAGS.log_tag+"/omegas",'wb') as f:
      pickle.dump(importance_weights, f)
    tools.visualize_importance_weights(importance_weights, FLAGS.summary_dir+FLAGS.log_tag)

  if FLAGS.extract_nearest_features:
    stime=time.time()
    # FLAGS.dataset = 'esatv3_expert/2500'
    # data.prepare_data(FLAGS, model.input_size, datatypes=['train'])
    source_dataset=copy.deepcopy(data.full_set['train'])
    print("prepare source data duration: {0:0.0f}".format(time.time()-stime))
    stime=time.time()
    
    # FLAGS.dataset = 'esatv3_expert/mini'
    FLAGS.dataset = 'real_drone'
    data.prepare_data(FLAGS, model.input_size, datatypes=['train'])
    target_dataset=copy.deepcopy(data.full_set['train'])
    print("prepare target data duration: {0:0.0f}".format(time.time()-stime))
    stime=time.time()
    
    feature_extractor=tools.NearestFeatures(model, source_dataset, target_dataset)
    print("calculate features: {0:0.0f}".format(time.time()-stime))
    stime=time.time()
    feature_extractor.create_graph(FLAGS.summary_dir+FLAGS.log_tag)
    print("create graph duration: {0:0.0f}".format(time.time()-stime))
    stime=time.time()
    feature_extractor.calculate_differences(FLAGS.summary_dir+FLAGS.log_tag)
    print("calculate difference duration: {0:0.0f}".format(time.time()-stime))

  # import pdb; pdb.set_trace()
  # if FLAGS.visualize_deep_dream_of_output:
  #   tools.deep_dream_of_extreme_control(FLAGS, model)

  # if FLAGS.visualize_activations:
  #   tools.visualize_activations(FLAGS,model) 

  # if FLAGS.visualize_control_activation_maps and 'CAM' in FLAGS.network:
  #   tools.visualize_control_activation_maps(FLAGS,model) 

