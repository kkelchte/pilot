import time
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

  for index, ok, batch in data.generate_batch(mode):
    data_loading_time+=(time.time()-start_data_time)
    start_calc_time=time.time()
    if ok:
      if len(batch[0]['img'].shape) > 3:
        if len(hidden_states) != 0 and FLAGS.sliding_tbptt:
          h_t, c_t = (torch.from_numpy(hidden_states[0]),torch.from_numpy(hidden_states[1]))
        else:
          # for each sample in batch h_t is LxH --> has to become LxBxH
          hs, cs = [], []
          for _ in batch:
            h, c = tools.get_hidden_state(_['prev_imgs'], model)
            hs.append(torch.squeeze(h))
            cs.append(torch.squeeze(c))
          h_t=torch.stack(hs, dim=1)
          c_t=torch.stack(cs, dim=1)
        # assert(inputs.shape[0] == h_t.size()[1])
        inputs = np.array([_['img'] for _ in batch])
        if inputs.shape[0] != h_t.size()[1]:
          print("offline.py: h_t {0} and inputs {1} dont fit: ".format(h_t.size(),inputs.shape))
          # import pdb; pdb.set_trace()
        inputs=(torch.from_numpy(inputs).type(torch.FloatTensor).to(model.device),(h_t.to(model.device),c_t.to(model.device)))
      else:
        inputs = np.array([_['img'] for _ in batch])
      targets = np.array([[_['ctr']] for _ in batch])

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
    # ep+=1

    print('\n {0} : start episode: {1}'.format(FLAGS.log_tag, epoch))

    debug=False
    if debug: print("ALERT ONLY VALIDATION")
    # ----------- train episode: update importance weights on training data
    # sumvar = run_episode('train', {}, model, ep==FLAGS.max_episodes-1 and FLAGS.update_importance_weights)    
    if not debug: sumvar = run_episode('train', {}, model)    
    
    # ----------- validate episode
    if debug: sumvar = run_episode('validation', {}, model)
    # sumvar = run_episode('validation', sumvar, model, ep==FLAGS.max_episodes-1 and FLAGS.update_importance_weights)
    if not debug: sumvar = run_episode('validation', sumvar, model)

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
    sumvar = run_episode('test', {}, model)  
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
      targets = np.array([[_['ctr']] for _ in batch])
      predictions, losses = model.predict(inputs, targets)
      print losses.keys()
      import pdb; pdb.set_trace()
      # --> loss is already mean ==> have to keep raw losses.
      hard_input=[np.array(x) for _,x in reversed(sorted(zip(list(losses['imitation_learning']), inputs.tolist())))][0]
      tools.visualize_saliency_of_output(FLAGS, model, hard_input)
      break

  if FLAGS.calculate_importance_weights:
    # tools.calculate_importance_weights(model, data.get_all_inputs('validation'), level='neuron')
    importance_weights=tools.calculate_importance_weights(model, data.get_all_inputs('train'), level='neuron')
    import pickle
    with open(FLAGS.summary_dir+FLAGS.log_tag+"/omegas",'wb') as f:
      pickle.dump(importance_weights, f)

    # import pdb; pdb.set_trace()
  # if FLAGS.visualize_deep_dream_of_output:
  #   tools.deep_dream_of_extreme_control(FLAGS, model)

  # if FLAGS.visualize_activations:
  #   tools.visualize_activations(FLAGS,model) 

  # if FLAGS.visualize_control_activation_maps and 'CAM' in FLAGS.network:
  #   tools.visualize_control_activation_maps(FLAGS,model) 

