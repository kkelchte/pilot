import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import sys, os, os.path
import shutil
import time
import random
import torch
import torch.backends.cudnn as cudnn
# from model import Model
import data
import tools
from replay_buffer import ReplayBuffer


# import tensorflow as tf

# FLAGS = None
# model = None

"""
This module scripts the procedure of running over episodes of training, validation and testing offline.
The data is collected from the data module.
"""

# FIELDS
loss_window=[]
last_loss_window_mean=0
last_loss_window_std=0
on_plateau=False

# if load data in ram keep testset in ram
testdata=[]

labels={'sc':0, 'lc': -1, 'rc': 1}



def interpret_loss_window(loss_window_mean, loss_window_std, model, x):
  """Adjust the global loss_window field, last mean and std of the plateau and whether current window is on of off a plateau.
  """
  global loss_window, last_loss_window_mean, last_loss_window_std, on_plateau

  # If peak is detected and on a plateau, put on_plateau to false.
  if on_plateau and loss_window_mean > last_loss_window_mean+last_loss_window_std:
    on_plateau=False

  # if plateau is detected and continual learning, update importance weights
  if model.FLAGS.continual_learning \
    and loss_window_mean < model.FLAGS.loss_window_mean_threshold \
    and loss_window_std < model.FLAGS.loss_window_std_threshold \
    and not on_plateau:

    last_loss_window_mean=loss_window_mean
    last_loss_window_std=loss_window_std
    on_plateau=True

    omegas_old=model.omegas[:]
    model.star_variables=[]
    gradients=tools.calculate_importance_weights(model=model,input_images=list(x))
    for pindex, p in enumerate(model.net.parameters()):
      gradients[pindex]=torch.from_numpy(gradients[pindex]).type(torch.FloatTensor).to(model.device)
      if len(omegas_old) != 0:
        model.omegas.append(1/model.count_updates*gradients[pindex]+(1-1/model.count_updates)*omegas_old[pindex])
      else:
        model.omegas.append(gradients[pindex])
      model.star_variables.append(p.data.clone().detach().to(model.device))
    model.count_updates+=1

def method(model, experience, replaybuffer, sumvar={}):
  """Learn on infinite stream of supervised data
  input: 
    image: (C,H,W) RGB image of model's size
    label: (1,A) supervised command
    returns nothing. 
  """
  global loss_window, last_loss_window_mean, last_loss_window_std, on_plateau

  image=experience['state']

  label=experience['trgt'] if 'trgt' in experience.keys() else None
  

  # annotate frame with predicted control logfolder/control_annotated
  if model.FLAGS.save_annotated_images:
    tools.save_annotated_images(image, label, model)

  # create CAM image and save in logfolder/CAM
  if model.FLAGS.save_CAM_images:
    try:
      tools.save_CAM_images(image, model, label=label)
    except Exception as e:
      print(e.args)
  if model.FLAGS.evaluate:
    return
  # save experience in buffer
  replaybuffer.add(experience)

  # if len(replaybuffer) < FLAGS.buffer_size or (len(replaybuffer) < FLAGS.min_buffer_size and FLAGS.min_buffer_size != -1): return
  if replaybuffer.size() < model.FLAGS.buffer_size or (replaybuffer.size() < model.FLAGS.min_buffer_size and model.FLAGS.min_buffer_size != -1): return

  if model.FLAGS.batch_size == -1:
    # perform a training step on data in replaybuffer 
    data=replaybuffer.get_all_data(max_batch_size=model.FLAGS.max_batch_size)
  else:
    raise(NotImplementedError("Online.py only works with batch size -1 by taking the full buffer in, current batch size is {0}".format(model.FLAGS.batch_size)))
  
  # take gradient steps
  for gs in range(model.FLAGS.gradient_steps):
    if model.FLAGS.batch_size != -1:
      data=replaybuffer.sample_batch(model.FLAGS.batch_size)
    epoch, predictions, losses, hidden_states = model.train(data['state'],data['trgt'])
    # add loss value to window
    if gs==0: 
      loss_window.append(np.mean(losses['imitation_learning']))
      if len(loss_window)>model.FLAGS.loss_window_length: del loss_window[0]

  # calculate mean and standard deviation to detect plateau or peak
  interpret_loss_window(np.mean(loss_window), np.std(loss_window), model, data['state'])
  
  # update hard buffer
  replaybuffer.update(model.FLAGS.buffer_update_rule, losses['total'], model.FLAGS.train_every_N_steps)
  
  # save some values for logging
  for k in losses.keys():
    sumvar[k]=np.mean(losses[k])
  sumvar['loss_window_means']=np.mean(loss_window)
  sumvar['loss_window_stds']=np.std(loss_window)
  sumvar['plateau_tags']=on_plateau

  # get all metrics of this episode and add them to summary variables
  tags_not_to_print=[]
  msg="epoch : {0}".format(model.epoch)
  for k in sumvar.keys(): msg="{0}, {1} : {2}".format(msg, k, sumvar[k]) if k not in tags_not_to_print else msg
  print("time: {0}, {1}".format(time.strftime('%H.%M.%S'),msg))
  f=open(model.FLAGS.summary_dir+model.FLAGS.log_tag+"/tf_log",'a')
  f.write(msg+'\n')
  f.close()
  sys.stdout.flush()
  if model.FLAGS.tensorboard:
    model.summarize(sumvar)

  # save model every now and then
  if int(model.epoch/model.FLAGS.gradient_steps) % model.FLAGS.save_every_num_epochs == 1:
    stime=time.time()
    model.save(model.FLAGS.summary_dir+model.FLAGS.log_tag, replaybuffer=replaybuffer)
    print("model saving duration with replaybuffer: {0:0.3f}".format(time.time()-stime))

def evaluate_forest_trails(model, testset):
  """Evaluate loss on test data
  """
  test_results={k:{} for k in testset[0].keys()}
  for cam in ['sc','rc','lc']:
    for run_index, run in enumerate(testset):
      for img_index in range(len(run[cam])):
        label=np.expand_dims(labels[cam], axis=-1)
        if not model.FLAGS.load_data_in_ram and len(testset)!=len(testdata):
          image=tools.load_rgb(im_file=run[cam][img_index], 
                                im_size=model.input_size, 
                                im_mode='CHW',
                                im_norm='scaled' if model.FLAGS.scaled_input else 'none',
                                im_means=model.FLAGS.scale_means,
                                im_stds=model.FLAGS.scale_stds)
        else:
          image=testdata[run_index][cam][img_index]
        _, losses, _ = model.predict(np.expand_dims(np.asarray(image), axis=0), np.array([label]))
        for k in losses.keys():
          tools.save_append(test_results[cam], k, np.mean(losses[k]))
  sumvar={}
  test_results['tot']={k:np.mean([np.mean(test_results[cam][k]) for cam in ['sc', 'rc', 'lc']]) for k in test_results['sc'].keys()}
  for cam in test_results.keys():
    for k in test_results[cam].keys():
      sumvar['test_'+cam+'_'+k]=np.mean(test_results[cam][k])
      
  if model.FLAGS.tensorboard:
    model.summarize(sumvar)
  
  msg="frame : {0}".format(model.epoch)
  for k in sumvar.keys(): msg="{0}, {1} : {2}".format(msg, k, sumvar[k])
  print("time: {0}, {1}".format(time.strftime('%H.%M.%S'),msg))
  
def evaluate(model, testset, FLAGS):
  """Evaluate loss on test data
  """
  test_results={}
  for run in testset:
    for sample_index in range(0,len(run['num_imgs']),10):
      if not FLAGS.load_data_in_ram:
        image=tools.load_rgb(im_file=os.path.join(run['name'],'RGB', '{0:010d}.jpg'.format(run['num_imgs'][sample_index])), 
                                im_size=model.input_size, 
                                im_mode='CHW',
                                im_norm='scaled' if FLAGS.scaled_input else 'none',
                                im_means=FLAGS.normalize_means,
                                im_stds=FLAGS.normalize_stds)
      else:
        image=run['imgs'][sample_index]
      label=np.expand_dims(run['controls'][sample_index],axis=-1)
      _, losses, _ = model.predict(np.expand_dims(np.asarray(image), axis=0), np.array([label]))
      for k in losses.keys():
        tools.save_append(test_results, k, np.mean(losses[k]))     
  sumvar={}
  for k in test_results:
    sumvar['test_'+k]=np.mean(test_results[k])

  msg="frame : {0}".format(model.epoch)
  for k in sumvar.keys(): msg="{0}, {1} : {2}".format(msg, k, sumvar[k])
  print("time: {0}, {1}".format(time.strftime('%H.%M.%S'),msg))
        
def run(_FLAGS, model):
  global epoch, labels, testdata
  FLAGS=_FLAGS
  start_time=time.time()

  # first see if a replaybuffer is within the my-model torch checkpoint.
  replaybuffer=tools.load_replaybuffer_from_checkpoint(FLAGS)
  if not replaybuffer: #other wise create a new.
    replaybuffer=ReplayBuffer(buffer_size=FLAGS.buffer_size, random_seed=FLAGS.random_seed, action_normalization=FLAGS.normalized_output and FLAGS.discrete)

  if FLAGS.save_annotated_images:
    if os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag+'/RGB'): shutil.rmtree(FLAGS.summary_dir+FLAGS.log_tag+'/RGB')
    os.makedirs(FLAGS.summary_dir+FLAGS.log_tag+'/RGB')

  # Load data of forest_trail_dataset
  if FLAGS.dataset == 'forest_trail_dataset':
    if FLAGS.data_root[0] != '/':  # 2. Pilot_data directory for saving data
      FLAGS.data_root=os.environ['HOME']+'/'+FLAGS.data_root
    # Data preparations for forest trail dataset
    original_dir=os.path.join(FLAGS.data_root,'forest_trail_dataset')
    
    # corresponding steering directions according to camera
    labels={'sc':0, 'lc': -FLAGS.action_bound, 'rc': FLAGS.action_bound}
    def load_trail_data(runs,subsample=1):
      dataset=[]
      for run in runs:
        # prepare run by loading image locations in a dict and clipping at shortest length
        run_dir= original_dir+'/'+run
        print(run_dir)
        images={}
        for cam in ['rc','sc','lc']:
          images[cam]=[]
          frames_dir=[run_dir+'/videos/'+cam+'/'+f for f in os.listdir(run_dir+'/videos/'+cam) if os.path.isdir(run_dir+'/videos/'+cam+'/'+f)][-1]
          for img_index in range(0,len(os.listdir(frames_dir))+10000,subsample):
            if os.path.isfile(frames_dir+'/frame'+str(img_index)+'.jpg'):
              images[cam].append(frames_dir+'/frame'+str(img_index)+'.jpg')
        # clip lengths of camera's according to shortest sequence
        shortest_length=min([len(images[cam]) for cam in images.keys()])
        for cam in images.keys(): 
          images[cam] = images[cam][:shortest_length]
          print(len(images[cam]))
        dataset.append(images)
      return dataset

    trainset=load_trail_data(["{0:03d}".format(r) for r in range(4,11)])
    # testset=load_trail_data(["{0:03d}".format(r) for r in range(11,12)],subsample=100)  
    # testset=load_trail_data(["{0:03d}".format(r) for r in range(4,9)],subsample=100)  
    testset=load_trail_data(["{0:03d}_eva".format(r) for r in range(4,11)])  
    
    if FLAGS.load_data_in_ram: # load test data in ram so evaluation goes faster
      testdata=[]
      for run in testset:
        rundata={}
        for cam in ['rc', 'lc', 'sc']:
          rundata[cam]=[]
          for img in run[cam]:
            image=tools.load_rgb(im_file=img, 
                                im_size=model.input_size, 
                                im_mode='CHW',
                                im_norm='scaled' if FLAGS.scaled_input else 'none',
                                im_means=FLAGS.normalize_means,
                                im_stds=FLAGS.normalize_stds)
            rundata[cam].append(image)
        testdata.append(rundata)
    # avoid evaluations for each replay buffer filling
    last_evaluation_epoch=0
    for run_index, images in enumerate(trainset):
      for img_index in range(len(images['lc'])):
        for cam in images.keys():
          label=np.expand_dims(labels[cam], axis=-1)
          image=tools.load_rgb(im_file=images[cam][img_index], 
                                im_size=model.input_size, 
                                im_mode='CHW',
                                im_norm='scaled' if FLAGS.scaled_input else 'none',
                                im_means=FLAGS.normalize_means,
                                im_stds=FLAGS.normalize_stds)
          experience={'state':image,'trgt':label}
          method(model, experience, replaybuffer)
          if int(model.epoch/FLAGS.gradient_steps)%100 == 50 and model.epoch != last_evaluation_epoch:
            evaluate_forest_trails(model, testset)
            last_evaluation_epoch=model.epoch
  else:
    last_evaluation_epoch=0
    data.prepare_data(FLAGS, model.input_size)
    for run in data.full_set['train']:
      concat_frames=[]
      for sample_index in range(len(run['num_imgs'])):
        if not FLAGS.load_data_in_ram:
          image=tools.load_rgb(im_file=os.path.join(run['name'],'RGB', '{0:010d}.jpg'.format(run['num_imgs'][sample_index])), 
                                  im_size=model.input_size, 
                                  im_mode='CHW',
                                  im_norm='scaled' if FLAGS.scaled_input else 'none',
                                  im_means=FLAGS.normalize_means,
                                  im_stds=FLAGS.normalize_stds)
        else:
            image=run['imgs'][sample_index]
        if '3d' in FLAGS.network or 'nfc' in FLAGS.network: 
          concat_frames.append(image)
          if len(concat_frames) < FLAGS.n_frames: continue
          if '3d' in FLAGS.network:
            image=np.concatenate(concat_frames,axis=0)
          else:
            image=np.asarray(concat_frames)
          concat_frames = concat_frames[-FLAGS.n_frames+1:] 
        label=np.expand_dims(run['controls'][sample_index],axis=-1)
        experience={'state':image,'trgt':label}
        method(model, experience, replaybuffer)

        if int(model.epoch/FLAGS.gradient_steps)%100 == 50 and model.epoch != last_evaluation_epoch:
          evaluate(model, data.full_set['validation'], FLAGS)
          last_evaluation_epoch=model.epoch
  


