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
import matplotlib.pyplot as plt

# import tensorflow as tf

FLAGS = None
model = None

"""
This module scripts the procedure of running over episodes of training, validation and testing offline.
The data is collected from the data module.
"""

# Initialize buffers
hard_buffer=[]
loss_window=[]
last_loss_window_mean=0
last_loss_window_std=0
on_plateau=False

# if load data in ram keep testset in ram
testdata=[]

labels={'sc':0, 'lc': -1, 'rc': 1}

def method(model, image, label):
  """Learn on infinite stream of supervised data
  input: 
    image: (C,H,W) RGB image of model's size
    label: (1,A) supervised command
    returns nothing. 
  """
  global hard_buffer, loss_window, last_loss_window_mean, last_loss_window_std, on_plateau
  sumvar={}

  # save experience in buffer
  hard_buffer.append({'state':image,'trgt':label})
  
  if len(hard_buffer) < FLAGS.batch_size: return
  
  x=np.array([_['state'] for _ in hard_buffer])
  y=np.array([_['trgt'] for _ in hard_buffer])
  # take training steps
  for gs in range(FLAGS.gradient_steps):
    epoch, predictions, losses, hidden_states = model.train(x,y)
    # add loss value to window
    if gs==0: 
      loss_window.append(np.mean(losses['imitation_learning']))
      if len(loss_window)>FLAGS.loss_window_length: del loss_window[0]
  
  # calculate mean and standard deviation to detect plateau or peak        
  loss_window_mean=np.mean(loss_window)
  loss_window_std=np.std(loss_window)
  
  if on_plateau and loss_window_mean > last_loss_window_mean+last_loss_window_std:
    on_plateau=False
    # print("peak detected")

  if FLAGS.continual_learning \
    and loss_window_mean < FLAGS.loss_window_mean_threshold \
    and loss_window_std < FLAGS.loss_window_std_threshold \
    and not on_plateau:
    last_loss_window_mean=loss_window_mean
    last_loss_window_std=loss_window_std
    on_plateau=True
    print("calculate importance weights")
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
  
  # update hard buffer
  hard_loss=losses['total']
  try:
    sorted_inputs=[np.asarray(lx) for _,lx in reversed(sorted(zip(hard_loss.tolist(),x),key= lambda f:f[0]))]
    sorted_targets=[ly for _,ly in reversed(sorted(zip(hard_loss.tolist(),y),key= lambda f:f[0]))]
  except:
    print("Failed to update hard_buffer.")
  hard_buffer=[{'state':sorted_inputs[i],'trgt':sorted_targets[i]} for i in range(min(FLAGS.buffer_size,len(sorted_inputs)))]
  
  # save some values for logging
  for k in losses.keys():
    sumvar[k]=np.mean(losses[k])
  sumvar['loss_window_means']=loss_window_mean
  sumvar['loss_window_stds']=loss_window_std
  sumvar['plateau_tags']=on_plateau

  # annotate frame with predicted control
  if FLAGS.save_annotated_images:
    ctr,_,_=model.predict(np.expand_dims(image,axis=0))
    plt.imshow(image.transpose(1,2,0).astype(np.float32)+0.5)
    plt.plot((image.shape[1]/2,image.shape[1]/2+ctr[0]*50), (image.shape[2]/2,image.shape[2]/2), linewidth=5, markersize=12,color='b')
    plt.plot((image.shape[1]/2,image.shape[1]/2+label*50), (image.shape[2]/2+10,image.shape[2]/2+10), linewidth=5, markersize=12,color='g')
    plt.axis('off')
    plt.text(x=5,y=image.shape[2]-10,s='Expert',color='g')
    plt.text(x=5,y=image.shape[2]-20,s='Student',color='b')
    plt.savefig(FLAGS.summary_dir+FLAGS.log_tag+'/RGB/{0:010d}.jpg'.format(len(imitation_loss)))
  
  # get all metrics of this episode and add them to var
  tags_not_to_print=[]
  msg="frame : {0}".format(model.epoch)
  for k in sumvar.keys(): msg="{0}, {1} : {2}".format(msg, k, sumvar[k]) if k not in tags_not_to_print else msg
  print("time: {0}, {1}".format(time.strftime('%H.%M.%S'),msg))
  f=open(FLAGS.summary_dir+FLAGS.log_tag+"/tf_log",'a')
  f.write(msg+'\n')
  f.close()
  sys.stdout.flush()
  if FLAGS.tensorboard:
    model.summarize(sumvar)
  if model.epoch % 100 == 0:
    model.save(FLAGS.summary_dir+FLAGS.log_tag)


def evaluate(model, testset):
  """Evaluate loss on test data
  """
  test_results={k:{} for k in testset[0].keys()}
  for cam in ['sc','rc','lc']:
    for run_index, run in enumerate(testset):
      for img_index in range(len(run[cam])):
        label=np.expand_dims(labels[cam], axis=-1)
        if not FLAGS.load_data_in_ram and len(testset)!=len(testdata):
          image=tools.load_rgb(im_file=run[cam][img_index], 
                                im_size=model.input_size, 
                                im_mode='CHW',
                                im_norm='shifted' if FLAGS.shifted_input else 'none',
                                im_means=FLAGS.scale_means,
                                im_stds=FLAGS.scale_stds)
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
      
  if FLAGS.tensorboard:
    model.summarize(sumvar)
  
  msg="frame : {0}".format(model.epoch)
  for k in sumvar.keys(): msg="{0}, {1} : {2}".format(msg, k, sumvar[k])
  print("time: {0}, {1}".format(time.strftime('%H.%M.%S'),msg))
  



def run(_FLAGS, model):
  global FLAGS, epoch, labels, testdata
  FLAGS=_FLAGS
  start_time=time.time()

  if FLAGS.save_annotated_images:
    if os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag+'/RGB'): shutil.rmtree(FLAGS.summary_dir+FLAGS.log_tag+'/RGB')
    os.makedirs(FLAGS.summary_dir+FLAGS.log_tag+'/RGB')

  # Load data
  if FLAGS.dataset == 'forest_trail_dataset':
    if FLAGS.data_root[0] != '/':  # 2. Pilot_data directory for saving data
      FLAGS.data_root=os.environ['HOME']+'/'+FLAGS.data_root
    # Data preparations for forest trail dataset
    original_dir=FLAGS.data_root+'forest_trail_dataset'
    
    # corresponding steering directions according to camera
    labels={'sc':0, 'lc': -FLAGS.action_bound, 'rc': FLAGS.action_bound}
    def load_trail_data(runs,subsample=1):
      dataset=[]
      for run in runs:
        # prepare run by loading image locations in a dict and clipping at shortest length
        run_dir= original_dir+'/'+run
        print run_dir
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
                                im_norm='shifted' if FLAGS.shifted_input else 'none',
                                im_means=FLAGS.scale_means,
                                im_stds=FLAGS.scale_stds)
            rundata[cam].append(image)
        testdata.append(rundata)
    
    for run_index, images in enumerate(trainset):
      for img_index in range(len(images['lc'])):
        for cam in images.keys():
          label=np.expand_dims(labels[cam], axis=-1)
          image=tools.load_rgb(im_file=images[cam][img_index], 
                                im_size=model.input_size, 
                                im_mode='CHW',
                                im_norm='shifted' if FLAGS.shifted_input else 'none',
                                im_means=FLAGS.scale_means,
                                im_stds=FLAGS.scale_stds)
          method(model, image, label)
          if model.epoch%100 == 50:
            evaluate(model, testset)
  else:
    data.prepare_data(FLAGS, model.input_size)
    for run in data.full_set['train']:
      for sample_index in range(len(run['num_imgs'])):
        if not FLAGS.load_data_in_ram:
          image=tools.load_rgb(im_file=os.path.join(run['name'],'RGB', '{0:010d}.jpg'.format(run['num_imgs'][sample_index])), 
                                  im_size=model.input_size, 
                                  im_mode='CHW',
                                  im_norm='shifted' if FLAGS.shifted_input else 'none',
                                  im_means=FLAGS.scale_means,
                                  im_stds=FLAGS.scale_stds)
        else:
            image=run['imgs'][sample_index]
        label=np.expand_dims(run['controls'][sample_index],axis=-1)
        method(model, image, label)
  


