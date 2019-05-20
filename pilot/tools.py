#!/usr/bin/python
import model
#from lxml import etree as ET
import xml.etree.cElementTree as ET

import numpy as np
# import tensorflow as tf
import torch

import os,sys,time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


import skimage.transform as sm
import skimage.io as sio

# FLAGS = tf.app.flags.FLAGS

""" 
Practical tools used by main, data and model
"""
def print_dur(duration_time):
  duration = duration_time #(time.time()-start_time)
  m, s = divmod(duration, 60)
  h, m = divmod(m, 60)
  return "time: %dh:%02dm:%02ds" % (h, m, s)

def save_append(mydict, mykey, myval):
  """append the value in the dict with corresponding key.
  if key error, than create new entry in dict.
  """
  if mykey in mydict.keys():
    if hasattr(myval, '__contains__') and myval.shape != ():
      mydict[mykey].extend(list(myval))
    else:
      mydict[mykey].append(myval)
  else:
    if hasattr(myval, '__contains__') and myval.shape != ():
      mydict[mykey]=list(myval)
    else:
      mydict[mykey]=[myval]
  return mydict

# ===========================
#   Save settings
# ===========================
def save_config(FLAGS, logfolder, file_name = "configuration"):
  """
  save all the FLAG values in a config file / xml file
  """
  print("[tools] Save configuration to: {}".format(logfolder))
  root = ET.Element("conf")
  flg = ET.SubElement(root, "flags")
  
  flags_dict=FLAGS.__dict__
  for f in sorted(flags_dict.keys()):
    # print f, flags_dict[f]
    e = ET.SubElement(flg, f, name=f) 
    e.text = str(flags_dict[f])
    e.tail = "\n  "
  tree = ET.ElementTree(root)
  tree.write(os.path.join(logfolder,file_name+".xml"), encoding="us-ascii", xml_declaration=True, method="xml")

# ===========================
#   Load settings
# ===========================
def load_config(FLAGS, modelfolder, file_name = "configuration"):
  """
  save all the FLAG values in a config file / xml file
  """
  print("[tools] Load configuration from: ", modelfolder)
  tree = ET.parse(os.path.join(modelfolder,file_name+".xml"))
  boollist=['auxiliary_depth', 'discrete','shifted_input','scaled_input','skew_input']
  intlist=['n_frames', 'num_outputs']
  floatlist=['depth_multiplier','speed','action_bound','turn_speed']
  stringlist=['network', 'data_format']
  for child in tree.getroot().find('flags'):
    try :
      if child.attrib['name'] in boollist:
        FLAGS.__setattr__(child.attrib['name'], child.text=='True')
        print 'set:', child.attrib['name'], child.text=='True'
      elif child.attrib['name'] in intlist:
        FLAGS.__setattr__(child.attrib['name'], int(child.text))
        print 'set:', child.attrib['name'], int(child.text)
      elif child.attrib['name'] in floatlist:
        FLAGS.__setattr__(child.attrib['name'], float(child.text))
        print 'set:', child.attrib['name'], float(child.text)
      elif child.attrib['name'] in stringlist:
        # Temporary hack to load models from doshico
        # if not FLAGS.network != 'mobile_nfc': 
        FLAGS.__setattr__(child.attrib['name'], str(child.text))
        print 'set:', child.attrib['name'], str(child.text)
      # Temporary hack to load models from doshico
      # elif child.attrib['name'] == 'n_fc':
      #   FLAGS.network='mobile_nfc'
      #   print 'set: network to mobile_nfc'
    except : 
      print 'couldnt set:', child.attrib['name'], child.text
      pass

  return FLAGS


def load_replaybuffer_from_checkpoint(FLAGS):
  """Go to checkpoint directory, look for my-model file, load it in with torch, return replaybuffer
  """
  replaybuffer=None
  if os.path.isfile(FLAGS.checkpoint_path+'/my-model'):
    try:
      if not 'gpu' in FLAGS.device:
        checkpoint=torch.load(FLAGS.checkpoint_path+'/my-model', map_location='cpu')
      else:  
        checkpoint=torch.load(FLAGS.checkpoint_path+'/my-model')
      replaybuffer=checkpoint['replaybuffer']
    except Exception as e:
      print("[Tools]: failed to load replay buffer from {0} due to {1}".format(FLAGS.checkpoint_path, e.message))
    else:
      print("[Tools]: successfully loaded replaybuffer from {0}".format(FLAGS.checkpoint_path))
  return replaybuffer

# ===========================
#   Load rgb image
# ===========================
def load_rgb(im_file="",im_object=[],im_size=[3,128,128], im_mode='CHW', im_norm='none', im_means=[0,0,0], im_stds=[1,1,1]):
  """Load an RGB image file and return a numpy array of type float16 with values between [0:1]
  args:
  im_file: absolute path to file
  im_size: list output image size, obviously in de corresponding image mode
  im_mode: CHW for channel-first and HWC for channel-last image
  im_norm: 'none', 'shifted' ~ move from 0:1 to -0.5:0.5, 'scale' for each channel (x-mean)/std
  return:
  numpy array
  """
  if im_file!="":
    if not os.path.isfile(im_file):
      raise IOError("File not found: {0}".format(im_file))
    img = sio.imread(im_file)
  elif im_object!=[]:
    img = im_object
  else:
    raise IOError("tools: load_rgb: no im_file or im_object provided.")
  # for pytorch: swap channels from last to first dimension
  if im_mode != 'HWC':
    img = np.swapaxes(img,1,2)
    img = np.swapaxes(img,0,1)
    scale_height = int(np.floor(img.shape[1]/im_size[1]))
    scale_width = int(np.floor(img.shape[2]/im_size[2]))
    img = img[:,::scale_height,::scale_width]
    img=sm.resize(img,(3,im_size[1],im_size[2]),mode='constant').astype(np.float16)
    if im_norm=='scaled':
      for i in range(3): 
        img[i,:,:]-=im_means[i]
        img[i,:,:]/=im_stds[i]
  else:
    scale_height = int(np.floor(img.shape[0]/im_size[0]))
    scale_width = int(np.floor(img.shape[1]/im_size[1]))
    img = img[::scale_height,::scale_width,:]
    img=sm.resize(img,im_size,mode='constant').astype(np.float16)
    if im_norm=='scaled':
      for i in range(3): 
        img[:,:,i]-=im_means[i]
        img[:,:,i]/=im_stds[i]

  if im_norm=='shifted':
    img -= 0.5

  if im_norm=='skewinput':
    img *= 255


  return img

# ===========================
#   Load Depth image
# ===========================
def load_depth(im_file="",im_size=[128,128], im_norm='none', im_mean=0, im_std=1, min_depth=0, max_depth=5):
  """Load an depth image file and return a numpy array of type float16 with values between [0:1]
  args:
  im_file: absolute path to file
  im_size: list output image size, obviously in de corresponding image mode
  im_norm: 'none', 'shifted' ~ move from 0:1 to -0.5:0.5, 'scale' for each channel (x-mean)/std
  im_mean: float for depth mean
  im_std: float for depth std
  min_depth: float in m for minimum trustable depth (d<minimum -> minimum)
  max_depth: float in m for maximum trustable depth (d>maximum -> maximum)
  return:
  numpy array
  """
  if not os.path.isfile(im_file):
    raise IOError("File not found: {0}".format(im_file))
  img = sio.imread(im_file)
  # for pytorch: swap channels from last to first dimension
  scale_height = int(np.floor(img.shape[0]/im_size[0]))
  scale_width = int(np.floor(img.shape[1]/im_size[1]))
  img = img[::scale_height,::scale_width]
  img=sm.resize(img,im_size,order=1,mode='constant',preserve_range=True).astype(np.float16)
  img[img<10]=0
  # scale to expected range between 0 and 5m
  img=img * (1/255. * 5.)
  # clip to minimum and maximum depth
  img=np.minimum(np.maximum(img, min_depth),max_depth)
  # scale to range 0:1
  img/=5.
  if im_norm=='scaled':
    for i in range(3): 
      img[:,:,i]-=im_means[i]
      img[:,:,i]/=im_stds[i]
  if im_norm=='shifted':
    img -= 0.5
  return img

# ==============================
#   Save annotated image
# ==============================
def save_annotated_images(image, label, model):
  """Save annotated image in logfolder/control_annotated 
  """
  
  if not os.path.isdir(model.FLAGS.summary_dir+model.FLAGS.log_tag+'/control_annotated'): 
    os.makedirs(model.FLAGS.summary_dir+model.FLAGS.log_tag+'/control_annotated')
  ctr,_,_=model.predict(np.expand_dims(image,axis=0))
  plt.cla()
  image_postprocess = image.transpose(1,2,0).astype(np.float32)+0.5 if model.FLAGS.shifted_input else image.transpose(1,2,0).astype(np.float32)
  if '3d' in model.FLAGS.network: image_postprocess = image_postprocess[:,:,-3:]
  plt.imshow(image_postprocess)
  plt.plot((image.shape[1]/2,image.shape[1]/2-ctr[0]*50), (image.shape[2]/2,image.shape[2]/2), linewidth=5, markersize=12,color='b')
  plt.plot((image.shape[1]/2,image.shape[1]/2-label*50), (image.shape[2]/2+10,image.shape[2]/2+10), linewidth=5, markersize=12,color='g')
  plt.axis('off')
  plt.text(x=5,y=image.shape[2]-10,s='Expert',color='g')
  plt.text(x=5,y=image.shape[2]-20,s='Student',color='b')
  
  image_index=len(os.listdir(model.FLAGS.summary_dir+model.FLAGS.log_tag+'/control_annotated'))
  plt.savefig(model.FLAGS.summary_dir+model.FLAGS.log_tag+'/control_annotated/{0:010d}.jpg'.format(image_index))


def save_CAM_images(image, model, label=None):
  """Save a CAM activation map of current image and save in logfolder/CAM
  """
  # import gradcam
  from PIL import Image
  # from misc_functions import get_example_params, save_class_activation_images, apply_colormap_on_image
  import matplotlib.cm as mpl_color_map
  import copy

  if not os.path.isdir(model.FLAGS.summary_dir+model.FLAGS.log_tag+'/CAM'): 
    os.makedirs(model.FLAGS.summary_dir+model.FLAGS.log_tag+'/CAM')
  
  ctr,_,_ = model.predict(np.expand_dims(image,0))
  grad_cam = GradCam(model.net.network.to(torch.device('cpu')), target_layer=len(model.net.network.features)-1)

  target_classes = [0]
  if model.FLAGS.discrete:
    # in discrete case take label output if provided otherwise loop over all options.
    target_classes=list(range(model.FLAGS.action_quantity)) if not label else [model.FLAGS.continuous_to_bins(label)-1] 
  


  fig, ax=plt.subplots(len(target_classes), 2, squeeze=False)
  # Generate cam mask
  for class_index, target_class in enumerate(target_classes):
    
    prep_img=torch.from_numpy(np.expand_dims(image,0)).type(torch.float32)
    # get CAM activation
    cam = grad_cam.generate_cam(prep_img, target_class)
    # specify hsv color map from matplotlib
    color_map = mpl_color_map.get_cmap('hsv')
    no_trans_heatmap = color_map(cam)
        
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    
    # post process input image
    image_postprocess = image.transpose(1,2,0)+0.5 if model.FLAGS.shifted_input else image.transpose(1,2,0)
    if '3d' in model.FLAGS.network: image_postprocess = image_postprocess[:,:,-3:]
    ax[class_index,0].imshow(image_postprocess.astype(np.float32))
    # add controls
    ax[class_index,0].plot((image_postprocess.shape[0]/2,image_postprocess.shape[0]/2), (image_postprocess.shape[1]/2-5,image_postprocess.shape[1]/2+15), linewidth=3, markersize=12,color='w')
    ax[class_index,0].plot((image_postprocess.shape[0]/2,image_postprocess.shape[0]/2-ctr[0]*50), (image_postprocess.shape[1]/2,image_postprocess.shape[1]/2), linewidth=5, markersize=12,color='b')
    ax[class_index,0].plot((image_postprocess.shape[0]/2,image_postprocess.shape[0]/2-label*50), (image_postprocess.shape[1]/2+10,image_postprocess.shape[1]/2+10), linewidth=5, markersize=12,color='g')
    ax[class_index,0].text(x=5,y=image_postprocess.shape[1]-10,s='Expert',color='g')
    ax[class_index,0].text(x=5,y=image_postprocess.shape[1]-20,s='Student',color='b')
    ax[class_index,0].axis('off')  

    # Apply heatmap on image
    original_image = Image.fromarray(((image_postprocess)*255).astype(np.uint8))
    heatmap_on_image = Image.new("RGBA", original_image.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, original_image.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    
    image_index=len(os.listdir(model.FLAGS.summary_dir+model.FLAGS.log_tag+'/CAM'))
    
    ax[class_index,1].imshow(heatmap_on_image)

    # if label==None:
    #   heatmap_on_image.save(model.FLAGS.summary_dir+model.FLAGS.log_tag+'/CAM/{0:010d}_{1}.png'.format(image_index, target_class), 'PNG')
    # else:
    #   # annotate control and save with pyplot
    #   plt.cla()
    #   plt.imshow(heatmap_on_image)
    ax[class_index,1].plot((heatmap_on_image.size[0]/2,heatmap_on_image.size[0]/2), (heatmap_on_image.size[1]/2-5,heatmap_on_image.size[1]/2+15), linewidth=3, markersize=12,color='w')
    ax[class_index,1].plot((heatmap_on_image.size[0]/2,heatmap_on_image.size[0]/2-ctr[0]*50), (heatmap_on_image.size[1]/2,heatmap_on_image.size[1]/2), linewidth=5, markersize=12,color='b')
    ax[class_index,1].plot((heatmap_on_image.size[0]/2,heatmap_on_image.size[0]/2-label*50), (heatmap_on_image.size[1]/2+10,heatmap_on_image.size[1]/2+10), linewidth=5, markersize=12,color='g')
    ax[class_index,1].axis('off')
    ax[class_index,1].text(x=5,y=heatmap_on_image.size[1]-10,s='Expert',color='g')
    ax[class_index,1].text(x=5,y=heatmap_on_image.size[1]-20,s='Student',color='b')

  model.net.network.to(model.device)
  fig.savefig(model.FLAGS.summary_dir+model.FLAGS.log_tag+'/CAM/{0:010d}.png'.format(image_index), bbox_inches='tight')
  plt.close()
  plt.cla()
  plt.clf()

# ==============================
#   Obtain Hidden State of LSTM 
# ==============================
def get_hidden_state(input_images, model, device='cpu', astype='tensor'):
  """
  A recurrent model is evaluated over the input images and the final hidden output and state is returned.
  args:
  input_images: ndarray of shape [T,C,H,W]
  model:  model with LSTM net
  device: 'cpu' or 'gpu'
  returns tuple of torch tensors for cell and hidden state
  EXTENSION: add device option...
  """
  # device = torch.device("cuda:0" if torch.cuda.is_available() and device=='gpu' else "cpu")
  device = torch.device("cuda:0")
  if not model.net.rnn: raise(ValueError("Network does not contain rnn part."))
  # move model to device:
  # stime=time.time()
  # print("move model: {0}".format(time.time()-stime))
  h_t,c_t = model.net.get_init_state(1)
  # print("[tools] Obtaining hidden state after image sequence with length {0}".format(len(input_images)))
  # for index in range(len(input_images)):
  #   inputs=(torch.from_numpy(np.expand_dims(np.expand_dims(input_images[index],0),0)).type(torch.FloatTensor).to(device), 
  #           (h_t.to(device), c_t.to(device)))
  #   outputs, (h_t, c_t)=model.net.forward(inputs)
  # print input_images.shape
  # import pdb; pdb.set_trace()
  if len(input_images) != 0:
    # model.net.to(device)
    inputs=(torch.from_numpy(np.expand_dims(input_images,0)).type(torch.FloatTensor).to(device),(h_t.to(device), c_t.to(device)))
    outputs, (h_t,c_t) = model.net.forward(inputs)
    # model.net.to(model.device)
  
  return (h_t.detach().cpu(), c_t.detach().cpu()) if astype=='tensor' else (h_t.detach().cpu().numpy(), c_t.detach().cpu().numpy())

# ===========================
#   Visualization Techniques
# ===========================
def visualize_saliency_of_output(FLAGS, model, input_images=[], filter_pos=-1, cnn_layer=-1):
  """
  Extract the salience map of the output control node(s) for a predefined set of visualization images.
  You can define the input_images as a list, if nothing is defined it uses the predefined images.
  It saves the images in FLAGS.summary_dir+FLAGS.log_tag+'/saliency_maps/....png'.
  FLAGS: the settings defined in main.py
  model: the loaded pilot-model object
  input_images: one or more images in an numpy array
  """
  # Extraction of guided backpropagation saliency maps from output  
  # Create saliency map with guided backpropagation
  import matplotlib as mpl
  mpl.use('Agg')
  import matplotlib.pyplot as plt

  inputs=torch.from_numpy(np.expand_dims(input_images[0],0)).type(torch.FloatTensor)
  inputs.requires_grad=True  
  if filter_pos != -1 or cnn_layer != -1:
    raise NotImplementedError('cnn layer and filter position can not be defined yet')
  else:
    from guided_backprop import GuidedBackprop
    BP = GuidedBackprop(model.net.network)
    target = 1 if FLAGS.discrete else 0
    gradient = BP.generate_gradients(inputs, target)
    gradient = gradient - gradient.min()
    gradient /= gradient.max()

    plt.imshow(gradient.transpose(1,2,0))
    plt.savefig(FLAGS.summary_dir+FLAGS.log_tag+'/saliency_maps.jpg',bbox_inches='tight')


# ==============================
#   Calculate importance weights
# ==============================
def calculate_importance_weights(model, input_images=[], level='neuron'):
  """
  Importance weights are calculated in the same way as the memory away synapsys for the model provided.
  For each image in input_images, the absolute value of the gradients are calculated 
  for each part of the network with respect to the input and averaged over the images.
  The level tag defines on which level of specificity the importance weight should be calculated.
  Options are: 'neuron'(default), 'filter', 'layer'

  The importance weights are solely estimated for convolutional and linear operations,
  not for batch normalization.

  Returns:
  a list of importance weights in the shape corresponding to the level of specificity:
  - neuron: same shape as the parameter (ChannelsxSizexSizexHiddenUnits)
  - filter: 1D array with length HiddenUnits
  - layer: 1 integer for each layer
  """
  print("[tools] calculate_importance_weights")
  # collect importance / gradients in list
  gradients=[0. for p in model.net.parameters()]
  stime=time.time()
  hidden_states=()
  model.net.zero_grad()
  for img_index in range(len(input_images)-model.FLAGS.n_frames): #loop over input images
    if img_index%100==0: print img_index
    # ensure no gradients are still in the network
    if not 'LSTM' in model.FLAGS.network:
      model.net.zero_grad()
    # adjust input for nfc, 3dcnn, LSTM
    if '3d' in model.FLAGS.network:
      imgs=input_images[img_index:img_index+model.FLAGS.n_frames]
      img=np.concatenate(imgs, axis=0)
    elif 'nfc' in model.FLAGS.network:
      img=np.asarray(input_images[img_index:img_index+model.FLAGS.n_frames])
    elif 'LSTM' in model.FLAGS.network:
      img=np.asarray(input_images[img_index:img_index+1])
    else:
      img=input_images[img_index]
    
    img=np.expand_dims(img,0)
    inputs=torch.from_numpy(img).type(torch.float32).to(model.device)

    if not 'LSTM' in model.FLAGS.network:
      # forward pass of one image through the network
      y_pred=model.net(inputs)
      # backward pass from the 2-norm of the output
      torch.norm(y_pred, 2, dim=1).backward()    
    else:
      if len(hidden_states) != 0:
        h_t, c_t = (hidden_states[0],hidden_states[1])
      else:
        h_t, c_t = get_hidden_state([],model) 
      inputs=(inputs,(h_t.to(model.device),c_t.to(model.device)))
      y_pred, hidden_states=model.net(inputs)
      torch.norm(y_pred, 2, dim=-1).backward(retain_graph=True)    
    
    for pindex, p in enumerate(model.net.parameters()):
      try:
        g=p.grad.data.clone().detach().cpu().numpy()
        gradients[pindex]+=np.abs(g)/len(input_images)
      except Exception as e:
        print(e.message)
        pass

  # # In one track for time considerations:
  # # ensure no gradients are still in the network
  # model.net.zero_grad()
  # # forward pass of one image through the network
  # y_pred=model.net(torch.from_numpy(np.asarray(input_images)).type(torch.float32).to(model.device))
  # # backward pass from the 2-norm of the output
  # torch.sum(torch.norm(y_pred,2,dim=1)).backward()
  # for pindex, p in enumerate(model.net.parameters()):
  #     g=p.grad.data.clone().detach().cpu().numpy()
  #     gradients[pindex]+=np.abs(g)/len(input_images)

  print("[tools] duration {0}".format(time.time()-stime))
  if level == 'neuron':
    return gradients
  elif level == 'filter':
    raise NotImplementedError
  else:
    raise NotImplementedError



def visualize_importance_weights(importance_weights, log_folder):
  """
  plot for each layer the percentage of non-zero importance weights ~ 'occupied space'
  if histogram: plot a histogram over each layer's weights.
  Note that importance weights of biases is not taken into account.
  """
  # import matplotlib as mpl
  # mpl.use('Agg')
  import matplotlib.pyplot as plt

  freespace=[]
  occupied=[]
  for index, iw in enumerate(importance_weights):
      # ignore the biases
      if isinstance(iw, float) or len(iw.shape)==1: continue
      iw=iw.flatten()
      assert(len(iw[iw==0])+len(iw[iw!=0])==len(iw))
      freespace.append(float(len(iw[iw==0]))/len(iw))
      occupied.append(float(len(iw[iw!=0]))/len(iw))
  

  plt.bar(range(len(occupied)),100)
  plt.bar(range(len(occupied)),[o*100 for o in occupied])
  # for i,v in enumerate(occupied):
  #     plt.text(i-0.25, 5, "{0:d}%".format(int(v*100)))
  plt.xlabel('Layers')
  plt.ylabel('Proportion')
  plt.tight_layout()
  plt.savefig(log_folder+'/occupancy.png')

  plt.cla()
  plt.bar(range(len(freespace)),100)
  plt.bar(range(len(freespace)),[o*100 for o in freespace])
  for i,v in enumerate(freespace):
      plt.text(i-0.25, 5, "{0:d}%".format(int(v*100)))
  plt.xlabel('Layers')
  plt.ylabel('Proportion')
  plt.tight_layout()
  plt.savefig(log_folder+'/freespace.png')

  if True:
    # histogram for each layer
    num_layers=sum([1 for iw in importance_weights if not isinstance(iw,float) and len(iw.shape) > 1])
    f, axes = plt.subplots(num_layers, 1, figsize=(5,3*num_layers), sharex=True)
    index=0
    for iw in importance_weights:
      try:
        if not isinstance(iw,int) and len(iw.shape)==1: continue
        axes[index].set_title('Layer {0}'.format(index))
        axes[index].hist(iw.flatten(), bins=50)
        index+=1
      except:
        pass

    plt.tight_layout()
    plt.savefig(log_folder+'/histogram_importance_weights.png')





class GradCam():
  """
      Based on: Utku Ozbulak - github.com/utkuozbulak
      Produces class activation map
  """
  def __init__(self, model, target_layer):
    self.model = model
    self.model.eval()
    self.target_layer=target_layer

  def save_gradient(self, grad):
    self.gradients = grad

  def forward_pass_on_convolutions(self, x):
    """
        Does a forward pass on convolutions, hooks the function at given layer
    """
    conv_output = None
    for module_pos, module in self.model.features._modules.items():
      x = module(x)  # Forward
      if int(module_pos) == self.target_layer:
        x.register_hook(self.save_gradient)
        conv_output = x  # Save the convolution output on that layer
    return conv_output, x

  def forward_pass(self, x):
    """
        Does a full forward pass on the model
    """
    # Forward pass on the convolutions
    conv_output, x = self.forward_pass_on_convolutions(x)
    x = x.view(x.size(0), -1)  # Flatten
    # Forward pass on the classifier
    x = self.model.classifier(x)
    return conv_output, x


  def generate_cam(self, input_image, target_class=None):
    from PIL import Image
    # Full forward pass
    # conv_output is the output of convolutions at specified layer
    # model_output is the final output of the model (1, 1000)
    conv_output, model_output = self.forward_pass(input_image)
    if target_class is None:
      target_class = np.argmax(model_output.data.numpy())
    # Target for backprop
    one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
    one_hot_output[0][target_class] = 1
    # Zero grads
    self.model.features.zero_grad()
    self.model.classifier.zero_grad()
    # Backward pass with specified target
    model_output.backward(gradient=one_hot_output, retain_graph=True)
    # Get hooked gradients
    guided_gradients = self.gradients.data.numpy()[0]
    # Get convolution outputs
    target = conv_output.data.numpy()[0]
    # Get weights from gradients
    
    # ADJUSTMENT 1
    # weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
    weights = np.sum(np.abs(guided_gradients),axis=(1,2))/(guided_gradients.shape[1]*guided_gradients.shape[2])
    
    # print('weights: ',np.amin(weights), np.amax(weights))
    # Create empty numpy array for cam
    cam = np.zeros(target.shape[1:], dtype=np.float32)
    # cam = np.ones(target.shape[1:], dtype=np.float32)

    # use only the top 10% most important feature maps to create visualization
    threshold=np.percentile(np.abs(weights), 10)
    
    # Multiply each weight with its conv output and then, sum
    for i, w in enumerate(weights):
      # ADJUSTMENT 2
      # if w >= threshold: cam += w*(10*guided_gradients[i] + target[i, :, :])
      if w >= threshold: cam += w*np.sign(guided_gradients[i])*target[i, :, :]
      # if w >= threshold: cam += w * target[i, :, :]
      # cam += w * target[i, :, :]
      # print('adding feature map ',i, np.amin(cam), np.amax(cam))
    # import pdb; pdb.set_trace()
    # ADJUSTMENT 3
    # cam = np.maximum(cam, 0)
    cam = np.tanh(cam)
    # print( np.amin(cam), np.amax(cam))
    
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                   input_image.shape[3]), Image.ANTIALIAS))
    # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
    # supports resizing numpy matrices, however, when I moved the repository to PIL, this
    # option is out of the window. So, in order to use resizing with ANTIALIAS feature of PIL,
    # I briefly convert matrix to PIL image and then back.
    # If there is a more beautiful way, send a PR.
    return cam



# def get_endpoint_activations(inputs, model):
# 	'''Run forward through the network for this batch and return all activations
# 	of all intermediate endpoints
# 	[not used]
# 	'''
# 	tensors = [ model.endpoints[ep] for ep in model.endpoints]
# 	activations = model.sess.run(tensors, feed_dict={model.inputs:inputs})
# 	return [ a.reshape(-1,1) for a in activations]

# def fig2buf(fig):
#   """
#   Convert a plt fig to a numpy buffer
#   """
#   # draw the renderer
#   fig.canvas.draw()

#   # Get the RGBA buffer from the figure
#   w,h = fig.canvas.get_width_height()
#   buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
#   buf.shape = (h, w, 4)
  
#   # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#   buf = np.roll(buf, 3, axis = 2 )
#   buf = buf[0::1,0::1] #slice to make image 4x smaller and use only the R channel of RGBA
#   buf = buf[0::1,0::1, 0:3] #slice to make image 4x smaller and use only the R channel of RGBA
#   return buf

# def plot_depth(inputs, depth_targets, model):
#   '''plot depth predictions and return np array as floating image'''
#   depths = model.sess.run(model.pred_depth, feed_dict={model.inputs: inputs})
#   n=3
#   fig = plt.figure(1, figsize=(5,5))
#   fig.suptitle('depth predictions', fontsize=20)
#   for i in range(n):
#     plt.axis('off') 
#     plt.subplot(n, 3, 1+3*i)
#     if FLAGS.n_fc: plt.imshow(inputs[i][:,:,0+3*(FLAGS.n_frames-1):]*1/255.)
#     else : plt.imshow(inputs[i][:,:,0:3]*1/255.)
#     plt.axis('off') 
#     plt.subplot(n, 3, 2+3*i)
#     plt.imshow(depths[i]*1/5.)
#     plt.axis('off') 
#     if len(depth_targets)!=0:
#       plt.subplot(n, 3, 3+3*i)
#       plt.imshow(depth_targets[i]*1/5.)
#       plt.axis('off')
#   buf=fig2buf(fig)
#   return np.asarray([buf])


# """
# Plot activations
# """

# def load_images(file_names, im_size):
#   """
#   Load a series of images defined in a list of full paths.
#   Returns a numpy array of preprocessed images.
#   """
#   import skimage.io as sio
#   import skimage.transform as sm

#   # load images in numpy array inputs
#   inputs=[]
#   for img_file in file_names:
#       img = sio.imread(img_file)
#       scale_height = int(np.floor(img.shape[0]/im_size[0]))
#       scale_width = int(np.floor(img.shape[1]/im_size[1]))
#       img = sm.resize(img[::scale_height,::scale_width],im_size,mode='constant').astype(float)
#       inputs.append(img)
#   return np.asarray(inputs)

# def deprocess_image(x, one_channel=False):
#     # normalize tensor: center on 0., ensure std is 0.1
#     x -= x.mean()
#     x /= (x.std() + 1e-5)
#     x *= 0.1

#     # clip to [0, 1]
#     x += 0.5
#     x = np.clip(x, 0, 1)

#     # convert to RGB array
#     x *= 1
#     #     x = x.transpose((1, 2, 0))
#     x = np.clip(x, 0, 1).astype('float')

#     # make it to one channel by taking max over 3 channels
#     if one_channel: x = np.amax(x,axis=2)
    
#     return x

# def matplotlibprove(x):
#   """
#   bg: matploblit imshow can only images of shape (n,m) (n,m,3) or (n,m,4)
#   if your image is (n,m,1), change it to (n,m)
#   x: numpy nd array
#   """
#   if x.shape[-1] == 1:
#     if np.ndim(x) == 3:
#       x=x[:,:,0]
#     elif np.ndim(x) == 4:
#       x=x[:,:,:,0]
#   return x

# def visualize_saliency_of_output(FLAGS, model, input_images=[]):
#   """
#   Extract the salience map of the output control node(s) for a predefined set of visualization images.
#   You can define the input_images as a list, if nothing is defined it uses the predefined images.
#   It saves the images in FLAGS.summary_dir+FLAGS.log_tag+'/saliency_maps/....png'.
#   FLAGS: the settings defined in main.py
#   model: the loaded pilot-model object
#   input_images: a list of strings with absolute paths to images used to extract the maps 
#   overlay: plot activations in overlay over input image 
#   """
#   if len(input_images) == 0:
#     # use predefined images
#     img_dir='/esat/opal/kkelchte/docker_home/pilot_data/visualization_images'
#     input_images=sorted([img_dir+'/'+f for f in os.listdir(img_dir)])

#   print("[tools.py]: extracting saliency maps of {0} in {1}".format([os.path.basename(i) for i in input_images], os.path.dirname(input_images[0])))
  
  
#   inputs = load_images(input_images, model.input_size[1:])

#   print  'shape: ',inputs.shape

#   if 'nfc' in FLAGS.network:
#     inputs = np.concatenate([inputs]*FLAGS.n_frames, axis=-1)
  
#   # extract deconvolution
#   import tf_cnnvis

#   # layers = ['c']
#   # layers=['MobilenetV1_1/control/Conv2d_1c_1x1/Conv2D']
#   # layers=['MobilenetV1_1/control/Conv2d_1c_1x1/Conv2D','MobilenetV1_1/AvgPool_1a/AvgPool']

#   # layers = [str(i.name) for i in model.sess.graph.get_operations() if 'outputs' in i.name and not 'activations' in i.name and not 'gradients' in i.name]
#   layers = [model.endpoints['eval']['outputs'].name[:-2]] #cut out :0 in the end to change name from tensor to operation name
#   # layers = ['outputs']
  
#   # results = tf_cnnvis.activation_visualization(sess_graph_path = model.sess, 
#   #                                               value_feed_dict = {model.inputs : inputs}, 
#   #                                               layers=layers)
#   results = tf_cnnvis.deconv_visualization(sess_graph_path = model.sess, 
#                                             value_feed_dict = {model.inputs : inputs}, 
#                                             layers=layers)

#   # Normalize deconvolution within 0:1 range
#   num_rows=0
#   clean_results={} 
#   # Loop over layers
#   for k in results.keys():
#     clean_results[k]=[]
#     # Loop over channels
#     for c in range(len(results[k])):
#       num_rows+=1
#       clean_results[k].append(np.zeros((results[k][c].shape[0:3])))
#       # Loop over images
#       for i in range(results[k][c].shape[0]):
#         clean_results[k][c][i]=deprocess_image(results[k][c][i],one_channel=True)
#   if num_rows > 6:
#     print("[tools.py]: There are too many columns to create a proper image.")
#     return

#   # create one combined image with each input image on each column
#   fig, axes = plt.subplots(num_rows+1,min(len(input_images),5),figsize=(23, 4*(2*len(results.keys())+1)))
#   # fig, axes = plt.subplots(num_columns+1,min(len(input_images),5),figsize=(23, 4*(2*len(results.keys())+1)))
#   # add original images in first row
#   for i in range(axes.shape[1]):
#     axes[0, i].set_title(os.path.basename(input_images[i]).split('.')[0])
#     axes[0, i].imshow(matplotlibprove(inputs[i]), cmap='inferno')
#     axes[0, i].axis('off')
  
#   # experts=np.asarray([[k]*(FLAGS.action_quantity if FLAGS.discrete else 1) for v in sorted(model.factor_offsets.values()) for k in model.factor_offsets.keys() if model.factor_offsets[k]==v]).flatten()

#   # add deconvolutions over the columns
#   row_index = 1
#   for k in results.keys(): # go over layers
#     for c in range(len(results[k])): # add each channel in 2 new column
#       for i in range(axes.shape[1]): # fill row going over input images
#         # axes[row_index, i].set_title(k.split('/')[1]+'/'+k.split('/')[2]+'_'+str(c))
#         axes[row_index, i].set_title(k+'_'+str(c))
#         # axes[row_index, i].set_title(experts[c])
        
#         axes[row_index, i].imshow(np.concatenate((inputs[i],np.expand_dims(clean_results[k][c][i],axis=2)), axis=2))
#         axes[row_index, i].axis('off')
#       # row_index+=2
#       row_index+=1
#   # plt.show()
#   plt.savefig(FLAGS.summary_dir+FLAGS.log_tag+'/saliency_maps.jpg',bbox_inches='tight')

# def deep_dream_of_extreme_control(FLAGS,model,input_images=[],num_iterations=10,step_size=0.1):
#   """
#   Function that for each of the input image adjust a number of iterations.
#   It creates an image corresponding to strong left and strong right turn.
#   For continuous control it uses gradient ascent and descent.
#   For discrete control it uses the two extreme nodes for gradient ascent.
#   In the end an image is created by substracting the two extreme-control images.
#   """
#   if len(input_images) == 0:
#     # use predefined images
#     img_dir='/esat/opal/kkelchte/docker_home/pilot_data/visualization_images'
#     input_images=sorted([img_dir+'/'+f for f in os.listdir(img_dir)])

#   print("[tools.py]: extracting deep dream maps of {0} in {1}".format([os.path.basename(i) for i in input_images], os.path.dirname(input_images[0])))
  
#   # experts=np.asarray([[k]*(FLAGS.action_quantity if FLAGS.discrete else 1) for v in sorted(model.factor_offsets.values()) for k in model.factor_offsets.keys() if model.factor_offsets[k]==v]).flatten()

#   inputs = load_images(input_images, model.input_size[1:])
  
#   # collect gradients for output endpoint of evaluation model
#   grads={}
#   with tf.device('/cpu:0'):
#     output_tensor = model.endpoints['eval']['outputs']
#     for i in range(output_tensor.shape[1].value):
#       layer_loss = output_tensor[:,i]
#       gradients = tf.gradients(layer_loss, model.inputs)[0]
#       gradients /= (tf.sqrt(tf.reduce_mean(tf.square(gradients))) + 1e-5)
#       grads[output_tensor.name+'_'+str(i)]=gradients


#   # apply gradient ascent for all outputs and each input image
#   # if number of outputs ==1 apply gradient descent for contrast
#   if len(grads.keys())== 1:
#     opposite_results={}
#   else:
#     opposite_results=None

#   import copy
#   results = {}
#   for gk in grads.keys(): 
#     results[gk]=copy.deepcopy(inputs)
#     if isinstance(opposite_results,dict): opposite_results[gk]=copy.deepcopy(inputs)

#   for step in range(num_iterations):
#     if step%10==0: print "{0} step: {1}".format(time.ctime(), step)
#     for i,gk in enumerate(sorted(grads.keys())):
#       results[gk] += step_size * model.sess.run(grads[gk], {model.inputs: results[gk]})
#       if isinstance(opposite_results,dict):
#         opposite_results[gk] -= step_size * model.sess.run(grads[gk], {model.inputs: opposite_results[gk]})

#   # Normalize results within 0:1 range
#   clean_results={}
#   for gk in results.keys():
#     clean_results[gk]=[]
#     for i in range(results[gk].shape[0]):
#       clean_results[gk].append(deprocess_image(results[gk][i], one_channel=True))
#       # results[gk][i]=deprocess_image(results[gk][i], one_channel=True)
#       if isinstance(opposite_results,dict):
#         opposite_results[gk][i]=deprocess_image(opposite_results[gk][i])

#   # combine adjust input images in one overview image
#   # one column for each input image
#   # one row with each extreme control for separate and difference images
#   num_rows=1+len(results.keys())
#   fig, axes = plt.subplots(num_rows ,min(len(input_images),5),figsize=(23, 4*(len(grads.keys())+1)))
#   # fig, axes = plt.subplots(num_rows ,min(len(input_images),5),figsize=(23, 4*(len(grads.keys())+1)))
#   # add original images in first row
#   for i in range(axes.shape[1]):
#     axes[0, i].set_title(os.path.basename(input_images[i]).split('.')[0])
#     axes[0, i].imshow(matplotlibprove(inputs[i]), cmap='inferno')
#     axes[0, i].axis('off')

#   # add for each filter the modified input
#   row_index=1
#   for gk in sorted(results.keys()):
#     for i in range(axes.shape[1]):
#       # print gk
#       # axes[row_index, i].set_title('Grad Asc: '+gk.split('/')[1]+'/'+gk[-1])   
#       axes[row_index, i].set_title('Grad Asc: '+gk)
#       # axes[row_index, i].set_title(experts[row_index-1])

#       axes[row_index, i].imshow(np.concatenate((inputs[i],np.expand_dims(clean_results[gk][i],axis=2)), axis=2), cmap='inferno')
#       # axes[row_index, i].imshow(matplotlibprove(results[gk][i]), cmap='inferno')
#       axes[row_index, i].axis('off')
#     row_index+=1
#   # In cas of continouos controls: visualize the gradient descent and difference
#   # if isinstance(opposite_results,dict):
#   #   for gk in opposite_results.keys():
#   #       for i in range(axes.shape[1]):
#   #         # axes[row_index, i].set_title('Grad Desc: '+gk.split('/')[1])   
#   #         axes[row_index, i].set_title('Grad Desc: '+gk)   
#   #         axes[row_index, i].imshow(matplotlibprove(opposite_results[gk][i]), cmap='inferno')
#   #         axes[row_index, i].axis('off')
#   #       row_index+=1
    
#   #   # add difference
#   #   for gk in opposite_results.keys():
#   #       for i in range(axes.shape[1]):
#   #         # axes[row_index, i].set_title('Diff: '+gk.split('/')[1])   
#   #         axes[row_index, i].set_title('Diff: '+gk)   
#   #         axes[row_index, i].imshow(matplotlibprove(deprocess_image((opposite_results[gk][i]-results[gk][i])**2)), cmap='inferno')
#   #         axes[row_index, i].axis('off')
#   #       row_index+=1
#   # else:
#   #   # add difference between 2 exteme actions
#   #   gk_left=sorted(results.keys())[0]
#   #   gk_right=sorted(results.keys())[-1]
#   #   for i in range(axes.shape[1]):
#   #     # axes[row_index, i].set_title('Diff : '+gk.split('/')[1])   
#   #     axes[row_index, i].set_title('Diff : '+gk)   
#   #     axes[row_index, i].imshow(matplotlibprove(deprocess_image((results[gk_left][i]-results[gk_right][i])**2)), cmap='inferno')
#   #     axes[row_index, i].axis('off')
#   #   row_index+=1
  
  
#   plt.savefig(FLAGS.summary_dir+FLAGS.log_tag+'/control_dream_maps.jpg',bbox_inches='tight')
#   # plt.show()

# def visualize_activations(FLAGS,model,input_images=[],layers = ['c']):
#   """
#   Use cnn_vis to extract the activations on the input images.
#   """
#   if len(input_images) == 0:
#     # use predefined images
#     img_dir='/esat/opal/kkelchte/docker_home/pilot_data/visualization_images'
#     input_images=sorted([img_dir+'/'+f for f in os.listdir(img_dir)])
#   inputs = load_images(input_images, model.input_size[1:])
  
#   import tf_cnnvis
  
#   results = tf_cnnvis.activation_visualization(sess_graph_path = model.sess, 
#                                           value_feed_dict = {model.inputs : inputs}, 
#                                           layers=layers)
  
#   # combine activations in one subplot 
#   # --> number is currently too large to create reasonable subplot


#   # fig.canvas.tostring_rgb and then numpy.fromstring

# """
# Plot Control Activation Maps
# """
# def visualize_control_activation_maps(FLAGS, model, input_images=[]):
#   """
#   The control activation maps assumes that there is a global average pooling step at the end before the decision layer.
#   """
#   # load input
#   if len(input_images) == 0:
#     # use predefined images
#     img_dir='/esat/opal/kkelchte/docker_home/pilot_data/visualization_images'
#     input_images=sorted([img_dir+'/'+f for f in os.listdir(img_dir)])
#   inputs = load_images(input_images, model.input_size[1:])
  
#   # evaluate input to get activation maps
#   weights, activation_maps = model.sess.run([[v for v in tf.trainable_variables() if v.name == 'outputs/kernel:0'][0],
#                                             model.endpoints['eval']['activation_maps']], {model.inputs: inputs})

#   # combine the activation maps
#   activation_maps = np.dot(activation_maps,np.squeeze(weights))
  
#   if len(activation_maps.shape) != 4: activation_maps = np.expand_dims(activation_maps, axis=-1)

#   # create a nice plot with on the columns the different images and the rows the different experts

#   number_of_maps = activation_maps.shape[-1] 

#   fig, axes = plt.subplots(number_of_maps+1, # number of rows
#                           activation_maps.shape[0], # number of columns
#                           figsize=(23, 5*(number_of_maps+1)))
  
#   # fill first row with original image
#   for i in range(axes.shape[1]):
#     axes[0, i].set_title(os.path.basename(input_images[i]).split('.')[0])
#     axes[0, i].imshow(matplotlibprove(inputs[i]))
#     axes[0, i].axis('off')

#   # get expert names for titling
#   experts=np.asarray([[k]*(FLAGS.action_quantity if FLAGS.discrete else 1) for v in sorted(model.factor_offsets.values()) for k in model.factor_offsets.keys() if model.factor_offsets[k]==v]).flatten()

#   # add following rows for different experts with different upscaled activation maps
#   # for j in range(activation_maps.shape[-1]): # loop over diferent outputs
#   for j in range(number_of_maps): # loop over diferent outputs
#     for i in range(axes.shape[1]):
#       axes[j+1, i].set_title(experts[j])
#       # pure upscaled heat maps:
#       axes[j+1, i].imshow(matplotlibprove(activation_maps[i,:,:,j]), cmap='seismic')
#       # concatenated in alpha channels:
#       # axes[j+1, i].imshow(np.zeros(inputs[i].shape[0:3]))
#       # axes[j+1, i].imshow(matplotlibprove(np.concatenate((inputs[i], deprocess_image(sm.resize(activation_maps[i,:,:,j],inputs[i].shape[0:2]+(1,),order=1,mode='constant', preserve_range=True))), axis=2)))
#       axes[j+1, i].axis('off')

#   plt.savefig(FLAGS.summary_dir+FLAGS.log_tag+'/control_activation_maps.jpg',bbox_inches='tight')
#   print("saved control_activation_maps")
#   # plt.show()
#   # import pdb; pdb.set_trace()

