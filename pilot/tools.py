#!/usr/bin/python
import model

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import os,sys,time


FLAGS = tf.app.flags.FLAGS

""" 
Practical tools used by main, data and model
"""
def print_dur(duration_time):
  duration = duration_time #(time.time()-start_time)
  m, s = divmod(duration, 60)
  h, m = divmod(m, 60)
  return "time: %dh:%02dm:%02ds" % (h, m, s)

def get_endpoint_activations(inputs, model):
	'''Run forward through the network for this batch and return all activations
	of all intermediate endpoints
	[not used]
	'''
	tensors = [ model.endpoints[ep] for ep in model.endpoints]
	activations = model.sess.run(tensors, feed_dict={model.inputs:inputs})
	return [ a.reshape(-1,1) for a in activations]

def fig2buf(fig):
  """
  Convert a plt fig to a numpy buffer
  """
  # draw the renderer
  fig.canvas.draw()

  # Get the RGBA buffer from the figure
  w,h = fig.canvas.get_width_height()
  buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
  buf.shape = (h, w, 4)
  
  # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
  buf = np.roll(buf, 3, axis = 2 )
  buf = buf[0::1,0::1] #slice to make image 4x smaller and use only the R channel of RGBA
  buf = buf[0::1,0::1, 0:3] #slice to make image 4x smaller and use only the R channel of RGBA
  return buf

def plot_depth(inputs, depth_targets, model):
  '''plot depth predictions and return np array as floating image'''
  depths = model.sess.run(model.pred_depth, feed_dict={model.inputs: inputs})
  n=3
  fig = plt.figure(1, figsize=(5,5))
  fig.suptitle('depth predictions', fontsize=20)
  for i in range(n):
    plt.axis('off') 
    plt.subplot(n, 3, 1+3*i)
    if FLAGS.n_fc: plt.imshow(inputs[i][:,:,0+3*(FLAGS.n_frames-1):]*1/255.)
    else : plt.imshow(inputs[i][:,:,0:3]*1/255.)
    plt.axis('off') 
    plt.subplot(n, 3, 2+3*i)
    plt.imshow(depths[i]*1/5.)
    plt.axis('off') 
    if len(depth_targets)!=0:
      plt.subplot(n, 3, 3+3*i)
      plt.imshow(depth_targets[i]*1/5.)
      plt.axis('off')
  buf=fig2buf(fig)
  return np.asarray([buf])


"""
Plot activations
"""

def load_images(file_names, im_size):
  """
  Load a series of images defined in a list of full paths.
  Returns a numpy array of preprocessed images.
  """
  import skimage.io as sio
  import skimage.transform as sm

  # load images in numpy array inputs
  inputs=[]
  for img_file in file_names:
      img = sio.imread(img_file)
      scale_height = int(np.floor(img.shape[0]/im_size[0]))
      scale_width = int(np.floor(img.shape[1]/im_size[1]))
      img = sm.resize(img[::scale_height,::scale_width],im_size,mode='constant').astype(float)
      inputs.append(img)
  return np.asarray(inputs)

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 1
    #     x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 1).astype('float')
    return x

def matplotlibprove(x):
  """
  bg: matploblit imshow can only images of shape (n,m) (n,m,3) or (n,m,4)
  if your image is (n,m,1), change it to (n,m)
  x: numpy nd array
  """
  if x.shape[-1] == 1:
    if np.ndim(x) == 3:
      x=x[:,:,0]
    elif np.ndim(x) == 4:
      x=x[:,:,:,0]
  return x

def visualize_saliency_of_output(FLAGS, model, input_images=[]):
  """
  Extract the salience map of the output control node(s) for a predefined set of visualization images.
  You can define the input_images as a list, if nothing is defined it uses the predefined images.
  It saves the images in FLAGS.summary_dir+FLAGS.log_tag+'/saliency_maps/....png'.
  FLAGS: the settings defined in main.py
  model: the loaded pilot-model object
  input_images: a list of strings with absolute paths to images used to extract the maps 
  overlay: plot activations in overlay over input image 
  """
  if len(input_images) == 0:
    # use predefined images
    img_dir='/esat/opal/kkelchte/docker_home/pilot_data/visualization_images'
    input_images=sorted([img_dir+'/'+f for f in os.listdir(img_dir)])

  print("[tools.py]: extracting saliency maps of {0} in {1}".format([os.path.basename(i) for i in input_images], os.path.dirname(input_images[0])))
  
  
  inputs = load_images(input_images, model.input_size[1:])

  print  'shape: ',inputs.shape

  if 'nfc' in FLAGS.network:
    inputs = np.concatenate([inputs]*FLAGS.n_frames, axis=-1)
  
  # extract deconvolution
  import tf_cnnvis

  # layers = ['c']
  # layers=['MobilenetV1_1/control/Conv2d_1c_1x1/Conv2D']
  # layers=['MobilenetV1_1/control/Conv2d_1c_1x1/Conv2D','MobilenetV1_1/AvgPool_1a/AvgPool']

  # layers = [str(i.name) for i in model.sess.graph.get_operations() if 'outputs' in i.name and not 'activations' in i.name and not 'gradients' in i.name]
  layers = [model.endpoints['eval']['outputs'].name[:-2]] #cut out :0 in the end to change name from tensor to operation name
  # layers = ['outputs']
  
  # results = tf_cnnvis.activation_visualization(sess_graph_path = model.sess, 
  #                                               value_feed_dict = {model.inputs : inputs}, 
  #                                               layers=layers)
  results = tf_cnnvis.deconv_visualization(sess_graph_path = model.sess, 
                                            value_feed_dict = {model.inputs : inputs}, 
                                            layers=layers)

  # Normalize deconvolution within 0:1 range
  num_rows=0
  # Loop over layers
  for k in results.keys():
    # Loop over channels
    for c in range(len(results[k])):
      num_rows+=1
      # Loop over images
      for i in range(results[k][c].shape[0]):
        results[k][c][i]=deprocess_image(results[k][c][i])
  # if num_rows > 6:
  #   print("[tools.py]: There are too many columns to create a proper image.")
  #   return

  # create one combined image with each input image on each column
  fig, axes = plt.subplots(2*num_rows+1,min(len(input_images),5),figsize=(15, int(5.5*(2*len(results.keys())+1))))
  # fig, axes = plt.subplots(num_columns+1,min(len(input_images),5),figsize=(23, 4*(2*len(results.keys())+1)))
  # add original images in first row
  for i in range(axes.shape[1]):
    axes[0, i].set_title(os.path.basename(input_images[i]).split('.')[0])
    axes[0, i].imshow(matplotlibprove(inputs[i]), cmap='inferno')
    axes[0, i].axis('off')
  
  # add deconvolutions over the columns
  row_index = 1
  for k in results.keys(): # go over layers
    for c in range(len(results[k])): # add each channel in 2 new column
      for i in range(axes.shape[1]): # fill row going over input images
        # axes[row_index, i].set_title(k.split('/')[1]+'/'+k.split('/')[2]+'_'+str(c))
        axes[row_index, i].set_title(k+'_'+str(c))
        axes[row_index, i].imshow(matplotlibprove(results[k][c][i]), cmap='inferno')
        axes[row_index, i].axis('off')
        axes[row_index+1,i].imshow(matplotlibprove((results[k][c][i]+inputs[i])/2), cmap='inferno')
        axes[row_index+1,i].axis('off')
      row_index+=2
      # row_index+=1
  # plt.show()
  plt.savefig(FLAGS.summary_dir+FLAGS.log_tag+'/saliency_maps.jpg',bbox_inches='tight')

def deep_dream_of_extreme_control(FLAGS,model,input_images=[],num_iterations=10,step_size=0.1):
  """
  Function that for each of the input image adjust a number of iterations.
  It creates an image corresponding to strong left and strong right turn.
  For continuous control it uses gradient ascent and descent.
  For discrete control it uses the two extreme nodes for gradient ascent.
  In the end an image is created by substracting the two extreme-control images.
  """
  if len(input_images) == 0:
    # use predefined images
    img_dir='/esat/opal/kkelchte/docker_home/pilot_data/visualization_images'
    input_images=sorted([img_dir+'/'+f for f in os.listdir(img_dir)])

  print("[tools.py]: extracting deep dream maps of {0} in {1}".format([os.path.basename(i) for i in input_images], os.path.dirname(input_images[0])))
  
  inputs = load_images(input_images, model.input_size[1:])
  
  # collect gradients for output endpoint of evaluation model
  grads={}
  with tf.device('/cpu:0'):
    output_tensor = model.endpoints['eval']['outputs']
    for i in range(output_tensor.shape[1].value):
      layer_loss = output_tensor[:,i]
      gradients = tf.gradients(layer_loss, model.inputs)[0]
      gradients /= (tf.sqrt(tf.reduce_mean(tf.square(gradients))) + 1e-5)
      grads[output_tensor.name+'_'+str(i)]=gradients


  # apply gradient ascent for all outputs and each input image
  # if number of outputs ==1 apply gradient descent for contrast
  if len(grads.keys())== 1:
    opposite_results={}
  else:
    opposite_results=None

  import copy
  results = {}
  for gk in grads.keys(): 
    results[gk]=copy.deepcopy(inputs)
    if isinstance(opposite_results,dict): opposite_results[gk]=copy.deepcopy(inputs)

  for step in range(num_iterations):
    if step%10==0: print "{0} step: {1}".format(time.ctime(), step)
    for i,gk in enumerate(sorted(grads.keys())):
      results[gk] += step_size * model.sess.run(grads[gk], {model.inputs: results[gk]})
      if isinstance(opposite_results,dict):
        opposite_results[gk] -= step_size * model.sess.run(grads[gk], {model.inputs: opposite_results[gk]})

  # Normalize results within 0:1 range
  for gk in results.keys():
    for i in range(results[gk].shape[0]):
      results[gk][i]=deprocess_image(results[gk][i])
      if isinstance(opposite_results,dict):
        opposite_results[gk][i]=deprocess_image(opposite_results[gk][i])

  # combine adjust input images in one overview image
  # one column for each input image
  # one row with each extreme control for separate and difference images
  num_rows=1+len(results.keys())+1 if not isinstance(opposite_results,dict) else 1+2+1
  fig, axes = plt.subplots(num_rows ,min(len(input_images),5),figsize=(23, 4*(len(grads.keys())+1)))
  # add original images in first row
  for i in range(axes.shape[1]):
    axes[0, i].set_title(os.path.basename(input_images[i]).split('.')[0])
    axes[0, i].imshow(matplotlibprove(inputs[i]), cmap='inferno')
    axes[0, i].axis('off')

  # add for each filter the modified input
  row_index=1
  for gk in sorted(results.keys()):
    for i in range(axes.shape[1]):
      # print gk
      # axes[row_index, i].set_title('Grad Asc: '+gk.split('/')[1]+'/'+gk[-1])   
      axes[row_index, i].set_title('Grad Asc: '+gk)   
      axes[row_index, i].imshow(matplotlibprove(results[gk][i]), cmap='inferno')
      axes[row_index, i].axis('off')
    row_index+=1
  # In cas of continouos controls: visualize the gradient descent and difference
  if isinstance(opposite_results,dict):
    for gk in opposite_results.keys():
        for i in range(axes.shape[1]):
          # axes[row_index, i].set_title('Grad Desc: '+gk.split('/')[1])   
          axes[row_index, i].set_title('Grad Desc: '+gk)   
          axes[row_index, i].imshow(matplotlibprove(opposite_results[gk][i]), cmap='inferno')
          axes[row_index, i].axis('off')
        row_index+=1
    
    # add difference
    for gk in opposite_results.keys():
        for i in range(axes.shape[1]):
          # axes[row_index, i].set_title('Diff: '+gk.split('/')[1])   
          axes[row_index, i].set_title('Diff: '+gk)   
          axes[row_index, i].imshow(matplotlibprove(deprocess_image((opposite_results[gk][i]-results[gk][i])**2)), cmap='inferno')
          axes[row_index, i].axis('off')
        row_index+=1
  else:
    # add difference between 2 exteme actions
    gk_left=sorted(results.keys())[0]
    gk_right=sorted(results.keys())[-1]
    for i in range(axes.shape[1]):
      # axes[row_index, i].set_title('Diff : '+gk.split('/')[1])   
      axes[row_index, i].set_title('Diff : '+gk)   
      axes[row_index, i].imshow(matplotlibprove(deprocess_image((results[gk_left][i]-results[gk_right][i])**2)), cmap='inferno')
      axes[row_index, i].axis('off')
    row_index+=1
  
  
  plt.savefig(FLAGS.summary_dir+FLAGS.log_tag+'/control_dream_maps.jpg',bbox_inches='tight')
  # plt.show()

def visualize_activations(FLAGS,model,input_images=[],layers = ['c']):
  """
  Use cnn_vis to extract the activations on the input images.
  """
  if len(input_images) == 0:
    # use predefined images
    img_dir='/esat/opal/kkelchte/docker_home/pilot_data/visualization_images'
    input_images=sorted([img_dir+'/'+f for f in os.listdir(img_dir)])
  inputs = load_images(input_images, model.input_size[1:])
  
  import tf_cnnvis
  
  results = tf_cnnvis.activation_visualization(sess_graph_path = model.sess, 
                                          value_feed_dict = {model.inputs : inputs}, 
                                          layers=layers)
  
  # combine activations in one subplot 
  # --> number is currently too large to create reasonable subplot


  # fig.canvas.tostring_rgb and then numpy.fromstring

