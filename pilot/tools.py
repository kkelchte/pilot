#!/usr/bin/python
import model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
