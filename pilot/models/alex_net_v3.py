"""
Version of Alexnet with batch normalization and dropout
"""

import tensorflow as tf
import numpy as np

# Arguments:
# inputs: Tensor input.
# axis: An int, the axis that should be normalized (typically the features axis). For instance, after a Convolution2D layer with data_format="channels_first", set axis=1 in BatchNormalization.
# momentum: Momentum for the moving average.
# epsilon: Small float added to variance to avoid dividing by zero.
# center: If True, add offset of beta to normalized tensor. If False, beta is ignored.
# scale: If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling can be done by the next layer.
# beta_initializer: Initializer for the beta weight.
# gamma_initializer: Initializer for the gamma weight.
# moving_mean_initializer: Initializer for the moving mean.
# moving_variance_initializer: Initializer for the moving variance.
# beta_regularizer: Optional regularizer for the beta weight.
# gamma_regularizer: Optional regularizer for the gamma weight.
# beta_constraint: An optional projection function to be applied to the beta weight after being updated by an Optimizer (e.g. used to implement norm constraints or value constraints for layer weights). The function must take as input the unprojected variable and must return the projected variable (which must have the same shape). Constraints are not safe to use when doing asynchronous distributed training.
# gamma_constraint: An optional projection function to be applied to the gamma weight after being updated by an Optimizer.
# training: Either a Python boolean, or a TensorFlow boolean scalar tensor (e.g. a placeholder). Whether to return the output in training mode (normalized with statistics of the current batch) or in inference mode (normalized with moving statistics). NOTE: make sure to set this parameter correctly, or else your training/inference will not work properly.
# trainable: Boolean, if True also add variables to the graph collection GraphKeys.TRAINABLE_VARIABLES (see tf.Variable).
# name: String, the name of the layer.
# reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
# renorm: Whether to use Batch Renormalization (https://arxiv.org/abs/1702.03275). This adds extra variables during training. The inference is the same for either value of this parameter.
# renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to scalar Tensors used to clip the renorm correction. The correction (r, d) is used as corrected_value = normalized_value * r + d, with r clipped to [rmin, rmax], and d to [-dmax, dmax]. Missing rmax, rmin, dmax are set to inf, 0, inf, respectively.
# renorm_momentum: Momentum used to update the moving means and standard deviations with renorm. Unlike momentum, this affects training and should be neither too small (which would add noise) nor too large (which would give stale estimates). Note that momentum is still applied to get the means and variances for inference.
# fused: if None or True, use a faster, fused implementation if possible. If False, use the system recommended implementation.
# virtual_batch_size: An int. By default, virtual_batch_size is None, which means batch normalization is performed across the whole batch. When virtual_batch_size is not None, instead perform "Ghost Batch Normalization", which creates virtual sub-batches which are each normalized separately (with shared gamma, beta, and moving statistics). Must divide the actual batch size during execution.
# adjustment: A function taking the Tensor containing the (dynamic) shape of the input tensor and returning a pair (scale, bias) to apply to the normalized values (before gamma and beta), only during training. For example, if axis==-1, adjustment = lambda shape: ( tf.random_uniform(shape[-1:], 0.93, 1.07), tf.random_uniform(shape[-1:], -0.1, 0.1)) will scale the normalized value by up to 7% up or down, then shift the result by up to 0.1 (with independent scaling and bias for each feature but shared across all examples), and finally apply gamma and/or beta. If None, no adjustment is applied. Cannot be specified if virtual_batch_size is specified.

# See also tutorial here: https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-mnist-tutorial/README_BATCHNORM.md
# with slides here: https://docs.google.com/presentation/d/e/2PACX-1vRouwj_3cYsmLrNNI3Uq5gv5-hYp_QFdeoan2GlxKgIZRSejozruAbVV0IMXBoPsINB7Jw92vJo2EAM/pub?slide=id.g187d73109b_1_1242

# Extra notes on parameters: 
# axis should change to 1 if dense connection layer
# scale is not used as we work with relu's
    

def alexnet(inputs,
            num_outputs=1,
            dropout_rate=0,
            reuse=None,
            is_training=False,
            verbose=False):
  end_points={}
  
  # TOWER ONE
  end_point = 'conv_1'
  l1 = tf.layers.conv2d(inputs, 96, kernel_size=[11,11], strides=4, padding='valid', activation=None, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l1: {}".format(l1.shape))
  end_points[end_point]=l1
  end_point='bn_1'
  bn1 = tf.layers.batch_normalization(l1, axis=-1, momentum=0.999, epsilon=0.00001, center=True, scale=False, training=is_training, name=end_point, reuse=reuse)
  end_points[end_point]=bn1
  end_point='relu_1'
  relu1 = tf.nn.relu(bn1, name=end_point)
  end_points[end_point]=relu1    
  end_point = 'pool_1'
  p1=tf.layers.max_pooling2d(relu1, pool_size=3, strides=2, padding='valid',name=end_point)
  if verbose: print("shape p1: {}".format(p1.shape))
  end_points[end_point]=p1
  
  # TOWER TWO
  end_point = 'conv_2'
  l2=tf.layers.conv2d(p1, 256, kernel_size=[5,5], strides=1, padding='same', activation=None, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l2: {}".format(l2.shape))
  end_points[end_point]=l2
  end_point='bn_2'
  bn2 = tf.layers.batch_normalization(l2, axis=-1, momentum=0.999, epsilon=0.00001, center=True, scale=False, training=is_training, name=end_point, reuse=reuse)
  end_points[end_point]=bn2
  end_point='relu_2'
  relu2 = tf.nn.relu(bn2, name=end_point)
  end_points[end_point]=relu2    
  end_point = 'pool_2'
  p2=tf.layers.max_pooling2d(relu2, pool_size=3, strides=2, padding='valid',name=end_point)
  if verbose: print("shape p2: {}".format(p2.shape))
  end_points[end_point]=p2

  # TOWER THREE
  end_point = 'conv_3'
  l3=tf.layers.conv2d(p2, 384, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l3: {}".format(l3.shape))
  end_points[end_point]=l3
  end_point = 'conv_4'
  l4=tf.layers.conv2d(l3, 384, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l4: {}".format(l4.shape))
  end_points[end_point]=l4
  end_point = 'conv_5'
  l5=tf.layers.conv2d(l4, 256, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l5: {}".format(l5.shape))
  end_points[end_point]=l5
  end_point = 'pool_5'
  p5=tf.layers.max_pooling2d(l5, pool_size=3, strides=2, padding='valid', name=end_point)
  if verbose: print("shape p5: {}".format(p5.shape))
  end_points[end_point]=p5
  p5 = tf.reshape(p5, (-1,1,6*6*256))
  
  if dropout_rate != 0:
      end_point = 'dropout_5'
      p5 = tf.layers.dropout(p5, dropout_rate)
      end_points[end_point]=p5
  
  end_point = 'fc_6'
  l6=tf.layers.conv1d(p5, filters=4096, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l6: {}".format(l6.shape))
  end_points[end_point]=l6
  
  if dropout_rate != 0:
      end_point = 'dropout_6'
      l6 = tf.layers.dropout(l6, dropout_rate)
      end_points[end_point]=l6
  
  end_point = 'fc_7'
  l7=tf.layers.conv1d(l6, filters=4096, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l7: {}".format(l7.shape))
  end_points[end_point]=l7

  end_point = 'fc_8'
  l8=tf.layers.conv1d(l7, filters=num_outputs, kernel_size=1, strides=1, padding='valid', activation=tf.nn.tanh if num_outputs == 1 else tf.nn.relu, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l8: {}".format(l8.shape))
  end_points[end_point]=l8
  end_point = 'outputs'
  outputs = tf.squeeze(l8, [1], name=end_point)
  if verbose: print("shape outputs: {}".format(outputs.shape))
  end_points[end_point]=outputs
  
  return end_points

default_image_size=[227,227,3]
