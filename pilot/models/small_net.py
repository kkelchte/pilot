"""
Code based on inception net from SLIM and https://github.com/vonclites/squeezenet/blob/master/squeezenet.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib import layers

"""
Default values for conv2d tensorflow 1.4:
  padding='SAME',
  data_format=None,
  rate=1, --> for atrous/dilated convolutions
  activation_fn=tf.nn.relu,
  normalizer_fn=None,
  normalizer_params=None,
  weights_initializer=initializers.xavier_initializer(),
  weights_regularizer=None,
  biases_initializer=tf.zeros_initializer(),
  biases_regularizer=None,
  reuse=None,
  variables_collections=None,
  outputs_collections=None,
  trainable=True,
  scope=None

Default values for conv2d tensorflow 1.8:
  padding='valid',
  data_format='channels_last',
  dilation_rate=(1, 1),
  activation=None, <--!!!
  use_bias=True,
  kernel_initializer=None,
  bias_initializer=tf.zeros_initializer(),
  kernel_regularizer=None,
  bias_regularizer=None,
  activity_regularizer=None,
  kernel_constraint=None,
  bias_constraint=None,
  trainable=True,
  name=None,
  reuse=None

"""



def smallnet(inputs,
             num_classes=1000,
             is_training=True,
             reuse=None,
             dropout_keep_prob=0.5,
             weight_decay=0.00004,
             stddev=0.09,
             initializer='xavier',
             random_seed=123):
    """A basic small fully convolutional network."""
    with tf.variable_scope('smallnet', values=[inputs], reuse=reuse) as sc:
      y1=layers.conv2d(inputs, num_outputs=3*4, kernel_size=[3,3], stride=4, padding='VALID') #1.8: 'valid' is small letters and is default
      y2=layers.conv2d(y1, num_outputs=3*4*4, kernel_size=[3,3], stride=4, padding='VALID')
      y3=layers.conv2d(y2, num_outputs=3*4*4*4, kernel_size=[3,3], stride=2, padding='VALID')
      y4=layers.conv2d(y3, num_outputs=3*4*4*4*2, kernel_size=[1,1], stride=2, padding='VALID')
      logits=layers.conv2d(y4, num_outputs=num_classes, kernel_size=[1,1], stride=2, padding='VALID', activation_fn=None)
      logits = tf.squeeze(logits, [1,2])

      return logits, {}

default_image_size=[100,100,3]
