"""
Version of Squeezenet adjust to personal interface but based on 
https://github.com/vonclites/squeezenet/blob/master/networks/squeezenet.py
"""

import tensorflow as tf

import numpy as np


def squeezenet(inputs,
            num_outputs=1,
            dropout_rate=0,
            reuse=None,
            is_training=False,
            verbose=False):
  """Squeeze net implementation with batch normalization without skip connections."""
  end_points={}
  # TOWER 1 conv2d+bn+relu+maxpool
  end_point = '1_pad'
  paddings = tf.constant([[0,0],[2, 1], [2, 1],[0,0]])
  pad_1 = tf.pad(inputs, paddings, "CONSTANT")
  if verbose: print("shape pad_1: {}".format(pad_1.shape))
  end_points[end_point]=pad_1
  
  end_point = '1_conv'
  l1 = tf.layers.conv2d(pad_1, 96, kernel_size=[7,7], strides=2, padding='valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l1: {}".format(l1.shape))
  end_points[end_point]=l1

  end_point = '1_pool'
  p1=tf.layers.max_pooling2d(l1, pool_size=3, strides=2, padding='valid',name=end_point)
  if verbose: print("shape p1: {}".format(p1.shape))
  end_points[end_point]=p1
  
  # TOWER 2 Firemodule
  end_point = '2_squeeze_conv_1'
  l2_s1 = tf.layers.conv2d(p1, 16, kernel_size=[1,1], strides=1, padding='valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l2_s1: {}".format(l2_s1.shape))
  end_points[end_point]=l2_s1

  end_point = '2_expand_conv_1'
  l2_e1 = tf.layers.conv2d(l2_s1, 64, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l2_e1: {}".format(l2_e1.shape))
  end_points[end_point]=l2_e1
  
  end_point = '2_expand_conv_3'
  l2_e3 = tf.layers.conv2d(l2_s1, 64, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l2_e3: {}".format(l2_e3.shape))
  end_points[end_point]=l2_e3

  end_point = '2_concat'
  l2_c=tf.concat([l2_e1, l2_e3], 3,name=end_point)
  if verbose: print("shape l2_c: {}".format(l2_c.shape))
  end_points[end_point]=l2_c

  # TOWER 3 Firemodule
  end_point = '3_squeeze_conv_1'
  l3_s1 = tf.layers.conv2d(l2_c, 16, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l3_s1: {}".format(l3_s1.shape))
  end_points[end_point]=l3_s1

  end_point = '3_expand_conv_1'
  l3_e1 = tf.layers.conv2d(l3_s1, 64, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l3_e1: {}".format(l3_e1.shape))
  end_points[end_point]=l3_e1
  
  end_point = '3_expand_conv_3'
  l3_e3 = tf.layers.conv2d(l3_s1, 64, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l3_e3: {}".format(l3_e3.shape))
  end_points[end_point]=l3_e3

  end_point = '3_concat'
  l3_c=tf.concat([l3_e1, l3_e3], 3,name=end_point)
  if verbose: print("shape l3_c: {}".format(l3_c.shape))
  end_points[end_point]=l3_c

  # TOWER 4 Firemodule with batch normalization and max pool
  end_point = '4_squeeze_conv_1'
  l4_s1 = tf.layers.conv2d(l3_c, 32, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l4_s1: {}".format(l4_s1.shape))
  end_points[end_point]=l4_s1

  end_point = '4_expand_conv_1'
  l4_e1 = tf.layers.conv2d(l4_s1, 128, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l4_e1: {}".format(l4_e1.shape))
  end_points[end_point]=l4_e1
  
  end_point = '4_expand_conv_3'
  l4_e3 = tf.layers.conv2d(l4_s1, 128, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l4_e3: {}".format(l4_e3.shape))
  end_points[end_point]=l4_e3

  end_point = '4_concat'
  l4_c=tf.concat([l4_e1, l4_e3], 3,name=end_point)
  if verbose: print("shape l4_c: {}".format(l4_c.shape))
  end_points[end_point]=l4_c
 
  end_point = '4_pool'
  p4=tf.layers.max_pooling2d(l4_c, pool_size=3, strides=2, padding='valid',name=end_point)
  if verbose: print("shape p4: {}".format(p4.shape))
  end_points[end_point]=p4
  
  # TOWER 5 Firemodule
  end_point = '5_squeeze_conv_1'
  l5_s1 = tf.layers.conv2d(p4, 32, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l5_s1: {}".format(l5_s1.shape))
  end_points[end_point]=l5_s1

  end_point = '5_expand_conv_1'
  l5_e1 = tf.layers.conv2d(l5_s1, 128, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l5_e1: {}".format(l5_e1.shape))
  end_points[end_point]=l5_e1
  
  end_point = '5_expand_conv_3'
  l5_e3 = tf.layers.conv2d(l5_s1, 128, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l5_e3: {}".format(l5_e3.shape))
  end_points[end_point]=l5_e3

  end_point = '5_concat'
  l5_c=tf.concat([l5_e1, l5_e3], 3,name=end_point)
  if verbose: print("shape l5_c: {}".format(l5_c.shape))
  end_points[end_point]=l5_c

  # TOWER 6 Firemodule
  end_point = '6_squeeze_conv_1'
  l6_s1 = tf.layers.conv2d(l5_c, 48, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l6_s1: {}".format(l6_s1.shape))
  end_points[end_point]=l6_s1

  end_point = '6_expand_conv_1'
  l6_e1 = tf.layers.conv2d(l6_s1, 192, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l6_e1: {}".format(l6_e1.shape))
  end_points[end_point]=l6_e1
  
  end_point = '6_expand_conv_3'
  l6_e3 = tf.layers.conv2d(l6_s1, 192, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l6_e3: {}".format(l6_e3.shape))
  end_points[end_point]=l6_e3

  end_point = '6_concat'
  l6_c=tf.concat([l6_e1, l6_e3], 3,name=end_point)
  if verbose: print("shape l6_c: {}".format(l6_c.shape))
  end_points[end_point]=l6_c

  # TOWER 7 Firemodule
  end_point = '7_squeeze_conv_1'
  l7_s1 = tf.layers.conv2d(l6_c, 48, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l7_s1: {}".format(l7_s1.shape))
  end_points[end_point]=l7_s1

  end_point = '7_expand_conv_1'
  l7_e1 = tf.layers.conv2d(l7_s1, 192, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l7_e1: {}".format(l7_e1.shape))
  end_points[end_point]=l7_e1
  
  end_point = '7_expand_conv_3'
  l7_e3 = tf.layers.conv2d(l7_s1, 192, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l7_e3: {}".format(l7_e3.shape))
  end_points[end_point]=l7_e3

  end_point = '7_concat'
  l7_c=tf.concat([l7_e1, l7_e3], 3,name=end_point)
  if verbose: print("shape l7_c: {}".format(l7_c.shape))
  end_points[end_point]=l7_c

  # TOWER 8 Firemodule + batch norm + max pool
  end_point = '8_squeeze_conv_1'
  l8_s1 = tf.layers.conv2d(l7_c, 64, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l8_s1: {}".format(l8_s1.shape))
  end_points[end_point]=l8_s1

  end_point = '8_expand_conv_1'
  l8_e1 = tf.layers.conv2d(l8_s1, 256, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l8_e1: {}".format(l8_e1.shape))
  end_points[end_point]=l8_e1
  
  end_point = '8_expand_conv_3'
  l8_e3 = tf.layers.conv2d(l8_s1, 256, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l8_e3: {}".format(l8_e3.shape))
  end_points[end_point]=l8_e3

  end_point = '8_concat'
  l8_c=tf.concat([l8_e1, l8_e3], 3,name=end_point)
  if verbose: print("shape l8_c: {}".format(l8_c.shape))
  end_points[end_point]=l8_c

  end_point = '8_pool'
  p8=tf.layers.max_pooling2d(l8_c, pool_size=3, strides=2, padding='valid',name=end_point)
  if verbose: print("shape p8: {}".format(p8.shape))
  end_points[end_point]=p8
  
  # TOWER 9 Firemodule
  end_point = '9_squeeze_conv_1'
  l9_s1 = tf.layers.conv2d(p8, 64, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l9_s1: {}".format(l9_s1.shape))
  end_points[end_point]=l9_s1

  end_point = '9_expand_conv_1'
  l9_e1 = tf.layers.conv2d(l9_s1, 256, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l9_e1: {}".format(l9_e1.shape))
  end_points[end_point]=l9_e1
  
  end_point = '9_expand_conv_3'
  l9_e3 = tf.layers.conv2d(l9_s1, 256, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l9_e3: {}".format(l9_e3.shape))
  end_points[end_point]=l9_e3

  end_point = '9_concat'
  l9_c=tf.concat([l9_e1, l9_e3], 3,name=end_point)
  if verbose: print("shape l9_c: {}".format(l9_c.shape))
  end_points[end_point]=l9_c

  # TOWER 10 Conv2d and avgpool
  end_point = '10_conv'
  l10 = tf.layers.conv2d(l9_c, num_outputs, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.tanh if num_outputs == 1 else tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l10: {}".format(l10.shape))
  end_points[end_point]=l10

  end_point = '10_avgpool'
  p10=tf.layers.max_pooling2d(l10, pool_size=13, strides=1, padding='valid',name=end_point)
  if verbose: print("shape p10: {}".format(p10.shape))
  end_points[end_point]=p10

  end_point = 'outputs'
  outputs = tf.squeeze(p10, [1,2], name=end_point)
  if verbose: print("shape outputs: {}".format(outputs.shape))
  end_points[end_point]=outputs

  return end_points 

default_image_size=[224,224,3]

