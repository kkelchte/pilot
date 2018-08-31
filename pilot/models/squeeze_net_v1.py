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
  end_point = 'pad_1'
  paddings = tf.constant([[0,0],[2, 1], [2, 1],[0,0]])
  pad_1 = tf.pad(inputs, paddings, "CONSTANT")
  if verbose: print("shape pad_1: {}".format(pad_1.shape))
  end_points[end_point]=pad_1
  
  end_point = 'conv_1'
  l1 = tf.layers.conv2d(pad_1, 96, kernel_size=[7,7], strides=2, padding='valid', activation=None, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
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
  
  # TOWER 2 Firemodule
  end_point = 'squeeze_2_conv_1'
  l2_s1 = tf.layers.conv2d(p1, 16, kernel_size=[1,1], strides=1, padding='valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l2_s1: {}".format(l2_s1.shape))
  end_points[end_point]=l2_s1

  end_point = 'expand_2_conv_1'
  l2_e1 = tf.layers.conv2d(l2_s1, 64, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l2_e1: {}".format(l2_e1.shape))
  end_points[end_point]=l2_e1
  
  end_point = 'expand_2_conv_3'
  l2_e3 = tf.layers.conv2d(l2_s1, 64, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l2_e3: {}".format(l2_e3.shape))
  end_points[end_point]=l2_e3

  end_point = 'concat_2'
  l2_c=tf.concat([l2_e1, l2_e3], 3,name=end_point)
  if verbose: print("shape l2_c: {}".format(l2_c.shape))
  end_points[end_point]=l2_c

  # TOWER 3 Firemodule
  end_point = 'squeeze_3_conv_1'
  l3_s1 = tf.layers.conv2d(l2_c, 16, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l3_s1: {}".format(l3_s1.shape))
  end_points[end_point]=l3_s1

  end_point = 'expand_3_conv_1'
  l3_e1 = tf.layers.conv2d(l3_s1, 64, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l3_e1: {}".format(l3_e1.shape))
  end_points[end_point]=l3_e1
  
  end_point = 'expand_3_conv_3'
  l3_e3 = tf.layers.conv2d(l3_s1, 64, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l3_e3: {}".format(l3_e3.shape))
  end_points[end_point]=l3_e3

  end_point = 'concat_3'
  l3_c=tf.concat([l3_e1, l3_e3], 3,name=end_point)
  if verbose: print("shape l3_c: {}".format(l3_c.shape))
  end_points[end_point]=l3_c

  # TOWER 4 Firemodule with batch normalization and max pool
  end_point = 'squeeze_4_conv_1'
  l4_s1 = tf.layers.conv2d(l3_c, 32, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l4_s1: {}".format(l4_s1.shape))
  end_points[end_point]=l4_s1

  end_point = 'expand_4_conv_1'
  l4_e1 = tf.layers.conv2d(l4_s1, 128, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l4_e1: {}".format(l4_e1.shape))
  end_points[end_point]=l4_e1
  
  end_point = 'expand_4_conv_3'
  l4_e3 = tf.layers.conv2d(l4_s1, 128, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l4_e3: {}".format(l4_e3.shape))
  end_points[end_point]=l4_e3

  end_point = 'concat_4'
  l4_c=tf.concat([l4_e1, l4_e3], 3,name=end_point)
  if verbose: print("shape l4_c: {}".format(l4_c.shape))
  end_points[end_point]=l4_c
 
  end_point='bn_4'
  bn4 = tf.layers.batch_normalization(l4_c, axis=-1, momentum=0.999, epsilon=0.00001, center=True, scale=False, training=is_training, name=end_point, reuse=reuse)
  end_points[end_point]=bn4
  
  p4=tf.layers.max_pooling2d(bn4, pool_size=3, strides=2, padding='valid',name=end_point)
  if verbose: print("shape p4: {}".format(p4.shape))
  end_points[end_point]=p4
  
  # TOWER 5 Firemodule
  end_point = 'squeeze_5_conv_1'
  l5_s1 = tf.layers.conv2d(p4, 32, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l5_s1: {}".format(l5_s1.shape))
  end_points[end_point]=l5_s1

  end_point = 'expand_5_conv_1'
  l5_e1 = tf.layers.conv2d(l5_s1, 128, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l5_e1: {}".format(l5_e1.shape))
  end_points[end_point]=l5_e1
  
  end_point = 'expand_5_conv_3'
  l5_e3 = tf.layers.conv2d(l5_s1, 128, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l5_e3: {}".format(l5_e3.shape))
  end_points[end_point]=l5_e3

  end_point = 'concat_5'
  l5_c=tf.concat([l5_e1, l5_e3], 3,name=end_point)
  if verbose: print("shape l5_c: {}".format(l5_c.shape))
  end_points[end_point]=l5_c

  # TOWER 6 Firemodule
  end_point = 'squeeze_6_conv_1'
  l6_s1 = tf.layers.conv2d(l5_c, 48, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l6_s1: {}".format(l6_s1.shape))
  end_points[end_point]=l6_s1

  end_point = 'expand_6_conv_1'
  l6_e1 = tf.layers.conv2d(l6_s1, 192, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l6_e1: {}".format(l6_e1.shape))
  end_points[end_point]=l6_e1
  
  end_point = 'expand_6_conv_3'
  l6_e3 = tf.layers.conv2d(l6_s1, 192, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l6_e3: {}".format(l6_e3.shape))
  end_points[end_point]=l6_e3

  end_point = 'concat_6'
  l6_c=tf.concat([l6_e1, l6_e3], 3,name=end_point)
  if verbose: print("shape l6_c: {}".format(l6_c.shape))
  end_points[end_point]=l6_c

  # TOWER 7 Firemodule
  end_point = 'squeeze_7_conv_1'
  l7_s1 = tf.layers.conv2d(l6_c, 48, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l7_s1: {}".format(l7_s1.shape))
  end_points[end_point]=l7_s1

  end_point = 'expand_7_conv_1'
  l7_e1 = tf.layers.conv2d(l7_s1, 192, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l7_e1: {}".format(l7_e1.shape))
  end_points[end_point]=l7_e1
  
  end_point = 'expand_7_conv_3'
  l7_e3 = tf.layers.conv2d(l7_s1, 192, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l7_e3: {}".format(l7_e3.shape))
  end_points[end_point]=l7_e3

  end_point = 'concat_7'
  l7_c=tf.concat([l7_e1, l7_e3], 3,name=end_point)
  if verbose: print("shape l7_c: {}".format(l7_c.shape))
  end_points[end_point]=l7_c

  # TOWER 8 Firemodule + batch norm + max pool
  end_point = 'squeeze_8_conv_1'
  l8_s1 = tf.layers.conv2d(l7_c, 64, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l8_s1: {}".format(l8_s1.shape))
  end_points[end_point]=l8_s1

  end_point = 'expand_8_conv_1'
  l8_e1 = tf.layers.conv2d(l8_s1, 256, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l8_e1: {}".format(l8_e1.shape))
  end_points[end_point]=l8_e1
  
  end_point = 'expand_8_conv_3'
  l8_e3 = tf.layers.conv2d(l8_s1, 256, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l8_e3: {}".format(l8_e3.shape))
  end_points[end_point]=l8_e3

  end_point = 'concat_8'
  l8_c=tf.concat([l8_e1, l8_e3], 3,name=end_point)
  if verbose: print("shape l8_c: {}".format(l8_c.shape))
  end_points[end_point]=l8_c

  end_point='bn_8'
  bn8 = tf.layers.batch_normalization(l8_c, axis=-1, momentum=0.999, epsilon=0.00001, center=True, scale=False, training=is_training, name=end_point, reuse=reuse)
  end_points[end_point]=bn8

  p8=tf.layers.max_pooling2d(bn8, pool_size=3, strides=2, padding='valid',name=end_point)
  if verbose: print("shape p8: {}".format(p8.shape))
  end_points[end_point]=p8
  
  # TOWER 9 Firemodule
  end_point = 'squeeze_9_conv_1'
  l9_s1 = tf.layers.conv2d(p8, 64, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l9_s1: {}".format(l9_s1.shape))
  end_points[end_point]=l9_s1

  end_point = 'expand_9_conv_1'
  l9_e1 = tf.layers.conv2d(l9_s1, 256, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l9_e1: {}".format(l9_e1.shape))
  end_points[end_point]=l9_e1
  
  end_point = 'expand_9_conv_3'
  l9_e3 = tf.layers.conv2d(l9_s1, 256, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l9_e3: {}".format(l9_e3.shape))
  end_points[end_point]=l9_e3

  end_point = 'concat_9'
  l9_c=tf.concat([l9_e1, l9_e3], 3,name=end_point)
  if verbose: print("shape l9_c: {}".format(l9_c.shape))
  end_points[end_point]=l9_c

  # TOWER 10 Conv2d and avgpool
  end_point = 'conv_10'
  l10 = tf.layers.conv2d(l9_c, num_outputs, kernel_size=[1,1], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
  if verbose: print("shape l10: {}".format(l10.shape))
  end_points[end_point]=l10

  end_point = 'avgpool_10'
  p10=tf.layers.max_pooling2d(l10, pool_size=13, strides=1, padding='valid',name=end_point)
  if verbose: print("shape p10: {}".format(p10.shape))
  end_points[end_point]=p10

  end_point = 'outputs'
  outputs = tf.squeeze(p10, [1,2], name=end_point)
  if verbose: print("shape outputs: {}".format(outputs.shape))
  end_points[end_point]=outputs

  return end_points 

default_image_size=[224,224,3]


