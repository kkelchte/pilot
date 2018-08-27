"""
Code based on inception net from SLIM and https://github.com/vonclites/squeezenet/blob/master/squeezenet.py
"""

import tensorflow as tf
import numpy as np

def alexnet(inputs,
            num_outputs=1,
            dropout_rate=0,
            reuse=None,
            verbose=False):
    """A basic alex net."""
    end_points={}
    
    # default activation of relu
    end_point = 'conv_1'
    l1=tf.layers.conv2d(inputs, 96, kernel_size=[11,11], strides=4, padding='valid', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l1: {}".format(l1.shape))
    end_points[end_point]=l1
    end_point = 'pool_1'
    p1=tf.layers.max_pooling2d(l1, pool_size=3, strides=2, padding='valid',name=end_point)
    if verbose: print("shape p1: {}".format(p1.shape))
    end_points[end_point]=p1
    end_point = 'conv_2'
    l2=tf.layers.conv2d(p1, 256, kernel_size=[5,5], strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l2: {}".format(l2.shape))
    end_points[end_point]=l2
    end_point = 'pool_2'
    p2=tf.layers.max_pooling2d(l2, pool_size=3, strides=2, padding='valid',name=end_point)
    if verbose: print("shape p2: {}".format(p2.shape))
    end_points[end_point]=p2
    end_point = 'conv_3'
    l3=tf.layers.conv2d(p2, 384, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l3: {}".format(l3.shape))
    end_points[end_point]=l3
    end_point = 'conv_4'
    l4=tf.layers.conv2d(l3, 384, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l4: {}".format(l4.shape))
    end_points[end_point]=l4
    end_point = 'conv_5'
    l5=tf.layers.conv2d(l4, 256, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l5: {}".format(l5.shape))
    end_points[end_point]=l5
    end_point = 'pool_5'
    p5=tf.layers.max_pooling2d(l5, pool_size=3, strides=2, padding='valid', name=end_point)
    if verbose: print("shape p5: {}".format(p5.shape))
    end_points[end_point]=p5
    reshaped_p5 = tf.reshape(p5, (-1,1,6*6*256))
    end_point = 'fc_6'
    l6=tf.layers.conv1d(reshaped_p5, filters=4096, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l6: {}".format(l6.shape))
    end_points[end_point]=l6
    end_point = 'fc_7'
    l7=tf.layers.conv1d(l6, filters=4096, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l7: {}".format(l7.shape))
    end_points[end_point]=l7
    end_point = 'fc_8'
    l8=tf.layers.conv1d(l7, filters=num_outputs, kernel_size=1, strides=1, padding='valid', activation=tf.nn.tanh if num_outputs == 1 else tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l8: {}".format(l8.shape))
    end_points[end_point]=l8
    end_point = 'outputs'
    outputs = tf.squeeze(l8, [1], name=end_point)
    if verbose: print("shape outputs: {}".format(outputs.shape))
    end_points[end_point]=outputs
    
    return end_points

default_image_size=[227,227,3]
