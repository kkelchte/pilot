"""
Version 
"""

import tensorflow as tf
import numpy as np

# input downscaled to 128x128x1
def tinynet(inputs,
            num_outputs=1,
            dropout_rate=0,
            reuse=None,
            is_training=False,
            verbose=False,
            scope=""):
    """A basic alex net."""
    end_points={}
    
    end_point=scope+'conv_1'
    ep=tf.layers.conv2d(inputs, filters=10, kernel_size=[6,6], strides=3, padding='valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(), name=end_point, reuse=reuse)
    end_points[end_point]=ep
    print("shape maps from conv_1: {}".format(ep.shape))
    
    end_point=scope+'conv_2'
    ep=tf.layers.conv2d(ep, filters=20, kernel_size=[3,3], strides=2, padding='valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(), name=end_point, reuse=reuse)
    end_points[end_point]=ep                    
    print("shape maps from conv_2: {}".format(ep.shape))
    
    end_point=scope+'conv_3'
    ep=tf.layers.dense(tf.reshape(ep,[-1,20*20*20]), 1024, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(), name=end_point, reuse=reuse)
    end_points[end_point]=ep                    
    print("shape maps from conv_3: {}".format(ep.shape))

    end_point=scope+'outputs'
    # print num_outputs
    ep=tf.layers.dense(ep, num_outputs, activation=None, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(), name=end_point, reuse=reuse)
    end_points[end_point]=ep
    # end_points[end_point]=tf.squeeze(ep,[1,2],name=end_point+'_squeeze')
    
    return end_points

# default_image_size=[227,227,3]
# default_image_size=[127,127,3]
default_image_size=[128,128,3]
