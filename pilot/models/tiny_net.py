"""
Version of Alexnet with smaller input size and less weights
"""

import tensorflow as tf
import numpy as np

# input downscaled to 128x128x1
def tinynet(inputs,
            num_outputs=1,
            dropout_rate=0,
            reuse=None,
            is_training=False,
            verbose=False):
    """A basic alex net."""
    end_points={}
    
    end_point='conv_1'
    ep=tf.layers.conv2d(inputs, filters=10, kernel_size=[6,6], strides=3, padding='valid', activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    end_points[end_point]=ep
    print("shape conv_1: {}".format(ep.shape))
    
    end_point='conv_2'
    ep=tf.layers.conv2d(ep, filters=20, kernel_size=[3,3], strides=2, padding='valid', activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    end_points[end_point]=ep                    
    print("shape conv_2: {}".format(ep.shape))
    
    end_point='outputs'
    print num_outputs
    ep=tf.layers.conv2d(ep, filters=num_outputs, kernel_size=[20,20], strides=1, padding='valid', activation=None, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    end_points[end_point]=tf.squeeze(ep,[1,2],name=end_point+'_squeeze')
    
    return end_points

# default_image_size=[227,227,3]
# default_image_size=[127,127,3]
default_image_size=[128,128,3]
