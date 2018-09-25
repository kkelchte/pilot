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
            verbose=False,
            scope=""):
    """A basic alex net."""
    end_points={}
    
    end_point=scope+'conv_1'
    ep=tf.layers.conv2d(inputs, filters=10, kernel_size=[6,6], strides=3, padding='valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    end_points[end_point]=ep
    print("shape conv_1: {}".format(ep.shape))
    
    end_point=scope+'activation_maps'
    ep=tf.layers.conv2d(ep, filters=256, kernel_size=[3,3], strides=2, padding='valid', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    end_points[end_point]=ep                    
    print("shape activation_maps: {}".format(ep.shape))
    
    end_point=scope+'avg_pool'
    ep=tf.layers.average_pooling2d(ep, pool_size=20, strides=1, padding='valid',name=end_point)
    end_points[end_point]=ep                    
    print("shape avg_pool: {}".format(ep.shape))
    
    end_point=scope+'outputs'
    ep=tf.layers.conv2d(ep, filters=num_outputs, kernel_size=[1,1], strides=1, padding='valid', activation=None, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    end_points[end_point]=tf.squeeze(ep,[1,2],name=end_point+'_squeeze')
    
    return end_points

# default_image_size=[227,227,3]
# default_image_size=[127,127,3]
default_image_size=[128,128,3]
