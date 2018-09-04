"""
Version of Alexnet with smaller input size and less weights
"""

import tensorflow as tf
import numpy as np

# input downscaled to 128x128x1
def alexnet(inputs,
            num_outputs=1,
            dropout_rate=0,
            reuse=None,
            is_training=False,
            verbose=False):
    """A basic alex net."""
    end_points={}
    
    # TOWER ONE
    end_point = 'conv_1'
    l1 = tf.layers.conv2d(inputs, 32, kernel_size=[11,11], strides=4, padding='valid', activation=None, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
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
    l2=tf.layers.conv2d(p1, 64, kernel_size=[5,5], strides=1, padding='same', activation=None, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
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
    l3=tf.layers.conv2d(p2, 64, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l3: {}".format(l3.shape))
    end_points[end_point]=l3
    end_point = 'conv_4'
    l4=tf.layers.conv2d(l3, 64, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l4: {}".format(l4.shape))
    end_points[end_point]=l4
    end_point = 'conv_5'
    l5=tf.layers.conv2d(l4, 64, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l5: {}".format(l5.shape))
    end_points[end_point]=l5
    end_point = 'pool_5'
    p5=tf.layers.max_pooling2d(l5, pool_size=3, strides=1 , padding='valid', name=end_point)
    if verbose: print("shape p5: {}".format(p5.shape))
    end_points[end_point]=p5
    p5 = tf.reshape(p5, (-1,1,4*4*64))
    
    if dropout_rate != 0:
        end_point = 'dropout_5'
        p5 = tf.layers.dropout(p5, dropout_rate)
        end_points[end_point]=p5
    
    end_point = 'fc_6'
    l6=tf.layers.conv1d(p5, filters=1024, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
    if verbose: print("shape l6: {}".format(l6.shape))
    end_points[end_point]=l6
    
    if dropout_rate != 0:
        end_point = 'dropout_6'
        l6 = tf.layers.dropout(l6, dropout_rate)
        end_points[end_point]=l6
    
    end_point = 'fc_7'
    l7=tf.layers.conv1d(l6, filters=1024, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=end_point, reuse=reuse)
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

# default_image_size=[227,227,3]
# default_image_size=[127,127,3]
default_image_size=[127,127,1]
