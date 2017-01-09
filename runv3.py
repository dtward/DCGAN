import tensorflow as tf
import numpy as np



# some ops
def affine(x, m, name='Affine', A_stddev=0.02, b_value=0.0):
    ''' affine map.  first dimension of x is batch.  
    it should be a vector of dimension n
    we map to dimension m
    '''
    n = x.get_shape[1]
    # a new scope
    with tf.variable_scope(name):
        # create new variables (unless we are in a reuse scope, then fined the variables)
        A = tf.get_variable(name='A', shape=[n,m], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=A_stddev))
        b = tf.get_variable(name='b', shape=[m], dtype=tf.float32, initializer=tf.constant_initializer(value=b_value))
        return tf.matmul(x,m) + b

def conv2d(x, m, kernel_h=5, kernel_w=5, stride_h=2, stride_w=2, stddev=0.02, name='conv2d'):
    '''
    map to m features,
    '''
    n = x.get_shape[3] # note it goes batch, height width channels
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel', shape=[kernel_h, kernel_w, n, m], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=stddev))
        return tf.nn.conv2d(x, kernel, strides=[1,stride_h, stride_w, 1], padding='SAME')

def conv2dT(x, output_shape, kernel_h=5, kernel_w=5, stride_h=2, stride_w=2, stddev=0.02, name='conv2dT'):
    ''' transpose of convolution, upsampling'''
    n = x.get_shape[3]
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel', shape=[kernel_h, kernel_w, output_shape[3], n], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=stddev))
        # note kernel shape channels is "backwards"
        return tf.nn.conv2d_transpose(x, kernel, output_shape=output_shape, strides=[1, stride_h, stride_w, 1], padding='SAME')

# that should be everything I need for ops

# the other important thing is batch normalization
def batchnorm(x, epsilon=1e-5, momentum = 0.9, is_training=True, name='batch_norm'):
    reuse = not is_training
    return tf.contrib.layers.batch_norm(
        inputs=x,
        decay=momentum,
        updates_collections=None,
        epsilon=epsilon,
        scale=True,
        is_training=is_training,
        scope=name,
        reuse=reuse)
# let's think about this
# is a function the best way to do this?

    
