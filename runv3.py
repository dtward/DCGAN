import tensorflow as tf

import numpy as np



# some ops
def affine(x, m, name='Affine', A_stddev=0.02, b_value=0.0):
    ''' affine map.  first dimension of x is batch.  
    it should be a vector of dimension n
    we map to dimension m
    '''
    n = x.get_shape()[1]
    # a new scope
    with tf.variable_scope(name):
        # create new variables (unless we are in a reuse scope, then fined the variables)
        A = tf.get_variable(name='A', shape=[n,m], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=A_stddev))
        b = tf.get_variable(name='b', shape=[m], dtype=tf.float32, initializer=tf.constant_initializer(value=b_value))
        return tf.matmul(x,A) + b

def conv2d(x, m, kernel_h=5, kernel_w=5, stride_h=2, stride_w=2, stddev=0.02, name='conv2d'):
    '''
    map to m features,
    '''
    n = x.get_shape()[3] # note it goes batch, height width channels
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel', shape=[kernel_h, kernel_w, n, m], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=stddev))
        return tf.nn.conv2d(x, kernel, strides=[1,stride_h, stride_w, 1], padding='SAME')

def conv2dT(x, m, kernel_h=5, kernel_w=5, stride_h=2, stride_w=2, stddev=0.02, name='conv2dT'):
    ''' 
    transpose of convolution, upsampling, output m channels
    note that the way I'm doing it I can't go from odd to even
    or even to odd.  i think that's thre reason for including
    the output_shape argument
    '''
    b,h,w,n = x.get_shape().as_list()
    output_shape = [b,h*stride_h,w*stride_w,m]
    # this output shape is not working
    # it seems to work with "as list"
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel', shape=[kernel_h, kernel_w,m,n], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=stddev))
        # note kernel shape channels is "backwards"
        return tf.nn.conv2d_transpose(x, kernel, output_shape=output_shape, strides=[1, stride_h, stride_w, 1], padding='SAME')

# that should be everything I need for ops

# the other important thing is batch normalization
def batchnorm(x, epsilon=1e-5, momentum = 0.9, is_training=True, name='batch_norm'):
    return tf.contrib.layers.batch_norm(
        inputs=x,
        decay=momentum,
        updates_collections=None, # this is a simple way to do everything without dealing with features I din't understand
        epsilon=epsilon,
        scale=True,
        is_training=is_training,
        scope=name)
# i'm not using thereuse argument
# it will be true if I'm in a reuse scope, and not otherwise
# thisis how i want it
# let's think about this
# is a function the best way to do this?

def lrelu(x):
    return tf.maximum(x,0.01*x)


NB = 10 # batch
NZ = 20 # input dim
NH = 28 # image height
NW = 28 # image width
NC = 1 # image channels
N0 = 256 # first layer, number of features
N1 = 128



def generator(z):
    with tf.variable_scope('generator') as scope:
        h0 = tf.reshape(tf.nn.relu(batchnorm(affine(z,N0*NH*NW/4/4,name='h0'),name='h0')), [-1, NH/4, NW/4, N0]) # 7x7
        h1 = tf.nn.relu(batchnorm(conv2dT(h0,N1,name='h1'),name='h1')) # 14x14
        #h2 = tf.nn.tanh(batchnorm(conv2dT(h1,NC,name='h2'),name='h2'))*0.5 + 0.5 # 28x28
        h2 = tf.nn.tanh(conv2dT(h1,NC,name='h2'))*0.5 + 0.5 # 28x28
        # woah that was so easy
        # I don't think I need batchnorm at the output
        return h2


def sampler(z):
    ''' 
    exacly the same as above but with reuse and no training on batchnorm 
    '''
    with tf.variable_scope('generator', reuse=True) as scope:
        h0 = tf.reshape(tf.nn.relu(batchnorm(affine(z,N0*NH*NW/4/4,name='h0'),is_training=False,name='h0')), [-1, NH/4, NW/4, N0]) # 7x7
        h1 = tf.nn.relu(batchnorm(conv2dT(h0,N1,name='h1'),is_training=False,name='h1')) # 14x14
        #h2 = tf.nn.tanh(batchnorm(conv2dT(h1,NC,name='h2'),is_training=False,name='h2'))*0.5 + 0.5 # 28x28
        h2 = tf.nn.tanh(conv2dT(h1,NC,name='h2'))*0.5 + 0.5 # 28x28
        return h2

        
def discriminator(image,reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        h0 = lrelu(batchnorm(conv2d(image,N1,name='h0'),name='h0')) # 14x14
        h1 = lrelu(batchnorm(conv2d(image,N0,name='h1'),name='h1')) # 7x7
        h2 = affine(tf.reshape(h1,[-1,N0*NH*NW/4/4]),1,name='h2') # no nonlinearity because we will use the sigmoid in the loss function
        return h2



z = tf.placeholder(dtype=tf.float32,shape=[NB,NZ])
g = generator(z)
s = sampler(z)
image = tf.placeholder(dtype=tf.float32,shape=[NB,NH,NW,NC])
d_true = discriminator(image)
d_false = discriminator(g,reuse=True)

# get the loss function
l_d_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_true,tf.ones_like(d_true)))
l_d_false = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_false,tf.zeros_like(d_true)))
l_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_true,tf.ones_lide(d_true))) # instead of negative d_false, this is betterfor training

l_d = l_d_true + l_d_false

