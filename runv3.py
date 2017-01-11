import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import random

#from PIL import Image

# image loader
import glob
import os
class ImageLoader(object):
    ''' for now all the images should be the same size '''
    def __init__(self,image_dir,image_filter='*.jpg'):
        self.image_dir = image_dir
        self.image_filter = image_filter
        self.image_files = glob.glob(os.path.join(self.image_dir + '/' + self.image_filter))
        self.counter = 0 # will loop through files
        self.n_files = len(self.image_files)
    def next_batch(self,n):
        for i in xrange(n):
            if self.counter == 0:
                # shuffle in place
                random.shuffle(self.image_files)
            #J = Image.open(self.image_files[self.counter])
            J = plt.imread(self.image_files[self.counter])
            if i == 0:
                nh,nw,nc = J.shape
                I = np.zeros([n,nh,nw,nc],dtype=np.float32)
            I[i,:,:,:] = J.astype(np.float32)/255.0
            self.counter += 1
            # counter will wrap around
            self.counter %= self.n_files
        return I

image_dir = 'cats2'
image_dir = 'cats3' # note that I cropped this one better
use_mnist = False

if not use_mnist:
    imageLoader = ImageLoader(image_dir)

# some operations I will use with proper scoping
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
        b = tf.get_variable(name='b',shape=[m],dtype=tf.float32,initializer=tf.constant_initializer(value=0.0))
        return tf.nn.conv2d(x, kernel, strides=[1,stride_h, stride_w, 1], padding='SAME') + b

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
        b = tf.get_variable(name='b',shape=[m],dtype=tf.float32,initializer=tf.constant_initializer(value=0.0))
        return tf.nn.conv2d_transpose(x, kernel, output_shape=output_shape, strides=[1, stride_h, stride_w, 1], padding='SAME') + b

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

def lrelu(x,leak=0.2):
    return tf.maximum(x,leak*x)


NB = 32 # batch
NZ = 100 # input dim
NH = 28 # image height
NW = 28 # image width
NC = 1 # image channels
NG = [512,128]
ND = [128,512]



# for cats
NB = 64
NH = 64                         # image height
NW = 64                         # image width
NC = 3                          # image channels
NG = [32,16,8] # not hugely many
NG = [64, 32, 16]
NG = [256,64,32]
NG = [512,256,128,64]
ND = NG[::-1] # a nice mirror image

# retest with mnist, but with list in generator
#NG = [16,8] # note that the generator will output NC channels
#NH = 28
#NW = 28
#NC = 1
#ND = [32,16] # note that the discriminator will output 1 number

# make sure its compatible with image size
def check_layers(N,name):
    
    for i,n in enumerate(N):
        if NH/2**i != float(NH)/float(2**i) or NW/2**i != float(NW
)/float(2**i):
            raise ValueError('{} layers are not compatible with image size.  Note that only up and downsampling by a factor of 2 are available currently.'.format(name))
check_layers(NG,'Generator')
check_layers(ND,'Discriminator')
# I don't think this is right for checking based on including NC


# instead of specifying each layer, I'd like to give a list
# then I can make an arbitrary number of layers


def generator(z,reuse=False,is_training=None):
    if is_training is None:
        is_training = not reuse
    print('Creating generator with reuse {} and is_training {}.'.format(reuse,is_training))
    # get the width and height of each layer
    NH_list = [NH/(2**i) for i in range(len(NG)+1)]
    NH_list = NH_list[::-1]
    NW_list = [NW/(2**i) for i in range(len(NG)+1)]
    NW_list = NW_list[::-1]

    with tf.variable_scope('generator',reuse=reuse) as scope:
        # first we have a linear layer mapping dimension up to NG[0]
        h = tf.reshape(tf.nn.relu(batchnorm(affine(z,NG[0]*NH_list[0]*NW_list[0],name='h0'),name='h0',is_training=is_training)), [-1, NH_list[0], NW_list[0], NG[0]])
        print('...Created affine connected layer from random input to {} channels.'.format(NG[0]))
        # now we iterate through layers
        for i in range(1,len(NG)):
            h = tf.nn.relu(batchnorm(conv2dT(h,NG[i],name='h{}'.format(i)),name='h{}'.format(i),is_training=is_training))
            print('...Created relu transposed convolution layer with batchnorm and {} channels.'.format(NG[i]))

        # last we go to NC (channels)
        h = tf.nn.tanh(conv2dT(h,NC,name='ho'))*0.5 + 0.5
        print('...Created tanh transposed convolution layer with {} channels.'.format(NC))
        print('Done.')
        return h



        
def discriminator(image,reuse=False):
    print('Creating discriminator with reuse {}.'.format(reuse))

    # as before, first get sizes
    NH_list = [NH/(2**i) for i in range(len(ND)+1)]
    NW_list = [NW/(2**i) for i in range(len(ND)+1)]
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        # loop through
        for i in range(len(ND)):
            if i == 0:  # no batchnorm
                h = lrelu(conv2d(image,ND[i],name='h0'))
                print('...Created leaky relu convolution layer with {} channels.'.format(ND[i]))
            else:
                h = lrelu(batchnorm(conv2d(h,ND[i],name='h{}'.format(i)),name='h{}'.format(i)))
                print('...Created leaky relu convolution layer with batchnorm and {} channels.'.format(ND[i]))

        # output 1 number
        h = affine(tf.reshape(h,[-1,ND[-1]*NH_list[-1]*NW_list[-1]]),1,name='ho')
        print('...Created affine layer with one logit output.')
        print('Done.')
        return h

z = tf.placeholder(dtype=tf.float32,shape=[NB,NZ])
g = generator(z)
s = generator(z,reuse=True) # a sampler
image = tf.placeholder(dtype=tf.float32,shape=[NB,NH,NW,NC])
d_true = discriminator(image)
d_false = discriminator(g,reuse=True)

# get the loss function
l_d_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_true,tf.ones_like(d_true)))
l_d_false = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_false,tf.zeros_like(d_true)))
l_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_false,tf.ones_like(d_true))) # better to give wrong label then do negative of d_false

l_d = l_d_true + l_d_false

l_d_summary = tf.scalar_summary('l_d_summary',l_d)

l_g_summary = tf.scalar_summary('l_g_summary',l_g)

# optimizers
variables = tf.trainable_variables()
learning_rate = 0.0002
momentum = 0.5
optimize_d = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1 = 0.5).minimize(l_d,var_list=[v for v in variables if 'discriminator' in v.name])
optimize_g = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1 = 0.5).minimize(l_g,var_list=[v for v in variables if 'generator' in v.name])




# now we need to do some training
if use_mnist:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # I should obviously SAVE my network
    # then I can load it back here
    n_summary = 10
    print('beginning training, will print summary every {} iterations'.format(n_summary))
    for i in range(100000):
        z_train = np.random.random([NB,NZ])

        # I'd like to be able to this with things other than mnist
        if use_mnist:
            image_train = mnist.train.next_batch(NB)[0]
            image_train.shape=[NB,NH,NW,NC]
        else:
            image_train = imageLoader.next_batch(NB)

        sess.run(optimize_d,feed_dict={z:z_train,image:image_train})
        
        # I'm not sure if I should resample z_train here
        # not resampling seems to work
        #z_train = np.random.random([NB,NZ])
        sess.run(optimize_g,feed_dict={z:z_train})
        
        #z_train = np.random.random([NB,NZ])
        sess.run(optimize_g,feed_dict={z:z_train})

        # I don't understand tensorflow summary
        if not i%n_summary:
            print('iteration: {}'.format(i))
            print('discriminator loss: {}'.format(l_d.eval(feed_dict={z:z_train,image:image_train})))
            print('generator loss: {}'.format(l_g.eval(feed_dict={z:z_train})))
            # do a square number less than or equal to NB
            NS = 64
            # apparantly I can't change the batch size, probably the way I set it up, no biggy, just choose NS less than NB
            z_train = np.random.random([NB,NZ])
            I = s.eval(feed_dict={z:z_train})
            plt.clf()
            if NC == 1:
                plt.imshow(I[1,:,:,0],cmap='gray',interpolation='none')
            else:
                plt.imshow(I[0,:,:,:],interpolation='none')
            plt.pause(0.1)
            # now I'd like to write something out!
            SNS = int(np.sqrt(NS))
            J = np.zeros([NH*SNS,NW*SNS,NC])
            count = 0
            for j in range(SNS):
                for k in range(SNS):
                    J[j*NH:(j+1)*NH,k*NW:(k+1)*NW,:] = I[count,:,:,:]
                    count += 1
            if NC == 3:
                plt.imsave('iter_{}.png'.format(i),J)
            elif NC == 1:
                plt.imsave('iter_{}.png'.format(i),np.tile(J,[1,1,3]))
            
            
    
    




