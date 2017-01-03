from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import numpy as np
import matplotlib.pyplot as plt
plt.close('all') # for repeated sessions
plt.ion() # interactive, use ioff for off


import tensorflow as tf


# first build classifier network
# none will be the arbitray batch size
# input
#x = tf.placeholder(tf.float32, shape=[None, 784],name='x')
# output I want to be a prob distribution for "REAL" "FAKE"
# this y will be the real data
#y_ = tf.placeholder(tf.float32, shape=[None, 2],name='y_')

# some helper functions for making variables
def weight_variable(shape,**input_dict):
  initial = tf.truncated_normal(shape, stddev=0.02) 
  return tf.Variable(initial,**input_dict)

def bias_variable(shape,**input_dict):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,**input_dict)


# convolution layers
# first is input (4d, batch width height channels)
# second is filter (4d, height width in_channels out_channels )
# strides: A list of ints. 1-D of length 4. The stride of the sliding window for each dimension of input. Must be in the same order as the dimension specified with format.
def conv2d(x, W, **input_dict):
  return tf.nn.conv2d(x, W, 
                      strides=[1, 1, 1, 1], 
                      padding='SAME',
                      **input_dict)
# max pool for downsampling
# value: A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
# ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
# strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
# padding: A string, either 'VALID' or 'SAME'. The padding algorithm. See the comment here
def max_pool_2x2(x,**input_dict):
  return tf.nn.max_pool(x, 
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], 
                        padding='SAME',
                        **input_dict)


def batch_normalize(x,w,b):
  ''' 
  batch normalize the input to a layer
  I also need to introduce more parameters so I can output the identity transform
  w and b are input scale and shift so we can recover identity
  '''
  xbar = tf.reduce_mean(x,0) # the batch axis
  x0 = x - xbar
  x02 = x0*x0
  e = 0.01
  return x0 / tf.sqrt(tf.reduce_mean(x02,0) + e*e) * w + b


# NOTE, better to normalize wx+b than x (so says paper), because mixture is more gaussian?



batch_normalize_generator = True
batch_normalize_discriminator = True


D = 50 # number of dimensions in uniform distribution
N1 = 32 # first layer
N2 = 16 # second 
N3 = 1 # third

# we need a generator network
# a network that generates 28x28 images
# by transforming a D dimensional input
z = tf.placeholder(tf.float32, shape=[None,D], name='z') # a row because we mult on right

# now we project to more features
w1 = weight_variable([D,N1*7*7],name='w1')
b1 = bias_variable([N1],name='b1')
up1 = tf.matmul(z,w1,name='up1')
if batch_normalize_generator:
  w1batch = tf.Variable(tf.constant(1.0,shape=[7,7,N1]))
  b1batch = tf.Variable(tf.constant(0.1,shape=[7,7,N1]))
  image1 = tf.nn.relu( batch_normalize(tf.reshape(up1,[-1,7,7,N1])+b1,w1batch,b1batch), name='image1')
else:
  image1 = tf.nn.relu( tf.reshape(up1,[-1,7,7,N1])+b1, name='image1')


# now we want to convolve and upsample
# use a 5x5 filter
# note that a fractionally strided convolution is a conv2d_transpose
# no need to upsample specifically
# arguments are
#value: A 4-D Tensor of type float and shape [batch, height, width, in_channels] for NHWC data format or [batch, in_channels, height, width] for NCHW data format.
#filter: A 4-D Tensor with the same type as value and shape [height, width, output_channels, in_channels]. filter's in_channels dimension must match that of value.
#output_shape: A 1-D Tensor representing the output shape of the deconvolution op.
#strides: A list of ints. The stride of the sliding window for each dimension of the input tensor.
#padding: A string, either 'VALID' or 'SAME'. The padding algorithm. See the comment here
w2 = weight_variable([5,5,N2,N1])
b2 = bias_variable([N2])
#batch_size = 10 # it seems I have to set this here
# actually I don't think I need to do this
batch_size = tf.shape(image1)[0]
if batch_normalize_generator:
  w2batch = tf.Variable(tf.constant(1.0,shape=[14, 14, N2])) # should be the size of image 1
  b2batch = tf.Variable(tf.constant(0.1,shape=[14, 14, N2]))
  image2 = tf.nn.relu( batch_normalize( tf.nn.conv2d_transpose(image1,w2,output_shape=[batch_size,14,14,N2],strides=[1,2,2,1],padding='SAME') + b2 , w2batch, b2batch) )
else:
  image2 = tf.nn.relu(tf.nn.conv2d_transpose(image1,w2,output_shape=[batch_size,14,14,N2],strides=[1,2,2,1],padding='SAME') + b2 )


# again
w3 = weight_variable([5,5,N3,N2])
b3 = bias_variable([N3])
# I may want to use tanh like the paper suggests
if batch_normalize_generator:
  w3batch = tf.Variable(tf.constant(1.0,shape=[28,28,N3])) # what is the size of I2
  b3batch = tf.Variable(tf.constant(0.1,shape=[28,28,N3]))
  image3 = tf.nn.sigmoid(batch_normalize(tf.nn.conv2d_transpose(image2,w3,output_shape=[batch_size,28,28,N3],strides=[1,2,2,1],padding='SAME') + b3, w3batch,b3batch))
else:
  image3 = tf.nn.sigmoid(tf.nn.conv2d_transpose(image2,w3,output_shape=[batch_size,28,28,N3],strides=[1,2,2,1],padding='SAME') + b3)

# I want between 0 and 1, tanh seems to be between -1 and 1
# but the paper says tanh so just do it

# now we want to connect the output of this to the input of the next

# I need to essentially have two copies of the descriminator
# one is hooked up to the generator
# and one is not
# to do this we just have to make sure we use the same variables

# my descriminator can be so simple
# input is 28x 28, output is 14x14
w3_ = weight_variable([5,5,N3,N2], name='w3_')
b3_ = bias_variable([N2],name='b3_')
# this one is hooked up

if batch_normalize_discriminator:
  w3batch_ = tf.Variable(tf.constant(1.0,shape=[14,14,N2]))
  b3batch_ = tf.Variable(tf.constant(0.1,shape=[14,14,N2]))
  image3_ = tf.nn.relu(batch_normalize(tf.nn.conv2d(image3,w3_,strides=[1,2,2,1],padding='SAME')+b3_,w3batch_,b3batch_),name='image3_')
else:
  image3_ = tf.nn.relu(tf.nn.conv2d(image3,w3_,strides=[1,2,2,1],padding='SAME')+b3_,name='image3_')

# this one is not 
image_ph = tf.placeholder(tf.float32,shape=[None,28,28,1],name='image_ph')
if batch_normalize_discriminator:
  image3__ = tf.nn.relu(batch_normalize(tf.nn.conv2d(image_ph,w3_,strides=[1,2,2,1],padding='SAME')+b3_,w3batch_,b3batch_),name='image3__')
else:
  image3__ = tf.nn.relu(tf.nn.conv2d(image_ph,w3_,strides=[1,2,2,1],padding='SAME')+b3_,name='image3__')

# but they're using the same vairables
# note use strided convolutions instead of maxpool
# NOTE two underscores is for POSITIVE, one underscore is hooked up and is for negative

# next layer, we go to 7x7
w2_ = weight_variable([5,5,N2,N1],name='w2_')
b2_ = bias_variable([N1],name='b2_')

if batch_normalize_discriminator:
  w2batch_ = tf.Variable(tf.constant(1.0,shape=[7,7,N1]))
  b2batch_ = tf.Variable(tf.constant(0.1,shape=[7,7,N1]))
  image2_ = tf.nn.relu(batch_normalize(tf.nn.conv2d(image3_,w2_,strides=[1,2,2,1],padding='SAME')+b2_,w2batch_,b2batch_),name='image2_')
  image2__ = tf.nn.relu(batch_normalize(tf.nn.conv2d(image3__,w2_,strides=[1,2,2,1],padding='SAME')+b2_,w2batch_,b2batch_),name='image2__')
else:
  image2_ = tf.nn.relu(tf.nn.conv2d(image3_,w2_,strides=[1,2,2,1],padding='SAME')+b2_,name='image2_')
  image2__ = tf.nn.relu(tf.nn.conv2d(image3__,w2_,strides=[1,2,2,1],padding='SAME')+b2_,name='image2__')

# next we go to one number
image2_flat_ = tf.reshape( image2_ , [-1, 7*7*N1 ],name='image2_flat_')
image2_flat__ = tf.reshape( image2__ , [-1, 7*7*N1 ],name='image2_flat__')
ws_ = weight_variable([7*7*N1,1],name='ws_');
bs_ = bias_variable([1],name='bs_')
scalar_ = tf.add( tf.matmul( image2_flat_ , ws_) , bs_, name='scalar_')
scalar__ = tf.add( tf.matmul( image2_flat__ , ws_) , bs_, name='scalar__')
# there's no reason to batch normalize here

# now normally I would put this through a sigmoid to get a number in 0,1
# instead we'll just use the appropriate cost function
is_real = tf.placeholder(tf.float32,shape=[None,1],name='is_real')

cross_entropy_ = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(scalar_,is_real),name='cross_entropy_')

cross_entropy__ = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(scalar__,is_real) , name='cross_entropy__')

# the negative is for maximizing
negative_cross_entropy_ = tf.mul( cross_entropy_, -1.0 , name='negative_cross_entropy_')

# I don't think the above is right, because the softmax normalizes the inputs to 0 1 by dividing by the sum!  Since I'm only doing one channell, there is nothing to sum over
# I need to use sigmoid and not softmax

# an online version of this I found did something different
# rather than using a placeholder is_real, it just inputs the appropriate term (ones or zeros)
# and rather than taking a negative, it inputs ones instead of zeros!
# i.e. instead of maximizing log(1-p) (second term), we end up minimizing log(p) (first term)



# now we'd like to have three trainings
# train discriminator on real images (minimize cross_entropy__)
# train discriminator on fake images (minimize cross_entropy_)
# train generator (maximize cross_entropy_)
# for the second two I must be sure I'm only optimizing over the right variables
# I can do this with the keyword argument var_list
step_size_gen = 0.0002*100
step_size_dis = 0.0002
# set gen step to 0 to see if I can train the dis
# the dis seems to be training with batch norm, but it's really slow
# with 1e-5 it didn't seem to move
# with 1e-4 it seems to move, but slow
# with 1e-3 it seems to train up in a couple hundred




optimizer_gen = tf.train.AdamOptimizer(step_size_gen)
optimizer_dis = tf.train.AdamOptimizer(step_size_dis)
#optimizer_gen = tf.train.GradientDescentOptimizer(step_size_gen)
#optimizer_dis = tf.train.GradientDescentOptimizer(step_size_dis)

var_list_generator=[w1,b1,w2,b2,w3,b3]
if batch_normalize_generator:
  var_list_generator = [w1,w1batch,b1,b1batch,w2,w2batch,b2,b2batch,w3,w3batch,b3,b3batch]
generator_train = optimizer_gen.minimize(negative_cross_entropy_,var_list=var_list_generator)

var_list_discriminator=[w3_,b3_,w2_,b2_,ws_,bs_]
if batch_normalize_discriminator:
  var_list_discriminator=[w3_,w3batch_,b3_,b3batch_,w2_,w2batch_,b2_,b2batch_,ws_,bs_]
discriminator_train_positive = optimizer_dis.minimize(cross_entropy__,var_list=var_list_discriminator)
discriminator_train_negative = optimizer_dis.minimize(cross_entropy_,var_list=var_list_discriminator)

# now we flatten it to 1D with a dot product
with tf.Session() as sess:
  batch_size = 128
  batch_size = 32
  sess.run( tf.initialize_all_variables() )

  niter = 1000000
  nshow = 100
  for _ in xrange(niter):
    # train descriminator
    batch_ = mnist.train.next_batch(batch_size)
    input_ = np.reshape(batch_[0],[batch_size,28,28,1])
    
    # is my generator getting any better at all?
    # if I stop optimizing my discriminator, my generator gives perfect (i.e. negative accuracy goes to zero)
    if _ < 300 or False: # test this
      sess.run(discriminator_train_positive,feed_dict={is_real:np.ones((batch_size,1)),image_ph:input_})
      #sess.run(discriminator_train_negative,feed_dict={z:np.random.random([batch_size,D]),is_real:np.zeros((batch_size,1))})
      sess.run(discriminator_train_negative,feed_dict={z:np.ones([batch_size,D]),is_real:np.zeros((batch_size,1))})

    # train generator
    # for debugging it may be interesting to see what happens if I input a constant
    n_gen = 10
    for gen_loop in range(n_gen):
      #sess.run(generator_train,feed_dict={z:np.random.random([batch_size,D]),is_real:np.zeros((batch_size,1))})
      sess.run(generator_train,feed_dict={z:np.ones([batch_size,D]),is_real:np.zeros((batch_size,1))})

    if not _%nshow:
      #out_test = image3.eval(feed_dict={z:np.random.random([batch_size,D])})
      out_test = image3.eval(feed_dict={z:np.ones([batch_size,D])})      
      plt.imshow(out_test[0,:,:,0],cmap='gray')
      #plt.imshow((out_test[0,:,:,0] - np.min(out_test[0,:,:,0]))/(np.max(out_test[0,:,:,0] - np.min(out_test[0,:,:,0]))),cmap='gray')
      plt.pause(0.1)
      plt.imsave('sample_iter_{:06d}.png'.format(_),out_test[0,:,:,0],cmap='gray')

      # set this up for display so I can see if everything's working
      #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      #print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
      testbatch = 100
      correct_prediction_  = tf.equal(tf.cast(scalar_  > 0.0,tf.float32),is_real)
      correct_prediction__ = tf.equal(tf.cast(scalar__ > 0.0,tf.float32),is_real)

      accuracy_ = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))
      accuracy__ = tf.reduce_mean(tf.cast(correct_prediction__, tf.float32))

      # recall that two underscores is for positive
      # one underscore is for negative

      #print('negatives accuracy: {}'.format( accuracy_.eval(feed_dict={z:np.random.random([testbatch,D]), is_real:np.zeros((testbatch,1))})))
      print('negatives accuracy: {}'.format( accuracy_.eval(feed_dict={z:np.ones([testbatch,D]), is_real:np.zeros((testbatch,1))})))

      print('positives accuracy: {}'.format( accuracy__.eval(feed_dict={image_ph:np.reshape(mnist.train.next_batch(testbatch)[0],[testbatch,28,28,1]), is_real:np.ones((testbatch,1))})))
      # looks like its working!
      # my classifier is working well
      # my generator is not so hot, maybe it needs a lot of training


# fixed cost function
# use sigmoid (1d) instead of softmax (nd)


# important according to paper is batch normalization
# I've implemented it
# paper also says something about fully connected layer that I don't understand


# training my generator with constant input, NOTHING HAPPENS, output is the same every time
# without batch normalization it still doesn't change
