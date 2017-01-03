import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.interactive(True)

batch_size = 128 # from paper
batch_size = 32
batch_size = 64

random_dimension = 100
random_dimension = 50 # from paper

# image size
image_row = 28
image_col = 28
image_c = 1

# convolution
conv_size = 5

# generator hidden layers (output is 1 image, or 3 for color)
NG = [64,32,image_c]  
# for now I'll do image_c as output, maybe I can figure out something else later
# note paper says 1024!
# code says 1024 then 64


# discriminator hidden layers (output is 1)
ND = [32, 64]

# helper functions
def rand_variable(shape,**input_dict):
    return tf.Variable( tf.truncated_normal(shape, stddev=0.02) , **input_dict)

def const_variable(shape,value=0.1,**input_dict):
    return tf.Variable(tf.constant(value,shape=shape), **input_dict)



def batch_normalize(x,w,b,**input_dict):
  ''' 
  batch normalize the input to a layer
  I also need to introduce more parameters so I can output the identity transform
  w and b are input scale and shift so we can recover identity
  add bias at end so I can give it a name
  '''
  xbar = tf.reduce_mean(x,0) # the batch axis
  x0 = x - xbar
  x02 = x0*x0
  e = 0.01
  tmp = x0 / tf.sqrt(tf.reduce_mean(x02,0) + e*e) * w 
  return tf.add(tmp,b,**input_dict)


z = tf.placeholder(tf.float32, shape=[None,random_dimension],name='z')

# note about reuse in scope
# if it is True, we will search for a variable and raise an error if itis not found
# if it is false, we will create a variable and raise an error if it exists



def generator(z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse) as scope:
        batch_size = tf.shape(z)[0]
        wup = tf.get_variable(name='wup', shape=[random_dimension,NG[0]*image_row/4*image_col/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        zup = tf.matmul(z, wup, name='zup')
        bup = tf.get_variable(name='bup', shape=[NG[0]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
        image0 = tf.add(tf.reshape(zup, shape=[-1,image_row/4,image_col/4,NG[0]]), bup,name='image0')
        # batch normalize it
        #wb0 = tf.get_variable(name='wb0', shape=[image_row/4,image_col/4,NG[0]], dtype=tf.float32, initializer=tf.constant_initializer(value=1.0))
        #bb0 = tf.get_variable(name='bb0', shape=[image_row/4,image_col/4,NG[0]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        #image0norm = batch_normalize(image0,wb0,bb0, name='image0norm')
        # no need to normalize because there is no covariate shift
        image0norm = image0


        # relu
        image1 = tf.nn.relu(image0norm,name='image1')
        
        # now convolutions
        w1 = tf.get_variable(name='w1',shape=[conv_size,conv_size,NG[1],NG[0]],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02)) # note order of the N's is backwards for the transpose
        b1 = tf.get_variable(name='b1', shape=[NG[1]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
        image1conv = tf.add(tf.nn.conv2d_transpose(image1,w1,output_shape=[batch_size,image_row/2,image_col/2,NG[1]],strides=[1,2,2,1],padding='SAME') , b1, name='image1conv')
        wb1 = tf.get_variable(name='wb1', shape=[image_row/2,image_col/2,NG[1]], dtype=tf.float32, initializer=tf.constant_initializer(value=1.0))
        bb1 = tf.get_variable(name='bb1', shape=[image_row/2,image_col/2,NG[1]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        image1norm = batch_normalize(image1conv,wb1,bb1,name='image1norm')
        image2 = tf.nn.relu(image1norm,name='image2')

        # now output an image of channels image_c
        w2 = tf.get_variable(name='w2',shape=[conv_size,conv_size,NG[2],NG[1]], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable(name='b2', shape=[NG[2]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
        image2conv = tf.add(tf.nn.conv2d_transpose(image2,w2,output_shape=[batch_size,image_row,image_row,NG[2]],strides=[1,2,2,1],padding='SAME'),b2,name='image2conv')
        wb2 = tf.get_variable(name='wb2',shape=[image_row,image_col,NG[2]],dtype=tf.float32,initializer=tf.constant_initializer(value=1.0))
        bb2 = tf.get_variable(name='bb2',shape=[image_row,image_col,NG[2]],dtype=tf.float32,initializer=tf.constant_initializer(value=0.0))
        image2norm = batch_normalize(image2conv,wb2,bb2,name='image2norm')
        # scale from 0 to 1
        image3 = tf.add(tf.mul(tf.tanh(image2norm), 0.5),0.5,name='image3')

        return image3
    
        
G = generator(z)
'''                
# some tests        



tvars = tf.trainable_variables()
print([t.name for t in tvars])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    I = G.eval(feed_dict={z:np.random.random([batch_size,random_dimension])})
plt.imshow(I[0,:,:,0],cmap='gray')
plt.pause(0.1)
'''


def discriminator(image,reuse=False):
    ''' here we need a reuse, because we'll have two of these networks and want to share variables.  The second time and subsequent we call reuse '''
    with tf.variable_scope('discriminator',reuse=reuse) as scope:
        batch_size = tf.shape(image)[0]

        w0 = tf.get_variable(name='w0', shape=[5,5,image_c,ND[0]], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        b0 = tf.get_variable(name='b0', shape=[ND[0]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
        image0 = tf.add(tf.nn.conv2d(image, w0, strides=[1,2,2,1], padding='SAME'), b0, name='image0')
        # don't normalize because no covariate shift
        image0norm = image0
        image1 = tf.nn.relu(image0norm,name='image1')
        
        # next hidden layer
        w1 = tf.get_variable(name='w1', shape=[5,5,ND[0],ND[1]], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable(name='b1', shape=[ND[1]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
        image1 = tf.add(tf.nn.conv2d(image1, w1, strides=[1,2,2,1], padding='SAME'), b1, name='image1')
        batch_w1 = tf.get_variable(name='batch_w1', shape=[image_row/4,image_col/4,ND[1]], dtype=tf.float32, initializer=tf.constant_initializer(value=1.0))
        batch_b1 = tf.get_variable(name='batch_b1', shape=[image_row/4,image_col/4,ND[1]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        image1norm = batch_normalize(image1,batch_w1,batch_b1,name='image1norm')
        image2 = tf.nn.relu(image1norm,name='image2')



        # "flatten"
        image2flat = tf.reshape(image2,[batch_size,ND[1]*image_row/4*image_col/4],name='image3flat')
        wflat = tf.get_variable(name='wflat', shape=[ND[1]*image_row/4*image_col/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02) )
        bflat = tf.get_variable(name='bfalt', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
        
        logit = tf.add(tf.matmul(image2flat, wflat), bflat, name='logit')

        # batch normalize? no whatever
        return logit


        
        
        
Dfalse = discriminator(G) # for false cases, connect to other network

image = tf.placeholder(tf.float32,shape=[None, image_row, image_col, image_c])
Dtrue = discriminator(image,reuse=True) # for true cases


# now loss functions
loss_dis_true = tf.nn.sigmoid_cross_entropy_with_logits(Dtrue,tf.ones_like(Dtrue))
loss_dis_false = tf.nn.sigmoid_cross_entropy_with_logits(Dfalse,tf.zeros_like(Dfalse))
# instead of taking negative, and minimize, a trick is to just provide the wrong label!
loss_gen = tf.nn.sigmoid_cross_entropy_with_logits(Dfalse, tf.ones_like(Dfalse))


# now get some optimizers going on
learning_rate = 0.0002 # from paper

momentum = 0.5 # from paper

var_list_dis = [v for v in tf.trainable_variables() if 'discriminator' in v.name]
var_list_gen = [v for v in tf.trainable_variables() if 'generator' in v.name]


optimizer_dis_true = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(loss_dis_true,var_list = var_list_dis)

optimizer_dis_false = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(loss_dis_false, var_list=var_list_dis)

optimizer_gen = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(loss_gen,var_list=var_list_gen)


# load the mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# now we're ready to train!
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    ntrain = 10000
    for train_loop in xrange(ntrain):
        training_data = mnist.train.next_batch(batch_size)[0]
        training_data.shape =[batch_size,image_row,image_col,image_c]
        sess.run(optimizer_dis_true,feed_dict={image:training_data})
        sess.run(optimizer_dis_false,feed_dict={z:np.random.random([batch_size,random_dimension])})

        n_gen = 20 # code on github says 2, paper says 1 or even less than 1
        for _ in range(n_gen):
            sess.run(optimizer_gen,feed_dict={z:np.random.random([batch_size,random_dimension])})

        nshow = 100
        if not train_loop%nshow:
            print('iteration {}'.format(train_loop))
            testbatch = 100
            training_data = mnist.train.next_batch(testbatch)[0]
            training_data.shape =[testbatch,image_row,image_col,image_c]
            # get accuracy
            accuracy_true = tf.reduce_mean(tf.cast(Dtrue>=0,tf.float32))
            accuracy_false = tf.reduce_mean(tf.cast(Dfalse<0,tf.float32))
            print('negatives accuracy: {}'.format( accuracy_false.eval(feed_dict={z:np.random.random([testbatch,random_dimension])})))
            print('positives accuracy: {}'.format( accuracy_true.eval(feed_dict={image:training_data})))

            # show an example
            I = G.eval(feed_dict={z:np.random.random([batch_size,random_dimension])})
            plt.clf()
            plt.imshow(I[0,:,:,0],cmap='gray', interpolation='none')
            plt.pause(0.01)
            plt.imsave('v2_sample_iter_{:06d}.png'.format(train_loop),I[0,:,:,0],cmap='gray')            



        




#with tf.Session() as sess: tmp = D.eval(feed_dict={image:training_data})
    
    

        



# what worked?
# it seemed like it was working for the first thousand or so
# with n_gen = 2
# batch size 32
# learning rate 0.0002
# but after a thousand it just went nowhere
