#!/usr/bin/python3

import tensorflow as tf

"""Testing convolutional neural networks
Didn't feel like running it through, so I don't know if it actually works"""

def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape))

def bias_variable(shape):
	return tf.Variable(tf.constant(.1, shape=shape))

def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

w_conv1 = weight_variable([5,5,1,8])
b_conv1 = bias_variable([8])

#convolution
x_4d = tf.reshape(x, [-1,28,28,1])
h1 = conv2d(x_4d, w_conv1) + b_conv1
#relu
h1 = tf.nn.relu(h1)
#pooling
h1 = max_pool_2x2(h1)

w_conv2 = weight_variable([5,5,8,16])
b_conv2 = bias_variable([16])

h2 = max_pool_2x2(tf.nn.relu(conv2d(h1, w_conv2)+b_conv2))

w_fc1 = weight_variable([7*7*16,1024])
b_fc1 = bias_variable([1024])
h2_flat = tf.reshape(h2, [-1,7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h2_flat, w_fc1)+b_fc1)

w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
pred = tf.matmul(h_fc1, w_fc2)+b_fc2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train = tf.train.AdamOptimizer(1e-4).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

batch_size = 100

for i in range(20):
	print('on iteration',i)
	num_batches = int(mnist.train.num_examples/batch_size)
	print(num_batches,'batches to do')
	for j in range(num_batches):
		print('on batch',j)
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		sess.run(train, feed_dict={x:batch_x, y:batch_y})
	#if (i%10 == 0):
		#print('cost is',sess.run(cost, feed_dict={x:mnist.train.images, y:mnist.train.labels}))
		#print('accuracy on train is',sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels}))

print('accuracy on test is',sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels}))

	
