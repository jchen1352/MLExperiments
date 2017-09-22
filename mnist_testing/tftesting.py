#!/usr/bin/python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder('float', [None,784])
y = tf.placeholder('float', [None,10])

w1 = tf.Variable(tf.random_normal([784,200]))
b1 = tf.Variable(tf.zeros([200]))
w2 = tf.Variable(tf.random_normal([200,10]))
b2 = tf.Variable(tf.zeros([10]))

h1 = tf.nn.relu(tf.matmul(x,w1)+b1)
pred = tf.matmul(h1,w2)+b2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train = tf.train.GradientDescentOptimizer(.5).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))

batch_size = 100

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(20):
	print('on iteration',i)
	num_batches = int(mnist.train.num_examples/batch_size)
	for j in range(num_batches):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		sess.run(train, feed_dict={x:batch_x, y:batch_y})
	if (i%10 == 0):
		print('cost is',sess.run(cost, feed_dict={x:mnist.train.images, y:mnist.train.labels}))
		print('accuracy on train is',sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels}))

print('accuracy on test is',sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels}))
