#!/usr/bin/python3

import gym
import tensorflow as tf
import numpy as np
import random

def weight_var(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=.1))

def bias_var(shape):
        return tf.Variable(tf.constant(.1, shape=shape))

class NeuralNetwork():
	def __init__(self, layer_sizes, learn_rate, session):
		#layer_sizes includes input and output layers
		self.sess = session

		self.num_weights = len(layer_sizes)-1
		self.weights = []
		self.biases = []
		for i in range(len(layer_sizes)-1):
			self.weights.append(weight_var([layer_sizes[i],layer_sizes[i+1]]))
			self.biases.append(bias_var([layer_sizes[i+1]]))

		self.x = tf.placeholder('float',[None,layer_sizes[0]])
		self.y = tf.placeholder('float',[None,layer_sizes[-1]])
		self.learn_rate = learn_rate
		self.pred = self.setup_network(self.x)
		self.train_step = self.setup_train()

		self.sess.run(tf.global_variables_initializer())

	def setup_network(self, x):
		for i in range(self.num_weights):
			if i == 0:
				h = tf.nn.relu(tf.matmul(x, self.weights[0]) + self.biases[0])
			elif i == self.num_weights-1:
				h = tf.matmul(h, self.weights[i]) + self.biases[i]
			else:
				h = tf.nn.relu(tf.matmul(h, self.weights[i]) + self.biases[i])
		return h

	def forward_pass(self, x):
		return self.sess.run(self.pred, feed_dict={self.x:x})

	def setup_train(self):
		cost = tf.reduce_mean(tf.squared_difference(self.y, self.pred))
		self.cost = cost
		train = tf.train.AdamOptimizer(self.learn_rate).minimize(cost)
		return train

	def train(self, x, y, p=False):
		if p:
			#print the cost function
			print(self.sess.run(self.cost, feed_dict={self.x:x,self.y:y}))
		self.sess.run(self.train_step, feed_dict={self.x:x,self.y:y})

	def setup_copy(self, nn):
		weight_copy = [self.weights[i].assign(nn.weights[i]) for i in range(len(self.weights))]
		bias_copy = [self.biases[i].assign(nn.biases[i]) for i in range(len(self.biases))]
		self.copy = weight_copy + bias_copy

	def run_copy(self):
		for c in self.copy:
			self.sess.run(c)

class EnvWrapper():
	def __init__(self, env, nn1, nn2, gamma=.9, max_replay_size=10000):
		self.env = env
		self.a_size = env.action_space.n
		self.o_size = env.observation_space.shape[0]
		self.nn1 = nn1
		self.nn2 = nn2
		self.gamma = gamma
		self.replays = []
		self.max_replay_size = max_replay_size
		self.last_run = []

		self.observation = self.action = self.reward = self.new_obs = None
		self.done = True

	def reset(self):
		self.observation = self.env.reset()
		self.action = self.reward = self.new_obs = None
		self.done = False

	def choose_action(self, epsilon=0):
		if random.random() < epsilon:
			self.action = self.env.action_space.sample()
		else:
			qvals = self.nn1.forward_pass(np.reshape(self.observation, [1,self.o_size]))
			self.action = np.argmax(qvals)

	def step(self):
		self.new_obs, self.reward, self.done, _ = self.env.step(self.action)

	def add_replay(self):
		if len(self.replays) >= self.max_replay_size:
			self.replays.pop(0)
		self.replays.append((self.observation, self.action, self.reward, self.new_obs, self.done))

	def update_obs(self):
		self.observation = self.new_obs
		self.new_obs = None

	def run(self, epsilon=0, render=False):
		self.reset()
		reward = 0
		while not self.done:
			if render:
				env.render()
			self.choose_action(epsilon)
			self.step()
			self.add_replay()
			self.update_obs()
			reward += self.reward
		return reward

	def get_train_target(self, replay):
		observation, action, reward, new_obs, done = replay
		observation = np.reshape(observation, [1,self.o_size])
		qvals = self.nn1.forward_pass(observation)
		new_obs = np.reshape(new_obs, [1,self.o_size])
		maxQ = np.max(self.nn2.forward_pass(new_obs))
		if not done:
			reward += maxQ * self.gamma
		y = np.reshape(qvals, [self.a_size])
		y[action] = reward
		return y

	def get_train_batch(self, batch_size):
		#add the last run
		#last_run = np.array(self.last_run)
		#last_x = np.vstack(last_run[:,0])
		#last_y = np.array([self.get_train_target(step) for step in last_run])

		#add a random sample from the replay
		l = len(self.replays)
		if l > batch_size:
			b = random.sample(range(l), batch_size)
		else:
			b = list(range(l))
		replays = np.array(self.replays)[b]
		replays_x = np.vstack(replays[:,0])
		replays_y = np.array([self.get_train_target(replay) for replay in replays])

		#batch_x = np.concatenate((last_x, replays_x),axis=0)
		#batch_y = np.concatenate((last_y, replays_y),axis=0)
		#return batch_x, batch_y
		return replays_x, replays_y

def accuracy(pred, y):
	correct = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	return tf.reduce_mean(tf.cast(correct, 'float'))

env = gym.make('CartPole-v1')
a_size = env.action_space.n
o_size = env.observation_space.shape[0]
#Tune the number and size of hidden layers
layers = [o_size,80,a_size]
learn_rate = .001
sess = tf.InteractiveSession()
nn1 = NeuralNetwork(layers, learn_rate, sess)
nn2 = NeuralNetwork(layers, learn_rate, sess)
nn2.setup_copy(nn1)
ew = EnvWrapper(env, nn1, nn2, gamma=.95)
sess.graph.finalize()

#Tune these parameters as necessary
epochs = 10000
batch_size = 50
epsilon = .8
epsilon_final = .1
epsilon_dec = .005
train_steps = 20
copy_period = 1
for epoch in range(epochs):
	reward = ew.run(epsilon)
	x, y = ew.get_train_batch(batch_size)
	for i in range(train_steps):
		ew.nn1.train(x,y,p=(i==0 or i==train_steps-1))
	print('finished epoch {}, total reward {}'.format(epoch, reward))
	if epsilon > epsilon_final:
		epsilon -= epsilon_dec
	if epoch % copy_period == 0:
		nn2.run_copy()

print(ew.run())
