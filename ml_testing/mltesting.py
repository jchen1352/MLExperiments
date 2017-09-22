#!/usr/bin/python3

import pandas as pd
import numpy as np
from scipy.special import expit

#returns the logistic hypothesis
#theta is a column vector of the parameters
#x is a matrix where the rows are training examples and the columns are features
def h(theta, x):
	return expit(np.dot(x, theta))

#returns the logistic cost function
#theta is a column vector of the parameters
#x is a matrix where the rows are training examples and the columns are features
#y is a column vector of the actual categories for the training examples
def J(theta, x, y):
	m = x.shape[0] #number of training examples
	_h = h(theta, x)
	return (np.dot(-y, np.log(_h)) - np.dot(1-y, np.log(1-_h)))/m

#returns theta after performing gradient descent
#theta is a column vector of the parameters
#x is a matrix where the rows are training examples and the columns are features
#y is a column vector of the actual categories for the training examples
#alpha is the rate of convergence
def gradient_descent(theta, x, y, alpha):
	#do a fixed number of iterations
	m = x.shape[0]
	for i in range(10000):
		theta -= np.dot(x.T, h(theta, x) - y) * alpha/m
		#if i%100 == 0:
			#print(J(theta,x,y))
			#print(theta)
	return theta

#returns 0 or 1 based on the hypothesis
#theta is a column vector of the parameters
#x is a matrix where the rows are training examples and the columns are features
def predict(theta, x):
	prediction = h(theta, x) > .5
	return prediction.astype(float)

#normalizes a matrix x based on the values of the features
def normalize(x):
	mean = np.mean(x,axis=0)
	std = np.std(x,axis=0)
	return (x-mean)/std

#add a column of ones at the beginning of a matrix
def add_ones(x):
	return np.concatenate([np.ones([x.shape[0],1]),x], axis=1)

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris_file = "iris-species/Iris.csv"
dataset = pd.read_csv(iris_file, names=names)
dataset = dataset.drop(dataset.index[0])       #drop the first row because it contains names
dataset = np.array(dataset) #convert to numpy array
dataset[dataset == 'Iris-setosa'] = 0
dataset[dataset == 'Iris-versicolor'] = 1
partial_dataset = np.concatenate([dataset[0:35],dataset[50:85]]) #get 35 training examples from each of the first 2 categories
#replace names with 0 or 1
partial_dataset = partial_dataset.astype(np.float)


x = partial_dataset[:,:-1] #cut off the last column
x = normalize(x)
x = add_ones(x) #add a column of ones for bias term
y = partial_dataset[:,-1] #get only the last column

theta = np.ones(x.shape[1]) #initialize theta to zeros
print(x.shape, y.shape, theta.shape)

theta = gradient_descent(theta, x, y, 1)
print(theta)
print(J(theta,x,y))

test_data = np.concatenate([dataset[35:50],dataset[85:100]]) #get the test data we didn't use
test_data = test_data.astype(np.float)
print(test_data.shape)

x_test = test_data[:,:-1]
x_test = normalize(x_test)
x_test = add_ones(x_test)
y_test = test_data[:,-1]
prediction = predict(theta, x_test)
print(prediction)
print(y_test)

