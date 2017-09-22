#!/usr/bin/python3

import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import expit
import time

#add a column of ones at the beginning of a matrix
def add_ones(x):
    return np.concatenate([np.ones([x.shape[0],1]),x], axis=1)

#using a sigmoid function
def g(x):
    return expit(x)

#returns the hypothesis of the neural network using forward propagation
#theta is a list of 2d numpy arrays containing the weights
#x is a matrix, rows are training examples and columns are features
#x does not have a bias term
def h(theta, x):
    for theta_j in theta:
        x = add_ones(x)
        x = g(np.dot(x, theta_j.T))
    return x

#returns a list of weights initialized to random values
#hidden_layers is a list of ints containing the number of nodes in each hidden layer
def initialize_theta(num_inputs, num_outputs, hidden_layers):
    theta = []
    layers = [num_inputs] + hidden_layers + [num_outputs]
    for i in range(len(layers)-1):
        theta.append(2*np.random.rand(layers[i+1], layers[i]+1)-1) #-1 to 1
    return theta

#cost function for neural network
#theta is the list of weights
#x is the training data
#y is the expected results
#lda is lambda, the regularization parameter
def J(theta, x, y, lda):
    _h = h(theta, x)
    m = x.shape[0] #number of training examples
    #add the logistic cost function
    j = (-1/m) * np.sum(y*np.log(_h) + (1-y)*np.log(1-_h))
    #add the regularization term
    j += (lda/2/m) * sum([np.sum((theta_j*theta_j)[1:,:]) for theta_j in theta])
    return j

#performs the back propagation algorithm
#returns a list with the same dimensions as theta
#theta is the list of weights
#x is the training data
#y is the expected results
#lda is lambda, the regularization parameter
def back_propagate(theta, x, y, lda):
    #gradient is the average of all the gradient_i for each training example
    #has the same dimensions as theta
    gradient = []
    for theta_j in theta:
        gradient.append(np.zeros(theta_j.shape))

    #iterate through all training examples
    for ex in range(len(x)):
        #a is a list of the output for each layer
        a = [x[ex]]
        for theta_j in theta:
            #appending the 1 is the bias term
            a_l = g(np.dot(theta_j, np.append(1,a[-1])))
            a.append(a_l)

        #delta is a list of the "error" vectors for each layer
        delta = [a[-1]-y[ex]]
        #iterate backwards but append to delta forwards, so we will reverse delta at the end
        for i in range(len(theta)-1, 0, -1):
            #don't include bias when calculating delta_l
            delta_l = np.dot(theta[i].T[1:], delta[-1]) * a[i] * (1-a[i])
            delta.append(delta_l)
        delta = delta[::-1] #reverse delta

        #gradient_i is a list of matrices containing the partial derivatives for the current training set
        #it will have the same dimensions as theta
        gradient_i = []
        for i in range(len(theta)):
            #the algorithm uses delta of the next layer, but there is no delta layer 0
            #so delta[i] will give the correct value
            delta_l = delta[i].reshape([delta[i].shape[0],1]) #make it vertical
            a_l = add_ones(a[i].reshape([1,a[i].shape[0]])) #make it horizontal and add bias
            gradient_i_l = np.dot(delta_l, a_l)
            gradient_i.append(gradient_i_l)

        #update the overall gradient
        for i in range(len(gradient)):
            gradient[i] += gradient_i[i]

    m = x.shape[0] #number of training examples
    #average and regularize gradient
    for i in range(len(gradient)):
        reg = lda * theta[i] #regularization term
        reg[:,0] = 0 #don't regularize bias terms
        gradient[i] += reg
        gradient[i] /= m
    return gradient

#vectorized back propagation that performs the algorithm on batches of training data
#batch_size is the number of training examples to do at a time
def back_propagate_batch(theta, x, y, lda, batch_size):
    gradient = []
    for theta_j in theta:
        gradient.append(np.zeros(theta_j.shape))

    m = x.shape[0]

    for n in range(0, m, batch_size):
        x_b = x[n:n+batch_size] #get the batch of training sets
        actual_b_size = x_b.shape[0]
        #in case the batch has fewer sets because batch size doesn't divide m evenly

        #a is a list of the outputs for each layer in the network
        #each item in the list is a matrix with (number of training sets) rows and (size of layer) cols
        a = [x_b]
        #initialize a by performing forward propagation
        for theta_j in theta:
            a_l = g(np.dot(add_ones(a[-1]), theta_j.T))
            a.append(a_l)

        #delta is a list of the "error" terms associated with the nodes, same shape as a
        delta = [a[-1] - y[n:n+batch_size]]
        for i in range(len(theta)-1, 0, -1):
            #don't include bias
            delta_l = np.dot(delta[-1], theta[i][:,1:]) * a[i] * (1-a[i])
            delta.append(delta_l)
        delta = delta[::-1] #reverse the list delta

        for i in range(len(theta)):

            #multiply the stacks of matrices together
            #gradient_batch_l = np.matmul(delta_l, a_l)
            gradient_batch = np.zeros(theta[i].shape)
            for j in range(actual_b_size):
                delta_l = delta[i][j].reshape([delta[i][j].shape[0],1])
                a_l = add_ones(a[i][j].reshape([1,a[i][j].shape[0]]))

                gradient_batch += np.dot(delta_l, a_l)
            gradient[i] += gradient_batch

    #average and regularize gradient
    for i in range(len(gradient)):
        reg = lda * theta[i] #regularization term
        reg[:,0] = 0 #don't regularize bias terms
        gradient[i] += reg
        gradient[i] /= m
    return gradient


#a temporary function to check if back propagation is implemented correctly
#returns the approximate gradient for the given weight
#l is layer, i is output node, j is input node
def gradient_check(theta, x, y, lda, l, i, j):
    epsilon = .0001
    theta_plus = [np.copy(a) for a in theta]
    theta_plus[l][i,j] += epsilon
    theta_minus = [np.copy(a) for a in theta]
    theta_minus[l][i,j] -= epsilon
    return (J(theta_plus,x,y,lda) - J(theta_minus,x,y,lda))/2/epsilon

#performs gradient descent to minimize cost function, modifies and returns theta
#alpha is learning rate
def gradient_descent(theta, x, y, alpha):
    for i in range(10000):
        gradient = back_propagate_batch(theta, x, y, 1, 100)
        print('on iteration',i+1)
        for j in range(len(theta)):
            theta[j] -= alpha*gradient[j]
        if i%10==0:
            print('cost is:',J(theta,x,y,1))
    return theta

#gives a prediction for the inputs x and theta
def predict(theta, x):
    hypot = h(theta, x)
    maxes = np.argmax(hypot, axis=1)
    m = x.shape[0]
    prediction = np.zeros([m,10])
    prediction[np.arange(m), maxes] = 1
    return prediction

#returns the percentage of the prediction correct
def accuracy(prediction, y):
    correct = np.all(prediction==y, axis=1) #a 1d array of booleans indicating if the prediction matches y
    return np.sum(correct)/correct.shape[0]

with open('train-labels-idx1-ubyte', 'rb') as train_lbl:
    magic, num = struct.unpack('>II', train_lbl.read(8))
    labels = np.fromfile(train_lbl, dtype=np.uint8)
    #initialize y to be a matrix with each row representing a digit
    #for example a row representing 2 would be [0,0,1,0,0,0,0,0,0,0]
    y = np.zeros([num, 10])
    for i in range(len(labels)):
        y[i,labels[i]] = 1

with open('train-images-idx3-ubyte', 'rb') as train_img:
    magic, num, rows, cols = struct.unpack('>IIII', train_img.read(16))
    images = np.fromfile(train_img, dtype=np.uint8).reshape(num, rows, cols)
    x = np.reshape(images, [num, rows*cols]).astype('float')

with open('t10k-labels-idx1-ubyte', 'rb') as test_lbl:
    magic, num = struct.unpack('>II', test_lbl.read(8))
    test_labels = np.fromfile(test_lbl, dtype=np.uint8)
    test_y = np.zeros([num, 10])
    for i in range(len(test_labels)):
        test_y[i,test_labels[i]] = 1

with open('t10k-images-idx3-ubyte', 'rb') as test_img:
    magic, num, rows, cols = struct.unpack('>IIII', test_img.read(16))
    test_images = np.fromfile(test_img, dtype=np.uint8).reshape(num, rows, cols)
    test_x = np.reshape(test_images, [num, rows*cols]).astype('float')


#x = x[:10001,:]
#y = y[:10001,:]
n = x.shape[1] #number of features
theta = initialize_theta(n, 10, [200])

theta = gradient_descent(theta,x,y,.1)

prediction = predict(theta, test_x)
print(accuracy(prediction, test_y))

p = predict(theta, x)
print(accuracy(p, y))

img1 = images[0]
plt.imshow(img1, cmap='gray')
#plt.show()


