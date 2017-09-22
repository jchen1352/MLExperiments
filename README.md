This is the result of about a month of teaching myself a bit about machine learning. It contains three folders that correspond to three areas that I studied.

The folder ml_testing contains a simple logistic regression model trained on the Iris dataset, and was my first step into machine learning.

The folder mnist_testing contains various neural network models designed to solve the MNIST digit recognition task. nntesting.py contains a neural network implemented from scratch with numpy. Unfortunately, the network takes far too long to train, especially if numpy is not compiled with OpenBLAS integration for faster matrix multiplication, and the network seems only to work when trained on the entire dataset instead of random batches. The file theta_trained_1.npy contains the result of training the network for about two hours, and it achieves roughly 93 percent accuracy on the test set.

I then experimented with implementing neural networks in Tensorflow. tftesting.py contains another basic fully-connected neural network that trains much faster on the training data. cnntesting.py is a basic convolutional neural network that achieves better results on the MNIST data.

Finally, the folder drl_testing consists of my exploration into reinforcement learning. I continued to use Tensorflow for this. I started with qlearntest.py, a q-learning neural network that learns to play gridworld, a simple grid based game taken from a website. The network wins more after playing more games.

The program I spent the most time on is openai_gym_test.py. It also contains a q-learning neural network, and it aims to solve the Open AI gym Cart-Pole environment. The network successfully reaches the max score quite consistently, although the network does appear to have some problems with catastrophic interference in that it sometimes aprubtly gets low scores before rebounding back. Overall, I am quite pleased with its performance.
