from __future__ import division
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

import numpy as np

mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data, mnist.target.astype('int32'))

mnist_X = mnist_X/255.0

train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2)
test_y = np.eye(test_y.max()+1)[test_y]
train_y = np.eye(train_y.max()+1)[train_y]

def homework(train_X, test_X, train_y):
    EPOCH = 10
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2)
    class Linear(object):
        def __init__(self, indim, outdim):
            self.W = np.random.normal(0, np.sqrt(1./indim), (indim, outdim))
            self.b = np.zeros(outdim)
            self.x = None
            self.y = None
            self.dW = None
            self.db = None
            self.dx = None

        def forward(self, x):
            self.y = np.dot(x, self.W)+self.b
            self.x = x
            return self.y

        def backward(self, delta):
            self.dW = np.dot(self.x.T, delta)
            self.db = np.dot(np.ones(len(self.x)), delta)
            self.dx = np.dot(delta, self.W.T)
            return self.dx

        def update(self, lr):
            self.W = self.W-lr*self.dW
            self.b = self.b-lr*self.b

    class Activation(object):
        def __init__(self, function, deriv_function):
            self.x = None
            self.y = None
            self.function = function
            self.deriv_function = deriv_function

        def forward(self, x):
            self.x = x
            self.y = self.function(x)
            return self.y

        def backward(self, delta):
            self.dx = self.deriv_function(self.x)*delta
            return self.dx

        def update(self, lr):
            pass

    def relu(x):
        return np.where(x < 0, 0, x)
    def deriv_relu(x):
        return np.where(x < 0, 0, 1)
    def softmax(x):
        exp_x = np.exp(x)
        return exp_x/np.sum(exp_x, axis=1, keepdims=True)
    def deriv_softmax(x):
        return softmax(x)*(1-softmax(x))

    def f_props(x, Layers):
        for layer in Layers:
            x = layer.forward(x)
        return x

    def b_props(delta, Layers):
        for layer in Layers[::-1]:
            delta = layer.backward(delta)

    def updates(lr, Layers):
        for layer in Layers:
            layer.update(lr)

    Layers = [Linear(784, 256), Activation(relu, deriv_relu),
            #  Linear(512, 256), Activation(relu, deriv_relu),
            #  Linear(256, 128), Activation(relu, deriv_relu),
            #  Linear(128, 64),  Activation(relu, deriv_relu),
            #  Linear(64, 32),   Activation(relu, deriv_relu),
              Linear(256, 10),   Activation(softmax, deriv_softmax)]

    def train(X, t, lr=0.01):
        y = f_props(X, Layers)
        cost = np.sum(-t*np.log(y)-(1-t)*np.log(1-y))
        delta = y-t
        b_props(delta, Layers)
        updates(lr, Layers)

        y = f_props(X, Layers)
        cost = np.sum(-t*np.log(y)-(1-t)*np.log(1-y))
        return cost, y

    def test(X, t):
        y = f_props(X, Layers)
        cost = np.sum(-t*np.log(y)-(1-t)*np.log(1-y))
        return cost, y

    for epoch in xrange(EPOCH):
        for x, y in zip(train_X, train_y):
            cost = train(x[np.newaxis, :], y[np.newaxis, :])
        cost, pred_y = test(valid_X, valid_y)
        print "epoch:", epoch
        print "cost:",  cost
        print "f1_score:", f1_score(valid_y, pred_y, average='micro')

homework(train_X, test_X, train_y)









