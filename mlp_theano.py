from __future__ import division
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

rng =  np.random.RandomState(1234)
mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data.astype('float32'), mnist.target.astype('int32'))

mnist_X = mnist_X / 255.0

train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2)
def homework(train_X, test_X, train_y):
    #--- Multi Layer Perceptron
    class Layer(object):
        #- Constructor
        def __init__(self, in_dim, out_dim, function):
            self.in_dim   = in_dim
            self.out_dim  = out_dim
            self.function = function
            self.W        = theano.shared(rng.uniform(low=-np.sqrt(6./(in_dim+out_dim)), high=np.sqrt(6./(in_dim+out_dim)), size=(in_dim, out_dim)).astype(np.float32), name="W")
            self.b        = theano.shared(np.zeros(out_dim).astype(np.float32), name="b")
            self.params   = [self.W, self.b]
    
        #- Forward Propagation
        def f_prop(self, x):
            self.z = self.function(T.dot(x, self.W)+self.b)
            return self.z
    
    class Dropout(object):
        train = True
        def __init__(self, dropout_ratio):
            self.dropout_ratio = dropout_ratio
            self.params = []

        def f_prop(self, x):
            if not Dropout.train:
                return x
            scale = np.float32(1./(1-self.dropout_ratio))
            flag = np.random.rand(*x.shape) >= self.dropout_ratio
            self.mask = scale*flag
            return x*self.mask


    #--- Stochastic Gradient Descent
    class Sgd(object):
        def __init__(self, lr=np.float32(0.01)):
            self.lr = lr
        def __call__(self, params, g_params):
            updates = OrderedDict()
            for param, g_param in zip(params, g_params):
                updates[param] = param - self.lr*g_param
            return updates
    
    #--- Adagrad
    class Adagrad(object):
        def __init__(self, params, lr=np.float32(0.01), eps=np.float32(1e-8)):
            self.g2_sums = [theano.shared(np.zeros(param.shape.eval()).astype(np.float32)) for param in params]
            self.lr = lr
            self.eps = eps
    
        def __call__(self, params, g_params):
            updates = OrderedDict()
            for param, g_param, g2_sum in zip(params, g_params, self.g2_sums):
                g2_sum += g_param*g_param
                updates[param] = param-self.lr/(T.sqrt(g2_sum)+self.eps)*g_param
            return updates
    
    #---- RMSProp
    class Rmsprop(object):
        def __init__(self, params, lr=np.float32(0.01), alpha=np.float32(0.99), eps=np.float32(1e-8)):
            self.g2_sums = [theano.shared(np.zeros(param.shape.eval()).astype(np.float32)) for param in params]
            self.lr = lr
            self.alpha = alpha
            self.eps = eps
        def __call__(self, params, g_params):
            updates = OrderedDict()
            for param, g_param, g2_sum in zip(params, g_params, self.g2_sums):
                g2_sum = self.alpha*g2_sum+(np.float32(1)-self.alpha)*g_param*g_param
                updates[param] = param-self.lr/(T.sqrt(g2_sum)+self.eps)*g_param
            return updates

    def relu(x):
        return T.switch(T.gt(x, 0.), x, 0.)

    def leakly_relu(x):
        return T.switch(T.gt(x, 0.), x, x*np.float32(0.2))

    train_y = np.eye(10)[train_y.astype(np.int32)]
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2)
    
    layers = [
        Layer(784, 512, leakly_relu),
        Layer(512, 512, leakly_relu),
        Layer(512, 512, leakly_relu),
        Layer(512, 10, T.nnet.softmax)
    ]
    
    x = T.fmatrix('x')
    t = T.imatrix('t')
    
    params = []
    for i, layer in enumerate(layers):
        params += layer.params
        if i == 0:
            layer_out = layer.f_prop(x)
        else:
            layer_out = layer.f_prop(layer_out)
    
    y = layers[-1].z
    cost = T.mean(T.nnet.categorical_crossentropy(y, t))
    
    g_params = T.grad(cost=cost, wrt=params)
    
    optimizer = Rmsprop(params, eps=np.float32(0.02))
    updates = optimizer(params, g_params)
    
    train = theano.function(inputs=[x, t], outputs=cost, updates=updates, allow_input_downcast=True, name='train')
    valid = theano.function(inputs=[x, t], outputs=[cost, T.argmax(y, axis=1)], allow_input_downcast=True, name='valid')
    test  = theano.function(inputs=[x], outputs=T.argmax(y, axis=1), allow_input_downcast=True, name='test')
    
    batch_size = 100
    n_batches = train_X.shape[0]//batch_size
    for epoch in xrange(7):
        train_X, train_y = shuffle(train_X, train_y)
        for i in xrange(n_batches):
            start = i*batch_size
            end = start + batch_size
            train(train_X[start:end], train_y[start:end])
        valid_cost, pred_y = valid(valid_X, valid_y)
        print 'EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, axis=1).astype('int32'), pred_y, average='macro'))
    return test(test_X)

print f1_score(test_y, homework(train_X, test_X, train_y), average='micro')
