from __future__ import division
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

import numpy as np
import theano
import theano.tensor as T

mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data.astype('float32'), mnist.target.astype('int32'))

mnist_X = mnist_X / 255.0

train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2)

def homework(train_X, test_X, train_y):
    from theano.tensor.shared_randomstreams import RandomStreams
    rng = np.random.RandomState(1234)
    theano_rng = RandomStreams(rng.randint(1234))
    class Dropout(object):
        train = True
        def __init__(self, in_dim, dropout_ratio):
            self.dropout_ratio = np.float32(dropout_ratio)
            self.in_dim = in_dim
            self.params = []
            self.encode_function = lambda x : x
        def pretraining(self, x, y):
            return 0, 0

        def f_prop(self, x):
            if not Dropout.train:
                return x
            scale = np.float32(1./(1-self.dropout_ratio))
            flag = np.random.rand(self.in_dim) >= self.dropout_ratio
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
    sgd = Sgd()
    
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
    class Autoencoder:
        #- Constructor
        def __init__(self, visible_dim, hidden_dim, W, function):
            self.visible_dim = visible_dim
            self.hidden_dim  = hidden_dim
            self.function    = function
            self.W           = W
            self.a           = theano.shared(np.zeros(visible_dim).astype('float32'), name='a')
            self.b           = theano.shared(np.zeros(hidden_dim).astype('float32'), name='b')
            self.params      = [self.W, self.a, self.b]
    
        #- Encoder
        def encode(self, x):
            u = T.dot(x, self.W)+self.b
            y = self.function(u)
            return y
    
        #- Decoder
        def decode(self, x):
            u = T.dot(x, self.W.T)+self.a
            y = self.function(u)
            return y
    
        #- Forward Propagation
        def f_prop(self, x):
            y = self.encode(x)
            reconst_x = self.decode(y)
            return reconst_x
    
        #- Reconstruction Error
        def reconst_error(self, x, noise):
            tilde_x = x+noise
            reconst_x = self.f_prop(tilde_x)
            error = T.mean(T.sum(T.nnet.binary_crossentropy(reconst_x, x), axis=1))
            return error, reconst_x
    class Layer:
        #- Constructor
        def __init__(self, in_dim, out_dim, function):
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.function = function
            self.W = theano.shared(rng.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype('float32'), name='W')
            self.b = theano.shared(np.zeros(out_dim).astype('float32'), name='b')
            self.params = [self.W, self.b]
    
            self.set_pretraining()
    
        #- Forward Propagation
        def f_prop(self, x):
            self.u = T.dot(x, self.W) + self.b
            self.z = self.function(self.u)
            return self.z
    
        #- Set Pretraining
        def set_pretraining(self):
            ae = Autoencoder(self.in_dim, self.out_dim, self.W, self.function)
    
            x = T.fmatrix(name='x')
            noise = T.fmatrix(name='noise')
    
            cost, reconst_x = ae.reconst_error(x, noise)
            params = ae.params
            g_params = T.grad(cost=cost, wrt=params)
            updates = sgd(params, g_params)
    
            self.pretraining = theano.function(inputs=[x, noise], outputs=[cost, reconst_x], updates=updates, allow_input_downcast=True, name='pretraining')
    
            hidden = ae.encode(x)
            self.encode_function = theano.function(inputs=[x], outputs=hidden, allow_input_downcast=True, name='encode_function')


    layers = [
        Layer(784, 512, T.nnet.sigmoid),
        Dropout(512, 0.7),
        Layer(512, 256, T.nnet.sigmoid),
        Dropout(256, 0.8),
        Layer(256, 256, T.nnet.sigmoid),
        Dropout(256, 0.6),
        Layer(256,  10, T.nnet.softmax)
    ]
    X = train_X
    for l, layer in enumerate(layers[:-1]):
        corruption_level = np.float32(0.3)
        batch_size = 100
        n_batches = X.shape[0] // batch_size
    
        for epoch in xrange(10):
            X = shuffle(X)
            err_all = []
            for i in xrange(0, n_batches):
                start = i*batch_size
                end = start + batch_size
    
                noise = rng.binomial(size=X[start:end].shape, n=1, p=1-corruption_level)
                err, reconst_x = layer.pretraining(X[start:end], noise)
                err_all.append(err)
        X = layer.encode_function(X)
    x = T.fmatrix(name='x')
    t = T.imatrix(name='t')
    
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
    test  = theano.function([x], T.argmax(y, axis=1), name='test')
    batch_size = 100
    n_batches = train_X.shape[0]//batch_size
    
    train_y = np.eye(10)[train_y].astype('int32')
    for epoch in xrange(11):
        train_X, train_y = shuffle(train_X, train_y)
        for i in xrange(n_batches):
            start = i*batch_size
            end = start + batch_size
            Dropout.train = True
            train(train_X[start:end], train_y[start:end])
        Dropout.train = False
        cost, pred_y = valid(train_X, train_y)
        print "cost: {}, f1: {}".format(cost, f1_score(np.argmax(train_y, axis=1), pred_y, average="macro"))
    Dropout.train = False
    return test(test_X)

print f1_score(test_y, homework(train_X, test_X, train_y), average="macro")
