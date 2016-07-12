from __future__ import division
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import theano
import theano.tensor as T

mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data.astype('float32'), mnist.target.astype('int32'))

mnist_X = mnist_X / 255.0

train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2)

def homework(train_X, test_X, train_y):
    class Conv:
        #- Constructor
        def __init__(self, filter_shape, function, border_mode="valid", subsample=(1, 1)):
            
            self.function = function
            self.border_mode = border_mode
            self.subsample = subsample
            self.W = theano.shared(rng.uniform(low=-0.8, high=0.8, size=filter_shape).astype(np.float32), name='W')
            self.b = theano.shared(np.zeros(filter_shape[0]).astype(np.float32))
    
            self.params = [self.W, self.b]
            
        #- Forward Propagation
        def f_prop(self, x):
            conv_out = conv2d(x, self.W, border_mode=self.border_mode, subsample=self.subsample)
            self.z   = self.function(conv_out+self.b[np.newaxis, :, np.newaxis, np.newaxis])
            return self.z

        def init(self):
            self.W = theano.shared(rng.uniform(low=-0.8, high=0.8, size=filter_shape).astype(np.float32), name='W')
            self.b = theano.shared(np.zeros(filter_shape[0]).astype(np.float32))


    class Pooling:
        #- Constructor
        def __init__(self, pool_size=(2, 2), mode='max'):
            self.pool_size = pool_size
            self.mode = mode
            self.params = []
            
        #- Forward Propagation
        def f_prop(self, x):
            return pool.pool_2d(input=x, ds=self.pool_size, mode=self.mode)

        def init(self):
            pass

    class Flatten:
        #- Constructor
        def __init__(self, outdim=2):
            self.outdim = outdim
            self.params = []

        #- Forward Propagation
        def f_prop(self,x):
            return T.flatten(x, self.outdim)

        def init(self):
            pass

    class Layer:
        #- Constructor
        def __init__(self, in_dim, out_dim, function):
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.function = function
    
            self.W = theano.shared(rng.uniform(
                        low=-np.sqrt(6. / (in_dim + out_dim)),
                        high=np.sqrt(6. / (in_dim + out_dim)),
                        size=(in_dim,out_dim)
                    ).astype("float32"), name="W")       
            self.b =  theano.shared(np.zeros(out_dim).astype("float32"), name="b")
            self.params = [ self.W, self.b ]
            
        #- Forward Propagation
        def f_prop(self, x):
            self.z = self.function(T.dot(x, self.W) + self.b)
            return self.z

        def init(self):
            self.W = theano.shared(rng.uniform(
                        low=-np.sqrt(6. / (in_dim + out_dim)),
                        high=np.sqrt(6. / (in_dim + out_dim)),
                        size=(in_dim,out_dim)
                    ).astype("float32"), name="W")       
            self.b =  theano.shared(np.zeros(out_dim).astype("float32"), name="b")


    class BatchNormalization:
        def __init__(self, shape):
            self.gamma = theano.shared(rng.uniform(
                            low=-np.sqrt(6./(shape[0]+shape[1]+shape[2])), high=np.sqrt(6./(shape[0]+shape[1]+shape[2])),
                            size=shape).astype(np.float32), name="gamma"
                            )
            self.beta = theano.shared(rng.uniform(
                            low=-np.sqrt(6./(shape[0]+shape[1]+shape[2])), high=np.sqrt(6./(shape[0]+shape[1]+shape[2])),
                            size=shape).astype(np.float32), name="beta"
                            )
            self.params = [self.gamma, self.beta]

        def f_prop(self, x):
            mean = T.mean(x, axis=0)
            var = T.mean((x-mean)**2, axis=0)
            return T.nnet.bn.batch_normalization(x, self.gamma, self.beta, mean, var)

        def init(self):
            self.gamma = theano.shared(rng.uniform(
                            low=-np.sqrt(6./(shape[0]+shape[1]+shape[2])), high=np.sqrt(6./(shape[0]+shape[1]+shape[2])),
                            size=shape).astype(np.float32), name="gamma"
                            )
            self.beta = theano.shared(rng.uniform(
                            low=-np.sqrt(6./(shape[0]+shape[1]+shape[2])), high=np.sqrt(6./(shape[0]+shape[1]+shape[2])),
                            size=shape).astype(np.float32), name="beta"
                            )



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

    rng = np.random.RandomState(1234)
    train_y = np.eye(10)[train_y]
    train_X = train_X.reshape((-1, 1, 28, 28))
    test_X = test_X.reshape((-1, 1, 28, 28))

    activation = relu

    
    layers = [
        Conv((20, 1, 5, 5),activation),  # 28x28x 1 -> 24x24x20
        #BatchNormalization((24, 24, 20)),
        Pooling((2, 2)),                 # 24x24x20 -> 12x12x20
        Conv((50, 20, 5, 5),activation), # 12x12x20 ->  8x 8x50
        #BatchNormalization((8, 8, 50)),
        Pooling((2, 2)),                 #  8x 8x50 ->  4x 4x50
        Flatten(2),
        Layer(4*4*50, 500, activation),
        Layer(500, 10, T.nnet.softmax)
    ]

    x = T.ftensor4('x')
    t = T.imatrix('t')
    
    params = []
    layer_out = x
    for layer in layers:
        params += layer.params
        layer_out = layer.f_prop(layer_out)

    optimizer = Rmsprop(params, eps=np.float32(0.02))
    
    y = layers[-1].z
    
    cost = T.mean(T.nnet.categorical_crossentropy(y, t))
    
    g_params = T.grad(cost, params)
    updates = optimizer(params, g_params)
    
    train = theano.function(inputs=[x, t], outputs=cost, updates=updates, allow_input_downcast=True, name='train')
    valid = theano.function(inputs=[x, t], outputs=[cost, T.argmax(y, axis=1)], allow_input_downcast=True, name='valid')
    test  = theano.function(inputs=[x], outputs=y, name='test')


    batch_size = 100
    n_batches = train_X.shape[0]//batch_size
    for epoch in xrange(10):
        train_X, train_y = shuffle(train_X, train_y)
        for i in xrange(n_batches):
            start = i*batch_size
            end = start + batch_size
            train(train_X[start:end], train_y[start:end])
    pred_y = np.zeros((test_X.shape[0], 1))
    pred_y += test(test_X)

    return np.argmax(pred_y, axis=1)

homework(train_X, test_X, train_y)

    
