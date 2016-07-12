#coding: utf-8
from __future__ import division
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams

import cPickle
import numpy as np
import theano
import theano.tensor as T

rng = np.random.RandomState(1234)

def unpickle(file):
    with open(file, 'rb') as f:
        data = cPickle.load(f)
    return data

trn = [unpickle('./cifar_10/data_batch_%d' %i) for i in range(1,6)]
cifar_X_1 = np.concatenate([d['data'] for d in trn]).astype('float32')
cifar_y_1 = np.concatenate([d['labels'] for d in trn]).astype('int32')

tst = unpickle('./cifar_10/test_batch')
cifar_X_2 = tst['data'].astype('float32')
cifar_y_2 = np.array(tst['labels'], dtype='int32')

cifar_X = np.r_[cifar_X_1, cifar_X_2]
cifar_y = np.r_[cifar_y_1, cifar_y_2]

cifar_X = cifar_X / 255.

train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y, test_size=0.2)

def homework(train_X, test_X, train_y):
    # reshape data to 4-D
    train_X = train_X.reshape(-1, 3, 32, 32)
    test_X = test_X.reshape(-1, 3, 32, 32)

    # Preprocessing flip and crop
    flip_train_X = train_X[:, :, :, ::-1]
    train_X=np.concatenate((train_X, flip_train_X), axis=0)
    train_y = np.concatenate((train_y, train_y), axis=0)
    padded = np.pad(train_X, ((0, 0),(0, 0), (4, 4), (4, 4)), mode='constant')
    crops = np.random.randint(8, size=(len(train_X), 2))
    cropped_train_X = [padded[i, :, c[0]:(c[0]+32), c[1]:(c[1]+32)] for i, c in enumerate(crops)]
    cropped_train_X = np.array(cropped_train_X)
    train_X = np.concatenate((train_X, cropped_train_X), axis=0)
    train_y = np.concatenate((train_y, train_y), axis=0)

    def gcn(x):
        mean = np.mean(x, axis=(1,2,3), keepdims=True)
        std = np.std(x, axis=(1,2,3), keepdims=True)
        return (x - mean)/std
    class ZCAWhitening:
        
        def __init__(self, epsilon=1e-4):
            self.epsilon = epsilon
            self.mean = None
            self.ZCA_matrix = None
        
        def fit(self, x):
            x = x.reshape(x.shape[0],-1)
            self.mean = np.mean(x,axis=0)
            x -= self.mean
            cov_matrix = np.dot(x.T, x)/x.shape[1]
            A, d, _ = np.linalg.svd(cov_matrix)
            self.ZCA_matrix = np.dot(np.dot(A, np.diag(1./np.sqrt(d+self.epsilon))), A.T)
    
        def transform(self, x):
            shape = x.shape
            x = x.reshape(x.shape[0], -1)
            x -= self.mean
            x = np.dot(x,self.ZCA_matrix.T)
            return x.reshape(shape)
    class BatchNorm:
        #- Constructor
        def __init__(self, shape, epsilon=np.float32(1e-5)):
            self.shape = shape
            self.epsilon = epsilon
            
            self.gamma = theano.shared(np.ones(self.shape, dtype="float32"), name="gamma")
            self.beta = theano.shared(np.zeros(self.shape, dtype="float32"), name="beta")
            self.params = [self.gamma, self.beta]
            
        #- Forward Propagation
        def f_prop(self, x):
            if x.ndim == 2:
                mean = T.mean(x, axis=0, keepdims=True)
                std = T.sqrt(T.var(x, axis=0, keepdims=True)+self.epsilon)
            elif x.ndim == 4:
                mean = T.mean(x, axis=(0,2,3), keepdims=True)
                std = T.sqrt(T.var(x, axis=(0,2,3), keepdims=True)+self.epsilon)
            
            normalized_x = (x-mean)/std
            self.z = self.gamma*normalized_x+self.beta
            return self.z
    class Resnet:
        def __init__(self, shape, function):
            self.function = function
            self.bn1 = BatchNorm(shape)
            self.conv1 = Conv((shape[0], shape[0], 3, 3), border_mode=(1, 1))
            self.bn2 = BatchNorm(shape)
            self.conv2 = Conv((shape[0], shape[0], 3, 3), border_mode=(1, 1))
            self.params = self.bn1.params+self.conv1.params+self.bn2.params+self.conv2.params
        def f_prop(self, x):
            y = self.bn1.f_prop(x)
            y = self.function(y)
            y = self.conv1.f_prop(y)
            y = self.bn2.f_prop(y)
            y = self.function(y)
            y = self.conv2.f_prop(y)
            y = y+x
            return y
    class Conv:
        #- Constructor
        def __init__(self, filter_shape, function=lambda x: x, border_mode="valid", subsample=(1, 1)):
            
            self.function = function
            self.border_mode = border_mode
            self.subsample = subsample
            
            fan_in = np.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
            
            # Xavier
            self.W = theano.shared(rng.uniform(
                        low=-np.sqrt(6. / (fan_in + fan_out)),
                        high=np.sqrt(6. / (fan_in + fan_out)),
                        size=filter_shape
                    ).astype("float32"), name="W")
            self.b = theano.shared(np.zeros((filter_shape[0],), dtype="float32"), name="b")
            self.params = [self.W,self.b]
            
        #- Forward Propagation
        def f_prop(self, x):
            conv_out = conv2d(x, self.W, border_mode=self.border_mode, subsample=self.subsample)
            self.z = self.function(conv_out + self.b[np.newaxis, :, np.newaxis, np.newaxis])
            return self.z
    class Pooling:
        #- Constructor
        def __init__(self, pool_size=(2,2), padding=(0,0), mode='max'):
            self.pool_size = pool_size
            self.mode = mode
            self.padding = padding
            self.params = []
            
        #- Forward Propagation
        def f_prop(self, x):
            return pool.pool_2d(input=x, ds=self.pool_size, padding=self.padding, mode=self.mode, ignore_border=True)
    class Flatten:
        #- Constructor
        def __init__(self, outdim=2):
            self.outdim = outdim
            self.params = []
    
        #- Forward Propagation
        def f_prop(self,x):
            return T.flatten(x, self.outdim)
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
    class Activation:
        #- Constructor
        def __init__(self, function):
            self.function = function
            self.params = []
        
        #- Forward Propagation
        def f_prop(self, x):
            self.z = self.function(x)
            return self.z
    #--- Stochastic Gradient Descent
    def sgd(params, g_params, eps=np.float32(0.1)):
        updates = OrderedDict()
        for param, g_param in zip(params, g_params):
            updates[param] = param - eps*g_param
        return updates
    activation = T.nnet.relu
    
    layers = [                               # (チャネル数)x(縦の次元数)x(横の次元数)
        Conv((32, 3, 3, 3)),                 #   3x32x32 ->  32x30x30
        #Resnet((32, 30, 30), activation),
        BatchNorm((32, 30, 30)),
        Activation(activation),
        Pooling((2, 2)),                     #  32x30x30 ->  32x15x15
        #Resnet((32, 15, 15), activation),
        Conv((64, 32, 3, 3)),                #  32x15x15 ->  64x13x13
        BatchNorm((64, 13, 13)),
        Activation(activation),
        Pooling((2, 2)),                     #  64x13x13 ->  64x 6x 6
        Conv((128, 64, 3, 3)),               #  64x 6x 6 -> 128x 4x 4
        BatchNorm((128, 4, 4)),
        Activation(activation),
        Pooling((2, 2)),
        Flatten(2),
        Layer(128*2*2, 256, activation),
        Layer(256, 10, T.nnet.softmax)
    ]
    x = T.ftensor4('x')
    t = T.imatrix('t')
    
    params = []
    layer_out = x
    for layer in layers:
        params += layer.params
        layer_out = layer.f_prop(layer_out)
    
    y = layers[-1].z
    
    cost = T.mean(T.nnet.categorical_crossentropy(y, t))
    
    g_params = T.grad(cost, params)
    updates = sgd(params, g_params)
    
    train = theano.function(inputs=[x, t], outputs=cost, updates=updates, allow_input_downcast=True, name='train')
    valid = theano.function(inputs=[x, t], outputs=[cost, T.argmax(y, axis=1)], allow_input_downcast=True, name='valid')
    test  = theano.function(inputs=[x], outputs=T.argmax(y, axis=1), name='test')
    # Preprocessing
    zca = ZCAWhitening()
    zca.fit(gcn(train_X))
    zca_train_X = zca.transform(gcn(train_X))
    zca_train_y = train_y[:]
    batch_size = 100
    n_batches = zca_train_X.shape[0]//batch_size
    for epoch in xrange(10):
        zca_train_X, zca_train_y = shuffle(zca_train_X, zca_train_y)
        for i in xrange(n_batches):
            start = i*batch_size
            end = start + batch_size
            cost = train(zca_train_X[start:end], zca_train_y[start:end])
        #print 'Training cost: %.3f' % cost
        #valid_cost, pred_y = valid(zca_valid_X, zca_valid_y)
        #print 'EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(zca_valid_y, axis=1).astype('int32'), pred_y, average='macro'))
    return test(test_X)
print f1_score(train_y, homework(train_X, test_X, train_y), average='macro')
