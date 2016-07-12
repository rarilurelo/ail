from __future__ import division
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
import theano
import theano.tensor as T

rng = np.random.RandomState(42)
trng = RandomStreams(42)

def load_data(file_path):
    dataset = []
    vocab, tag = set(), set()
    for line in open(file_path):
        instance = [l.strip().split() for l in line.split('|||')]
        vocab.update(instance[0])
        tag.update(instance[1])
        dataset.append(instance)
    return dataset, vocab, tag

def encode_dataset(dataset, word2index, tag2index):
    X, y = [], []
    vocab = set(word2index.keys())
    for sentence, tags in dataset:
        X.append([word2index[word] if word in vocab else word2index['<unk>'] for word in sentence])
        y.append([tag2index[tag] for tag in tags])
    return X, y

train_data, train_vocab, train_tags = load_data('train.unk')
special_words = set(['<unk>'])

global word2index
global tag2index

word2index = dict(map(lambda x: (x[1], x[0]), enumerate(train_vocab | special_words)))
tag2index  = dict(map(lambda x: (x[1], x[0]), enumerate(train_tags)))

train_X, train_y = encode_dataset(train_data, word2index, tag2index)
train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2)

def homework(train_X, test_X, train_y):
    import time
    s=time.time()
    classes = np.arange(len(tag2index))
    train_y = [label_binarize(instance_y, classes).astype('int32') for instance_y in train_y]
    def sharedX(X, name='', dtype="float32"):
        return theano.shared(np.array(X, dtype=dtype), name=name)
    
    class Projection:
        def __init__(self, in_dim, out_dim, scale):
            self.V = sharedX(rng.randn(in_dim, out_dim) * scale, name='V')
            self.params = [self.V]
    
        def f_prop(self, x):
            x_emb = self.V[x]
            return x_emb
    class RNN_mul:
        def __init__(self, in_dim, hid_dim, scale):
            self.scale = scale
            self.hid_dim = hid_dim
    
            self.W_in  = sharedX(rng.randn(in_dim, hid_dim) * scale, name='W_in')
            self.W_rec = sharedX(np.identity(hid_dim) * scale, name='W_rec')
            self.b_rec = sharedX(rng.randn(hid_dim) * scale, name='b_rec')
            self.h_0   = sharedX(np.zeros(hid_dim), name='h_0')
    
            self.output_info = [self.h_0]
            self.params = [self.W_in, self.W_rec, self.b_rec]
            
            #self.bn = BatchNorm((hid_dim, ))
            #self.params += self.bn.params
    
        def f_prop(self, x):
            def step(x, h_tm1):
                h = T.tanh(T.dot(x, self.W_in)*T.dot(h_tm1, self.W_rec)+self.b_rec)
                return h
    
            h, _ = theano.scan(fn=step,
                               sequences=x,
                               outputs_info=self.output_info)
                               
            return h
    class BiLinear:
        def __init__(self, in_dim, out_dim, scale):
            self.W_out_f = sharedX(rng.randn(in_dim, out_dim)*scale, name='W_out_f')
            self.W_out_b = sharedX(rng.randn(in_dim, out_dim)*scale, name='W_out_b')
            self.b_out = sharedX(rng.randn(out_dim, )*scale, name='b_out')
            self.params = [self.W_out_f, self.W_out_b, self.b_out]
    
        def f_prop(self, x_f, x_b):
            z = T.dot(x_f, self.W_out_f)+T.dot(x_b, self.W_out_b)+self.b_out
            return z
    class Activation:
        def __init__(self, function=T.nnet.softmax):
            self.function = function
            self.params = []
    
        def f_prop(self, x):
            self.z = self.function(x)
            return self.z
    def sgd(cost, params, eps=np.float32(0.1), threshold=2):
        gparams = T.grad(cost, params)
        updates = OrderedDict()
        for param, gparam in zip(params, gparams):
            updates[param] = param - eps*gparam
        return updates
    def sgd_gp(cost, params, eps=np.float32(0.1), threshold=2):
        gparams = T.grad(cost, params)
        updates = OrderedDict()
        for param, gparam in zip(params, gparams):
            norm = gparam.norm(L=2)
            if T.lt(threshold, norm):
                gparam = threshold/norm*gparam
            updates[param] = param - eps*gparam
        return updates
    vocab_size = len(word2index)
    in_dim     = 300
    hid_dim    = 100
    out_dim    = len(tag2index)
    
    x = T.ivector('x')
    t = T.imatrix('t')
    
    layers = [
        Projection(vocab_size, in_dim, scale=0.3),
        RNN_mul(in_dim, hid_dim, scale=0.3),
        RNN_mul(in_dim, hid_dim, scale=0.3),
        BiLinear(hid_dim, out_dim, scale=0.3),
        Activation(T.nnet.softmax)
        ]
    params = []
    layer_out = x
    for i, layer in enumerate(layers):
        params += layer.params
        if i == 0:
            layer_out = layer.f_prop(x)
        elif i == 1:
            layer_out_f = layer.f_prop(layer_out)
        elif i == 2:
            layer_out_b = layer.f_prop(layer_out[::-1])
        elif i == 3:
            layer_out = layer.f_prop(layer_out_f, layer_out_b[::-1])
        else:
            layer_out = layer.f_prop(layer_out)
    
    y = layers[-1].z
    cost = T.mean(T.nnet.categorical_crossentropy(y, t))
    
    #optimizer = Rmsprop(params)
    optimizer = sgd
    #optimizer = sgd
    
    ## Define update graph
    updates = optimizer(cost, params) 
    
    ## Compile Function
    train = theano.function(inputs=[x,t], outputs=cost, updates=updates)
    #valid = theano.function(inputs=[x,t], outputs=[cost, T.argmax(y, axis=1)])
    test  = theano.function(inputs=[x], outputs=T.argmax(y, axis=1))
    epochs = 1
    
    for epoch in xrange(epochs):
        train_X, train_y = shuffle(train_X, train_y)  # Shuffle Samples !!
        for i, (instance_x, instance_y) in enumerate(zip(train_X, train_y)):
            cost = train(instance_x, instance_y)
            if i % 1000 == 0:
                print "EPOCH:: %i, Iteration %i, cost: %.3f" % (epoch + 1, i, cost)
        if time.time()-s > 58:
            break
        
        #true_y, pred_y, valid_cost = [], [], []
        #for instance_x, instance_y in zip(valid_X, valid_y):
        #    cost, pred = valid(instance_x, instance_y)
        #    true_y += list(np.argmax(instance_y, axis=1))
        #    pred_y += list(pred)
        #    valid_cost += cost
        #print 'EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, sum(valid_cost), f1_score(true_y, pred_y, average='macro'))
        #print 'Time: {0}min'.format((time.time()-s)/60)
    pred_y = []
    for instance_x in test_X:
        pred_y += list(test(instance_x))
    return pred_y
true_y = []
for instance_y in test_y:
    true_y += instance_y
    
f1_score(true_y, homework(train_X, test_X, train_y), average='macro')
