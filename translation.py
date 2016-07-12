from __future__ import division
from collections import OrderedDict, Counter
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
#from gensim.models.word2vec import Word2Vec
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
import theano
import theano.tensor as T

rng = np.random.RandomState(42)
trng = RandomStreams(42)

def build_vocab(file_path):
    f_vocab, e_vocab = set(), set()
    for line in open(file_path):
        f, e = [l.strip().split()[1:-1] for l in line.split('|||')]
        f_vocab.update(f)
        e_vocab.update(e)

    f_w2i = {w: np.int32(i+2) for i, w in enumerate(f_vocab)}
    e_w2i = {w: np.int32(i+2) for i, w in enumerate(e_vocab)}

    f_w2i['<s>'], f_w2i['</s>'] = np.int32(0), np.int32(1)
    e_w2i['<s>'], e_w2i['</s>'] = np.int32(0), np.int32(1)
    return set(f_w2i.keys()), set(e_w2i.keys()), f_w2i, e_w2i

def encode(sentence, vocab, w2i):
    encoded_sentence = []
    for w in sentence:
        if w in vocab:
            encoded_sentence.append(w2i[w])
        else:
            encoded_sentence.append(w2i['UNK'])
    return encoded_sentence

def decode(encoded_sentence, w2i):
    i2w = {i:w for w, i in w2i.items()}
    decoded_sentence = []
    for i in encoded_sentence:
        decoded_sentence.append(i2w[i])
    return decoded_sentence

def load_data(file_path, f_vocab, e_vocab, f_w2i, e_w2i):
    x, y = [], []
    for line in open(file_path):
        f, e = [l.strip().split() for l in line.split('|||')]
        f_enc = encode(f, f_vocab, f_w2i)
        e_enc = encode(e, e_vocab, e_w2i)
        x.append(f_enc)
        y.append(e_enc)
    return x, y

global f_vocab
global e_vocab

dataset_path = './train.zh-en'
f_vocab, e_vocab, f_w2i, e_w2i = build_vocab(dataset_path)
train_X, train_y = load_data(dataset_path, f_vocab, e_vocab, f_w2i, e_w2i)
train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2)
def homework(train_X, train_y):
    # WRITE ME!
    def sharedX(X, name=None, dtype="float32"):
        return theano.shared(np.array(X, dtype=dtype), name=name)
    
    class Projection:
        def __init__(self, in_dim, out_dim, scale):
            self.V = sharedX(rng.randn(in_dim, out_dim)*scale, name='V')
            self.params = [self.V]
    
        def f_prop(self, x):
            x_emb = self.V[x]
            return x_emb
    class LSTM:
        def __init__(self, in_dim, out_dim, scale, h_0=None, c_0=None):
            
            #- Input gate
            self.W_xi = sharedX(rng.randn(in_dim, out_dim)*scale, name='W_xi')
            self.W_hi = sharedX(rng.randn(out_dim, out_dim)*scale, name='W_hi')
            self.W_ci = sharedX(rng.randn(out_dim, out_dim)*scale, name='W_ci')
            self.b_i  = sharedX(rng.randn(out_dim)*scale, name='b_i')
            
            #- Forget gate
            self.W_xf = sharedX(rng.randn(in_dim, out_dim)*scale, name='W_xf')
            self.W_hf = sharedX(rng.randn(out_dim, out_dim)*scale, name='W_hf')
            self.W_cf = sharedX(rng.randn(out_dim, out_dim)*scale, name='W_cf')
            self.b_f  = sharedX(rng.randn(out_dim)*scale, name='b_f')
            
            #- Cell state
            self.W_xc = sharedX(rng.randn(in_dim, out_dim)*scale, name='W_xc')
            self.W_hc = sharedX(rng.randn(out_dim, out_dim)*scale, name='W_hc')
            self.b_c  = sharedX(rng.randn(out_dim)*scale, name='b_c')
            
            #- Output gate
            self.W_xo = sharedX(rng.randn(in_dim, out_dim)*scale, name='W_xo')
            self.W_ho = sharedX(rng.randn(out_dim, out_dim)*scale, name='W_ho')
            self.W_co = sharedX(rng.randn(out_dim, out_dim)*scale, name='W_co')
            self.b_o  = sharedX(rng.randn(out_dim)*scale, name='b_o')
    
            #- Initial state
            if h_0 is None:
                self.h_0 = sharedX(np.zeros(out_dim), name='h_0')
            else:
                self.h_0 = T.flatten(h_0)
            if c_0 is None:
                self.c_0 = sharedX(np.zeros(out_dim), name='c_0')
            else:
                self.c_0 = T.flatten(c_0)
    
            self.output_info = [self.h_0, self.c_0]
            self.params = [self.W_xf, self.W_hf, self.W_cf, self.b_f
                           , self.W_xi, self.W_hi, self.W_ci, self.b_i
                           , self.W_xc, self.W_hc, self.b_c
                           , self.W_xo, self.W_ho, self.W_co, self.b_o]
        
        def f_prop(self, x):
            def fn(x, h_tm1, c_tm1):
                # Input gate
                i_t = T.nnet.sigmoid(T.dot(x, self.W_xi)+T.dot(h_tm1, self.W_hi)+T.dot(c_tm1, self.W_ci)+self.b_i)
                
                # Forget gate
                f_t = T.nnet.sigmoid(T.dot(x, self.W_xf)+T.dot(h_tm1, self.W_hf)+T.dot(c_tm1, self.W_cf)+self.b_f)
                
                # Cell state
                c_t = f_t*c_tm1+i_t*T.tanh(T.dot(x, self.W_xc)+T.dot(h_tm1, self.W_hc)+self.b_c)
                
                # Output gate
                o_t = T.nnet.sigmoid(T.dot(x, self.W_xo)+T.dot(h_tm1, self.W_ho)+T.dot(c_t, self.W_co)+self.b_o)
                
                # Hidden state
                h_t = o_t*T.tanh(c_t)
                
                return h_t, c_t
            
            [h,c], _ = theano.scan(fn=fn,
                                   sequences=x,
                                   outputs_info=self.output_info
                                  )
            self.c = c
            
            return h
    class Linear:
        def __init__(self, in_dim, out_dim, scale):
            self.W_out = sharedX(rng.randn(in_dim, out_dim)*scale, name='W_out')
            self.b_out = sharedX(rng.randn(out_dim,)*scale, name='b_out')
            self.params = [self.W_out, self.b_out]
    
        def f_prop(self, x):
            z = T.dot(x, self.W_out) + self.b_out
            return z
    class Activation:
        def __init__(self, function):
            self.function = function
            self.params = []
    
        def f_prop(self, x):
            self.z = self.function(x)
            return self.z
    def sgd(cost, params, eps=np.float32(0.1)):
        g_params = T.grad(cost, params)
        updates = OrderedDict()
        for param, g_param in zip(params, g_params):
            updates[param] = param - eps*g_param
        return updates
    #Bidirectional
    x = T.ivector('x')
    t = T.ivector('t')

    # Target
    t_in = t[:-1]
    t_out = t[1:]
    vocab_size_f = len(f_vocab)
    vocab_size_e = len(e_vocab)
    in_dim  = 300
    hid_dim = 100
    out_dim = vocab_size_e
    scale=0.1


    def f_props(layers, x):
        layer_out = x
        for i, layer in enumerate(layers):
            if i == 0:
                layer_out = layer.f_prop(x)
            else:
                layer_out = layer.f_prop(layer_out)
        return layer_out

    encoder = [
        Projection(vocab_size_f, in_dim, scale=scale),
        LSTM(in_dim, hid_dim, scale=scale),
        LSTM(in_dim, hid_dim, scale=scale)
    ]

    layer_out = encoder[0].f_prop(x)
    layer_out_f = encoder[1].f_prop(layer_out)
    layer_out_b = encoder[2].f_prop(layer_out[::-1])

    BiLinear_hid = Linear(hid_dim*2, hid_dim, scale=scale)
    h_enc = BiLinear_hid.f_prop(T.concatenate([layer_out_f[-1], layer_out_b[-1]]))

    BiLinear_cel = Linear(hid_dim*2, hid_dim, scale=scale)
    c_enc = BiLinear_cel.f_prop(T.concatenate([encoder[1].c[-1], encoder[2].c[-1]]))

    decoder = [
        Projection(vocab_size_e, in_dim, scale=scale),
        LSTM(in_dim, hid_dim, h_0=h_enc, c_0=c_enc, scale=scale),
        Linear(hid_dim, out_dim, scale=scale),
        Activation(T.nnet.softmax)
    ]
    #Bidirection
    def join(layers):
        params = []
        for layer in layers:
            params += layer.params
        return params

    y = f_props(decoder, t_in)
    cost = T.mean(T.nnet.categorical_crossentropy(y, t_out))

    params = join(encoder + decoder + [BiLinear_hid] + [BiLinear_cel])
    updates = sgd(cost, params)
    
    train = theano.function(inputs=[x, t], outputs=cost, updates=updates)
    valid = theano.function(inputs=[x, t], outputs=cost)
    test  = theano.function(inputs=[x, t], outputs=[cost, T.argmax(y, axis=1)])
    epochs = 1
    for epoch in xrange(epochs):
        train_X, train_y = shuffle(train_X, train_y)  # Shuffle Samples !!
        for i, (instance_x, instance_y) in enumerate(zip(train_X, train_y)):
            train_cost = train(instance_x, instance_y)
            if i%100 == 0:
                pass
                print "EPOCH:: %i, Iteration %i, Training Cost: %.3f" % (epoch + 1, i, train_cost)
            if (i+1)%5000 == 0:
                break
    return test
test_fn = homework(train_X, train_y)
cost_test = 0
scale = 0.1
for instance_x, instance_y in zip(test_X, test_y):
    test_cost, _ = test_fn(instance_x, instance_y)
    cost_test += test_cost
print "scale: {0}, test cost: {1}".format(scale, cost_test/len(test_X))
