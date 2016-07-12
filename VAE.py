from __future__ import division
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import theano
import theano.tensor as T

#theano.config.floatX = "float32"
rng = np.random.RandomState(1234)
theano_rng = RandomStreams(rng.randint(1234))

mnist = fetch_mldata('MNIST original')
mnist_X = shuffle(mnist.data.astype('float32'), random_state=1234)

mnist_X = mnist_X/255.0

train_X, test_X = train_test_split(mnist_X, test_size=0.2)


def homework(train_X):

    class Layer:
        def __init__(self, in_dim, out_dim, function=lambda x : x):
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.function = function

            self.W = theano.shared(rng.uniform(
                        low=-np.sqrt(6./(in_dim+out_dim)),
                        high=np.sqrt(6./(in_dim+out_dim)),
                        size=(in_dim, out_dim)
                        ).astype('float32'), name='W'
                        )

            self.b = theano.shared(np.zeros(out_dim).astype('float32'), name='b')

            self.params = [self.W, self.b]

        def f_prop(self, x):
            self.u = T.dot(x, self.W)+self.b
            self.z = self.function(self.u)
            return self.z
    class VAE:
        def __init__(self, q, p, random=1234):
            self.q = q
            self.p = p
            self.srng = RandomStreams(seed=random)
        def q_f_prop(self, x):
            params = []
            layer_out = x
            for i, layer in enumerate(self.q[:-2]):
                params += layer.params
                layer_out = layer.f_prop(layer_out)

            params += self.q[-2].params
            mean = self.q[-2].f_prop(layer_out)

            params += self.q[-1].params
            var = self.q[-1].f_prop(layer_out)

            return mean, var, params

        def p_f_prop(self, x):
            params = []
            layer_out = x
            for i, layer in enumerate(self.p):
                params += layer.params
                layer_out = layer.f_prop(layer_out)
            mean = layer_out

            return mean, params

        def lower_bound(self, x):
            mean, var, q_params = self.q_f_prop(x)
            KL = -1./2*T.mean(T.sum(1+T.log(var)-mean**2-var, axis=1))

            epsilon = self.srng.normal(mean.shape)
            z = mean+epsilon*T.sqrt(var)

            _x, p_params = self.p_f_prop(z)
            log_likelihood = T.mean(T.sum(x*T.log(_x)+(1-x)*T.log(1-_x), axis=1))
            params = q_params+p_params

            lower_bound = [-KL, log_likelihood]

            return lower_bound, params

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

    def Adam(params, g_params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        updates = []
        i = theano.shared(0.)
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, g_params):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates

    class Sgd(object):
        def __init__(self, lr=0.01):
            self.lr = lr
        def __call__(self, params, g_params):
            updates = OrderedDict()
            for param, g_param in zip(params, g_params):
                updates[param] = param - self.lr*g_param
            return updates

    z_dim = 10

    q = [
            Layer(784, 200, T.nnet.relu),
            Layer(200, 200, T.nnet.relu),
            Layer(200, z_dim),                 # mean
            Layer(200, z_dim, T.nnet.softplus) # variance
        ]

    p = [
            Layer(z_dim, 200, T.nnet.relu),
            Layer(200, 200, T.nnet.relu),
            Layer(200, 784, T.nnet.sigmoid)
        ]

    model = VAE(q, p)

    x = T.fmatrix('x')
    lower_bound, params = model.lower_bound(x)

    g_params = T.grad(-T.sum(lower_bound), params)
    #optimizer = Rmsprop(params)
    optimizer = Sgd()
    updates = optimizer(params, g_params)

    train = theano.function(inputs=[x], outputs=lower_bound, updates=updates, allow_input_downcast=True, name='train')
    test = theano.function(inputs=[x], outputs=T.sum(lower_bound), allow_input_downcast=True, name='test')

    batch_size = 100
    n_batches = train_X.shape[0]//batch_size
    
    for epoch in xrange(30):
        rng.shuffle(train_X)
        lowerbound_all = []
        for i in xrange(n_batches):
            start = i*batch_size
            end = start+batch_size
            lowerbound = train(train_X[start:end])
            lowerbound_all.append(lowerbound)
        lowerbound_all = np.mean(lowerbound_all, axis=0)
        print 'Train Lower Bound:%lf (%lf, %lf)' % (np.sum(lowerbound_all), lowerbound_all[0], lowerbound_all[1])
    return test

test = homework(train_X)
test_lowerbound = test(test_X)
print 'Test Lower Bound:%lf' % test_lowerbound
