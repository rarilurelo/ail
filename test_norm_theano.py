import theano
import theano.tensor as T
import numpy as np

x = T.matrix('x')
y = x.norm(L=2)
fn = theano.function(inputs=[x], outputs=y, allow_input_downcast=True)

#print fn(np.array([1,2,3,3,9]))
print fn(np.array([[1,2,3,3],[1,1,1,1]]))
