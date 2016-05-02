from __future__ import division
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data, mnist.target.astype('int32'))

mnist_X = mnist_X/255.0

train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2)

def homework(train_X, train_y, test_X):
    from collections import Counter
    def k_nn(k, train_X, train_y, test_X):
        normalized_train_X = train_X/np.linalg.norm(train_X, ord=2, axis=1)[:, np.newaxis]
        normalized_test_X = test_X/np.linalg.norm(test_X, ord=2, axis=1)[:, np.newaxis]
        pred = []
        for test in normalized_test_X:
            dot_product = np.dot(normalized_train_X, test)
            k_max = []
            for _ in range(k):
                k_max.append(train_y[np.argmax(dot_product)])
                dot_product[np.argmax(dot_product)] = 0
            pred.append(Counter(k_max).most_common(1)[0][0])
        return pred
    
    
    X_train, valid_X, y_train, valid_y = train_test_split(train_X, train_y, test_size=0.2)
    k_f1_list = []
    for k in range(1, 2):
        pred = k_nn(k, X_train, y_train, valid_X)
        k_f1_list.append(f1_score(valid_y, pred, average='micro'))
    k = np.argmax(k_f1_list)+1
    pred_y = k_nn(k, train_X, train_y, test_X)
    return pred_y
print f1_score(test_y, homework(train_X, train_y, test_X), average='micro')

#dot_product = np.dot(normalized_test_X, normalized_train_X.T)
#del normalized_train_X
#del normalized_test_X
#gc.collect()
#k_maxes = [[] for i in test_X]
#for i in range(k):
#    max_ind = np.argmax(dot_product, axis=1)
#    for j, ind in enumerate(max_ind):
#        k_maxes[j].append(train_y[ind])
#        dot_product[j, ind] = 0
#pred = []
#del dot_product
#gc.collect()
#for k_max in k_maxes:
#    pred.append(Counter(k_max).most_common(1)[0][0])
#return pred


#k = 2
#def data_gen(n):
#	return train_X[train_y == n]
#train_X_num = [data_gen(i) for i in range(10)]
#inv_cov = [np.linalg.inv(np.cov(train_X_num[i], rowvar=0)+np.eye(784)*0.00001) for i in range(10)] // Making Inverse covariance matrices
#for i in range(10):
#	ivec = train_X_num[i] // ivec size is (number of 'i' data, 784)
#	ivec = ivec - test_X[:, np.newaxis, :] // This code is too much slowly, and using huge memory
#	iinv_cov = inv_cov[i]
#	d[i] = np.add.reduce(np.dot(ivec, iinv_cov)*ivec, axis=2).sort(1)[:, :k+1] // Calculate x.T inverse(sigma) x, and extract k-minimal distance
#ivec = train_X_num[0]
#tes = np.ones((14000, len(ivec), 784))
#tes[0, :, 0] = test_X
#iinv_cov = inv_cov[0]
#from collections import Counter
#def plot_number(x):
#	fig = plt.figure(figsize=(9, 9))
#	fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
#	ax = fig.add_subplot(10, 10, 2, xticks=[], yticks=[])
#	ax.imshow(x.reshape((28, 28)), cmap='gray')
#	plt.show()
#pred_y = [-1 for i in test_X]
#d = [-1 for i in range(10)]
##for i in range(10):
##	ivec = train_X_num[i]
##	ivec = ivec-test_X[:, np.newaxis, :]
##	iinv_cov = inv_cov[i]
##	print "one loop"
##	s = time.time()
##	d[i] = np.add.reduce(np.dot(ivec, iinv_cov)*ivec, axis=2).sort(1)[:, :k+1]
##	print s-time.time()
#ind = [0 for i in range(10)]
#argmin_list = [None for i in range(k)]
#
#
#
##for q, test_x in enumerate(test_X):
##	k_list = [0 for i in range(10)]
##	for i in range(10):
##		d_mat = train_X_num[i]-test_x
##		d_vec = np.sort(np.diag(d_mat.dot(inv_cov[i].dot(d_mat.T))))
##		k_list[i] = d_vec[0:k+1]
##	indicate = [0 for i in range(10)]
#	argmin_list = [None for i in range(k)]
#	for i in range(k):
#		min_list = [None for p in range(10)]
#		for j in range(10):
#			min_list[j] = k_list[j][indicate[j]]
#		argmin_list[i] = np.argmin(min_list)
#		indicate[argmin_list[i]] += 1
#	pred_y[q] = Counter(argmin_list).most_common(1)[0][0]
#print f1_score(test_y, pred_y)
#

#for i in range(10):
#	print train_X_num[i]-test_X[:, np.newaxis]

#plot_number(train_X_num[0][0])
#
#def homework(train_X, train_y, test_X):
#	return pred_y
