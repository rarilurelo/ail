from __future__ import division
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

import numpy as np
import matplotlib.pyplot as plt
def plot_number(x):
	fig = plt.figure(figsize=(9, 9))
	fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
	ax = fig.add_subplot(10, 10, 2, xticks=[], yticks=[])
	ax.imshow(x.reshape((28, 28)), cmap='gray')
	plt.show()

mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data, mnist.target.astype('int32'))

mnist_X = mnist_X/255.0

train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2)

k = 2
from collections import Counter
def data_gen(n):
	return train_X[train_y == n]
train_X_num = [data_gen(i) for i in range(10)]
print ( train_X_num[0]-test_X[0] ).shape
inv_cov = [np.linalg.inv(np.cov(train_X_num[i], rowvar=0)+np.eye(784)*0.00001) for i in range(10)]
pred_y = [-1 for i in test_X]
d = [-1 for i in range(10)]
print "start"
import time
s = time.time()
ivec = train_X_num[0]
print ivec.shape
t = time.time()
print t
tes = np.ones((14000, len(ivec), 784))
print time.time()-t
tes[0, :, 0] = test_X
print ivec.shape
iinv_cov = inv_cov[0]
print time.time()-s
#for i in range(10):
#	ivec = train_X_num[i]
#	ivec = ivec-test_X[:, np.newaxis, :]
#	iinv_cov = inv_cov[i]
#	print "one loop"
#	s = time.time()
#	d[i] = np.add.reduce(np.dot(ivec, iinv_cov)*ivec, axis=2).sort(1)[:, :k+1]
#	print s-time.time()
ind = [0 for i in range(10)]
argmin_list = [None for i in range(k)]



#for q, test_x in enumerate(test_X):
#	k_list = [0 for i in range(10)]
#	for i in range(10):
#		d_mat = train_X_num[i]-test_x
#		d_vec = np.sort(np.diag(d_mat.dot(inv_cov[i].dot(d_mat.T))))
#		k_list[i] = d_vec[0:k+1]
#	indicate = [0 for i in range(10)]
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
