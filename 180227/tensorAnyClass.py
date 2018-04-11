import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf

# パラメータ初期値
M=2#入力次元
K=3#クラス
n=100#クラスごとのデータ数
N=n*K#全入力データ数

batch_size = 50
n_batches = N // batch_size

X1 = np.random.randn(n,M) + np.array([0,10])
X2 = np.random.randn(n,M) + np.array([5,5])
X3 = np.random.randn(n,M) + np.array([10,0])

Y1 = np.array([[1,0,0] for i in range(n)])
Y2 = np.array([[0,1,0] for i in range(n)])
Y3 = np.array([[0,0,1] for i in range(n)])

X = np.concatenate((X1,X2,X3), axis=0)
Y = np.concatenate((Y1,Y2,Y3), axis=0)

# plt.plot(X[:,0],X[:,1],"o")
# plt.show()

W = tf.Variable(tf.random_uniform([M,K],1.0,-1.0))
b = tf.Variable(tf.zeros([K]))

x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K])
y = tf.nn.softmax(tf.matmul(x,W)+b)

# 交差エントロピー誤差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y),reduction_indices=[1]))

# 確率的勾配降下法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 分類の正確性確認
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))


# 実行初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#エポックごとにデータシャッフルして学習
for epoch in range(20):
	X_, Y_ = shuffle(X, Y)

	for i in range(n_batches):
		start = i * batch_size
		end = start + batch_size

		sess.run(train_step, feed_dict={
			x: X_[start:end],
			t: Y_[start:end]
		})

# 分類できているか確認
X_, Y_ = shuffle(X, Y)
classified = correct_prediction.eval(session=sess, feed_dict={
	x: X_[0:10],
	t: Y_[0:10]
})
prob = y.eval(session=sess, feed_dict={
	x: X_[0:10]
})

def y1(x):
	return (-(lW[0,0]-lW[0,1]) * x - (lb[0]-lb[1])) / (lW[1,0]-lW[1,1])

def y2(x):
	return (-(lW[0,1]-lW[0,2]) * x - (lb[1]-lb[2])) / (lW[1,1]-lW[1,2])

lW = sess.run(W)
lb = sess.run(b)

plt.plot(X[:,0],X[:,1],"o")
tmpx = np.arange(-2,12,0.1)
tmpy1 = y1(tmpx)
tmpy2 = y2(tmpx)
plt.plot(tmpx,tmpy1)
plt.plot(tmpx,tmpy2)
plt.show()

print()
print(classified)
print()
print(prob)
print()
print(lW)
print(lb)
