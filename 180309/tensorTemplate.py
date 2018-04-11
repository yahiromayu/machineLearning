import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def inference(x, keep_prob, n_in, n_hiddens, n_out):
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.zeros(shape)
		return tf.Variable(initial)

	#入力層 - 隠れ層、隠れ層 - 隠れ層
	for i, n_hidden in enumerate(n_hiddens):
		if i==0:
			input = x
			input_dim = n_in
		else:
			input = output
			input_dim = n_hiddens[i-1]

		W = weight_variable([input_dim, n_hidden])
		b = bias_variable([n_hidden])

		h = tf.nn.relu(tf.matmul(input, W) + b)
		output = tf.nn.dropout(h, keep_prob)

	#隠れ層 - 出力層
	W_out = weight_variable([n_hiddens[-1], n_out])
	b_out = bias_variable([n_out])
	y = tf.nn.softmax(tf.matmul(output, W_out) + b_out)
	return y


def loss(y, t):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
	return cross_entropy

def training(loss):
	optimizer = tf.train.GradientDescentOptimizer(0.005)
	train_step = optimizer.minimize(loss)
	return train_step

if __name__ == "__main__":
	np.random.seed(1)

	#mnistをローカルにダウンロード
	mnist = datasets.fetch_mldata("MNIST original", data_home=".")

	n = len(mnist.data)
	N = 10000

	#n個の数字からN個取り出す,X,yの値を出す
	indices = np.random.permutation(range(n))[:N]
	X = mnist.data[indices]
	y = mnist.target[indices]
	#1ofK表現に変換
	Y = np.eye(10)[y.astype(int)]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

	n_in = 784
	n_hiddens = [200, 200, 200]
	n_out = 10

	x = tf.placeholder(tf.float32, shape=[None, n_in])
	t = tf.placeholder(tf.float32, shape=[None, n_out])
	keep_prob = tf.placeholder(tf.float32)

	y = inference(x,keep_prob, n_in = n_in, n_hiddens = n_hiddens, n_out = n_out)
	loss = loss(y,t)
	train_step = training(loss)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	epochs = 30
	batch_size = 100
	p_keep = 0.5
	N_train = len(X_train)
	n_batches = N_train // batch_size

	for epoch in range(epochs):
		X_,Y_ = shuffle(X_train,Y_train)
		for i in range(n_batches):
			start = i*batch_size
			end = start + batch_size

			sess.run(train_step, feed_dict={
				x:X_[start:end],
				t:Y_[start:end],
				keep_prob:p_keep
			})
		loss_ = loss.eval(session=sess, feed_dict={
			x: X_train,
			t: Y_train,
			keep_prob: 1.0
		})
		accuracy_ = accuracy.eval(session=sess, feed_dict={
			x: X_train,
			t: Y_train,
			keep_prob: 1.0
		})

		print("epoch:",epoch,"  loss:",loss_,"  accuracy:",accuracy_)

	#評価
	testAccuracy = accuracy.eval(session = sess, feed_dict ={
		x:X_test,
		t:Y_test,
		keep_prob:1.0
	})
	print("Test Data : ",testAccuracy)