import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class EarlyStopping():
	def __init__(self, patience=0, verbose=0):
		self._step = 0
		self._loss = float("inf")
		self.patience = patience
		self.verbose = verbose

	def validate(self, loss):
		if self._loss < loss:
			self._step += 1
			if self._step > self.patience:
				if self.verbose:
					print("early_stopping")
				return True

		else:
			self._step = 0
			self._loss = loss

		return False


def inference(x, n_batch, maxlen=None, n_hidden=None, n_out=None):
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		return tf.Variable(initial)
	def bias_variable(shape):
		initial = tf.zeros(shape)
		return tf.Variable(initial)

	#RNN隠れ層初期化
	#cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
	#cell = tf.contrib.rnn.SimpleRNNCell(n_hidden, forget_bias=1.0)
	cell = tf.contrib.rnn.GRUCell(n_hidden)
	initial_state = cell.zero_state(n_batch, tf.float32)
	state = initial_state
	outputs = [] #隠れ層の出力を保存
	with tf.variable_scope("RNN"):
		for t in range(maxlen):
			if t > 0:
				tf.get_variable_scope().reuse_variables()
			(cell_output, state) = cell(x[:,t,:],state)
			outputs.append(cell_output)
	output = outputs[-1]

	V = weight_variable([n_hidden, n_out])
	c = bias_variable([n_out])
	y = tf.matmul(output, V) + c #線形活性

	return y

def loss(y, t):
	mse = tf.reduce_mean(tf.square(y-t)) #2乗平均誤差関数
	return mse

def training(loss):
	optimizer = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.999)
	train_step = optimizer.minimize(loss)
	return train_step

def mask(T=200):
	mask = np.zeros(T)
	indices = np.random.permutation(np.arange(T))[:2]
	mask[indices] = 1
	return mask

def toy_problem(N=10, T=100):
	signals = np.random.uniform(low=0.0, high=1.0, size=(N,T))
	masks = np.zeros((N, T))
	for i in range(N):
		masks[i] = mask(T)

	data = np.zeros((N, T, 2))
	data[:,:,0] = signals[:]
	data[:,:,1] = masks[:]
	target = (signals * masks).sum(axis=1).reshape(N, 1)

	return (data, target)

if __name__ == "__main__":
	np.random.seed(1)
	early_stopping = EarlyStopping(patience=5,verbose=1)
	history = {
		"val_loss":[]
	}

	N = 10000
	T = 200
	maxlen = T

	X,Y = toy_problem(N=N, T=T)

	N_train = int(N* 0.9)
	N_validation = N - N_train

	X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)

	n_in = len(X[0][0]) #1
	n_hidden = 20
	n_out = len(Y[0]) #1

	x = tf.placeholder(tf.float32, shape=[None, maxlen, n_in])
	t = tf.placeholder(tf.float32, shape=[None, n_out])
	n_batch = tf.placeholder(tf.int32, shape=[])

	y = inference(x, n_batch, maxlen=maxlen, n_hidden=n_hidden, n_out=n_out)
	loss = loss(y, t)
	train_step = training(loss)


	epochs = 200
	batch_size = 10

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	n_batches = N_train // batch_size

	for epoch in range(epochs):
		X_,Y_ = shuffle(X_train,Y_train)

		for i in range(n_batches):
			start = i * batch_size
			end = start + batch_size

			sess.run(train_step, feed_dict={
				x: X_[start:end],
				t: Y_[start:end],
				n_batch: batch_size
			})

		#検証データで表か
		val_loss = loss.eval(session=sess, feed_dict={
			x:X_validation,
			t:Y_validation,
			n_batch:N_validation
		})

		history["val_loss"].append(val_loss)
		print("epoch : ",epoch,"   validation loss : ",val_loss)

		#Early Stopping
		if early_stopping.validate(val_loss):
			break

	plt.plot(history["val_loss"])
	plt.show()