import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DNN(object):
	def __init__(self,n_in, n_hiddens, n_out):
		#初期化処理
		self.n_in = n_in
		self.n_hiddens = n_hiddens
		self.n_out = n_out
		self.weights = []
		self.biases = []

		self._x = None
		self._t = None
		self._keep_prob = None
		self._sess = None
		self._history = {
			"accuracy":[],
			"loss":[]
		}

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.zeros(shape)
		return tf.Variable(initial)

	def inference(self, x, keep_prob):
		#モデル定義
		#入力層-隠れ層、隠れ層-隠れ層
		for i,n_hidden in enumerate(self.n_hiddens):
			if i==0:
				input = x
				input_dim = self.n_in
			else:
				input = output
				input_dim = self.n_hiddens[i-1]

			self.weights.append(self.weight_variable([input_dim,n_hidden]))
			self.biases.append(self.bias_variable([n_hidden]))

			h = tf.nn.relu(tf.matmul(input, self.weights[-1])+self.biases[-1])
			output = tf.nn.dropout(h, keep_prob)

		#隠れ層-隠れ層
		self.weights.append(self.weight_variable([self.n_hiddens[-1], self.n_out]))
		self.biases.append(self.bias_variable([self.n_out]))

		y = tf.nn.softmax(tf.matmul(output, self.weights[-1]) + self.biases[-1])
		return y

	def loss(self, y, t):
		#cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(y),reduction_indices=[1]))
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),reduction_indices=[1]))
		return cross_entropy

	def training(self, loss):
		optimizer = tf.train.GradientDescentOptimizer(0.005)
		train_step = optimizer.minimize(loss)
		return train_step

	def accuracy(self, y, t):
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy

	def fit(self, X_train, Y_train, X_validation, Y_validation, epochs = 100, batch_size = 100, p_keep = 0.5, verbose = 1):
		#学習処理
		x = tf.placeholder(tf.float32, shape=[None, self.n_in])
		t = tf.placeholder(tf.float32, shape=[None, self.n_out])
		keep_prob = tf.placeholder(tf.float32)

		#evaluate()用に保持
		self._x = x
		self._t = t
		self._keep_prob = keep_prob

		y = self.inference(x, keep_prob)
		loss = self.loss(y, t)
		train_step = self.training(loss)
		accuracy = self.accuracy(y, t)

		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)

		#evaluate用に保持
		self._accuracy = accuracy
		self._sess = sess

		N_train = len(X_train)
		n_batches = N_train // batch_size

		for epoch in range(epochs):
			X_, Y_ = shuffle(X_train, Y_train)
			for i in range(n_batches):
				start = i * batch_size
				end = start + batch_size

				sess.run(train_step, feed_dict={
					x: X_[start:end],
					t: Y_[start:end],
					keep_prob: p_keep
				})
			loss_ = loss.eval(session=sess, feed_dict={
				x: X_validation,
				t: Y_validation,
				keep_prob: 1.0
			})
			accuracy_ = accuracy.eval(session=sess, feed_dict={
				x: X_validation,
				t: Y_validation,
				keep_prob: 1.0
			})
			#値の記録
			self._history["loss"].append(loss_)
			self._history["accuracy"].append(accuracy_)

			if verbose:
				print("epoch:",epoch,"  loss:",loss_,"  accuracy:",accuracy_)

		return self._history

	def evaluate(self, X_test, Y_test):
		#評価処理
		return self._accuracy.eval(session=self._sess, feed_dict={
			self._x: X_test,
			self._t: Y_test,
			self._keep_prob: 1.0
		})

if __name__ == "__main__":
	np.random.seed(1)

	#mnistをローカルにダウンロード
	mnist = datasets.fetch_mldata("MNIST original", data_home=".")

	n = len(mnist.data)
	N = 24000
	N_training = 20000
	N_validation = 4000

	#n個の数字からN個取り出す,X,yの値を出す
	indices = np.random.permutation(range(n))[:N]
	X = mnist.data[indices]
	y = mnist.target[indices]
	#1ofK表現に変換
	Y = np.eye(10)[y.astype(int)]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=N_training)
	X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=N_validation)

	#inference(),loss(),training()を1つにまとめる
	model = DNN(n_in=784,n_hiddens=[200,200,200],n_out=10)
	model.fit(X_train,Y_train,X_validation,Y_validation,epochs=50,batch_size=200,p_keep=0.5)
	accuracy = model.evaluate(X_test,Y_test)
	print("accuracy : ",accuracy)
	'''	
	#グラフ描画
	plt.rc("font",family="sefif")#フォント
	flg = plt.figure()#グラフ準備
	#データ書き込み
	plt.plot(range(50),model._history["accuracy"],label="acc",color="black")
	#軸の名前
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	#グラフ表示
	plt.show()
	#保存
	#plt.savefig("mnist_tensorflow.eps")
	'''

	fig = plt.figure()
	ax_acc = fig.add_subplot(111)#予測精度用の軸
	ax_acc.plot(range(50),model._history["accuracy"],label="acc",color="black")
	ax_loss = ax_acc.twinx()#損失用の軸設定
	ax_loss.plot(range(50),model._history["loss"],label="loss",color="gray")
	plt.xlabel("epochs")
	plt.show()
	#plt.savefig("mnist_tensorflow.eps")