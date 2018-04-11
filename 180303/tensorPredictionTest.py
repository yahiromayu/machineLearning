import tensorflow as tf
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

N = 1000
X,y = datasets.make_moons(N,noise=0.3)

for i in range(len(X)):
	if(y[i] == 0):
		plt.plot(X[i,0],X[i,1],"ro")
	else:
		plt.plot(X[i,0],X[i,1],"bo")


#訓練データとテストデータを分割
Y = y.reshape(N, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8)

#隠れ層
num_hidden = 3
print("using num_hidden : ",num_hidden)
print()

#入力、出力サイズ
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])

#入力層-隠れ層
W = tf.Variable(tf.truncated_normal([2, num_hidden]))
b = tf.Variable(tf.zeros([num_hidden]))
h = tf.nn.sigmoid(tf.matmul(x,W) + b)

#隠れ層-出力層
V = tf.Variable(tf.truncated_normal([num_hidden,1]))
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h,V) + c)

#交差エントロピー誤差
cross_entropy = -tf.reduce_sum(t * tf.log(y) + (1-t) * tf.log(1-y))

#学習率、方法
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

#パーセプトロンの発火が正解かどうか
correct_prediction = tf.equal(tf.to_float(tf.greater(y,0.5)),t)

#予測精度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#学習
batch_size = 20
n_batches = N //batch_size

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(500):
	X_, Y_ = shuffle(X_train, Y_train)

	for i in range(n_batches):
		start = i*batch_size
		end = start + batch_size

		sess.run(train_step, feed_dict={
			x:X_[start:end],
			t:Y_[start:end]
		})
	if(epoch % 50 == 0):
		print("NOW LOADING : ",epoch)

accuracy_rate = accuracy.eval(session = sess, feed_dict={
	x:X_test,
	t:Y_test
})
print("accuracy:",accuracy_rate)

plt.show()