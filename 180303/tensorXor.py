import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#正解データ
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

#入力層2、出力層1
x = tf.placeholder(tf.float32,shape=[None, 2])
t = tf.placeholder(tf.float32,shape=[None, 1])

#中間層パラメータ
#truncated_normalはゼロ行列ではない行列を生成
W = tf.Variable(tf.truncated_normal([2,2]))
b = tf.Variable(tf.zeros([2]))
h = tf.sigmoid(tf.matmul(x, W) + b)

#出力層パラメータ
V = tf.Variable(tf.truncated_normal([2,1]))
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h, V) + c)

#交差エントロピー誤差
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1-t) * tf.log(1-y))

#確率的勾配降下法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

#学習スタート
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(4000):
	sess.run(train_step, feed_dict = {
		x:X,
		t:Y
	})
	#途中経過を表示
	if epoch % 1000 == 0:
		print("epoch:", epoch)
		classified = correct_prediction.eval(session = sess, feed_dict = {
			x:X,
			t:Y
		})
		prob = y.eval(session=sess,feed_dict={
			x:X
		})

		print("classified:")
		print(classified)
		print()
		print("output probabolity:")
		print(prob)
