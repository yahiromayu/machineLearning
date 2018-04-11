import numpy as np
import tensorflow as tf

w = tf.Variable(tf.random_uniform([2,1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, shape=[None,2])
t = tf.placeholder(tf.float32, shape=[None,1])
y = tf.nn.sigmoid(tf.matmul(x,w)+b)

# 交差エントロピー誤差
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1-t) * tf.log(1-y))

# 学習率0.1で勾配降下法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# yが0.5以上で発火
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# 入力パラメータ
X = ([[0,0],[0,1],[1,0],[1,1]])
Y = ([[0],[1],[1],[1]])

# 初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 学習
for epoch in range(200):
	sess.run(train_step, feed_dict={
		x: X,
		t: Y
	})

# 結果確認(分類)
classified = correct_prediction.eval(session=sess, feed_dict={
	x: X,
	t: Y
})
print(classified)
print()

# 結果確認(確率)
prob = y.eval(session = sess, feed_dict={
	x: X,
	t: Y
})
print(prob)
print()

# 変数確認
print("w:", sess.run(w))
print("b:", sess.run(b))
