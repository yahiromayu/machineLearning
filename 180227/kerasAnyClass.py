import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.utils import shuffle

M=2#入力次元
K=3#クラス
n=100#クラスごとのデータ数
N=n*K#全入力データ数

# 入力データ
X1 = np.random.randn(n,M) + np.array([0,10])
X2 = np.random.randn(n,M) + np.array([5,5])
X3 = np.random.randn(n,M) + np.array([10,0])

Y1 = np.array([[1,0,0] for i in range(n)])
Y2 = np.array([[0,1,0] for i in range(n)])
Y3 = np.array([[0,0,1] for i in range(n)])

X = np.concatenate((X1,X2,X3), axis=0)
Y = np.concatenate((Y1,Y2,Y3), axis=0)

# 初期化、入力2、出力1
model = Sequential([
	Dense(input_dim=M, units=K),
	Activation("softmax")
])

# # 追加する書き方もできる
# model = Sequential()
# model.add(Dense(input_dim=2,units=1))
# model.add(Activation("sigmoid"))

# 確率的勾配降下法、1ofKの場合はbinary_crossentropyではなくcategorical_crossentropy
model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.1))

# 学習
minibatch_size = 50
model.fit(X,Y,epochs=20, batch_size=minibatch_size)


# 結果確認
# classesは分類結果を表すので合ってるかどうかはテストデータの出力データYと比較する
X_, Y_ = shuffle(X,Y)
classes = model.predict_classes(X_[0:10], batch_size=minibatch_size)
prob = model.predict_proba(X_[0:10], batch_size=1)

print()
print(np.argmax(model.predict(X_[0:10]), axis=1) == classes)
print()
print(prob)
