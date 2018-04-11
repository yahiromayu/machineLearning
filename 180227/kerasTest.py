import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# 初期化、入力2、出力1
model = Sequential([
	Dense(input_dim=2, units=1),
	Activation("sigmoid")
])

# # 追加する書き方もできる
# model = Sequential()
# model.add(Dense(input_dim=2,units=1))
# model.add(Activation("sigmoid"))

# 確率的勾配降下法
model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.1))

# 入力データ
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[1]])

# 学習実行
model.fit(X, Y, epochs=200, batch_size=1)


# 結果確認
# classesは分類結果を表すので合ってるかどうかはテストデータの出力データYと比較する
classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)

print()
print(Y == classes)
print()
print(prob)