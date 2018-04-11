import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

np.random.seed(0)
early_stopping = EarlyStopping(monitor="val_loss", patience=50,verbose=1)

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


'''
モデル設定
'''
n_in = len(X[0])	#784
n_hidden = 200
n_out = len(Y[0])	#10

model = Sequential()
model.add(Dense(n_hidden,input_dim=n_in))
model.add(Activation("sigmoid"))
model.add(Dense(n_out))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.01),metrics=["accuracy"])

'''
モデル学習
'''
epochs = 1000
batch_size = 100
hist = model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, validation_split=0.1,verbose=1)

'''
予測精度評価
'''
score = model.evaluate(X_test,Y_test, verbose=0)
print('test loss:', score[0])
print('test acc:', score[1])

loss = hist.history["loss"]
val_loss = hist.history["val_loss"]
acc = hist.history["acc"]
val_acc = hist.history["val_acc"]

nb_epochs = len(loss)
plt.plot(range(nb_epochs), loss)
plt.plot(range(nb_epochs), val_loss)
plt.show()

nb_epochs = len(acc)
plt.plot(range(nb_epochs), acc)
plt.plot(range(nb_epochs), val_acc)
plt.show()
