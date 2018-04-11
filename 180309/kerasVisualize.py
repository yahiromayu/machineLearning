import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def weight_variable(shape):
	return K.truncated_normal(shape, stddev=0.01)

np.random.seed(1)

#mnistをローカルにダウンロード
mnist = datasets.fetch_mldata("MNIST original", data_home=".")

n = len(mnist.data)
N_training = 25000
N_validation = 5000

#n個の数字からN個取り出す,X,yの値を出す
indices = np.random.permutation(range(n))[:N_training]
X_train = mnist.data[indices] / 255.0
y_train = mnist.target[indices]
#1ofK表現に変換
Y_train = np.eye(10)[y_train.astype(int)]

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=N_validation)

n_in = 784
n_hiddens = [200, 200, 200]
n_out = 10
activation = "relu"
p_keep = 0.5

model = Sequential()
for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
	model.add(Dense(n_hiddens[i],input_dim=input_dim,kernel_initializer=weight_variable))
	model.add(Activation(activation))
	model.add(Dropout(p_keep))

model.add(Dense(n_out))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.05),metrics=["accuracy"])

epochs = 50
batch_size = 200

hist = model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_validation,Y_validation))

val_acc = hist.history["val_acc"]
plt.rc("font",family="serif")
fig = plt.figure()
plt.plot(range(epochs),val_acc,label="acc",color="black")
plt.xlabel("epochs")
plt.show()
#plt.savefig("mnist_keras.eps")
