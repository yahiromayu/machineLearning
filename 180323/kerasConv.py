import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.datasets import cifar100,mnist
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

def weight_variable(shape):
	return K.truncated_normal(shape, stddev=0.01)

np.random.seed(1)
early_stopping = EarlyStopping(monitor="val_loss", patience=10,verbose=1)

'''
row = 32
col = 32
channel = 3

#データ
(X_train, y_train),(X_test, y_test) = cifar100.load_data()
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
'''
row = 28
col = 28
channel = 1

(X_train, y_train),(X_test, y_test) = mnist.load_data()
X_train = X_train.astype("float32").reshape(X_train.shape[0],row,col,channel)
X_test = X_test.astype("float32").reshape(X_test.shape[0],row,col,channel)
X_train /= 255
X_test /= 255

nb_classes = 100
Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes)

#モデル
model = Sequential()

model.add(Conv2D(128,3,input_shape=(row,col,channel)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(256,3,padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(2048))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(2048))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation("softmax"))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"])

epochs = 100
batch_size = 200

hist = model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_split=0.1,verbose=1,callbacks=[early_stopping])

#評価
score = model.evaluate(X_test,Y_test, verbose=0)
print('test loss:', score[0])
print('test acc:', score[1])
 
model.save('kerasConv.h5')

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