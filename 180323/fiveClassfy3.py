import numpy as np
from PIL import Image
import os
import sys
import glob
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import img_to_array, load_img, list_pictures
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

def main():
	early_stopping = EarlyStopping(monitor="val_loss", patience=10,verbose=1)

	#画像準備
	row = 64
	col = 64
	channel = 3
	X_train = []
	Y_train = []
	img_folders = glob.glob(os.path.abspath("./class/*"))
	for i,folder in enumerate(img_folders):
		#img_list = glob.glob(folder+"/extended/*.jpg")
		img_list = glob.glob(folder+"/*.jpg")
		for img_path in img_list:
			img = img_to_array(load_img(img_path, target_size=(row, col)))
			X_train.append(img)
			Y_train.append(i)
	X_train = np.asarray(X_train)/255
	Y_train = np.asarray(Y_train)
	nb_classes = len(img_folders)
	X_train,Y_train = shuffle(X_train,Y_train)
	Y_train = np_utils.to_categorical(Y_train,nb_classes)

	X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=0)
	
	#モデル
		
	model = Sequential()

	model.add(Conv2D(64,3,input_shape=(row,col,channel)))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))

	model.add(Conv2D(128,3,padding="same"))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))

	model.add(Conv2D(256,3,padding="same"))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))

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

	optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)
	model.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"])

	model.summary()

	epochs = 100
	batch_size = 200
	
	hist = model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_split=0.1,verbose=1,callbacks=[early_stopping])
	
	#評価
	score = model.evaluate(X_test,Y_test, verbose=0)
	print('test loss:', score[0])
	print('test acc:', score[1])
 
	model.save('fiveClassfy2.h5')

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
	
if __name__=="__main__":
	main()