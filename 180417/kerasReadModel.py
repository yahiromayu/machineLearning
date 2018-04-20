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
import cv2

def main():
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
	model = load_model("./h5/mideshi.h5")

	#model.summary()

	#評価
	score = model.evaluate(X_test,Y_test, verbose=0)
	print('test loss:', score[0])
	print('test acc:', score[1])
	
	print()
	print("n:NEXT  p:PREV  q:QUIT")
	num = 0
	while(1):
		print(X_test.shape[:4])
		exit(1)
		print(model.predict(X_test)[num]," : ",Y_test[num])
		processing_img = X_test[num]
		show_img = cv2.cvtColor(processing_img,cv2.COLOR_RGB2BGR)
		cv2.imshow("window",show_img)


		key = cv2.waitKey(0)
		if key == ord("n"):
			if num!=len(X_test)-1:
				num += 1
		elif key == ord("p"):
			if num!=0:
				num -= 1
		elif key == ord("q"):
			break

		cv2.destroyAllWindows()

if __name__=="__main__":
	main()