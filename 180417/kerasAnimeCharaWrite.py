import os
import shutil
import cv2
import numpy as np
from PIL import Image
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

def recognize(model, image):
	height, width = image.shape[:2]
	RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	predict = model.predict(np.reshape(cv2.resize(RGB_image/255,(64,64)),(1,64,64,3)))
	chara_num = max(enumerate(predict[0]),key=lambda x: x[1])[0]
	if chara_num == 0:
		image = cv2.rectangle(image, (0,0), (width,height), (204,102,255),10,4)
	elif chara_num == 1:
		image = cv2.rectangle(image, (0,0), (width,height), (102,204,255),10,4)
	elif chara_num == 2:
		image = cv2.rectangle(image, (0,0), (width,height), (255,255,153),10,4)
	elif chara_num == 3:
		image = cv2.rectangle(image, (0,0), (width,height), (0,0,0),10,4)

	return image

def video_anime_chara_write(video_file,h5):
	model = load_model(h5)

	classifier = cv2.CascadeClassifier("lbpcascade_animeface.xml")
	fourcc = cv2.VideoWriter_fourcc(*"MP4V")

	cap = cv2.VideoCapture(video_file)

	flg, frame = cap.read()
	height, width = frame.shape[:2]
	frame_rate = int((cap.get(cv2.CAP_PROP_FPS)+0.5))
	all_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print("ALL COUNT = ", all_frame)

	rec = cv2.VideoWriter("output.m4v", fourcc, frame_rate, (width,height))

	count = 0
	while(cap.isOpened()):
		if count != 0:
			flg,frame = cap.read()
		if flg == False:
			break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = classifier.detectMultiScale(gray)
		for coord in faces:
			if coord[2] > 50:
				face_image = frame[coord[1]:coord[1]+coord[3],coord[0]:coord[0]+coord[2]]
				face_image = recognize(model,face_image)
				frame[coord[1]:coord[1]+coord[3],coord[0]:coord[0]+coord[2]] = face_image

		rec.write(frame)
		print("finish ",count, " / ", all_frame)
		count += 1

	cap.release()
	rec.release()


def main():
	video = "./mideshi.mp4"
	h5 = "./h5/mideshi.h5"

	#引数は(映像データのパス、h5ファイルのパス)
	video_anime_chara_write(video,h5)

if __name__ == '__main__':
	main()