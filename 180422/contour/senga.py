import cv2
import numpy as np
import os

def make_contour_image(path,num):
	neiborhood24 = np.array([[1,1,1,1,1],
		[1,1,1,1,1],
		[1,1,1,1,1],
		[1,1,1,1,1],
		[1,1,1,1,1]],np.uint8)

	#グレイスケールで画像読み込み
	gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	#cv2.imwrite("./output/"+str(num)+"gray.jpg",gray)

	#白色膨張
	dilated = cv2.dilate(gray,neiborhood24, iterations=1)
	#cv2.imwrite("./output/"+str(num)+"dilated.jpg",dilated)

	#2つの差をとる
	diff = cv2.absdiff(dilated, gray)
	#cv2.imwrite("./output/"+str(num)+"diff.jpg",diff)

	#白黒反転
	contour = 255 - diff
	cv2.imwrite("./output/"+str(num)+"contour.jpg",contour)

	#二値化
	thresh = 220
	max_pixel = 255
	ret,img_dst = cv2.threshold(contour, thresh, max_pixel, cv2.THRESH_BINARY)
	cv2.imwrite("./output/"+str(num)+"output.jpg",img_dst)
	return img_dst


if __name__ == '__main__':
	files = os.listdir("./input/")
	count = 0

	for file in files:
		file = "./input/" + file
		make_contour_image(file, count)
		count = count + 1

