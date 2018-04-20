import shutil
import cv2
import numpy as np
import os

def main():
	print("Please push the number key!")
	number = 0
	while(1):
		img_path = "./output/img_%s.jpg" % str(number).zfill(8)
		if not os.path.isfile(img_path):
			number += 1
			#break
			continue
		img = cv2.imread(img_path)
		cv2.imshow("Please number Button to Distribute the Class.",img)

		key = cv2.waitKey(0)
		if key == ord("0"):
			if not os.path.exists("./class/0"):
				os.makedirs("./class/0/")
			shutil.move("./output/img_%s.jpg" % str(number).zfill(8), "./class/0/")
		elif key == ord("1"):
			if not os.path.exists("./class/1"):
				os.makedirs("./class/1/")
			shutil.move("./output/img_%s.jpg" % str(number).zfill(8), "./class/1/")
		elif key == ord("2"):
			if not os.path.exists("./class/2"):
				os.makedirs("./class/2/")
			shutil.move("./output/img_%s.jpg" % str(number).zfill(8), "./class/2/")
		elif key == ord("3"):
			if not os.path.exists("./class/3"):
				os.makedirs("./class/3/")
			shutil.move("./output/img_%s.jpg" % str(number).zfill(8), "./class/3/")
		elif key == ord("4"):
			if not os.path.exists("./class/4"):
				os.makedirs("./class/4/")
			shutil.move("./output/img_%s.jpg" % str(number).zfill(8), "./class/4/")
		elif key == ord("5"):
			if not os.path.exists("./class/5"):
				os.makedirs("./class/5/")
			shutil.move("./output/img_%s.jpg" % str(number).zfill(8), "./class/5/")
		elif key == ord("6"):
			if not os.path.exists("./class/6"):
				os.makedirs("./class/6/")
			shutil.move("./output/img_%s.jpg" % str(number).zfill(8), "./class/6/")
		elif key == ord("7"):
			if not os.path.exists("./class/7"):
				os.makedirs("./class/7/")
			shutil.move("./output/img_%s.jpg" % str(number).zfill(8), "./class/7/")
		elif key == ord("8"):
			if not os.path.exists("./class/8"):
				os.makedirs("./class/8/")
			shutil.move("./output/img_%s.jpg" % str(number).zfill(8), "./class/8/")
		elif key == ord("9"):
			if not os.path.exists("./class/9"):
				os.makedirs("./class/9/")
			shutil.move("./output/img_%s.jpg" % str(number).zfill(8), "./class/9/")
		elif key == ord("z"):
			number -= 1
			continue
		elif key == ord("q"):
			break

		number += 1
		cv2.destroyAllWindows()

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()