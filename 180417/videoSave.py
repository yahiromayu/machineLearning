import cv2
import os
import numpy as np

def main():
	ipt = cv2.VideoCapture("./mideshi.mp4")
	opt = cv2.VideoWriter("./output.m4v",cv2.VideoWriter_fourcc(*"MP4V"),int(ipt.get(cv2.CAP_PROP_FPS)),(int(ipt.get(cv2.CAP_PROP_FRAME_WIDTH)),int(ipt.get(cv2.CAP_PROP_FRAME_HEIGHT))))

	print(int(ipt.get(cv2.CAP_PROP_FPS)),(int(ipt.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(ipt.get(cv2.CAP_PROP_FRAME_WIDTH))))
	count = 0
	while(ipt.isOpened()):
		flg,frame = ipt.read()
		if flg == False:
			break

		cv2.imshow("",frame)
		cv2.waitKey(0)
		if count ==20:
			break

		opt.write(frame)
		print(count)
		count += 1

	ipt.release()
	opt.release()

if __name__ == '__main__':
	main()