import os
import shutil
import cv2

def video_anime_chara_trimming(video_file):
	image_dir = "./output/"
	if os.path.exists(image_dir):
		shutil.rmtree(image_dir)

	if not os.path.exists(image_dir):
		os.makedirs(image_dir)

	i = 0		#img number
	count = 0	#frame count
	classifier = cv2.CascadeClassifier("lbpcascade_animeface.xml")
	cap = cv2.VideoCapture(video_file)
	image_file = "img_%s.jpg"
	cut_frame_num = int((cap.get(cv2.CAP_PROP_FPS)+0.5))
	while(cap.isOpened()):
		count += 1
		flag, frame = cap.read()
		if flag == False:
			break
		if count % cut_frame_num != 0:
			continue
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = classifier.detectMultiScale(gray)
		for coord in faces:
			if coord[2] > 50:
				face_image = frame[coord[1]:coord[1]+coord[3],coord[0]:coord[0]+coord[2]]
				output = image_dir + image_file % str(i).zfill(8)
				cv2.imwrite(output, face_image)
				print("Save image : ",i)
				i += 1

	cap.release()


def main():
	video = "./mideshi2.mp4"
	video_anime_chara_trimming(video)

if __name__ == '__main__':
	main()