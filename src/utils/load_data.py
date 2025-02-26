
import cv2
import numpy as np


def load_video(video_path):
    
	cap = cv2.VideoCapture(video_path)
	rgb_frames = []
 
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		rgb_frames.append(frame)
	cap.release()
	X = np.array(rgb_frames)
 
	return X
