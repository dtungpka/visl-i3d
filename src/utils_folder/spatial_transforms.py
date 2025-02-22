

import cv2

class Resize:
	def __init__(self, size):
		if isinstance(size,(tuple,list)):
			self.size = size
		else:
			self.size = (size,size)
   
	def __call__(self, X):
		'''
		Resize the input array to the specified size.
		X: numpy array  T,H,W,C
		
		'''
		new_video = []
		for frame in X:
			new_video.append( cv2.resize(frame, self.size))
			
		return X