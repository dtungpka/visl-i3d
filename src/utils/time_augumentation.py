

import numpy as np


class AdjustNFrames:
	def __init__(self,n_frames):
		self.n_frames = n_frames
	def __call__(self,X):
		'''
		Adjust the number of frames in the input array to match the desired number of frames.
		X: numppy array  T,H,W,C
		
		'''
		if X.shape[0] < self.n_frames:
			X = np.tile(X, (self.n_frames // X.shape[0] + 1, 1, 1, 1))
		if X.shape[0] > self.n_frames:
			X = X[:self.n_frames]
   
		return X