


import torch
from torch.nn import functional as F

class ToTensor:
	def __init__(self):
		pass
   
	def __call__(self, X):
		'''
		
		X: numpy array  T,H,W,C ->  T,C,H,W
		Nomalize video 255 -> [0-1]
		return Tensor 
		'''
		video_tensor = torch.from_numpy(X).float()

  		# T,W,H,C -> C,T,W,H
		video_tensor = video_tensor.permute(-1,0,1,2)

		# Normalize to [0,1]
		video_tensor = video_tensor / 255.0  
		return video_tensor

def OneHotConvert(label,num_classes):
    '''
    label: int
    '''
    return  F.one_hot(torch.tensor(label, dtype=torch.long), num_classes= num_classes).float()