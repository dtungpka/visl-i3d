



import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join
import sys
from torch.nn import functional as F

#import every where in the source
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils_folder.load_data import load_video

#batch input 
def collate_fn(batch):

		batch_rgb_tensor = []
		batch_label_tensor = []
  
		for sample in batch:
			rgb_tensor = sample['rgb_tensor']
			label_tensor = sample['label_tensor']

			batch_rgb_tensor.append(rgb_tensor)
			batch_label_tensor.append(label_tensor)

		#list to tensor B,N,C,W,H
		batch_rgb_tensor = torch.cat(batch_rgb_tensor,dim = 0)
		batch_label_tensor = torch.cat(batch_label_tensor,dim =0)
  
		batch_data  = {
			'batch_rgb': batch_rgb_tensor,
			'label_rgb': batch_label_tensor
		}
  
		return batch_data

class RgbOnlyDataset(Dataset):
	def __init__(self, config):
		
		self.config = config
		
		if self.config['phase'] == 'train':
			# not implemented yet
			self.transforms = None
		elif self.config['phase'] == 'val':
			# not implemented yet
			self.transforms = None
		else:
			print(f"Phase {self.config['phase']} not found in ['tran','val]")
			
		self.root_dir = config['root_dir']
		self.num_classes = config['num_classes']
  
		self.all_video_paths = []
		self.all_labels = []
  
		for folder_name in os.listdir(self.root_dir):
			#folder_name strure AnPm --> n is class index
			label = folder_name.split('P')[0][1:]
			label = int(label)

			folder_path = join(self.root_dir,folder_name)

			rgb_path = join(folder_path,'rgb')
   
			for file_name in os.listdir(rgb_path):
				video_path = os.path.join(rgb_path,file_name)
				self.all_video_paths.append(video_path)
				self.all_labels.append(label)
	
		#batch data while using dataloader
		self.collate_fn = collate_fn 
  
	def __len__(self):
		return len(self.all_video_paths)
	def __getitem__(self, index):
	 
		video_path = self.all_video_paths[index]
		label = self.all_labels[index]
	
		video_np = load_video(video_path)

		video_tensor = torch.from_numpy(video_np).float()
		# T,W,H,C -> C,T,W,H
		video_tensor = video_tensor.permute(0,3,1,2)
  
		# Normalize to [0,1]
		video_tensor = video_tensor / 255.0  
  
		label_tensor = F.one_hot(torch.tensor(label, dtype=torch.long), num_classes=self.num_classes) 

		data_dir = {
			'rgb_tensor': video_tensor,
			'label_tensor':   label_tensor
		}
  
		return data_dir
				
	

if __name__ == "__main__":
	# Example usage with batch processing
	config = {
		'phase': 'train',
		'root_dir': '/work/21013187/SignLanguageRGBD/data/ver2_all_rgb_only',
		'num_classes': 200
		
	}
	
	visl2_dataset = RgbOnlyDataset(config)
	data_dir = visl2_dataset[0]
 
	print(data_dir['rgb_tensor'].shape)
	print(data_dir['label_tensor'].shape)