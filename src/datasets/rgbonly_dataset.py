


import os
import torch
from torch.utils.data import Dataset
from os.path import join
import sys


#import every where in the source
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils_folder.load_data import load_video
from utils_folder.time_augumentation import AdjustNFrames
from utils_folder.spatial_transforms import Resize
from utils_folder import ToTensor,OneHotConvert

#batch input 
def collate_fn(batch):

		batch_rgb_tensor = []
		batch_label_tensor = []
  
		for sample in batch:
			rgb_tensor = sample['rgb_tensor']
			label_tensor = sample['label_tensor']

			batch_rgb_tensor.append(rgb_tensor.unsqueeze(0))
			batch_label_tensor.append(label_tensor.unsqueeze(0))
		
		#list to tensor B,C,T,W,H
		batch_rgb_tensor = torch.cat(batch_rgb_tensor,dim = 0)
		batch_label_tensor = torch.cat(batch_label_tensor,dim =0)
  
		batch_data  = {
			'batch_rgb': batch_rgb_tensor,
			'batch_label': batch_label_tensor
		}
  
		return batch_data

class RgbOnlyDataset(Dataset):
	def __init__(self, config,mode):
		
		self.config = config
		self.root_dir = config['root_dir']
		self.num_classes = config['num_classes']
		self.n_frames = config['num_frames']
		self.img_size = config['img_size']
  
		self.array_video_transforms = [
				AdjustNFrames(self.n_frames),
				#cv2 resize
				Resize(self.img_size),
				#convert numpy arrays [0-255] T,H,W,C to tensor [0-1] with shape C,T,H,W
				ToTensor()
			]
  
		if mode == 'train':
			pass
		elif mode in ['val','test']:
			pass
		else:
			print(f"Phase {self.config['phase']} not found in ['tran','val]")
		
		
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
    
		self.mode = mode
		#batch data while using dataloader
		self.collate_fn = collate_fn 
		print(f"Phase {mode} totally has {len(self)} samples!")
	def __len__(self):
		return len(self.all_video_paths)
	def __getitem__(self, index):
	 
		video_path = self.all_video_paths[index]
		label = self.all_labels[index]
	
		rgb_data = load_video(video_path)
  
		for numpy_video_transform in self.array_video_transforms:
			rgb_data = numpy_video_transform(rgb_data)
  
		# label : int -> tensor num_classes [0,1,0,0,0,...,0]
		label_tensor = OneHotConvert(label,num_classes = self.num_classes)

		data_dir = {
			'rgb_tensor': rgb_data,
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