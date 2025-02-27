import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Use absolute imports when possible, with clear fallback
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from src.datasets.augmentation import SkeletonAugmentation, RGBDAugmentation
    from src.utils.skeleton_processor import MediaPipeProcessor
except ImportError:
    try:
        from datasets.augmentation import SkeletonAugmentation, RGBDAugmentation
        from utils.skeleton_processor import MediaPipeProcessor
    except ImportError:
        # Last resort, try relative import
        from .augmentation import SkeletonAugmentation, RGBDAugmentation
        from ..utils.skeleton_processor import MediaPipeProcessor

# Global optical flow instance.
optical_flow = cv2.optflow.createOptFlow_DualTVL1()

class Visl2Dataset(Dataset):
    """
    A unified dataset loader for the visl2 dataset.
    Expects each batch folder to have 'depth' (.npy) and 'rgb' (.avi) subfolders.
    Supports output types: 'rgb', 'rgbd', 'skeleton', and 'flow'.
    """
    def __init__(self, config, mode='train'):
        self.config = config
        self.dataset_path = config['paths'][f'{mode}_data_path']
        self.height = config['height']
        self.width = config['width']
        self.n_frames = config['n_frames']
        self.output = config['output']
        self.cache_folder = config.get('cache_folder')
        self.num_classes = config.get('num_classes', 100)
        
        # Create a MediaPipeProcessor if needed (singleton pattern)
        if self.output == 'skeleton':
            self.mediapipe_processor = MediaPipeProcessor()

        # Person selection configuration with default to 'all' if not present
        if 'person_selection' not in config:
            self.person_selection = {'mode': 'all'}
        else:
            self.person_selection = config['person_selection'].get(mode, {'mode': 'all'})
        
        # Build complete person list
        self.all_persons = sorted([d for d in os.listdir(self.dataset_path) 
                                 if os.path.isdir(os.path.join(self.dataset_path, d))])
        
        # Filter out classes if they exceed num_classes
        if self.num_classes is not None:
            self.all_persons = [p for p in self.all_persons if int(p.split('P')[0][1:]) <= self.num_classes]
        
        # Filter persons based on selection mode
        self.selected_persons = self._filter_persons()
        
        # Build file lists for selected persons
        self.depth_files, self.rgb_files, self.labels = self._build_file_lists()
        
        # Setup augmentation
        aug_config = config.get('augmentation', {})
        self.augmentation = None
        if aug_config.get('use_augmentation', False):
            if self.output == 'skeleton':
                self.augmentation = SkeletonAugmentation(
                    aug_config.get('augmentations'))
            elif self.output in ['rgb', 'rgbd', 'flow']:
                self.augmentation = RGBDAugmentation(
                    aug_config.get('augmentation'), 
                    self.output, (config.get('width'), config.get('height')))
                    
        print(f"Mode {mode} loaded with {len(self)} samples")

    def _filter_persons(self):
        mode = self.person_selection['mode']
        if mode == 'all':
            return self.all_persons
        elif mode == 'list':
            person_list = self.person_selection['persons']  # list of strings
            data_list = []
            for p in person_list:
                data_list.extend([_p for _p in self.all_persons if f'P{p}' in _p])
            return data_list
        elif mode == 'index':
            indices = self.person_selection['indices']  # list of integers or ranges
            data_list = []
            for i in indices:
                if isinstance(i, list):
                    for j in range(i[0], i[1]+1):
                        data_list.extend([_p for _p in self.all_persons if f'P{j}' in _p])
                else:
                    data_list.extend([_p for _p in self.all_persons if f'P{i}' in _p])
            return data_list
        
        return self.all_persons

    def _build_file_lists(self):
        depth_files = []
        rgb_files = []
        labels = []
        
        for person in self.selected_persons:
            person_path = os.path.join(self.dataset_path, person)
            depth_dir = os.path.join(person_path, 'depth')
            rgb_dir = os.path.join(person_path, 'rgb')
            
            if os.path.isdir(depth_dir) and os.path.isdir(rgb_dir):
                depth_list = sorted(os.listdir(depth_dir))
                for file in depth_list:
                    depth_fp = os.path.join(depth_dir, file)
                    rgb_fp = os.path.join(rgb_dir, file.replace('.npy', '.avi'))
                    if os.path.exists(rgb_fp):
                        depth_files.append(depth_fp)
                        rgb_files.append(rgb_fp)
                        labels.append(person)
        
        return depth_files, rgb_files, labels

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        try:
            label = self.labels[index]
            depth_fp = self.depth_files[index]
            rgb_fp = self.rgb_files[index]
            X = None

            # Check if we're in a worker process
            worker_info = getattr(torch.utils.data.get_worker_info(), 'id', None)
            is_worker = worker_info is not None

            # Optionally use caching for preprocessed samples
            if self.cache_folder:
                if not os.path.exists(self.cache_folder):
                    os.makedirs(self.cache_folder)
                cache_file = os.path.join(self.cache_folder,
                                         os.path.basename(depth_fp).replace('.npy','') + f"_{self.output}.npy")
                if os.path.exists(cache_file):
                    try:
                        X = np.load(cache_file)
                    except Exception as e:
                        print(f"Error loading cached file {cache_file}: {e}")
                        X = None

            if X is None:
                if self.output == 'skeleton':
                    # For worker processes, return default skeleton to avoid MediaPipe issues
                    X = np.zeros((64, 75, 3), dtype=np.float32)  # Default skeleton

                    # In main process, use MediaPipe
                    is_verbose = getattr(self, 'verbose', False)
                    X = self.mediapipe_processor.process_video(rgb_fp, show_progress=is_verbose)
                    
                    # If processing failed, create a default skeleton
                    if X is None:
                        X = np.zeros((1, 75, 3), dtype=np.float32)
                    
                    # If keypoints_to_use is defined, keep only those keypoints
                    if self.config.get('keypoints_to_use', None):
                        self.keypoints_to_use = self.config['keypoints_to_use']
                        X = X[:, self.keypoints_to_use]
                    
                    if self.cache_folder:
                        np.save(cache_file, X)
                
                # [Rest of the code for other output types remains the same]
                elif self.output in ['flow', 'rgbd']:
                    # Load depth data and preprocess
                    try:
                        read_depth = np.load(depth_fp)  # Expected shape: (time, H, W)
                        read_depth = read_depth[:-1]  # Remove the last frame as per previous logic
                        read_depth = np.expand_dims(read_depth, axis=-1)  # Now (time, H, W, 1)
                        n_frames_available = read_depth.shape[0]

                        if self.output == 'flow':
                            # For flow compute frame-to-frame optical flow
                            depth_uint8 = read_depth.astype(np.uint8)
                            frames = [cv2.resize(depth_uint8[j], (self.width, self.height))
                                    for j in range(n_frames_available)]
                            prev = frames[0]
                            flow = np.zeros((n_frames_available, self.height, self.width, 2), dtype=np.float32)
                            for _i in range(1, n_frames_available):
                                flow[_i] = optical_flow.calc(prev, frames[_i], None)
                                prev = frames[_i]
                            X = flow
                        else:
                            # For 'rgbd', load the rgb video and combine with depth
                            cap = cv2.VideoCapture(rgb_fp)
                            rgb_frames = []
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                rgb_frames.append(frame)
                            cap.release()
                            rgb_frames = np.array(rgb_frames)
                            if rgb_frames.shape[0] > n_frames_available:
                                rgb_frames = rgb_frames[:n_frames_available]
                            elif rgb_frames.shape[0] < n_frames_available:
                                read_depth = read_depth[:rgb_frames.shape[0]]
                            X = np.concatenate([rgb_frames, read_depth], axis=-1)  # Combined along channel axis
                    except Exception as e:
                        print(f"Error processing {depth_fp}: {e}")
                        # Create default empty array
                        X = np.zeros((1, self.height, self.width, 4 if self.output == 'rgbd' else 2), dtype=np.float32)
                else:
                    # For 'rgb', read the video (avi) file
                    try:
                        cap = cv2.VideoCapture(rgb_fp)
                        rgb_frames = []
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            rgb_frames.append(frame)
                        cap.release()
                        
                        if not rgb_frames:
                            raise ValueError("No frames found in video")
                            
                        X = np.array(rgb_frames)
                        
                        # Resize and normalize
                        X = X.astype(np.uint8)
                        X = np.array([cv2.resize(X[j], (self.width, self.height))
                                    for j in range(X.shape[0])]).astype(np.float32)
                        # Normalize to [-1, 1]
                        X = (X - np.min(X)) / (np.max(X) - np.min(X)) * 2.0 - 1.0
                    except Exception as e:
                        print(f"Error processing RGB video {rgb_fp}: {e}")
                        # Create default empty array
                        X = np.zeros((1, self.height, self.width, 3), dtype=np.float32)

                if self.cache_folder and X is not None:
                    np.save(cache_file, X)

            # Apply augmentation if defined
            if self.augmentation is not None and X is not None:
                try:
                    # Shuffle augmentation parameters once per sequence
                    frame_size = (self.height, self.width)
                    self.augmentation.shuffle(frame_size)
                    
                    # Apply augmentations to all frames at once
                    X = self.augmentation.apply(X)
                except Exception as e:
                    print(f"Augmentation error: {e}")
            
            # Ensure X has the right format and shape
            if X is None:
                if self.output == 'skeleton':
                    X = np.zeros((self.n_frames, 75, 3), dtype=np.float32)
                else:
                    X = np.zeros((self.n_frames, self.height, self.width, 3), dtype=np.float32)
            
            # Transpose for non-skeleton outputs
            if self.output != 'skeleton':
                X = np.transpose(X, (3, 0, 1, 2))
            
            # Handle frame count adjustments
            if X.shape[0] > self.n_frames:
                X = X[:self.n_frames]
            elif X.shape[0] < self.n_frames:
                n_pad = self.n_frames - X.shape[0]
                if self.output == 'skeleton':
                    pad = np.zeros((n_pad, X.shape[1], X.shape[2]), dtype=X.dtype)
                else:
                    pad = np.zeros((n_pad, X.shape[1], X.shape[2], X.shape[3]), dtype=X.dtype)
                X = np.concatenate([X, pad], axis=0)
            
            # Convert label to tensor
            label_idx = int(label.split('P')[0][1:]) - 1
            label_tensor = torch.tensor(label_idx, dtype=torch.long)

            # Convert data to tensor
            X = torch.FloatTensor(X)
            
            return {
                'data': X,
                'label': label_tensor
            }
            
        except Exception as e:
            print(f"Error in __getitem__ for index {index}: {e}")
            # Return a default/empty sample with correct shapes
            if self.output == 'skeleton':
                X = torch.zeros((self.n_frames, 75, 3), dtype=torch.float32)
            else:
                X = torch.zeros((3, self.n_frames, self.height, self.width), dtype=torch.float32)
            
            # Default to class 0
            label_tensor = torch.zeros(1, dtype=torch.long)
            
            return {
                'data': X,
                'label': label_tensor
            }

    # Add this as a method to the Visl2Dataset class
    def collate_fn(self, batch):
        """
        Custom collate function for Visl2Dataset with enhanced error handling.
        Expects each sample to be a dict with 'data' and 'label' keys.
        """
        if not batch:
            return None
        
        valid_samples = []
        for item in batch:
            # Check for valid data
            if item['data'] is not None and isinstance(item['data'], torch.Tensor) and item['label'] is not None:
                valid_samples.append(item)
        
        if not valid_samples:
            # Return dummy batch if no valid samples
            dummy_data = torch.zeros((1, self.n_frames, 75, 3) if self.output == 'skeleton' 
                                     else (1, 3, self.n_frames, self.height, self.width), dtype=torch.float32)
            dummy_labels = torch.zeros(1, dtype=torch.long)
            return dummy_data, dummy_labels
                
        # Extract data and labels
        batch_data = [item['data'] for item in valid_samples]
        batch_labels = [item['label'] for item in valid_samples]
            
        # Stack tensors along batch dimension
        try:
            batch_tensor = torch.stack(batch_data, dim=0)
            batch_labels = torch.stack(batch_labels, dim=0)
        except Exception as e:
            print(f"Error stacking batch: {e}")
            # Return the first valid sample as a batch of size 1
            if valid_samples:
                return valid_samples[0]['data'].unsqueeze(0), valid_samples[0]['label'].unsqueeze(0)
            else:
                # Return dummy batch
                dummy_data = torch.zeros((1, self.n_frames, 75, 3) if self.output == 'skeleton' 
                                         else (1, 3, self.n_frames, self.height, self.width), dtype=torch.float32)
                dummy_labels = torch.zeros(1, dtype=torch.long)
                return dummy_data, dummy_labels
        #print(batch_tensor.sum())
        return batch_tensor, batch_labels

# Register the dataset directly - this ensures it's available regardless of import method
if __name__ != "__main__": 
    # When imported as a module
    try:
        # Try to import DatasetRegistry - handle both cases
        try:
            from src.datasets import DatasetRegistry
        except ImportError:
            try:
                from datasets import DatasetRegistry
            except ImportError:
                # Last resort, assume it's in the same package
                from . import DatasetRegistry
                
        # Register the dataset
        DatasetRegistry.register('sign_dataset', Visl2Dataset)
        print("Registered Visl2Dataset as 'sign_dataset'")
    except ImportError as e:
        print(f"Error registering Visl2Dataset: {e}")

if __name__ == "__main__":
    # Example usage 
    config = {
        'paths': {
            'train_data_path': "/work/21010294/ViSL-2/Processed",
            'val_data_path': "/work/21010294/ViSL-2/Processed", 
            'test_data_path': "/work/21010294/ViSL-2/Processed"
        },
        'height': 224,
        'width': 224,
        'n_frames': 64,
        'batch_size': 64,
        'num_classes': 100,
        'output': 'skeleton',
        'cache_folder': "/work/21010294/ViSL-2/cache_64/",
        'person_selection': {
            'train': {
                'mode': 'index',
                'indices': [[6,99]]
            },
            'val': {
                'mode': 'index', 
                'indices': [[4,5]]
            },
            'test': {
                'mode': 'index',
                'indices': [[1,3]] 
            }
        },
        'augmentation': {
            'use_augmentation': True,
            'augmentations':{
                'temporal': {
                    'frame_skip_range': (1, 3),
                    'frame_duplicate_prob': 0.2,
                    'temporal_crop_scale': (0.8, 1.0),
                    'min_frames': 64
                },
                'spatial': {
                    'rotation_range': (-13, 13),
                    'squeeze_range': (0, 0.15), 
                    'perspective_ratio_range': (0, 1),
                    'joint_rotation_prob': 0.3,
                    'joint_rotation_range': (-4, 4)
                }
            }
        }
    }
    
    from torch.utils.data import DataLoader
    
    # Import and register the dataset when running as script
    try:
        from src.datasets import DatasetRegistry
    except ImportError:
        try:
            from datasets import DatasetRegistry
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
            from src.datasets import DatasetRegistry
    
    # Test dataset
    visl2_dataset = Visl2Dataset(config, mode='train')
    print(f"Dataset length: {len(visl2_dataset)}")
    
    # Test single sample
    sample = visl2_dataset[0]
    print(f"Sample data shape: {sample['data'].shape}")
    print(f"Sample label: {sample['label']}")
    
    # Test with DataLoader using dataset's own collate_fn
    dataloader = DataLoader(
        visl2_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        collate_fn=visl2_dataset.collate_fn  # Use dataset's own method instead
    )
    
    for batch_data, batch_labels in dataloader:
        print("Batch shape:", batch_data.shape)
        print("Batch labels:", batch_labels)
        print("Number of samples in batch:", len(batch_labels))
        break

    # Also register here when running as script
    DatasetRegistry.register('visl2_dataset', Visl2Dataset)