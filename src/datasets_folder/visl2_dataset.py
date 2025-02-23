import os
import math
import random
import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from PIL import Image
try:
    from src.datasets_folder.augmentation import SkeletonAugmentation, RGBDAugmentation
except:
    from datasets_folder.augmentation import SkeletonAugmentation, RGBDAugmentation

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global optical flow instance.
optical_flow = cv2.optflow.createOptFlow_DualTVL1()

class Visl2Dataset(IterableDataset):
    """
    A unified dataset loader for the visl2 dataset.
    Expects each batch folder to have 'depth' (.npy) and 'rgb' (.avi) subfolders.
    Supports output types: 'rgb', 'rgbd', and 'flow'.
    """
    def __init__(self, config,mode='train'):
        self.config = config
        self.dataset_path = config['paths'][f'{mode}_data_path']
        self.height = config['height']
        self.width = config['width']
        self.n_frames = config['n_frames']
        self.batch_size = config['batch_size']
        self.output = config['output']
        self.cache_folder = config.get('cache_folder')
        self.num_classes = config.get('num_classes', 100)


        if self.output == 'skeleton':
            HandLandmarker = mp.solutions.hands.Hands
            self.hand_landmarker = HandLandmarker(static_image_mode=False,min_detection_confidence=0.5,min_tracking_confidence=0.5)

            #Pose
            PoseLandmarker = mp.solutions.pose.Pose
            self.pose_landmarker = PoseLandmarker(static_image_mode=False,min_detection_confidence=0.5,min_tracking_confidence=0.5)

        # Person selection configuration
        self.person_selection = config['person_selection'][mode]
        
        # Build complete person list
        self.all_persons = sorted([d for d in os.listdir(self.dataset_path) 
                                 if os.path.isdir(os.path.join(self.dataset_path, d))])
        
        #filter out the classes if exceed the num_classes
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
                    self.output,(config.get('width'),config.get('height'))) 

    def _filter_persons(self):
        mode = self.person_selection['mode']
        if mode == 'all':
            return self.all_persons
        elif mode == 'list':
            person_list = self.person_selection['persons'] #self.person_selection['persons'] should be a list of strings
            data_list = []
            for p in person_list:
                data_list.extend([ _p for _p in self.all_persons if f'P{p}' in _p])
            return data_list
        elif mode == 'index':
            indices = self.person_selection['indices'] #self.person_selection['indices'] should be a list of integers
            #example indices = [ [1,5],2,[7,9] ] will select persons P1,P5,P2,P7,P9
            data_list = []
            #all_person is a list of all persons in the dataset (AnPm) with Pm the person index
            for i in indices:
                if isinstance(i,list):
                    for j in range(i[0],i[1]+1):
                        data_list.extend([ _p for _p in self.all_persons if f'P{j}' in _p])
                else:
                    data_list.extend([ _p for _p in self.all_persons if f'P{i}' in _p])
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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.labels)
        if worker_info is not None:
            # For multi-worker settings, split the workload.
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = start + worker_id * per_worker
            end = min(start + per_worker, len(self.labels))
        return self.get_data(start, end)

    def get_data(self, start, end):
        batch_data = []
        batch_labels = []
        
        for i in range(start, end):
            label = self.labels[i]
            depth_fp = self.depth_files[i]
            rgb_fp = self.rgb_files[i]
            X = None

            # Optionally use caching for preprocessed samples.
            if self.cache_folder:
                if not os.path.exists(self.cache_folder):
                    os.makedirs(self.cache_folder)
                cache_file = os.path.join(self.cache_folder,
                                          os.path.basename(depth_fp).replace('.npy','') + f"_{self.output}.npy")
                if os.path.exists(cache_file):
                    X = np.load(cache_file)

            if X is None:
                if self.output in ['flow', 'rgbd']:
                    # Load depth data and preprocess.
                    read_depth = np.load(depth_fp)  # Expected shape: (time, H, W)
                    read_depth = read_depth[:-1]  # Remove the last frame as per previous logic.
                    read_depth = np.expand_dims(read_depth, axis=-1)  # Now (time, H, W, 1)
                    n_frames_available = read_depth.shape[0]

                    if self.output == 'flow':
                        # For flow compute frame-to-frame optical flow.
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
                        # For 'rgbd', load the rgb video and combine with depth.
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
                        X = np.concatenate([rgb_frames, read_depth], axis=-1)  # Combined along channel axis.
                else:
                    # For 'rgb' or skeleton output, read the video (avi) file.
                    cap = cv2.VideoCapture(rgb_fp)
                    rgb_frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        rgb_frames.append(frame)
                    cap.release()
                    X = np.array(rgb_frames)


                #For skeleton output, use mediapipe to extract keypoints
                if self.output == 'skeleton':
                    #Extract pose and extract hand landmarks
                    pose_frames = []
                    hand_frames = []
                    for frame in X:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.pose_landmarker.process(frame)
                        pose_frames.append(results.pose_landmarks)
                        results = self.hand_landmarker.process(frame)
                        hand_frames.append(results.multi_hand_landmarks)
                    #Convert to numpy and keep only keypoints_to_use
                    pose_frames = np.array([[[landmark.x,landmark.y,landmark.z] for landmark in frame.landmark] for frame in pose_frames])
                    # Process hand landmarks, accounting for both hands
                    hand_frames_processed = []
                    for frame_hands in hand_frames:
                        if frame_hands is None or len(frame_hands) == 0:
                            # If no hands detected, use zeros 
                            frame_data = np.zeros((42, 3))  # 21 landmarks * 2 hands flattened
                        else:
                            # Process up to 2 hands
                            frame_data = np.zeros((42, 3))  # 21 landmarks * 2 hands flattened
                            for i, hand in enumerate(frame_hands[:2]):  # Only take first two hands if more are detected
                                landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand.landmark])
                                frame_data[i*21:(i+1)*21] = landmarks
                        hand_frames_processed.append(frame_data)
                    hand_frames = np.array(hand_frames_processed)
                    X = np.concatenate([pose_frames,hand_frames],axis=1)
                    #X = X.reshape(X.shape[0],-1)
                    X = X.astype(np.float32)

                    
                    #if keypoints_to_use is defined, keep only those keypoints, otherwise keep all
                    if self.config.get('keypoints_to_use', None):
                        self.keypoints_to_use = self.config['keypoints_to_use']
                        X = X[:,self.keypoints_to_use]
                    
                    
                    
                    
                    #X = X.reshape(X.shape[0],-1)


                # For non-flow outputs, resize and normalize the frames.
                elif self.output != 'flow':
                    X = X.astype(np.uint8)
                    X = np.array([cv2.resize(X[j], (self.width, self.height))
                                  for j in range(X.shape[0])]).astype(np.float32)
                    # Normalize to [-1, 1]
                    X = (X - np.min(X)) / (np.max(X) - np.min(X)) * 2.0 - 1.0

                # add padding if necessary
                if X.shape[0] < self.n_frames:
                    n_pad = self.n_frames - X.shape[0]
                    pad = np.zeros((n_pad, self.height, self.width, X.shape[-1]), dtype=X.dtype)
                    X = np.concatenate([X, pad], axis=0)
                    
                if X.shape[0] > self.n_frames:
                    X = X[:self.n_frames]
                    
                if self.cache_folder:
                        np.save(cache_file, X)

            # Apply augmentation if an augmentation pipeline is defined
            if self.augmentation is not None:
                # Shuffle augmentation parameters once per sequence
                frame_size = (self.height, self.width)
                self.augmentation.shuffle(frame_size)
                
                # Apply augmentations to all frames at once
                X = self.augmentation.apply(X)
                
            if self.output != 'skeleton':
                X = np.transpose(X, (3, 0, 1, 2))
            #re-check the shape if the time is the same
            
            if X.shape[0] > self.n_frames:
                X = X[:self.n_frames]
            elif X.shape[0] < self.n_frames:
                n_pad = self.n_frames - X.shape[0]
                if self.output == 'skeleton':
                    pad = np.zeros((n_pad, X.shape[1], X.shape[2]), dtype=X.dtype)
                else:
                    pad = np.zeros((n_pad, self.height, self.width, X.shape[-1]), dtype=X.dtype)
                X = np.concatenate([X, pad], axis=0)
                
            #label to one hot
            label = int(label.split('P')[0][1:])-1
            label_tensor = torch.zeros(self.num_classes)
            label_tensor[label] = 1
            label = label_tensor
            

        
                

            X_tensor = torch.FloatTensor(X)
            batch_data.append(X_tensor)
            batch_labels.append(label)

            # Yield batch when accumulated batch_size samples
            if len(batch_data) == self.batch_size:
                # Stack tensors along batch dimension
                batch_tensor = torch.stack(batch_data, dim=0)
                yield batch_tensor, batch_labels
                # Clear accumulators
                batch_data = []
                batch_labels = []

        # Yield remaining samples if any
        if batch_data:
            batch_tensor = torch.stack(batch_data, dim=0)
            yield batch_tensor, batch_labels

if __name__ == "__main__":
    # Example usage 
    config = {
        'paths': {
            'train_data_path': "/work/21010294/ViSL-2/Processed"
        },
        'height': 224,
        'width': 224,
        'n_frames': 32,
        'batch_size': 16,  
        'output': 'skeleton',
        'cache_folder': "/work/21010294/ViSL-2/cache/",
        'person_selection': {
            'train': {
                'mode': 'all'
            },
        },
        'augmentation': {
            'use_augmentation': True,
            'augmentations':{
                    'temporal': {
                        'frame_skip_range': (1, 3),        # Skip 1-3 frames
                        'frame_duplicate_prob': 0.2,        # 20% chance to duplicate a frame
                        'temporal_crop_scale': (0.8, 1.0),  # Temporal crop ratio
                        'min_frames': 32                    # Minimum frames to keep
                    },
                    'spatial': {
                        # For skeleton
                        'rotation_range': (-13, 13),
                        'squeeze_range': (0, 0.15),
                        'perspective_ratio_range': (0, 1),
                        'joint_rotation_prob': 0.3,
                        'joint_rotation_range': (-4, 4),
                        
                        # For RGB/RGBD
                        'flip_h_prob': 0.5,
                        'flip_v_prob': 0.0,
                        'brightness_range': (-0.2, 0.2),
                        'contrast_range': (-0.2, 0.2),
                        'saturation_range': (-0.2, 0.2),
                        'hue_range': (-0.1, 0.1),
                        'spatial_crop_scale': (0.8, 1.0),
                        'normalize': {
                            'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]
                        }
                    }
                }
        }
    }
    
    visl2_dataset = Visl2Dataset(config)
    for batch_data, batch_labels in visl2_dataset:
        print("Batch shape:", batch_data.shape)
        print("Batch labels:", batch_labels)
        print("Number of samples in batch:", len(batch_labels))
        break