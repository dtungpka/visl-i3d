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
    from src.datasets.augmentation import get_augmentation_pipeline
except:
    from datasets.augmentation import get_augmentation_pipeline

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
        
        # Filter persons based on selection mode
        self.selected_persons = self._filter_persons()
        
        # Build file lists for selected persons
        self.depth_files, self.rgb_files, self.labels = self._build_file_lists()
        
        # Setup augmentation
        aug_config = config.get('augmentation', {})
        self.augmentation = get_augmentation_pipeline(
            aug_config.get('augmentations'), 
            self.output
        ) if aug_config.get('use_augmentation', False) else None

    def _filter_persons(self):
        mode = self.person_selection['mode']
        if mode == 'all':
            return self.all_persons
        elif mode == 'list':
            person_list = self.person_selection['persons']
            return [p for p in self.all_persons if p in person_list]
        elif mode == 'index':
            indices = self.person_selection['indices']
            return [self.all_persons[i] for i in indices if i < len(self.all_persons)]
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
                    hand_frames = np.array([[[landmark.x,landmark.y,landmark.z] for landmark in hand.landmark] for hand in hand_frames])
                    X = np.concatenate([pose_frames,hand_frames],axis=-1)
                    X = X.reshape(X.shape[0],-1)
                    X = X.astype(np.float32)

                    self.keypoints_to_use = config['keypoints_to_use'] #list of indices to keep
                    X = X[:,self.keypoints_to_use]
                    X = X.reshape(X.shape[0],-1)


                # For non-flow outputs, resize and normalize the frames.
                elif self.output != 'flow':
                    X = X.astype(np.uint8)
                    X = np.array([cv2.resize(X[j], (self.width, self.height))
                                  for j in range(X.shape[0])]).astype(np.float32)
                    # Normalize to [-1, 1]
                    X = (X - np.min(X)) / (np.max(X) - np.min(X)) * 2.0 - 1.0

                # Adjust length to match desired n_frames.
                if X.shape[0] < self.n_frames:
                    X = np.tile(X, (self.n_frames // X.shape[0] + 1, 1, 1, 1))
                if X.shape[0] > self.n_frames:
                    X = X[:self.n_frames]

                # Apply augmentation frame‐wise if an augmentation pipeline is defined.
                if self.augmentation is not None and self.output in ['rgb', 'rgbd', 'skeleton']:
                    augmented_frames = []
                    for j in range(X.shape[0]):
                        if self.output == 'rgbd':
                            # Split RGB and depth channels
                            rgb_frame = X[j, :3]  # Shape: (H, W, 3)
                            depth_channel = X[j, 3]  # Shape: (H, W)
                            
                            # Convert to uint8 for augmentation
                            rgb_img = rgb_frame.astype(np.uint8)
                            
                            # Apply augmentation to RGB
                            aug_rgb = self.augmentation(rgb_img)
                            
                            # Convert augmented RGB to numpy
                            if isinstance(aug_rgb, torch.Tensor):
                                aug_rgb = aug_rgb.permute(1, 2, 0).numpy()  # Shape: (H, W, 3)
                            elif isinstance(aug_rgb, Image.Image):
                                aug_rgb = np.array(aug_rgb)  # Shape: (H, W, 3)
                            
                            # Resize depth to match augmented RGB dimensions
                            depth_resized = cv2.resize(depth_channel, (aug_rgb.shape[1], aug_rgb.shape[0]))  # Shape: (H, W)
                            depth_resized = np.expand_dims(depth_resized, axis=-1)  # Shape: (H, W, 1)
                             
                            # Concatenate along channel dimension
                            frame_aug = np.concatenate([aug_rgb, depth_resized], axis=-1)  # Shape: (H, W, 4)
                        elif self.output == 'rgb':
                            # For rgb, simply augment
                            frame = X[j].astype(np.uint8)
                            aug_frame = self.augmentation(frame)
                            if isinstance(aug_frame, torch.Tensor):
                                frame_aug = aug_frame.permute(1, 2, 0).numpy()
                            elif isinstance(aug_frame, Image.Image):
                                frame_aug = np.array(aug_frame)
                        elif self.output == 'skeleton':
                            frame = X[j].astype(np.float32)
                            frame_aug = self.augmentation(frame)
                            
                        augmented_frames.append(frame_aug)
                    # Stack along time dimension
                    X = np.stack(augmented_frames, axis=0)  # Change: Stack along time dimension first
                    # Transpose to (C, T, H, W)
                    X = np.transpose(X, (3, 0, 1, 2))
                else:
                    # Transpose X to (C, T, H, W)
                    X = np.transpose(X, (3, 0, 1, 2))

                if self.cache_folder:
                    np.save(cache_file, X)

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
    # Example usage with batch processing
    config = {
        'paths': {
            'train_data_path': "/work/21010294/ViSL-2/Processed"
        },
        'height': 224,
        'width': 224,
        'n_frames': 320,
        'batch_size': 16,  # Smaller batch size for testing
        'output': 'rgbd',
        'cache_folder': "/work/21010294/ViSL-2/cache/",
        'person_selection': {
            'mode': 'all'
        },
        'augmentation': {
            'use_augmentation': True,
            'augmentations': [
                {
                    'type': 'random_crop',
                    'size': [224, 224],
                    'scale': [0.8, 1.0]
                },
                {
                    'type': 'random_flip',
                    'horizontal': True,
                    'vertical': False
                }
            ]
        }
    }
    
    visl2_dataset = Visl2Dataset(config)
    for batch_data, batch_labels in visl2_dataset:
        print("Batch shape:", batch_data.shape)
        print("Batch labels:", batch_labels)
        print("Number of samples in batch:", len(batch_labels))
        break