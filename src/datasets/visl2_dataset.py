import os
import math
import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from PIL import Image
from src.datasets.augmentation import get_augmentation_pipeline  # NEW import

# Global optical flow instance.
optical_flow = cv2.optflow.createOptFlow_DualTVL1()

class Visl2Dataset(IterableDataset):
    """
    A unified dataset loader for the visl2 dataset.
    Expects each batch folder to have 'depth' (.npy) and 'rgb' (.avi) subfolders.
    Supports output types: 'rgb', 'rgbd', and 'flow'.
    """
    def __init__(self, dataset_path, height=224, width=224, n_frames=320, batch_size=1,
                 output='rgb', cache_folder=None, apply_aug=False, aug_config=None):
        self.dataset_path = dataset_path
        self.height = height
        self.width = width
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.output = output
        self.cache_folder = cache_folder
        self.apply_aug = apply_aug

        # Build file lists by scanning each batch folder.
        self.depth_files = []
        self.rgb_files = []
        self.labels = []
        for batch in sorted(os.listdir(self.dataset_path)):
            batch_path = os.path.join(self.dataset_path, batch)
            depth_dir = os.path.join(batch_path, 'depth')
            rgb_dir = os.path.join(batch_path, 'rgb')
            if os.path.isdir(depth_dir) and os.path.isdir(rgb_dir):
                depth_list = sorted(os.listdir(depth_dir))
                for file in depth_list:
                    depth_fp = os.path.join(depth_dir, file)
                    rgb_fp = os.path.join(rgb_dir, file.replace('.npy', '.avi'))
                    if os.path.exists(rgb_fp):
                        self.depth_files.append(depth_fp)
                        self.rgb_files.append(rgb_fp)
                        self.labels.append(batch)
        self.dataset_len = len(self.labels)

        # Use the unified augmentation pipeline if apply_aug is True.
        self.augmentation = get_augmentation_pipeline(aug_config, self.output) if self.apply_aug else None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = self.dataset_len
        if worker_info is not None:
            # For multi-worker settings, split the workload.
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = start + worker_id * per_worker
            end = min(start + per_worker, self.dataset_len)
        return self.get_data(start, end)

    def get_data(self, start, end):
        for i in range(start, end):
            label = self.labels[i]  # Label can be processed later as needed
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
                    # For 'rgb' output, read the video (avi) file.
                    cap = cv2.VideoCapture(rgb_fp)
                    rgb_frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        rgb_frames.append(frame)
                    cap.release()
                    X = np.array(rgb_frames)

                # For non-flow outputs, resize and normalize the frames.
                if self.output != 'flow':
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
                if self.augmentation is not None and self.output in ['rgb', 'rgbd']:
                    # Process each frame individually.
                    augmented_frames = []
                    for j in range(X.shape[0]):
                        # For rgbd, split rgb and depth if necessary.
                        if self.output == 'rgbd':
                            # Augment only the rgb channels (first three), then reattach the depth channel.
                            rgb_frame = X[j, :3, :, :]
                            # Convert to CHW numpy array to HWC for PIL.
                            rgb_img = np.transpose(rgb_frame, (1, 2, 0)).astype(np.uint8)
                            aug_rgb = self.augmentation(rgb_img)
                            aug_rgb = aug_rgb.numpy()
                            # Use the unaugmented depth (fourth channel).
                            depth_channel = X[j, 3:, :, :]
                            frame_aug = np.concatenate([aug_rgb, depth_channel], axis=0)
                        else:
                            # For rgb, simply augment.
                            frame = np.transpose(X[j], (1, 2, 0)).astype(np.uint8)
                            frame_aug = self.augmentation(frame).numpy()
                        augmented_frames.append(frame_aug)
                    X = np.stack(augmented_frames, axis=1)  # (C, T, H, W)
                else:
                    # Transpose X to (C, T, H, W)
                    X = np.transpose(X, (3, 0, 1, 2))

                if self.cache_folder:
                    np.save(cache_file, X)

            X_tensor = torch.FloatTensor(X)
            # In this example the label is kept as a string.
            yield X_tensor, label

if __name__ == "__main__":
    # Example usage:
    dataset_path = "path/to/visl2_dataset"  # change this to your dataset root folder
    cache_folder = "path/to/cache_folder"    # change this to your preferred cache folder (or None)
    visl2_dataset = Visl2Dataset(dataset_path, apply_aug=True, cache_folder=cache_folder, output='rgb')
    for data, label in visl2_dataset:
        print("Data shape:", data.shape, "Label:", label)
        break