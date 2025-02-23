import numpy as np
import torch
import random
from torchvision import transforms
from typing import Optional, List, Dict, Any, Tuple, Union

class SkeletonAugmentation:
    def __init__(self):
        # Rotation parameters
        self.rotation_angle = 0.0  # [-13, 13] degrees
        self.center = None  # Will be computed during shuffle

        # Squeeze parameters
        self.left_squeeze = 0.0   # [0, 0.15]
        self.right_squeeze = 0.0  # [0, 0.15]

        # Perspective parameters
        self.perspective_side = 'left'  # or 'right'
        self.perspective_ratio = 0.0    # [0, 1]
        self.perspective_matrix = None

        # Sequential rotation parameters
        self.joint_rotations = []  # List of (joint_idx, angle) tuples
        
        # Define arm joint indices (example - adjust based on your skeleton structure)
        self.left_arm_indices = [5, 6, 7, 8]  # Left shoulder to hand
        self.right_arm_indices = [9, 10, 11, 12]  # Right shoulder to hand
        self.arm_indices = self.left_arm_indices + self.right_arm_indices

    def shuffle(self, frame_size: Tuple[int, int]) -> None:
        """Shuffle augmentation parameters for next sequence"""
        H, W = frame_size
        self.center = (W/2, H/2)

        # In-plane rotation
        self.rotation_angle = random.uniform(-13, 13)

        # Squeeze
        self.left_squeeze = random.uniform(0, 0.15)
        self.right_squeeze = random.uniform(0, 0.15)

        # Perspective
        self.perspective_side = random.choice(['left', 'right'])
        self.perspective_ratio = random.uniform(0, 1)
        self._compute_perspective_matrix(frame_size)

        # Sequential joint rotation
        self.joint_rotations = []
        for joint_idx in self.arm_indices:
            if random.random() < 0.3:  # 3:10 chance
                angle = random.uniform(-4, 4)
                self.joint_rotations.append((joint_idx, angle))

    def _compute_perspective_matrix(self, frame_size: Tuple[int, int]) -> None:
        """Compute perspective transform matrix"""
        H, W = frame_size
        ratio = self.perspective_ratio
        
        if self.perspective_side == 'right':
            src_points = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            dst_points = np.float32([[0, 0], 
                                   [W * (1-ratio), ratio*H], 
                                   [W * (1-ratio), H-(ratio*H)], 
                                   [0, H]])
        else:  # left
            src_points = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            dst_points = np.float32([[W*ratio, ratio*H], 
                                   [W, 0], 
                                   [W, H], 
                                   [W*ratio, H-(ratio*H)]])
            
        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def _rotate_point(self, point: np.ndarray, angle: float, center: Tuple[float, float]) -> np.ndarray:
        """Rotate a point around a center by given angle in degrees"""
        angle_rad = np.radians(angle)
        cx, cy = center
        x, y = point
        
        x_shifted, y_shifted = x - cx, y - cy
        
        x_rotated = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
        y_rotated = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
        
        return np.array([x_rotated + cx, y_rotated + cy])

    def apply(self, skeleton: np.ndarray) -> np.ndarray:
        """Apply augmentations to skeleton data
        Args:
            skeleton: numpy array of shape (n_joints, 2) or (n_joints, 3)
        Returns:
            augmented skeleton of same shape
        """
        augmented = skeleton.copy()
        n_joints = skeleton.shape[0]
        
        # 1. Apply in-plane rotation
        for i in range(n_joints):
            if skeleton.shape[1] >= 2:  # Only rotate x,y coordinates
                augmented[i, :2] = self._rotate_point(
                    skeleton[i, :2], 
                    self.rotation_angle, 
                    self.center
                )

        # 2. Apply squeeze
        width = self.center[0] * 2  # Assuming center is middle of frame
        left_cut = width * self.left_squeeze
        right_cut = width * self.right_squeeze
        
        # Adjust x-coordinates
        mask = augmented[:, 0] < self.center[0]  # Left side points
        augmented[mask, 0] = (augmented[mask, 0] - left_cut) * (width / (width - left_cut - right_cut))
        mask = augmented[:, 0] >= self.center[0]  # Right side points
        augmented[mask, 0] = augmented[mask, 0] * (width / (width - left_cut - right_cut))

        # 3. Apply perspective transformation
        if self.perspective_matrix is not None:
            points = augmented[:, :2].reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(points.astype(np.float32), 
                                                        self.perspective_matrix)
            augmented[:, :2] = transformed_points.reshape(-1, 2)

        # 4. Apply sequential joint rotation
        for joint_idx, angle in self.joint_rotations:
            if joint_idx < n_joints - 1:  # Ensure we have next joint
                current = augmented[joint_idx, :2]
                next_joint = augmented[joint_idx + 1, :2]
                
                # Rotate next joint around current joint
                augmented[joint_idx + 1, :2] = self._rotate_point(
                    next_joint, 
                    angle, 
                    tuple(current)
                )

        return augmented

class RGBDAugmentation:
    def __init__(self, aug_config: Optional[Dict] = None, output_type: str = 'rgb'):
        self.output_type = output_type
        self.aug_config = aug_config
        
        if output_type == 'skeleton':
            self.skeleton_augmentor = SkeletonAugmentation()
        else:
            # Initialize RGB/RGBD augmentation parameters similar to previous implementation
            self.aug_params = {}
        
            # Initialize default parameters
            if aug_config is None:
                if output_type in ['rgb', 'rgbd']:
                    self.aug_params = {
                        'flip_h': False,
                        'flip_v': False,
                        'brightness': 0.0,
                        'contrast': 0.0,
                        'saturation': 0.0,
                        'hue': 0.0,
                        'crop_params': None,
                        'mean': [0.485, 0.456, 0.406] if output_type == 'rgb' else [0.5] * 4,
                        'std': [0.229, 0.224, 0.225] if output_type == 'rgb' else [0.25] * 4
                    }
                elif output_type == 'skeleton':
                    self.aug_params = {
                        'rotation': 0.0,
                        'scale': 1.0,
                        'shift_x': 0.0,
                        'shift_y': 0.0,
                        'flip_h': False
                    }
            else:
                self._parse_config(aug_config)

    def _parse_config(self, aug_config: List[Dict[str, Any]]) -> None:
        """Parse augmentation configuration and store parameters."""
        self.aug_params = {}
        
        for aug in aug_config:
            if aug['type'] == 'random_crop':
                self.aug_params['crop_size'] = tuple(aug['size'])
                self.aug_params['crop_scale'] = aug.get('scale', (0.8, 1.0))
            elif aug['type'] == 'random_flip':
                self.aug_params['flip_h_prob'] = 0.5 if aug.get('horizontal', False) else 0.0
                self.aug_params['flip_v_prob'] = 0.5 if aug.get('vertical', False) else 0.0
            elif aug['type'] == 'color_jitter':
                self.aug_params['brightness'] = aug.get('brightness', 0.2)
                self.aug_params['contrast'] = aug.get('contrast', 0.2)
                self.aug_params['saturation'] = aug.get('saturation', 0.2)
                self.aug_params['hue'] = aug.get('hue', 0.1)
            elif aug['type'] == 'normalize':
                self.aug_params['mean'] = aug.get('mean', [0.485, 0.456, 0.406] if self.output_type == 'rgb' else [0.5] * 4)
                self.aug_params['std'] = aug.get('std', [0.229, 0.224, 0.225] if self.output_type == 'rgb' else [0.25] * 4)

    def shuffle(self, frame_size: Tuple[int, int]) -> None:
        """Shuffle augmentation parameters for next sequence"""
        if self.output_type == 'skeleton':
            self.skeleton_augmentor.shuffle(frame_size)
        else:
            # Shuffle RGB/RGBD parameters
            if self.output_type in ['rgb', 'rgbd']:
                self.aug_params['flip_h'] = random.random() < self.aug_params.get('flip_h_prob', 0.5)
                self.aug_params['flip_v'] = random.random() < self.aug_params.get('flip_v_prob', 0.5)
                self.aug_params['brightness'] = random.uniform(-self.aug_params.get('brightness', 0.2),
                                                             self.aug_params.get('brightness', 0.2))
                self.aug_params['contrast'] = random.uniform(-self.aug_params.get('contrast', 0.2),
                                                           self.aug_params.get('contrast', 0.2))
                self.aug_params['saturation'] = random.uniform(-self.aug_params.get('saturation', 0.2),
                                                             self.aug_params.get('saturation', 0.2))
                self.aug_params['hue'] = random.uniform(-self.aug_params.get('hue', 0.1),
                                                       self.aug_params.get('hue', 0.1))
                
                if 'crop_size' in self.aug_params:
                    scale = self.aug_params['crop_scale']
                    size = self.aug_params['crop_size']
                    area = size[0] * size[1]
                    target_area = random.uniform(scale[0], scale[1]) * area
                    aspect_ratio = random.uniform(3/4, 4/3)
                    w = int(round(np.sqrt(target_area * aspect_ratio)))
                    h = int(round(np.sqrt(target_area / aspect_ratio)))
                    self.aug_params['crop_params'] = (h, w)
                    
            elif self.output_type == 'skeleton':
                self.aug_params['rotation'] = random.uniform(-30, 30)
                self.aug_params['scale'] = random.uniform(0.9, 1.1)
                self.aug_params['shift_x'] = random.uniform(-0.1, 0.1)
                self.aug_params['shift_y'] = random.uniform(-0.1, 0.1)
                self.aug_params['flip_h'] = random.random() < 0.5

    def apply(self, frame: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply consistent augmentation to a single frame"""
        if self.output_type == 'skeleton':
            return self.skeleton_augmentor.apply(frame)
        else:
            # Apply RGB/RGBD augmentations
            if self.output_type in ['rgb', 'rgbd']:
                # Convert to PIL Image
                if not isinstance(frame, torch.Tensor):
                    frame = transforms.ToPILImage()(frame)
                
                # Apply transformations
                if self.aug_params['flip_h']:
                    frame = transforms.functional.hflip(frame)
                if self.aug_params['flip_v']:
                    frame = transforms.functional.vflip(frame)
                
                frame = transforms.functional.adjust_brightness(frame, 1.0 + self.aug_params['brightness'])
                frame = transforms.functional.adjust_contrast(frame, 1.0 + self.aug_params['contrast'])
                frame = transforms.functional.adjust_saturation(frame, 1.0 + self.aug_params['saturation'])
                frame = transforms.functional.adjust_hue(frame, self.aug_params['hue'])
                
                if 'crop_params' in self.aug_params and self.aug_params['crop_params'] is not None:
                    h, w = self.aug_params['crop_params']
                    frame = transforms.functional.resize(frame, (h, w))
                
                # Convert to tensor and normalize
                frame = transforms.functional.to_tensor(frame)
                frame = transforms.functional.normalize(frame, 
                                                     self.aug_params['mean'],
                                                     self.aug_params['std'])
                
            elif self.output_type == 'skeleton':
                # Apply geometric transformations to skeleton data
                # Assuming frame shape is (num_joints, 2) or (num_joints, 3)
                frame = frame.copy()
                
                # Rotation
                theta = np.radians(self.aug_params['rotation'])
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s], [s, c]])
                frame[:, :2] = frame[:, :2] @ R.T
                
                # Scale
                frame[:, :2] *= self.aug_params['scale']
                
                # Shift
                frame[:, 0] += self.aug_params['shift_x']
                frame[:, 1] += self.aug_params['shift_y']
                
                # Horizontal flip
                if self.aug_params['flip_h']:
                    frame[:, 0] = -frame[:, 0]
            
            return frame

def get_augmentation_pipeline(aug_config=None, output_type='rgb'):
    """
    Returns an augmentation pipeline based on the provided config.
    If aug_config is None, a default pipeline is used.
    """
    if aug_config is None:
        # Default pipeline for rgb/rgbd outputs.
        if output_type in ['rgb', 'rgbd']:
            mean = [0.485, 0.456, 0.406] if output_type == 'rgb' else [0.5] * 4
            std = [0.229, 0.224, 0.225] if output_type == 'rgb' else [0.25] * 4
            pipeline = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        elif output_type == 'skeleton':
            #Shape of skeleton data is (n_frames, n_joints, 2)\\

            pass
        else:
            pipeline = None
        return pipeline

    # Else, build the pipeline based on the provided configuration.
    transform_list = []

    # Ensure image is a PIL image first.
    transform_list.append(transforms.ToPILImage())

    for aug in aug_config:
        if aug['type'] == 'random_crop':
            size = tuple(aug['size'])
            transform_list.append(transforms.RandomResizedCrop(size=size, scale=aug.get('scale', (0.8, 1.0))))
        elif aug['type'] == 'random_flip':
            if aug.get('horizontal', False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug.get('vertical', False):
                transform_list.append(transforms.RandomVerticalFlip())
        elif aug['type'] == 'color_jitter':
            transform_list.append(transforms.ColorJitter(
                brightness=aug.get('brightness', 0.2),
                contrast=aug.get('contrast', 0.2),
                saturation=aug.get('saturation', 0.2),
                hue=aug.get('hue', 0.1)
            ))
        elif aug['type'] == 'resize':
            size = tuple(aug['size'])
            transform_list.append(transforms.Resize(size))
        elif aug['type'] == 'normalize':
            if output_type in ['rgb', 'rgbd']:
                mean = aug.get('mean', [0.485, 0.456, 0.406]) if output_type == 'rgb' else aug.get('mean', [0.5] * 4)
                std = aug.get('std', [0.229, 0.224, 0.225]) if output_type == 'rgb' else aug.get('std', [0.25] * 4)
                transform_list.append(transforms.ToTensor())
                transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)

def apply_augmentation(image, aug_config=None, output_type='rgb'):
    pipeline = get_augmentation_pipeline(aug_config, output_type)
    return pipeline(image) if pipeline is not None else image