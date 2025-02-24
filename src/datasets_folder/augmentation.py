import numpy as np
import torch
import random
import cv2
from torchvision import transforms
from typing import Optional, List, Dict, Any, Tuple, Union
from tqdm import tqdm
import math

# MARK: VISL

class BaseAugmentation:
    """Base class for all augmentations"""
    def __init__(self, aug_config: Optional[Dict] = None):
        self.aug_params = {}
        self.temporal_params = {}
        
        # Default temporal augmentation parameters
        self.temporal_params = {
            'frame_skip_range': (1, 1),  # No skip by default
            'frame_duplicate_prob': 0.0,  # No duplication by default
            'temporal_crop_scale': (0.8, 1.0),  # Temporal crop range
            'min_frames': None  # Minimum frames required
        }
        
        if aug_config:
            self._parse_temporal_config(aug_config.get('temporal', {}))
    
    def _parse_temporal_config(self, temporal_config: Dict) -> None:
        """Parse temporal augmentation configuration"""
        self.temporal_params.update({
            'frame_skip_range': temporal_config.get('frame_skip_range', (1, 1)),
            'frame_duplicate_prob': temporal_config.get('frame_duplicate_prob', 0.0),
            'temporal_crop_scale': temporal_config.get('temporal_crop_scale', (0.8, 1.0)),
            'min_frames': temporal_config.get('min_frames', None)
        })
    
    def _apply_temporal_augmentation(self, frames: np.ndarray) -> np.ndarray:
        """Apply temporal augmentations to frames"""
        n_frames = len(frames)
        
        # 1. Temporal cropping
        min_frames = self.temporal_params['min_frames'] or n_frames
        scale = self.temporal_params['temporal_crop_scale']
        target_frames = int(random.uniform(scale[0], scale[1]) * n_frames)
        target_frames = max(min_frames, target_frames)
        
        if target_frames < n_frames:
            start_idx = random.randint(0, n_frames - target_frames)
            frames = frames[start_idx:start_idx + target_frames]
        
        # 2. Frame skipping
        skip_range = self.temporal_params['frame_skip_range']
        if skip_range[1] > 1:
            skip_rate = random.randint(skip_range[0], skip_range[1])
            frames = frames[::skip_rate]
        
        # 3. Frame duplication
        dup_prob = self.temporal_params['frame_duplicate_prob']
        if dup_prob > 0:
            duplicated_frames = []
            for frame in frames:
                duplicated_frames.append(frame)
                if random.random() < dup_prob:
                    duplicated_frames.append(frame)  # Duplicate frame
            frames = np.array(duplicated_frames)
        
        return frames

class SkeletonAugmentation(BaseAugmentation):
    def __init__(self, aug_config: Optional[Dict] = None):
        super().__init__(aug_config)
        
        # Skeleton-specific parameters
        self.left_arm_indices = [5, 6, 7, 8]
        self.right_arm_indices = [9, 10, 11, 12]
        self.arm_indices = self.left_arm_indices + self.right_arm_indices
        
        # Default spatial parameters
        self.aug_params = {
            'rotation_range': (-13, 13),
            'squeeze_range': (0, 0.15),
            'perspective_ratio_range': (0, 1),
            'joint_rotation_prob': 0.3,
            'joint_rotation_range': (-4, 4)
        }
        
        # Will hold computed matrix if any
        self.perspective_matrix = None
        self.rotation_angle = 0.0
        self.left_squeeze = 0.0
        self.right_squeeze = 0.0
        self.perspective_side = 'left'
        self.perspective_ratio = 0.0
        self.center = (0.0, 0.0)
        self.joint_rotations = []
        
        if aug_config:
            self._parse_spatial_config(aug_config.get('spatial', {}))
    
    def _parse_spatial_config(self, spatial_config: Dict) -> None:
        """Parse spatial augmentation configuration"""
        self.aug_params.update({
            'rotation_range': spatial_config.get('rotation_range', (-13, 13)),
            'squeeze_range': spatial_config.get('squeeze_range', (0, 0.15)),
            'perspective_ratio_range': spatial_config.get('perspective_ratio_range', (0, 1)),
            'joint_rotation_prob': spatial_config.get('joint_rotation_prob', 0.3),
            'joint_rotation_range': spatial_config.get('joint_rotation_range', (-4, 4))
        })
    
    def shuffle(self, frame_size: Tuple[int, int]) -> None:
        """Shuffle augmentation parameters once per sequence"""
        H, W = frame_size
        self.center = (W/2, H/2)
        
        # Spatial parameters
        self.rotation_angle = random.uniform(*self.aug_params['rotation_range'])
        self.left_squeeze = random.uniform(*self.aug_params['squeeze_range'])
        self.right_squeeze = random.uniform(*self.aug_params['squeeze_range'])
        self.perspective_side = random.choice(['left', 'right'])
        self.perspective_ratio = random.uniform(*self.aug_params['perspective_ratio_range'])
        
        self._compute_perspective_matrix(frame_size)
        
        # Joint rotations
        self.joint_rotations = [
            (joint_idx, random.uniform(*self.aug_params['joint_rotation_range']))
            for joint_idx in self.arm_indices
            if random.random() < self.aug_params['joint_rotation_prob']
        ]
    
    def _compute_perspective_matrix(self, frame_size: Tuple[int, int]) -> None:
        """Compute perspective transform matrix based on side & ratio"""
        H, W = frame_size
        ratio = self.perspective_ratio
        
        if ratio <= 0.0:
            self.perspective_matrix = None
            return
        
        src_points = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        if self.perspective_side == 'right':
            dst_points = np.float32([
                [0, 0],
                [W*(1-ratio), ratio*H],
                [W*(1-ratio), H-(ratio*H)],
                [0, H]
            ])
        else:  # left
            dst_points = np.float32([
                [W*ratio, ratio*H],
                [W, 0],
                [W, H],
                [W*ratio, H-(ratio*H)]
            ])
        
        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    def apply(self, frames: np.ndarray) -> np.ndarray:
        """Apply augmentations to multiple skeleton frames"""
        # Apply temporal augmentations first
        frames = self._apply_temporal_augmentation(frames)
        
        # Apply spatial augmentations to each frame
        augmented_frames = []
        pbar = tqdm(frames, desc='Applying SA', leave=False) if len(frames) > 100 else frames
        for skeleton in pbar:
            aug_skeleton = self._apply_spatial(skeleton)
            augmented_frames.append(aug_skeleton)
        
        return np.stack(augmented_frames)
    
    def _apply_spatial(self, skeleton: np.ndarray) -> np.ndarray:
        """Apply rotation, squeeze, perspective, and joint rotation to a single skeleton"""
        aug = skeleton.copy()
        n_joints = aug.shape[0]
        
        # 1. In-plane rotation around self.center
        for i in range(n_joints):
            aug[i, :2] = self._rotate_point(aug[i, :2], self.rotation_angle, self.center)
        
        # 2. Squeeze (left & right)
        width = self.center[0] * 2.0
        # Left side
        mask_left = aug[:, 0] < self.center[0]
        aug[mask_left, 0] = aug[mask_left, 0] - (width * self.left_squeeze)
        # Right side
        mask_right = aug[:, 0] >= self.center[0]
        aug[mask_right, 0] = aug[mask_right, 0] * (1.0 - self.right_squeeze)
        
        # 3. Perspective transform
        if self.perspective_matrix is not None:
            pts = aug[:, :2].reshape(-1, 1, 2).astype(np.float32)
            warped = cv2.perspectiveTransform(pts, self.perspective_matrix)
            aug[:, :2] = warped.reshape(-1, 2)
        
        # 4. Sequential joint rotations
        for (joint_idx, angle) in self.joint_rotations:
            if joint_idx < n_joints - 1:
                pivot = aug[joint_idx, :2]
                aug[joint_idx + 1, :2] = self._rotate_point(aug[joint_idx + 1, :2],
                                                            angle, pivot)
        
        return aug
    
    def _rotate_point(self, point: np.ndarray, angle_deg: float, center_pt: Tuple[float, float]) -> np.ndarray:
        """Rotate a 2D point around center_pt by angle_deg"""
        angle_rad = np.radians(angle_deg)
        cx, cy = center_pt
        x, y = point

        # Translate to origin
        x_shift, y_shift = x - cx, y - cy
        # Rotate
        x_rot = x_shift * np.cos(angle_rad) - y_shift * np.sin(angle_rad)
        y_rot = x_shift * np.sin(angle_rad) + y_shift * np.cos(angle_rad)
        # Translate back
        return np.array([x_rot + cx, y_rot + cy])

class RGBDAugmentation(BaseAugmentation):
    def __init__(self, aug_config: Optional[Dict] = None, output_type: str = 'rgb'
                 ,desired_output_size: Tuple[int, int] = (224, 224)):
        super().__init__(aug_config)
        self.output_type = output_type
        
        # Default spatial parameters
        self.aug_params = {
            'flip_h_prob': 0.5,
            'flip_v_prob': 0.0,
            'brightness_range': (-0.2, 0.2),
            'contrast_range': (-0.2, 0.2),
            'saturation_range': (-0.2, 0.2),
            'hue_range': (-0.1, 0.1),
            'spatial_crop_scale': (0.8, 1.0),
            'mean': [0.485, 0.456, 0.406] if output_type == 'rgb' else [0.5] * 4,
            'std': [0.229, 0.224, 0.225] if output_type == 'rgb' else [0.25] * 4
        }
        self.desired_output_size = desired_output_size
        
        if aug_config:
            self._parse_spatial_config(aug_config.get('spatial', {}))
    
    def _parse_spatial_config(self, spatial_config: Dict) -> None:
        """Parse spatial augmentation configuration"""
        if 'normalize' in spatial_config:
            self.aug_params['mean'] = spatial_config['normalize'].get('mean', self.aug_params['mean'])
            self.aug_params['std'] = spatial_config['normalize'].get('std', self.aug_params['std'])
        
        self.aug_params.update({
            'flip_h_prob': spatial_config.get('flip_h_prob', 0.5),
            'flip_v_prob': spatial_config.get('flip_v_prob', 0.0),
            'brightness_range': spatial_config.get('brightness_range', (-0.2, 0.2)),
            'contrast_range': spatial_config.get('contrast_range', (-0.2, 0.2)),
            'saturation_range': spatial_config.get('saturation_range', (-0.2, 0.2)),
            'hue_range': spatial_config.get('hue_range', (-0.1, 0.1)),
            'spatial_crop_scale': spatial_config.get('spatial_crop_scale', (0.8, 1.0))
        })
    
    def shuffle(self, frame_size: Tuple[int, int]) -> None:
        """Shuffle augmentation parameters"""
        self.current_params = {
            'flip_h': random.random() < self.aug_params['flip_h_prob'],
            'flip_v': random.random() < self.aug_params['flip_v_prob'],
            'brightness': random.uniform(*self.aug_params['brightness_range']),
            'contrast': random.uniform(*self.aug_params['contrast_range']),
            'saturation': random.uniform(*self.aug_params['saturation_range']),
            'hue': random.uniform(*self.aug_params['hue_range'])
        }
        
        # Spatial crop parameters
        H, W = frame_size
        scale = self.aug_params['spatial_crop_scale']
        target_area = random.uniform(scale[0], scale[1]) * H * W
        aspect_ratio = random.uniform(3/4, 4/3)
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        self.current_params['crop'] = (h, w)
    
    def apply(self, frames: np.ndarray) -> np.ndarray:
        """Apply augmentations to multiple frames"""
        # Apply temporal augmentations first
        frames = self._apply_temporal_augmentation(frames)
        
        # Apply spatial augmentations to each frame
        augmented_frames = []
        pbar = tqdm(frames, desc='Applying SA', leave=False) if len(frames) > 100 else frames
        for frame in pbar:
            aug_frame = self._apply_spatial(frame)
            augmented_frames.append(aug_frame)
        
        return np.stack(augmented_frames)
    
    def _apply_spatial(self, frame: np.ndarray) -> np.ndarray:
        """Apply spatial augmentations to a single frame"""
        # Convert to PIL Image if needed
        if not isinstance(frame, torch.Tensor):
            #norm to [0 255] and convert to uint8
            frame = frame * 255
            frame = frame.astype(np.uint8)
            frame = transforms.ToPILImage()(frame)
        
        # Apply transformations
        if self.current_params['flip_h']:
            frame = transforms.functional.hflip(frame)
        if self.current_params['flip_v']:
            frame = transforms.functional.vflip(frame)
        
        frame = transforms.functional.adjust_brightness(frame, 1.0 + self.current_params['brightness'])
        frame = transforms.functional.adjust_contrast(frame, 1.0 + self.current_params['contrast'])
        frame = transforms.functional.adjust_saturation(frame, 1.0 + self.current_params['saturation'])
        frame = transforms.functional.adjust_hue(frame, self.current_params['hue'])
        
        # Apply spatial crop
        h, w = self.current_params['crop']
        frame = transforms.functional.resize(frame, (h, w))
        
        # Resize to desired output size
        frame = transforms.functional.resize(frame, self.desired_output_size)
        
        # Convert to tensor and normalize
        frame = transforms.functional.to_tensor(frame)
        frame = transforms.functional.normalize(frame, self.aug_params['mean'], self.aug_params['std'])
        
        # Convert back to numpy array
        frame = frame.numpy()
        #norm to [0 1]
        frame = frame.astype(np.float32)
        
        return frame
    
    
    
    
    #MARK: CZECH AUGMENTATION
    

from src.utils_folder.body_normalization import BODY_IDENTIFIERS
from src.utils_folder.hand_normalization import HAND_IDENTIFIERS


HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]
ARM_IDENTIFIERS_ORDER = ["neck", "$side$Shoulder", "$side$Elbow", "$side$Wrist"]


def __random_pass(prob):
    return random.random() < prob


def __numpy_to_dictionary(data_array: np.ndarray) -> dict:
    """
    Supplementary method converting a NumPy array of body landmark data into dictionaries. The array data must match the
    order of the BODY_IDENTIFIERS list.
    """

    output = {}

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS):
        output[identifier] = data_array[:, landmark_index].tolist()

    return output


def __dictionary_to_numpy(landmarks_dict: dict) -> np.ndarray:
    """
    Supplementary method converting dictionaries of body landmark data into respective NumPy arrays. The resulting array
    will match the order of the BODY_IDENTIFIERS list.
    """

    output = np.empty(shape=(len(landmarks_dict["leftEar"]), len(BODY_IDENTIFIERS), 2))

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS):
        output[:, landmark_index, 0] = np.array(landmarks_dict[identifier])[:, 0]
        output[:, landmark_index, 1] = np.array(landmarks_dict[identifier])[:, 1]

    return output


def __rotate(origin: tuple, point: tuple, angle: float):
    """
    Rotates a point counterclockwise by a given angle around a given origin.

    :param origin: Landmark in the (X, Y) format of the origin from which to count angle of rotation
    :param point: Landmark in the (X, Y) format to be rotated
    :param angle: Angle under which the point shall be rotated
    :return: New landmarks (coordinates)
    """

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy


def __preprocess_row_sign(sign: dict):
    """
    Supplementary method splitting the single-dictionary skeletal data into two dictionaries of body and hand landmarks
    respectively.
    """

    sign_eval = sign

    if "nose_X" in sign_eval:
        body_landmarks = {identifier: [(x, y) for x, y in zip(sign_eval[identifier + "_X"], sign_eval[identifier + "_Y"])]
                          for identifier in BODY_IDENTIFIERS}
        hand_landmarks = {identifier: [(x, y) for x, y in zip(sign_eval[identifier + "_X"], sign_eval[identifier + "_Y"])]
                          for identifier in HAND_IDENTIFIERS}

    else:
        body_landmarks = {identifier: sign_eval[identifier] for identifier in BODY_IDENTIFIERS}
        hand_landmarks = {identifier: sign_eval[identifier] for identifier in HAND_IDENTIFIERS}

    return body_landmarks, hand_landmarks


def __wrap_sign_into_row(body_identifiers: dict, hand_identifiers: dict) -> dict:
    """
    Supplementary method for merging body and hand data into a single dictionary.
    """

    return {**body_identifiers, **hand_identifiers}


def augment_rotate(sign: dict, angle_range: tuple) -> dict:
    """
    AUGMENTATION TECHNIQUE. All the joint coordinates in each frame are rotated by a random angle up to 13 degrees with
    the center of rotation lying in the center of the frame, which is equal to [0.5; 0.5].

    :param sign: Dictionary with sequential skeletal data of the signing person
    :param angle_range: Tuple containing the angle range (minimal and maximal angle in degrees) to randomly choose the
                        angle by which the landmarks will be rotated from

    :return: Dictionary with augmented (by rotation) sequential skeletal data of the signing person
    """

    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)
    angle = math.radians(random.uniform(*angle_range))

    body_landmarks = {key: [__rotate((0.5, 0.5), frame, angle) for frame in value] for key, value in
                      body_landmarks.items()}
    hand_landmarks = {key: [__rotate((0.5, 0.5), frame, angle) for frame in value] for key, value in
                      hand_landmarks.items()}

    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


def augment_shear(sign: dict, type: str, squeeze_ratio: tuple) -> dict:
    """
    AUGMENTATION TECHNIQUE.

        - Squeeze. All the frames are squeezed from both horizontal sides. Two different random proportions up to 15% of
        the original frame's width for both left and right side are cut.

        - Perspective transformation. The joint coordinates are projected onto a new plane with a spatially defined
        center of projection, which simulates recording the sign video with a slight tilt. Each time, the right or left
        side, as well as the proportion by which both the width and height will be reduced, are chosen randomly. This
        proportion is selected from a uniform distribution on the [0; 1) interval. Subsequently, the new plane is
        delineated by reducing the width at the desired side and the respective vertical edge (height) at both of its
        adjacent corners.

    :param sign: Dictionary with sequential skeletal data of the signing person
    :param type: Type of shear augmentation to perform (either 'squeeze' or 'perspective')
    :param squeeze_ratio: Tuple containing the relative range from what the proportion of the original width will be
                          randomly chosen. These proportions will either be cut from both sides or used to construct the
                          new projection

    :return: Dictionary with augmented (by squeezing or perspective transformation) sequential skeletal data of the
             signing person
    """

    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)

    if type == "squeeze":
        move_left = random.uniform(*squeeze_ratio)
        move_right = random.uniform(*squeeze_ratio)

        src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)
        dest = np.array(((0 + move_left, 1), (1 - move_right, 1), (0 + move_left, 0), (1 - move_right, 0)),
                        dtype=np.float32)
        mtx = cv2.getPerspectiveTransform(src, dest)

    elif type == "perspective":

        move_ratio = random.uniform(*squeeze_ratio)
        src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)

        if __random_pass(0.5):
            dest = np.array(((0 + move_ratio, 1 - move_ratio), (1, 1), (0 + move_ratio, 0 + move_ratio), (1, 0)),
                            dtype=np.float32)
        else:
            dest = np.array(((0, 1), (1 - move_ratio, 1 - move_ratio), (0, 0), (1 - move_ratio, 0 + move_ratio)),
                            dtype=np.float32)

        mtx = cv2.getPerspectiveTransform(src, dest)

    else:

        print("Unsupported shear type provided.")
        return {}

    landmarks_array = __dictionary_to_numpy(body_landmarks)
    augmented_landmarks = cv2.perspectiveTransform(np.array(landmarks_array, dtype=np.float32), mtx)

    augmented_zero_landmark = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), mtx)[0][0]
    augmented_landmarks = np.stack([np.where(sub == augmented_zero_landmark, [0, 0], sub) for sub in augmented_landmarks])

    body_landmarks = __numpy_to_dictionary(augmented_landmarks)

    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


def augment_arm_joint_rotate(sign: dict, probability: float, angle_range: tuple) -> dict:
    """
    AUGMENTATION TECHNIQUE. The joint coordinates of both arms are passed successively, and the impending landmark is
    slightly rotated with respect to the current one. The chance of each joint to be rotated is 3:10 and the angle of
    alternation is a uniform random angle up to +-4 degrees. This simulates slight, negligible variances in each
    execution of a sign, which do not change its semantic meaning.

    :param sign: Dictionary with sequential skeletal data of the signing person
    :param probability: Probability of each joint to be rotated (float from the range [0, 1])
    :param angle_range: Tuple containing the angle range (minimal and maximal angle in degrees) to randomly choose the
                        angle by which the landmarks will be rotated from

    :return: Dictionary with augmented (by arm joint rotation) sequential skeletal data of the signing person
    """

    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)

    # Iterate over both directions (both hands)
    for side in ["left", "right"]:
        # Iterate gradually over the landmarks on arm
        for landmark_index, landmark_origin in enumerate(ARM_IDENTIFIERS_ORDER):
            landmark_origin = landmark_origin.replace("$side$", side)

            # End the process on the current hand if the landmark is not present
            if landmark_origin not in body_landmarks:
                break

            # Perform rotation by provided probability
            if __random_pass(probability):
                angle = math.radians(random.uniform(*angle_range))

                for to_be_rotated in ARM_IDENTIFIERS_ORDER[landmark_index + 1:]:
                    to_be_rotated = to_be_rotated.replace("$side$", side)

                    # Skip if the landmark is not present
                    if to_be_rotated not in body_landmarks:
                        continue

                    body_landmarks[to_be_rotated] = [__rotate(body_landmarks[landmark_origin][frame_index], frame,
                        angle) for frame_index, frame in enumerate(body_landmarks[to_be_rotated])]

    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


