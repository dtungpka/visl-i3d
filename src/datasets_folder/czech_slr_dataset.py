import os
import math
import random

import torch
import numpy as np
import pandas as pd

from torch.utils.data import IterableDataset
from typing import Optional

# For your existing augmentation functions:
# from augmentations import augment_rotate, augment_shear, augment_arm_joint_rotate
# from normalization.body_normalization import BODY_IDENTIFIERS, normalize_single_dict as normalize_single_body_dict
# from normalization.hand_normalization import HAND_IDENTIFIERS, normalize_single_dict as normalize_single_hand_dict
#
# Adjust import paths accordingly to match your local project.

# Example placeholders for your original augmentation and normalization.
# Replace these with your real versions.
def augment_rotate(landmarks_dict, angle_range):
    # Your real implementation here.
    return landmarks_dict

def augment_shear(landmarks_dict, shear_type, shear_range):
    # Your real implementation here.
    return landmarks_dict

def augment_arm_joint_rotate(landmarks_dict, prob, angle_range):
    # Your real implementation here.
    return landmarks_dict

BODY_IDENTIFIERS = [
    "leftEar", "rightEar", "leftEye", "rightEye",  # etc.
    # Fill this out with the actual list of body joints you use
]
HAND_IDENTIFIERS = [
    "handLeft", "handRight",
    # etc. Fill out with the actual list for each hand
]

def normalize_single_body_dict(landmarks_dict):
    # Your actual normalization
    return landmarks_dict

def normalize_single_hand_dict(landmarks_dict):
    # Your actual normalization
    return landmarks_dict

################################################################################
# Utility functions for your CSV loading and converting
################################################################################

def tensor_to_dictionary(landmarks_tensor: torch.Tensor) -> dict:
    """
    Convert a (T, J, 2) shape tensor into a dictionary mapping
    each joint name to an array of shape (T, 2).
    """
    data_array = landmarks_tensor.numpy()
    output = {}
    all_idents = BODY_IDENTIFIERS + HAND_IDENTIFIERS

    for landmark_index, identifier in enumerate(all_idents):
        output[identifier] = data_array[:, landmark_index]  # shape (T, 2)

    return output

def dictionary_to_tensor(landmarks_dict: dict) -> torch.Tensor:
    """
    Convert a dictionary of joint_name -> (T, 2) arrays into
    a (T, J, 2) tensor.
    """
    all_idents = BODY_IDENTIFIERS + HAND_IDENTIFIERS
    # Assume they all have the same sequence length T
    T = len(landmarks_dict[all_idents[0]])
    J = len(all_idents)
    output = np.empty(shape=(T, J, 2), dtype=np.float32)

    for landmark_index, identifier in enumerate(all_idents):
        output[:, landmark_index, :] = landmarks_dict[identifier]

    return torch.from_numpy(output)

def load_csv_data(csv_path: str):
    """
    Reads your CSV file and returns:
       data: list of shape (T, J, 2) as numpy arrays
       labels: list of integer label IDs
    """
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Example fix-ups so columns match "BODY_IDENTIFIERS/HAND_IDENTIFIERS"
    # You should adapt to your real logic:
    df.columns = [col.replace("_left_", "_0_").replace("_right_", "_1_") for col in df.columns]
    if "neck_X" not in df.columns:
        df["neck_X"] = [0 for _ in range(df.shape[0])]
        df["neck_Y"] = [0 for _ in range(df.shape[0])]

    labels = df["labels"].to_list()

    # Build list of (T, J, 2) data
    data = []
    all_idents = BODY_IDENTIFIERS + HAND_IDENTIFIERS

    # Each row is one sample: a variable-length sequence of landmarks
    for _, row in df.iterrows():
        # Suppose "leftEar_X" is a string of float values: "[0.1, 0.2, ...]"
        # Then ast.literal_eval is used to parse it; adapt as needed
        # (be sure to import ast if you have not already).
        sequence_length = len(eval(row[f"{all_idents[0]}_X"]))
        sample_np = np.zeros((sequence_length, len(all_idents), 2), dtype=np.float32)

        for j, ident in enumerate(all_idents):
            x_list = eval(row[f"{ident}_X"])
            y_list = eval(row[f"{ident}_Y"])
            # Fill T, 2
            for t in range(sequence_length):
                sample_np[t, j, 0] = x_list[t]
                sample_np[t, j, 1] = y_list[t]

        data.append(sample_np)

    return data, labels

################################################################################
# The new CzechSlrDataset class in a visl2-like structure
################################################################################

class CzechSlrDataset(IterableDataset):
    """
    A Czech SLR dataset in a style similar to Visl2Dataset:
       - It uses IterableDataset
       - Batches are yielded on the fly from __iter__
       - Accepts a config dictionary describing paths, batch size, etc.
    """

    def __init__(self, config: dict, mode: str = "train"):
        """
        Expects config to have structure like:
          {
            "paths": {
              "train_data_path": "path/to/train.csv",
              "val_data_path":   "path/to/val.csv",
              "test_data_path":  "path/to/test.csv"
            },
            "batch_size": 16,
            "num_frames": 64,
            "num_classes": 10,
            "augmentation": {
               "use_augmentation": True,
               ... # your augment params
            },
            "person_selection": { "train": {...}, ... } # optional
          }
        """
        super().__init__()

        self.config = config
        self.mode = mode

        # Pull out relevant items from config
        self.csv_path = self.config["paths"][f"{mode}_data_path"]
        self.batch_size = self.config.get("batch_size", 16)
        self.n_frames = self.config.get("num_frames", 64)  # If you'd like
        self.num_classes = self.config.get("num_classes", 5)  # Example
        self.use_augmentation = self.config.get("augmentation", {}).get("use_augmentation", False)
        self.augment_prob = 0.5  # Or place in config
        self.normalize = True     # Or place in config

        # Load entire dataset into memory
        self.all_data, self.all_labels = load_csv_data(self.csv_path)

        # Print summary
        print(f"[CzechSlrDataset] Loaded {len(self.all_data)} samples from {self.csv_path} (mode={mode}).")

        # If you want a basic collate_fn so that you can do
        # DataLoader(dataset, collate_fn=dataset.collate_fn), define one:
        self.collate_fn = None  # For an IterableDataset, usually not used.

    def __iter__(self):
        """
        The IterableDataset entry point. Splits data across workers if using
        num_workers > 0 in your DataLoader, then yields mini-batches.
        """
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.all_labels)
        if worker_info is not None:
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            ws = start + worker_id * per_worker
            we = min(ws + per_worker, end)
        else:
            ws, we = start, end

        yield from self._get_data(ws, we)

    def _get_data(self, start: int, end: int):
        """
        Generator that yields mini-batches from all_data[start:end].
        """
        batch_data = []
        batch_labels = []

        for i in range(start, end):
            item_data = self.all_data[i]   # shape (T, J, 2)
            item_label = self.all_labels[i]

            # Convert item_data to torch.Tensor
            item_tensor = torch.from_numpy(item_data)  # shape (T, J, 2)
            # Convert to dict for augmentation:
            landmarks_dict = tensor_to_dictionary(item_tensor)

            # Possibly do random augmentation
            if self.use_augmentation and random.random() < self.augment_prob:
                # Example: choose one of a few augmentations
                selected_aug = random.randrange(4)
                if selected_aug == 0:
                    landmarks_dict = augment_rotate(landmarks_dict, (-13, 13))
                elif selected_aug == 1:
                    landmarks_dict = augment_shear(landmarks_dict, "perspective", (0, 0.1))
                elif selected_aug == 2:
                    landmarks_dict = augment_shear(landmarks_dict, "squeeze", (0, 0.15))
                elif selected_aug == 3:
                    landmarks_dict = augment_arm_joint_rotate(landmarks_dict, 0.3, (-4, 4))

            # Possibly normalize
            if self.normalize:
                landmarks_dict = normalize_single_body_dict(landmarks_dict)
                landmarks_dict = normalize_single_hand_dict(landmarks_dict)

            # Convert back to torch
            item_tensor = dictionary_to_tensor(landmarks_dict)
            # Shift the range if you like:
            # item_tensor = item_tensor - 0.5

            # If you want to handle "num_frames" / "n_frames" logic:
            seq_len = item_tensor.shape[0]
            if seq_len > self.n_frames:
                # Truncate
                item_tensor = item_tensor[: self.n_frames]
            elif seq_len < self.n_frames:
                # Pad by repeating or zeros
                pad_len = self.n_frames - seq_len
                pad_zeros = torch.zeros((pad_len, item_tensor.shape[1], item_tensor.shape[2]))
                item_tensor = torch.cat([item_tensor, pad_zeros], dim=0)

            # Convert label to one-hot if your pipeline expects that.
            # Otherwise store it as int. E.g.:
            label_index = int(item_label - 1)  # if CSV labels start at 1
            label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
            if 0 <= label_index < self.num_classes:
                label_tensor[label_index] = 1.0

            # Accumulate
            batch_data.append(item_tensor)
            batch_labels.append(label_tensor)

            # Once we reach batch_size, yield
            if len(batch_data) == self.batch_size:
                yield torch.stack(batch_data, dim=0), torch.stack(batch_labels, dim=0)
                batch_data = []
                batch_labels = []

        # Yield any leftover
        if batch_data:
            yield torch.stack(batch_data, dim=0), torch.stack(batch_labels, dim=0)
            batch_data = []
            batch_labels = []

if __name__ == "__main__":
    example_config = {
        "paths": {
            "train_data_path": "path/to/czech_train.csv",
            "val_data_path": "path/to/czech_val.csv",
            "test_data_path": "path/to/czech_test.csv",
        },
        "batch_size": 4,
        "num_frames": 32,
        "num_classes": 5,
        "augmentation": {
            "use_augmentation": True
        },
        # person_selection if needed, or anything else your pipeline uses
    }

    dataset = CzechSlrDataset(example_config, mode="train")
    for batch_X, batch_y in dataset:
        print("Batch shapes:", batch_X.shape, batch_y.shape)
        break
