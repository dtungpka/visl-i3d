import os
import yaml
import sys
import torch
import gc
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from src.datasets import DatasetRegistry
from src.datasets.visl2_dataset import Visl2Dataset

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config_path = "src/config/visl2_spoter.yaml"
    config = load_config(config_path)
    
    # Set up dataset with empty skeletons for workers
    dataset_config = config['dataset'].copy()
    dataset_config['num_workers'] = 2
    dataset_config['cache_folder'] = None  # Disable caching for test
    
    print("Available datasets:", DatasetRegistry.list_datasets())
    
    # Register dataset manually if needed
    if 'visl2' not in DatasetRegistry.list_datasets():
        DatasetRegistry.register('visl2', Visl2Dataset)
    
    val_dataset = DatasetRegistry.get_dataset('visl2', dataset_config, mode='val')
    print(f"Created dataset with {len(val_dataset)} samples")
    
    # Create a dataloader with worker_init_fn to initialize each worker
    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=getattr(val_dataset, 'collate_fn', None),
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Just load one batch to test
    print("Loading first batch...")
    start = time.time()
    batch = next(iter(dataloader))
    elapsed = time.time() - start
    
    print(f"Loaded batch in {elapsed:.2f} seconds")
    print(f"Batch shape: {batch[0].shape}, labels: {batch[1]}")
    
    # Try loading a few more batches
    print("\nLoading a few more batches...")
    for i, (inputs, labels) in enumerate(dataloader):
        print(f"Batch {i+1}: shape={inputs.shape}, labels={labels}")
        if i >= 2:  # Just test 3 batches total
            break

if __name__ == "__main__":
    main()