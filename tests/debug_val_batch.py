import os
import yaml
import sys
import torch
import gc
import numpy as np
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Set memory limit for workers
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from src.datasets import DatasetRegistry
from src.datasets.visl2_dataset import Visl2Dataset
from src.utils.skeleton_processor import MediaPipeProcessor

# Initialize MediaPipe processor upfront
mediapipe_processor = MediaPipeProcessor()

# Register dataset manually
DatasetRegistry.register('visl2', Visl2Dataset)

def print_queue_status(processor):
    """Print current status of the MediaPipe processor queue"""
    status = processor.get_queue_status()
    print("\nMediaPipe Queue Status:")
    print(f"Queue size: {status['queue_size']}")
    print(f"Active requests: {status['active_requests']}")
    print(f"Cached results: {status['cached_results']}")
    
    if status['in_progress']:
        print("\nIn Progress:")
        for req_id, progress in status['in_progress'].items():
            print(f"  Request {req_id}: {progress:.1%} complete")
    else:
        print("No requests in progress")
    print()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def test_single_sample(dataset, idx=0):
    """Test loading a single sample directly"""
    try:
        print(f"Trying to load sample {idx} directly...")
        start_time = time.time()
        sample = dataset[idx]
        elapsed = time.time() - start_time
        print(f"Successfully loaded sample {idx} in {elapsed:.2f} seconds")
        print(f"Data shape: {sample['data'].shape}")
        print(f"Label: {sample['label']}")
        return True
    except Exception as e:
        print(f"Error loading sample {idx}: {e}")
        return False

def test_val_batch(config_path):
    config = load_config(config_path)
    
    # Force garbage collection before starting
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Available datasets:", DatasetRegistry.list_datasets())
    
    # Create validation dataset with reduced workers
    dataset_name = config['dataset']['name']
    dataset_config = config['dataset'].copy()
    dataset_config['num_workers'] = 0  # Start with 0 workers
    dataset_config['batch_size'] = 1  # Small batch size
    
    try:
        # First disable caching to prevent corrupt cache files
        if 'cache_folder' in dataset_config:
            print("Temporarily disabling cache for debugging")
            original_cache = dataset_config['cache_folder']
            dataset_config['cache_folder'] = None
        
        # Set verbose mode for MediaPipe progress tracking
        dataset_config['verbose'] = True
        
        val_dataset = DatasetRegistry.get_dataset(dataset_name, dataset_config, mode='val')
        print(f"Created validation dataset with {len(val_dataset)} samples")
        
        # Test loading a few individual samples directly
        success = 0
        for i in range(min(5, len(val_dataset))):
            if test_single_sample(val_dataset, i):
                success += 1
        
        print(f"Successfully loaded {success}/5 individual samples")
        
        # Try loading with simple DataLoader first - ALWAYS use num_workers=0 for debug
        print("\nTesting with DataLoader(num_workers=0)...")
        dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Keep this at 0 for reliable debugging
            collate_fn=getattr(val_dataset, 'collate_fn', None)
        )
        
        # Load one batch
        print("Attempting to load one validation batch...")
        start_time = time.time()
        batch = next(iter(dataloader))
        elapsed = time.time() - start_time
        print(f"Successfully loaded batch in {elapsed:.2f} seconds!")
        print(f"Batch shape: {batch[0].shape}, labels: {batch[1]}")
        
        # If that worked, try with more samples
        print("\nTesting with larger batch (still num_workers=0)...")
        dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Keep this at 0 for reliable debugging
            collate_fn=getattr(val_dataset, 'collate_fn', None)
        )
        
        start_time = time.time()
        batch = next(iter(dataloader))
        elapsed = time.time() - start_time
        print(f"Successfully loaded larger batch in {elapsed:.2f} seconds!")
        print(f"Batch shape: {batch[0].shape}, labels: {batch[1]}")
        
        # Now try with a worker
        print("\nTesting with 1 worker...")
        dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=1,
            collate_fn=getattr(val_dataset, 'collate_fn', None)
        )
        
        start_time = time.time()
        batch = next(iter(dataloader))
        elapsed = time.time() - start_time
        print(f"Successfully loaded batch with worker in {elapsed:.2f} seconds!")
        
        # Print queue status periodically during processing
        print("\nMonitoring processor queue status...")
        for _ in range(3):
            print_queue_status(mediapipe_processor)
            time.sleep(2)
            
    except Exception as e:
        print(f"Error testing validation batch: {e}")
    finally:
        # Print final queue status
        print("\nFinal queue status:")
        print_queue_status(mediapipe_processor)
        
        # Clean up
        gc.collect()
        torch.cuda.empty_cache()
        # Shutdown the MediaPipe processor
        mediapipe_processor.shutdown()

if __name__ == "__main__":
    config_path = "src/config/visl2_spoter.yaml"
    test_val_batch(config_path)