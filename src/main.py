import os
import yaml
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))  # Go one level up

from models import ModelRegistry
from datasets import DatasetRegistry
from src.utils.utils import get_optimizer, get_criterion, set_seed
from utils.train_utils import train_model_loop, evaluate_model_loop

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_dataloaders(config):
    """Initialize and return all required dataloaders based on config"""
    dataset_name = config['dataset']['name']
    dataloader_dict = {}
    
    print(f"Available datasets: {DatasetRegistry.list_datasets()}")
    print(f"Setting up dataloaders for dataset: {dataset_name}")
    
    # Set reasonable worker count based on the system
    num_workers = config['dataset'].get('num_workers', 4)
    config['dataset']['num_workers'] = num_workers
    
    for mode in ['train', 'val', 'test']:
        # Create a copy of the dataset config
        dataset_config = config['dataset'].copy()
        
        # Use fewer workers for validation and test to prevent memory issues
        if mode != 'train':
            dataset_config['num_workers'] = min(2, num_workers)
            
        try:
            dataloader = DatasetRegistry.get_dataloader(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                mode=mode
            )
            dataloader_dict[mode] = dataloader
            print(f"Successfully created {mode} dataloader")
        except Exception as e:
            print(f"Error creating {mode} dataloader: {str(e)}")
            if mode == 'train':
                raise  # Only re-raise for train mode since it's required
    
    if 'train' not in dataloader_dict:
        raise ValueError("No training dataloader was created. Check dataset configuration.")
    
    return dataloader_dict

def train_model(model, config):
    """Train model using configuration"""
    # Set random seed if specified
    if 'seed' in config:
        set_seed(config['seed'])
    
    # Setup optimizer and criterion
    optimizer = get_optimizer(model.parameters(), config)
    criterion = get_criterion(config)
    
    # Setup dataloaders
    dataloaders = setup_dataloaders(config)
    
    # Train the model
    return train_model_loop(model, dataloaders, optimizer, criterion, config)

def evaluate_model(model, config):
    """Evaluate model using configuration"""
    # Setup criterion
    criterion = get_criterion(config)
    
    # Setup test dataloader
    dataset_name = config['dataset']['name']
    dataset_config = config['dataset'].copy()
    dataset_config['mode'] = 'test'
    test_loader = DatasetRegistry.get_dataloader(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        mode='test'
    )
    
    # Evaluate the model
    return evaluate_model_loop(model, test_loader, criterion, config)

def main(config_path):
    config = load_config(config_path)

    # Create model instance
    model_config = config['model']
    model = ModelRegistry.get_model(
        model_config['model_name'], config=config)

    # Train or evaluate based on config
    if config.get('mode') == 'train':
        train_model(model, config)
    elif config.get('mode') == 'evaluate':
        evaluate_model(model, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    main(args.config)