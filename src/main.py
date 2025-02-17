import os
import yaml
from src.models import ModelRegistry, train_model, evaluate_model
from src.datasets import DatasetRegistry  # NEW import from registry package
from src.datasets.loader import DataLoader
from src.utils import setup_logging

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    config = load_config(config_path)

    setup_logging(config.get('logging', {}))

    # Create model instance.
    model = ModelRegistry.get_model(
        config['model_selection'],
        num_classes=config.get('model', {}).get('num_classes', 10)
    )

    # Create data loader.
    data_loader = DataLoader(config['data'])
    
    # Load dataset dynamically via DatasetRegistry.
    dataset_name = config['data'].get('dataset', 'visl2')  # Default to 'visl2'
    DatasetClass = DatasetRegistry.get_dataset(dataset_name)

    # Pass the augmentation config from YAML.
    aug_config = config['augmentation']['augmentations'] if config['augmentation']['use_augmentation'] else None

    dataset = DatasetClass(
        dataset_path=config['data']['train_data_path'],
        apply_aug=config['augmentation']['use_augmentation'],
        aug_config=aug_config,
        output=config['data'].get('output', 'rgb')
    )

    # Train or evaluate based on config.
    if config.get('mode') == 'train':
        train_model(model, config)
    elif config.get('mode') == 'evaluate':
        evaluate_model(model, config)

if __name__ == "__main__":
    main()