import os
import importlib
import torch
from typing import Dict, Any, List
from torch.utils.data import DataLoader

class DatasetRegistry:
    _datasets: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, dataset_class: Any) -> None:
        cls._datasets[name] = dataset_class

    @classmethod
    def get_dataset(cls, dataset_name: str, dataset_config: dict, mode: str) -> Any:
        if dataset_name not in cls._datasets:
            raise ValueError(f"Dataset {dataset_name} not found in registry")
        return cls._datasets[dataset_name](config=dataset_config, mode=mode)

    @classmethod
    def get_dataloader(cls, dataset_name: str, dataset_config: dict, mode: str) -> DataLoader:
        dataset = cls.get_dataset(dataset_name, dataset_config, mode)
        
        # Extract dataloader parameters from config
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        shuffle = mode == 'train' and dataset_config.get('shuffle', True)
        
        # Check if the dataset is iterable (no need for shuffle or sampler)
        from torch.utils.data import IterableDataset
        if isinstance(dataset, IterableDataset):
            return DataLoader(
                dataset,
                batch_size=None,  # Batch size is handled in the dataset for IterableDataset
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            # Regular map-style dataset
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=cls.default_collate_fn
            )

    @staticmethod
    def default_collate_fn(batch: List[dict]) -> tuple:
        """
        Default collate function that handles dictionary-based dataset outputs.
        
        Args:
            batch: List of dictionaries, each with 'data' and 'label' keys
            
        Returns:
            Tuple of (batched_data, batched_labels)
        """
        if not batch:
            return None
            
        # Check if batch contains dictionaries with expected keys
        if isinstance(batch[0], dict) and 'data' in batch[0] and 'label' in batch[0]:
            # Extract data and labels
            batch_data = [item['data'] for item in batch]
            batch_labels = [item['label'] for item in batch]
            
            # Stack tensors along batch dimension
            batch_tensor = torch.stack(batch_data, dim=0)
            batch_labels = torch.stack(batch_labels, dim=0)
            
            return batch_tensor, batch_labels
        else:
            # Fallback to default behavior
            return torch.utils.data.default_collate(batch)

    @classmethod
    def list_datasets(cls) -> list:
        return list(cls._datasets.keys())

def _load_datasets():
    """Dynamically load all dataset files in the datasets directory"""
    current_dir = os.path.dirname(__file__)
    for file in os.listdir(current_dir):
        if file.endswith('_dataset.py'):
            dataset_name = file[:-3]  # Remove .py
            try:
                module = importlib.import_module(f'.{dataset_name}', package='src.datasets_folder')
            except ImportError:
                try:
                    module = importlib.import_module(f'.{dataset_name}', package='datasets_folder')
                except ImportError:
                    print(f"Could not import dataset module {dataset_name}")

# Load all datasets when the package is imported
_load_datasets()

# Export public API
__all__ = ['DatasetRegistry']