import os
import importlib.util
from typing import Dict, Any
from torch.utils.data import DataLoader, IterableDataset

class DatasetRegistry:
    _datasets: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, dataset_class: Any) -> None:
        cls._datasets[name] = dataset_class
        print(f"Registered dataset: {name}")

    @classmethod
    def get_dataset(cls, dataset_name: str, dataset_config: dict, mode: str) -> Any:
        if dataset_name not in cls._datasets:
            raise ValueError(f"Dataset {dataset_name} not found in registry. Available datasets: {cls.list_datasets()}")
        return cls._datasets[dataset_name](config=dataset_config, mode=mode)

    @classmethod
    def get_dataloader(cls, dataset_name: str, dataset_config: dict, mode: str) -> DataLoader:
        dataset = cls.get_dataset(dataset_name, dataset_config, mode)
        
        # Extract dataloader parameters from config
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        shuffle = mode == 'train' and dataset_config.get('shuffle', True)
        
        # Use dataset's collate_fn if available, otherwise use default
        collate_fn = getattr(dataset, 'collate_fn', None)
        
        # Check if the dataset is iterable
        if isinstance(dataset, IterableDataset):
            return DataLoader(
                dataset,
                batch_size=None,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )
        else:
            # Regular map-style dataset
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )

    @classmethod
    def list_datasets(cls) -> list:
        return list(cls._datasets.keys())

def _load_datasets():
    """
    Dynamically load all dataset files in the datasets directory using spec-based imports
    which are more reliable than module-based imports across different Python environments.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file in os.listdir(current_dir):
        if file.endswith('_dataset.py'):
            dataset_name = file[:-3]  # Remove .py extension
            module_path = os.path.join(current_dir, file)
            
            try:
                # Use spec-based import which is more reliable for local files
                spec = importlib.util.spec_from_file_location(dataset_name, module_path)
                if spec is None:
                    print(f"Could not create spec for {dataset_name}")
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"Successfully imported {dataset_name}")
                
                # The dataset class should register itself in its own file
                # If not, we can try to detect and register it here
                if hasattr(module, dataset_name[:-8].capitalize() + "Dataset"):  # e.g. "visl2" from "visl2_dataset"
                    dataset_class = getattr(module, dataset_name[:-8].capitalize() + "Dataset")
                    # Only register if not already registered
                    if dataset_name[:-8] not in DatasetRegistry._datasets:
                        DatasetRegistry.register(dataset_name[:-8], dataset_class)
                    
            except Exception as e:
                print(f"Error importing {dataset_name}: {e}")

# Load all datasets when the package is imported
_load_datasets()

# Export public API
__all__ = ['DatasetRegistry']