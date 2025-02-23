# This file initializes the datasets package, allowing for easy imports of the data loading and augmentation functionalities.
import importlib
from torch.utils.data import DataLoader
from . import *

class DatasetRegistry:
    _datasets = {}

    @classmethod
    def register(cls, name, dataset_cls):
        cls._datasets[name.lower()] = dataset_cls

    @classmethod
    def get_dataset(cls, name):
        key = name.lower()
        if key in cls._datasets:
            return cls._datasets[key]
        else:
            
            module_name = f".{key}_dataset"
            module = importlib.import_module(module_name, package="datasets_folder")
            
            # Search for a class in the module with name starting with the dataset key (case-insensitive).
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and key in attr.lower():
                    cls._datasets[key] = obj
                    return obj
            raise ImportError(f"Dataset {name} not found.")
    @classmethod
    def get_dataloader(cls,dataset_name,dataset_config,mode):
        
        dataset_cls = cls.get_dataset(dataset_name)
        dataset =  dataset_cls(dataset_config,mode = mode)
        
        #batch size, num_workers, ...
        dataloder_config = dataset_config['dataloader_config']
        
        return DataLoader(dataset,collate_fn=dataset.collate_fn,**dataloder_config)