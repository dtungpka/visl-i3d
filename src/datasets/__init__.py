# This file initializes the datasets package, allowing for easy imports of the data loading and augmentation functionalities.
import importlib

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
            # Assuming the module name is {name}_dataset and it exists in this package.
            module_name = f"src.datasets.{key}_dataset"
            module = importlib.import_module(module_name)
            # Search for a class in the module with name starting with the dataset key (case-insensitive).
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and key in attr.lower():
                    cls._datasets[key] = obj
                    return obj
            raise ImportError(f"Dataset {name} not found.")