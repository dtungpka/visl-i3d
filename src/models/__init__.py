import os
import importlib
from typing import Dict, Type, Any

class ModelRegistry:
    _models: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, model_class: Any) -> None:
        cls._models[name] = model_class

    @classmethod
    def get_model(cls, name: str, **kwargs) -> Any:
        if name not in cls._models:
            raise ValueError(f"Model {name} not found in registry")
        return cls._models[name](**kwargs)

    @classmethod
    def list_models(cls) -> list:
        return list(cls._models.keys())

def _load_models():
    """Dynamically load all model files in the models directory"""
    current_dir = os.path.dirname(__file__)
    for file in os.listdir(current_dir):
        if file.endswith('_model.py'):
            model_name = file[:-3]  # Remove .py
            try:
                module = importlib.import_module(f'.{model_name}', package='src.models')
            except:
                module = importlib.import_module(f'.{model_name}', package='models')

def train_model(model: Any, config: dict):
    """Generic training function that delegates to model's train method"""
    return model.train(config)

def evaluate_model(model: Any, config: dict):
    """Generic evaluation function that delegates to model's evaluate method"""
    return model.evaluate(config)

def inference(model: Any, input_data: Any):
    """Generic inference function that delegates to model's inference method"""
    return model.inference(input_data)

# Load all models when the package is imported
_load_models()

# Export public API
__all__ = ['ModelRegistry', 'train_model', 'evaluate_model', 'inference']