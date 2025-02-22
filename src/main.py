import os
import yaml
try:
    from src.models import ModelRegistry, train_model, evaluate_model
    # from src.utils import setup_logging
except:
    from models import ModelRegistry, train_model, evaluate_model
    # from utils import setup_logging



def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'phuoc_rgb_only_config.yaml')
    config = load_config(config_path)

    # setup_logging(config.get('logging', {}))

    # Create model instance.
    model_config = config['model']
    model = ModelRegistry.get_model(
        model_config['model_name'],
        num_classes=config.get('model', {}).get('num_classes', 10)
    )


    # Train or evaluate based on config.
    if config.get('mode') == 'train':
        train_model(model, config)
    elif config.get('mode') == 'evaluate':
        evaluate_model(model, config)

if __name__ == "__main__":
    main()