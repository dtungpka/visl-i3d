from torchvision import transforms

def get_augmentation_pipeline(aug_config=None, output_type='rgb'):
    """
    Returns an augmentation pipeline based on the provided config.
    If aug_config is None, a default pipeline is used.
    """
    if aug_config is None:
        # Default pipeline for rgb/rgbd outputs.
        if output_type in ['rgb', 'rgbd']:
            mean = [0.485, 0.456, 0.406] if output_type == 'rgb' else [0.5] * 4
            std = [0.229, 0.224, 0.225] if output_type == 'rgb' else [0.25] * 4
            pipeline = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            pipeline = None
        return pipeline

    # Else, build the pipeline based on the provided configuration.
    transform_list = []

    # Ensure image is a PIL image first.
    transform_list.append(transforms.ToPILImage())

    for aug in aug_config:
        if aug['type'] == 'random_crop':
            size = tuple(aug['size'])
            transform_list.append(transforms.RandomResizedCrop(size=size, scale=aug.get('scale', (0.8, 1.0))))
        elif aug['type'] == 'random_flip':
            if aug.get('horizontal', False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug.get('vertical', False):
                transform_list.append(transforms.RandomVerticalFlip())
        elif aug['type'] == 'color_jitter':
            transform_list.append(transforms.ColorJitter(
                brightness=aug.get('brightness', 0.2),
                contrast=aug.get('contrast', 0.2),
                saturation=aug.get('saturation', 0.2),
                hue=aug.get('hue', 0.1)
            ))
        elif aug['type'] == 'resize':
            size = tuple(aug['size'])
            transform_list.append(transforms.Resize(size))
        elif aug['type'] == 'normalize':
            if output_type in ['rgb', 'rgbd']:
                mean = aug.get('mean', [0.485, 0.456, 0.406]) if output_type == 'rgb' else aug.get('mean', [0.5] * 4)
                std = aug.get('std', [0.229, 0.224, 0.225]) if output_type == 'rgb' else aug.get('std', [0.25] * 4)
                transform_list.append(transforms.ToTensor())
                transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)

def apply_augmentation(image, aug_config=None, output_type='rgb'):
    pipeline = get_augmentation_pipeline(aug_config, output_type)
    return pipeline(image) if pipeline is not None else image