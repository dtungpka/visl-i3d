import torch
from torch import nn
import torch.nn.functional as F
from . import ModelRegistry
try:
    from src.datasets import DatasetRegistry
except:
    from datasets import DatasetRegistry
    
from tqdm import tqdm


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(0, (self.kernel_size[dim] - self.stride[dim]))
        else:
            return max(0, (self.kernel_size[dim] - (s % self.stride[dim])))

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        x = F.pad(x, (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b))
        return super().forward(x)

class Unit3D(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1), padding=0, activation_fn=F.relu, use_batch_norm=True, use_bias=False):
        super(Unit3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, output_channels, kernel_shape, stride=stride, padding=padding, bias=use_bias)
        self.batch_norm = nn.BatchNorm3d(output_channels) if use_batch_norm else None
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()
        self.branch1x1 = Unit3D(in_channels, out_channels[0], kernel_shape=(1, 1, 1), stride=(1, 1, 1))
        self.branch5x5 = nn.Sequential(
            Unit3D(in_channels, out_channels[1], kernel_shape=(1, 1, 1)),
            Unit3D(out_channels[1], out_channels[2], kernel_shape=(5, 5, 5))
        )
        self.branch3x3 = nn.Sequential(
            Unit3D(in_channels, out_channels[3], kernel_shape=(1, 1, 1)),
            Unit3D(out_channels[3], out_channels[4], kernel_shape=(3, 3, 3))
        )
        self.branch_pool = nn.Sequential(
            MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            Unit3D(in_channels, out_channels[5], kernel_shape=(1, 1, 1))
        )

    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch5x5(x)
        branch3 = self.branch3x3(x)
        branch4 = self.branch_pool(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class InceptionI3d(nn.Module):
    """
    Inception-v1 I3D architecture.
    The model is introduced in:
    Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
    https://arxiv.org/abs/1705.07750
    """
    def __init__(self, num_classes=157, spatial_squeeze=True, final_endpoint='Logits', 
                 in_channels=3, dropout_keep_prob=0.5, **kwargs):
        super(InceptionI3d, self).__init__()
        self.num_classes = num_classes
        self.spatial_squeeze = spatial_squeeze
        self.final_endpoint = final_endpoint
        self.dropout_keep_prob = dropout_keep_prob
        self.in_channels = in_channels
        
        # Build the network architecture
        self.build()
        
        # Initialize weights
        self._initialize_weights()

    def build(self):
        """Builds the I3D architecture."""
        # First conv block
        self.conv3d_1a_7x7 = Unit3D(self.in_channels, 64, kernel_shape=(7, 7, 7), stride=(2, 2, 2))
        
        # First maxpool
        self.maxpool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        
        # Inception modules
        self.inception_3b = InceptionModule(64, [64, 96, 128, 16, 32, 32], name='Inception_3b')
        self.inception_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64], name='Inception_3c')
        
        # Second maxpool
        self.maxpool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        
        # More inception modules
        self.inception_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64], name='Inception_4b')
        self.inception_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64], name='Inception_4c')
        self.inception_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64], name='Inception_4d')
        
        # Third maxpool
        self.maxpool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Final inception modules
        self.inception_5b = InceptionModule(512, [112, 144, 288, 32, 64, 64], name='Inception_5b')
        self.inception_5c = InceptionModule(528, [256, 160, 320, 32, 128, 128], name='Inception_5c')
        
        # Dropout
        self.dropout = nn.Dropout(p=1 - self.dropout_keep_prob)
        
        # Final convolution for classification
        self.logits = nn.Conv3d(832, self.num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        # Conv block 1
        x = self.conv3d_1a_7x7(x)
        x = self.maxpool3d_2a_3x3(x)
        
        # Inception blocks 3
        x = self.inception_3b(x)
        x = self.inception_3c(x)
        x = self.maxpool3d_4a_3x3(x)
        
        # Inception blocks 4
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.maxpool3d_5a_2x2(x)
        
        # Inception blocks 5
        x = self.inception_5b(x)
        x = self.inception_5c(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification
        x = self.logits(x)
        
        if self.spatial_squeeze:
            x = F.adaptive_avg_pool3d(x, (1, 1, 1))
            x = x.view(x.size(0), -1)
        
        return x

    def train_model(self, config: dict):
        """Training logic implementation."""
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            **config['optimizer_config']
        )
        
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        
        dataset_name = config['dataset']['name']
       
        self.dataloader_dict = {}
        for mode in ['train', 'val', 'test']:
            dataset_current_config = config['dataset'][mode]
            # add extra model infor 
            dataset_current_config = self.preprocessing_dataset_config(dataset_current_config)
            #add dataloader
            self.dataloader_dict[mode] = DatasetRegistry.get_dataloader(dataset_name = dataset_name,
                                                                  dataset_config = dataset_current_config,
                                                                  mode=mode)
        self.train_loop()
    def train_loop(self):
        
        """Training loop implementation."""
        
        self.train()  # Set model to training mode
        total_epochs = self.config['hyperparameters']['num_epochs']
        for epoch in range(self.config['hyperparameters']['num_epochs']):
            for mode in ['train', 'val']:
                pbar = tqdm(enumerate(self.dataloader_dict[mode]),total = len(self.dataloader_dict[mode]))
                pbar.set_description_str(f"{epoch}/{total_epochs}")
                for batch_idx, data in pbar:
                    import pdb;pdb.set_trace()
                    
    def preprocessing_dataset_config(self,config):
        config['num_classes']  = self.config['model']['num_classes']
        return config
    def evaluate(self, config: dict):
        """Evaluation logic implementation."""
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Evaluation loop implementation would go here
            pass
            
    def inference(self, input_data):
        """Inference logic implementation."""
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            return self.forward(input_data)

    def replace_logits(self, num_classes):
        """Replace the final classification layer for transfer learning."""
        self.logits = nn.Conv3d(832, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self._initialize_weights()

    def load_pretrained_weights(self, state_dict):
        """Load pretrained weights into the model."""
        self.load_state_dict(state_dict)

# Register the model with the registry
ModelRegistry.register('i3d', InceptionI3d)
