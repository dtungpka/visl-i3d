from torch import nn
import torch.nn.functional as F
from . import ModelRegistry

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
    def __init__(self, num_classes=157, spatial_squeeze=True, final_endpoint='Logits', in_channels=3, dropout_keep_prob=0.5):
        super(InceptionI3d, self).__init__()
        self.num_classes = num_classes
        self.spatial_squeeze = spatial_squeeze
        self.final_endpoint = final_endpoint
        self.dropout_keep_prob = dropout_keep_prob

        self.build()

    def build(self):
        self.conv3d_1a_7x7 = Unit3D(3, 64, kernel_shape=(7, 7, 7), stride=(2, 2, 2))
        self.maxpool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.inception_3b = InceptionModule(64, [64, 128, 128, 128, 128, 32], name='Inception_3b')
        self.inception_3c = InceptionModule(256, [128, 128, 128, 128, 128, 64], name='Inception_3c')
        self.maxpool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.inception_4b = InceptionModule(256, [192, 192, 192, 192, 192, 64], name='Inception_4b')
        self.inception_4c = InceptionModule(256, [192, 192, 192, 192, 192, 64], name='Inception_4c')
        self.inception_4d = InceptionModule(256, [192, 192, 192, 192, 192, 64], name='Inception_4d')
        self.maxpool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.inception_5b = InceptionModule(256, [192, 192, 192, 192, 192, 64], name='Inception_5b')
        self.inception_5c = InceptionModule(256, [192, 192, 192, 192, 192, 64], name='Inception_5c')
        self.logits = nn.Conv3d(256, self.num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1a_7x7(x)
        x = self.maxpool3d_2a_3x3(x)
        x = self.inception_3b(x)
        x = self.inception_3c(x)
        x = self.maxpool3d_4a_3x3(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.maxpool3d_5a_2x2(x)
        x = self.inception_5b(x)
        x = self.inception_5c(x)
        x = self.logits(x)
        if self.spatial_squeeze:
            x = F.adaptive_avg_pool3d(x, (1, 1, 1))
            x = x.view(x.size(0), -1)
        return x

    def replace_logits(self, num_classes):
        self.logits = nn.Conv3d(256, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    def extract_features(self, x):
        # This method can be implemented to extract features from intermediate layers
        pass

class InceptionI3d:
    def __init__(self, num_classes: int, **kwargs):
        self.num_classes = num_classes
        # Initialize your model architecture here
        
    def train(self, config: dict):
        # Implement training logic
        pass
        
    def evaluate(self, config: dict):
        # Implement evaluation logic
        pass
        
    def inference(self, input_data):
        # Implement inference logic
        pass

# Register the model
ModelRegistry.register('i3d', InceptionI3d)