import torch
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from models import ModelRegistry
from datasets import DatasetRegistry

class SimpleRGBModel(nn.Module):
    """
    A simple 3D CNN model for video classification using RGB input.
    This demonstrates how to create a new model compatible with the framework.
    """
    def __init__(self, config: dict):
        super().__init__()
        # Extract model parameters from config
        self.num_classes = config['model'].get('num_classes', config['dataset']['num_classes'])
        self.in_channels = config['model'].get('in_channels', 3)  # RGB has 3 channels
        
        print(f"Creating SimpleRGBModel with {self.num_classes} classes")
        
        # Define a simple 3D CNN architecture
        self.features = nn.Sequential(
            # First 3D conv block
            nn.Conv3d(self.in_channels, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            
            # Second 3D conv block
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Third 3D conv block
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        
        # Calculate the size of flattened features based on input dimensions
        self._calculate_flat_size(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config['model'].get('dropout', 0.5)),
            nn.Linear(512, self.num_classes)
        )
    
    def _calculate_flat_size(self, config):
        """Calculate the size of flattened features after convolutional layers"""
        # Get input dimensions from config
        height = config['dataset'].get('height', 224)
        width = config['dataset'].get('width', 224)
        frames = config['dataset'].get('n_frames', 32)
        
        # Calculate feature map size after convolutions and pooling
        h = height // 8  # After 3 pooling layers with stride 2
        w = width // 8   # After 3 pooling layers with stride 2
        t = frames // 4   # After temporal pooling
        
        self.flat_size = 256 * t * h * w

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Input tensor of shape (batch_size, channels, frames, height, width)
        """
        # Apply 3D CNN feature extraction
        features = self.features(x)
        
        # Flatten the features
        features = features.view(features.size(0), -1)
        
        # Apply classifier
        out = self.classifier(features)
        
        return out
    
    def train_model(self, config: dict):
        """Training implementation"""
        # Setup training components
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config['hyperparameters']['learning_rate'],
            weight_decay=config['hyperparameters'].get('weight_decay', 0.0001)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize dataloaders
        dataset_name = config['dataset']['name']
        self.dataloader_dict = {}
        
        for mode in ['train', 'val', 'test']:
            dataset_config = config['dataset'].copy()
            self.dataloader_dict[mode] = DatasetRegistry.get_dataloader(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                mode=mode
            )
        
        print(f"Training on device: {self.device}")
        self.to(self.device)
        
        # Training loop
        self._train_loop(config)

    def _train_loop(self, config):
        """Main training loop with metric logging"""
        num_epochs = config['hyperparameters']['num_epochs']
        top_val_acc = 0
        checkpoint_index = 0
        losses, train_accs, val_accs = [], [], []
        experiment_name = config.get('experiment_name', 'rgb_experiment')
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss, train_acc = self._run_epoch('train')
            losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation phase
            self.eval()
            with torch.no_grad():
                val_loss, val_acc = self._run_epoch('val')
            val_accs.append(val_acc)
            
            # Log epoch statistics
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save checkpoints if enabled
            if config.get('save_checkpoints', False) and val_acc > top_val_acc:
                top_val_acc = val_acc
                ckpt_dir = f"out-checkpoints/{experiment_name}"
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(self.state_dict(), 
                          os.path.join(ckpt_dir, f"checkpoint_{checkpoint_index}.pth"))
                print(f"Saved checkpoint with val_acc: {val_acc:.4f}")
            
            # Reset top accuracy every 10 epochs
            if (epoch + 1) % 10 == 0:
                top_val_acc = 0
                checkpoint_index += 1
        
        # Plot training statistics if requested
        if config.get('plot_stats', False):
            self._save_training_plots(losses, train_accs, val_accs, experiment_name)
        
        print("\nTraining complete!")

    def _run_epoch(self, mode):
        """Run one epoch of training or validation"""
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.dataloader_dict[mode], desc=f'{mode} epoch')
        for batch in pbar:
            # Get batch data
            inputs, labels = self._prepare_batch(batch)
            
            # Forward pass
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass in training mode
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = outputs.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1), 'acc': correct / total})
            
        return total_loss / len(self.dataloader_dict[mode]), correct / total

    def _prepare_batch(self, batch):
        """Prepare batch data for training/validation"""
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        return inputs, labels
        
    def _save_training_plots(self, losses, train_accs, val_accs, experiment_name):
        """Save training metrics as plots"""
        os.makedirs("out-img", exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"out-img/{experiment_name}_loss.png")
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_accs, label='Training Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"out-img/{experiment_name}_accuracy.png")

    def evaluate(self, config: dict):
        """Evaluation implementation"""
        self.eval()
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.to(self.device)
        
        # Setup test dataloader
        dataset_name = config['dataset']['name']
        dataset_config = config['dataset'].copy()
        test_loader = DatasetRegistry.get_dataloader(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            mode='test'
        )
        
        # Initialize metrics
        correct = 0
        total = 0
        
        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                inputs, labels = self._prepare_batch(batch)
                outputs = self(inputs)
                pred = outputs.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        return {'accuracy': accuracy}

# Register the model with the registry
ModelRegistry.register('simple_rgb_model', SimpleRGBModel)