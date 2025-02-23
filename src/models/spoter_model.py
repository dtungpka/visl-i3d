import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from tqdm import tqdm

from models import ModelRegistry
from datasets_folder import DatasetRegistry

def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])

class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Modified TransformerDecoderLayer omitting self-attention operation
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(SPOTERTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        del self.self_attn

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class SPOTER(nn.Module):
    """
    SPOTER (Sign POse-based TransformER) architecture for sign language recognition
    """
    def __init__(self, config: dict):
        super().__init__()
        hidden_dim = config['model']['hidden_dim']
        num_classes = config['model'].get('num_classes', config['dataset']['num_classes'])
        num_heads = config['model'].get('num_heads', 9)
        num_encoder_layers = config['model'].get('num_encoder_layers', 6)
        num_decoder_layers = config['model'].get('num_decoder_layers', 6)
        dim_feedforward = config['model'].get('dim_feedforward', 2048)
        dropout = config['model'].get('dropout', 0.1)
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        print(f"SPOTER model with {num_classes} classes and hidden dim {hidden_dim}")
        
        # Input projection to match hidden dimension
        input_size = 75 * 3  # MediaPipe gives 33 pose + 42 hand landmarks, each with x,y,z
        self.input_projection = nn.Linear(input_size, hidden_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.rand(1, 1, hidden_dim))
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Classification head
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        
        # Custom decoder layers
        custom_decoder_layer = SPOTERTransformerDecoderLayer(
            self.transformer.d_model, 
            self.transformer.nhead,
            dim_feedforward,
            dropout,
            "relu"
        )
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, 
                                                    self.transformer.decoder.num_layers)

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape (batch_size, seq_len, num_keypoints, 3)
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
    
        
        # Flatten landmarks for each frame
        x = inputs.reshape(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, num_keypoints*3)
        
        # Project to hidden dimension
        x = self.input_projection(x)  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Prepare for transformer: (seq_len, batch_size, hidden_dim)
        x = x.transpose(0, 1)
        
        # Prepare query: (1, batch_size, hidden_dim)
        query = self.class_query.unsqueeze(0).expand(1, batch_size, -1)
        
        # Pass through transformer
        out = self.transformer(x, query)
        
        # Get classification output
        out = self.linear_class(out.transpose(0, 1))
        
        return out.squeeze(1)  # Shape: (batch_size, num_classes)

    def train_model(self, config: dict):
        """Training implementatione"""
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
            dataset_config['mode'] = mode
            dataset_config = self._preprocess_dataset_config(dataset_config)
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
        """Main training loop"""
        num_epochs = config['hyperparameters']['num_epochs']
        best_val_acc = 0.0
        
        for epoch in tqdm(range(num_epochs), desc='Training'):
            # Training phase
            self.train()
            train_loss, train_acc = self._run_epoch('train')
            
            # Validation phase
            self.eval()
            with torch.no_grad():
                val_loss, val_acc = self._run_epoch('val')
            
            # Print epoch statistics
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.state_dict(), 'best_spoter.pth')

    def _run_epoch(self, mode):
        """Run one epoch of training or validation"""
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.dataloader_dict[mode], desc=f'{mode} epoch')
        for batch in pbar:
            # Get batch data
            inputs, labels = self._prepare_batch(batch)
            inputs = inputs.squeeze(0)
            labels = labels.squeeze(0)
            
            # print("inputs.shape:", inputs.shape)
            # print("labels.shape:", labels.shape)
            
            # Forward pass
            outputs = self(inputs)
            loss = self.criterion(outputs.squeeze(0), labels)
            
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
        inputs,labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
                           
        return inputs, labels

    def _preprocess_dataset_config(self, config):
        """Preprocess dataset configuration"""
        config['num_classes'] = self.num_classes
        return config

    def evaluate(self, config: dict):
        """Evaluation implementation"""
        self.eval()
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.to(self.device)
        
        # Setup test dataloader
        dataset_name = config['dataset']['name']
        dataset_config = config['dataset'].copy()
        dataset_config['mode'] = 'test'
        dataset_config = self._preprocess_dataset_config(dataset_config)
        test_loader = DatasetRegistry.get_dataloader(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            mode='test'
        )
        
        # Evaluate
        correct = 0
        total = 0
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

# Register the model
ModelRegistry.register('spoter', SPOTER)
