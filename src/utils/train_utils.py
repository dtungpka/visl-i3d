import torch
import os
from tqdm import tqdm
import numpy as np
from src.utils.utils import prepare_batch, save_training_plots, save_model
from torch.utils.data.dataloader import _BaseDataLoaderIter

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Run one epoch of training, supporting both map-style and iterable datasets"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    
    # Use tqdm with an unknown total for iterable datasets
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Get batch data
        inputs, labels = prepare_batch(batch, device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        batch_count += 1
        total_loss += loss.item()
        pred = outputs.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        
        # Update progress bar
        pbar.set_postfix({'loss': total_loss / batch_count, 'acc': correct / total})
            
    return total_loss / batch_count, correct / total

def validate(model, dataloader, criterion, device):
    """Run validation, supporting both map-style and iterable datasets"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for batch in pbar:
            # Get batch data
            inputs, labels = prepare_batch(batch, device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Update metrics
            batch_count += 1
            total_loss += loss.item()
            pred = outputs.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
            
            # Update progress bar
            pbar.set_postfix({'loss': total_loss / batch_count, 'acc': correct / total})
                
    return total_loss / batch_count, correct / total

def train_model_loop(model, dataloaders, optimizer, criterion, config):
    """Main training loop with metric logging and checkpoints"""
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    
    num_epochs = config['hyperparameters']['num_epochs']
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    experiment_name = config.get('experiment_name', 'experiment')
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_epoch(model, dataloaders['train'], optimizer, criterion, device)
        losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        val_loss, val_acc = validate(model, dataloaders['val'], criterion, device)
        val_accs.append(val_acc)
        
        # Log epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save checkpoints if enabled
        if config.get('save_checkpoints', False):
            # Create checkpoint directories if necessary
            ckpt_dir = f"out-checkpoints/{experiment_name}"
            os.makedirs(ckpt_dir, exist_ok=True)
            if train_acc > top_train_acc:
                top_train_acc = train_acc
                save_model(model, os.path.join(ckpt_dir, f"checkpoint_t_{checkpoint_index}.pth"))
            if val_acc > top_val_acc:
                top_val_acc = val_acc
                save_model(model, os.path.join(ckpt_dir, f"checkpoint_v_{checkpoint_index}.pth"))
        
        # Reset top accuracies every 10 epochs and update checkpoint index
        if (epoch + 1) % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1
        
        # Record learning rate progress
        lr_progress.append(optimizer.param_groups[0]["lr"])
    
    # Plot and save metrics if requested
    if config.get('plot_stats', False) or config.get('plot_lr', False):
        save_training_plots(losses, train_accs, val_accs, lr_progress, experiment_name)
    
    print("\nTraining complete.")
    return model

def evaluate_model_loop(model, dataloader, criterion, config):
    """Evaluation implementation, supporting both map-style and iterable datasets"""
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    model.eval()
    
    batch_count = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = prepare_batch(batch, device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
            batch_count += 1
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return {'accuracy': accuracy}