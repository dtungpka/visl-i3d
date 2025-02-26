import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch import nn, optim

def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0

def log_metrics(metrics, log_file='metrics.log'):
    with open(log_file, 'a') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_batch(batch, device):
    """Prepare batch data for training/validation, handling different batch formats"""
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        inputs, labels = batch
        inputs = inputs.to(device)
        
        # Handle different label types (long tensor, one-hot encoded, etc.)
        if isinstance(labels, torch.Tensor):
            if labels.dtype == torch.float32 and labels.dim() > 1:
                # Assuming one-hot encoded labels
                labels = labels.to(device)
            else:
                # Regular class indices
                labels = labels.to(device)
        else:
            labels = torch.tensor(labels, dtype=torch.long).to(device)
        
        return inputs, labels
    else:
        # For custom batch formats, you may need to extend this
        raise ValueError(f"Unsupported batch format: {type(batch)}")

def get_optimizer(model_parameters, config):
    """Create optimizer based on configuration"""
    optimizer_name = config['hyperparameters'].get('optimizer', 'adam').lower()
    lr = config['hyperparameters'].get('learning_rate', 0.001)
    weight_decay = config['hyperparameters'].get('weight_decay', 0.0001)
    
    if optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config['hyperparameters'].get('momentum', 0.9)
        return optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

def get_criterion(config):
    """Create loss function based on configuration"""
    loss_name = config['hyperparameters'].get('loss', 'cross_entropy').lower()
    
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"Loss function {loss_name} not supported")

def save_training_plots(losses, train_accs, val_accs, lr_progress, experiment_name):
    """Save training plots for loss, accuracy, and learning rate"""
    os.makedirs("out-img", exist_ok=True)
    
    # Plot loss and accuracy
    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
    ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")
    if val_accs:
        ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
    ax.grid()
    fig.savefig(f"out-img/{experiment_name}_loss.png")
    plt.close(fig)
    
    # Plot learning rate
    if lr_progress:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="Learning Rate", title="")
        ax1.grid()
        fig1.savefig(f"out-img/{experiment_name}_lr.png")
        plt.close(fig1)