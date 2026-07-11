"""
Utility functions: logging, early stopping, loss functions
"""
import os
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn


class Logger:
    """Dual logger that writes to both console and file."""
    
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
        
        # Clear/create log file
        with open(log_path, 'w') as f:
            f.write(f"=" * 70 + "\n")
            f.write(f"Experiment Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 70 + "\n\n")
    
    def log(self, message):
        """Log message to both console and file."""
        print(message)
        with open(self.log_path, 'a') as f:
            f.write(message + "\n")
    
    def section(self, title):
        """Log a section header."""
        sep = "=" * 70
        self.log(f"\n{sep}")
        self.log(title)
        self.log(sep)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class LpLoss:
    """Relative Lp loss for operator learning."""
    
    def __init__(self, p=2, size_average=True):
        self.p = p
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 
            self.p, dim=1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, dim=1)
        
        if self.size_average:
            return torch.mean(diff_norms / (y_norms + 1e-6))
        else:
            return torch.sum(diff_norms / (y_norms + 1e-6))

    def __call__(self, x, y):
        return self.rel(x, y)


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)