# train.py
import sys
import os
import argparse
import time
import random
import numpy as np
import torch
import torch.optim as optim

# Robustly add parent directory to path to find optimizers/ regardless of execution dir
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

# Assumes you have these optimizers available in ../optimizers
# If not, comment out the imports you don't have
try:
    from optimizers import Cadam, cubic_damping_opt, iKFAD, LDHD
except ImportError:
    print("Warning: Custom optimizers not found in path. Ensure 'optimizers' module exists.")

from dataloader import get_dataloaders
from model import get_model

def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            _, predicted_labels = torch.max(logits, dim=1) 
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
    return correct_pred.float()/num_examples * 100

def compute_loss(model, data_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_buffer = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            loss_buffer += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            
    return loss_buffer / total_samples if total_samples > 0 else 0

def get_optimizer(model, optim_name, hp, device):
    params = model.parameters()
    if optim_name == 'msgd':
        print('h: ', hp['lr'], 'momentum: ', hp['momentum'])
        return optim.SGD(params, lr=hp['lr'], momentum=hp['momentum'])
    elif optim_name == 'adam':
        print('h: ', hp['lr'], 'betas: ', hp['betas'])
        return optim.Adam(params, lr=hp['h'], betas=hp['betas'])
    elif optim_name == 'cadam':
        print('h: ', hp['h'], 'gamma: ', hp['gamma'], 'c: ', hp['c'], 'alpha: ', hp['alpha'], 'eps: ', 1e-8)
        return Cadam(params, h=hp['h'], gamma=hp['gamma'], c=hp['c'], 
                     alpha=hp['alpha'], eps=1e-8, device=device)
    elif optim_name == 'cadam0':
        print('h: ', hp['h'], 'gamma: ', 0., 'c: ', hp['c'], 'alpha: ', hp['alpha'], 'eps: ', 1e-8)
        return Cadam(params, h=hp['h'], gamma=0., c=hp['c'], 
                     alpha=hp['alpha'], eps=1e-8, device=device)
    elif optim_name == 'cd':
        print('h: ', hp['h'], 'gamma: ', hp['gamma'], 'c: ', hp['c'])
        return cubic_damping_opt(params, h=hp['h'], gamma=hp['gamma'], c=hp['c'], device=device)
    elif optim_name == 'ldhd':
        print('h: ', hp['h'], 'gamma: ', hp['gamma'])
        return LDHD(params, h=hp['h'], gamma=hp['gamma'], device=device)
    elif optim_name == 'cd0':
        print('h: ', hp['h'], 'gamma: ', 0., 'c: ', hp['c'])
        return cubic_damping_opt(params, h=hp['h'], gamma=0., c=hp['c'], device=device)
    elif optim_name == 'ikfad':
        print('h: ', hp['h'], 'gamma: ', hp['gamma'], 'alpha: ', hp['alpha'], 'mu: ', hp['mu'])
        return iKFAD(params, h=hp['h'], gamma=hp['gamma'], alpha=hp['alpha'], mu=hp['mu'], device=device)
    elif optim_name == 'ikfad0':
        print('h: ', hp['h'], 'gamma: ', 0., 'alpha: ', hp['alpha'], 'mu: ', hp['mu'])
        return iKFAD(params, h=hp['h'], gamma=0., alpha=hp['alpha'], mu=hp['mu'], device=device)
    else:
        raise ValueError(f"Optimizer {optim_name} not implemented.")

def train_engine(optim_name, hp, num_epochs, device, metric, seed):
    # 1. Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    print('Seed is: ', seed)
    
    start_time = time.time()
    
    # 2. Data & Model
    print(f"Loading data for CIFAR10...", flush=True)
    # Increased batch size to 128 (standard for ViT stability)
    train_loader, valid_loader, test_loader = get_dataloaders(batch_size=128)
    
    print(f"Initializing NanoViT for CIFAR10...", flush=True)
    model = get_model(device)
    
    # 3. Optimizer
    print(f"Initializing optimizer: {optim_name}", flush=True)
    optimizer = get_optimizer(model, optim_name, hp, device)
    
    criterion = torch.nn.CrossEntropyLoss()

    # Checkpoint setup
    checkpoint_dir = os.path.join('cifar10', 'checkpoints', optim_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Track best validation loss
    best_val_loss = float('inf')

    # 4. Loop
    print("Starting training...", flush=True)
    for epoch in range(num_epochs):
        model.train()
        
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            logits = model(images)
            train_loss = criterion(logits, labels)
            
            # Step Logic
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Accumulate metrics
            batch_size = labels.size(0)
            train_loss_accum += train_loss.item() * batch_size
            
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += batch_size

        # Calculate Epoch Metrics
        avg_train_loss = train_loss_accum / train_total
        avg_train_acc = 100.0 * train_correct / train_total

        # Validation phase
        valid_loss = compute_loss(model, valid_loader, device)
        valid_acc = compute_accuracy(model, valid_loader, device)
        
        # Log One Line Summary
        print(f'Epoch {epoch+1:04d}/{num_epochs:04d} | '
              f'Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | '
              f'Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}% | '
              f'Time: {(time.time() - start_time)/60:.2f} min', flush=True)

        # Save Checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss,
            'acc': valid_acc
        }, os.path.join(checkpoint_dir, 'last.pth'))

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'acc': valid_acc
            }, os.path.join(checkpoint_dir, 'best.pth'))
            print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}", flush=True)

    print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min', flush=True)
    print(f'Test accuracy: {compute_accuracy(model, test_loader, device):.2f}%', flush=True)
    
    del model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    return best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10 Train Engine')
    parser.add_argument('--optimizer', type=str, required=True, help='cadam, msgd, etc')
    parser.add_argument('--metric', type=str, default='loss', help='Unused, always val_loss')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=123)
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate / h')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=1.0) 
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    args = parser.parse_args()
    
    hp = {
        'lr': args.lr, 
        'h': args.lr, 
        'momentum': args.momentum, 
        'betas': (args.beta1, args.beta2), 
        'gamma': args.gamma, 
        'c': args.c, 
        'alpha': args.alpha, 
        'mu': args.mu 
    }
    
    result = train_engine(args.optimizer, hp, args.epochs, args.device, args.metric, args.seed)
    print(f"Final Result (Best Valid Loss): {result}")