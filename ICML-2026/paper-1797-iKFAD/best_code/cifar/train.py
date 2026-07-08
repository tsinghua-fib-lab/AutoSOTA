import sys
import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Robustly add parent directory to path to find optimizers/ regardless of execution dir
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from optimizers import Cadam, cubic_damping_opt, iKFAD
from dataloader import get_dataloaders
from model import get_model

def compute_accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        total_correct  = 0
        total_num_data = 0
        for batch in data_loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()    
            total_num_data += inputs.size(0)
    return 100 * total_correct/total_num_data

def compute_loss(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_num_data = 0
    for batch in data_loader:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        total_num_data += inputs.size(0)
    return total_loss / total_num_data 

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
    elif optim_name == 'cd':
        print('h: ', hp['h'], 'gamma: ', hp['gamma'], 'c: ', hp['c'])
        return cubic_damping_opt(params, h=hp['h'], gamma=hp['gamma'], c=hp['c'], device=device)
    elif optim_name == 'ikfad':
        print('h: ', hp['h'], 'gamma: ', hp['gamma'], 'alpha: ', hp['alpha'], 'mu: ', hp['mu'])
        return iKFAD(params, h=hp['h'], gamma=hp['gamma'], alpha=hp['alpha'], mu=hp['mu'], device=device)
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
    print(f"Loading data for CIFAR-10...", flush=True)
    train_loader, valid_loader, test_loader = get_dataloaders(batch_size=64)
    
    print(f"Initializing model for CIFAR-10...", flush=True)
    model = get_model(device)
    
    # 3. Optimizer
    print(f"Initializing optimizer: {optim_name}", flush=True)
    optimizer = get_optimizer(model, optim_name, hp, device)
    
    # 4. Loss Function
    criterion = nn.CrossEntropyLoss()
    
    valid_losses = np.zeros(num_epochs)
    valid_accs = np.zeros(num_epochs)
    
    # Checkpoint setup
    # Create a subfolder for the specific optimizer
    checkpoint_dir = os.path.join('cifar', 'checkpoints', optim_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    best_acc = 0.0

    # 5. Loop
    print("Starting training...", flush=True)
    for epoch in range(num_epochs):
        model.train()
        
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            
            outputs = model(inputs)
            train_loss = criterion(outputs, targets)
            
            # Step Logic
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss_accum += train_loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += inputs.size(0)

        # Calculate Epoch Metrics
        avg_train_loss = train_loss_accum / train_total
        avg_train_acc = 100.0 * train_correct / train_total

        # Validation phase
        valid_loss = compute_loss(model, valid_loader, device)
        valid_acc = compute_accuracy(model, valid_loader, device)
        
        valid_losses[epoch] = valid_loss
        valid_accs[epoch]   = valid_acc
        
        # Log One Line Summary
        print(f'Epoch {epoch+1:04d}/{num_epochs:04d} | '
              f'Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | '
              f'Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}% | '
              f'Time: {(time.time() - start_time)/60:.2f} min', flush=True)

        # Save Checkpoints
        # 1. Save Last (Clean filename inside the subfolder)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss,
            'acc': valid_acc
        }, os.path.join(checkpoint_dir, 'last.pth'))

        # 2. Save Best (based on Accuracy)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'acc': best_acc
            }, os.path.join(checkpoint_dir, 'best.pth'))

    print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min', flush=True)
    print(f'Test accuracy: {compute_accuracy(model, test_loader, device):.2f}%', flush=True)
    
    del model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    if metric == 'acc':
        return valid_accs[-1]
    else:
        return valid_losses[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR Train Engine')
    parser.add_argument('--optimizer', type=str, required=True, help='cadam, msgd, etc')
    parser.add_argument('--metric', type=str, default='loss', help='loss or acc')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=123)
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate / h')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=1.0) 
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    args = parser.parse_args()
    
    hp = {
        'lr': args.lr, 
        'h': args.lr, 
        'momentum': args.momentum, 
        'betas': (args.beta1, args.beta2), 
        'weight_decay': args.weight_decay,
        'gamma': args.gamma, 
        'c': args.c, 
        'alpha': args.alpha, 
        'mu': args.mu 
    }
    
    result = train_engine(args.optimizer, hp, args.epochs, args.device, args.metric, args.seed)
    print(f"Final Result ({args.metric}): {result}")