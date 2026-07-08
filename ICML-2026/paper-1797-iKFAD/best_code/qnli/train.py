# train.py

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

from optimizers import Cadam, cubic_damping_opt, iKFAD
from dataloader import get_dataloaders
from model import get_model

def compute_accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for batch in data_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs['logits']
            _, predicted_labels = torch.max(logits, dim=1) 
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
    return correct_pred.float()/num_examples * 100

def compute_loss(model, data_loader, device):
    model.eval()
    loss_buffer = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
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
        return Cadam(params, h=hp['h'], gamma=hp['gamma'], c=hp['c'], 
                     alpha=hp['alpha'], eps=1e-8, device=device)
    elif optim_name == 'cd':
        print('h: ', hp['h'], 'gamma: ', hp['gamma'], 'c: ', hp['c'])
        return cubic_damping_opt(params, h=hp['h'], gamma=hp['gamma'], c=hp['c'], device=device)
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
    print(f"Loading data for QNLI...", flush=True)
    train_loader, valid_loader, test_loader = get_dataloaders(batch_size=16)
    
    print(f"Initializing model for QNLI...", flush=True)
    model = get_model(device)
    
    # 3. Optimizer
    print(f"Initializing optimizer: {optim_name}", flush=True)
    optimizer = get_optimizer(model, optim_name, hp, device)
    
    # Checkpoint setup
    checkpoint_dir = os.path.join('qnli', 'checkpoints', optim_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Track best validation loss
    best_val_loss = float('inf')

    # 4. Epoch 0 Log (Pre-training)
    valid_loss_0 = compute_loss(model, valid_loader, device)
    valid_acc_0 = compute_accuracy(model, valid_loader, device)
    print(f'Epoch 0000/{num_epochs:04d} | Step 0 | '
          f'Train Loss: N/A    | Train Acc: N/A    | '
          f'Valid Loss: {valid_loss_0:.4f} | Valid Acc: {valid_acc_0:.2f}% | '
          f'Time: 0.00 min', flush=True)

    # 5. Loop
    print("Starting training...", flush=True)
    global_step = 0  # <--- ADDED

    for epoch in range(num_epochs):
        model.train()
        
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0
        
        total_steps = len(train_loader)
        log_interval = max(1, total_steps // 10)
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            train_loss = outputs['loss']
            
            # Step Logic
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Accumulate metrics
            batch_size = labels.size(0)
            train_loss_accum += train_loss.item() * batch_size
            
            logits = outputs['logits']
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += batch_size

            global_step += 1  # <--- ADDED

            # Log every 10th of epoch (but using global step)
            if global_step % log_interval == 0:  # <--- CHANGED
                cur_train_loss = train_loss_accum / train_total
                cur_train_acc = 100.0 * train_correct / train_total
                
                # Check validation metrics for the log
                cur_valid_loss = compute_loss(model, valid_loader, device)
                cur_valid_acc = compute_accuracy(model, valid_loader, device)
                model.train() # Return to training mode
                
                print(f'Epoch {epoch+1}/{num_epochs} | Step {global_step} | '  # <--- CHANGED
                      f'Train Loss: {cur_train_loss:.4f} | Train Acc: {cur_train_acc:.2f}% | '
                      f'Valid Loss: {cur_valid_loss:.4f} | Valid Acc: {cur_valid_acc:.2f}% | '
                      f'Time: {(time.time() - start_time)/60:.2f} min', flush=True)

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

    # Always return best_val_loss for the sweep
    return best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SST2 Train Engine')
    parser.add_argument('--optimizer', type=str, required=True, help='cadam, msgd, etc')
    parser.add_argument('--metric', type=str, default='loss', help='Unused, always val_loss')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=123)
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate / h')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
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
