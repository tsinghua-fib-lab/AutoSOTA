# run_best_seeds.py
import sys
import os
import argparse
import time
import random
import numpy as np
import torch

# Add parent directory to path to find optimizers/ and dataloader/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from dataloader import get_dataloaders
from model import get_model

# Import functions directly from train.py to ensure consistency
sys.path.append(current_dir)
from train import compute_loss, compute_accuracy, get_optimizer

# ---------------------------------------------------------
# Single Seed Run Logic
# ---------------------------------------------------------

def run_single_seed(optim_name, hp, device, seed, num_epochs=2):
    # 1. Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    
    train_loader, _, test_loader = get_dataloaders(batch_size=128)
    
    # 3. Model & Optimizer
    model = get_model(device)
    optimizer = get_optimizer(model, optim_name, hp, device)
    criterion = torch.nn.CrossEntropyLoss() # Added criterion
    
    # 4. Settings for Evaluation
    total_batches = len(train_loader)
    
    # Set interval equal to total_batches so it only triggers at the end of the epoch
    eval_interval = total_batches 
        
    print(f"\n[Seed {seed}] Starting {num_epochs} epochs. Steps/epoch: {total_batches}. Eval interval: {eval_interval}", flush=True)
    
    seed_results = []

    for epoch in range(num_epochs):
        model.train()
        
        # Accumulators for training metrics over the interval
        interval_loss = 0.0
        interval_correct = 0
        interval_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # --- Forward ---
            logits = model(images)
            loss = criterion(logits, labels)
            
            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Accumulate Train Metrics ---
            batch_size = labels.size(0)
            interval_loss += loss.item() * batch_size
            
            _, predicted = torch.max(logits, 1)
            interval_correct += (predicted == labels).sum().item()
            interval_samples += batch_size

            # --- Evaluation Logic: Once per epoch ---
            current_step = batch_idx + 1
            if current_step % eval_interval == 0:
                
                # 1. Calculate Train Metrics (Interval Average)
                train_loss = interval_loss / interval_samples
                train_acc = (interval_correct / interval_samples) * 100.0
                
                # 2. Calculate Test Metrics
                test_loss = compute_loss(model, test_loader, device)
                test_acc = compute_accuracy(model, test_loader, device)
                
                # 3. Log
                print(f"  S{seed} | Ep {epoch+1} | Step {current_step:03d}/{total_batches} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                      f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%", flush=True)
                
                seed_results.append({
                    'seed': seed,
                    'epoch': epoch + 1,
                    'step': current_step,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc
                })
                
                # 4. Reset Interval Accumulators
                interval_loss = 0.0
                interval_correct = 0
                interval_samples = 0
                
                # Ensure model is back in train mode
                model.train()
                
    return seed_results

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViT Multi-Seed Runner')
    
    # Config
    parser.add_argument('--optimizer', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs per seed")
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=2e-5)
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
    
    # 10 Seeds (0-9)
    seeds = range(10)
    
    print(f"Starting Multi-Seed Run for Optimizer: {args.optimizer}")
    print(f"Device: {args.device} | Epochs: {args.epochs}")
    print(f"Hyperparameters: {hp}")
    print("=" * 60)
    
    start_time = time.time()
    all_results = []
    
    for s in seeds:
        seed_data = run_single_seed(args.optimizer, hp, args.device, s, args.epochs)
        all_results.extend(seed_data)
        
        # Clear GPU memory between seeds
        torch.cuda.empty_cache()

    total_time = (time.time() - start_time)/60
    print("=" * 60)
    print(f"Completed 10 seeds in {total_time:.2f} minutes.")