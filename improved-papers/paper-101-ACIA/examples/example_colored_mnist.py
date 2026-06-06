"""
Example: Training on ColoredMNIST
==================================

This example demonstrates how to train a causal representation model
on the ColoredMNIST dataset, where color spuriously correlates with
digit parity in different ways across environments.

Environment 1: P(red|even) = 0.75
Environment 2: P(red|even) = 0.25

The model learns to ignore color and focus on digit shape.
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset
from acia.datasets import ColoredMNIST, visualize_cmnist_results
from acia.models import CausalRepresentationNetwork
from acia.models.training import CausalOptimizer


def main():
    # Configuration
    config = {
        'batch_size': 128,
        'learning_rate': 1e-4,
        'n_epochs': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Create datasets
    print("Loading ColoredMNIST datasets...")
    train_e1 = ColoredMNIST(env='e1', root='./data', train=True)
    train_e2 = ColoredMNIST(env='e2', root='./data', train=True)
    test_e1 = ColoredMNIST(env='e1', root='./data', train=False)
    test_e2 = ColoredMNIST(env='e2', root='./data', train=False)
    
    # Combine training environments
    train_data = ConcatDataset([train_e1, train_e2])
    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Test loaders for both environments
    test_loader_e1 = DataLoader(test_e1, batch_size=config['batch_size'], shuffle=False)
    test_loader_e2 = DataLoader(test_e2, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = CausalRepresentationNetwork().to(config['device'])
    optimizer = CausalOptimizer(
        model,
        batch_size=config['batch_size'],
        lr=config['learning_rate']
    )
    
    # Training loop
    print("\nStarting training...")
    history = {
        'train_loss': [],
        'R1': [],
        'R2': [],
        'train_acc': [],
        'test_acc_e1': [],
        'test_acc_e2': []
    }
    
    for epoch in range(config['n_epochs']):
        # Training
        model.train()
        epoch_metrics = {'pred_loss': [], 'R1': [], 'R2': [], 'train_acc': []}
        
        for batch_idx, (x, y, e) in enumerate(train_loader):
            x = x.to(config['device'])
            y = y.to(config['device'])
            e = e.to(config['device'])
            
            # Training step
            metrics = optimizer.train_step(x, y, e)
            
            # Track metrics
            epoch_metrics['pred_loss'].append(metrics['pred_loss'])
            epoch_metrics['R1'].append(metrics['R1'])
            epoch_metrics['R2'].append(metrics['R2'])
            
            # Compute accuracy
            with torch.no_grad():
                _, _, logits = model(x)
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean().item()
                epoch_metrics['train_acc'].append(acc)
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{config['n_epochs']} - "
                      f"Batch {batch_idx}/{len(train_loader)} - "
                      f"Loss: {metrics['pred_loss']:.4f} - "
                      f"R1: {metrics['R1']:.4f} - "
                      f"R2: {metrics['R2']:.4f}")
        
        # Evaluation on both test environments
        model.eval()
        test_acc_e1 = evaluate(model, test_loader_e1, config['device'])
        test_acc_e2 = evaluate(model, test_loader_e2, config['device'])
        
        # Store epoch metrics
        history['train_loss'].append(sum(epoch_metrics['pred_loss']) / len(epoch_metrics['pred_loss']))
        history['R1'].append(sum(epoch_metrics['R1']) / len(epoch_metrics['R1']))
        history['R2'].append(sum(epoch_metrics['R2']) / len(epoch_metrics['R2']))
        history['train_acc'].append(sum(epoch_metrics['train_acc']) / len(epoch_metrics['train_acc']))
        history['test_acc_e1'].append(test_acc_e1)
        history['test_acc_e2'].append(test_acc_e2)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Train Acc: {history['train_acc'][-1]*100:.2f}%")
        print(f"  Test Acc (E1): {test_acc_e1*100:.2f}%")
        print(f"  Test Acc (E2): {test_acc_e2*100:.2f}%")
        print(f"  R1: {history['R1'][-1]:.4f}")
        print(f"  R2: {history['R2'][-1]:.4f}\n")
    
    # Save model
    print("Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, 'coloredmnist_model.pt')
    
    # Visualize results
    print("Generating visualizations...")
    visualize_cmnist_results(
        model,
        test_loader_e1,
        config['device'],
        'coloredmnist_results.png'
    )
    
    print("Training complete!")
    print(f"Final Test Accuracy (E1): {history['test_acc_e1'][-1]*100:.2f}%")
    print(f"Final Test Accuracy (E2): {history['test_acc_e2'][-1]*100:.2f}%")


def evaluate(model, loader, device):
    """Evaluate model on a test set."""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y, e in loader:
            x = x.to(device)
            y = y.to(device)
            
            _, _, logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total


if __name__ == '__main__':
    main()
