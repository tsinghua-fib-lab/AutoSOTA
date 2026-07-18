"""
Toy Diffusion Model Training Script with Configuration Management

This script trains a diffusion model on various 2D toy datasets with a flexible
configuration system that allows easy switching between datasets and hyperparameters.

Usage Examples:
1. Train with default checkerboard dataset:
   config = get_preset_config('checkerboard')

2. Train with swissroll dataset:
   config = get_preset_config('swissroll')

3. Train with custom parameters:
   config = get_preset_config('checkerboard', num_epochs=1000, learning_rate=5e-4)

4. Load a previously saved experiment:
   config, model, data_dict, optimizer, scheduler = load_model_and_data('outputs_checkerboard_20250807_123456')

5. Continue training from a checkpoint:
   config, model, final_loss = continue_training_from_checkpoint('outputs_checkerboard_20250807_123456', additional_epochs=200)

6. Create dataloader from loaded data:
   dataloader = create_dataloader_from_loaded_data(data_dict, batch_size=512)

Supported Datasets:
- 'checkerboard': Checkerboard pattern with configurable squares and variance
- 'swissroll': Swiss roll manifold with configurable curl and noise
- '3gm': Three Gaussian mixture
- 'thin_3gm': Thin three Gaussian mixture
- 'any_gm': Custom Gaussian mixture with configurable components
- 'single_point': Single point dataset

Output Structure:
- config.json: All hyperparameters and dataset configuration
- trained_model.pth: Model state dict, optimizer state, and training metadata
- dataset.pkl: Generated dataset and related information

Key Functions:
- get_preset_config(): Create configuration with preset defaults
- create_dataset(): Generate dataset based on configuration
- load_model_and_data(): Load saved model, config, and data
"""

import os
from datetime import datetime
import json
import pickle
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from TSR_toy.model.toy_model import ToyModel
from TSR_toy.sampler.toy_samplers import T, beta, alpha, alpha_bar
from TSR_toy.data.toy_data import create_dataset


device = torch.device("cuda:0")

def get_preset_config(**kwargs):
    """
    Get a preset configuration for different datasets.
    
        **kwargs: Additional parameters to override defaults
    
    Returns:
        dict: Configuration dictionary
    """
    base_config = {
        # Training hyperparameters
        'pred_type': 'flow',
        'batch_size': 1000,
        'num_epochs': 500,
        'learning_rate': 1e-3,
        'hidden_dim': 128,
        'pos_emb_dim': 64,
        't_emb_dim': 32,
        'cond_emb_dim': 0,
        'num_classes': 1,
        'cond_drop_prob': 0.0,
        'n_mlp_layers': 8,
        'n_resblocks': 10,
        
        # Dataset configuration
        'dataset_name': '1d_gm',
        'num_data_points': 100000,
        'data_dim': 2,  # Add data dimension parameter
        
        # Dataset-specific parameters
        'checkerboard_params': {
            'num_squares': 6,
        },
        'swissroll_params': {
            'curly_factor': 2.0,
            'noise': 0.5
        },
        '3gm_params': {},
        'thin_3gm_params': {},
        '6gm_params': {
            'n_components': 6,
            'means': [
                        [-0.5, 0], [-1, 1], [-1, -1],
                        [0.5, 0], [1, 1], [1, -1]
                    ],
            'variances': 0.02,
            'weights': [1, 1, 1, 1, 1, 1],
        },
        '6gm_horizontal_params': {
            'n_components': 6,
            'means': [
                        [-2.5, 0], [-1.5, 0], [-0.5, 0], [0.5, 0], [1.5, 0], [2.5, 0]
                    ],
            'variances': 0.02,
            'weights': [1, 1, 1, 1, 1, 1],
        },
        '1d_gm_params': {
            'mus': [-1.5, 0.0, 1.5],
            'variances': 0.1,
            'weights': [1, 1, 1],
        },
        'single_point_params': {
            'point': [0, 0]
        },
        
        # Training metadata
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }
    
    # Override with any provided kwargs
    base_config.update(kwargs)
    return base_config



def load_model_and_data(output_dir, device=None):
    """
    Load a previously saved model, configuration, and dataset.
    
    Args:
        output_dir: Directory containing the saved files (config.json, trained_model.pth, dataset.pkl)
        device: Device to load the model on. If None, uses the device from config or defaults to cuda:0
    
    Returns:
        tuple: (config, model, data_dict, optimizer, scheduler)
            - config: Configuration dictionary
            - model: Loaded model with weights
            - data_dict: Dictionary containing dataset and metadata
            - optimizer: Optimizer with loaded state (optional)
            - scheduler: Scheduler with loaded state (optional)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config_path = os.path.join(output_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loading experiment from: {output_dir}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Training timestamp: {config.get('timestamp', 'Unknown')}")
    
    # Load model checkpoint
    model_path = os.path.join(output_dir, 'trained_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model with same architecture
    model = ToyModel(
        betas=beta.to(device),
        alphas=alpha.to(device),
        alpha_bars=alpha_bar.to(device),
        pred_type=config['pred_type'],
        data_dim=config.get('data_dim', 2),  # Use data_dim from config, default to 2
        n_layers=config['n_mlp_layers'],
        n_resblocks=config['n_resblocks'],
        hidden_dim=config['hidden_dim'],
        pos_emb_dim=config['pos_emb_dim'],
        t_emb_dim=config['t_emb_dim'],
        cond_emb_dim=config['cond_emb_dim'],
        num_classes=config['num_classes'],
        cond_drop_prob=config['cond_drop_prob'],
        device=device
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Optionally recreate optimizer and scheduler
    optimizer = None
    scheduler = None
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'scheduler_state_dict' in checkpoint:
        iter_per_epoch = config.get('iter_per_epoch', 1)  # Default fallback
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer if optimizer else torch.optim.Adam(model.parameters()), 
            T_max=config['num_epochs'] * iter_per_epoch, 
            eta_min=1e-6
        )
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load dataset
    data_path = os.path.join(output_dir, 'dataset.pkl')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Convert numpy arrays back to tensors on the correct device
    if 'data' in data_dict:
        data_dict['data_tensor'] = torch.tensor(data_dict['data'], device=device, dtype=torch.float32)
    if 'data_samples' in data_dict:
        data_dict['data_samples_tensor'] = torch.tensor(data_dict['data_samples'], device=device, dtype=torch.float32)
    
    final_loss = checkpoint.get('final_loss', 'N/A')
    print(f"Model loaded successfully!")
    print(f"Final training loss: {final_loss}")
    print(f"Data shape: {data_dict['data'].shape}")
    print(f"Model on device: {next(model.parameters()).device}")
    
    return config, model, data_dict, optimizer, scheduler


def main():
    
    parser = argparse.ArgumentParser(description="Training diffusion/flow model for Toy data")
    
    parser.add_argument("--data", type=str, 
                        choices=["checkerboard", "swissroll"],
                        required=True, help="Type of Supported Toy Data")

    parser.add_argument("--save_dir", type=str, 
                        default="./output",
                        help="Directory for saved ckpts and results")
    
    parser.add_argument("--model_type", type=str, 
                        choices=["flow", "diffusion"],
                        help="Type of Supported Models")
    
    args = parser.parse_args()
    
    config= get_preset_config(dataset_name = args.data, pred_type=args.model_type)

    # Create dataset based on configuration
    data, labels, xy_lims, initweight_str, gm = create_dataset(config, device)
    data_samples = data[:10000]

    dataset = TensorDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    iter_per_epoch = len(data_loader)

    model = ToyModel(
        betas=beta.to(device),
        alphas=alpha.to(device),
        alpha_bars=alpha_bar.to(device),
        pred_type=config['pred_type'],
        data_dim=config['data_dim'],
        n_layers=config['n_mlp_layers'],
        n_resblocks=config['n_resblocks'],
        hidden_dim=config['hidden_dim'],
        pos_emb_dim=config['pos_emb_dim'],
        t_emb_dim=config['t_emb_dim'],
        cond_emb_dim=config['cond_emb_dim'],
        num_classes=config['num_classes'],
        cond_drop_prob=config['cond_drop_prob'],
        device=device,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'] * iter_per_epoch, eta_min=1e-6)

    for epoch in range(config['num_epochs']):
        epoch_loss = 0.0
        for iter, data_batch in enumerate(data_loader):
            x_0, cond = data_batch
            x_0 = x_0.to(device)
            cond = cond.to(device)

            if config['pred_type'] == 'flow':
                t = torch.rand(size=(), device=device).item()
            else:
                t = torch.randint(low=0, high=T, size=(), device=device).item()

            x_t, target, _ = model.add_noise(x_0, t)
            model_output = model(x_t, t, cond)
            loss = nn.functional.mse_loss(model_output, target)  # Scale loss by 1 - alpha_bar_t
            # print(f"iter {iter+1 + epoch * len(data_loader)}, loss: {loss.item():.6f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1} completed. Avg loss: {avg_loss:.6f}")

    # Create output directory
    output_dir = f"{args.save_dir}/{config['dataset_name']}_{config['pred_type']}"
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")

    # Save trained model
    model_path = os.path.join(output_dir, 'trained_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_loss': avg_loss,
        'config': config
    }, model_path)
    print(f"Trained model saved to: {model_path}")

    # Save dataset and related data
    data_path = os.path.join(output_dir, 'dataset.pkl')
    data_dict = {
        'data': data.cpu().numpy(),
        'data_samples': data_samples.cpu().numpy(),
        'xy_lims': xy_lims,
        'initweight_str': initweight_str,
        'gm': gm,
        'dataset_name': config['dataset_name']
    }
    with open(data_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Dataset saved to: {data_path}")

    print(f"\nAll outputs saved in directory: {output_dir}")
    print(f"- Configuration: config.json")
    print(f"- Trained model: trained_model.pth")
    print(f"- Dataset: dataset.pkl")

if __name__ == "__main__":
    main()