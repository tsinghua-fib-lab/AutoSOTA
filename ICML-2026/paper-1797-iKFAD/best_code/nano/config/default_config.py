# config/default_config.py
import torch

def get_default_config():
    return {
        # I/O
        'out_dir': None,
        'eval_interval': None,
        'log_interval': None,
        'eval_iters': None,
        'eval_only': None,
        'always_save_checkpoint': None,
        'ckpt_interval': None,
        
        # Data
        'dataset': None,
        'gradient_accumulation_steps': None,
        'batch_size': None,
        'block_size': None,
        
        # Model
        'n_layer': None,
        'n_head': None,
        'n_embd': None,
        'dropout': None,
        'bias': None,
        'flash_attn': None,
        
        # Optimizer
        'optimizer_name': 'adamw', # 'adamw', 'sgd', etc.
        'learning_rate': None,
        'max_iters': None,
        'weight_decay': None,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': None,
        
        # Scheduler
        'decay_lr': None,
        'warmup_iters': None,
        'lr_decay_iters': None,
        'min_lr': None,
        
        # System
        'device': 'cuda',
        'dtype': None,
        'compile': None,
        'seed': None,
    }
