# train.py
"""
Modularized Single GPU Training Script (Pre-training).
Usage: python train.py config/train_gpt_small.py
"""
import os
import time
import math
import pickle
import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np
import torch
from model import GPTConfig, GPT
from config.default_config import get_default_config

# -----------------------------------------------------------------------------
# The Trainer Class
# -----------------------------------------------------------------------------
class GPTTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_system()
        self.setup_data()
        self.setup_model()
        
    def setup_system(self):
        """Configures device, seeds, and directories."""
        os.makedirs(self.config['out_dir'], exist_ok=True)
        torch.manual_seed(self.config['seed'])
        print("Seed is: ", self.config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True 
        
        self.device = self.config['device']
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config['dtype']]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        
        print(f"System: {self.device} | Precision: {self.config['dtype']}")

    def setup_data(self):
        """Loads memory-mapped datasets."""
        data_dir = os.path.join('data', self.config['dataset'])
        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        # Check for meta file (vocab size)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        self.meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.meta_vocab_size = meta['vocab_size']
            print(f"Found vocab_size = {self.meta_vocab_size} (from {meta_path})")

    def setup_model(self):
        """Initializes model, optimizer, and scaler."""
        print("Initializing model from scratch...")
        
        model_args = dict(
            n_layer=self.config['n_layer'], 
            n_head=self.config['n_head'], 
            n_embd=self.config['n_embd'], 
            block_size=self.config['block_size'],
            bias=self.config['bias'], 
            vocab_size=None, 
            dropout=self.config['dropout'], 
            flash_attn=self.config['flash_attn']
        )
        
        # Determine vocab size
        model_args['vocab_size'] = self.meta_vocab_size if self.meta_vocab_size is not None else 50304
        
        # Create Model
        gptconf = GPTConfig(**model_args)
        self.model = GPT(gptconf)
        self.model.to(self.device)
        
        # Optimizer
        opt_kwargs = self.config.get('optimizer_params', {})
        self.optimizer = self.model.configure_optimizers(
            self.config['weight_decay'],
            self.device_type,
            self.config.get('optimizer_name'),
            opt_kwargs
        )
        
        # GradScaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config['dtype'] == 'float16'))
        
        # Compile
        if self.config['compile'] and hasattr(torch, 'compile'):
            print("Compiling model...")
            self.model = torch.compile(self.model)
            
        self.model_args = model_args # Save for checkpointing

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config['block_size'], (self.config['batch_size'],))
        x = torch.stack([torch.from_numpy((data[i:i+self.config['block_size']]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.config['block_size']]).astype(np.int64)) for i in ix])
        
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['val', 'train']: 
            losses = torch.zeros(self.config['eval_iters'])
            for k in range(self.config['eval_iters']):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def save_checkpoint(self, iter_num, raw_model):
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': self.model_args,
            'iter_num': iter_num,
            'config': self.config,
        }
        print(f"Saving checkpoint to {self.config['out_dir']}")
        torch.save(checkpoint, os.path.join(self.config['out_dir'], f'ckpt_{iter_num}.pt'))

    def train(self):
        iter_num = 0
        best_val_loss = float('inf')
        t0 = time.time()
        raw_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        
        X, Y = self.get_batch('train')
        
        print("Starting training loop...")
        while True:

            # Evaluate and Checkpoint
            if iter_num % self.config['eval_interval'] == 0:
                losses = self.estimate_loss()
                if torch.isnan(losses['val']) or torch.isnan(losses['train']):
                    print(f"NaN detected at step {iter_num}. returning infinity.")
                    return float('inf')
                
                print(f"step {iter_num}: val loss {losses['val']:.4f}, train loss {losses['train']:.4f}")

                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    
                if iter_num % self.config['ckpt_interval'] == 0 and iter_num > 0:
                    self.save_checkpoint(iter_num, raw_model)

            # Forward Backward (with Gradient Accumulation)
            for _ in range(self.config['gradient_accumulation_steps']):
                with self.ctx:
                    _, loss = self.model(X, Y)
                    # scale loss for gradient accumulation
                    loss = loss / self.config['gradient_accumulation_steps'] 
                
                X, Y = self.get_batch('train') # Prefetch
                self.scaler.scale(loss).backward()

            # Clip and Step
            if self.config['grad_clip'] != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Logging
            # t1 = time.time()
            # dt = t1 - t0
            # t0 = t1
            # if iter_num % self.config['log_interval'] == 0:
            #     lossf = loss.item() * self.config['gradient_accumulation_steps']
            #     print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

            iter_num += 1
            if iter_num > self.config['max_iters']:
                break
        return best_val_loss

# -----------------------------------------------------------------------------
# 3. Main Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train GPT on a single GPU')
    parser.add_argument('config', type=str, help='Path to the python configuration file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")

    # Load configuration cleanly
    config = get_default_config()
    
    print(f"Loading configuration from {args.config}...")
    
    # Execute config file in a separate local dictionary to prevent global pollution
    user_config = {}
    with open(args.config) as f:
        exec(f.read(), {}, user_config)
    
    # Update defaults with user config
    # We only take keys that match our known configuration or basic types
    for k, v in user_config.items():
        if k in config:
            config[k] = v
        elif not k.startswith('_') and isinstance(v, (int, float, bool, str)):
            # Allow new keys if they are simple types (logging metadata)
            config[k] = v

    # Initialize and run trainer
    trainer = GPTTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()