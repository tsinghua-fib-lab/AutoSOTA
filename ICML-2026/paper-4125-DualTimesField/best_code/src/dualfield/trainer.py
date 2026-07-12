import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path


class DualFieldTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=1000,
        device='cuda',
        save_dir='./checkpoints',
        sparsity_lambda=0.01,
        smoothness_lambda=0.001
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sparsity_lambda = sparsity_lambda
        self.smoothness_lambda = smoothness_lambda
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=lr * 0.01)
        
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'reconstruction': [], 'sparsity': [], 'active_atoms': []}
        
    def train_epoch(self, epoch):
        self.model.train()
        self.model.set_epoch(epoch)
        
        total_loss = 0.0
        total_rec = 0.0
        total_sparsity = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            x, t = batch
            x = x.to(self.device)
            t = t.to(self.device)
            
            self.optimizer.zero_grad()
            
            output, ctf_output, dgf_output = self.model(x, t)
            
            rec_loss = F.mse_loss(output, x)
            sparsity_loss = self.model.dgf.compute_sparsity_loss()
            smoothness_loss = self.model.ctf.compute_smoothness_loss(t)
            
            loss = rec_loss + self.sparsity_lambda * sparsity_loss + self.smoothness_lambda * smoothness_loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_rec += rec_loss.item()
            total_sparsity += sparsity_loss.item()
            num_batches += 1
        
        return total_loss / num_batches, total_rec / num_batches, total_sparsity / num_batches
    
    def validate(self):
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x, t = batch
                x = x.to(self.device)
                t = t.to(self.device)
                
                output, _, _ = self.model(x, t)
                loss = F.mse_loss(output, x)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, log_interval=100):
        for epoch in range(self.epochs):
            train_loss, rec_loss, sparsity_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.scheduler.step()
            
            active_atoms = self.model.dgf.get_active_atoms()

            self.history['train_loss'].append(train_loss)
            self.history['reconstruction'].append(rec_loss)
            self.history['sparsity'].append(sparsity_loss)
            self.history['active_atoms'].append(active_atoms)
            
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, 'best.pt')
            else:
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self.save_checkpoint(epoch, 'best.pt')
            
            if (epoch + 1) % log_interval == 0:
                print(f'Epoch {epoch+1}/{self.epochs}')
                print(f'  Train Loss: {train_loss:.6f}, Rec: {rec_loss:.6f}')
                print(f'  Active Atoms: {active_atoms}')
                if val_loss is not None:
                    print(f'  Val Loss: {val_loss:.6f}')
        
        self.save_checkpoint(self.epochs, 'final.pt')
        return self.history
    
    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        return checkpoint['epoch']
