import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any 
from logdiff.cs_classifier.metrics import MultiLabelGroupAcc

# --- Trainer: Separate CE Loss for Each Task ---
class ClassifierTrainer(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 num_classes_per_label:list[int],
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 noise_scheduler: Any = None,
                 epochs: int = 100
                 ):
        super().__init__()
        
        # Save hyperparams, ignoring model and scheduler objects
        self.save_hyperparameters(ignore=['model', 'noise_scheduler'])
        self.model = model
        self.val_acc = MultiLabelGroupAcc(num_classes_per_label)
        self.train_acc = MultiLabelGroupAcc(num_classes_per_label)
        self.num_classes_per_label = num_classes_per_label
        
        self.total_outputs = sum(num_classes_per_label)
        self.epochs = epochs
        alphas_cumprod = torch.as_tensor(noise_scheduler.alphas_cumprod, dtype=torch.float32)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.num_timesteps = len(sqrt_alphas_cumprod)


    def configure_optimizers(self):
        """
        Optimizer and scheduler configuration.
        """
        optimizer = self.hparams.optimizer(self.model.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer, T_max=self.epochs)
        return [optimizer],[{"scheduler": scheduler, "interval": "step"}]

    def _prepare_for_metrics(self, y, logits_tuple):
        """
        Recombines the separate CE outputs into the single tensor format 
        required by the custom MultiLabelGroupAcc metric.
        """
        combined_logits = torch.cat(logits_tuple, dim=1)
        
        targets = []
        for i, n_cls in enumerate(self.num_classes_per_label):
            t_one_hot = F.one_hot(y[:, i], num_classes=n_cls).float()
            targets.append(t_one_hot)
        combined_target = torch.cat(targets, dim=1)
        return combined_logits, combined_target

    def _add_noise_to_batch(self, x0: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a random forward diffusion step to the clean image x0.
        
        Returns:
            xt (torch.Tensor): Noisy image.
            t (torch.Tensor): Timestep used for noise.
        """
        B = x0.shape[0]
        device = x0.device
        
        t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)
        alpha_t = self.sqrt_alphas_cumprod[t].reshape(B, 1, 1, 1)
        one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(B, 1, 1, 1)
        epsilon = torch.randn_like(x0, device=device)
        xt = alpha_t * x0 + one_minus_alpha_t * epsilon
        return xt, t


    def training_step(self, batch, batch_idx):
        x0, y = batch['X'], batch['label']              # x0 shape: [B, C, H, W], y shape: [B, num_labels]
        xt, t = self._add_noise_to_batch(x0, y)
        combined_logits = self.model(xt, t)

        total_loss = 0.0
        for i, logits in enumerate(combined_logits):
            target = y[:, i]
            loss = F.cross_entropy(logits, target)
            total_loss += loss
            self.log(f"train_loss_task{i}", loss)
        
        combined_logits, combined_target = self._prepare_for_metrics(y, combined_logits)

        self.train_acc.update(combined_logits, combined_target)
        self.log("train_loss", total_loss)
        return total_loss

    
    def validation_step(self, batch, batch_idx):
        x0, y = batch['X'], batch['label']             
        xt, t = self._add_noise_to_batch(x0, y)

        combined_logits = self.model(xt, t)

        total_loss = 0.0
        for i, logits in enumerate(combined_logits):
            target = y[:, i]
            loss = F.cross_entropy(logits, target)
            total_loss += loss
            self.log(f"val_loss_task{i}", loss)
        
        combined_logits, combined_target = self._prepare_for_metrics(y, combined_logits)

        self.val_acc.update(combined_logits, combined_target)   
        return total_loss
    

    def on_validation_epoch_end(self):
        for i in range(len(self.num_classes_per_label)):
            self.log(f'val_accuracy_{i}',self.val_acc[i].compute(), prog_bar=True,on_epoch=True,sync_dist=True)
        self.val_acc.reset()
        return {}

    def on_train_epoch_end(self):
        for i in range(len(self.num_classes_per_label)):
            self.log(f'train_accuracy_{i}',self.train_acc[i].compute(),on_epoch=True,sync_dist=True)
        self.train_acc.reset()
        return {}

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = self.model.state_dict()
        return checkpoint