import torch
from torch import nn as nn
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from torch.nn import functional as F
from typing import  List, Optional
from diffusers import AutoencoderKL

class ComposableDiffusion(pl.LightningModule):

    def __init__(self, 
                model,
                noise_scheduler: DDPMScheduler,           
                vae: Optional[AutoencoderKL] = None,
                lambda_coind: float = 0.0,
                null_token: Optional[List] = None,
                  **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model','noise_scheduler','sampling_pipe','vae'])
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.lambda_coind = lambda_coind
        self.vae = vae
        if self.vae is not None:
            for param in self.vae.parameters():
                param.requires_grad = False

        self.coind_loss_type = kwargs.get('coind_loss_type',None) #regular or theoritical
        self.coind_masking = kwargs.get('coind_masking','pairwise') #pairwise or one
        self.p_null_mask = kwargs.get("p_null_mask",0.2) #ideal 0.2,0.3 any one is fine
        self.null_token = null_token
    
    def prepare_labels(self, y, y_null):
        """
        y: (batch_size, num_cols)
        y_null: (batch_size, num_cols)
        returns: [(batch_size, num_cols), (batch_size*4, num_cols)]
            y_diffusion_obj: [x% of batch with y and 1-x% of y_null] or [x% of (batch_size, num_cols) with y[i] and 1-x% of y_null[i]]
            y_coind_obj: [y,y_null,y_i,y_-i]
        """
        p_null = self.p_null_mask
        batch_size, num_cols = y.size()

        is_unconditional = torch.rand(batch_size, 1, device=y.device) < p_null
        y_single_feature = y_null.clone()
        feature_to_keep = torch.randint(
            0, num_cols, (batch_size, 1), device=y.device
        )
        feature_mask = torch.zeros(batch_size, num_cols, dtype=torch.bool, device=y.device)
        feature_mask.scatter_(dim=1, index=feature_to_keep, value=True)
        y_single_feature[feature_mask] = y[feature_mask]
        y_diffusion_obj = torch.where(is_unconditional, y_null, y_single_feature)
            
        #this is i,-i setting for 2 columns everything is same but difference is for multi column values.
        batch_size, num_cols = y.size()
        all_y_idx = torch.arange(num_cols).repeat(batch_size,1)    
        y_indices = torch.argsort(torch.rand(*all_y_idx.shape[:2]), dim=1)
        y_idx = torch.gather(all_y_idx, dim=-1, index=y_indices)[:,:2]
        x_idx = torch.arange(batch_size*4)
        y_null = y_null.repeat(4,1)
        y_coind_obj = y.clone().repeat(4,1)
        if self.coind_masking == 'pairwise':  #for two variables pairwise = one 
            y_coind_obj[x_idx[:batch_size],y_idx[:batch_size,0]] = y_null[x_idx[:batch_size],y_idx[:batch_size,0]]
            y_coind_obj[x_idx[batch_size:batch_size*3],:] = y_null[x_idx[:batch_size*2],:]
            y_coind_obj[x_idx[batch_size:batch_size*2],y_idx[:batch_size,0]] = y[x_idx[:batch_size],y_idx[:batch_size,0]]
        elif self.coind_masking == 'one':
            y_coind_obj[x_idx[:batch_size],y_idx[:batch_size,0]] = y_null[x_idx[:batch_size],y_idx[:batch_size,0]]
            y_coind_obj[x_idx[batch_size:batch_size*3],:] = y_null[x_idx[:batch_size*2],:]
            y_coind_obj[x_idx[batch_size:batch_size*2],y_idx[:batch_size,0]] = y[x_idx[:batch_size],y_idx[:batch_size,0]]    
        else:
            raise NotImplementedError
        return y_diffusion_obj,y_coind_obj

    
    def training_step(self, batch, batch_idx):
        x0, y = batch['X'], batch['label']
        if 'null_label' in batch.keys():
            y_null = batch['null_label']
        else:
            y_null = torch.Tensor(self.null_token).repeat(x0.shape[0], 1).long().to(x0.device)

        if self.vae is not None:
            x0 = self.vae.encode(x0)[0].mode()
            x0 = x0 * self.vae.config.scaling_factor
            
        y_diffusion_obj, y_coind_obj = self.prepare_labels(y, y_null)

        noise = torch.randn_like(x0)
        batch_size = x0.size(0)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()

        xt = self.noise_scheduler.add_noise(x0, noise, timesteps)
        noise_pred = self.model(xt, timesteps,y_diffusion_obj)
        l_diffusion = F.mse_loss(noise_pred, noise)
        l_coind = 0.0

        ################### INDEPENDENCE ############################

        batch_size, num_cols = y.size()
        num_dim = len(xt.shape)
        xt = xt.repeat(4,*(1 for _ in range(num_dim-1)))  #xt = xt.repeat(4,1,1,1) #replaced 4 with 2
        timesteps = timesteps.repeat(4)

        if self.lambda_coind > 0.0:    
            noise_pred_new = self.model(xt, timesteps,y_coind_obj).chunk(4,dim=0)
            l_coind = F.mse_loss(noise_pred_new[0]+noise_pred_new[1], noise_pred_new[2]+noise_pred_new[3])
            if self.coind_loss_type == 'theoritical':
                l = torch.sqrt(l_diffusion) + self.lambda_coind*torch.sqrt(l_coind)
            else:
                l = l_diffusion + self.lambda_coind*l_coind
        else:
            with torch.no_grad():
                noise_pred_new = self.model(xt, timesteps,y_coind_obj).chunk(4,dim=0)
                l_coind = F.mse_loss(noise_pred_new[0]+noise_pred_new[1], noise_pred_new[2]+noise_pred_new[3])
            l = l_diffusion
        
        ################### END INDEPENDENCE ############################
        self.log_dict({'diffusion_loss': l_diffusion, 'train_loss': l,  'coind_loss':l_coind}, prog_bar=True,on_epoch=True,sync_dist=True)
        return {'loss': l}
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.model.parameters())
        scheduler = self.hparams.scheduler(optimizer)
        return [optimizer],[{"scheduler": scheduler, "interval": "step"}]
    
    def validation_step(self,batch,batch_idx):
        x0, y = batch['X'], batch['label']
        if 'null_label' in batch.keys():
            y_null = batch['null_label']
        else:
            y_null = torch.Tensor(self.null_token).repeat(x0.shape[0], 1).long().to(x0.device)

        if self.vae is not None:
            x0 = self.vae.encode(x0)[0].mode()
            x0 = x0 * self.vae.config.scaling_factor

        y_diffusion_obj, y_coind_obj = self.prepare_labels(y, y_null)

        noise = torch.randn_like(x0)
        batch_size = x0.size(0)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        xt = self.noise_scheduler.add_noise(x0, noise, timesteps)
        noise_pred = self.model(xt, timesteps, y_diffusion_obj)

        with torch.no_grad():
            l_diffusion  = F.mse_loss(noise_pred, noise)
            batch_size, num_cols = y.size()
            num_dim = len(xt.shape)
            xt = xt.repeat(4,*(1 for _ in range(num_dim-1)))  #xt = xt.repeat(4,1,1,1) #replaced 4 with 2
            timesteps = timesteps.repeat(4)
            noise_pred_new = self.model(xt, timesteps,y_coind_obj).chunk(4,dim=0)
            l_coind = F.mse_loss(noise_pred_new[0]+noise_pred_new[1], noise_pred_new[2]+noise_pred_new[3])
            if self.coind_loss_type == 'theoritical':
                l = torch.sqrt(l_diffusion) + self.lambda_coind*torch.sqrt(l_coind)
            else:
                l = l_diffusion + self.lambda_coind*l_coind
        
        self.log_dict({'diffusion_loss': l_diffusion, 'val_loss': l,  'coind_loss':l_coind}, prog_bar=True,on_epoch=True,sync_dist=True)
        return {'loss': l}


class Lace(ComposableDiffusion):

    def training_step(self, batch, batch_idx):
        
        x0,y,y_null = batch['X'],batch['label'],batch['label_null']

        if self.vae is not None:
            x0 = self.vae.encode(x0)[0].mode()
            x0 = x0 * self.vae.config.scaling_factor
            
        
        y_diffusion_obj,y_coind_obj = self.prepare_labels(y,y_null)

        noise = torch.randn_like(x0)
        batch_size = x0.size(0)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()

        xt = self.noise_scheduler.add_noise(x0, noise, timesteps)
        
        noise_pred = self.model(xt, timesteps,y_diffusion_obj)
        noise = noise.unsqueeze(1).repeat(1,2,*(1 for _ in range(len(noise_pred.shape)-2)))
        train_loss  = F.mse_loss(noise_pred, noise)

        self.log_dict({'train_loss': train_loss}, prog_bar=True,on_epoch=True,sync_dist=True)
        return {'loss': train_loss}
    
    def validation_step(self,batch,batch_idx):
        x0,y,y_null = batch['X'],batch['label'],batch['label_null']

        if self.vae is not None:
            x0 = self.vae.encode(x0)[0].mode()
            x0 = x0 * self.vae.config.scaling_factor

        y_diffusion_obj,y_coind_obj = self.prepare_labels(y,y_null)

        noise = torch.randn_like(x0)
        batch_size = x0.size(0)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        xt = self.noise_scheduler.add_noise(x0, noise, timesteps)

        noise_pred = self.model(xt, timesteps,y_diffusion_obj)
        noise = noise.unsqueeze(1).repeat(1,2,*(1 for _ in range(len(noise_pred.shape)-2)))
        l_diffusion  = F.mse_loss(noise_pred, noise)
        
        self.log_dict({'val_loss': l_diffusion}, prog_bar=True,on_epoch=True,sync_dist=True)
        return {'loss': l_diffusion}

