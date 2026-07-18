import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from diffusers import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from typing import Optional, List
from logdiff.metrics import JSD, CS, R2, Quality

def prepare_and_query(y,null_token,guidance_scale):
    B,Q = y.size()
    query = null_token.repeat(Q,1).to(dtype=y.dtype,device=y.device)
    query = query.reshape(B,Q,Q)
    for i in range(Q):
        query[:,i,i] = y[:,i]
    guidance_scale = [guidance_scale]*Q
    return query,guidance_scale


class ANDTrackerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def config(self):
        return self.model.config

    @torch.no_grad()
    def forward(self, xt, t, query, guidance_dict=None, null_token=None, ll_state=None, scheduler=None, **kwargs):
        x0_uncond_noise = self.model(xt, t, null_token)
        if query is None:
            return x0_uncond_noise, None

        if isinstance(guidance_dict, dict):
            guidance_scale = guidance_dict.get("atom", 1.0)
        else:
            guidance_scale = 1.0

        prepared_query, guidance_scales = prepare_and_query(query, null_token, guidance_scale)

        composed_noise = x0_uncond_noise
        for i, scale in enumerate(guidance_scales):
            x_cond_noise = self.model(xt, t, prepared_query[:, i, :])
            composed_noise = composed_noise + scale * (x_cond_noise - x0_uncond_noise)

        return composed_noise, None
    

class JSDTracker(Callback):
    def __init__(self, num_classes_per_label: int,log_interval: int = 1,):
            super().__init__()
            self.log_interval = log_interval
            self.num_classes_per_label = num_classes_per_label
    def setup(self, trainer, pl_module, stage):
        if stage == 'fit':            
            self.jsd_metric = JSD(self.num_classes_per_label).to(pl_module.device)
    def on_validation_batch_start(self,
                                  trainer: pl.Trainer,
                                  pl_module: pl.LightningModule,
                                  batch: torch.Tensor,
                                  batch_idx: int,
                                  dataloader_idx: int =0):
        if batch_idx ==0  and (trainer.current_epoch +1)% self.log_interval == 0:
            self.jsd_metric.update(batch, pl_module.model,pl_module.noise_scheduler) 
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch +1)% self.log_interval == 0:
            for metric_name, metric in self.jsd_metric.compute().items():
                pl_module.log(f'val/{metric_name}', metric, prog_bar=True,on_epoch=True,sync_dist=True)
            self.jsd_metric.reset() 


class GenerationMetrics(Callback):
    def __init__(self, 
                    sampling_pipe: DiffusionPipeline,
                    num_inference_steps: int = 50,
                    vae: Optional[AutoencoderKL] = None,
                    log_interval: int = 1,
                    metrics: List[str] = ["quality","cs","r2"],
                    num_classes_per_label: List[int] = [10,10],
                    guidance_scale: float = 7.5,
                    output_dir: Optional[str] = None,
                    classifier: Optional[nn.Module] = None,
                    classifer_checkpoint: Optional[str] = None
                   ):
            super().__init__()
            if "quality" in metrics and output_dir is None:
                raise ValueError("Output directory is required")
            if "cs" in metrics and classifier is None:
                raise ValueError("Classifier checkpoint is required")
            self.sampling_pipe_partial = sampling_pipe
            self.num_inference_steps = num_inference_steps
            self.output_dir = output_dir
            self.log_interval = log_interval
            self.vae = vae
            self.guidance_scale = guidance_scale
            self.num_classes_per_label = num_classes_per_label
            if classifier is not None:
                self.model = classifier
                self.model.load_state_dict(torch.load(classifer_checkpoint, weights_only=False)["state_dict"])
            self.metrics = metrics

    def setup(self, trainer, pl_module, stage):
        if trainer.is_global_zero:
            os.makedirs(self.output_dir, exist_ok=True)
        trainer.strategy.barrier()

        if stage == 'fit':     
            if self.vae is not None:
                self.vae = self.vae.to(pl_module.device)
                output_shape = (self.vae.config.in_channels,self.vae.config.sample_size,self.vae.config.sample_size)
            else:
                output_shape = (pl_module.model.config.in_channels,pl_module.model.config.sample_size,pl_module.model.config.sample_size)

            net = ANDTrackerWrapper(pl_module.model)

            self.sampling_pipe = self.sampling_pipe_partial(net=net, scheduler=pl_module.noise_scheduler, vae=self.vae)
            
            if "cs" in self.metrics:
                self.cs_metric_logger = {
                    "train": CS(classifier = self.model).to(pl_module.device),
                    "val": CS(classifier = self.model).to(pl_module.device)
                }
            if "r2" in self.metrics:
                self.r2_metric_logger = {
                    "train": R2(output_shape).to(pl_module.device),
                    "val": R2(output_shape).to(pl_module.device)
                }
            if "quality" in self.metrics:
                self.quality = {
                    "train": Quality(save_dir=f"{self.output_dir}/train",save_style='grid'),
                    "val": Quality(save_dir=f"{self.output_dir}/val",save_style='grid')
                }
 
    def on_train_batch_start(self,
                                  trainer: pl.Trainer,
                                  pl_module: pl.LightningModule,
                                  batch: torch.Tensor,
                                  batch_idx: int,
                                  dataloader_idx: int =0):
        if batch_idx ==0 and (trainer.current_epoch +1)% self.log_interval == 0:
            pl_module.model.eval()
            self.and_metric_tracking(batch,epoch=trainer.current_epoch,state="train")
            pl_module.model.train()
        
    def on_validation_batch_start(self,
                                  trainer: pl.Trainer,
                                  pl_module: pl.LightningModule,
                                  batch: torch.Tensor,
                                  batch_idx: int,
                                  dataloader_idx: int =0):
        if batch_idx ==0  and (trainer.current_epoch +1)% self.log_interval == 0:
            self.and_metric_tracking(batch,epoch=trainer.current_epoch,state="val")
    
    def and_metric_tracking(self, batch, epoch,state):
        """
        We focus on AND composition for tracking the training metrics
        """
        x, y, null_token = batch["X"], batch["label"], batch["label_null"]    
        generated_images = self.sampling_pipe(batch_size=x.size(0), 
                    num_inference_steps=self.num_inference_steps,
                    return_dict=True,
                    use_clipped_model_output=True,
                    query=y,
                    guidance_dict={"atom": self.guidance_scale},
                    null_token=null_token).images
        query, guidance_scale = prepare_and_query(y, null_token, self.guidance_scale)
        
        if self.vae is not None:
            with torch.no_grad():
                x = self.vae.decode(x[:8]/ self.vae.config.scaling_factor)[0]
                x = (x.clamp(-1, 1) + 1) / 2 
        if "quality" in self.metrics:
            self.quality[state].update(generated_images,x)
        if "cs" in self.metrics:    
            self.cs_metric_logger[state].update(generated_images,query,null_token,guidance_scale)
        if "r2" in self.metrics:
            self.r2_metric_logger[state].update(generated_images,x)
        return 
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch +1)% self.log_interval == 0:
            if "cs" in self.metrics:
                for metric_name, metric in self.cs_metric_logger["val"].compute().items():
                    pl_module.log(f'val/{metric_name}', metric, prog_bar=True,on_epoch=True,sync_dist=True)
                self.cs_metric_logger["val"].reset()
            if "r2" in self.metrics:
                for metric_name, metric in self.r2_metric_logger["val"].compute().items():
                    pl_module.log(f'val/{metric_name}', metric, prog_bar=True,on_epoch=True,sync_dist=True)
                self.r2_metric_logger["val"].reset()

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch +1)% self.log_interval == 0:
            if "cs" in self.metrics:
                for metric_name, metric in self.cs_metric_logger["train"].compute().items():
                    pl_module.log(f'train/{metric_name}', metric, prog_bar=True,on_epoch=True,sync_dist=True)
                self.cs_metric_logger["train"].reset()
            if "r2" in self.metrics:
                for metric_name, metric in self.r2_metric_logger["train"].compute().items():
                    pl_module.log(f'train/{metric_name}', metric, prog_bar=True,on_epoch=True,sync_dist=True)
                self.r2_metric_logger["train"].reset()

