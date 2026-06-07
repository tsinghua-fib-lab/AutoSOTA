"""Mask-guided self-supervised trainer for MASS pretraining.

The trainer builds the self-supervised data pipeline, optimizes the Iris model
with auto-generated masks as visual priors, maintains EMA weights, and can run
in-context validation on processed segmentation datasets.
"""

import os
import time
import logging
import json
from copy import deepcopy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import numpy as np

from utils.distributed import (
    is_master, 
    get_rank, 
    get_local_rank,
    get_world_size, 
    is_distributed,
    reduce_tensor
)
from utils.checkpoint import (
    save_checkpoint, 
    resume_from_checkpoint,
    prepare_checkpoint_state
)
from utils.metrics_utils import (
    AverageMeter, 
    ProgressMeter, 
    TimeMeter
)
from training.evaluator import Evaluator
from utils.registry import (
    get_optimizer, 
    get_scheduler, 
    get_criterion,
    get_dataset,
    register_trainer
)


@register_trainer("mask_guided_self_supervised")
class MaskGuidedSelfSupervisedTrainer:
    """
    Trainer class for mask-guided self-supervised training of 3D medical image segmentation models.
    
    This trainer supports:
    - Self-supervised training with auto-generated masks
    - Mixed precision training
    - Distributed Data Parallel (DDP)
    - TorchCompile
    - EMA model tracking
    - Checkpoint saving/resuming
    - In-context evaluation only (no regular validation)
    - Ensemble support for in-context evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize self-supervised trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
        
        self.epochs = config.get('epochs', 100)
        self.start_epoch = 0
        self.best_metric = float('-inf')
        self.current_iter = 0
        
        logging.info('Start self-supervised trainer initialization')

        self._setup_logging()
        self._setup_model()
        self._setup_optimizer()
        self._setup_criterion()
        self._setup_data()
        
        logging.info('Self-supervised trainer initialization done')
        
        if 'resume' in config.get('run', {}):
            self._resume_checkpoint()
    
    def _setup_model(self):
        """Set up model with DDP and optional EMA."""
        from models.iris import Iris
        
        model_config = deepcopy(self.config.get('model', {}))
        model_name = model_config.pop('type', 'iris')

        import models
        from utils.registry import get_model
        model_cls = get_model(model_name)
        self.model = model_cls(**model_config).to(self.device)

        if self.config.get('ema', {}).get('enabled', False):
            ema_config = self.config.get('ema', {})
            self.ema_model = model_cls(**model_config).to(self.device)
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.copy_(param.data)
            self.ema_decay = ema_config.get('decay', 0.999)
        else:
            self.ema_model = None
        
        compile_config = self.config.get('compile', {})
        if compile_config.get('enabled', False):
            compile_backend = compile_config.get('backend', 'inductor')
            compile_mode = compile_config.get('mode', 'default')
            logging.info(f"Compiling model with backend={compile_backend}, mode={compile_mode}")
            self.model = torch.compile(self.model, backend=compile_backend, mode=compile_mode)
            if self.ema_model is not None:
                self.ema_model = torch.compile(self.ema_model, backend=compile_backend, mode=compile_mode)
        
        if is_distributed():
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                find_unused_parameters=False
            )  
            if self.ema_model is not None:
                # EMA is read-only during backprop, so it does not need DDP.
                self.ema_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.ema_model)
            
        if is_master():
            total_params = sum(p.numel() for p in self.model.parameters())
            
            encoder_params = sum(p.numel() for p in self.model.encoder.parameters())
            
            decoder_params = sum(p.numel() for p in self.model.decoder.parameters())
            
            task_embedding_params = sum(p.numel() for p in self.model.task_embedding.parameters())
            for i in range(self.model.num_prior_stage):
                buffer = getattr(self.model, f'task_prior_{i}')
                task_embedding_params += buffer.numel()
            
            logging.info(f"=== Model Parameters Breakdown ===")
            logging.info(f"Encoder:        {encoder_params/1e6:>8.2f}M ({encoder_params/total_params*100:>5.1f}%)")
            logging.info(f"Decoder:        {decoder_params/1e6:>8.2f}M ({decoder_params/total_params*100:>5.1f}%)")
            logging.info(f"Task Embedding: {task_embedding_params/1e6:>8.2f}M ({task_embedding_params/total_params*100:>5.1f}%)")
            logging.info(f"Total:          {total_params/1e6:>8.2f}M")
            logging.info(f"==================================")
    
    def _setup_optimizer(self):
        """Set up optimizer and learning rate scheduler."""
        opt_config = deepcopy(self.config.get('optimizer', {}))
        opt_type = opt_config.pop('type', 'lamb')
        
        optimizer_cls = get_optimizer(opt_type)
        self.optimizer = optimizer_cls(
            self.model.parameters(), 
            **opt_config
        )
        

        scheduler_config = deepcopy(self.config.get('scheduler', {}))
        if scheduler_config:
            scheduler_name = scheduler_config.pop('type', 'cosine')
            
            if scheduler_name == 'OneCycleLR' and scheduler_config.get('total_steps') is None:
                epochs = self.config.get('epochs', 100)
                max_iter_per_epoch = self.config.get('max_iter_per_epoch', 400)
                total_steps = epochs * max_iter_per_epoch
                scheduler_config['total_steps'] = total_steps
                logging.info(f"Setting OneCycleLR total_steps to {total_steps}")
            
            scheduler_cls = get_scheduler(scheduler_name)
            self.scheduler = scheduler_cls(
                self.optimizer, 
                **scheduler_config
            )

        else:
            self.scheduler = None
        
        amp_config = self.config.get('amp', {})
        if amp_config.get('enabled', False):
            amp_dtype_str = amp_config.get('dtype', 'float16')
            
            if amp_dtype_str == 'float16':
                self.amp_dtype = torch.float16
                self.scaler = GradScaler('cuda')
                logging.info("Using FP16 with GradScaler")
            elif amp_dtype_str == 'bfloat16':
                self.amp_dtype = torch.bfloat16
                self.scaler = None
                logging.info("Using BF16 without GradScaler")
            else:
                logging.warning(f"Unrecognized dtype: {amp_dtype_str}, defaulting to FP16")
                raise ValueError(f"Unrecognized dtype: {amp_dtype_str}")
                
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
            self.amp_dtype = None

    
    def _setup_criterion(self):
        """Set up loss functions."""
        loss_config = deepcopy(self.config.get('loss', {}))
        
        
        self.loss_fns = {}
        self.loss_weights = {}
        
        for loss_name, loss_params in loss_config.items():
            if isinstance(loss_params, dict):
                weight = loss_params.pop('weight', 1.0)
                loss_type = loss_params.pop('type')
                loss_fn = get_criterion(loss_type)(**loss_params)
                
                self.loss_fns[loss_name] = loss_fn.to(self.device)
                self.loss_weights[loss_name] = weight
            else:
                self.loss_weights[loss_name] = loss_params
    
    def _create_sampler(self, dataset, shuffle=False, is_train=False):
        """
        Helper to create appropriate samplers based on mode and dataset.
        
        Args:
            dataset: Dataset to create sampler for
            shuffle: Whether to shuffle the data
            is_train: Whether this is for training mode
        
        Returns:
            Appropriate sampler for the dataset and mode
        """
        weight_list = None
        if is_train:
            weight_list = getattr(dataset, 'weight_list', None)
            if weight_list is None and hasattr(dataset, 'get_weight_list'):
                weight_list = dataset.get_weight_list()

        if is_train and weight_list is not None and len(weight_list) > 0:
            samples_weight = torch.as_tensor(weight_list, dtype=torch.double)
            loader_config = self.config.get('data', {}).get('loader', {})
            batch_size = int(loader_config.get('batch_size', 1))
            num_samples = int(self.config.get('max_iter_per_epoch', len(dataset))) * batch_size
            generator = None
            if is_distributed():
                generator = torch.Generator()
                generator.manual_seed(int(self.config.get('seed', 42)) + get_rank())

            # The pretraining dataset is virtually expanded via __len__/modulo,
            # so weighted sampling operates over the real scan list with replacement.
            if is_master():
                logging.info(
                    f"Using weighted pretraining sampler over {len(samples_weight)} scans "
                    f"for {num_samples} samples per epoch"
                )
            return torch.utils.data.WeightedRandomSampler(
                samples_weight,
                num_samples,
                replacement=True,
                generator=generator,
            )
        elif is_distributed():
            return torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=shuffle
            )
        # Non-distributed evaluation
        else:
            return None

    def _setup_data(self):
        """Set up data loaders for self-supervised training and in-context evaluation."""
        data_config = self.config.get('data', {})
        loader_config = data_config.get('loader', {})
        augmentation_config = self.config.get('augmentation', {})
        
        # The pretraining loader uses image .npy files plus HDF5 auto masks.
        train_dataset_config = deepcopy(data_config.get('train', {}))
        train_dataset_name = train_dataset_config.pop('type', 'MaskGuidedSelfSupervisedDataset')
        
        train_dataset_params = {k: v for k, v in train_dataset_config.items() if k != 'type'}
        train_dataset_params['augmentation_config'] = augmentation_config 
        train_dataset_params['device'] = self.device
        train_dataset = get_dataset(train_dataset_name)(**train_dataset_params)
        

        train_sampler = self._create_sampler(train_dataset, shuffle=True, is_train=True)
        num_workers = loader_config.get('num_workers', 4)
        train_loader_kwargs = {
            "batch_size": loader_config.get('batch_size', 1),
            "shuffle": train_sampler is None and loader_config.get('shuffle', True),
            "sampler": train_sampler,
            "num_workers": num_workers,
            "pin_memory": loader_config.get('pin_memory', True),
            "drop_last": loader_config.get('drop_last', False),
            "persistent_workers": loader_config.get('persistent_workers', True) and num_workers > 0,
        }
        if num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = loader_config.get('prefetch_factor', 2)
        self.train_loader = DataLoader(train_dataset, **train_loader_kwargs)
        
        # Optional segmentation validation uses GT .npy references, separate
        # from the self-supervised pretraining data stream.
        incontext_dataset_config = data_config.get('incontext', {})
        incontext_dataset_name = incontext_dataset_config.get('type')
        self.incontext_dataset_names = incontext_dataset_config.get('datasets', [])
        
        ensemble_size = self.config.get('incontext_evaluation', {}).get('ensemble_size', 1)
        
        self.incontext_loaders = {}
        
        if incontext_dataset_name and self.incontext_dataset_names:
            for dataset_name in self.incontext_dataset_names:
                single_dataset_params = {k: v for k, v in incontext_dataset_config.items() if k not in ['type', 'datasets']}
                single_dataset_params['datasets'] = [dataset_name]
                single_dataset_params['mode'] = 'test_incontext'
                single_dataset_params['num_references_per_class'] = ensemble_size
                # Processed arrays are [D, H, W], so target_spacing is stored as (z, y, x).
                single_dataset_params.setdefault('spacing', self.config.get('target_spacing', [1.5, 1.5, 1.5]))

                dataset = get_dataset(incontext_dataset_name)(**single_dataset_params)
                sampler = self._create_sampler(dataset, shuffle=False, is_train=False)

                self.incontext_loaders[dataset_name] = DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=2,
                    pin_memory=loader_config.get('pin_memory', True),
                    persistent_workers=loader_config.get('persistent_workers', True) and loader_config.get('num_workers', 4) > 0
                )
                logging.info(f"Created in-context loader for dataset: {dataset_name}")
        else:
            self.incontext_dataset_names = []
            logging.info("In-context evaluation disabled")
        
        self.max_iter_per_epoch = self.config.get('max_iter_per_epoch', len(self.train_loader))
        
        logging.info(f"Self-supervised training on {len(self.train_loader)} batches")
        logging.info(f"In-context evaluation datasets: {self.incontext_dataset_names}")
    
    def _setup_logging(self):
        """Set up logging and tensorboard."""

        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        
        if is_master():
            log_dir = os.path.join(self.config['run']['output_dir'], 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_handlers = [
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.config['run']['output_dir'], 'train.log'))
            ]
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s', 
                handlers=log_handlers,
                force=True
            )
            
            self.writer = SummaryWriter(log_dir)
            
            logging.info("Logging initialized (master process only)")
        else:
            logging.basicConfig(
                level=logging.WARNING,
                handlers=[logging.NullHandler()],
                force=True
            )
            self.writer = None

    def _resume_checkpoint(self):
        """Resume a full training checkpoint, including optimizer and scheduler."""
        self.start_epoch, self.current_iter, self.best_metric = resume_from_checkpoint(
            self.config,
            self.model,
            self.optimizer,
            self.ema_model,
            self.scaler,
            self.scheduler
        )
    
    def train(self):
        """Main training loop."""
        logging.info(f"Starting training from epoch {self.start_epoch} to {self.epochs}")
        logging.info(f"Starting from iteration {self.current_iter}")
        
        checkpoint_freq_epoch = self.config.get('checkpoint_freq_epoch', 1)  # Save every N epochs
        checkpoint_freq_iter = self.config.get('checkpoint_freq_iter', None)  # Save every N iterations
        
        iterations_per_epoch = self.max_iter_per_epoch
        actual_start_epoch = self.current_iter // iterations_per_epoch if self.current_iter > 0 else self.start_epoch

        for epoch in range(actual_start_epoch, self.epochs):
            if is_distributed() and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            self._train_epoch(epoch)

            if (epoch + 1) % checkpoint_freq_epoch == 0:
                self._save_checkpoint(epoch, tag='latest')
                logging.info(f"Saved checkpoint at epoch {epoch+1} as 'latest'")

            
            if self.incontext_loaders and (epoch + 1) % self.config.get('val_frequency', 1) == 0:
                incontext_metrics = self._validate_incontext_per_dataset(epoch)
                incontext_avg_metrics = self._calculate_average_metrics(incontext_metrics)
                
                current_metric = incontext_avg_metrics['dice_mean']
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
                else:
                    is_best = False
                
            
                self._save_checkpoint(epoch, tag=f"epoch_{epoch+1}", is_best=is_best)
                self._save_checkpoint(epoch, tag="latest")

                self._save_metrics(epoch, incontext_metrics, incontext_avg_metrics)

        
        
        if self.writer is not None:
            self.writer.close()
        
        logging.info("Training completed!")
    

    def _compute_loss(self, output, target):
        """
        Compute combined loss from all loss components.
        
        Args:
            output: Model output (predictions)
            target: Ground truth target
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        losses = {}
        
        for loss_name, loss_fn in self.loss_fns.items():
            loss = loss_fn(output, target)
            
            weighted_loss = loss * self.loss_weights[loss_name]
            total_loss += weighted_loss
            losses[loss_name] = loss.item()
        
        return total_loss, losses

    def _train_epoch(self, epoch: int):
        """
        Train one epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.model.train()
        
        batch_time = AverageMeter('Time', ':.4f')
        data_time = AverageMeter('Data', ':.4f')
        losses = AverageMeter('Loss', ':.4f')
        
        loss_meters = {name: AverageMeter(name, ':.4f') for name in self.loss_fns.keys()}
        
        meters = [batch_time, data_time, losses] + list(loss_meters.values())
        progress = ProgressMeter(
            min(len(self.train_loader), self.max_iter_per_epoch),
            meters,
            prefix=f"Epoch: [{epoch+1}/{self.epochs}]"
        )

        checkpoint_freq_iter = self.config.get('checkpoint_freq_iter', None)
        
        iterations_per_epoch = min(len(self.train_loader), self.max_iter_per_epoch)
        start_iter_in_epoch = self.current_iter % iterations_per_epoch if self.current_iter > 0 else 0
        
        if start_iter_in_epoch > 0:
            logging.info(f"Resuming epoch {epoch+1} from iteration {start_iter_in_epoch}/{iterations_per_epoch}")
        
        end = time.time()
        for i, batch in enumerate(self.train_loader):
            if i < start_iter_in_epoch:
                continue
                
            if i >= self.max_iter_per_epoch:
                break
            
            data_time.update(time.time() - end)
            
            # Batch layout: target view is optimized against masks defined by
            # the paired reference view.
            tgt_img = batch[0].to(self.device)
            tgt_mask = batch[1].to(self.device)
            ref_img = batch[2].to(self.device)
            ref_mask = batch[3].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    outputs = self.model(tgt_img, ref_img, ref_mask, update_buffer=False)
                    loss, individual_losses = self._compute_loss(outputs, tgt_mask)
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            else:
                outputs = self.model(tgt_img, ref_img, ref_mask, update_buffer=False)
                loss, individual_losses = self._compute_loss(outputs, tgt_mask)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.ema_model is not None:
                # EMA weights provide a smoother checkpoint for validation and
                # release-time inference.
                with torch.no_grad():
                    for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                        ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

                    for (name, buffer), (ema_name, ema_buffer) in zip(
                        self.model.named_buffers(), 
                        self.ema_model.named_buffers()
                    ):
                        if 'num_batches_tracked' in name:
                            continue
                        ema_buffer.data.mul_(self.ema_decay).add_(buffer.data, alpha=1 - self.ema_decay)


            if is_distributed():
                loss = reduce_tensor(loss)
                for name, indiv_loss in individual_losses.items():
                    individual_losses[name] = reduce_tensor(torch.tensor(indiv_loss, device=self.device)).item()

            losses.update(loss.item(), tgt_img.size(0))
            for name, indiv_loss in individual_losses.items():
                loss_meters[name].update(indiv_loss, tgt_img.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config.get('print_freq', 10) == 0:
                progress.display(i)

            self.current_iter += 1

            if checkpoint_freq_iter is not None and self.current_iter % checkpoint_freq_iter == 0:
                self._save_checkpoint(epoch, tag=f"iter_{self.current_iter}")
                logging.info(f"Saved checkpoint at iteration {self.current_iter}")
            
            if is_master() and self.writer is not None and i % self.config.get('log_freq', 10) == 0:
                global_step = self.current_iter
                self.writer.add_scalar('Train/Loss', losses.avg, global_step)
                for name, loss_meter in loss_meters.items():
                    self.writer.add_scalar(f'Train/{name}', loss_meter.avg, global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)

        if is_master() and self.writer is not None:
            logging.info(f"Epoch [{epoch+1}/{self.epochs}] completed - Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.writer.add_scalar('epoch/train_loss', losses.avg, epoch+1)
            for name, loss_meter in loss_meters.items():
                self.writer.add_scalar(f'epoch/train_{name}', loss_meter.avg, epoch+1)
                
    
    def _validate_incontext_per_dataset(self, epoch: int) -> List[Dict[str, Any]]:
        """
        Run in-context evaluation on all in-context datasets.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            List of dictionaries with metrics for each in-context dataset
        """
        logging.info(f"Running in-context evaluation for epoch {epoch+1}")
        
        model_to_eval = self.ema_model if self.ema_model is not None else self.model
        model_to_eval.eval()
        
        incontext_metrics = []
        ensemble_size = self.config.get('incontext_evaluation', {}).get('ensemble_size', 1)
        
        dice_total = 0.0
        asd_total = 0.0
        hd_total = 0.0
        
        for dataset_id in self.incontext_dataset_names:
            logging.info(f"Running in-context evaluation on {dataset_id}")
            evaluator = Evaluator(
                model_to_eval, 
                self.incontext_loaders[dataset_id], 
                self.config, 
                device=self.device, 
                incontext=True
            )
            
            dice_mean, asd_mean, hd_mean = evaluator.run()
            
            dice_avg = float(np.mean(dice_mean))
            asd_avg = float(np.mean(asd_mean))
            hd_avg = float(np.mean(hd_mean))
            
            dice_total += dice_avg
            asd_total += asd_avg
            hd_total += hd_avg
            
            metrics = {
                'dataset': dataset_id,
                'ensemble_size': ensemble_size,
                'dice_mean': dice_avg,
                'asd_mean': asd_avg,
                'hd_mean': hd_avg,
                'dice_per_class': dice_mean.tolist(),
                'asd_per_class': asd_mean.tolist(),
                'hd_per_class': hd_mean.tolist()
            }
            
            incontext_metrics.append(metrics)
            self._log_metrics(epoch, dataset_id, metrics, is_incontext=True)  # Log per-dataset metrics
        
        num_datasets = len(self.incontext_dataset_names)
        if num_datasets > 0:
            combined_metrics = {
                'dataset': 'combined_incontext',
                'ensemble_size': ensemble_size,
                'dice_mean': dice_total / num_datasets,
                'asd_mean': asd_total / num_datasets,
                'hd_mean': hd_total / num_datasets
            }
            incontext_metrics.append(combined_metrics)
            
            logging.info(f"Combined In-context Validation - "
                        f"Dice: {combined_metrics['dice_mean']:.4f}, "
                        f"ASD: {combined_metrics['asd_mean']:.4f}, "
                        f"HD: {combined_metrics['hd_mean']:.4f}")
        
        self._log_overall_metrics(epoch, incontext_metrics, is_incontext=True)
        
        return incontext_metrics

    def _calculate_average_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average metrics across datasets.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Dictionary with average metrics
        """
        if not metrics_list:
            return {'dice_mean': 0.0, 'asd_mean': 0.0, 'hd_mean': 0.0}
        
        dataset_metrics = [m for m in metrics_list if m['dataset'] != 'combined_incontext']
        
        if not dataset_metrics:
            return {'dice_mean': 0.0, 'asd_mean': 0.0, 'hd_mean': 0.0}
        
        dice_sum = sum(m['dice_mean'] for m in dataset_metrics)
        asd_sum = sum(m['asd_mean'] for m in dataset_metrics)
        hd_sum = sum(m['hd_mean'] for m in dataset_metrics)
        n = len(dataset_metrics)
        
        return {
            'dice_mean': dice_sum / n,
            'asd_mean': asd_sum / n,
            'hd_mean': hd_sum / n
        }
    
    def _save_checkpoint(self, epoch: int, tag: str = "latest", is_best: bool = False):
        """Save a resumable training checkpoint.

        Release checkpoints should be stripped before upload because this state
        also contains optimizer, scheduler, and run configuration.
        """
        state = prepare_checkpoint_state(
            model=self.model,
            optimizer=self.optimizer,
            ema_model=self.ema_model,
            epoch=epoch + 1,
            iteration=self.current_iter,
            best_metric=self.best_metric,
            config=self.config,
            scaler=self.scaler,
            scheduler=self.scheduler
        )
        
        save_checkpoint(
            state,
            self.config,
            tag=tag,
            is_best=is_best
        )
    

    
    def _save_metrics(self, epoch: int, incontext_metrics: List[Dict], incontext_avg_metrics: Dict):
        """
        Save evaluation metrics to JSON file.
        
        Args:
            epoch: Current epoch number
            incontext_metrics: In-context validation metrics for each dataset
            incontext_avg_metrics: Average metrics across all in-context datasets
        """
        if not is_master():
            return
            
        output_dir = self.config['run']['output_dir']
        metrics_dir = os.path.join(output_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_data = {
            'epoch': epoch + 1,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'incontext_validation': incontext_metrics,
            'incontext_average_metrics': incontext_avg_metrics,
            'best_metric': self.best_metric
        }
        
        metrics_file = os.path.join(metrics_dir, f'metrics_epoch_{epoch+1}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logging.info(f"Saved evaluation metrics to {metrics_file}")
    
    def _log_metrics(self, epoch, dataset_name, metrics, is_incontext=False):
        """
        Helper method to log metrics to console and tensorboard with optimized grouping.
        
        Args:
            epoch: Current epoch number
            dataset_name: Name of the dataset
            metrics: Dictionary of metrics
            is_incontext: Whether these are in-context evaluation metrics
        """
        mode_str = 'In-context ' if is_incontext else ''
        logging.info(f"{mode_str}Validation on {dataset_name} - "
                    f"Dice: {metrics['dice_mean']:.4f}, "
                    f"ASD: {metrics['asd_mean']:.4f}, "
                    f"HD: {metrics['hd_mean']:.4f}")
        
        if 'dice_per_class' in metrics and metrics['dice_per_class'] is not None:
            dice_per_class = metrics['dice_per_class']
            asd_per_class = metrics['asd_per_class']
            hd_per_class = metrics['hd_per_class']
            
            logging.info(f"{mode_str}Per-class metrics for {dataset_name}:")
            
            logging.info(f"  {'Class':<10} {'Dice':<10} {'ASD':<10} {'HD':<10}")
            logging.info(f"  {'-'*40}")
            
            for cls_idx, (dice, asd, hd) in enumerate(zip(dice_per_class, asd_per_class, hd_per_class)):
                logging.info(f"  {cls_idx:<10} {dice:<10.4f} {asd:<10.2f} {hd:<10.2f}")
        
        if self.writer is not None and is_master():
            prefix = 'InContext' if is_incontext else 'Val'
            
            if dataset_name != 'combined_incontext':
                # Per-dataset metrics keep TensorBoard plots easy to filter.
                self.writer.add_scalar(f'{prefix}/{dataset_name}/Dice/avg_dice', metrics['dice_mean'], epoch + 1)
                
                self.writer.add_scalar(f'{prefix}/{dataset_name}/ASD/avg_asd', metrics['asd_mean'], epoch + 1)
                
                self.writer.add_scalar(f'{prefix}/{dataset_name}/HD/avg_hd', metrics['hd_mean'], epoch + 1)
                
                if 'dice_per_class' in metrics and metrics['dice_per_class'] is not None:
                    for cls_idx, (dice, asd, hd) in enumerate(zip(
                        metrics['dice_per_class'], 
                        metrics['asd_per_class'], 
                        metrics['hd_per_class']
                    )):
                        self.writer.add_scalar(f'{prefix}/{dataset_name}/Dice/class_{cls_idx}', dice, epoch + 1)
                        self.writer.add_scalar(f'{prefix}/{dataset_name}/ASD/class_{cls_idx}', asd, epoch + 1)
                        self.writer.add_scalar(f'{prefix}/{dataset_name}/HD/class_{cls_idx}', hd, epoch + 1)


    def _log_overall_metrics(self, epoch, incontext_metrics, is_incontext=True):
        """
        Log overall metrics across all datasets to tensorboard.
        
        Args:
            epoch: Current epoch number
            incontext_metrics: List of metric dictionaries for each dataset
            is_incontext: Whether these are in-context evaluation metrics
        """
        if self.writer is None or not is_master():
            return
        
        prefix = 'InContext' if is_incontext else 'Val'
        
        dice_total = 0.0
        asd_total = 0.0
        hd_total = 0.0
        dataset_count = 0
        
        # Log each dataset and the macro-average across datasets.
        for metrics in incontext_metrics:
            dataset_name = metrics['dataset']
            
            if dataset_name == 'combined_incontext':
                continue
            
            self.writer.add_scalar(f'{prefix}/Overall/Dice/{dataset_name}', metrics['dice_mean'], epoch + 1)
            self.writer.add_scalar(f'{prefix}/Overall/ASD/{dataset_name}', metrics['asd_mean'], epoch + 1)
            self.writer.add_scalar(f'{prefix}/Overall/HD/{dataset_name}', metrics['hd_mean'], epoch + 1)
            
            dice_total += metrics['dice_mean']
            asd_total += metrics['asd_mean']
            hd_total += metrics['hd_mean']
            dataset_count += 1
        
        if dataset_count > 0:
            self.writer.add_scalar(f'{prefix}/Overall/Dice/average', dice_total / dataset_count, epoch + 1)
            self.writer.add_scalar(f'{prefix}/Overall/ASD/average', asd_total / dataset_count, epoch + 1)
            self.writer.add_scalar(f'{prefix}/Overall/HD/average', hd_total / dataset_count, epoch + 1)
