"""Downstream segmentation finetuning trainer.

This release example adapts a pretrained MASS/Iris model to a supervised
segmentation dataset, with standard optimization, validation, checkpointing,
and optional use of pretrained weights.
"""

import os
import time
import logging
import json
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from utils.distributed import (
    is_master, get_rank, get_local_rank, get_world_size, 
    is_distributed, reduce_tensor
)
from utils.checkpoint import (
    save_checkpoint, resume_from_checkpoint, prepare_checkpoint_state
)
from utils.metrics_utils import AverageMeter, ProgressMeter, TimeMeter
from training.evaluator import Evaluator, RegularEvaluator
from utils.registry import (
    get_optimizer, get_scheduler, get_criterion, 
    get_dataset, get_model, register_trainer
)

@register_trainer("finetuning")
class FineTuningTrainer:
    """
    Downstream segmentation finetuning example for MASS/Iris checkpoints.
    It supports full finetuning and probing with learnable task priors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize finetuning trainer."""
        self.config = config
        self.device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
        
        self.epochs = config.get('epochs', 100)
        self.start_epoch = 0
        self.best_metric = float('-inf')
        self.current_iter = 0
        
        self.finetune_config = config.get('finetuning', {})
        
        self._setup_amp()
        
        logging.info('Start finetuning trainer initialization')
        
        self._setup_logging()
        self._setup_model()
        self._setup_criterion()
        self._setup_data()
        
        if self.model_type == 'iris':
            self._setup_learnable_priors()
        else:
            self.learnable_priors = None
            self.ema_learnable_priors = None
            
        self._setup_optimizer()
        
        logging.info('Finetuning trainer initialization done')
        
        if 'resume' in config.get('run', {}):
            self._resume_checkpoint()

    def _setup_amp(self):
        """Setup Automatic Mixed Precision training."""
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
                self.amp_dtype = torch.float16
                self.scaler = GradScaler('cuda')
                
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
            self.amp_dtype = None

    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for components."""
        param_groups = []
        
        finetune_mode = self.finetune_config.get('mode', 'full')
        
        model_params = []
        classifier_params = []
        other_model_params = []
        
        # Give the segmentation head and task priors their own learning rates.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                model_params.append(param)
                if 'out' in name or 'decoder' in name or 'head' in name:
                    classifier_params.append(param)
                else:
                    other_model_params.append(param)
        
        prior_params = []
        if self.model_type == 'iris' and self.learnable_priors is not None:
            for param in self.learnable_priors.parameters():
                if param.requires_grad:
                    prior_params.append(param)
        
        if is_master():
            logging.info(f"=== Optimizer Setup ({finetune_mode} mode) ===")
            if finetune_mode == 'probing':
                logging.info(f"  - Classifier parameters: {sum(p.numel() for p in classifier_params)/1e6:.2f}M")
                logging.info(f"  - Other model parameters (should be 0): {sum(p.numel() for p in other_model_params)/1e6:.2f}M")
            else:
                logging.info(f"  - Total model parameters: {sum(p.numel() for p in model_params)/1e6:.2f}M")
                logging.info(f"    - Classifier: {sum(p.numel() for p in classifier_params)/1e6:.2f}M")
                logging.info(f"    - Other model: {sum(p.numel() for p in other_model_params)/1e6:.2f}M")
            if prior_params:
                logging.info(f"  - Learnable prior parameters: {sum(p.numel() for p in prior_params)/1e6:.2f}M")
        
        if model_params:
            param_groups.append({
                'params': model_params,
                'lr': self.finetune_config.get('model_lr', 0.0001),
                'name': 'model'
            })
        
        if prior_params:
            param_groups.append({
                'params': prior_params,
                'lr': self.finetune_config.get('embedding_lr', 0.001),
                'name': 'priors'
            })
        
        if not param_groups:
            raise ValueError("No parameters to optimize! Check freezing configuration.")
        
        optimizer_config = self.config.get('optimizer', {}).copy()
        optimizer_type = optimizer_config.pop('type')
        optimizer_config.pop('lr', None)
        
        self.optimizer = get_optimizer(optimizer_type)(
            param_groups, 
            **optimizer_config
        )
        
        if is_master():
            for group in self.optimizer.param_groups:
                logging.info(f"  - {group['name']} learning rate: {group['lr']:.6f}")
        
        scheduler_config = self.config.get('scheduler', {}).copy()
        if scheduler_config:
            scheduler_type = scheduler_config.pop('type')
            
            if 'total_steps' in scheduler_config and scheduler_config['total_steps'] is None:
                scheduler_config['total_steps'] = self.steps_per_epoch * self.epochs
            
            self.scheduler = get_scheduler(scheduler_type)(
                self.optimizer,
                **scheduler_config
            )
        else:
            self.scheduler = None
    
    def _setup_model(self):
        """Set up an Iris/MASS model for downstream segmentation finetuning."""
        model_config = self.config.get('model', {}).copy()
        model_name = model_config.pop('type', 'iris').lower()
        if model_name != 'iris':
            raise ValueError(
                "The open-source finetuning example only supports model.type='iris'. "
                f"Got: {model_name}"
            )

        self.model_type = 'iris'
        model_cls = get_model('iris')
        self.model = model_cls(**model_config).to(self.device)

        state_dict = None
        pretrained_path = self.finetune_config.get('pretrained_checkpoint')
        if pretrained_path not in [None, False, 'None', 'False', 'none', 'false', '']:
            logging.info(f"Loading MASS checkpoint from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            prefer_ema = self.finetune_config.get('use_ema_checkpoint', True)

            # EMA checkpoints are usually the best starting point after SSL pretraining.
            if prefer_ema and 'ema_model' in checkpoint:
                state_dict = checkpoint['ema_model']
                logging.info("Using EMA weights from pretrained checkpoint")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logging.warning(f"Missing keys in checkpoint: {missing_keys[:20]}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:20]}")
        else:
            logging.info("No pretrained checkpoint provided. Finetuning from random initialization.")

        finetune_mode = self.finetune_config.get('mode', 'full')
        if finetune_mode == 'probing':
            logging.info("=== PROBING MODE (iris) ===")
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            for param in self.model.task_embedding.parameters():
                param.requires_grad = False
            logging.info("Frozen encoder, decoder, and task embedding; priors/head remain trainable")
        elif finetune_mode == 'full':
            logging.info("=== FULL FINETUNING MODE (iris) ===")
        else:
            raise ValueError(f"Unknown finetuning mode: {finetune_mode}. Must be 'full' or 'probing'")

        if self.config.get('ema', {}).get('enabled', False):
            self.use_ema = True
            ema_config = self.config.get('ema', {})
            self.ema_model = model_cls(**model_config).to(self.device)
            self.ema_model.load_state_dict(self.model.state_dict(), strict=False)

            if finetune_mode == 'probing':
                for param in self.ema_model.encoder.parameters():
                    param.requires_grad = False
                for param in self.ema_model.decoder.parameters():
                    param.requires_grad = False
                for param in self.ema_model.task_embedding.parameters():
                    param.requires_grad = False

            self.ema_decay = ema_config.get('decay', 0.999)
            self.use_prior_ema = ema_config.get('prior_ema', True)
            self.ema_learnable_priors = None
        else:
            self.use_ema = False
            self.ema_model = None
            self.use_prior_ema = False
            self.ema_learnable_priors = None

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
                find_unused_parameters=(finetune_mode == 'probing'),
            )
            if self.ema_model is not None:
                self.ema_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.ema_model)

        if is_master():
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info("=== Model Summary ===")
            logging.info(f"Total parameters: {total_params/1e6:.2f}M")
            logging.info(f"Trainable parameters: {trainable_params/1e6:.2f}M")
            logging.info(f"Frozen parameters: {(total_params - trainable_params)/1e6:.2f}M")

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
    
    def _create_sampler(self, dataset, shuffle):
        """Create distributed sampler if needed."""
        if is_distributed():
            return torch.utils.data.distributed.DistributedSampler(
                dataset, 
                shuffle=shuffle,
                num_replicas=get_world_size(),
                rank=get_rank()
            )
        return None  


    def _setup_data(self):
        """Setup data loaders."""
        data_config = self.config.get('data', {})
        loader_config = data_config.get('loader', {})
        augmentation_config = self.config.get('augmentation', {})
        
        train_config = deepcopy(data_config.get('train', {}))
        train_dataset_type = train_config.pop('type', 'FineTuningDataset')
        train_config.setdefault('spacing', self.config.get('target_spacing', [1.5, 1.5, 1.5]))
        train_dataset_params = {k: v for k, v in train_config.items() if k != 'type'}
        train_dataset_params['augmentation_config'] = augmentation_config
        
        train_dataset = get_dataset(train_dataset_type)(**train_dataset_params)
        self._sync_finetune_classes(train_dataset)
        
        train_sampler = self._create_sampler(train_dataset, shuffle=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=loader_config.get('batch_size', 1),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=loader_config.get('num_workers', 4),
            pin_memory=loader_config.get('pin_memory', True),
            drop_last=loader_config.get('drop_last', False),
            persistent_workers=loader_config.get('persistent_workers', True) and loader_config.get('num_workers', 4) > 0
        )
        
        val_config = deepcopy(data_config.get('val', {}))
        if val_config:
            val_dataset_type = val_config.pop('type')
            val_config.setdefault('spacing', self.config.get('target_spacing', [1.5, 1.5, 1.5]))
            val_dataset_params = {
                k: v for k, v in val_config.items() if k != 'type'
            }
            
            val_dataset = get_dataset(val_dataset_type)(**val_dataset_params)
            self._check_val_classes(val_dataset)
            
            val_sampler = self._create_sampler(val_dataset, shuffle=False)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                sampler=val_sampler,
                num_workers=loader_config.get('num_workers', 4),
                pin_memory=loader_config.get('pin_memory', True),
                persistent_workers=loader_config.get('persistent_workers', True) and loader_config.get('num_workers', 4) > 0
            )
        else:
            self.val_loader = None
        
        self.steps_per_epoch = self._resolve_steps_per_epoch()
        logging.info(
            f"Training loader has {len(self.train_loader)} batches; "
            f"running {self.steps_per_epoch} optimization steps per epoch"
        )

    def _resolve_steps_per_epoch(self) -> int:
        """Resolve the number of optimization steps in each finetuning epoch."""
        if len(self.train_loader) == 0:
            raise ValueError("Training loader is empty. Check dataset paths and splits.")

        max_iter_per_epoch = self.config.get('max_iter_per_epoch', None)
        if max_iter_per_epoch is None:
            return len(self.train_loader)

        max_iter_per_epoch = int(max_iter_per_epoch)
        if max_iter_per_epoch <= 0:
            raise ValueError("max_iter_per_epoch must be positive or null.")
        return max_iter_per_epoch

    def _sync_finetune_classes(self, train_dataset):
        """Use dataset foreground classes as the source of truth for priors."""
        if not hasattr(train_dataset, 'foreground_classes'):
            raise ValueError(
                "FineTuningTrainer requires the train dataset to expose "
                "`foreground_classes`. Pass `foreground_classes` in the dataset "
                "config or define the dataset label map in data/split.py."
            )

        self.foreground_classes = list(train_dataset.foreground_classes)
        self.num_classes = len(self.foreground_classes)
        if self.num_classes == 0:
            raise ValueError("No foreground classes found for finetuning.")

        configured_num_classes = self.finetune_config.get('num_classes')
        if configured_num_classes not in [None, 'null', 'None', 'auto']:
            configured_num_classes = int(configured_num_classes)
            if configured_num_classes != self.num_classes:
                logging.warning(
                    "finetuning.num_classes=%s does not match the dataset "
                    "foreground class count=%s. Using the dataset value. To "
                    "finetune a subset, set data.train.foreground_classes and "
                    "data.val.foreground_classes.",
                    configured_num_classes,
                    self.num_classes,
                )

        self.finetune_config['num_classes'] = self.num_classes
        logging.info(
            "Finetuning foreground classes (%d): %s",
            self.num_classes,
            self.foreground_classes,
        )

    def _check_val_classes(self, val_dataset):
        """Ensure validation returns the same class channels as training."""
        if not hasattr(val_dataset, 'foreground_classes'):
            return

        val_classes = list(val_dataset.foreground_classes)
        if val_classes != self.foreground_classes:
            raise ValueError(
                "Validation foreground_classes must match training "
                f"foreground_classes. train={self.foreground_classes}, "
                f"val={val_classes}"
            )
    

    def _setup_learnable_priors(self):
        """Initialize Iris learnable priors from supervised training labels.

        For each foreground class, the trainer encodes all available class crops
        and averages their task embeddings. These priors become trainable
        class-specific task tokens for downstream finetuning.
        """
        assert self.model_type == 'iris', "Learnable priors only applicable to Iris models"
        
        num_classes = self.num_classes
        num_stages = self.finetune_config.get('num_prior_stages', 3)
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        training_size = self.config['data']['train'].get('training_size', [128, 128, 128])
        if isinstance(training_size, int):
            training_size = [training_size, training_size, training_size]
        
        if not hasattr(self, 'train_loader'):
            raise RuntimeError("train_loader not initialized. Make sure _setup_data() is called before _setup_learnable_priors()")
        
        train_dataset = self.train_loader.dataset
        
        self.learnable_priors = nn.ParameterList()
        
        logging.info(f"Computing initial embeddings from all available samples...")
        logging.info(f"Dataset has {num_classes} foreground classes: {self.foreground_classes}")
        
        # Encode reference crops without updating normalization/dropout state.
        model.eval()
        
        all_class_embeddings = []
        
        
        with torch.no_grad():
            for class_idx in range(num_classes):
                class_label = self.foreground_classes[class_idx]
                logging.info(f"Processing class {class_idx}/{num_classes-1} (label={class_label})...")
                
                class_samples = train_dataset.get_all_samples_for_class(class_idx)
                
                if len(class_samples) == 0:
                    logging.warning(f"No samples found for class {class_idx} (label={class_label}), using random initialization")
                    stage_embeddings_for_class = []
                    for stage_idx in range(num_stages):
                        dim = getattr(model.encoder, f"down{4-stage_idx}").out_ch
                        if hasattr(model, f'task_prior_{stage_idx}'):
                            task_prior_buffer = getattr(model, f'task_prior_{stage_idx}')
                            num_tokens = task_prior_buffer.shape[1]
                        else:
                            num_tokens = 11
                        
                        random_emb = torch.randn(1, num_tokens, dim, device=self.device) * 0.02
                        stage_embeddings_for_class.append(random_emb)
                    all_class_embeddings.append(stage_embeddings_for_class)
                    continue
                
                stage_embeddings = [[] for _ in range(num_stages)]
                valid_samples = 0
                
                for sample_idx, sample in enumerate(class_samples):
                    try:
                        img = np.load(sample['img_npy_path'], mmap_mode='r').astype(np.float32)
                        lab = np.load(sample['gt_npy_path'], mmap_mode='r').astype(np.float32)
                        
                        class_mask = (lab == class_label)
                        if not class_mask.any():
                            continue
                        
                        indices = np.where(class_mask)
                        center = [int(np.mean(indices[i])) for i in range(3)]
                        
                        # Crop around foreground so the learned prior starts
                        # from a real visual example of this class.
                        crop_slices = []
                        for i in range(3):
                            half_size = training_size[i] // 2
                            start = max(0, center[i] - half_size)
                            end = min(img.shape[i], start + training_size[i])
                            if end - start < training_size[i]:
                                start = max(0, end - training_size[i])
                            crop_slices.append(slice(start, end))
                        
                        img_crop = img[crop_slices[0], crop_slices[1], crop_slices[2]]
                        lab_crop = lab[crop_slices[0], crop_slices[1], crop_slices[2]]
                        
                        pad_widths = []
                        for i in range(3):
                            pad_before = (training_size[i] - img_crop.shape[i]) // 2
                            pad_after = training_size[i] - img_crop.shape[i] - pad_before
                            pad_widths.append((max(0, pad_before), max(0, pad_after)))
                        
                        if any(sum(p) > 0 for p in pad_widths):
                            img_crop = np.pad(img_crop, pad_widths, mode='constant', constant_values=0)
                            lab_crop = np.pad(lab_crop, pad_widths, mode='constant', constant_values=0)
                        
                        img_tensor = torch.from_numpy(img_crop).unsqueeze(0).unsqueeze(0).to(self.device)
                        lab_tensor = torch.from_numpy(lab_crop).unsqueeze(0).unsqueeze(0).to(self.device)
                        
                        class_mask_tensor = (lab_tensor == class_label).float()
                        
                        if class_mask_tensor.sum() == 0:
                            continue
                        
                        if self.use_amp:
                            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                                ref_feat_list = model.encode_image_feature(img_tensor)
                                embeddings = model.encode_visual_prior(ref_feat_list, class_mask_tensor)
                        else:
                            ref_feat_list = model.encode_image_feature(img_tensor)
                            embeddings = model.encode_visual_prior(ref_feat_list, class_mask_tensor)
                        
                        for stage_idx, emb in enumerate(embeddings):
                            if stage_idx < num_stages:
                                stage_embeddings[stage_idx].append(emb)
                        
                        valid_samples += 1
                        
                        if (sample_idx + 1) % 10 == 0:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    
                    except Exception as e:
                        logging.warning(f"Failed to compute embedding for class {class_idx} (label={class_label}), sample {sample_idx}: {e}")
                        continue
                
                logging.info(f"Class {class_idx} (label={class_label}): processed {valid_samples}/{len(class_samples)} samples successfully")
                
                stage_embeddings_for_class = []
                for stage_idx in range(num_stages):
                    if stage_embeddings[stage_idx]:
                        # Average real examples into one learnable prior per class.
                        stacked = torch.cat(stage_embeddings[stage_idx], dim=0)
                        stacked = torch.squeeze(stacked, dim=1)
                        avg_embedding = stacked.mean(dim=0, keepdim=True)
                        stage_embeddings_for_class.append(avg_embedding)
                    else:
                        # Classes without valid crops still need an optimizable prior.
                        dim = getattr(model.encoder, f"down{4-stage_idx}").out_ch
                        if hasattr(model, f'task_prior_{stage_idx}'):
                            task_prior_buffer = getattr(model, f'task_prior_{stage_idx}')
                            num_tokens = task_prior_buffer.shape[1]
                        else:
                            num_tokens = 11
                        
                        random_emb = torch.randn(1, num_tokens, dim, device=self.device) * 0.02
                        stage_embeddings_for_class.append(random_emb)
                
                all_class_embeddings.append(stage_embeddings_for_class)
        
        model.train()
        
        for stage_idx in range(num_stages):
            stage_priors = []
            for class_embeddings in all_class_embeddings:
                stage_priors.append(class_embeddings[stage_idx])
            
            stage_priors = torch.cat(stage_priors, dim=0)
            
            stage_priors = nn.Parameter(stage_priors.contiguous())
            self.learnable_priors.append(stage_priors)
            
            logging.info(f"Stage {stage_idx}: initialized priors with shape {list(stage_priors.shape)}")
        
        if self.use_prior_ema:
            self.ema_learnable_priors = nn.ParameterList()
            for stage_priors in self.learnable_priors:
                ema_priors = nn.Parameter(stage_priors.detach().clone())
                ema_priors.requires_grad = False
                self.ema_learnable_priors.append(ema_priors)
            logging.info("Initialized EMA learnable priors")
        
        logging.info(f"Initialized learnable priors: {num_classes} classes, {num_stages} stages using all available samples")

    
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
    
    def train(self):
        """Main training loop."""
        self.validate(-1)
        for epoch in range(self.start_epoch, self.epochs):
            if is_distributed():
                self.train_loader.sampler.set_epoch(epoch)
            
            # Training epoch
            self.train_epoch(epoch)
            
            if self.scheduler and hasattr(self.scheduler, 'step'):
                self.scheduler.step()
            
            if self.val_loader and (epoch + 1) % self.config.get('val_frequency', 5) == 0:
                self.validate(epoch)
            
            if is_master() and (epoch + 1) % self.config.get('checkpoint_freq_epoch', 1) == 0:
                self.save_checkpoint(epoch)

    def train_epoch(self, epoch):
        """Train for one epoch with proper AMP handling."""
        self.model.train()
        
        batch_time = AverageMeter('Time', ':.4f')
        data_time = AverageMeter('Data', ':.4f')
        losses = AverageMeter('Loss', ':.4f')
        
        loss_meters = {name: AverageMeter(name, ':.4f') for name in self.loss_fns.keys()}
        
        meters = [batch_time, data_time, losses] + list(loss_meters.values())
        progress = ProgressMeter(
            self.steps_per_epoch,
            meters,
            prefix=f"Epoch: [{epoch+1}/{self.epochs}]"
        )
        
        end = time.time()
        train_iter = iter(self.train_loader)
        for i in range(self.steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                # Few-shot runs can intentionally request more steps than one
                # pass over the loader; restart to draw new random crops/augs.
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            data_time.update(time.time() - end)
            
            query_img = batch['query_img'].to(self.device, non_blocking=True)
            query_lab = batch['query_lab'].to(self.device, non_blocking=True)
            
            # class_indices are 0-based positions into foreground_classes, not
            # raw dataset label values.
            class_indices = batch.get('class_indices', None)
            if class_indices is not None:
                class_indices = class_indices.to(self.device, non_blocking=True)
            
            self.current_iter = epoch * self.steps_per_epoch + i
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    loss, individual_losses = self.train_step(query_img, query_lab, class_indices)
                
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
                loss, individual_losses = self.train_step(query_img, query_lab, class_indices)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            if self.use_ema:
                with torch.no_grad():
                    for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                        ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
                    
                    for buffer, ema_buffer in zip(self.model.buffers(), self.ema_model.buffers()):
                        ema_buffer.data.mul_(self.ema_decay).add_(buffer.data, alpha=1 - self.ema_decay)
                    
                    if self.use_prior_ema and self.learnable_priors is not None:
                        for prior, ema_prior in zip(self.learnable_priors, self.ema_learnable_priors):
                            ema_prior.data.mul_(self.ema_decay).add_(prior.data, alpha=1 - self.ema_decay)
            

            
            losses.update(loss.item(), query_img.size(0))
            for name, indiv_loss in individual_losses.items():
                loss_meters[name].update(indiv_loss, query_img.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.config.get('print_freq', 10) == 0:
                progress.display(i)
            
            if self.writer and i % self.config.get('log_freq', 50) == 0:
                global_step = epoch * self.steps_per_epoch + i
                self.writer.add_scalar('Train/Loss', losses.avg, global_step)
                for name, loss_meter in loss_meters.items():
                    self.writer.add_scalar(f'Train/{name}', loss_meter.avg, global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
            
            self.current_iter += 1
        
        if is_master() and self.writer is not None:
            logging.info(f"Epoch [{epoch+1}/{self.epochs}] completed - Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.writer.add_scalar('epoch/train_loss', losses.avg, epoch+1)
            for name, loss_meter in loss_meters.items():
                self.writer.add_scalar(f'epoch/train_{name}', loss_meter.avg, epoch+1)


    def train_step(self, query_img, query_lab, class_indices):
        """Run one optimization forward pass with class-specific Iris priors."""
        
        if self.model_type == 'iris':
            batch_size = query_img.size(0)
            
            # Gather only the class priors needed by each sample in the batch.
            encoded_prior_list = []
            
            for stage_idx, stage_priors in enumerate(self.learnable_priors):
                batch_priors = []
                
                for b in range(batch_size):
                    sample_indices = class_indices[b]
                    sample_indices = sample_indices.to(stage_priors.device)
                    if sample_indices.numel() > 0:
                        min_index = int(sample_indices.min().item())
                        max_index = int(sample_indices.max().item())
                        if min_index < 0 or max_index >= stage_priors.shape[0]:
                            raise ValueError(
                                "class_indices are out of bounds for learnable "
                                f"priors: valid=[0, {stage_priors.shape[0] - 1}], "
                                f"got=[{min_index}, {max_index}]. Check that "
                                "the train/val dataset foreground_classes match "
                                "the learned prior count."
                            )
                    sample_priors = stage_priors[sample_indices]
                    batch_priors.append(sample_priors)
                
                # Shape after stacking: [B, num_classes, num_tokens, dim].
                batch_stage_priors = torch.stack(batch_priors)
                encoded_prior_list.append(batch_stage_priors)
            
            model = self.model.module if hasattr(self.model, 'module') else self.model
            predictions = model.forward_with_encoded_prior(
                tgt_img=query_img,
                encoded_prior_list=encoded_prior_list
            )
        else:
            predictions = self.model(query_img)
        
        total_loss = 0
        individual_losses = {}
        
        for loss_name, loss_fn in self.loss_fns.items():
            loss = loss_fn(predictions, query_lab)
            weighted_loss = loss * self.loss_weights.get(loss_name, 1.0)
            total_loss += weighted_loss
            individual_losses[loss_name] = loss.item()
        
        return total_loss, individual_losses
    
    def validate(self, epoch):
        """Validation step."""
        if not self.val_loader:
            return
        
        logging.info(f"Validating epoch {epoch+1}")
        
        model_to_eval = self.ema_model if self.ema_model is not None else self.model
        
        priors_to_eval = None
        if self.model_type == 'iris':
            # Validation uses the same learned prior tensors as training; EMA
            # priors are preferred when available.
            priors_to_eval = self.ema_learnable_priors if self.use_prior_ema and self.ema_learnable_priors is not None else self.learnable_priors
        
            evaluator = Evaluator(
                model=model_to_eval,
                data_loader=self.val_loader,
                config=self.config,
                device=self.device
            )
        else:
            evaluator = RegularEvaluator(
                model=model_to_eval,
                data_loader=self.val_loader,
                config=self.config,
                device=self.device
            )
        
        if self.model_type == 'iris':
            dice_mean, asd_mean, hd_mean = evaluator.run(learnable_priors=priors_to_eval)
        else:
            dice_mean, asd_mean, hd_mean = evaluator.run()
        
        dice_avg = float(np.mean(dice_mean))
        asd_avg = float(np.mean(asd_mean))
        hd_avg = float(np.mean(hd_mean))
        
        metrics = {
            'dataset': 'finetuning',
            'dice_mean': dice_avg,
            'asd_mean': asd_avg,
            'hd_mean': hd_avg,
            'dice_per_class': dice_mean.tolist(),
            'asd_per_class': asd_mean.tolist(),
            'hd_per_class': hd_mean.tolist()
        }
        
        self._log_metrics(epoch, 'finetuning', metrics)
        
        if dice_avg > self.best_metric:
            self.best_metric = dice_avg
            if is_master():
                self.save_checkpoint(epoch, is_best=True)
                logging.info(f"New best Dice score: {self.best_metric:.4f}")

    def _log_metrics(self, epoch, dataset_name, metrics):
        """
        Helper method to log metrics to console and tensorboard.
        
        Args:
            epoch: Current epoch number
            dataset_name: Name of the dataset
            metrics: Dictionary of metrics
        """
        logging.info(f"Validation on {dataset_name} - "
                    f"Dice: {metrics['dice_mean']:.4f}, "
                    f"ASD: {metrics['asd_mean']:.4f}, "
                    f"HD: {metrics['hd_mean']:.4f}")
        
        if 'dice_per_class' in metrics and metrics['dice_per_class'] is not None:
            dice_per_class = metrics['dice_per_class']
            asd_per_class = metrics['asd_per_class']
            hd_per_class = metrics['hd_per_class']
            
            logging.info(f"Per-class metrics for {dataset_name}:")
            logging.info(f"{'Class':<8} | {'Dice':<8} | {'ASD':<8} | {'HD':<8}")
            logging.info("-" * 40)
            
            for i, (dice, asd, hd) in enumerate(zip(dice_per_class, asd_per_class, hd_per_class)):
                logging.info(f"{i:<8} | {dice:<8.4f} | {asd:<8.2f} | {hd:<8.2f}")
            
            dice_str = ", ".join([f"{d:.3f}" for d in dice_per_class])
            logging.info(f"Dice per class [{dataset_name}]: [{dice_str}]")
        
        if is_master() and self.writer is not None:
            dataset_prefix = f"val/{dataset_name}"
            
            self.writer.add_scalar(f'{dataset_prefix}/dice_mean', metrics['dice_mean'], epoch+1)
            self.writer.add_scalar(f'{dataset_prefix}/asd_mean', metrics['asd_mean'], epoch+1)
            self.writer.add_scalar(f'{dataset_prefix}/hd_mean', metrics['hd_mean'], epoch+1)
            
            if 'dice_per_class' in metrics and metrics['dice_per_class'] is not None:
                class_prefix = f"{dataset_prefix}/classes"
                
                for i, (dice, asd, hd) in enumerate(zip(
                    metrics['dice_per_class'], 
                    metrics['asd_per_class'], 
                    metrics['hd_per_class']
                )):
                    self.writer.add_scalar(f'{class_prefix}/class_{i}/dice', dice, epoch+1)
                    self.writer.add_scalar(f'{class_prefix}/class_{i}/asd', asd, epoch+1)
                    self.writer.add_scalar(f'{class_prefix}/class_{i}/hd', hd, epoch+1)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint including learnable priors and their EMA (only for Iris)."""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
            'model_type': self.model_type,
            'num_classes': self.num_classes
        }
        
        if self.model_type == 'iris' and self.learnable_priors is not None:
            checkpoint['learnable_priors'] = self.learnable_priors.state_dict()
        
        if self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        if self.use_ema:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
            
        if self.use_prior_ema and self.ema_learnable_priors is not None:
            checkpoint['ema_learnable_priors'] = self.ema_learnable_priors.state_dict()
        
        filename = 'best.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth'
        filepath = os.path.join(self.config['run']['output_dir'], filename)
        torch.save(checkpoint, filepath)
        
        logging.info(f"Saved checkpoint: {filepath}")
        
        if is_best and self.model_type == 'iris':
            # Export the priors that validation used for the best checkpoint.
            priors_to_save = self.ema_learnable_priors if self.use_prior_ema and self.ema_learnable_priors is not None else self.learnable_priors
            
            priors_path = os.path.join(self.config['run']['output_dir'], 'learned_priors.pth')
            torch.save({
                'priors': [p.detach().cpu() for p in priors_to_save],
                'num_classes': self.num_classes,
                'class_names': self.finetune_config.get('class_names', {}),
                'is_ema': self.use_prior_ema and priors_to_save is self.ema_learnable_priors
            }, priors_path)
            logging.info(f"Saved learned priors: {priors_path} (EMA: {self.use_prior_ema and priors_to_save is self.ema_learnable_priors})")
    
    def _resume_checkpoint(self):
        """Resume from checkpoint."""
        resume_path = self.config['run']['resume']
        checkpoint = torch.load(resume_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.model_type == 'iris' and 'learnable_priors' in checkpoint:
            self.learnable_priors.load_state_dict(checkpoint['learnable_priors'])
        
        if self.use_ema and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            
        if self.use_prior_ema and 'ema_learnable_priors' in checkpoint:
            self.ema_learnable_priors.load_state_dict(checkpoint['ema_learnable_priors'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.scheduler and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        self.start_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        
        logging.info(f"Resumed from epoch {self.start_epoch}")
