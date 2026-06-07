"""Downstream classification trainer using the MASS encoder.

The trainer wraps supervised 3D classification experiments, including metrics
such as AUC/F1/balanced accuracy, checkpointing, and optional initialization
from a pretrained MASS encoder.
"""

import os
import time
import logging
import json
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
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

from utils.distributed import (
    is_master, get_rank, get_local_rank, get_world_size,
    is_distributed, reduce_tensor
)
from utils.checkpoint import (
    save_checkpoint, prepare_checkpoint_state
)
from utils.metrics_utils import AverageMeter, ProgressMeter
from utils.registry import (
    get_optimizer, get_scheduler, get_criterion,
    get_dataset, register_trainer
)
import gc

class AttentionPooling(nn.Module):
    """Attention-based pooling layer."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            pooled: [B, C]
        """
        attn = self.attention(x)  # [B, 1, D, H, W]
        attn = torch.softmax(attn.view(x.size(0), -1), dim=1)  # [B, D*H*W]
        attn = attn.view(x.size(0), 1, x.size(2), x.size(3), x.size(4))

        out = (x * attn).sum(dim=[2, 3, 4])  # [B, C]
        return out


class ClassificationHead(nn.Module):
    """Classification head with pooling."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pooling_type: str = 'avg',
        dropout: float = 0.5
    ):
        """
        Args:
            in_channels: Number of input channels from encoder
            num_classes: Number of output classes
            pooling_type: 'avg', 'max', 'attention', 'none'
            dropout: Dropout rate
        """
        super().__init__()

        self.pooling_type = pooling_type

        # Pooling layer
        if pooling_type == 'avg':
            self.pool = nn.AdaptiveAvgPool3d(1)
        elif pooling_type == 'max':
            self.pool = nn.AdaptiveMaxPool3d(1)
        elif pooling_type == 'attention':
            self.pool = AttentionPooling(in_channels)
        elif pooling_type == 'none':
            self.pool = None
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map from encoder [B, C, D, H, W]
        Returns:
            logits: [B, num_classes]
        """

        # Pool
        if self.pooling_type in ['avg', 'max']:
            x = self.pool(x)  # [B, C, 1, 1, 1]
            x = x.view(x.size(0), -1)  # [B, C]
        elif self.pooling_type == 'attention':  # attention
            x = self.pool(x)  # [B, C]
        elif self.pooling_type == 'none':
            pass
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Classify
        x = self.dropout(x)
        x = self.fc(x)

        return x


@register_trainer("classification")
class ClassificationTrainer:
    """
    Trainer for classification finetuning.

    Loads pretrained encoder, adds classification head, and finetunes.
    Supports freezing/unfreezing different model parts.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize classification trainer."""
        self.config = config
        self.device = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")

        self.epochs = config.get('epochs', 100)
        self.start_epoch = 0
        self.best_metric = 0.0  # Classification uses accuracy (higher is better)
        self.current_iter = 0

        self.classification_config = config.get('classification', {})
        self.multi_label = bool(self.classification_config.get('multi_label', False))
        self._sync_classification_config()

        logging.info('Start classification trainer initialization')

        self._setup_logging()
        self._setup_amp()
        self._setup_model()
        self._setup_data()
        self._setup_criterion()
        self._setup_optimizer()

        logging.info('Classification trainer initialization done')

        # Resume from checkpoint if requested
        if 'resume' in config.get('run', {}):
            self._resume_checkpoint()

    def _sync_classification_config(self):
        """Keep model head, datasets, and criterion on the same class count."""
        data_config = self.config.get('data', {})
        train_config = data_config.get('train', {})
        val_config = data_config.get('val', {})

        candidate_values = []
        for source_name, source_config in (
            ('classification', self.classification_config),
            ('data.train', train_config),
            ('data.val', val_config),
        ):
            value = source_config.get('num_classes')
            if value in [None, 'auto', 'Auto', 'AUTO']:
                continue
            candidate_values.append((source_name, int(value)))

        if not candidate_values:
            raise ValueError(
                "Classification requires `num_classes` in either "
                "`classification`, `data.train`, or `data.val`."
            )

        # Prefer the dataset value if present, because it controls label loading
        # and class weights. The model head must match it.
        dataset_values = [
            value for source_name, value in candidate_values
            if source_name.startswith('data.')
        ]
        resolved_num_classes = dataset_values[0] if dataset_values else candidate_values[0][1]

        for source_name, value in candidate_values:
            if value != resolved_num_classes:
                logging.warning(
                    "%s.num_classes=%s does not match resolved "
                    "classification num_classes=%s. Using %s for the model "
                    "head, dataset, and loss.",
                    source_name,
                    value,
                    resolved_num_classes,
                    resolved_num_classes,
                )

        if train_config.get('num_classes') not in [None, 'auto', 'Auto', 'AUTO']:
            train_num_classes = int(train_config['num_classes'])
            if train_num_classes != resolved_num_classes:
                raise ValueError(
                    "data.train.num_classes must match the resolved class "
                    f"count. Got {train_num_classes} vs {resolved_num_classes}."
                )

        if val_config.get('num_classes') not in [None, 'auto', 'Auto', 'AUTO']:
            val_num_classes = int(val_config['num_classes'])
            if val_num_classes != resolved_num_classes:
                raise ValueError(
                    "data.val.num_classes must match the resolved class "
                    f"count. Got {val_num_classes} vs {resolved_num_classes}."
                )

        self.num_classes = resolved_num_classes
        self.classification_config['num_classes'] = self.num_classes
        self.config.setdefault('classification', {})['num_classes'] = self.num_classes

        for split_name in ('train', 'val'):
            split_config = self.config.setdefault('data', {}).setdefault(split_name, {})
            split_config['num_classes'] = self.num_classes
            split_config['multi_label'] = self.multi_label

        logging.info(
            "Classification task: num_classes=%d, multi_label=%s",
            self.num_classes,
            self.multi_label,
        )

    def _setup_amp(self):
        """Setup Automatic Mixed Precision."""
        amp_config = self.config.get('amp', {})
        if amp_config.get('enabled', False):
            amp_dtype_str = amp_config.get('dtype', 'float16')

            if amp_dtype_str == 'float16':
                self.amp_dtype = torch.float16
                self.scaler = GradScaler('cuda')
                logging.info("Using FP16 with GradScaler")
            elif amp_dtype_str == 'bfloat16':
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # No scaler needed for BF16
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
            logging.info("AMP disabled, using FP32")

    def _setup_logging(self):
        """Setup logging and tensorboard."""
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        if is_master():
            output_dir = Path(self.config['run']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = output_dir

            log_handlers = [
                logging.StreamHandler(),
                logging.FileHandler(output_dir / 'train.log')
            ]

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=log_handlers,
                force=True
            )

            self.writer = SummaryWriter(log_dir=output_dir / 'tensorboard')

            with open(output_dir / 'config.json', 'w') as f:
                json.dump(self.config, f, indent=2)

            logging.info("Logging initialized (master process only)")
        else:
            logging.basicConfig(
                level=logging.WARNING,
                handlers=[logging.NullHandler()],
                force=True
            )
            self.output_dir = None
            self.writer = None

    def _setup_model(self):
        """Setup an Iris/MASS encoder and classification head."""
        encoder_config = self.classification_config.get('encoder', {})
        pretrained_path = encoder_config.get('pretrained_checkpoint')
        self.encoder_type = encoder_config.get('type', 'iris').lower()
        if self.encoder_type != 'iris':
            raise ValueError(
                "The open-source classification linear probing example only supports encoder.type='iris'. "
                f"Got: {self.encoder_type}"
            )
        freeze_encoder = bool(encoder_config.get('freeze', True))

        head_config = self.classification_config.get('head', {})
        pooling_type = head_config.get('pooling_type', 'avg')
        dropout = float(head_config.get('dropout', 0.5))

        model_config = self.config.get('model', {}).copy()
        model_config.pop('type', None)

        from models.iris import Iris

        full_model = Iris(**model_config)
        self.encoder = full_model.encoder

        # The classification head consumes the deepest encoder feature map.
        if 'channels' in model_config and model_config['channels']:
            encoder_channels = model_config['channels'][-1]
        else:
            base_ch = model_config.get('base_ch', 32)
            encoder_channels = 16 * base_ch

        if pretrained_path not in [None, False, 'None', 'False', 'none', 'false', '']:
            logging.info(f"Loading MASS checkpoint from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            prefer_ema = encoder_config.get('use_ema_checkpoint', True)

            # Load into the full Iris module first so checkpoint key names match,
            # then keep only the encoder for classification.
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
            missing, unexpected = full_model.load_state_dict(state_dict, strict=False)
            if missing:
                logging.warning(f"Missing keys when loading pretrained weights: {missing[:20]}")
            if unexpected:
                logging.warning(f"Unexpected keys when loading pretrained weights: {unexpected[:20]}")
        else:
            logging.info("No pretrained checkpoint provided. Using random initialization.")

        if freeze_encoder:
            logging.info("Freezing encoder weights")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            logging.info("Encoder weights are trainable")

        self.encoder = self.encoder.to(self.device)
        self.classification_head = ClassificationHead(
            in_channels=encoder_channels,
            num_classes=self.num_classes,
            pooling_type=pooling_type,
            dropout=dropout,
        ).to(self.device)

        if is_distributed():
            self.encoder = nn.parallel.DistributedDataParallel(
                self.encoder,
                device_ids=[get_local_rank()],
                find_unused_parameters=False,
            )
            self.classification_head = nn.parallel.DistributedDataParallel(
                self.classification_head,
                device_ids=[get_local_rank()],
            )

        if is_master():
            encoder_params = sum(p.numel() for p in self.encoder.parameters())
            encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            head_params = sum(p.numel() for p in self.classification_head.parameters())
            logging.info(f"Encoder parameters: {encoder_params/1e6:.2f}M (trainable: {encoder_trainable/1e6:.2f}M)")
            logging.info(f"Classification head parameters: {head_params/1e6:.2f}M")

    @staticmethod
    def _select_feature_map(features):
        """Select the deepest feature map from the Iris encoder output."""
        if isinstance(features, (list, tuple)):
            return features[0]
        return features

    def _setup_criterion(self):
        """Setup loss criterion with class balancing."""

        class_weights = None
        if hasattr(self.train_loader.dataset, 'class_weights'):
            class_weights_np = self.train_loader.dataset.class_weights
            if class_weights_np is not None:
                class_weights = torch.from_numpy(class_weights_np).float().to(self.device)

        if self.multi_label:
            # Multi-label tasks use independent binary targets per class.
            if class_weights is not None:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
                if is_master():
                    logging.info(f"Using BCEWithLogitsLoss with pos_weight: {class_weights.cpu().numpy()}")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                if is_master():
                    logging.info("Using BCEWithLogitsLoss without pos_weight")
        else:
            # Single-label tasks use one mutually exclusive class target.
            criterion_config = self.config.get('criterion', {}).copy()

            if criterion_config:
                criterion_type = criterion_config.pop('type', 'cross_entropy')

                if class_weights is not None and 'weight' not in criterion_config:
                    criterion_config['weight'] = class_weights

                self.criterion = get_criterion(criterion_type)(**criterion_config)
            else:
                if class_weights is not None:
                    self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                    if is_master():
                        logging.info(f"Using CrossEntropyLoss with class weights: {class_weights.cpu().numpy()}")
                else:
                    self.criterion = nn.CrossEntropyLoss()
                    if is_master():
                        logging.info("Using CrossEntropyLoss without class weights")


    def _setup_data(self):
        """Setup dataloaders."""
        data_config = self.config.get('data', {})
        loader_config = data_config.get('loader', {})
        augmentation_config = self.config.get('augmentation', {})

        train_config = data_config.get('train', {}).copy()
        train_config['augmentation_config'] = augmentation_config
        train_config['num_classes'] = self.num_classes
        train_config['multi_label'] = self.multi_label
        train_dataset = get_dataset('classification')(**train_config)

        if is_distributed():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                shuffle=True
            )
        else:
            train_sampler = None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=loader_config.get('batch_size', 4),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=loader_config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=loader_config.get('persistent_workers', True) and loader_config.get('num_workers', 4) > 0,
        )

        val_config = data_config.get('val', {}).copy()
        val_config['num_classes'] = self.num_classes
        val_config['multi_label'] = self.multi_label
        val_dataset = get_dataset('classification')(**val_config)

        if val_dataset.num_classes != train_dataset.num_classes:
            raise ValueError(
                "Train and validation classification datasets disagree on "
                f"num_classes: train={train_dataset.num_classes}, "
                f"val={val_dataset.num_classes}"
            )

        if is_distributed():
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                shuffle=False
            )
        else:
            val_sampler = None

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
        )

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        params = []

        # Encoder parameters (if not frozen)
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if encoder_params:
            params.append({
                'params': encoder_params,
                'lr': float(self.classification_config.get('encoder_lr', 1e-5)),
                'name': 'encoder'
            })


        # Head parameters
        params.append({
            'params': self.classification_head.parameters(),
            'lr': float(self.classification_config.get('head_lr', 1e-3)),
            'name': 'head'
        })

        optimizer_config = self.config.get('optimizer', {}).copy()
        optimizer_type = optimizer_config.pop('type', 'adam')
        optimizer_config.pop('lr', None)  # Use param group LRs

        self.optimizer = get_optimizer(optimizer_type)(params, **optimizer_config)

        scheduler_config = self.config.get('scheduler', {}).copy()
        if scheduler_config:
            scheduler_type = scheduler_config.pop('type')

            if 'total_steps' in scheduler_config and scheduler_config['total_steps'] is None:
                scheduler_config['total_steps'] = len(self.train_loader) * self.epochs

            self.scheduler = get_scheduler(scheduler_type)(self.optimizer, **scheduler_config)
        else:
            self.scheduler = None

        if is_master():
            logging.info("Optimizer parameter groups:")
            for group in self.optimizer.param_groups:
                logging.info(f"  - {group['name']}: lr={group['lr']}")

    def train(self):
        """Main training loop."""
        logging.info("Starting training")

        for epoch in range(self.start_epoch, self.epochs):
            if is_distributed():
                self.train_loader.sampler.set_epoch(epoch)

            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            if self.scheduler:
                self.scheduler.step()

            val_metrics = self._validate(epoch)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            if is_master():
                self._log_metrics(epoch, train_metrics, val_metrics)

            is_best = val_metrics['accuracy'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['accuracy']

            if is_master():
                self._save_checkpoint(epoch, is_best)

        if is_master():
            logging.info(f"Training completed. Best accuracy: {self.best_metric:.4f}")
            self.writer.close()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.encoder.train()
        self.classification_head.train()

        # Keep frozen encoders in eval mode so normalization layers stay fixed.
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()

        losses = AverageMeter('Loss', ':.4f')
        accuracies = AverageMeter('Acc', ':.4f')
        batch_time = AverageMeter('Time', ':6.3f')

        # Track class-wise stats in addition to the global accuracy.
        per_class_correct = np.zeros(self.num_classes)
        per_class_total = np.zeros(self.num_classes)
        per_class_losses = [[] for _ in range(self.num_classes)]

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()

            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    features = self._select_feature_map(self.encoder(images))

                    logits = self.classification_head(features)

                    loss = self.criterion(logits, labels)

                    if self.multi_label:
                        per_sample_loss = F.binary_cross_entropy_with_logits(
                            logits, labels, reduction='none'
                        ).mean(dim=1)
                    else:
                        per_sample_loss = F.cross_entropy(logits, labels, reduction='none')

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.classification_head.parameters()),
                        max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.classification_head.parameters()),
                        max_norm=1.0
                    )
                    self.optimizer.step()
            else:
                features = self._select_feature_map(self.encoder(images))
                logits = self.classification_head(features)
                loss = self.criterion(logits, labels)

                if self.multi_label:
                    per_sample_loss = F.binary_cross_entropy_with_logits(
                        logits, labels, reduction='none'
                    ).mean(dim=1)
                else:
                    per_sample_loss = F.cross_entropy(logits, labels, reduction='none')

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.classification_head.parameters()),
                    max_norm=1.0
                )
                self.optimizer.step()

            if self.multi_label:
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == labels).float().mean()

                preds_np = preds.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                for c in range(self.num_classes):
                    correct = (preds_np[:, c] == labels_np[:, c]).sum()
                    per_class_correct[c] += correct
                    per_class_total[c] += len(labels_np)

                    class_mask = labels_np[:, c] == 1
                    if class_mask.sum() > 0:
                        per_class_losses[c].extend(
                            per_sample_loss[class_mask].detach().cpu().numpy().tolist()
                        )
            else:
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean()

                preds_np = preds.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                per_sample_loss_np = per_sample_loss.detach().cpu().numpy()

                for c in range(self.num_classes):
                    class_mask = labels_np == c
                    if class_mask.sum() > 0:
                        per_class_total[c] += class_mask.sum()
                        per_class_correct[c] += (preds_np[class_mask] == c).sum()
                        per_class_losses[c].extend(per_sample_loss_np[class_mask].tolist())

            if is_distributed():
                loss = reduce_tensor(loss)
                acc = reduce_tensor(acc)

            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))
            batch_time.update(time.time() - batch_start, images.size(0))

            if batch_idx % 10 == 0 and is_master():
                logging.info(
                    f"Epoch [{epoch+1}/{self.epochs}] "
                    f"Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Acc: {accuracies.val:.4f} ({accuracies.avg:.4f})"
                )

            if is_master() and self.writer is not None and batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', losses.avg, global_step)
                self.writer.add_scalar('Train/Accuracy', accuracies.avg, global_step)
                self.writer.add_scalar('Train/BatchTime', batch_time.avg, global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)

        per_class_acc = {}
        per_class_loss = {}
        for c in range(self.num_classes):
            if per_class_total[c] > 0:
                per_class_acc[f'class_{c}'] = per_class_correct[c] / per_class_total[c]
            else:
                per_class_acc[f'class_{c}'] = 0.0

            if len(per_class_losses[c]) > 0:
                per_class_loss[f'class_{c}'] = np.mean(per_class_losses[c])
            else:
                per_class_loss[f'class_{c}'] = 0.0

        return {
            'loss': losses.avg,
            'accuracy': accuracies.avg,
            'per_class_accuracy': per_class_acc,
            'per_class_loss': per_class_loss
        }


    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.encoder.eval()

        self.classification_head.eval()

        all_preds = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label']

                if self.use_amp:
                    with autocast(device_type='cuda', dtype=self.amp_dtype):
                        features = self._select_feature_map(self.encoder(images))
                        logits = self.classification_head(features)

                else:
                    features = self._select_feature_map(self.encoder(images))
                    logits = self.classification_head(features)


                all_logits.append(logits.cpu())
                all_labels.append(labels)

                if self.multi_label:
                    preds = (torch.sigmoid(logits) > 0.5).cpu()
                else:
                    preds = torch.argmax(logits, dim=1).cpu()

                all_preds.append(preds)

        # Concatenate all results
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_logits = torch.cat(all_logits, dim=0).float().numpy()

        # Gather from all processes if distributed
        if is_distributed():
            world_size = get_world_size()
            all_preds_list = [None] * world_size
            all_labels_list = [None] * world_size
            all_logits_list = [None] * world_size

            dist.all_gather_object(all_preds_list, all_preds)
            dist.all_gather_object(all_labels_list, all_labels)
            dist.all_gather_object(all_logits_list, all_logits)

            all_preds = np.concatenate(all_preds_list, axis=0)
            all_labels = np.concatenate(all_labels_list, axis=0)
            all_logits = np.concatenate(all_logits_list, axis=0)

        metrics = self._calculate_metrics(all_preds, all_labels, all_logits)

        if is_master():
            logging.info(
                f"Validation Epoch [{epoch+1}/{self.epochs}] "
                f"Acc: {metrics['accuracy']:.4f} "
                f"BMAC: {metrics.get('balanced_accuracy', 0):.4f} "
                f"F1: {metrics.get('f1', 0):.4f} "
                f"AUROC: {metrics.get('auroc', 0):.4f}"
            )

        return metrics

    def _calculate_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        logits: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics including per-class metrics."""
        metrics = {}

        # Overall accuracy
        if self.multi_label:
            metrics['accuracy'] = ((preds == labels).sum() / labels.size)
        else:
            metrics['accuracy'] = (preds == labels).mean()

        if not self.multi_label:
            # Overall metrics
            metrics['balanced_accuracy'] = balanced_accuracy_score(labels, preds)

            if self.num_classes == 2:
                metrics['f1'] = f1_score(labels, preds)
            else:
                metrics['f1'] = f1_score(labels, preds, average='macro')

            if self.num_classes == 2:
                probs = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
                metrics['auroc'] = roc_auc_score(labels, probs)
            else:
                probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
                metrics['auroc'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')

            # Per-class metrics
            per_class_acc = {}
            per_class_ba = {}
            per_class_f1 = {}
            per_class_auroc = {}

            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

            for c in range(self.num_classes):
                # Accuracy
                class_mask = labels == c
                if class_mask.sum() > 0:
                    per_class_acc[f'class_{c}'] = (preds[class_mask] == c).mean()
                else:
                    per_class_acc[f'class_{c}'] = 0.0

                # Binary metrics for this class vs rest
                binary_labels = (labels == c).astype(int)
                binary_preds = (preds == c).astype(int)

                if len(np.unique(binary_labels)) > 1:  # Check if both classes present
                    per_class_ba[f'class_{c}'] = balanced_accuracy_score(binary_labels, binary_preds)
                    per_class_f1[f'class_{c}'] = f1_score(binary_labels, binary_preds)
                    per_class_auroc[f'class_{c}'] = roc_auc_score(binary_labels, probs[:, c])
                else:
                    per_class_ba[f'class_{c}'] = 0.0
                    per_class_f1[f'class_{c}'] = 0.0
                    per_class_auroc[f'class_{c}'] = 0.0

            metrics['per_class_accuracy'] = per_class_acc
            metrics['per_class_balanced_accuracy'] = per_class_ba
            metrics['per_class_f1'] = per_class_f1
            metrics['per_class_auroc'] = per_class_auroc

        else:
            probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
            metrics['auroc'] = roc_auc_score(labels, probs, average='macro')
            metrics['f1'] = f1_score(labels, preds, average='macro')

            label_balanced_accs = []
            per_class_acc = {}
            per_class_ba = {}
            per_class_f1 = {}
            per_class_auroc = {}

            for c in range(self.num_classes):
                # Per-class metrics
                per_class_acc[f'class_{c}'] = (preds[:, c] == labels[:, c]).mean()

                if len(np.unique(labels[:, c])) > 1:
                    ba = balanced_accuracy_score(labels[:, c], preds[:, c])
                    per_class_ba[f'class_{c}'] = ba
                    per_class_f1[f'class_{c}'] = f1_score(labels[:, c], preds[:, c])
                    per_class_auroc[f'class_{c}'] = roc_auc_score(labels[:, c], probs[:, c])
                    label_balanced_accs.append(ba)
                else:
                    per_class_ba[f'class_{c}'] = 0.0
                    per_class_f1[f'class_{c}'] = 0.0
                    per_class_auroc[f'class_{c}'] = 0.0

            metrics['balanced_accuracy'] = float(np.mean(label_balanced_accs)) if label_balanced_accs else 0.0
            metrics['per_class_accuracy'] = per_class_acc
            metrics['per_class_balanced_accuracy'] = per_class_ba
            metrics['per_class_f1'] = per_class_f1
            metrics['per_class_auroc'] = per_class_auroc

        return metrics


    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to tensorboard including per-class metrics."""
        # Overall training metrics
        self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch + 1)
        self.writer.add_scalar('epoch/train_accuracy', train_metrics['accuracy'], epoch + 1)

        # Per-class training metrics
        if 'per_class_accuracy' in train_metrics:
            for class_name, acc in train_metrics['per_class_accuracy'].items():
                self.writer.add_scalar(f'train_per_class/accuracy/{class_name}', acc, epoch + 1)

        if 'per_class_loss' in train_metrics:
            for class_name, loss in train_metrics['per_class_loss'].items():
                self.writer.add_scalar(f'train_per_class/loss/{class_name}', loss, epoch + 1)

        # Overall validation metrics
        for key, value in val_metrics.items():
            if not key.startswith('per_class'):
                self.writer.add_scalar(f'epoch/val_{key}', value, epoch + 1)

        # Per-class validation metrics
        if 'per_class_accuracy' in val_metrics:
            for class_name, acc in val_metrics['per_class_accuracy'].items():
                self.writer.add_scalar(f'val_per_class/accuracy/{class_name}', acc, epoch + 1)

        if 'per_class_balanced_accuracy' in val_metrics:
            for class_name, ba in val_metrics['per_class_balanced_accuracy'].items():
                self.writer.add_scalar(f'val_per_class/balanced_accuracy/{class_name}', ba, epoch + 1)

        if 'per_class_f1' in val_metrics:
            for class_name, f1 in val_metrics['per_class_f1'].items():
                self.writer.add_scalar(f'val_per_class/f1/{class_name}', f1, epoch + 1)

        if 'per_class_auroc' in val_metrics:
            for class_name, auroc in val_metrics['per_class_auroc'].items():
                self.writer.add_scalar(f'val_per_class/auroc/{class_name}', auroc, epoch + 1)

        # Learning rate
        if self.scheduler:
            for i, param_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(
                    f'lr/{param_group["name"]}',
                    param_group['lr'],
                    epoch + 1
                )


    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save classification head and encoder state for resume/evaluation."""
        state = {
            'epoch': epoch + 1,
            'iteration': self.current_iter,
            'best_metric': self.best_metric,
            'config': self.config,
        }

        state['encoder_state_dict'] = (
            self.encoder.module.state_dict() if is_distributed()
            else self.encoder.state_dict()
        )
        state['head_state_dict'] = (
            self.classification_head.module.state_dict() if is_distributed()
            else self.classification_head.state_dict()
        )

        state['optimizer'] = self.optimizer.state_dict()
        if self.scheduler: state['scheduler'] = self.scheduler.state_dict()
        if self.scaler: state['scaler'] = self.scaler.state_dict()

        save_checkpoint(
            state,
            self.config,
            tag='latest',
            is_best=is_best
        )

        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                state,
                self.config,
                tag=f'epoch_{epoch+1}',
                is_best=False
            )
            logging.info(f"Saved periodic checkpoint at epoch {epoch+1}")

        if is_best:
            logging.info(f"Saved best checkpoint with accuracy: {self.best_metric:.4f}")


    def _resume_checkpoint(self):
        """Resume from checkpoint."""
        resume_path = self.config['run']['resume']
        logging.info(f"Resuming from {resume_path}")

        checkpoint = torch.load(resume_path, map_location='cpu')

        if is_distributed():
            self.encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
            self.classification_head.module.load_state_dict(checkpoint['head_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.classification_head.load_state_dict(checkpoint['head_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.scheduler and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        if self.scaler and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.current_iter = checkpoint.get('iteration', 0)

        logging.info(f"Resumed from epoch {self.start_epoch}, iteration {self.current_iter}")
