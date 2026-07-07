"""Dataset configuration utilities for FLAME training."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Mapping

import torch
from torch.utils.data import DataLoader, ConcatDataset

from .localforgerydataset import LocalForgeryDataset
from .train_utils import custom_collate_fn
from .validation_manifest import apply_manifest_to_dataset, manifest_entries_by_name

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    path: str
    enabled: bool = True
    allow_multiple_targets: bool = False
    authentic_ratio: float = 0.0
    is_training: bool = True
    sample_ratio: float = 1.0  # Ratio of samples to use from this dataset (0.0 to 1.0)
    force_resize: Optional[bool] = None
    
    # Optional overrides for dataset-specific settings
    img_size: Optional[int] = None
    contrastive_blur: Optional[bool] = None
    perturbation_type: Optional[str] = None
    perturbation_intensity: Optional[float] = None
    authentic_source_dir: Optional[str] = None  # Optional; authentic images use dataset root/source


@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader."""
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = False


@dataclass
class DatasetManagerConfig:
    """Main configuration for dataset management."""
    data_root: str = "FLAME/data"
    train_datasets: List[DatasetConfig] = None
    val_datasets: List[DatasetConfig] = None
    train_loader: DataLoaderConfig = None
    val_loader: DataLoaderConfig = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.train_datasets is None:
            self.train_datasets = []
        if self.val_datasets is None:
            self.val_datasets = []
        if self.train_loader is None:
            self.train_loader = DataLoaderConfig()
        if self.val_loader is None:
            self.val_loader = DataLoaderConfig()


class DatasetManager:
    """Manages dataset creation and configuration."""
    
    def __init__(self, config: DatasetManagerConfig):
        self.config = config
        
    def create_dataset(
        self,
        dataset_config: DatasetConfig,
        # Global defaults
        img_size: int = 512,
        contrastive_blur: bool = False,
        perturbation_type: str = "gaussian_blur/gaussian_noise",
        perturbation_intensity: float = 0.75,
        authentic_source_dir: Optional[str] = None,
        force_resize: bool = False,
    ) -> Optional[LocalForgeryDataset]:
        """Create a single dataset from configuration."""
        if not dataset_config.enabled:
            logger.info("Dataset %s is disabled, skipping", dataset_config.name)
            return None
            
        # Resolve full path
        if os.path.isabs(dataset_config.path):
            full_path = dataset_config.path
        else:
            full_path = os.path.join(self.config.data_root, dataset_config.path)
            
        if not os.path.exists(full_path):
            logger.warning("Dataset path %s not found; skipping %s", full_path, dataset_config.name)
            return None
            
        # Use dataset-specific overrides or global defaults
        final_img_size = dataset_config.img_size or img_size
        final_contrastive_blur = dataset_config.contrastive_blur if dataset_config.contrastive_blur is not None else contrastive_blur
        final_perturbation_type = dataset_config.perturbation_type or perturbation_type
        final_perturbation_intensity = dataset_config.perturbation_intensity if dataset_config.perturbation_intensity is not None else perturbation_intensity
        final_force_resize = force_resize if dataset_config.force_resize is None else dataset_config.force_resize
        
        try:
            dataset = LocalForgeryDataset(
                full_path,
                img_size=final_img_size,
                allow_multiple_targets=dataset_config.allow_multiple_targets,
                is_training=dataset_config.is_training,
                authentic_ratio=dataset_config.authentic_ratio,
                authentic_source_dir=None,
                sample_ratio=dataset_config.sample_ratio,
                force_resize=final_force_resize,
            )
            
            logger.info(
                "Created dataset %s: %s samples from %s",
                dataset_config.name,
                len(dataset),
                full_path
            )
            return dataset
            
        except Exception as e:
            logger.error("Failed to create dataset %s: %s", dataset_config.name, e)
            return None
    
    def create_train_datasets(
        self,
        img_size: int = 512,
        contrastive_blur: bool = False,
        perturbation_type: str = "gaussian_blur/gaussian_noise",
        perturbation_intensity: float = 0.75,
        authentic_ratio: float = 0.0,
        authentic_source_dir: Optional[str] = None,
        force_resize: bool = False,
    ) -> tuple[ConcatDataset, DataLoader]:
        """Create training datasets and dataloader."""
        datasets = []
        
        for dataset_config in self.config.train_datasets:
            # Override authentic_ratio for training datasets with global setting
            config_copy = DatasetConfig(**asdict(dataset_config))
            if config_copy.authentic_ratio == 0.0:  # Only override if not explicitly set
                config_copy.authentic_ratio = authentic_ratio
                
            dataset = self.create_dataset(
                config_copy,
                img_size=img_size,
                contrastive_blur=contrastive_blur,
                perturbation_type=perturbation_type,
                perturbation_intensity=perturbation_intensity,
                authentic_source_dir=None,
                force_resize=force_resize,
            )
            if dataset is not None:
                datasets.append(dataset)
        
        if not datasets:
            raise ValueError("No training datasets found! Please check your configuration and data directory.")
        
        # Combine datasets
        combined_dataset = ConcatDataset(datasets)
        total_samples = sum(len(d) for d in datasets)
        
        logger.info(
            "Combined training dataset | samples=%s sources=%s",
            total_samples,
            len(datasets),
        )
        
        # Create dataloader
        train_num_workers = self.config.train_loader.num_workers
        train_loader = DataLoader(
            combined_dataset,
            batch_size=self.config.train_loader.batch_size,
            shuffle=self.config.train_loader.shuffle,
            num_workers=train_num_workers,
            pin_memory=self.config.train_loader.pin_memory,
            drop_last=self.config.train_loader.drop_last,
            collate_fn=custom_collate_fn,
            persistent_workers=(
                self.config.train_loader.persistent_workers and train_num_workers > 0
            ),
        )
        
        return combined_dataset, train_loader
    
    def create_val_datasets(
        self,
        img_size: int = 512,
        contrastive_blur: bool = False,
        perturbation_type: str = "gaussian_blur/gaussian_noise",
        perturbation_intensity: float = 0.75,
        authentic_ratio: float = 0.0,
        authentic_source_dir: Optional[str] = None,
        force_resize: bool = False,
        validation_manifest: Optional[Mapping[str, Any]] = None,
    ) -> tuple[Optional[ConcatDataset], Optional[DataLoader]]:
        """Create validation datasets and dataloader."""
        datasets = []
        manifest_by_name = (
            manifest_entries_by_name(validation_manifest)
            if validation_manifest is not None
            else {}
        )
        
        for dataset_config in self.config.val_datasets:
            config_copy = DatasetConfig(**asdict(dataset_config))
            if config_copy.authentic_ratio == 0.0 and authentic_ratio > 0.0:
                config_copy.authentic_ratio = authentic_ratio
            if dataset_config.name in manifest_by_name:
                # Apply fixed manifests against the full sample pool.  If we
                # let LocalForgeryDataset consume sample_ratio first, the
                # manifest may reference samples that were randomly dropped.
                config_copy.sample_ratio = 1.0

            dataset = self.create_dataset(
                config_copy,
                img_size=img_size,
                contrastive_blur=contrastive_blur,
                perturbation_type=perturbation_type,
                perturbation_intensity=perturbation_intensity,
                authentic_source_dir=None,
                force_resize=force_resize,
            )
            if dataset is not None:
                if dataset_config.name in manifest_by_name:
                    apply_manifest_to_dataset(
                        dataset,
                        manifest_by_name[dataset_config.name],
                        dataset_name=dataset_config.name,
                    )
                    logger.info(
                        "Applied validation manifest for %s: %s samples",
                        dataset_config.name,
                        len(dataset),
                    )
                datasets.append(dataset)
        
        if not datasets:
            logger.warning("No validation datasets found.")
            return None, None
        
        # Combine datasets
        combined_dataset = ConcatDataset(datasets)
        total_samples = sum(len(d) for d in datasets)
        
        logger.info(
            "Combined validation dataset | samples=%s sources=%s",
            total_samples,
            len(datasets),
        )
        
        # Create dataloader
        val_num_workers = self.config.val_loader.num_workers
        val_loader = DataLoader(
            combined_dataset,
            batch_size=self.config.val_loader.batch_size,
            shuffle=self.config.val_loader.shuffle,
            num_workers=val_num_workers,
            pin_memory=self.config.val_loader.pin_memory,
            drop_last=self.config.val_loader.drop_last,
            collate_fn=custom_collate_fn,
            persistent_workers=(
                self.config.val_loader.persistent_workers and val_num_workers > 0
            ),
        )
        
        return combined_dataset, val_loader


def load_dataset_config(config_path: str) -> DatasetManagerConfig:
    """Load dataset configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert dictionary to dataclass
    train_datasets = [DatasetConfig(**ds) for ds in config_dict['train_datasets']]
    val_datasets = [DatasetConfig(**ds) for ds in config_dict['val_datasets']]
    train_loader = DataLoaderConfig(**config_dict['train_loader'])
    val_loader = DataLoaderConfig(**config_dict['val_loader'])
    
    return DatasetManagerConfig(
        data_root=config_dict.get('data_root', 'FLAME/data'),
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        train_loader=train_loader,
        val_loader=val_loader
    )


def save_dataset_config(config: DatasetManagerConfig, config_path: str) -> None:
    """Save dataset configuration to JSON file."""
    config_dict = {
        'data_root': config.data_root,
        'train_datasets': [asdict(ds) for ds in config.train_datasets],
        'val_datasets': [asdict(ds) for ds in config.val_datasets],
        'train_loader': asdict(config.train_loader),
        'val_loader': asdict(config.val_loader)
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def create_default_config() -> DatasetManagerConfig:
    """Create a default dataset configuration matching current setup."""
    return DatasetManagerConfig(
        data_root="FLAME/data",
        train_datasets=[
            DatasetConfig(
                name="magicbrush_multiedit_train",
                path="magicbrush_multiedit_train",
                enabled=True,
                allow_multiple_targets=False,
                authentic_ratio=0.0,  # Will be overridden by global setting
                is_training=True
            ),
            DatasetConfig(
                name="sid_train",
                path="large_sid_train",
                enabled=True,
                allow_multiple_targets=False,
                authentic_ratio=0.0,  # SID doesn't use authentic ratio
                is_training=True
            ),
        ],
        val_datasets=[
            DatasetConfig(
                name="magicbrush_val",
                path="full_magicbrush_val",
                enabled=True,
                allow_multiple_targets=False,
                authentic_ratio=0.0,  # Usually validation doesn't need authentic samples
                is_training=False
            ),
            DatasetConfig(
                name="sid_val",
                path="sid_validation",
                enabled=True,
                allow_multiple_targets=False,
                authentic_ratio=0.0,
                is_training=False
            ),
        ],
        train_loader=DataLoaderConfig(
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False
        ),
        val_loader=DataLoaderConfig(
            batch_size=4,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False
        )
    )
