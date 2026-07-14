# src/config.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any

import wandb
from omegaconf import OmegaConf


class ConfigManager:
    def __init__(self, config_path: str = None, overrides: Dict[str, Any] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to YAML config file
            overrides: Dictionary of parameters to override from config
        """
        self.config_path = config_path or "configs/base_config.yaml"
        self.overrides = overrides or {}

        # Load config
        self.config = self._load_config()

        # Initialize wandb
        self._init_wandb()

    def _load_config(self) -> Dict[str, Any]:
        """Load and process configuration."""
        # Load base config
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Convert to OmegaConf for easier manipulation
        config = OmegaConf.create(config)

        # Apply overrides
        for k, v in self.overrides.items():
            OmegaConf.update(config, k, v)

        return config

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        wandb_config = self.config.get("wandb", {})
        if wandb_config.get("mode") != "disabled":
            wandb.init(
                project=wandb_config.get("project"),
                entity=wandb_config.get("entity"),
                config=OmegaConf.to_container(
                    self.config, resolve=True
                ),  # Changed from self.config.defaults
                mode=wandb_config.get("mode", "online"),
            )

    def get_config(self) -> Dict[str, Any]:
        """Get processed configuration."""
        return self.config  # Changed from self.config.defaults
