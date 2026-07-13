#!/usr/bin/env python3
"""Quick test to verify fm_post_transform rescaling parameter."""

import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="config")
def test_config(cfg: DictConfig):
    """Test what rescale parameter fm_post_transform receives."""

    print("="*80)
    print("FM POST TRANSFORM CONFIGURATION TEST")
    print("="*80)

    # Print task-level rescale
    print(f"\nTask rescale setting: {cfg.task.rescale}")

    # Print fm_post_transform rescale setting
    if 'fm_post_transform' in cfg.task:
        fm_config = cfg.task.fm_post_transform
        print(f"\nFM Post Transform config:")
        print(OmegaConf.to_yaml(fm_config))

        if 'training_params' in fm_config:
            training_params = fm_config.training_params
            print(f"\nTraining params:")
            print(OmegaConf.to_yaml(training_params))

            if 'rescale' in training_params:
                print(f"\n✅ Rescale parameter FOUND in training_params: {training_params.rescale}")
            else:
                print(f"\n❌ Rescale parameter NOT FOUND in training_params")
                print("   This means it will use the function default!")

    # Also check rope for comparison
    print("\n" + "="*80)
    print("ROPE CONFIGURATION (for comparison)")
    print("="*80)
    if 'rope' in cfg.task:
        rope_config = cfg.task.rope
        print(f"\nRoPE config:")
        print(OmegaConf.to_yaml(rope_config))

        if 'training_params' in rope_config:
            training_params = rope_config.training_params
            print(f"\nTraining params:")
            print(OmegaConf.to_yaml(training_params))

            if 'rescale' in training_params:
                print(f"\n✅ Rescale parameter FOUND in training_params: {training_params.rescale}")

if __name__ == "__main__":
    test_config()
