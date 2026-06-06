#!/usr/bin/env python3
"""
Evaluation script for Mind-the-Glitch paper.
Computes Pearson and Spearman correlations between VSM metric and oracle on the test dataset.
"""

import os
import sys
import gc

# Set environment variables before any imports
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/home/dataset-assist-0/pip_pkgs/g1748_tmp/hf_cache'
os.environ['HUGGINGFACE_HUB_TOKEN'] = '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'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use first two GPUs seen (mapped to 2,3)

# Add package paths
sys.path.insert(0, '/home/dataset-assist-0/pip_pkgs/g1748')
sys.path.insert(0, '/repo')

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.stats import pearsonr, spearmanr
from omegaconf import OmegaConf
import hydra
from safetensors.torch import load_file

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def load_cleandift_model():
    """Load the CleanDIFT feature extractor model."""
    config_dir = "/repo/external"
    cfg = OmegaConf.load(os.path.join(config_dir, 'sd21_feature_extractor.yaml'))
    cfg_model = cfg['model']

    print("Instantiating CleanDIFT model...")
    cleandift_model = hydra.utils.instantiate(cfg_model)
    cleandift_model = cleandift_model.cuda().bfloat16()

    # Load weights from local HF cache
    ckpt_pth = '/home/dataset-assist-0/pip_pkgs/g1748_tmp/hf_cache/models--CompVis--cleandift/snapshots/bf3a8d841ebdce7e212b61e42877f8fdaed81d58/cleandift_sd21_full.safetensors'

    if not os.path.exists(ckpt_pth):
        print(f"Downloading cleandift weights...")
        from huggingface_hub import hf_hub_download
        ckpt_pth = hf_hub_download(repo_id="CompVis/cleandift", filename="cleandift_sd21_full.safetensors")

    print(f"Loading cleandift weights from {ckpt_pth}")
    state_dict = load_file(ckpt_pth)
    cleandift_model.load_state_dict(state_dict, strict=True)
    cleandift_model = cleandift_model.eval()
    print("CleanDIFT model loaded successfully!")
    return cleandift_model


def load_mtg_model(cleandift_model):
    """Load the MTG model with pre-trained weights."""
    from mtg.models.mtg_model import MindTheGlitchModel

    # Load experiment config
    exp_cfg_path = '/home/dataset-assist-0/pip_pkgs/g1748_tmp/mtg_model/experiment_cfg.yaml'
    config = OmegaConf.load(exp_cfg_path)

    DEVICE = 'cuda'
    DTYPE = torch.bfloat16

    print("Creating MTG model...")
    model = MindTheGlitchModel(cleandift_model, config, DEVICE, DTYPE)

    # Load MTG model weights
    mtg_weights_path = '/home/dataset-assist-0/pip_pkgs/g1748_tmp/mtg_model/mtg_weights.safetensors'
    print(f"Loading MTG weights from {mtg_weights_path}")
    mtg_state_dict = load_file(mtg_weights_path)
    model.load_state_dict(mtg_state_dict, strict=False)
    model = model.eval()
    print("MTG model loaded successfully!")
    return model, config


def load_test_dataset():
    """Load the test dataset from HuggingFace."""
    from datasets import load_dataset

    print("Loading test dataset...")
    dataset = load_dataset(
        'abdo-eldesokey/mtg-dataset',
        split='test',
        token='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',
    )
    print(f"Loaded {len(dataset)} test samples")
    return dataset


def compute_oracle_score(obj_mask, part_mask):
    """Compute oracle score: 1 - (part_mask_area / obj_mask_area)."""
    obj_mask_arr = np.array(obj_mask.convert('L'))
    part_mask_arr = np.array(part_mask.convert('L'))

    obj_area = float((obj_mask_arr > 127).sum())
    part_area = float((part_mask_arr > 127).sum())

    if obj_area == 0:
        return 0.0

    oracle = 1.0 - (part_area / obj_area)
    return oracle


def run_evaluation():
    """Run the full evaluation pipeline."""
    # Load models
    cleandift_model = load_cleandift_model()
    model, config = load_mtg_model(cleandift_model)

    # Load dataset
    dataset = load_test_dataset()

    # Setup transform (768x768 as per experiment config)
    img_size = (768, 768)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    # Evaluation loop
    vsm_scores = []
    oracle_scores = []

    print(f"\nRunning inference on {len(dataset)} samples...")

    DEVICE = 'cuda'
    DTYPE = torch.bfloat16

    for i, sample in enumerate(dataset):
        try:
            # Get images from dataset
            img1_orig = sample['image_1_original'].convert('RGB')
            img2_orig = sample['image_2_original'].convert('RGB')
            img1_inpainted = sample['image_1_inpainted'].convert('RGB')
            img1_obj_mask = sample['image_1_object_mask'].convert('L')
            img2_obj_mask = sample['image_2_object_mask'].convert('L')
            img1_part_mask = sample['image_1_part_mask'].convert('L')
            img2_part_mask = sample['image_2_part_mask'].convert('L')

            # Compute oracle score
            oracle = compute_oracle_score(img1_obj_mask, img1_part_mask)

            # Resize images
            img1_inpainted_resized = img1_inpainted.resize(img_size)
            img2_orig_resized = img2_orig.resize(img_size)
            img1_obj_mask_resized = img1_obj_mask.resize(img_size, Image.NEAREST)
            img2_obj_mask_resized = img2_obj_mask.resize(img_size, Image.NEAREST)
            img1_part_mask_resized = img1_part_mask.resize(img_size, Image.NEAREST)

            # Run inference (original)
            with torch.no_grad():
                result = model.inference(
                    img1_p=img1_inpainted_resized,
                    img2_p=img2_orig_resized,
                    img1_obj_mask_p=img1_obj_mask_resized,
                    img2_obj_mask_p=img2_obj_mask_resized,
                    img1_part_mask_p=img1_part_mask_resized,
                    transform=transforms.ToTensor(),
                )

            metrics = result['metrics']
            vsm_orig = metrics.get('vsm_0.6', [0.0])[0] if isinstance(metrics.get('vsm_0.6', []), list) else metrics.get('vsm_0.6', 0.0)

            # TTA: horizontal flip
            img1_flip = img1_inpainted_resized.transpose(Image.FLIP_LEFT_RIGHT)
            img2_flip = img2_orig_resized.transpose(Image.FLIP_LEFT_RIGHT)
            mask1_obj_flip = img1_obj_mask_resized.transpose(Image.FLIP_LEFT_RIGHT)
            mask2_obj_flip = img2_obj_mask_resized.transpose(Image.FLIP_LEFT_RIGHT)
            mask1_part_flip = img1_part_mask_resized.transpose(Image.FLIP_LEFT_RIGHT)
            with torch.no_grad():
                result_flip = model.inference(
                    img1_p=img1_flip,
                    img2_p=img2_flip,
                    img1_obj_mask_p=mask1_obj_flip,
                    img2_obj_mask_p=mask2_obj_flip,
                    img1_part_mask_p=mask1_part_flip,
                    transform=transforms.ToTensor(),
                )
            metrics_flip = result_flip['metrics']
            vsm_flip = metrics_flip.get('vsm_0.6', [0.0])[0] if isinstance(metrics_flip.get('vsm_0.6', []), list) else metrics_flip.get('vsm_0.6', 0.0)

            vsm_score = (3.0 * vsm_orig + vsm_flip) / 4.0

            vsm_scores.append(vsm_score)
            oracle_scores.append(oracle)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(dataset)}] VSM: {vsm_score:.4f}, Oracle: {oracle:.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            import traceback; traceback.print_exc()
            continue

    # Compute correlations
    vsm_arr = np.array(vsm_scores)
    oracle_arr = np.array(oracle_scores)

    pearson_corr = pearsonr(vsm_arr, oracle_arr)[0]
    spearman_corr = spearmanr(vsm_arr, oracle_arr)[0]

    print(f"\n=== RESULTS ===")
    print(f"Num samples: {len(vsm_scores)}")
    print(f"VSM scores: mean={vsm_arr.mean():.4f}, std={vsm_arr.std():.4f}")
    print(f"Oracle scores: mean={oracle_arr.mean():.4f}, std={oracle_arr.std():.4f}")
    print(f"")
    print(f"Pearson Correlation (VSM vs Oracle): {pearson_corr:.4f}")
    print(f"Spearman Correlation (VSM vs Oracle): {spearman_corr:.4f}")
    print(f"")
    print(f"Paper reported: Pearson=0.448, Spearman=0.582")

    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'num_samples': len(vsm_scores),
        'vsm_mean': vsm_arr.mean(),
        'oracle_mean': oracle_arr.mean(),
    }


if __name__ == '__main__':
    results = run_evaluation()
