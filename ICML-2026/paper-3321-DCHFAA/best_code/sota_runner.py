#!/usr/bin/env python3
"""
SOTA optimization wrapper for step02_fourier.py.
Accepts JSON config to sweep classifier, feature, and threshold parameters
without modifying the original script.
"""

import sys
import json
import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- Import original module ---
sys.path.insert(0, '/repo')
import step02_fourier as sf


def run_with_config(config):
    """
    Run step02_fourier.main() with patched behavior based on config.

    Config keys:
        anno_1, attn_1: required (str)
        anno_2, attn_2: optional (str or null)
        tokenizer_name: str
        sliding_window: int
        penalty: 'l1' or 'l2' (default 'l2')
        C: float (default 1.0)
        class_weight: 'balanced' or None (default None)
        max_iter: int (default 1000)
        threshold_step: float (default 0.1)
        use_low_freq: bool (default False)
        feature_mode: 'mean_pool', 'hfer', 'entropy_weighted'
        random_state: int (default 42)
    """

    # --- Patch load_files to optionally include low-frequency features ---
    original_load_files = sf.load_files

    def patched_load_files(anno_file, attn_file, tokenizer_name=None, auth_token=None):
        use_low = config.get('use_low_freq', False)

        if not use_low:
            return original_load_files(anno_file, attn_file, tokenizer_name, auth_token)

        # Call original to get high-freq features
        result = original_load_files(anno_file, attn_file, tokenizer_name, auth_token)
        attn_tensor, new_tokens_tensors, context_tensors, labels, splits = result

        # Load low-frequency features from the same .pt file
        anno_path = sf.jsonl_path_dict[anno_file]
        attn_path = attn_file

        anno_data = []
        with open(anno_path, 'r') as f:
            for line in f:
                anno_data.append(json.loads(line))

        attn_data = []
        attn_data.extend(torch.load(attn_path, weights_only=False))

        anno_by_index = {a['index']: a for a in anno_data}
        matched_attn = []
        for a in attn_data:
            di = a['data_index']
            if di in anno_by_index:
                matched_attn.append(a)

        # Extend with low-freq features by stacking [high, low] along new dim 0
        extended_new = []
        extended_context = []
        for i, a in enumerate(matched_attn):
            if i < len(context_tensors):
                high_ctx = context_tensors[i]
                low_ctx = a['context_low_l2']
                extended_context.append(torch.stack([high_ctx, low_ctx], dim=0))

                high_new = new_tokens_tensors[i]
                low_new = a['new_tokens_low_l2']
                extended_new.append(torch.stack([high_new, low_new], dim=0))
            else:
                extended_context.append(context_tensors[i])
                extended_new.append(new_tokens_tensors[i])

        return attn_tensor, extended_new, extended_context, labels, splits

    sf.load_files = patched_load_files

    # --- Patch extract_time_series_features ---
    feat_mode = config.get('feature_mode', 'mean_pool')
    if feat_mode != 'mean_pool':
        original_extract = sf.extract_time_series_features

        def patched_extract_features(attn_tensor):
            features = []
            num_examples = len(attn_tensor)

            for i in range(num_examples):
                example = attn_tensor[i].clone()

                if feat_mode == 'hfer':
                    # HFER: sum(|high|^2) / sum(|all|^2) per head
                    if example.dim() == 5 and example.shape[0] == 2:
                        high = example[0]
                        low = example[1]
                        high_energy = (high ** 2).sum(dim=-1)
                        low_energy = (low ** 2).sum(dim=-1)
                        total_energy = high_energy + low_energy + 1e-10
                        hfer = (high_energy / total_energy).flatten().numpy()
                        features.append(hfer)
                    else:
                        try:
                            example_flat = example.view(-1, example.shape[-1])
                            example_flat = example_flat.transpose(0, 1)
                            fv = example_flat.mean(dim=0).numpy()
                            fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
                            features.append(fv)
                        except Exception:
                            continue
                else:
                    # Default: mean pool
                    try:
                        example_flat = example.view(-1, example.shape[-1])
                        example_flat = example_flat.transpose(0, 1)
                        fv = example_flat.mean(dim=0).numpy()
                        fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
                        features.append(fv)
                    except Exception:
                        continue

            return np.array(features)

        sf.extract_time_series_features = patched_extract_features

    # --- Patch split function to handle extended tensors ---
    original_convert = sf.convert_to_token_level

    def patched_convert_to_token_level(attn_tensor, new_tokens_tensors, context_tensors,
                                        labels, splits, sliding_window=1):
        # If tensors have extra dim 0 (high/low), we need to handle sliding window
        # on the last dim while preserving the stacking
        out_attn, out_new, out_ctx, out_labels, out_splits = original_convert(
            attn_tensor, new_tokens_tensors, context_tensors,
            labels, splits, sliding_window=sliding_window)
        return out_attn, out_new, out_ctx, out_labels, out_splits

    sf.convert_to_token_level = patched_convert_to_token_level

    # --- Patch train_test_split random_state ---
    rs = config.get('random_state', 42)

    def patched_train_test_split(*args, **kwargs):
        kwargs['random_state'] = rs
        return train_test_split(*args, **kwargs)

    sf.train_test_split = patched_train_test_split

    # --- Patch LogisticRegression ---
    penalty = config.get('penalty', 'l2')
    C_val = config.get('C', 1.0)
    class_weight = config.get('class_weight', None)
    max_iter = config.get('max_iter', 1000)

    class PatchedLR(LogisticRegression):
        def __init__(self, **kwargs):
            kwargs.setdefault('penalty', penalty)
            kwargs.setdefault('C', C_val)
            kwargs.setdefault('max_iter', max_iter)
            if class_weight is not None:
                kwargs.setdefault('class_weight', class_weight)
            if penalty == 'l1':
                kwargs.setdefault('solver', 'saga')
            super().__init__(**kwargs)

    sf.LogisticRegression = PatchedLR

    # --- Patch threshold search step ---
    threshold_step = config.get('threshold_step', 0.1)
    original_find_best = sf.find_best_threshold_on_validation

    def patched_find_best(y_val, y_val_proba, mode='macro', search_step=None):
        if search_step is None:
            search_step = threshold_step
        return original_find_best(y_val, y_val_proba, mode, search_step)

    sf.find_best_threshold_on_validation = patched_find_best

    # --- Run main ---
    anno_2 = config.get('anno_2', None)
    attn_2 = config.get('attn_2', None)
    if anno_2 in ('', 'null', None):
        anno_2 = None
    if attn_2 in ('', 'null', None):
        attn_2 = None

    sf.main(
        anno_file_1=config['anno_1'],
        attn_file_1=config['attn_1'],
        anno_file_2=anno_2,
        attn_file_2=attn_2,
        tokenizer_name=config.get('tokenizer_name', '/models/Llama-2-7b-chat-hf'),
        sliding_window=config.get('sliding_window', 1),
        classifier_path=None,
    )


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = json.loads(sys.stdin.read())
    run_with_config(config)
