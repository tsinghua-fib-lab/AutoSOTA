#!/usr/bin/env python3
"""
Single Experiment Runner - Combined Version

Runs a single watermark generation experiment from command-line config.
Combines run_experiment.py and watermark_generate.py functionality.
Designed to be called by experiment_scheduler.py for parallel execution.

Usage:
    python run_experiment_combined.py --config '{"model_key": "gpt2", "dataset_key": "wikipedia", ...}'
    python run_experiment_combined.py --config-file config.json
"""

import argparse
import json
import sys
import os
import math
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch

# Set default cache directory (will be overridden by config if specified)
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = './hf_cache'
if 'HF_DATASETS_CACHE' not in os.environ:
    os.environ['HF_DATASETS_CACHE'] = './hf_cache/datasets'

# Uncomment to force offline mode (requires pre-cached models via download_cache.py)
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList,LogitsProcessor
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import Tokenizer
sys.path.append("./watermark_reliability_release")
from watermark_processor import WatermarkBase, WatermarkDetector, WatermarkLogitsProcessor
from experiment_config import create_config, validate_config
from transformers import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
)
from itertools import tee, chain
from collections import Counter
import torch.nn.functional as F

##########################################################################
# Ngram iteration from nltk, extracted to remove the dependency
# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Eric Kafe <kafe.eric@gmail.com> (acyclic closures)
# URL: <https://www.nltk.org/>
# For license information, see https://github.com/nltk/nltk/blob/develop/LICENSE.txt
##########################################################################


def ngrams(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n - 1))
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.




class WatermarkLogitsProcessorWithKL(WatermarkLogitsProcessor):
    def __init__(
        self,
        *args,
        store_spike_ents: bool = False,
        store_kl_divergence: bool = False,
        entropy_threshold: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, store_spike_ents=store_spike_ents, **kwargs)

        self.store_kl_divergence = store_kl_divergence
        self.kl_divergences = None
        self.entropy_threshold = entropy_threshold  # fraction of max_entropy, 0.0 = disabled

    def _compute_kl_divergence(self, pre_scores, post_scores):
        kl_div = F.kl_div(
            F.log_softmax(pre_scores, dim=-1),
            F.log_softmax(post_scores, dim=-1),
            reduction='sum',
            log_target=True
        )
        return kl_div

    def _get_kl_divergences(self):
        kl_divs = [[] for _ in range(len(self.kl_divergences))]
        for b_idx, kl_tensor_list in enumerate(self.kl_divergences):
            for kl_tensor in kl_tensor_list:
                kl_divs[b_idx].append(kl_tensor.item())
        return kl_divs

    def _get_and_clear_stored_kl_divergences(self):
        kl_divs = self._get_kl_divergences()
        self.kl_divergences = None
        return kl_divs

    def __call__(self, input_ids, scores):
        if self.store_kl_divergence:
            pre_scores = scores.clone()

        # Entropy-gated watermark: skip bias for low-entropy tokens
        if self.entropy_threshold > 0.0 and input_ids.shape[-1] >= self.context_width:
            # Lazy initialization of RNG (same as parent __call__)
            self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng

            probs = torch.softmax(scores, dim=-1)
            log_probs = torch.log_softmax(scores, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            max_entropy = math.log(scores.shape[-1])
            entropy_ratio = entropy / max_entropy
            apply_watermark = entropy_ratio >= self.entropy_threshold

            scores_processed = scores.clone()
            for b_idx, input_seq in enumerate(input_ids):
                if self.self_salt:
                    greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
                else:
                    greenlist_ids = self._get_greenlist_ids(input_seq)
                if apply_watermark[b_idx]:
                    scores_processed[b_idx, greenlist_ids] = \
                        scores_processed[b_idx, greenlist_ids] + self.delta
            scores = scores_processed
        else:
            scores = super().__call__(input_ids, scores)

        if self.store_kl_divergence:
            if self.kl_divergences is None:
                self.kl_divergences = [[] for _ in range(input_ids.shape[0])]
            for b_idx in range(input_ids.shape[0]):
                kl = self._compute_kl_divergence(pre_scores[b_idx], scores[b_idx])
                self.kl_divergences[b_idx].append(kl)

        return scores


class WatermarkLogitsProcessorOPT(WatermarkLogitsProcessorWithKL):
    """
    OPT (Optimal) Watermark Logits Processor.

    Threshold-based watermarking (Wouters 2024): at each step, compute distortion
    cost B. If B <= delta (beta threshold), force all probability to green tokens;
    otherwise keep original distribution.
    """

    def __init__(self, *args, eps: float = 1e-12,
                 store_b_values: bool = False, store_gamma_values: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.store_b_values = store_b_values
        self.store_gamma_values = store_gamma_values
        self.b_values = None
        self.gamma_values = None
        self.watermark_applied = None

    def _compute_b_value(self, p, green_mask):
        """Compute distortion cost B(p_t, G_t) and green probability mass Gamma_t."""
        p_safe = torch.clamp(p, min=self.eps)
        log_p = torch.log(p_safe)
        Gamma = p[green_mask].sum()
        Gamma_safe = torch.clamp(Gamma, min=self.eps, max=1.0 - self.eps)
        p_log_p = p_safe * log_p
        S = p_log_p.sum()
        S_G = p_log_p[green_mask].sum()
        numerator = Gamma_safe * S - S_G
        denominator = Gamma_safe * (1.0 - Gamma_safe)
        B = numerator / denominator
        return B, Gamma

    def _compute_kl_divergence(self, pre_scores, post_scores):
        """KL divergence handling -inf logits from OPT red token zeroing."""
        p = torch.softmax(pre_scores, dim=-1)
        q = torch.softmax(post_scores, dim=-1)
        p_safe = torch.clamp(p, min=self.eps)
        q_safe = torch.clamp(q, min=self.eps)
        log_ratio = torch.log(q_safe) - torch.log(p_safe)
        kl_terms = q * log_ratio
        kl_terms = torch.where(q > self.eps, kl_terms, torch.zeros_like(kl_terms))
        return kl_terms.sum()

    def _apply_opt_watermark(self, scores, green_mask, Gamma):
        """Set red token logits to -inf (force green sampling)."""
        scores[~green_mask] = float('-inf')
        return scores

    def _get_b_values(self):
        if self.b_values is None:
            return None
        return [[v.item() if torch.is_tensor(v) else v for v in vl] for vl in self.b_values]

    def _get_gamma_values(self):
        if self.gamma_values is None:
            return None
        return [[v.item() if torch.is_tensor(v) else v for v in vl] for vl in self.gamma_values]

    def _get_watermark_applied(self):
        if self.watermark_applied is None:
            return None
        return [list(flags) for flags in self.watermark_applied]

    def _get_and_clear_stored_values(self):
        """Get all stored values (KL, B, Gamma, applied flags) and clear."""
        result = {
            'b_values': self._get_b_values(),
            'gamma_values': self._get_gamma_values(),
            'watermark_applied': self._get_watermark_applied(),
            'kl_divergence': self._get_kl_divergences() if self.store_kl_divergence else None
        }
        self.b_values = None
        self.gamma_values = None
        self.watermark_applied = None
        self.kl_divergences = None
        return result

    def __call__(self, input_ids, scores):
        if self.store_kl_divergence:
            pre_scores = scores.clone()

        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        if self.store_b_values and self.b_values is None:
            self.b_values = [[] for _ in range(input_ids.shape[0])]
        if self.store_gamma_values and self.gamma_values is None:
            self.gamma_values = [[] for _ in range(input_ids.shape[0])]
        if self.watermark_applied is None:
            self.watermark_applied = [[] for _ in range(input_ids.shape[0])]

        for b_idx, input_seq in enumerate(input_ids):
            greenlist_ids = self._get_greenlist_ids(input_seq)
            green_mask = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
            green_mask[greenlist_ids] = True

            p = torch.softmax(scores[b_idx], dim=-1)
            B, Gamma = self._compute_b_value(p, green_mask)

            if self.store_b_values:
                self.b_values[b_idx].append(B.detach())
            if self.store_gamma_values:
                self.gamma_values[b_idx].append(Gamma.detach())

            if B.item() <= self.delta:
                scores[b_idx] = self._apply_opt_watermark(scores[b_idx], green_mask, Gamma)
                self.watermark_applied[b_idx].append(True)
            else:
                self.watermark_applied[b_idx].append(False)

        if self.store_kl_divergence:
            if self.kl_divergences is None:
                self.kl_divergences = [[] for _ in range(input_ids.shape[0])]
            for b_idx in range(input_ids.shape[0]):
                kl = self._compute_kl_divergence(pre_scores[b_idx], scores[b_idx])
                self.kl_divergences[b_idx].append(kl)

        return scores

class WatermarkDetectorNgram(WatermarkDetector):
    def _get_green_at_T_booleans(
        self, input_ids, ngram_to_watermark_lookup
    ):
        """Now keep track of which token is cosidered ngram"""

        green_token_mask, green_token_mask_unique, offsets = [], [], []
        used_ngrams = {}
        unique_ngram_idx = 0
        ngram_examples = ngrams(input_ids.cpu().tolist(),self.context_width + 1 - self.self_salt)

        for ngram_example in ngram_examples:
            green_token_mask.append(ngram_to_watermark_lookup[ngram_example])
            if self.ignore_repeated_ngrams:
                if ngram_example in used_ngrams:
                    pass
                else:
                    used_ngrams[ngram_example] = True
                    unique_ngram_idx += 1
                    green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
            else:
                green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                unique_ngram_idx += 1
            offsets.append(unique_ngram_idx - 1)
        used = set()
        ngram_mask = []
        for off in offsets:
            if off not in used:
                ngram_mask.append(True)
                used.add(off)
            else:
                ngram_mask.append(False)
        return (
            torch.tensor(green_token_mask, dtype=torch.bool),
            torch.tensor(green_token_mask_unique, dtype=torch.bool),
            torch.tensor(offsets, dtype=torch.long),
            torch.tensor(ngram_mask, dtype=torch.bool),
        )
    

    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = True,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        return_ngram_mask: bool = True,
    ):
        ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(input_ids)
        ids = input_ids
        if ids.dim() == 2:
            ids = ids[0]

        green_token_mask, green_unique, offsets, ngram_mask = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )

        if self.ignore_repeated_ngrams:
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())
        else:
            num_tokens_scored = sum(frequencies_table.values())
            assert num_tokens_scored == len(input_ids) - self.context_width + self.self_salt
            green_token_count = sum(
                freq * outcome
                for freq, outcome in zip(frequencies_table.values(), ngram_to_watermark_lookup.values())
            )

        assert green_token_count == green_unique.sum()

        score_dict = {}
        if return_num_tokens_scored:
            score_dict["num_tokens_scored"] = num_tokens_scored
        if return_num_green_tokens:
            score_dict["num_green_tokens"] = int(green_token_count)
        if return_green_fraction:
            score_dict["green_fraction"] = float(green_token_count / num_tokens_scored)
        if return_z_score:
            score_dict["z_score"] = float(self._compute_z_score(green_token_count, num_tokens_scored))
        if return_p_value:
            z = score_dict.get("z_score")
            if z is None:
                z = float(self._compute_z_score(green_token_count, num_tokens_scored))
            score_dict["p_value"] = float(self._compute_p_value(z))

        if return_green_token_mask:
            score_dict["green_token_mask"] = green_token_mask.tolist()

        if return_ngram_mask:
            score_dict['ngram_included_mask'] = ngram_mask.tolist()

        if return_z_at_T:
            sizes = torch.arange(1, len(green_unique) + 1, device=green_unique.device)
            seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.gamma * sizes
            seq_z_score_denom = torch.sqrt(sizes * self.gamma * (1 - self.gamma))
            z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
            z_score_at_T = z_score_at_effective_T[offsets]
            score_dict["z_score_at_T"] = z_score_at_T

        return score_dict

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer(config):
    """Load model and tokenizer with automatic cache fallback."""
    print(f"[PROGRESS] Loading model: {config['model_name']}")

    # Get cache directory from config or environment variable
    cache_dir = config.get('cache_dir', None)
    if cache_dir is None:
        cache_dir = os.getenv('HF_HOME', None)

    # Check if forced offline mode
    is_offline_forced = os.getenv('TRANSFORMERS_OFFLINE', '0') == '1'
    model_name = config['model_name']

    # Helper function for two-phase loading (cache first, then download fallback)
    def load_with_fallback(component_name, load_fn):
        """Try loading from cache first, fallback to download if allowed."""
        try:
            # Phase 1: Try loading from cache only
            print(f"[PROGRESS] Attempting to load {component_name} from cache")
            result = load_fn(local_files_only=True)
            print(f"[PROGRESS] Loaded {component_name} from cache")
            return result
        except (OSError, ValueError, EnvironmentError) as cache_error:
            # Cache miss or corrupted
            if is_offline_forced:
                # Cannot fallback in forced offline mode
                raise RuntimeError(
                    f"Failed to load {component_name} from cache in offline mode.\n"
                    f"Model: {model_name}\n"
                    f"Cache directory: {cache_dir}\n"
                    f"Error: {str(cache_error)}\n\n"
                    f"Please run: python download_cache.py --models {config.get('model_key', 'all')}"
                ) from cache_error
            else:
                # Phase 2: Fallback to download
                print(f"[PROGRESS] Cache miss for {component_name}, downloading from Hugging Face Hub...")
                result = load_fn(local_files_only=False)
                print(f"[PROGRESS] {component_name} downloaded and cached successfully")
                return result

    # Load tokenizer with cache fallback
    tokenizer = load_with_fallback(
        "tokenizer",
        lambda local_files_only: AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
    )

    # Use bf16 for large models (>2GB VRAM) to avoid OOM
    vram_gb = config.get('model_vram_gb', 0)
    use_bf16 = vram_gb > 2
    model_kwargs = dict(cache_dir=cache_dir)
    if use_bf16:
        model_kwargs['dtype'] = torch.bfloat16
        print(f"[PROGRESS] Using bfloat16 for {model_name} (vram_gb={vram_gb})")

    # Load model with cache fallback
    model = load_with_fallback(
        "model",
        lambda local_files_only: AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            **model_kwargs
        )
    )

    # Handle models without a pad token (e.g. Llama 3)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        print(f"[PROGRESS] Set pad_token to eos_token for {model_name}")

    device = 'cuda'  # if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    print(f"[PROGRESS] Model loaded on {device}")
    if cache_dir:
        print(f"[PROGRESS] Using cache directory: {cache_dir}")
    return model, tokenizer, device


def setup_watermark(config, tokenizer, device):
    """Setup watermark processor and detector based on watermark_type in config."""
    watermark_type = config.get('watermark_type', 'soft')

    if watermark_type == 'opt':
        print("[PROGRESS] Setting up OPT watermark processor and detector")
        watermark_processor = WatermarkLogitsProcessorOPT(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=config['gamma'],
            delta=config['delta'],
            seeding_scheme=config['seeding_scheme'],
            select_green_tokens=True,
            store_kl_divergence=True,
            store_b_values=True,
            store_gamma_values=True,
        )
    else:
        entropy_threshold = config.get('entropy_threshold', 0.0)
        print(f"[PROGRESS] Setting up Soft (KGW) watermark processor and detector" +
              (f" (entropy_threshold={entropy_threshold})" if entropy_threshold > 0 else ""))
        watermark_processor = WatermarkLogitsProcessorWithKL(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=config['gamma'],
            delta=config['delta'],
            seeding_scheme=config['seeding_scheme'],
            select_green_tokens=True,
            store_kl_divergence=True,
            entropy_threshold=entropy_threshold,
        )

    watermark_detector = WatermarkDetectorNgram(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=config['gamma'],
        delta=config['delta'],
        seeding_scheme=config['seeding_scheme'],
        device=device,
        tokenizer=tokenizer,
        z_threshold=config['z_threshold'],
        normalizers=[],
        ignore_repeated_ngrams=True,
        select_green_tokens=True
    )

    return watermark_processor, watermark_detector


def load_dataset_stream(config):
    """Load dataset with automatic cache usage."""
    print(f"[PROGRESS] Loading dataset: {config['dataset_name']}")

    # Get cache directory from config or environment variable
    cache_dir = config.get('cache_dir', None)
    
    if cache_dir is None:
        cache_dir = os.getenv('HF_HOME', None)
    # Set datasets cache to be in a subdirectory of the main cache
    # This ensures datasets are stored alongside models in ./hf_cache/datasets/
    original_datasets_cache = os.environ.get('HF_DATASETS_CACHE')
    if cache_dir:
        datasets_cache = str(Path(cache_dir))+'/datasets'
        os.environ['HF_DATASETS_CACHE'] = datasets_cache

    # Check if forced offline mode
    is_offline_forced = os.getenv('HF_DATASETS_OFFLINE', '0') == '1'

    try:
        print(f"[PROGRESS] Attempting to load dataset from cache")
        if cache_dir:
            print(f"[PROGRESS] Datasets cache directory: {datasets_cache}")

        # Import DownloadMode for controlling dataset loading behavior
        from datasets import DownloadMode

        # Try loading with cache-first approach (avoids unnecessary network calls)
        try:
            if config['dataset_config'] is not None:
                dataset = load_dataset(
                    config['dataset_name'],
                    config['dataset_config'],
                    split=config['split'],
                    streaming=config['streaming'],
                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
                )
            else:
                dataset = load_dataset(
                    config['dataset_name'],
                    split=config['split'],
                    streaming=config['streaming'],
                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
                )
            print(f"[PROGRESS] Dataset loaded from cache (offline mode)")
        except Exception as cache_err:
            # If cache loading fails, try with normal download mode
            print(f"[PROGRESS] Cache-only load failed, attempting with network access...")
            if config['dataset_config'] is not None:
                dataset = load_dataset(
                    config['dataset_name'],
                    config['dataset_config'],
                    split=config['split'],
                    streaming=config['streaming']
                )
            else:
                dataset = load_dataset(
                    config['dataset_name'],
                    split=config['split'],
                    streaming=config['streaming']
                )
            print(f"[PROGRESS] Dataset loaded successfully")

    except Exception as error:
        # If loading failed in offline mode, provide helpful error
        if is_offline_forced:
            raise RuntimeError(
                f"Failed to load dataset in offline mode.\n"
                f"Dataset: {config['dataset_name']} (config: {config.get('dataset_config')})\n"
                f"Cache directory: {cache_dir}\n"
                f"Error: {str(error)}\n\n"
                f"Please run: python download_cache.py --datasets {config.get('dataset_key', 'all')}"
            ) from error
        else:
            # Re-raise error for debugging (download will be attempted automatically)
            print(f"[ERROR] Failed to load dataset: {error}")
            raise
    finally:
        # Restore original environment variable
        if original_datasets_cache is not None:
            os.environ['HF_DATASETS_CACHE'] = original_datasets_cache
        elif 'HF_DATASETS_CACHE' in os.environ:
            del os.environ['HF_DATASETS_CACHE']

    dataset = dataset.shuffle(buffer_size=1000, seed=config['seed'])
    print(f"[PROGRESS] Dataset loaded, shuffling enabled seed:{config['seed']}")
    if cache_dir:
        print(f"[PROGRESS] Using cache directory: {cache_dir}")
    return dataset



def process_sentence(text, truncate_at, model, tokenizer, watermark_processor,
                    watermark_detector, config, device):
    """Process a single sentence through the watermark pipeline."""
    # Tokenize the input text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Skip if text is too short
    if len(tokens) < truncate_at:
        return None

    # Truncate at specified position
    truncated_tokens = tokens[:truncate_at]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    # Prepare input for generation
    inputs = tokenizer(truncated_text, return_tensors="pt").to(device)


    def generate_and_detect(apply_watermark):
        """Helper function to generate and detect for one variant."""
        # Generate with or without watermark
        if apply_watermark:
            logits_processor = LogitsProcessorList([
                watermark_processor,
            ])
        else:
            logits_processor = None
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config['max_new_tokens'],
                do_sample=config['do_sample'],
                temperature=config['temperature'],
                top_k=config.get('top_k', 0),
                top_p=config.get('top_p', 1.0),
                logits_processor=logits_processor
            )
        # Extract generated tokens (excluding prompt)
        generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        full_output = truncated_text + generated_text

        # Retrieve analysis values based on processor type
        kl_divergence_values = None
        b_values = None
        gamma_values = None
        watermark_applied = None

        if apply_watermark:
            if isinstance(watermark_processor, WatermarkLogitsProcessorOPT):
                opt_analysis = watermark_processor._get_and_clear_stored_values()
                if opt_analysis:
                    kl_divergence_values = opt_analysis['kl_divergence'][0] if opt_analysis['kl_divergence'] else None
                    b_values = opt_analysis['b_values'][0] if opt_analysis['b_values'] else None
                    gamma_values = opt_analysis['gamma_values'][0] if opt_analysis['gamma_values'] else None
                    watermark_applied = opt_analysis['watermark_applied'][0] if opt_analysis['watermark_applied'] else None
            elif watermark_processor.store_kl_divergence:
                kl_divergence_values = watermark_processor._get_and_clear_stored_kl_divergences()
                if kl_divergence_values and len(kl_divergence_values) > 0:
                    kl_divergence_values = kl_divergence_values[0]

        detection_result = watermark_detector.detect(
            tokenized_text=generated_tokens.to(watermark_detector.device),
            return_prediction=True,
            return_scores=True,
            return_green_token_mask=True
            )
        ngraminfo=detection_result.get('ngram_included_mask', [])
        green_token_mask=detection_result.get('green_token_mask', [])
        T = len(generated_tokens)
        n = watermark_detector.context_width + 1 - watermark_detector.self_salt
        if kl_divergence_values is not None:
            assert len(kl_divergence_values) == T, f"KL divergence length mismatch: {len(kl_divergence_values)} vs {T}"
        assert len(ngraminfo) == T - n + 1, f"Ngram info length mismatch: {len(ngraminfo)} vs {T - n + 1}"

        # Summary stats from detection (always computed)
        summary = {
            'z_score': float(detection_result.get('z_score', 0.0)),
            'prediction': bool(detection_result.get('prediction', False)),
            'p_value': float(detection_result.get('p_value', 1.0)),
            'green_fraction': float(detection_result.get('green_fraction', 0.0)),
            'num_tokens_scored': int(detection_result.get('num_tokens_scored', 0)),
            'num_tokens_generated': T,
            'mean_kl': float(np.mean(kl_divergence_values)) if kl_divergence_values else None,
        }

        return {
            'generated_text': generated_text,
            'full_output': full_output,
            'generated_tokens': generated_tokens.cpu().tolist(),
            'green_token_mask': green_token_mask,
            'ngram_info': ngraminfo,
            'kl_divergence': kl_divergence_values,
            'b_values': b_values,
            'gamma_values': gamma_values,
            'watermark_applied': watermark_applied,
            'summary': summary,
        }
    result = {
        'original_text': text,
        'truncated_text': truncated_text,
        'watermarked': generate_and_detect(apply_watermark=True),
    }

    # Filter by minimum generated length
    min_length = config.get('min_generated_length', 200)
    if len(result['watermarked']['generated_text']) < min_length:
        return None
    return result


def run_generation(config):

    """Run the text generation (no statistics calculation)."""
    print("="*80)
    print(f"STARTING GENERATION: {config['experiment_name']}")
    print("="*80)
    print(f"Model: {config['model_name']}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Samples: {config['num_samples']}")
    print(f"Gamma: {config['gamma']}, Delta: {config['delta']}")
    print(f"Temperature: {config['temperature']}, Do Sample: {config['do_sample']}")
    print("="*80)

    # Load model and setup
    model, tokenizer, device = load_model_and_tokenizer(config)
    watermark_processor, watermark_detector = setup_watermark(config, tokenizer, device)
    dataset = load_dataset_stream(config)

    # Process sentences
    results = []
    successful = 0

    print(f"[PROGRESS] Processing {config['num_samples']} sentences...")

    progress_bar = tqdm(total=config['num_samples'], desc="Processing")

    for idx, sample in enumerate(dataset):
        if successful >= config['num_samples']:
            break

        # Extract text from sample
        text_field_value = sample[config['text_field']]

        # Special handling for lfqa dataset where 'answers' is a dict
        if isinstance(text_field_value, dict) and 'text' in text_field_value:
            # Extract first answer text from the answers dict
            text = text_field_value['text'][0].strip() if text_field_value['text'] else ""
        else:
            text = text_field_value.strip()

        # Split into sentences (simple split on periods)
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 0]

        # Process first valid sentence
        for sentence in sentences:
            try:
                result = process_sentence(
                    sentence,
                    truncate_at=config['truncate_at'],
                    model=model,
                    tokenizer=tokenizer,
                    watermark_processor=watermark_processor,
                    watermark_detector=watermark_detector,
                    config=config,
                    device=device
                )

                if result is not None:
                    results.append(result)
                    successful += 1
                    progress_bar.update(1)
                    break
            except Exception as e:
                print(f"[ERROR] Failed to process sentence: {e}")
                import traceback
                traceback.print_exc()
                continue

    progress_bar.close()

    print(f"[PROGRESS] Generated {len(results)} sentences successfully")

    save_mode = config.get('save_mode', 'stats')  # 'stats' (B), 'compact' (A), 'full' (original)

    # Compute experiment-level summary stats from per-sample summaries
    z_scores = [r['watermarked']['summary']['z_score'] for r in results]
    predictions = [r['watermarked']['summary']['prediction'] for r in results]
    green_fractions = [r['watermarked']['summary']['green_fraction'] for r in results]
    kl_means = [r['watermarked']['summary']['mean_kl'] for r in results
                if r['watermarked']['summary']['mean_kl'] is not None]

    experiment_summary = {
        'num_samples': len(results),
        'mean_z_score': float(np.mean(z_scores)) if z_scores else None,
        'std_z_score': float(np.std(z_scores)) if z_scores else None,
        'median_z_score': float(np.median(z_scores)) if z_scores else None,
        'mean_green_fraction': float(np.mean(green_fractions)) if green_fractions else None,
        'mean_kl': float(np.mean(kl_means)) if kl_means else None,
        'tpr_z4': float(np.mean(predictions)) if predictions else None,  # TPR at default threshold
    }
    # TPR at multiple thresholds
    for thresh in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        experiment_summary[f'tpr_z{thresh:.0f}'] = float(np.mean([z > thresh for z in z_scores])) if z_scores else None

    print(f"[SUMMARY] z_score: {experiment_summary['mean_z_score']:.2f} +/- {experiment_summary['std_z_score']:.2f}")
    print(f"[SUMMARY] TPR@4: {experiment_summary['tpr_z4']:.3f}, green_frac: {experiment_summary['mean_green_fraction']:.3f}")
    if experiment_summary['mean_kl'] is not None:
        print(f"[SUMMARY] mean_KL: {experiment_summary['mean_kl']:.4f}")

    # Prepare output based on save_mode
    if save_mode == 'full':
        # Original: save everything (per-token arrays, text, etc.)
        sample_results = results
    elif save_mode == 'compact':
        # Compact keeps the per-token data needed by attacks, quality metrics,
        # green-token-rate summaries, and Pareto aggregation while dropping OPT
        # internals that are only needed for low-level debugging.
        sample_results = []
        for r in results:
            wm = r['watermarked']
            sample_results.append({
                'original_text': r['original_text'],
                'truncated_text': r['truncated_text'],
                'watermarked': {
                    'generated_tokens': wm['generated_tokens'],
                    'generated_text': wm['generated_text'],
                    'green_token_mask': wm['green_token_mask'],
                    'ngram_info': wm['ngram_info'],
                    'kl_divergence': wm['kl_divergence'],
                    'summary': wm['summary'],
                },
            })
    else:
        # Stats only (default): just per-sample summary, no tokens/text
        sample_results = []
        for r in results:
            sample_results.append({
                'watermarked': {
                    'summary': r['watermarked']['summary'],
                },
            })

    output = {
        'experiment_name': config['experiment_name'],
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'num_samples': len(results),
        'experiment_summary': experiment_summary,
        'results': sample_results,
    }

    return output


def save_results(output, output_dir):
    """Save generation results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{output['experiment_name']}.json"
    filepath = output_path / filename

    print(f"[PROGRESS] Saving results to {filepath}")

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[PROGRESS] Results saved successfully")
    return str(filepath)


def run_single_experiment(config, verbose=True):
    """
    Run a single watermark generation experiment.

    Args:
        config: Experiment configuration dictionary
        verbose: Print progress messages

    Returns:
        tuple: (success: bool, output_path: str or None)
    """
    try:
        # Validate config
        validate_config(config)

        # Ensure output directory exists
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config to temporary file
        config_file = output_dir / f"{config['experiment_name']}_config.json"

        # Check if experiment already exists
        if f"{config['experiment_name']}.json" in os.listdir(output_dir):
            if verbose:
                print(f"[WORKER] Experiment already exists: {config['experiment_name']}")
            return True, str(output_dir / f"{config['experiment_name']}.json")

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        if verbose:
            print(f"[WORKER] Running: {config['experiment_name']}")
            print(f"[WORKER] Model: {config['model_key']}, Dataset: {config['dataset_key']}")
            print(f"[WORKER] Gamma: {config['gamma']}, Delta: {config['delta']}")

        # Set seed for reproducibility
        set_seed(config['seed'])

        # Run generation directly (no subprocess)
        output = run_generation(config)

        # Save results
        output_path = save_results(output, config['output_dir'])

        if verbose:
            print(f"[WORKER] Success: {output_path}")

        return True, str(output_path)

    except Exception as e:
        if verbose:
            print(f"[WORKER] Exception: {e}")
        return False, None


def main():
    parser = argparse.ArgumentParser(description='Run single watermark experiment')

    # Config input methods
    config_group = parser.add_mutually_exclusive_group(required=False)
    config_group.add_argument('--config', type=str,
                            help='JSON string with experiment config')
    config_group.add_argument('--config-file', type=str,#default='results\GRID_m_opt-125m_d_wikipedia_g0.10_d0.00000_t0.50_s0_config.json',
                            help='Path to JSON config file')

    # Individual parameters (alternative to config)
    parser.add_argument('--model', type=str, default='gpt2', help='Model key (e.g., gpt2)')
    parser.add_argument('--dataset', type=str, default='lfqa', help='Dataset key (e.g., wikipedia)')
    parser.add_argument('--gamma', type=float, default=0.25, help='Gamma parameter')
    parser.add_argument('--delta', type=float, default=6, help='Delta parameter')
    parser.add_argument('--num-samples', type=int, default=200, help='Number of samples')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    parser.add_argument('--watermark-type', type=str, default='soft',
                       choices=['soft', 'opt'],
                       help='Watermark type: soft (KGW green-list bias) or opt (OPT threshold)')
    parser.add_argument('--verbose', action='store_true', default=True,
                      help='Print progress messages')
    parser.add_argument('--quiet', action='store_true',
                      help='Suppress output')
    parser.add_argument('--save-mode', type=str, default='stats',
                      choices=['stats', 'compact', 'full'],
                      help='Output mode: stats (default, summary only, smallest), '
                           'compact (tokens + plot/attack metrics + summary), '
                           'full (everything including OPT internals)')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = json.loads(args.config)
    elif args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        # Build config from individual arguments (using defaults if not provided)
        config = create_config(
            model_key=args.model,
            dataset_key=args.dataset,
            num_samples=args.num_samples,
            gamma=args.gamma,
            delta=args.delta,
            experiment_name=args.experiment_name
        )
        config['watermark_type'] = args.watermark_type

    # Set save_mode in config (CLI overrides config file)
    if 'save_mode' not in config:
        config['save_mode'] = args.save_mode

    # Run experiment
    verbose = args.verbose and not args.quiet
    success, output_path = run_single_experiment(config, verbose=verbose)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
