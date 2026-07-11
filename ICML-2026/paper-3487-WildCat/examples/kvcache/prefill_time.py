# Compute prefill time for SnapKV and CompressKV
# Based on https://github.com/NVIDIA/kvpress/blob/main/notebooks/speed_and_memory.ipynb
#
# Example usage: python prefill_time.py

import warnings
from time import time
from tqdm import tqdm
import pickle
import os

import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import torch
from transformers import AutoModelForCausalLM, pipeline
from transformers import DynamicCache, QuantoQuantizedCache
from transformers.utils.logging import disable_progress_bar
import transformers

from kvpress.presses.snapkv_press import SnapKVPress
from presses.compresskv_press import CompressKV

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
disable_progress_bar()

device = "cuda:0"
ckpt = "Qwen/Qwen2.5-7B-Instruct"

# The model is loaded each time inside the function to avoid potential GPU leakage
def get_prefilling_stats(press, n_tokens, cache_implementation="dynamic"):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    idle_peak_memory = torch.cuda.max_memory_allocated()
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="auto").to(device)
    initial_peak_memory = torch.cuda.max_memory_allocated()

    inputs =torch.arange(n_tokens).reshape([1, n_tokens]).to(device)
    # Model warmup (for better prefilling time estimation)
    with torch.no_grad():
        model(inputs[:, :100])
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Compute cache size and prefilling time
    with torch.no_grad(), press(model):
        if cache_implementation == "dynamic":
            cache = DynamicCache()
        elif cache_implementation == "quantized":
            cache = QuantoQuantizedCache(config=model.config, nbits=4)
        else:
            raise NotImplementedError(f"Cache {cache_implementation} not yet implemented")

        start = time()
        model(inputs, num_logits_to_keep=1, past_key_values=cache)
        prefilling_time = time() - start

        del cache
        
    
    peak_memory = torch.cuda.max_memory_allocated()
    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {"Prefilling time": prefilling_time}

def combine_stats(prefilling_stats, generation_stats):
    """Combines prefilling and generation data, then plots."""
    combined_stats = {}
    for compression_ratio in prefilling_stats:
        combined_stats[compression_ratio] = dict()
        combined_stats[compression_ratio]['Prefilling time'] = prefilling_stats[compression_ratio]['Prefilling time']
        
    return combined_stats

# Compute SnapKV prefill time
snapkv_stats = {}
compression_ratios = [0.75]
for n_tokens in [32_000]:
    prefilling_stats = {compression_ratio : get_prefilling_stats(
        press=SnapKVPress(compression_ratio=compression_ratio), n_tokens=n_tokens)
                        for compression_ratio in tqdm(compression_ratios)}
    generation_stats = None
    
    snapkv_stats[n_tokens] = combine_stats(prefilling_stats, generation_stats)
    print(snapkv_stats)

compresskv_stats = {}

# Compute CompressKV prefill time
r = 12
compression_ratios = [0.75] 
for n_tokens in [32_000]: 
    prefilling_stats = {compression_ratio : get_prefilling_stats(
        press=CompressKV(bin_r=r, compression_ratio=compression_ratio), n_tokens=n_tokens)
                        for compression_ratio in tqdm(compression_ratios)}
    generation_stats = None
    compresskv_stats[n_tokens] = combine_stats(prefilling_stats, generation_stats)
    print(compresskv_stats)