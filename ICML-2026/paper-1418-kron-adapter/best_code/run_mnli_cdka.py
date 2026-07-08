#!/usr/bin/env python3
"""Wrapper to run CDKA MNLI experiment with paper settings."""

import os
import sys

# Set environment
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/autosota_cache/hf'
os.environ['HF_HUB_CACHE'] = '/autosota_cache/hf/hub'
os.environ['HF_DATASETS_CACHE'] = '/autosota_cache/hf/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/autosota_cache/hf'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Monkey-patch huggingface_hub to follow 308 redirects from hf-mirror
import requests
from requests.adapters import HTTPAdapter

# Store original send method
_original_adapter_send = HTTPAdapter.send

def _patched_adapter_send(self, request, **kwargs):
    """Ensure redirects are followed for HEAD requests."""
    # Always allow redirects
    kwargs.setdefault('allow_redirects', True)
    return _original_adapter_send(self, request, **kwargs)

# Apply patch
HTTPAdapter.send = _patched_adapter_send

# Also monkey-patch the Session.head method to ensure allow_redirects
_original_session_head = requests.Session.head
def _patched_session_head(self, url, **kwargs):
    kwargs.setdefault('allow_redirects', True)
    return _original_session_head(self, url, **kwargs)
requests.Session.head = _patched_session_head

print('[*] HF mirror 308 redirect patch applied')
print(f'[*] HF_ENDPOINT={os.environ["HF_ENDPOINT"]}')

# Test model loading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print('[*] Testing t5-base model load...')
tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
print(f'[*] Tokenizer loaded, vocab: {len(tokenizer)}')

# Now run the experiment via hydra
import subprocess
import sys

# Build command for CDKA MNLI T5-Base experiment
cmd = [
    sys.executable, '-u', '/repo/run_exp.py',
    '+model=t5base',
    '+peft=all',
    '++peft.lora_r1=3',
    '++peft.lora_r2=3',
    '++peft.lora_r=1',
    '++peft.lora_alpha=16',
    '+init=default',
    '+dataset_name=mnli',
    '++seed=0',
]

print(f'[*] Running: {" ".join(cmd)}')
os.execvp(sys.executable, cmd)

# Monkey-patch to force local cache for known models
import huggingface_hub.file_download as fd_mod
import huggingface_hub.utils._http as http_mod

_original_request_wrapper = fd_mod._request_wrapper

def _patched_request_wrapper(**kwargs):
    kwargs['allow_redirects'] = True
    return _original_request_wrapper(**kwargs)

fd_mod._request_wrapper = _patched_request_wrapper

# Pre-cache the model loading functions
import functools
import transformers

_original_model_from_pretrained = transformers.PreTrainedModel.from_pretrained.__func__

@classmethod
@functools.wraps(_original_model_from_pretrained)
def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    # If model name is google-t5/t5-base, redirect to local cache
    if isinstance(pretrained_model_name_or_path, str) and 'google-t5/t5-base' in pretrained_model_name_or_path:
        local_path = '/autosota_cache/hf/hub/models--google-t5--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'
        if os.path.isdir(local_path):
            pretrained_model_name_or_path = local_path
    return _original_model_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

transformers.PreTrainedModel.from_pretrained = _patched_from_pretrained
