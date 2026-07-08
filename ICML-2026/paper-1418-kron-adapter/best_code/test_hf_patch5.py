import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/autosota_cache/hf'

# Patch _request_wrapper in file_download module
import huggingface_hub.file_download as fd

_original_request_wrapper = fd._request_wrapper

def _patched_request_wrapper(**kwargs):
    # Force allow_redirects=True for HEAD requests to hf-mirror
    kwargs['allow_redirects'] = True
    return _original_request_wrapper(**kwargs)

fd._request_wrapper = _patched_request_wrapper

print('Patched _request_wrapper! Testing model load...')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
print(f'Tokenizer loaded: {len(tokenizer)}')

import torch
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base', torch_dtype=torch.float32)
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')
print('SUCCESS!')
