import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/autosota_cache/hf'

# Patch _request_wrapper
import huggingface_hub.file_download as fd
_original_rw = fd._request_wrapper
def _patched_rw(**kwargs):
    kwargs['allow_redirects'] = True
    return _original_rw(**kwargs)
fd._request_wrapper = _patched_rw

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
print(f'Tokenizer OK, vocab={len(tokenizer)}')

print('Loading model...')
model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base', torch_dtype=torch.float32)
print(f'Model OK, params={sum(p.numel() for p in model.parameters()):,}')
print('SUCCESS!')
