import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/autosota_cache/hf'

# Direct patch on requests.Session class level
import requests

_original_head = requests.Session.head
def _patched_head(self, url, **kwargs):
    kwargs['allow_redirects'] = True
    return _original_head(self, url, **kwargs)

requests.Session.head = _patched_head

print('Patched requests.Session.head! Testing model load...')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
print(f'Tokenizer loaded: {len(tokenizer)}')

import torch
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base', torch_dtype=torch.float32)
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')
print('SUCCESS!')
