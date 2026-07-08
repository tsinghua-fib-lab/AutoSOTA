import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/autosota_cache/hf'

# Patch at the source: monkey-patch huggingface_hub.file_download module
import huggingface_hub.file_download as fd_mod
import huggingface_hub.utils._http as http_mod

# Get the module-level get_session function
_original_get_session = http_mod.get_session

def _patched_get_session():
    s = _original_get_session()
    _oh = s.head
    def _ph(url, **kw):
        kw['allow_redirects'] = True
        return _oh(url, **kw)
    s.head = _ph
    return s

http_mod.get_session = _patched_get_session

# Also patch fd_mod._get_metadata_or_catch_error to handle 308
_orig_get_metadata = fd_mod._get_metadata_or_catch_error

def _patched_get_metadata(url, **kwargs):
    try:
        return _orig_get_metadata(url, **kwargs)
    except Exception:
        pass
    raise RuntimeError(f'Still failed to get metadata for {url}')

print('Patched! Testing model load...')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
print(f'Tokenizer loaded: {len(tokenizer)}')

import torch
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base', torch_dtype=torch.float32)
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')
print('SUCCESS!')
