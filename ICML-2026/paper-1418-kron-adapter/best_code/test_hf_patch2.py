import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/autosota_cache/hf'

# Monkey-patch the huggingface_hub session
from huggingface_hub import get_session

session = get_session()
_original_head = session.head

def _patched_head(url, **kwargs):
    kwargs['allow_redirects'] = True
    return _original_head(url, **kwargs)

session.head = _patched_head

# Now also need to make future get_session() calls return the patched session
from huggingface_hub import _http
_original_get_session = _http.get_session
_original_get_session2 = _http._get_session_from_cache

_sessions_cache = {}
def _patched_get_session():
    global _sessions_cache
    if 'default' not in _sessions_cache:
        s = _original_get_session()
        _oh = s.head
        def _ph(url, **kw):
            kw['allow_redirects'] = True
            return _oh(url, **kw)
        s.head = _ph
        _sessions_cache['default'] = s
    return _sessions_cache['default']

_http.get_session = _patched_get_session

print('Patched! Testing model load...')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
print(f'Tokenizer loaded: {len(tokenizer)}')

# Also test model load
import torch
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base', torch_dtype=torch.float32)
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')
print('SUCCESS!')
