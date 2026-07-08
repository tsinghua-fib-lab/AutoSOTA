import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/autosota_cache/hf'

# Patch _request_wrapper
import huggingface_hub.file_download as fd
_orw = fd._request_wrapper
def _prw(**kwargs):
    kwargs['allow_redirects'] = True
    return _orw(**kwargs)
fd._request_wrapper = _prw

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load from cache snapshot directory directly
model_path = '/autosota_cache/hf/hub/models--google-t5--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'

print(f'Loading from: {model_path}')
print(f'Files: {os.listdir(model_path)}')

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    local_files_only=True
)
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')
print('SUCCESS!')
