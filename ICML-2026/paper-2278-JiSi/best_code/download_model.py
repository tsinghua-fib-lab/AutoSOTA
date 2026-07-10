import os
import sys

# Remove proxy env vars that might cause issues with httpx
for var in ['ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']:
    os.environ.pop(var, None)

# Set HF endpoint
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

print("Starting download of gte-Qwen2-7B-instruct...")
model_dir = snapshot_download(
    repo_id="Alibaba-NLP/gte-Qwen2-7B-instruct",
    local_dir="/models/gte-Qwen2-7B-instruct",
    token=os.environ.get("HF_TOKEN"),
    resume_download=True,
)
print(f"Model downloaded to: {model_dir}")
print("Download complete!")
