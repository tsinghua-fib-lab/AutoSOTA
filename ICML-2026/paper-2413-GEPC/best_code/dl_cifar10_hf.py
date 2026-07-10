import os, sys, tarfile
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Try huggingface_hub to download from torch/cifar10
from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "torch/cifar10"
token = os.environ.get("HF_TOKEN", None)

print(f"Listing files in {repo_id}...")
try:
    files = list_repo_files(repo_id, token=token)
    print(f"Files: {files[:10]}")
except Exception as e:
    print(f"list_repo_files failed: {e}")
    # Try direct download
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="cifar-10-python.tar.gz",
            local_dir="/datasets",
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"Downloaded to {path}")
    except Exception as e2:
        print(f"hf_hub_download also failed: {e2}")
