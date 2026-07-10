import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import list_repo_files, hf_hub_download, HfApi

api = HfApi()

# Search for cifar10 datasets
try:
    # Try specific repos
    repos_to_try = [
        "uoft-cs/cifar10",
        "cifar10-resized/cifar10",
        "activeeon/cifar10",
        "tanganke/cifar-10",
    ]
    for repo in repos_to_try:
        try:
            files = list_repo_files(repo, repo_type="dataset")
            print(f"{repo}: {files[:10]}")
        except Exception as e:
            print(f"{repo}: {type(e).__name__}")
except Exception as e:
    print(f"Error: {e}")
