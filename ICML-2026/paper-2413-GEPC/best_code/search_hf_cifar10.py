import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import HfApi, list_repo_files

api = HfApi()

# Search for datasets with cifar10 in name
try:
    results = api.list_datasets(search="cifar-10-python", limit=10)
    for r in results:
        print(f"  {r.id}")
except Exception as e:
    print(f"Search error: {e}")

# Also try listing known repos
repos = [
    "cifar10",
    "uoft-cs/cifar10", 
    "tanganke/cifar10",
]
for repo in repos:
    try:
        files = list_repo_files(repo, repo_type="dataset")
        print(f"\n{repo}: {files[:15]}")
    except Exception as e:
        pass
