from huggingface_hub import list_repo_files, hf_hub_download
import os, shutil

repo_id = "ottogin/locality-diffusion-baselines"

all_files = list_repo_files(repo_id)
files_to_download = [f for f in all_files if f.startswith("models/baseline_unet/")]

for f in files_to_download:
    cached_path = hf_hub_download(repo_id=repo_id, filename=f)  # download into HF cache
    rel = f.removeprefix("models/")  # strip leading "models/"
    dst = os.path.join("base_models", rel)  # base_models/baseline_unet/...
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(cached_path, dst)

print("done, count =", len(files_to_download))
