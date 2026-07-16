import gzip
import os
import shutil
import tarfile

from huggingface_hub import snapshot_download

root_dir = "./DataDecide-eval-instances"

### 1. download
snapshot_download(
    repo_id="allenai/DataDecide-eval-instances",
    repo_type="dataset",
    local_dir=root_dir,
)

### 2. unzip
dirs_to_visit = [root_dir]

while dirs_to_visit:
    current_dir = dirs_to_visit.pop()
    with os.scandir(current_dir) as it:
        for entry in it:
            path = entry.path

            if entry.is_dir():
                if ".cache" in path.split(os.sep):
                    continue
                dirs_to_visit.append(path)
                continue

            if path.endswith(".jsonl.gz"):
                dest = path[:-3]
                if os.path.exists(dest):
                    print(f"[jsonl.gz] Skipping {dest}")
                else:
                    with gzip.open(path, "rb") as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"[jsonl.gz] Wrote {dest}")

            if path.endswith(".tar.gz"):
                without_gz = os.path.splitext(path)[0]
                dest_dir = os.path.splitext(without_gz)[0]
                if os.path.isdir(dest_dir):
                    print(f"[tar.gz] Skipping {dest_dir}")
                    dirs_to_visit.append(dest_dir)
                else:
                    os.makedirs(dest_dir, exist_ok=True)
                    with tarfile.open(path, "r:gz") as tar:
                        tar.extractall(path=dest_dir)
                    print(f"[tar.gz] Extracted {path} -> {dest_dir}")
                    dirs_to_visit.append(dest_dir)
