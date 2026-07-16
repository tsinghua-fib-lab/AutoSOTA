import os
from huggingface_hub import HfApi

api = HfApi()

repo_id = "yuhengtu/irsl_datadecide"
repo_type = "dataset"
root = "."

for dirpath, _, filenames in os.walk(root):
    for fname in filenames:
        if fname.endswith(".parquet"):
            fpath = os.path.join(dirpath, fname)
            rel = os.path.basename(fpath)

            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=rel,
                repo_id=repo_id,
                repo_type=repo_type,
            )
