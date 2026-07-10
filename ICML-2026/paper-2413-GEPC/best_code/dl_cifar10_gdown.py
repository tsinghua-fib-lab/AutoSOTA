import os, sys, tarfile, shutil

# Use gdown to download from Google Drive
# CIFAR-10 on Google Drive: https://drive.google.com/file/d/1hX9r2kQ_nIeQBJmH3x2vB7YJB8XFtVXk/view
import gdown

file_id = "1hX9r2kQ_nIeQBJmH3x2vB7YJB8XFtVXk"
url = f"https://drive.google.com/uc?id={file_id}"
output = "/autosota_cache/tmp/cifar-10-python.tar.gz"
datasets_dir = "/datasets"

os.makedirs("/autosota_cache/tmp", exist_ok=True)

print(f"Downloading CIFAR-10 via gdown from Google Drive...")
gdown.download(url, output, quiet=False)

if os.path.exists(output) and os.path.getsize(output) > 0:
    print(f"Downloaded {os.path.getsize(output)} bytes, extracting...")
    with tarfile.open(output, 'r:gz') as tar:
        tar.extractall(path=datasets_dir)
    os.remove(output)
    print(f"Extracted to {datasets_dir}/cifar-10-batches-py/")
else:
    print("Download failed")
    sys.exit(1)
