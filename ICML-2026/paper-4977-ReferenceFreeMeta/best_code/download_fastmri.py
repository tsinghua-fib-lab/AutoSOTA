import os, requests, sys

os.environ["no_proxy"] = ""
os.environ["https_proxy"] = "http://172.17.0.1:17890"
os.environ["http_proxy"] = "http://172.17.0.1:17890"

token = os.environ.get("HF_TOKEN")
headers = {"Authorization": "Bearer " + token}

print("Fetching file list...", flush=True)
api_url = "https://huggingface.co/api/datasets/btoto3/fastmri-dl"
resp = requests.get(api_url, headers=headers, timeout=30)
data = resp.json()
siblings = data["siblings"]

train_files = [s for s in siblings if "singlecoil_train" in s["rfilename"]]
test_files = [s for s in siblings if "singlecoil_test" in s["rfilename"]]
print("Train: %d, Test: %d" % (len(train_files), len(test_files)), flush=True)

local_dir = "/datasets/fastmri_dl"
os.makedirs(local_dir, exist_ok=True)

# Download 5 train + 5 test
download_list = train_files[:5] + test_files[:5]
for i, f_info in enumerate(download_list):
    fname = f_info["rfilename"]
    local_name = fname.replace("/", "_")
    local_path = os.path.join(local_dir, local_name)
    
    if os.path.exists(local_path):
        sz = os.path.getsize(local_path)
        print("[%d/%d] EXISTS: %s (%.1fMB)" % (i+1, len(download_list), local_name, sz/1024/1024), flush=True)
        continue
    
    url = "https://huggingface.co/datasets/btoto3/fastmri-dl/resolve/main/%s" % fname
    print("[%d/%d] Downloading: %s" % (i+1, len(download_list), fname), flush=True)
    
    resp = requests.get(url, headers=headers, timeout=180, stream=True)
    total_size = int(resp.headers.get("Content-Length", 0))
    
    downloaded = 0
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
    
    print("  Done: %.1fMB" % (downloaded / 1024 / 1024), flush=True)

print("All downloads complete!", flush=True)
