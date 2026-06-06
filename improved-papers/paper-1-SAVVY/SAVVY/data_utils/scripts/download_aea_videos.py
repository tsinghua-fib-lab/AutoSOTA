"""
AEA download data code for SAVVY-Bench data preprocessing - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""


import json
from pathlib import Path
import requests
from tqdm import tqdm
import hashlib
import zipfile

def calculate_file_hash(filepath):
    sha1 = hashlib.sha1()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha1.update(chunk)
    return sha1.hexdigest()

def download_file_with_resume(url, filepath, expected_hash=None):
    if filepath.exists() and expected_hash:
        if calculate_file_hash(filepath) == expected_hash:
            return

    filepath.parent.mkdir(parents=True, exist_ok=True)
    resume_pos = filepath.stat().st_size if filepath.exists() else 0

    headers = {'Range': f'bytes={resume_pos}-'} if resume_pos > 0 else {}
    response = requests.get(url, stream=True, headers=headers)
    total_size = int(response.headers.get('content-length', 0)) + resume_pos

    mode = 'ab' if resume_pos > 0 else 'wb'
    with open(filepath, mode) as f, tqdm(
        desc=filepath.name,
        total=total_size,
        initial=resume_pos,
        unit='B',
        unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def extract_zip_file(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def download_scene(scene_name, config, aea_data_dir):
    scene_dir = aea_data_dir / scene_name
    scene_dir.mkdir(exist_ok=True)

    print(f"Processing: {scene_name}")

    scene_data = config["sequences"][scene_name]

    if "main_vrs" in scene_data:
        vrs_info = scene_data["main_vrs"]
        vrs_path = scene_dir / "recording.vrs"
        download_file_with_resume(vrs_info["download_url"], vrs_path, vrs_info.get("sha1sum"))

    slam_dir = scene_dir / "mps" / "slam"
    slam_dir.mkdir(parents=True, exist_ok=True)

    for file_type in ["mps_slam_trajectories", "mps_slam_points"]:
        if file_type in scene_data:
            file_info = scene_data[file_type]
            zip_path = scene_dir / file_info["filename"]
            download_file_with_resume(file_info["download_url"], zip_path, file_info.get("sha1sum"))
            extract_zip_file(zip_path, slam_dir)

def main():
    script_dir = Path(__file__).parent
    savvy_bench_root = script_dir.parent

    aea_data_dir = savvy_bench_root / "aea" / "aea_data"
    aea_data_dir.mkdir(parents=True, exist_ok=True)

    config_file = savvy_bench_root / "Aria Everyday Activities Dataset.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    video_ids_file = savvy_bench_root / "aea" / "video_ids.txt"
    with open(video_ids_file, 'r') as f:
        scenes = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"Downloading {len(scenes)} scenes...")

    for scene in scenes:
        download_scene(scene, config, aea_data_dir)

if __name__ == "__main__":
    main()
