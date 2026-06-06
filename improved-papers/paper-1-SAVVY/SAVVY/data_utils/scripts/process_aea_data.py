"""
AEA data preprocessing code for SAVVY-Bench data preprocessing - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""
import sys
from pathlib import Path
import subprocess

def process_scene(scene_name, egolifter_path, aea_raw_root, aea_processed_root):
    scene_raw_dir = aea_raw_root / scene_name
    scene_processed_dir = aea_processed_root / scene_name
    scene_processed_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        sys.executable,
        str(egolifter_path / "scripts" / "process_project_aria_3dgs.py"),
        "--vrs_file", str(scene_raw_dir / "recording.vrs"),
        "--mps_data_dir", str(scene_raw_dir / "mps" / "slam"),
        "--output_dir", str(scene_processed_dir)
    ], check=True)

    source_points = scene_raw_dir / "mps" / "slam" / "semidense_points.csv.gz"
    target_points = scene_processed_dir / "global_points.csv.gz"
    target_points.symlink_to(source_points.absolute())

    subprocess.run([
        sys.executable,
        str(egolifter_path / "scripts" / "rectify_aria.py"),
        "-i", str(aea_processed_root),
        "-o", str(aea_processed_root),
        "-s", scene_name
    ], check=True, cwd=str(egolifter_path))

def main():
    script_dir = Path(__file__).parent
    savvy_bench_root = script_dir.parent

    egolifter_path = savvy_bench_root / "egolifter"
    aea_raw_root = savvy_bench_root / "aea" / "aea_data"
    aea_processed_root = savvy_bench_root / "aea" / "aea_processed"

    aea_raw_root.mkdir(parents=True, exist_ok=True)
    aea_processed_root.mkdir(parents=True, exist_ok=True)

    video_ids_file = savvy_bench_root / "aea" / "video_ids.txt"
    with open(video_ids_file, 'r') as f:
        scenes = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    for scene in scenes:
        process_scene(scene, egolifter_path, aea_raw_root, aea_processed_root)

if __name__ == "__main__":
    main()
