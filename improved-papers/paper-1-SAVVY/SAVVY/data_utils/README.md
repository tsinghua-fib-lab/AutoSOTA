# SAVVY-Bench Data Preprocessing Utilities

This directory (`data_utils/`) contains data preprocessing tools for SAVVY-Bench, the first benchmark for dynamic 3D spatial reasoning in audio-visual environments, introduced in [SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing](https://arxiv.org/pdf/2506.05414). It provides downloading and preprocessing scripts for [Aria Everyday Activities Dataset, Meta Reality Labs-R](https://www.projectaria.com/datasets/aea/) videos used in SAVVY-Bench.

**Note:** This is part of the [SAVVY repository](https://github.com/shlizee/savvy/tree/main/SAVVY). For the main SAVVY algorithm code, see the parent directory.

<!-- ## SAVVY-Bench Dataset

The benchmark dataset is also available on Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("ZijunCui/SAVVY-Bench")
``` -->

## Setup Environment

### Step 1: Create Conda Environment
```bash
conda create -n savvy-bench python=3.10 -y
conda activate savvy-bench

# For AEA pre-processing
pip install requests tqdm numpy scipy opencv-python imageio open3d matplotlib tyro pillow natsort

# Install Project Aria Tools (ESSENTIAL for VRS file processing)
pip install 'projectaria-tools[all]'

# Install PyTorch
# CPU version:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or GPU (optional):
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Step 2: Initialize Submodules

Please ensure the egolifter submodule is initialized:

```bash
# From the SAVVY repo root
git submodule update --init --recursive
```

**Egolifter:** The EgoLifter submodule (`data_utils/egolifter/`) is unmodified from the [original source](https://github.com/facebookresearch/egolifter) and is licensed under Apache License 2.0.

## Download and Process Aria Everyday Activities Videos

### Step 1: Access the Dataset
1. Visit [Aria Everyday Activities Dataset](https://www.projectaria.com/datasets/aea/)
2. Follow the simple instructions to access the dataset
3. Download the `Aria Everyday Activities Dataset.json` file and place it in the data_utils/ directory

**Expected Download Size: 120.51 GB total**

| File Type                                   | Count    | Size      |
|---------------------------------------------|----------|-----------|
| main_vrs (Raw VRS recordings)               | 53 files | 93.09 GB  |
| mps_artifacts (Processing artifacts)        | 53 files | 10.94 GB  |
| mps_slam_points (3D point clouds)           | 53 files | 9.97 GB   |
| video_main_rgb (RGB preview videos)         | 53 files | 5.54 GB   |
| mps_slam_trajectories (Camera trajectories) | 53 files | 938.68 MB |
| mps_slam_calibration (Calibration data)     | 53 files | 55.73 MB  |
| mps_eye_gaze (Eye gaze tracking)            | 53 files | 2.42 MB   |
| annotations (Metadata)                      | 53 files | 282.03 KB |
| mps_slam_summary (Summary files)            | 53 files | 38.77 KB  |

### Step 2: Data Download and Undistortion
```bash
# Navigate to data_utils directory
cd data_utils

conda activate savvy-bench

# Download AEA data
python scripts/download_aea_videos.py

# Process and undistort
python scripts/process_aea_data.py

# Extract audio
python scripts/extract_vrs_audio.py aea/aea_data aea/aea_processed

# Convert to video
python scripts/process_videos.py aea/aea_processed aea/aea_processed
```

**Note:** Audio and video processing automatically trim segments based on timestamps in `aea/video_timestamps.txt`. The special sequences `loc2_script3_seq3_rec1`, `loc2_script3_seq3_rec2` are split into 4 segments each (seq31, seq32, seq33, seq34).

### Step 3: Verify Processing
After completion, you should have the following structure within `data_utils/`:

**Raw Data:** `data_utils/aea/aea_data/` (53 raw VRS recordings with MPS SLAM data)

**Processed Data:** `data_utils/aea/aea_processed/` with structure:

```
data_utils/aea/aea_processed/
├── loc1_script2_seq1_rec1/
│   ├── audio/
│   │   └── loc1_script2_seq1_rec1.wav  # Timestamp-trimmed audio (mic 5,6 @ 48kHz)
│   ├── video/
│   │   └── loc1_script2_seq1_rec1.mp4  # Timestamp-trimmed video (20 FPS, undistorted)
│   ├── images/                         # Undistorted RGB frames
│   │   ├── frame_000001.jpg
│   │   └── ...
│   └── transforms.json                 # Camera poses & intrinsics (3DGS format)
├── loc1_script2_seq1_rec2/
│   └── ...
└── [56 total sequences including seq31-34 splits]
```
## Merge audio and video

Use `scripts/merge.sh` to iterate over all processed sequences (`audio/` and `video/`) and muxes the trimmed audio back into the corresponding video files.

**Prerequisites**:
- `ffmpeg` must be installed and available on PATH.
    - On Debian/Ubuntu: `sudo apt install ffmpeg`
    - Conda: `conda install -c conda-forge ffmpeg`

**Details**:
- This is useful when you need the final muxed videos for visualization or downstream evaluation.
- Reads `${sequence}/video/${sequence}.mp4` and `${sequence}/audio/${sequence}.wav` for every sequence directory under `aea/aea_processed`.
- Copies the video stream, encodes audio to AAC (48 kHz, stereo), and writes the result to `${sequence}/video_merged/${sequence}.mp4`.
- Existing files in `video_merged/` will be overwritten.  

**Usage**:

```bash
# from repo root
cd SAVVY/data_utils/scripts
bash merge.sh
```

## Citation

```bibtex
@article{chen2025savvy,
    title={SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing},
    author={Mingfei Chen and Zijun Cui and Xiulong Liu and Jinlin Xiang and Caleb Zheng and Jingyuan Li and Eli Shlizerman},
    year={2025},
    eprint={2506.05414},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
