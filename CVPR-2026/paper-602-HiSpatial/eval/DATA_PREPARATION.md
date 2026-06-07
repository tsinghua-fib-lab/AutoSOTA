# Benchmark Data Preparation

CV-Bench, RoboSpatial, SpatialRGPT are loaded automatically via
huggingFace datasets and require no manual
download. The three benchmarks below need additional files prepared beforehand.


## EmbSpatial

Download `embspatial_bench.json` from the HuggingFace dataset repository:

<https://huggingface.co/datasets/Phineas476/EmbSpatial-Bench>

Then pass the path to the eval script:

```bash
python eval/eval_emb_spatial.py \
    --benchmark_path /path/to/embspatial_bench.json \
    --vlm_model_path lhzzzzzy/HiSpatial-3B \
    --save_path results/embspatial
```


## 3DSRBench

Download `3dsrbench_v1_vlmevalkit_circular.tsv` from the HuggingFace dataset repository:

<https://huggingface.co/datasets/ccvl/3DSRBench>

Then pass the path to the eval script:

```bash
python eval/eval_3dsrbench.py \
    --tsv_path /path/to/3dsrbench_v1_vlmevalkit_circular.tsv \
    --vlm_model_path lhzzzzzy/HiSpatial-3B \
    --save_path results/3dsrbench
```


## Q-Spatial (ScanNet split)

The QSpatial-ScanNet split requires local ScanNet RGB-D data. Please follow the
[official ScanNet instructions](http://www.scan-net.org/) to download the dataset.

Each scene folder should contain the following structure:

```
scannet_images/
└── scene0015_00/
    ├── color/
    │   ├── 0.jpg
    │   └── ...
    ├── depth/
    │   ├── 0.png
    │   └── ...
    └── intrinsics_depth.npy
```

Then pass the root directory to the eval script:

```bash
python eval/eval_q_spatial.py \
    --scannet_images_dir /path/to/scannet_images \
    --vlm_model_path lhzzzzzy/HiSpatial-3B \
    --save_path results/qspatial
```
