# OoD Datasets

The proposed OoD Datasets are available on [Hugging Face](https://huggingface.co/datasets/Simom0/OoD-Datasets).

Each dataset contains two different versions, a *single* and a *multi* split, comprising respectively a single anomaly object or multiple ones in each single scan. Download the zip files and extract them.

### SemanticKITTI-OoD

SemanticKITTI-OoD is derived from the [SemanticKITTI](https://semantic-kitti.org/) validation sequence (08). The structure is as follows:

```
SemanticKITTI-OoD
├── kitti-ood/
|   └── sequences/
|       └── 08/
|           ├── velodyne/
|               ├── 000000.bin
|               └── ...
|           └── labels/
|               ├── 000000.label
|               └── ...
└── kitti-ood-multi/
    └── ...
```

### SemanticPOSS-OoD

SemanticPOSS-OoD is derived from the [SemanticPOSS](http://www.poss.pku.edu.cn/semanticposs.html) test sequence (02). The structure is as follows:

```
SemanticPOSS-OoD
├── poss-ood/
|   └── dataset/
|       └── sequences/
|           └── 02/
|               ├── velodyne/
|                   ├── 000001.bin
|                   └── ...
|               └── labels/
|                   ├── 000001.label
|                   └── ...
└── poss-ood-multi/
    └── ...
```

### nuScenes-OoD

nuScenes-OoD is generated from the [nuScenes](https://www.nuscenes.org) validation set (converted into KITTI format). The structure is as follows:

```
nuScenes-OoD
├── nuscenes-ood/
|    └── sequences/
|        ├── 0003/
|        |   ├── velodyne/
|        |       ├── 000000.bin
|        |       └── ...
|        |   └── labels/
|        |       ├── 000000.label
|        |       └── ...
|        ├── 0012/
|        └── ...
└── nuscenes-ood-multi/
    └── ...
```

### Anomaly labels

For both SemanticKITTI-OoD and SemanticPOSS-OoD datasets, anomaly points are labeled with label 2, following STU settings. For nuScenes-OoD, as label 2 is already used, we selected 100 as value to identify anomalous objects.

| Dataset | Label ID |
|---|---|
| STU | 2|
| SemanticKITTI-OoD | 2 |
| SemanticPOSS-OoD | 2 |
| nuScenes-OoD | 100 |

## Visualization

To visualize the datasets with ground truth annotations and/or predictions run the following scripts. The `vispy` package is required.

```bash
### SemanticKITTI-OoD
python3 visualize_ood_dataset.py --dataset /path/to/kitti-ood/ --config config/labels/semantic-kitti.yaml --sequence 08 [--predictions /path/to/predictions/]

### SemanticPOSS-OoD
python3 visualize_ood_dataset.py --dataset /path/to/poss-ood/ --config config/labels/semantic-poss.yaml --sequence 02 [--predictions /path/to/predictions/]

### nuScenes-OoD
python3 visualize_ood_dataset.py --dataset /path/to/nuscenes-ood/ --config config/labels/nuscenes.yaml --sequence 000X [--predictions /path/to/predictions/]
```

## Dataset Generation

Available soon

### License

The proposed OoD dataset are based on the SemanticKITTI, SemanticPOSS and nuScenes benchmarks and therefore we distribute the data under Creative Commons Attribution-NonCommercial-ShareAlike license. You are free to share and adapt the data, but have to give appropriate credit and may not use the work for commercial purposes. Please refer to the original license of each dataset.

| OoD Dataset | Original Dataset | Original License | Reference |
|---|---|---|---|
| SemanticKITTI-OoD | SemanticKITTI | CC BY-NC-SA 4.0 | [link](https://semantic-kitti.org/dataset.html#licence) |
| SemanticPOSS-OoD | SemanticPOSS | CC BY-NC-SA 3.0 | [link](http://www.poss.pku.edu.cn/semanticposs.html) |
| nuScenes-OoD | nuScenes | CC BY-NC-SA 4.0 | [link](https://www.nuscenes.org/terms-of-use) |

Specifically, you should cite our work ([PDF](https://arxiv.org/pdf/2604.23604)):

```
@inproceedings{mosco2026learning,
    title={Learning to Identify Out-of-Distribution Objects for 3D LiDAR Anomaly Segmentation},
    author={Mosco, Simone and Fusaro, Daniel and Pretto, Alberto},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2026}
}
```

But also the original datasets SemanticKITTI, SemanticPOSS and nuScenes:

```
@inproceedings{behley2019iccv,
  author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
  title = {{SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences}},
  booktitle = {Proc. of the IEEE/CVF International Conf.~on Computer Vision (ICCV)},
  year = {2019}
}
```

```
@inproceedings{pan2020semanticposs,
  title={Semanticposs: A point cloud dataset with large quantity of dynamic instances},
  author={Pan, Yancheng and Gao, Biao and Mei, Jilin and Geng, Sibo and Li, Chengkun and Zhao, Huijing},
  booktitle={2020 IEEE intelligent vehicles symposium (IV)},
  pages={687--693},
  year={2020},
  organization={IEEE}
}
```

```
@inproceedings{caesar2020nuscenes,
  title={nuscenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11621--11631},
  year={2020}
}

```