![Python ==3.8](https://img.shields.io/badge/Python-==3.8-yellow.svg)
![PyTorch ==1.12.0](https://img.shields.io/badge/PyTorch-==1.12.0-blue.svg)

# [Neruips2025] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification
The official repository for MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification [[pdf]](https://openreview.net/attachment?id=7jg26Fd1ra&name=pdf)

### Prepare Datasets

```bash
mkdir data
```
Download the person datasets [RGBNT201](https://drive.google.com/drive/folders/1EscBadX-wMAT56_It5lXY-S3-b5nK1wH), [RGBNT100](https://pan.baidu.com/s/1xqqh7N4Lctm3RcUdskG0Ug) (code：rjin), and the [MSVR310](https://drive.google.com/file/d/1IxI-fGiluPO_Ies6YjDHeTEuVYhFdYwD/view?usp=drive_link).

### Installation

```bash
pip install -r requirements.txt
```

### Prepare ViT Pre-trained Models

You need to download the pretrained CLIP model: [ViT-B-16](https://pan.baidu.com/s/1YPhaL0YgpI-TQ_pSzXHRKw) (Code：52fu)

## Training

You can train the MDReID with:

```bash
python train_net.py --config_file configs/RGBNT201/MDReID.yml
```
**Some examples:**
```bash
python train_net.py --config_file configs/RGBNT201/MDReID.yml
```

1. The device ID to be used can be set in config/defaults.py

2. If you need to train on the RGBNT100 and MSVR310 datasets, please ensure the corresponding path is modified accordingly.


## Evaluation

```bash
python test_net.py --config_file 'choose which config to test' --model_path 'your path of trained checkpoints'
```

**Some examples:**
```bash
python test_net.py --config_file configs/MSVR310/MDReID.yml --model_path MSVR310_MDReIDbest.pth
```

#### Results
|  Dataset  | Rank@1 | mAP  | Model |
|:---------:|:------:|:----:| :------: |
| RGBNT201  |  85.2  | 82.1 | [model](https://drive.google.com/file/d/14kd4rUdSJT26UKKQs_Emxx9iL0bi69da/view?usp=drive_link) |
| RGBNT100  |  95.6  | 85.3 | [model](https://drive.google.com/file/d/1g8Y28L7MUauVDq_XVFngXYWMULOd8X4M/view?usp=drive_link) |
| MSVR310   |  68.9  | 51.0 | [model](https://drive.google.com/file/d/1eTkFMuM7Efpphnwv0ZT5Iix1yv2Db6mb/view?usp=drive_link) |


## Citation
Please kindly cite this paper in your publications if it helps your research:
```bash
@inproceedings{feng2025mdreid,
  title={MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification},
  author={Yingying, Feng and Jie, Li and Jie, Hu and Yukang, Zhang and Lei, Tan and Jiayi, Ji},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgement
Our code is based on [TOP-ReID](https://github.com/924973292/TOP-ReID)[1]

## References
[1]Wang Yuhao, Liu Xuehu, Zhang Pingping, Lu Hu, Tu Zhengzheng, Lu Huchuan. 2024. TOP-ReID: Multi-Spectral Object Re-identification with Token Permutation. Proceedings of the AAAI Conference on Artificial Intelligence. 38. 5758-5766.

## Contact

If you have any questions, please feel free to contact us. E-mail: [tanlei@stu.xmu.edu.cn](mailto:lei.tan@nus.edu.sg)
