# REALM
> **[CVPR 2026] REALM: An MLLM-Agent Framework for Open-World Reasoning Segmentation and Editing on 3D Gaussian Splatting**

> [[Project Page]](https://changyueshi.github.io/REALM/)

## **🔧Preparation**
Clone this repository
```
git clone https://github.com/ChangyueShi/REALM-Code
cd REALM-Code
```
prepare the environment
```
conda create -n realm python=3.8 -y
conda activate realm 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
## **📊Dataset**
The dataset format is consistent with the [GS-Group](https://huggingface.co/mqye/Gaussian-Grouping/tree/main)

```
data
|____bear
|____lerf
| |____figurines
|____mipnerf360
| |____counter
```


## **🚀Training and Inference**
Optimize the 3D feature field
```
bash script/train.sh lerf/figurines 1
```
Inference using MLLM-Agent
```
bash run_seg.sh
```