<div align="center">
<h1>[CVPR 2026] MixerCSeg </h1>
<h3>MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention</h3>

Zilong Zhao<sup>1</sup>,
Zhengming Ding<sup>2</sup>,
Pei Niu<sup>1</sup>, 
Wenhao Sun<sup>1</sup>, 
Feng Guo<sup>1</sup>, <sup>*</sup>

<sup>1</sup>  School of Qilu Transportation, Shandong University, China, <sup>2</sup>  Department of Computer Science, Tulane University,  USA.

Paper: ([arXiv 2603.01361](https://arxiv.org/abs/2603.01361))

</div>


<!-- ## 
* [**updates**](#white_check_mark-updates)
* [**abstract**](#abstract)
* [**overview**](#overview--derivations)
* [**main results**](#main-results)
* [**getting started**](#getting-started)
* [**star history**](#star-history)
* [**citation**](#citation)
* [**acknowledgment**](#acknowledgment) -->


## 💥 News 💥
* **`Feb. 28th, 2026`**: **We have released the checkpoints and inference results for all datasets !**
* **`Feb. 27th, 2026`**: **We have released the code for MixerCSeg!**
* **`Feb. 23th, 2026`**: **MixerCSeg has been accepted to CVPR 2026 !**



## Abstract

<p align="center">
    <img src="./figure/overview.png" alt="Overview" />
</p>
Feature encoders play a key role in pixel-level crack segmentation by shaping the representation of fine textures and thin structures. Existing CNN-, Transformer-, and Mamba-based models each capture only part of the required spatial or structural information, leaving clear gaps in modeling complex crack patterns. To address this, we present MixerCSeg, a mixer architecture designed like a coordinated team of specialists, where CNN-like pathways focus on local textures, Transformer-style paths capture global dependencies, and Mamba-inspired flows model sequential context within a single encoder. At the core of MixerCSeg is the TransMixer, which explores Mamba’s latent attention behavior while establishing dedicated pathways that naturally express both locality and global awareness. To further enhance structural fidelity, we introduce a spatial block processing strategy and a Direction-guided Edge Gated Convolution (DEGConv) that strengthens edge sensitivity under irregular crack geometries with minimal computational overhead. A Spatial Refinement Multi-Level Fusion (SRF) module is then employed to refine multi-scale details without increasing complexity. Extensive experiments on multiple crack segmentation benchmarks show that MixerCSeg achieves state-of-the-art performance with only 2.05 GFLOPs and 2.54 M parameters, demonstrating both efficiency and strong representational capability.




## Installation
```
conda create -n MixerCSeg python=3.10 -y
conda activate MixerCSeg

pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

pip install -U openmim
mim install mmcv-full

pip install -r requirements.txt

cd VMamba/models/kernels/selective_scan/
python setup.py install

pip install numpy==1.23
```



## Getting Started
### Train your model

You can modify the parameters in the main.py file and run it with the following command:
```
python main.py --dataset_path [your_dataset_path]
```

### Test
You can perform inference on checkpoints using the following command. **Note:** Please set the dataset file at line 17 of test.py, and specify the checkpoint location at line 24:
```
python test.py
```
Calculate performance metrics using the following command. Please ensure to configure your result path:
```
python eval/evaluate.py --result_path [your_results_path]
```

### Datasets and Checkpoints
- The datasets used in our work: [CamCrack79, DeepCrack, Crack500 and CrackMap](https://drive.google.com/drive/folders/1H-YwfW6gc--23bDyC92kQzTFK1a4KGmW?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto).
- Obtain our [visualization results](https://drive.google.com/drive/folders/1yDdaMIrMsjmPcp2bssOG2-hKQi5UclKl?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto) and [checkpoints](https://drive.google.com/drive/folders/13cfgbyYiBMcxdLLkQTB984RyDQOiJ1B4?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto) on four datasets.
- Move the downloaded checkpoint file to the designated path: `./checkpoints/weights/checkpoint_[your_dataset_name]/`



## Acknowledgment
This project is based on [SCSegamba](https://github.com/Karl1109/SCSegamba), [VMamba](https://github.com/MzeroMiko/VMamba), [HiddenMambaAttn](https://github.com/AmeenAli/HiddenMambaAttn), [LongMamba](https://github.com/GATECH-EIC/LongMamba) and [DeciMamba](https://github.com/assafbk/DeciMamba), thanks for their excellent works.


## Citation
If you are using our MixerCSeg for your research, please cite the following paper:
```
@article{zhao2026mixercseg,
  title={MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention},
  author={Zhao, Zilong and Ding, Zhengming and Niu, Pei and Sun, Wenhao and Guo, Feng},
  journal={arXiv preprint arXiv:2603.01361},
  year={2026}
}
```

## Concat

If you have any other questions, feel free to contact me at **zzl000503@163.com**.





