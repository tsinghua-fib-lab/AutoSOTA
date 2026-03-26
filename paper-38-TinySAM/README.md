# TinySAM
### **TinySAM: Pushing the Envelope for Efficient Segment Anything Model**

*Han Shu, Wenshuo Li, Yehui Tang, Yiman Zhang, Yihao Chen, Houqiang Li, Yunhe Wang, Xinghao Chen*

*AAAI 2025*

<a href="https://arxiv.org/abs/2312.13789"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red&logo=arxiv"></a>
<a href="https://huggingface.co/spaces/merve/tinysam"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=Demo&color=yellow"></a>
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/shuh15/TinySAM)

https://github.com/user-attachments/assets/c4bb50ab-bc20-419d-8c22-e36fe6a661ee

## Updates
* **2025/01/09**: TinySAM is accepted by AAAI 2025.
* **2024/01/06**: [Demo](https://openxlab.org.cn/apps/detail/shuh15/TinySAM) of TinySAM is now available in **OpenXLab**. Thanks for the GPU grant.
* **2023/12/27**: [Models](https://huggingface.co/merve/tinysam) and [demo](https://huggingface.co/spaces/merve/tinysam) of TinySAM are now available in **Hugging Face**. Thanks for [merveenoyan](https://github.com/merveenoyan).
* **2023/12/27**: Pre-trained models and codes of [Q-TinySAM](#usage) (quantized variant) are released.
* **2023/12/27**: [Evaluation](#evaluation) codes for zero-shot instance segmentation task on COCO are released.
* **2023/12/22**: Pre-trained models and codes of TinySAM are released both in [Pytorch](https://github.com/xinghaochen/TinySAM) and [Mindspore](https://gitee.com/mindspore/models/tree/master/research/cv/TinySAM).

## Overview

We propose a framework to obtain a tiny segment anything model (**TinySAM**) while maintaining the strong zero-shot performance. We first propose a full-stage knowledge distillation method with online hard prompt sampling strategy to distill a lightweight student model. We also adapt the post-training quantization to the promptable segmentation task and further reducing the computational cost. Moreover, a hierarchical segmenting everything strategy is proposed to accelerate the everything inference by with almost no performance degradation. With all these proposed methods, our TinySAM leads to orders of magnitude computational reduction and pushes the envelope for efficient segment anything task. Extensive experiments on various zero-shot transfer tasks demonstrate the significantly advantageous performance of our TinySAM against counterpart methods.

![framework](./fig/framework.png)
<div align=center>
<sup>Figure 1: Overall framework and zero-shot results of TinySAM.</sup>
</div>

![everything](./fig/everything.png)
<div align=center>
<sup>Figure 2: Our hierarchical strategy for everything mode.</sup>
</div>

![vis](./fig/vis.png)
<div align=center>
<sup>Figure 3: Visualization results of TinySAM.</sup>
</div>

## Requirements
The code requires `python>=3.7` and we use `torch==1.10.2` and `torchvision==0.11.3`. To visualize the results, `matplotlib>=3.5.1` is also required.  
- python 3.7
- pytorch == 1.10.2
- torchvision == 0.11.3
- matplotlib==3.5.1

## Usage

1. Download [checkpoints](#evaluation) into the directory of *weights*.

2. Run the demo code for single prompt of point or box.

```
python demo.py
```
3. Run the demo code for hierarchical segment everything strategy.
```
python demo_hierachical_everything.py
```

4. Run the demo code for quantization inference.
```
python demo_quant.py
```

## Evaluation
We follow the setting of original [SAM](https://arxiv.org/abs/2304.02643) paper and evaluate the zero-shot instance segmentaion on COCO and LVIS dataset. The experiment results are described as followed.

| Model               | FLOPs (G) |COCO AP (%) | LVIS AP (%)| 
| ------------------- | -------- | ------- |------- |
| SAM-H                 |2976| 46.6/46.5*     | 44.7       | 
| SAM-L                 |1491| 46.2/45.5*     | 43.5       | 
| SAM-B                 |487| 43.4/41.0*     | 40.8       | 
| FastSAM                 |344| 37.9     | 34.5       | 
| MobileSAM            | 42.0|41.0     | 37.0       | 
| **TinySAM**  [\[ckpt\]](https://github.com/xinghaochen/TinySAM/releases/download/3.0/tinysam_42.3.pth)       | 42.0|42.3     | 38.6       | 
| **Q-TinySAM**  [\[ckpt\]](https://github.com/xinghaochen/TinySAM/releases/download/2.0/tinysam_w8a8.pth)            | 20.3|41.4     | 37.2      | 

<sup>\* Results of single output ([`multimask_output=False`](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py#L98)). </sup></br>

First download the detection boxes ([`coco_instances_results_vitdet.json`](https://github.com/xinghaochen/TinySAM/releases/download/2.0/coco_instances_results_vitdet.json)) produced by ViTDet model, as well as the ground-truth instance segmentation labels([`instances_val2017.json`](https://github.com/xinghaochen/TinySAM/releases/download/2.0/instances_val2017.json)) and put them into `eval/json_files`. 
Related json files for LVIS dataset are available in [`lvis_instances_results_vitdet.json`](https://github.com/xinghaochen/TinySAM/releases/download/3.0/lvis_instances_results_vitdet.json) and [`lvis_v1_val.json`](https://github.com/xinghaochen/TinySAM/releases/download/3.0/lvis_v1_val.json).

Run the following code to perform evaluation for zero-shot instance segmentation on COCO dataset.
```
cd eval; sh eval_coco.sh
```


## Acknowledgements
We thank the following projects: [SAM](https://github.com/facebookresearch/segment-anything), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [TinyViT](https://github.com/microsoft/Cream).

## Citation
```bibtex
@article{tinysam,
  title={TinySAM: Pushing the Envelope for Efficient Segment Anything Model},
  author={Shu, Han and Li, Wenshuo and Tang, Yehui and Zhang, Yiman and Chen, Yihao and Li, Houqiang and Wang, Yunhe and Chen, Xinghao},
  journal={arXiv preprint arXiv:2312.13789},
  year={2023}
}
```

## License

This project is licensed under <a rel="license" href="License.txt"> Apache License 2.0</a>. Redistribution and use should follow this license.
