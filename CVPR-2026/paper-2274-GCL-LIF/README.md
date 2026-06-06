# Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model
Code implementation for [Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model](https://arxiv.org/abs/2410.07547) (*CVPR 2026*).

## 👨‍💻 Quick Usage
```
python main.py --dataset CIFAR100 --datadir /home/to/dataset --use_ter --mixup --net_arch resnet18 --dev 0 --opt_backprop --use_parallel --mode compression

python main_inf.py --dataset CIFAR100 --datadir /home/to/dataset --use_ter --net_arch resnet18 --dev 0 --batchsize 50 --checkpoint_path /home/to/checkpoint --use_parallel --mode compression
```

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝:

```bibtex
@inproceedings{hao2026hdlif,
  title={Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model},
  author={Hao, Zecheng and Huang, Yifan and Xu, Zijie and Liu, Wenxuan and Tang, Yuanhong and Yu, Zhaofei and Huang, Tiejun},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2026}
}
```