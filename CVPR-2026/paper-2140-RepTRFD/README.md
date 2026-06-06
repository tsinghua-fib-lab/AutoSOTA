# RepTRFD

Official implementation of **"Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery" (CVPR 2026)**.

---

## Dataset

The datasets used in our experiments are publicly available.

- **Test datasets** can be downloaded from:  
  https://drive.google.com/drive/folders/1rphapDHEcFwBZXH-nHEKFGMynZabOKPT?usp=sharing

- The **SHOT dataset** used in our paper is available at:  
  https://drive.google.com/drive/folders/1lboszDEitPdZaJivdm3LCr_22SGopqw6?usp=drive_link

After downloading, please place the files into the `data/` directory.

---

## Directory Structure

The project should be organized as follows:

```text
RepTRFD/
├── data/                           # Datasets directory
│   ├── airplane.tiff
│   ├── Toy.mat
│   ├── Washington_DC.mat
│   ├── news.mat
│   ├── 0809.png
│   └── mario011.ply
│
├── model.py                        # Core RepTRFD network architectures
├── utils.py                        # Utility functions
│
├── Demo_inpainting.py              # Image / Video Inpainting
├── Demo_denoising.py               # MSI / HSI Denoising
├── Demo_super_resolution.py        # Image Super-Resolution
└── Demo_point_cloud.py             # Point Cloud Recovery
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{xu2026reparameterized,
  title={Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery},
  author={Xu, Yangyang and Ke, Junbo and Wen, You-Wei and Wang, Chao},
  journal={arXiv preprint arXiv:2603.01034},
  year={2026}
}
```
