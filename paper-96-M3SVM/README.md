# private_multiclass_svm

Official Python Implementation of [**"Multi-Class Support Vector Machine with Differential Privacy"** (NeurIPS 2025)](https://openreview.net/forum?id=Jz4PENUsRo)

## How to run

### 1. Setup
* **Dataset:** Refer to the `dataset` folder.
* **Requirements:** Install the necessary packages using the following command:
  ```bash
  pip install -r requirements.txt
### 2. Run
- Refer to the **run** folder detailed implementation including hyperparameter setting with .ipynb file.

- Gradient Perturbation (Opacus)

  ```bash
	python main-opacus.py --data Cornell --optimizer adam --num_epoch 20 --lr_scale 0.1 --eps 2.0
- Weight Perturbation (Sklearn)

  ```bash
	python main-sklearn.py --data=Cornell

### 3. Citation
We utilize the official GitHub of M3SVM (AAAI24): https://github.com/zz-haooo/M3SVM

Use the BibTeX below for citation.
  ```bash 
@inproceedings{
park2025multiclass,
title={Multi-Class Support Vector Machine with Differential Privacy},
author={Jinseong Park and Yujin Choi and Jaewook Lee},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={[https://openreview.net/forum?id=Jz4PENUsRo](https://openreview.net/forum?id=Jz4PENUsRo)}
}
