# TimePFN
This is an official implementation of [TimePFN: Effective Multivariate Time Series Forecasting with Synthetic Data (AAAI 2025)](https://arxiv.org/abs/2502.16294).

This repository contains the codebase of the TimePFN. We recommend using a conda virtual environment to load the dependencies listed in `requirements.txt`.

We provide the model checkpoint, testing, training, and fine-tuning scripts. Please check `pfn_scripts`. For the datasets, please refer to iTransformer's `datasets.zip` [gdrive link](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=sharing).

Download them and put them under the directory `./datasets`.

To generate synthetic datasets for the pretraining task, please refer to the directory `synthetic_data_generation`. Please read the comments and directives in the bash scripts.


## Citation

```
@inproceedings{taga2025timepfn,
  title={TimePFN: Effective multivariate time series forecasting with synthetic data},
  author={Taga, Ege Onur and Ildiz, Muhammed Emrullah and Oymak, Samet},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={19},
  pages={20761--20769},
  year={2025}
}
```

## Acknowledgement

We thank to the following repositories for their valuable code contributions, which helped immensely:

- iTransformer (https://github.com/thuml/iTransformer)
- PatchTST (https://github.com/yuqinie98/PatchTST)
- Reformer (https://github.com/lucidrains/reformer-pytorch)
- Informer (https://github.com/zhouhaoyi/Informer2020)
- Autoformer (https://github.com/thuml/Autoformer)
