# Repeat After Me: Transformers are Better than State Space Models at Copying

## About

This part of the repository was built from [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/abs/2402.01032), specifically the synthetic experiment part.

This repository gathers the experiments for the paper . The experiments divide in two parts: 

## Installation

<tt>pip install causal-conv1d>=1.1.0</tt> : an efficient implementation of a simple causal Conv1d layer used inside the Mamba block.
<tt>pip install mamba-ssm</tt> : the core Mamba package.
<tt>pip install names</tt> : names package to randomly sample names in the phone-book experiment.

Other requirements:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+
- transformers 4.35+
- datasets 2.14+

## We need to cite

```
@article{jelassi2024repeat,
  title={Repeat after me: Transformers are better than state space models at copying},
  author={Jelassi, Samy and Brandfonbrener, David and Kakade, Sham M and Malach, Eran},
  journal={arXiv preprint arXiv:2402.01032},
  year={2024}
}
```

