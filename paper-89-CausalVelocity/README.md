# causal-velocity

This is the official repository of the paper [Distinguishing Cause from Effect with Causal Velocity Models](https://arxiv.org/abs/2502.05122). The script `experiment.py` can be used to reproduce the experimental results. `score_probe.ipynb` can be used to reproduce Figures 3 and 4. 

Users are invited to read the `demo.ipynb` notebook to get a quick idea of how velocity models can be used for cause-effect inference. It also gives an idea of how to understand and interpret the velocity parametrization of SCMs. 

The main codebase has few dependencies beyond base JAX and [optax](https://github.com/google-deepmind/optax) for optimization. To actually evaluate the causal curves, we need [diffrax](https://docs.kidger.site/diffrax/) for numerical integration. The subdirectory `/loci` is entirely cloned from the original authors of [LOCI](https://github.com/aleximmer/loci), and as such has its own dependencies. It is only used to evaluate their method on our newly generated datasets. 

```bash
pip install -r requirements.txt
```
