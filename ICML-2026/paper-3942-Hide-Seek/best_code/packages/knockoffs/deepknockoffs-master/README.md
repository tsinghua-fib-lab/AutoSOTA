Deep Knockoffs
==============

This repository provides a Python package for sampling approximate
model-X knockoffs using deep generative models.

Accompanying paper: https://arxiv.org/abs/1811.06687.
Published in the Journal of the American Statistical Association (https://doi.org/10.1080/01621459.2019.1660174)

To learn more about the algorithm implemented in this package, visit  https://web.stanford.edu/group/candes/deep-knockoffs/ and read the accompanying paper.

To learn more about the broader framework of knockoffs, visit https://web.stanford.edu/group/candes/knockoffs/.

## Software dependencies

The code contained in this repository was tested on the following configuration of Python:

- python=3.6.5
- numpy=1.14.0
- scipy=1.0.0
- pytorch=0.4.1
- cvxpy=1.0.10
- cvxopt=1.2.0
- pandas=0.23.4

## Installation guide

```bash
cd DeepKnockoffs
python setup.py install --user
```

## Examples

 - [examples/toy-example.ipynb](examples/toy-example.ipynb) A usage example on a toy problem with multivariate Gaussian variables is available in the form of a Jupyter Notebook.
 - [examples/experiments-1.ipynb](examples/experiments-1.ipynb) Code to train the machine used in the paper.
 - [examples/experiments-2.ipynb](examples/experiments-2.ipynb) Code to compute the goodness-of-fit diagnostics for the machine used in the paper.
 - [examples/experiments-3.ipynb](examples/experiments-3.ipynb) Code to perform the controlled variable selection experiments in the paper.
 - [examples/data-preprocessing.ipynb](examples/data-preprocessing.ipynb) Example of how to pre-process data containing extremely correlated variables.

## License

This software is distributed under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html) and it comes with ABSOLUTELY NO WARRANTY.
