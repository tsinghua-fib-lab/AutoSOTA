# Training Molecular Dynamics Potentials in JAX

[**Documentation**](https://chemtrain.readthedocs.io/en/latest/) | [**Getting Started**](#getting-started) | [**Installation**](#installation) | [**Contents**](#contents) | [**Contact**](#contact)

[![PyPI version](https://badge.fury.io/py/chemtrain.svg)](https://badge.fury.io/py/chemtrain)
[![Documentation Status](https://readthedocs.org/projects/chemtrain/badge/?version=latest)](https://chemtrain.readthedocs.io/en/latest/?badge=latest)
[![Test](https://github.com/tummfm/chemtrain/actions/workflows/test.yml/badge.svg)](https://github.com/tummfm/chemtrain/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-chemtrain-yellow)](https://doi.org/10.1016/j.cpc.2025.109512)
[![Paper](https://img.shields.io/badge/Paper-chemtrain--deploy-yellow)
](https://doi.org/10.1021/acs.jctc.5c00996)


Neural Networks are promising models for enhancing the accuracy of classical molecular
simulations. However, the training of accurate models is challenging.
**chemtrain** is a framework for learning sophisticated Neural Network potential
models by combining customizable training routines with advanced training
algorithms.
This combination enables the inclusion of high-quality reference data from
simulations and experiments and lowering the computational demand of training
through complementing algorithms with different advantages.

**chemtrain** is written in JAX, integrating with the differentiable MD engine
[JAX, M.D.](https://github.com/jax-md/jax-md)
Therefore, **chemtrain** leverages end-to-end differentiable
physics and hardware acceleration through GPUs to provide flexibility at scale.


## Getting Started

To get started with chemtrain and with the most important algorithms,
we provide simple toy examples.
These examples are simple to run on the CPU and sufficient to illustrate the basic
concepts of the algorithms:

- [Force Matching](./examples/force_matching.ipynb)
- [DiffTre](./examples/difftre.ipynb)
- [Prior Simulation](./examples/prior_simulation.ipynb)
- [Relative Entropy Maximization](./examples/relative_entropy.ipynb)

For a more extensive overview of implemented algorithms, please refer to the
documentation of the ``trainers`` module.

To see the usage of chemtrain in real examples, we implemented the training
procedures of some recent papers:

- [CG Alaninine Dipeptide in Implicit Water](./examples/CG_alanine_dipeptide.ipynb)
- [CG Water on Structural Data](./examples/CG_water_difftre.ipynb)
- [AT Titanium on Fused Simulation and Experimental Data](./examples/AT_titanium_fused_training.ipynb)

We recommend viewing the examples in the [reference documentation](https://chemtrain.readthedocs.io/en/latest/).

## Installation

**chemtrain** can be installed with pip:

```shell
pip install chemtrain --upgrade
```

The above command installs **JAX for CPU**.
Running **chemtrain on the GPU** requires the installation of a special JAX
version.
Please follow the
[JAX Installation Instructions](https://github.com/google/jax#installation).

> **Note:** Chemtrain installs ``jax == 0.4.30`` which is, in principle,
> incompatible with ``jax_md <= 0.1.29`` but resolves an
> [XLA issue](https://github.com/google/jax/issues/17730) which can prevent
> training. By importing ``chemtrain`` or the ``jax_md_mod`` module
> **before importing** ``jax_md``, the compatibility is restored by a simple
> patch.

### Additional Packages

Some parts of **chemtrain** require additional packages.
To install these, provide the `all` option.

```shell
pip install 'chemtrain[all]' --upgrade
```

### Installation from Source

The lines below install **chemtrain** from source for development purposes.

```shell
git clone git@github.com:tummfm/chemtrain.git
pip install -e '.[all,docs,test]'
```

This command additionally installs the requirements to run the tests

```shell
pytest tests
```

and to build the documentation (e.g., in HTML)

```shell
make -C docs html
```

## Contents

Within the repository, we provide the following directories:

``chemtrain/``
: Source code of the **chemtrain** package. The package consists of the
  following submodules:

  - ``data`` Loading and preprocessing of microscopic reference data
  - ``ensemble`` Sampling from and evaluating quantities for ensembles
  - ``learn`` Lower level implementations of training algorithms
  - ``quantity`` Learnable microscopic and macroscopic quantities
  - ``trainers`` High-level API to training algorithms

``docs/``
: Source code of the documentation.

``examples/``
: Example Jupyter Notebooks as provided in the documentation. Additionally,
  the ``examples/data/`` folder contains some example data for the toy examples.
  The other Jupyter Notebooks download data automatically from the sources
  provided in the original papers.

``jax_md_mod/``
: Source code of the JAX, M.D. modifications. In the long term, we aim to integrate these modifications into the main JAX, M.D. repository.

``tests/``
: Unit test for the **chemtrain** package, supplementing the testing trough
  a reproduction of published paper results.


## Citation

If you use chemtrain or chemtrain-deploy, please cite the following [paper](https://www.sciencedirect.com/science/article/pii/S0010465525000153):

```bibtex
@article{fuchs2025chemtrain,
    title = {chemtrain: Learning deep potential models via automatic differentiation and statistical physics},
    journal = {Computer Physics Communications},
    volume = {310},
    pages = {109512},
    year = {2025},
    issn = {0010-4655},
    doi = {https://doi.org/10.1016/j.cpc.2025.109512},
    url = {https://www.sciencedirect.com/science/article/pii/S0010465525000153},
    author = {Paul Fuchs and Stephan Thaler and Sebastien Röcken and Julija Zavadlav},
}
```

```bibtex
@article{fuchsChemtrainDeploy2025,
  title = {Chemtrain-{{Deploy}}: {{A Parallel}} and {{Scalable Framework}} for {{Machine Learning Potentials}} in {{Million-Atom MD Simulations}}},
  author = {Fuchs, Paul and Chen, Weilong and Thaler, Stephan and Zavadlav, Julija},
  year = {2025},
  month = jul,
  journal = {Journal of Chemical Theory and Computation},
  publisher = {American Chemical Society},
  issn = {1549-9618},
  doi = {10.1021/acs.jctc.5c00996}
}
```

## Contributing
Contributions are always welcome! Please open a pull request to discuss the code
additions.

## Contact
For questions or discussions, please open an Issue on GitHub.
