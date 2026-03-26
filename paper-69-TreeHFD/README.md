# **`treehfd`**

<div align="center">
    <h3>Explain XGBoost models with the Hoeffding functional decomposition.</h3>
</div>
<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg">
  <img src="https://img.shields.io/badge/python-3.11--3.14-blue">
  <img src="https://img.shields.io/pypi/v/treehfd">
  <img src="https://github.com/ThalesGroup/treehfd/actions/workflows/ci.yml/badge.svg?branch=main">
  <img src="https://codecov.io/github/thalesgroup/treehfd/graph/badge.svg?token=KAV27C99T4"/>
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json">
  <img src="https://microsoft.github.io/pyright/img/pyright_badge.svg">
  <img src="https://img.shields.io/maintenance/yes/2099?style=flat-square">
</p>  

Clément Bénard (Thales cortAIx-Labs)


----


## Overview 🌐

<div align="justify">

**`treehfd`** is a Python module to compute the Hoeffding functional decomposition
of XGBoost models (Chen and Guestrin, 2016) with dependent input variables, using
the TreeHFD algorithm. 
This decomposition breaks down XGBoost models into a sum of functions of one or 
two variables (respectively main effects and interactions), which are intrinsically
explainable, while preserving the accuracy of the initial black-box model.

The TreeHFD algorithm is introduced in the following article: 
[**Bénard, C. (2025). Tree Ensemble Explainability through the Hoeffding Functional Decomposition and TreeHFD Algorithm.
In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), in press.**](https://openreview.net/pdf?id=dRLWcpBQxS)

The documentation is available at [**Read the Docs**](https://treehfd.readthedocs.io/en/latest/).

**`treehfd`** currently supports XGBoost models for regression and binary classification with numeric input variables,
and categorical variables should be one-hot encoded.
**`treehfd`** essentially relies on `xgboost`, `numpy`, `scipy`, and `scikit-learn`.

</div>


## Installation 🛠️
To install **`treehfd`**, run: `pip install treehfd`

<div align="justify">

To install from source, first clone this repository, create
and activate a Python virtual environnement using **Python version >= 3.11**
(`conda` or `uv` can also be used), and run pip at the root directory of the repository:

</div>

```console
python -m venv .treehfd-env
source .treehfd-env/bin/activate
pip install .
```


## Example 🔎

<div align="justify">

The output of **`treehfd`** is illustrated with a standard dataset, the California Housing dataset,
where housing prices are predicted from various features, such as the house locations, and
characteristics of the house blocks. The code is provided below and in [examples](examples).

The following figure shows the two main decomposition components for the `longitude` and `Latitude`
variables of a black-box `xgboost` model, fitted with default parameters. 
The **`treehfd`** decomposition is displayed in blue, and compared to the decomposition
induced by `TreeSHAP` with interactions in red, which is directly implemented in `xgboost` package.
In particular, **`treehfd`** clearly identifies the peak of housing prices in the `Longitude` main effect,
corresponding to the San Francisco Bay Area (-122°25'), whereas `TreeSHAP` does not really detect it,
because main effects are entangled with interactions, while they are orthogonal in the Hoeffding decomposition.
 **`treehfd`** also highlights that house prices are lower in northern and eastern California.

</div>

<div align="center">
<img src="examples/fig_housing_main_effects_shap.png" alt="drawing" width="800"/>
<h4>Main treehfd components for the California Housing dataset.</h4>
</div>


```python
# Load packages.
import xgboost as xgb
from sklearn.datasets import fetch_california_housing

from treehfd import XGBTreeHFD

if __name__ == "__main__":

    # Fetch California Housing data.
    california_housing = fetch_california_housing()
    X = california_housing.data
    y = california_housing.target

    # Fit XGBoost model.
    xgb_model = xgb.XGBRegressor(eta=0.3, n_estimators=100, max_depth=5)
    xgb_model = xgb_model.fit(X, y)

    # Fit TreeHFD.
    treehfd_model = XGBTreeHFD(xgb_model)
    treehfd_model.fit(X)

    # Compute TreeHFD predictions.
    y_main, y_order2 = treehfd_model.predict(X)
```


## Scientific Background

<div align="justify">

The Hoeffding decomposition was introduced by Hoeffding (1948)
for independent input variables. More recently, Stone (1994) and Hooker (2007)
extended the decomposition to the case of dependent input variables, where uniqueness
of the decomposition is enforced by hierarchical orthogonality constraints. The estimation
from a data sample in the dependent case is a notoriously difficult problem, and 
therefore, the Hoeffding decomposition has long remain an abstract theoretical tool.
TreeHFD computes the Hoeffding decomposition of tree ensembles, based on a 
discretization of the hierarchical orthogonality constraints over the tree partitions.
Such decomposition is proved to be piecewise constant over these partitions,
and the values in each cell for all components are given by solving a least square problem for each tree.

</div>


## Documentation 📖

<div align="justify">

The documentation is available at [Read the Docs](https://treehfd.readthedocs.io/en/latest/),
and is generated with `sphinx` package. To build the documentation from source,
first install `sphinx` and the relevant extensions.
Then, go to the `docs` folder and build the documentation.

</div>

```console
pip install sphinx sphinx-rtd-theme
cd docs
sphinx-build -M html ./source ./build
```
Finally, open the html file `build/html/index.html` with a web browser to display the
documentation.


## Tests ✅

To run the tests, install and execute `pytest` with:
```console
pip install pytest
pytest
```


## Contributions ⛏️

Contributions are of course very welcome!
If you are interested in contributing to ``treehfd``, start by reading the [Contributing guide](CONTRIBUTING.md).


## License ⚖️

This package is distributed under the Apache 2.0 license. All dependencies have their own license. 
In particular, `xgboost` relies on NVIDIA proprietary modules for the optional use of GPU.


## References 📜

Please use the following citation to refer to ``treehfd``:
```r
@inproceedings{
  benard2025tree,
  title={Tree Ensemble Explainability through the Hoeffding Functional Decomposition and Tree{HFD} Algorithm},
  author={Clement Benard},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=dRLWcpBQxS}
}
```
<br>

<div align="justify">

Hoeffding, W. (1948). A Class of Statistics with Asymptotically Normal Distribution.
The Annals of Mathematical Statistics, 19:293 – 325.

Stone, C.J. (1994). The use of polynomial splines and their tensor products in
multivariate function estimation. The Annals of Statistics, 22:118–171.

Hooker, G. (2007). Generalized functional anova diagnostics for high-dimensional functions
of dependent variables. Journal of Computational and Graphical Statistics, 16:709–732.

Chastaing, G., Gamboa, F., and Prieur, C. (2012). Generalized Hoeffding-Sobol decomposition
for dependent variables - application to sensitivity analysis. Electronic Journal of
Statistics, 6:2420 – 2448.

Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings
of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,
pages 785–794, New York. ACM.

Lengerich, B., Tan, S., Chang, C.-H., Hooker, G., and Caruana, R. (2020). Purifying interaction
effects with the functional anova: An efficient algorithm for recovering identifiable additive
models. In International Conference on Artificial Intelligence and Statistics, pages 2402–2412. PMLR.

Bénard, C. (2025). Tree Ensemble Explainability through the Hoeffding Functional Decomposition and
TreeHFD Algorithm. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), in press. 

</div>
