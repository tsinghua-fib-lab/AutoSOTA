# Tutorials

Five interactive notebooks walk through OTP-FM end-to-end on synthetic and
real-world data. Each notebook is also the source for one of the experiments in
the paper.

```{toctree}
:maxdepth: 1

01_quickstart_gaussians
02_singlecell_eb
03_gulf_of_mexico
04_beijing_airquality
05_exact_gaussian_solutions
```

| Notebook | What it covers |
| --- | --- |
| [01 - Quickstart on Gaussians](01_quickstart_gaussians.ipynb) | End-to-end training + sampling on synthetic 2D Gaussian marginals. |
| [02 - Single-cell (Embryoid Body)](02_singlecell_eb.ipynb) | Trajectory inference from scRNA-seq data. |
| [03 - Gulf of Mexico](03_gulf_of_mexico.ipynb) | Modeling ocean currents in the Gulf of Mexico. |
| [04 - Beijing air quality](04_beijing_airquality.ipynb) | Multimarginal forecasting of pollutant concentrations. |
| [05 - Exact Gaussian solutions](05_exact_gaussian_solutions.ipynb) | Closed-form dynamic OT with potentials for Gaussian marginals. |

```{note}
Notebooks can be run locally with ``pixi run jupyter lab``.
```
