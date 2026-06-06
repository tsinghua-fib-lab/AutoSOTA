# Full Reproducibility

While the README was a simple showcase of the package, this is an in-depth step-by-step guide to reproduce the results in the paper. If your aim is to reproduce, you should follow this document.

## Setting up the Environment

The environment use for reproducibility is a little bit different, as we would need to run baselines in R and also CATENet which uses JaX. To do so, please run the following command to create a conda environment:

```bash
conda env create -f env_reproduce.yaml
conda activate reproduce
```

We also have some R dependencies for baseline methods. Open up the `R` console (hit `R` in the terminal) and run the following commands:

```r
install.packages("remotes")
remotes::install_version("Matrix", version = "1.5-4")
install.packages("grf")
install.packages("bartCause")
```

## Extra Environment Variables

You will also need to set the following environment variables using `python-dotenv`:

```bash
# the directory where everything will be logged and saved with git ignoring it
dotenv set OUTPUT_DIR <absolute_path_to_your_output_dir>
```

## Additional Notebooks

Apart from the notebooks that were described in the [README](README.md), once you have the environment set up, you can run the following notebooks:

1. [`notebooks/causal_effect_full.ipynb`](notebooks/causal_effect_full.ipynb): Similar to the `causal_effect` notebook, but with *all* the baselines, including CATENet and the R baselines.
2. [`notebooks/qini.ipynb`](notebooks/qini.ipynb): Uplift modelling results and Qini curves.
