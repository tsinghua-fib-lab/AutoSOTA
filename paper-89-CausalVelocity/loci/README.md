[![arXiv](https://img.shields.io/badge/arXiv-2306.11078-b31b1b.svg)](https://arxiv.org/abs/2210.09054)
[![Venue](https://img.shields.io/badge/venue-ICML_2023-darkblue)](https://proceedings.mlr.press/v202/immer23a.html)

# Code for Location-Scale Causal Inference (LOCI) estimator for bivariate causal model identification

## Setup

Requires `python>=3.8`. Install dependencies with
```
pip install -r requirements.txt
```

## Example Usage
To use the LOCI estimator on some pair x, y in the default setting (with neural network and subsequent independence test) use it like
```python
from causa.loci import loci
score = loci(x, y)
# The score is such that score > 0 indicates x -> y and score < 0 indicates y -> x.
print(score)
```

To use LOCI with a particular dataset provided in this repository and also plot the estimator fit and scaled residuals that are used for the independence test, the following snippet might be helpful:
```python
from sklearn.preprocessing import StandardScaler
from causa.loci import loci
from causa.datasets import MNU
from causa.utils import plot_pair

# We use the MNU pair 55 as an example and standardize the data
dataset = MNU(55, preprocessor=StandardScaler(), double=True)
x, y = dataset.cause.flatten().numpy(), dataset.effect.flatten().numpy()

# This function call shows all options of LOCI, we return_function to visualize the estimator
score, f_forward, f_reverse = loci(
    x, y, independence_test=True, neural_network=True, 
    return_function=True, n_steps=5000
)
plot_pair(x, y, f_forward, f_reverse)
```
![MNU_False](https://user-images.githubusercontent.com/7715036/221351915-fedaa072-7dff-4805-8b87-c50a862f935c.png#gh-light-mode-only)
![MNU_True](https://user-images.githubusercontent.com/7715036/221351917-ea3fa7f2-971d-4317-822b-60cce218292a.png#gh-dark-mode-only)

## Reproduce All Results 
First, create all necessary commands to be executed with
```
python commands/jobs_generate.py > commands/jobs.sh
python commands/jobs_estimator_generate.py > commands/jobs_estimator.sh
```
Create the result directory `mkdir results`.
All commands written to `commands/jobs.sh` and `commands/jobs_estimator.sh` need to be executed and can be run entirely in parallel.
The results will be written to `results/`.

## Run Individual Experiments
Instead of re-running the entire set of jobs, one can run individual causal pair benchmark pairs as follows
```
python run_individual.py --pair_id PAIR_ID --config configs/ANs_standardized.gin
```
The results will be printed after the run.

To run an individual estimator benchmark case, run
```
python run_estimator_benchmark.py --seed SEED --knot_heuristic {sqrt|lin}
```

## Produce Figures and Tables
First, create a directory to store the figure PDFs with `mkdir paper_figures`.
Then run 
```
python generate_figures_and_tables.py
```
The resulting figures will be saved to the created directory.
The overall latex result tables are printed to the command line output.

## Repository structure
```
├── causa  
│   ├── datasets.py  # bivariate causal data sets loaded from /data
│   ├── het_ridge.py  # implementation of our convex LSNM estimator
│   ├── hsic.py  # independence test
│   ├── iterative_fgls.py  # baseline LSNM estimator
│   ├── ml.py  # neural network estimators
│   └── utils.py  # neural network utilities
├── commands
│   ├── jobs_estimator_generate.py  # generate bash commands for estimator benchmark
│   └── jobs_generate.py  # generate bash commands for causal pairs
├── configs  # configs for causal pairs
├── baseline_results  # results for baseline methods (GRCI, QCCD, etc.)
├── data  # data sets
├── requirements.txt  # pip requirements
├── run_estimator_benchmark.py  # estimator benchmark script
├── run_individual.py  # causal pair benchmark script
└── generate_figures_and_tables.py  # create PDF figures and print tex tables
```
