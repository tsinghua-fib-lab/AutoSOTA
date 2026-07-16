This contains utilites for running scaling law fits on our results.

### Quick Start

```sh
git lfs install # .ipynb files are tracked with git lfs! (brew install git-lfs)
pip install -r requirements.txt

# Dry-run fit scaling laws on a subset of results
python fit_scaling_laws.py --dry-run

# Render result tables using our ladder fits
python render_tables.py --hf-path allenai/DataDecide-eval-results
```

**Folder structure**
- `remote/` - contains utilities for pulling evaluation results into a single prediction file
- `utils/` - contains dataloading, ladder usage and plotting code
- `notebooks/` - contains interactive notebooks for inspecting scaling law fits

### Fit scaling laws

**Setup**

```sh
# (Optional) If you want to edit the ladder fitting code, you can install locally!
git clone -b datados https://github.com/allenai/OLMo-ladder OLMo-ladder
cd OLMo-ladder
pip install -e ".[plotting]"
```

**How to run**

```sh
# Fit scaling laws with "--dry-run" to use a subset of results for tesing
python fit_scaling_laws.py --result-path ladder_predictions_dry_run.csv --dry-run

# Fit on all metrics!
python fit_scaling_laws.py --result-path ladder_predictions.csv

# Fit + push results to HuggingFace
python fit_scaling_laws.py --result-path ladder_predictions.csv --push-to-hf allenai/DataDecide-eval-results

# Render tables in our paper using OUR ladder fits
python render_tables.py --hf-path allenai/DataDecide-eval-results

# Render tables in our paper using YOUR ladder fits
python render_tables.py --result-path ladder_predictions.csv
```

### Notebooks

Scaling laws often have to be de-bugged by looking at the curve fits themselves. Here are some utilities for doing so:

- `notebooks/example_results.ipynb` -- Example of rendering training curves
- `notebooks/scaling_law_figures.ipynb` -- Render paper figures to inspect scaling law fits
- `notebooks/scaling_law_tables.ipynb` -- Interactive version of table rendering
- `notebooks/math_and_code.ipynb` -- Tables for results on MBPP, HumanEval, etc.