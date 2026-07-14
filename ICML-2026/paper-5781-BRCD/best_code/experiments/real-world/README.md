# README: Multi-Python setup for `main-ob.py` and `main-ss.py`

This repo intentionally uses **different Python versions** to run different RCA methods on OB, sockshop, and petshop experiments.
Both `main-ob.py` and `main-ss.py` select which methods to import based on your interpreter version:

- **Python 3.12**: imports the “py312” method set (many baseline / graph / ML methods)
- **Python 3.10**: imports a “py310” method set (e.g., `brcd`, `idint`, `score_ordering`, `smooth_traversal`, `cholesky`)
- **Python 3.8**: imports the “rcd/ht/e_diagnosis/mmrcd” method set (includes `pyrca`-based methods)

Because the dependency constraints differ, the most reliable workflow is to create **three isolated environments**.

---

## What runs in which Python?

When you run `python main-ob.py ...` or `python main-ss.py ...`, your Python version controls what’s available:

- **Python 3.12**
  - Methods imported by the scripts include: `baro`, `causalrca`, `circa`,  `easyrca`, `rcg`, `simplerca_helper`, `microdig`, `shapleyiq`, etc.
- **Python 3.10**
  - Methods imported by the scripts include: `score_ordering`, `smooth_traversal`, `cholesky`, `idint`, `brcd`
- **Python 3.8**
  - Methods imported by the scripts include: `rcd`, `mmrcd`, plus wrappers that rely on **PyRCA**:  `e_diagnosis`

If you try `--method` that isn’t imported in your current Python version, the script will error during argument parsing.

---

## Option A (recommended): Conda/Mamba environments (py38/py310/py312)

### 0) Install Conda (Miniconda/Mambaforge) and basic OS packages

On Ubuntu/Debian you’ll usually want these system packages for building wheels and graphviz:

```bash
sudo apt update -y
sudo apt install -y build-essential graphviz
```

If you plan to use the **py38** env (for `ht`/`e_diagnosis` via PyRCA), you’ll also want Java:

```bash
sudo apt install -y openjdk-11-jre-headless
```

### 1) Create the environments from YAML

From the repo root (`/local/scratch/a/lee4094/RCAEval`):

```bash
conda env create -f environment-py312.yml
conda env create -f environment-py310.yml
conda env create -f environment-py38.yml
```

Notes:
- `environment.yml` is provided as a convenience alias for the **py312** environment.
- If you use `mamba`, replace `conda` with `mamba` for faster solving.

### 2) Post-install step (py38 only)

The RCD-mode setup in this repo may require linking a customized PC implementation:

```bash
conda activate rcaeval-py38
bash script/link.sh
```

---

## Extra step for Python 3.8: `PyRCA` dependency for RCD 

The Python 3.8 methods `rcd` are implemented as wrappers around the **PyRCA** library ([`salesforce/PyRCA`](https://github.com/salesforce/PyRCA)).

When setting up the **py38** environment for running these methods, make sure `pyrca` is installed. You can either use the vendored `PyRCA` folder in this repo (as shown below) or clone the official repository:

```bash
git clone https://github.com/salesforce/PyRCA.git
cd PyRCA
pip install .
```

In the `venv` example above, we assume you placed the cloned `PyRCA` folder under `real-world/` and then run:

```bash
pip install -e ./PyRCA
```

For more details on PyRCA and its RCA methods, see the official README: [`https://github.com/salesforce/PyRCA`](https://github.com/salesforce/PyRCA).

---

## Extra step for Python 3.12: `tigramite` dependency

Some methods used in the Python 3.12 environment (for example those in `RCAEval/graph_construction/pcmci.py` and `RCAEval/e2e/microcause.py` / `easyrca.py`) depend on the [`tigramite`](https://github.com/jakobrunge/tigramite) package for causal discovery on time series.

When setting up the **py312** environment, make sure `tigramite` is available, either by:

- **Installing via pip/conda** (if available in your channel), or  
- **Cloning the official repository** and installing it locally:

```bash
git clone https://github.com/jakobrunge/tigramite.git
cd tigramite
python setup.py install
```

See the official Tigramite README and docs for details on versions and optional dependencies: [`https://github.com/jakobrunge/tigramite`](https://github.com/jakobrunge/tigramite).

---

## Option B: Pure `venv` (works, but you must manage Python installs yourself)

The repo’s `docs/SETUP.md` shows a `venv`-based setup for Python 3.12 (default) and 3.8 (rcd).
If you also need Python 3.10 (for the py310 methods), you’ll need to install it on your machine first.

Example (Python 3.10):

```bash
python3.10 -m venv env-py310
. env-py310/bin/activate
pip install -U pip wheel
pip install -e ".[default]"
```

Example (Python 3.8):

```bash
python3.8 -m venv env-py38
. env-py38/bin/activate
pip install -U pip wheel
pip install -e ".[rcd]"
pip install -e ./PyRCA
bash script/link.sh
```

Example (Python 3.12):

```bash
python3.12 -m venv env-py312
. env-py312/bin/activate
pip install -U pip wheel
pip install -e ".[default]"
```

---

## Running `main-ob.py` and `main-ss.py`

### General pattern

```bash
python main-ob.py --method <METHOD> --dataset <DATASET> [--test] [--length 20] [--tdelta 0]
python main-ss.py --method <METHOD> --dataset <DATASET> [--test] [--length 20] [--tdelta 0]
```

Both scripts will **auto-download datasets** on first run (requires internet access).

### Example commands

**Python 3.12 (py312 methods):**

```bash
conda activate rcaeval-py312
python main-ob.py --method baro --dataset online-boutique --test
```

**Python 3.10 (py310 methods):**

```bash
conda activate rcaeval-py310
python main-ss.py --method brcd --dataset sock-shop-2 --test
```

**Python 3.8 (RCD/PyRCA-based methods):**

```bash
conda activate rcaeval-py38
python main-ob.py --method rcd --dataset online-boutique --test
python main-ob.py --method ht --dataset online-boutique --test
python main-ob.py --method e_diagnosis --dataset online-boutique --test
```

---

## Troubleshooting

- **`pyagrum` install issues**
  - Prefer installing from `conda-forge` (the provided `environment*.yml` includes `conda-forge`).
- **Method not found / “not defined”**
  - Your `--method` must match what gets imported for your Python version. Switch envs (py38/py310/py312).
- **Slow first run**
  - First run downloads datasets and may take time; `--test` runs only a small subset.


