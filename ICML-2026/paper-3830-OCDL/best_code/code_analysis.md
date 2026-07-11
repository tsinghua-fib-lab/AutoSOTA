# Code Analysis — DVM-AD (Paper 3830)

## Evaluation path
- **Script**: `run_benchmark.py`
- **Entry point**: `python3 run_benchmark.py`
- **Data loading**: UCI Cardiotocography via `ucimlrepo.fetch_ucirepo(id=193)`
- **Split**: 70/30 stratified, 5 seeds (0-4)
- **Training**: One-class (normal-only from train split)
- **Metric**: AUROC via `sklearn.metrics.roc_auc_score(y_te, scores)`
- **Output format**: `DVM-AD Cardiotocography AUROC: XX.XX +/- Y.YY%`

## Train/inference path
- **Model**: `dvmad.DVMAD` (PyOD wrapper) -> `dvmad.core.DVMADCore` (NumPy impl)
- **Fit**: `DVMADCore.fit(X_train)` — augments with artificial reference point, computes discriminants via generalized eigen-decomposition
- **Predict**: `DVMADCore.predict(X_test)` — projects test points, computes NN distance in projected space

## Config path
- **run_benchmark.py** lines 38-41: `EPSILON_SEL`, `EPSILON_TOL`, `MODE`, `ARTIFICIAL_MODE`
- **dvmad/core.py** `DVMADCore.__init__`: eps, mode, artificial_mode
- **dvmad/dvmad.py** `DVMAD.__init__`: contamination, mode, eps, artificial_mode

## Metric parser
- Parse from stdout: `DVM-AD Cardiotocography AUROC: XX.XX +/- Y.YY%`
- Primary metric is the mean value

## Reusable resources
- No pre-downloaded paper data mounts
- Data fetched from UCI via ucimlrepo at runtime
- Cache paths: /autosota_cache, /datasets, /models (all mounted, writable)

## Risky files
- `dvmad/core.py`: Core algorithm — changes here affect all eval paths
- `run_benchmark.py`: Evaluation script — do NOT change split logic, metric computation, or data loading

## Safe modification targets
- `dvmad/core.py`:
  - `_construct_reference_point()` (lines 82-90): Reference point construction
  - `_compute_discriminants_unchecked()` lines 116-131: S_S computation, eigendecomposition, pseudo-inverse
  - `compute_discriminants()` eigenvalue selection mask (lines 138-148)
- `run_benchmark.py`:
  - EPSILON_SEL, MODE, ARTIFICIAL_MODE parameters
  - Preprocessing steps (before DVMAD fit, after data loading)
  - Post-processing (ensemble scoring)

## Red-line constraints
- Do NOT change: data loading, split logic, metric computation, label mapping, seed set
- All optimization changes must be unsupervised (no test-label leakage)
- FAISS 1.7.4 must remain compatible (numpy 1.26.0 constraint)
