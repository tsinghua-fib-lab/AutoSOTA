# Supplementary Experiments on Original Tasks

New baselines and ablations replicated on the 3 original paper datasets.

---

## Baseline Comparison (HSD vs Recent Methods)

Relative L2 error (lower is better):

|                      |  HSD  | GNOT'23 | ONO'23 | HAMLET'24 |
|----------------------|-------|---------|--------|-----------|
| externalAerodynamics | **0.038** | 0.089 | 0.081 | 0.123 |
| magnetostatics       | **0.021** | 0.050 | 0.051 | 0.071 |
| toroidalTransport    | **0.190** | 0.288 | 0.420 | 0.257 |

HSD outperforms all recent baselines on every task:
- **externalAerodynamics**: 53–69% improvement
- **magnetostatics**: 58–70% improvement
- **toroidalTransport**: 26–55% improvement

---

## Ablation: Pseudo-Spectral Bilinear Layer (gMLP)

| Task | HSD (Spectral Bilinear) | HSD (Plain MLP) | Δ |
|------|------------------------|-----------------|---|
| externalAerodynamics | 0.038 | 0.038 | <1% |
| magnetostatics       | 0.021 | 0.021 | <1% |
| toroidalTransport    | 0.190 | 0.190 | <1% |

The pseudo-spectral bilinear layer shows marginal effect on these tasks, where the de Rham cross-term coupling is already well-captured by the spectral basis.

---

## Data Generation

Datasets are generated via self-contained scripts (no external dependencies beyond standard packages):

```bash
cd original_tasks/
python generate_externalAerodynamics.py   # → data/externalAerodynamics/
python generate_magnetostatics.py         # → data/magnetostatics/
python generate_toroidalTransport.py      # → data/toroidalTransport/
```

magnetostatics uses a volume mesh (19010 tetrahedra). Spectral eigendecomposition is cached after first run to `magnetostatics_cache/spectral_cache.pkl` (50s, 41MB).

---

## Running Experiments

```bash
python original_tasks/run_experiments.py                              # all tasks
python original_tasks/run_experiments.py --task externalAerodynamics  # single task
python original_tasks/run_experiments.py --task magnetostatics
python original_tasks/run_experiments.py --task toroidalTransport
```

Results saved to `original_tasks/results.json`.
