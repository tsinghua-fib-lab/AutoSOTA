# Baselines

The paper compares SyNGLER against several graph-generation baselines.
We do **not** vendor the baseline source in this release; instead, each
`baselines/<name>/` directory contains a thin runner + YAML configs that
wrap the upstream implementation.

## Convention

Each subdirectory follows the same layout:

```
baselines/<name>/
├── README.md         # Upstream URL + install steps
├── configs/          # YAML configs per setting (sparse-sim, dblp, yelp, ...)
└── run.py            # Wrapper: reads config, calls upstream
```

To use a baseline:

1. Read `baselines/<name>/README.md` and `git clone` the upstream
   implementation into `baselines/<name>/src/` (the runner expects this
   path; symlinks are fine).
2. Install the baseline's own dependencies (usually a `requirements.txt`
   in the upstream repo).
3. Generate the dataset of interest with `python data/prepare_real_data.py`
   or `python data/generate_sparse_sim.py` — see `data/README.md`.
4. Launch:
   ```
   python baselines/<name>/run.py --config baselines/<name>/configs/<setting>.yaml \
       --output runs/<name>/<setting>/seed=<S>
   ```

## Baselines included

| Name        | Bundled? | Paper section            | Upstream (if needed)                                       |
|-------------|----------|--------------------------|------------------------------------------------------------|
| GRAN        | yes      | Sparse sim (Appendix)    | https://github.com/lrjconan/GRAN (original; we ship SyNGLER fork) |
| EDGE        | yes      | Sparse sim (Appendix)    | https://github.com/tufts-ml/graph-generation-EDGE (original; we ship SyNGLER fork) |
| VGAE        | yes      | Sparse sim (main + appx) | (paper's own implementation)                               |
| CELL        | yes      | Real data                | https://github.com/hheidrich/CELL (original; we ship a snapshot) |
| HiGen       | yes      | Real data                | https://github.com/Karami-m/HiGen_main (original; we ship SyNGLER fork) |
| LGD         | no       | Real data                | https://github.com/zhouc20/LatentGraphDiffusion (OOMs on n≳1000) |
| GraphMaker  | no       | Appendix F note          | https://github.com/Graph-COM/GraphMaker                    |

### Rep counts

GRAN / EDGE on sparse simulation use **20 Monte Carlo replications** (each
replication retrains the model). All other settings inherit the rep count
from the paper's main tables — see each `<name>/README.md` for specifics.
