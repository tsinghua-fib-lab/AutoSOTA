# Orbit MMD evaluation

Computes orbit MMD distance (4-node graphlets, 15 orbits) between
reference and generated graphs using ORCA + Gaussian-TV kernel.

## Setup

Build ORCA (a C++ utility) once:

```bash
cd syngler/evaluation/orca   # paper authors will need to vendor or link this
g++ -O2 -std=c++11 -o orca orca.cpp
```

(If you don't have the ORCA source, fetch it from
<https://file.biolab.si/biolab/supp/orca/orca.html> and place `orca`
binary at `syngler/evaluation/orca/orca`.)

## Run

```bash
python -m syngler.evaluation.orbit \
    --reference data/real/dblp/generator/seed=0.npy \
    --samples_dir runs/syngler/dblp/syngr/samples \
    --output runs/eval_paper/syngler/dblp/orbit_mmd.json
```

The script reports `orbit_mmd ± std` over the supplied samples vs the
reference.
