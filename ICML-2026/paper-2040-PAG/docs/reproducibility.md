# Reproducibility

This page describes how to rebuild PAG and reproduce local benchmark runs with the dataset scripts.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The default build creates:

```text
build/libpag_core.a
build/PAG
```

## Dataset Files

Each dataset directory should contain:

```text
base.fbin
query.fbin
gt1000.ibin
```

See [data_format.md](data_format.md) for the binary layout.

The prepared benchmark files can be downloaded from:

```text
https://huggingface.co/datasets/ckadzh8/pag-benchmark-data
```

```bash
python -m pip install "huggingface_hub[hf_xet]"
hf download ckadzh8/pag-benchmark-data --repo-type dataset \
  --include "data/*" --local-dir .
```

## Run Benchmarks

```bash
./scripts/run_glove.sh
./scripts/run_sift.sh
./scripts/run_music.sh
```

Each script builds the index if the configured index directory does not exist. If the index directory already exists, the script loads it and runs the search benchmark.

Use `TOPK` for the current query result size and `MAX_SEARCH_K` for the largest `TOPK` the built index should support. `TARGET_DEGREE` controls the graph target degree `M`; level-0 adjacency storage can hold up to `2M` neighbors. The dataset scripts default to `TOPK=100` and `MAX_SEARCH_K=1000`.

The command-line output has three columns:

```text
efs    Recall    QPS
```

## Interpreting Results

- Higher `efs` usually improves recall and lowers QPS.
- Compare runs at the same `top_k`, metric, dataset, and build parameters.
- For online workloads, evaluate the intended insertion order and query distribution instead of relying only on static-query benchmarks.
