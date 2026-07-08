# Data Format

PAG benchmark scripts use simple binary matrix files. All integer headers are little-endian `uint32_t`.

## Float Matrix: `.fbin`

`base.fbin` and `query.fbin` store row-major `float32` vectors:

```text
uint32_t rows
uint32_t dim
float data[rows][dim]
```

Example:

```text
base.fbin
  rows = number of database vectors
  dim  = vector dimension
  data = rows * dim float32 values
```

## Ground Truth Matrix: `.ibin`

`gt1000.ibin` stores row-major `uint32_t` vector IDs:

```text
uint32_t rows
uint32_t k
uint32_t ids[rows][k]
```

Each row contains the exact nearest-neighbor IDs for one query. Distances are not stored. The benchmark code computes recall by comparing returned labels with the first `top_k` IDs from each row.

When using the command-line benchmark directly, the row and dimension arguments must match the file headers. For subset runs, create separate subset files with matching headers.

## Expected Dataset Layout

The provided scripts expect this directory shape:

```text
data/
  glove/
    base.fbin
    query.fbin
    gt1000.ibin
  sift/
    base.fbin
    query.fbin
    gt1000.ibin
  music/
    base.fbin
    query.fbin
    gt1000.ibin
```

Large datasets should stay outside Git history. The repository should contain scripts and documentation, not multi-gigabyte benchmark files.

## Download Benchmark Data

The prepared GloVe, SIFT, and Music benchmark files are hosted at:

```text
https://huggingface.co/datasets/ckadzh8/pag-benchmark-data
```

Download the expected `data/` tree into the repository root with:

```bash
python -m pip install "huggingface_hub[hf_xet]"
hf download ckadzh8/pag-benchmark-data --repo-type dataset \
  --include "data/*" --local-dir .
```

## Metric Conventions

- L2: lower distance is better.
- Cosine: vectors are normalized internally; higher cosine similarity is better.
- MIPS: higher inner product is better.

Ground truth must be generated with the same metric used by the benchmark run.
