# Benchmark Summary

This page summarizes local validation runs for PAG on three metrics:

- GloVe: L2
- SIFT: cosine
- Music: maximum inner product search

The benchmarks use `base.fbin`, `query.fbin`, and `gt1000.ibin` files under each dataset directory. Recall is computed against the top-`k` ground-truth IDs.

## Dataset Shapes

| dataset | metric | base vectors | queries | dimension | ground truth |
|---|---|---:|---:|---:|---:|
| glove | L2 | 1,193,514 | 1,000 | 200 | top 1000 |
| sift | cosine | 10,000,000 | 1,000 | 128 | top 1000 |
| music | MIPS | 1,000,000 | 10,000 | 100 | top 1000 |

## Static Search

Static indexes were built once per dataset and searched at multiple `ef_search` values. The table below reports the first and tenth points of each sweep.

| dataset | topk | first ef recall / QPS | tenth ef recall / QPS |
|---|---:|---:|---:|
| glove | 100 | ef=100, R=0.83029, QPS=11786.9 | ef=1000, R=0.96935, QPS=1518.9 |
| glove | 1000 | ef=1000, R=0.958095, QPS=966.5 | ef=10000, R=0.994477, QPS=100.0 |
| sift | 100 | ef=100, R=0.86440, QPS=6510.1 | ef=1000, R=0.99666, QPS=1250.9 |
| sift | 1000 | ef=1000, R=0.961292, QPS=644.7 | ef=10000, R=0.999688, QPS=104.5 |
| music | 100 | ef=100, R=0.949302, QPS=15117.6 | ef=1000, R=0.999012, QPS=2056.2 |
| music | 1000 | ef=1000, R=0.983465, QPS=1042.1 | ef=10000, R=0.999598, QPS=104.6 |

## Static and Online Comparison

The online check builds the first `N - 10000` vectors, inserts the final `10000` vectors one by one, and searches against the full `gt1000.ibin` ground truth.

| dataset | topk | ef | static recall / QPS | online recall / QPS |
|---|---:|---:|---:|---:|
| glove | 1000 | 10000 | 0.994477 / 100.0 | 0.994445 / 97.1 |
| sift | 1000 | 10000 | 0.999688 / 104.5 | 0.999719 / 101.7 |
| music | 1000 | 10000 | 0.999598 / 104.6 | 0.999520 / 104.1 |

## Build and Insertion Time

| dataset | static build | online initial build | online insert total | online insert average |
|---|---:|---:|---:|---:|
| glove | 95.17 s | 91.77 s | 21.60 s | 2159.9 us/vector |
| sift | 124.03 s | 114.64 s | 3.61 s | 361.4 us/vector |
| music | 106.68 s | 102.06 s | 11.04 s | 1104.3 us/vector |

## Notes

- `ef_search` controls the search work per query; higher values usually improve recall and reduce QPS.
- The static packed index is the fastest and most reproducible deployment path when updates can be batched.
- Dynamic workloads should be validated with representative insertion order, query distribution, and recall targets.
