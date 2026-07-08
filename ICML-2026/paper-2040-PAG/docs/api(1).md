# API Guide

PAG exposes a narrow public facade and keeps the internal build and search pipeline inside the implementation.

## C++ API

The public header is:

```cpp
#include "pag_index.h"
```

Installed consumers include:

```cpp
#include <pag/pag_index.h>
```

The main type is:

```cpp
pag::Index
```

Stable operations:

- `build`: construct an index from row-major `float32` vectors
- `load`: load a saved static index
- `save`: persist the current index
- `search`: single-query approximate nearest neighbor search
- `search_batch`: batch search over row-major queries with OpenMP parallelism
- `add`: single-vector online insertion with an automatically assigned label
- `insert`: single-vector online insertion with a caller-provided label
- `add_batch`: batch online insertion with automatically assigned labels
- `insert_batch`: batch online insertion with caller-provided labels

## Build Options

Common fields:

```cpp
pag::BuildOptions build;
build.index_path = "./pag_index";
build.metric = pag::Metric::L2;
build.max_search_k = 100;
build.ef_construction = 200;
build.target_degree = 16;
build.projection_levels = 64;
```

`max_search_k` is the largest query `top_k` supported by the index. A search request with `top_k > max_search_k` throws an exception.
The benchmark scripts expose the same idea as `MAX_SEARCH_K`; `TOPK` is only the current query result size.

`target_degree` is the graph target degree `M`. The base-layer adjacency capacity is `2 * target_degree`, while selected new-node neighbors are capped by `target_degree`.

`projection_levels` must be a positive multiple of `8`. The per-level projection code width is fixed internally by the current 4-bit encoding format and is not a user-tunable parameter.

Supported metrics:

- `pag::Metric::L2`
- `pag::Metric::Cosine`
- `pag::Metric::MaximumInnerProduct`

For MIPS, set `max_search_k` to the largest `top_k` expected at query time.
MIPS builds use dataset order so that the same insertion semantics are available for online workloads.

## Search Options

```cpp
pag::SearchOptions search;
search.top_k = 10;
search.ef_search = 100;
```

`ef_search` controls search work. Higher values generally improve recall and reduce QPS.

Use `search()` for one query and `search_batch()` for a row-major query batch:

```cpp
auto one = index.search(query.data(), search);
auto many = index.search_batch(queries.data(), query_count, search);
```

Both static and online indexes support batch search. A batch call uses the same `SearchOptions` for every query and parallelizes the query loop with OpenMP.

## Online Insertion

Online mode starts from an initial graph. It is not an empty-index append-only builder.

Build the initial online graph with the same graph parameters that later insertions should use:

```cpp
pag::BuildOptions build;
build.index_path = "./pag_online";
build.mode = pag::IndexMode::Online;
build.max_elements = initial_count + reserve_count;
build.max_search_k = 100;
build.ef_construction = 200;
build.target_degree = 16;
build.projection_levels = 64;

pag::Index index;
index.build(base.data(), initial_count, dim, build);
```

`target_degree`, `ef_construction`, `projection_levels`, `metric`, `max_search_k`, and `max_elements` are fixed by this `build()` call. Later online insertions do not take a separate `target_degree`; they use the graph configuration already stored in the index.

The initial batch must contain at least the construction beam width:

```text
min_initial_count = min(max(ef_construction, 2 * target_degree), 100)
```

With the default API settings, this is `100` initial vectors. Use a representative initial batch for the target workload and keep `search.top_k <= build.max_search_k`.

Search and insert can then be interleaved:

```cpp
pag::SearchOptions search;
search.top_k = 10;
search.ef_search = 100;

auto before = index.search(query.data(), search);

pag::Label assigned = index.add(new_vector.data());
index.insert(other_vector.data(), 123456);

auto after = index.search(query.data(), search);
```

Use `add()` when PAG should assign the next sequential label. Use `insert()` when the caller provides the label; provide a fresh label to keep search results unambiguous.

For update blocks where no query needs to run between individual insertions, use the batch APIs:

```cpp
auto assigned_labels = index.add_batch(new_vectors.data(), new_count);
index.insert_batch(labeled_vectors.data(), labels.data(), labeled_count);
```

`add_batch()` and `insert_batch()` use OpenMP to parallelize the insertion loop. If the workload needs a search after every inserted vector, use the single-vector `add()` / `insert()` calls in the intended order.

Python uses the same model:

```python
build = pag.BuildOptions()
build.index_path = "./pag_online"
build.metric = pag.Metric.L2
build.mode = pag.IndexMode.Online
build.max_elements = initial_count + reserve_count
build.max_search_k = 100
build.ef_construction = 200
build.target_degree = 16
build.projection_levels = 64

index = pag.Index()
index.build(initial_vectors, build)

ids, distances = index.search(queries, top_k=10, ef_search=100)
assigned = index.add(new_vector)
index.insert(other_vector, 123456)
assigned_batch = index.add_batch(new_vectors)
index.insert_batch(labeled_vectors, labels)
```

## Persistence

In static mode, `build()` writes the packed static layout and reloads it internally, so the resulting object is immediately searchable.

In online mode, `build()` keeps a mutable in-memory graph so that `add()` and `insert()` can continue. Calling `save()` finalizes the mutable graph into the packed static layout and persists it; after that, the object is no longer in online-insertion mode.

Saved index directories contain:

```text
index.bin
info.bin
permutation.bin
```

Load an existing index:

```cpp
pag::LoadOptions load;
load.index_path = "./pag_index";
load.metric = pag::Metric::L2;

pag::Index index;
index.load(load);
```

## Python API

Install:

```bash
python -m pip install .
```

Use:

```python
import numpy as np
import pag

base = np.random.random((100000, 128)).astype("float32")
queries = np.random.random((1000, 128)).astype("float32")

build = pag.BuildOptions()
build.index_path = "./pag_index"
build.metric = pag.Metric.L2
build.max_search_k = 100
build.ef_construction = 200
build.target_degree = 16
build.projection_levels = 64

index = pag.Index()
index.build(base, build)
ids, distances = index.search(queries, top_k=10, ef_search=100)
```

Python exposes the same stable operations as C++: `build`, `load`, `save`, `search`, `add`, `insert`, `add_batch`, and `insert_batch`. Input vectors should be numpy `float32` arrays in row-major layout. `search()` accepts one query vector or a 2D query batch and returns `(ids, distances)` numpy arrays. A 2D `search()` call uses the C++ batch search path.

## Integrator Notes

- The per-level projection code width is fixed by the 4-bit encoding format.
- `max_search_k` controls the largest query `top_k` supported without rebuilding.
- Build, batch search, and batch insertion use the OpenMP runtime thread pool. Set `OMP_NUM_THREADS` to cap CPU threads; `schedule(dynamic)` controls work distribution, not the number of threads.
- Use the batch APIs for parallel work inside one `pag::Index` call. Do not mutate the same online index from multiple application threads at the same time.
- For dynamic workloads, benchmark representative update sequences and choose a rebuild cadence that matches the application recall target.
