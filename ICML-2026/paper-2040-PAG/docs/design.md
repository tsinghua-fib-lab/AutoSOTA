# Design Overview

PAG is a graph-based approximate nearest neighbor index. It combines graph traversal with projection-derived edge metadata so that search can reject unlikely neighbors before running full vector scoring.

## Public Model

The public API exposes one main type:

```cpp
pag::Index
```

The stable operations are:

- `build`: construct an index from a row-major vector matrix
- `load`: load a previously saved static index
- `save`: persist the current index
- `search`: return approximate nearest neighbors
- `add`: append one vector with an automatically assigned label
- `insert`: append one vector with a caller-provided label

The implementation keeps graph construction, projection storage, pruning, and packed search layout behind this facade.

## Metrics

PAG supports:

- L2 distance
- cosine similarity
- maximum inner product search

Cosine vectors are normalized internally. MIPS queries use metric-specific scoring so that inner-product ranking is preserved while sharing the same graph-search interface.

## Build Pipeline

The static build path has five main stages.

1. **Vector preparation**

   Input vectors are padded for SIMD-friendly access. Cosine vectors are normalized. The build also computes norm and energy statistics used by graph construction and projection layout.

2. **Projection setup**

   PAG divides the padded vector space into balanced projection subspaces and creates projection directions for each subspace. These directions are used to encode edge metadata and evaluate pruning bounds during search.

3. **Graph construction**

   Initial seed points are selected, then remaining vectors are inserted into the graph. Each new vector searches the existing graph for candidate neighbors and connects through a heuristic neighbor selection step.
   The public `target_degree` parameter is the target degree `M`; level-0 adjacency storage allows up to `2M` neighbors to accommodate reverse links.

4. **Edge projection storage**

   After graph construction, PAG computes and stores compact projection records for graph edges. These records support projection-based rejection during traversal.

5. **Packed search layout**

   Static indexes are packed into a cache-friendly layout for search. The packed layout stores graph edges, vector records, labels, and projection metadata in a form optimized for the AVX-512 kernels.

## Search Pipeline

Search starts from a fixed set of entry points and maintains a bounded working set of candidates.

- **TFB working set** keeps the active frontier compact.
- **PRT filtering** uses projection records to reject neighbors whose predicted score cannot improve the current working set.
- **PES edge storage** keeps per-edge projection information close to the graph adjacency data.
- Full vector scoring is used for candidates that pass the projection filter.

The `ef_search` parameter controls how much work search can spend. Larger values generally improve recall at lower QPS.

## Online Updates

Online mode starts from an initial graph built by `Index::build` with `BuildOptions::mode = IndexMode::Online`. It then accepts single-vector `add` / `insert` calls and batch `add_batch` / `insert_batch` calls until the configured `max_elements` capacity is reached.

Online insertion stores the new vector, searches the current graph for candidate neighbors, connects the new node, and updates the mutable graph in memory. Search is available immediately after each insertion.

For dynamic deployments:

- choose `max_elements` with enough spare capacity;
- keep query `top_k` no larger than build-time `max_search_k`;
- build the initial graph from a representative batch;
- validate recall under the intended insertion order and query distribution;
- rebuild periodically if the application requires static-index-level recall after many updates.

## Static and Online Roles

The static packed index is the primary high-throughput path for workloads that can rebuild or refresh indexes in batches. Online mode is intended for workloads that need immediate in-memory updates between rebuilds.

Batch search is available on both static and online indexes. Batch insertion is available only in online mode; it is intended for update blocks where no query needs to be interleaved between individual inserted vectors.
