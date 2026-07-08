# PAG Code Analysis for SOTA Optimization

## Evaluation Path
- **CLI binary**: `./build/PAG` (built from `main.cpp` + `pag.cpp`)
- **Entry point**: `main.cpp` → `RunPAG()` → `RunPAGWorkflow()`
- **Build phase**: `RunPAGWorkflow()` in `pag.cpp` lines ~1417-1817
  - Reads base vectors from `.fbin` file
  - Builds projection vectors, permutation
  - Inserts points in parallel using OpenMP
  - Completes PES (Projection Edge Set), encodes projections
  - Packs search layout, saves index
  - Outputs "PAG build time: N seconds" on stderr
- **Search/benchmark phase**: `RunSearchBenchmark()` in `pag.cpp` lines ~380-468
  - Loads saved index from disk
  - Runs search at multiple ef_search levels (10 points by default)
  - Outputs: `efs\tRecall\tQPS\n` (tab-separated)
- **Metric parsing**: Parse line where `efs` equals target (e.g., 1000), extract Recall and QPS columns

## Config Path
- `pag_config.h`: `PAGRunConfig` struct
  - `ef_construction`, `target_degree`, `projection_levels` control index quality
  - `base_count`, `query_count`, `dim`, `result_k` control data dimensions
  - `metric_name`, `build_order`, `max_search_k` control behavior
- Build system: `CMakeLists.txt` - `PAG_USE_AVX512=ON`, `PAG_NATIVE_ARCH=OFF`

## Key Source Files
1. **pag.cpp** (1817 lines): CLI, build workflow, search benchmark
2. **paglib/pag_index_core.h** (2600 lines): Core index data structures, vector encoding, distance computation
3. **paglib/pag_search_engine.inc** (1566 lines): TFB search engine with PRT filtering
4. **paglib/pag_search_primitives.h** (491 lines): SIMD primitives for projection evaluation
5. **paglib/pag_build_pipeline.inc** (742 lines): Edge projection record writing, build insertion
6. **paglib/space_l2.h**: L2 distance kernels (AVX-512, AVX, SSE)
7. **paglib/space_ip.h**: Inner product kernels (AVX-512, AVX, SSE)

## Risk Classification
### Safe modification targets:
- `pag.cpp`: Benchmark loop (permutation hoisting - Idea 08), build I/O (pre-load vectors - Idea 07)
- `paglib/pag_search_engine.inc`: Query implementation (SIMD optimization - Idea 09)
- `pag_index.h` / `pag_config.h`: Parameter extensions (no eval protocol changes)

### Risky modification targets:
- `paglib/pag_index_core.h`: Vector encoding (int8 quantization - Idea 01), distance computation
- `paglib/pag_search_primitives.h`: PRT evaluation (must preserve correctness)
- `paglib/pag_build_pipeline.inc`: Edge record format changes

### Do NOT modify:
- Ground truth data, metric computation formulas, recall calculation
- Dataset split (base/query vectors)
- Output format (tab-separated efs/Recall/QPS table)
- Evaluation protocol (ef_search sweep behavior)

## Known Levers
- **ef_construction** (100-10000): Higher values → better recall at cost of build time
- **target_degree M** (16-64): Higher M → better recall, more memory
- **projection_levels L** (8-128, multiple of 8): Controls PRT granularity
- **ef_search**: Runtime efficiency knob (trade QPS for recall)
- **Metric**: l2, cosine, mips
- **Build order**: dataset_order (default)

## Baseline Configuration
- ef_construction=1000, M=32, L=128, L2 metric
- 999K base vectors, 1K queries, d=1536
- QPS=527 at ef=1000 (recall=0.9994)
- Build time: 884s
- Index size: 6.8GB
