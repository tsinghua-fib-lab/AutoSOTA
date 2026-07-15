# Code Analysis — Paper 6233 (OC-space)

## Evaluation Path

`eval_credit_octree.py` → `load_credit_pareto_models()` → foreach model:
1. `enumerate_ocs_to_disk()` (depth_first_ocenum.py) — enumerate OC-space boxes to Zarr
2. `run_adversarial_robustness_tasks()` (verification.py) → `emp_robustness_octree_index()`:
   - Build `OCTreeIndex` (index/index.py) from OC-space shards
   - Query 500 correctly classified test instances via `find_closest_adversarial_example()`
3. Collect per-model times, compute geometric mean

## Key Files

| File | Role | Safe to Modify |
|------|------|----------------|
| `eval_credit_octree.py` | Main eval script | Yes — parameters, loops |
| `src/depth_first_ocenum.py` | OC-space enumeration | Yes — buffer, algorithm |
| `src/index/index.py` | OCTreeIndex, RootboxIndex, PosNegIndex | Yes — index construction, queries |
| `src/verification.py` | Verification methods, shard iteration | Yes — numba decorators, methods |
| `src/util.py` | Distance functions, dataset loading | Yes — numba decorators, functions |
| `data/raw/OC-space_paper_compression.txt` | Compressed models (114MB) | NO — paper data |
| `/datasets/prada/` | Cached prada datasets | NO — test data |

## Metric Parser

Parse stdout for `Geometric mean time: VALUE ms`. Also saved to JSON at `SAVE_DIR/credit_octree_results.json`.

## Numba Hot Path

All distance functions in `util.py` and `verification.py` use `@numba.njit` without `cache=True`.
Key functions: `linf`, `dist_to_boxes`, `dist_to_closest_box`, `min_dist`, `overlaps_batch`, `overlaps`, `equals`.

## OCTreeIndex Hot Path

`find_closest_adversarial_example()` (index.py:492-501):
1. Compute distance in same bin
2. Use generator_boxes with condition lambda to filter by distance
3. Compute min distance for each yielded batch

The `condition` lambda re-computes `dist_to_boxes` for already-seen boxes.
The generator traverses the tree, filtering nodes by the condition.

## Safe Modification Targets

1. Add `cache=True` to numba functions (util.py, verification.py) — no logic change
2. Add `fastmath=True` to numba functions — small floating-point diffs possible
3. Memoize distance computations in `find_closest_adversarial_example`
4. Optimize `bulk_insert` to avoid box duplication
5. Auto-tune OCTreeIndex depth
6. Adjust buffer_size in enumerate_ocs_to_disk
7. FIPE pruning of AddTrees before LOP compression
8. Grid search (pruning, alpha) parameters

## Risky Files (DO NOT MODIFY)

- `data/raw/OC-space_paper_compression.txt` — paper data
- `/datasets/prada/` — test datasets
- `pyproject.toml` — dependencies (already working)
- Any file that changes the evaluation metric definition

## Reusable Resources

- Compressed models at `data/raw/OC-space_paper_compression.txt`
- Prada dataset cache at `/datasets/prada`
- Gurobi license (for Kantchelian MILP baseline)
- OC-space cache at `/autosota_cache/oc_space`
