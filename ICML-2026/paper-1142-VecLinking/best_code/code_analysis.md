# Code Analysis — VecLinking (Paper 1142)

## Evaluation Path
- Entry: `veclink.py` → `main` → `test_clu(args, seed)`
- Eval command: `python3 veclink.py --dataset nfcorpus --emb1 mistral --emb2 openai --overlap_ratio 0.3 --n_seeds 15 --seed 42 --use_gpu true`
- Final output: `Final results: accuracy=<P>, recall=<R>` parsed from last INFO line
- F1 computed as `2*P*R/(P+R)`

## Core Algorithm Flow
1. **Data loading** (L517-650): Load embeddings from `.npy` files, create random partition with `overlap_ratio=0.3`
2. **Seed sampling** (L800-890): Sample `n_seeds=15` reference pairs from `ind_nonref` using `ref_method=random`
3. **Iterative refinement** (L1000-1600): Bernoulli ensemble-based mutual NN discovery
   - Each iteration runs `ensemble_reference_selection_bernoulli()` with scheduled views
   - Paper view schedule: `sf_t = 1 + c*log(g_t)`, `m_t = ceil(m0*sf_t)`, `s_t = ceil(ρ0*|L|/sf_t)`
   - Overlap inference via Otsu's method on posterior distribution
   - Convergence: stable `mutual_nn_ratio` (<1% change) over 10 iterations
   - Max 100 iterations, but typically converges in 10-20

## Key Files
- `veclink.py` — main pipeline + ensemble orchestration (1777 lines)
- `utils/ensemble_selection.py` — Bernoulli voting and posterior inference (~3000 lines)
- `utils/retrieval_util.py` — accuracy/recall computation, mutual pair finding, CSLS
- `utils/sample_methods.py` — seed sampling methods (random, furthest, cluster, nearest, localized)
- `utils/graph_util.py` — distance computation utilities
- `utils/procrustes_util.py` — Procrustes alignment refinement
- `graph_utils/distance_encoder.py` — distance encoding for reference selection
- `generate_embeddings_nfcorpus.py` — regenerate embeddings if needed

## Metric Parser
- `compute_accuracy_recall()` in `retrieval_util.py`:
  - Precision = correct / len(ref_indices1)
  - Recall = correct / len(ind_nonref)
  - F1 = 2*P*R/(P+R)
- NOT to be modified per constraints

## Known Levers
| Lever | Current | Range | Description |
|---|---|---|---|
| seed | 42 | any int | Random seed for reproducibility |
| n_seeds | 15 | 1-N | Number of seed anchor pairs |
| overlap_ratio | 0.3 | 0.01-0.5 | Overlap ratio for random partition |
| ensemble_strategy | furthest | random/cluster/furthest/nearest | Anchor selection strategy within views |
| ensemble_n_ensembles | 5 | 1-20 | Base number of ensemble views m0 |
| ensemble_subset_ratio | 0.4 | 0.1-1.0 | Base per-view anchor fraction ρ0 |
| ensemble_vote_threshold | 0.6 | 0.0-1.0 | Vote threshold for pair selection |
| schedule_c | 0.3 | 0.0-1.0 | View-schedule growth constant |
| overlap_inference_method | otsu | threshold/adaptive/otsu/gmm/elbow/expected/gap | Overlap pair inference method |
| posterior_threshold | 0.1 | 0.01-0.5 | Posterior threshold (for threshold method) |
| max_iter | 100 | 10-200 | Max iterations |
| fp16 | true | bool | FP16 for ensemble computation |
| ref_method | random | random/furthest/cluster_centroids/nearest/localized_cluster | Seed sampling method |
| ref_filter_ratio | 0.9 | 0.5-1.0 | Reference filter keep ratio |
| use_procrustes | false | bool | Procrustes refinement |
| csls_neighborhood | 50 | 0-100 | CSLS neighborhood size |
| concat_seed_pairs | false | bool | Concatenate seeds to ensemble refs |

## NFCorpus Dataset
- 3633 documents from BEIR
- Embeddings at `/repo/embeddings/corpus_embeddings_{mistral,openai}_nfcorpus.npy`
- Mistral: 1024-dim, OpenAI: 1536-dim
- Overlap: 1074 document pairs (30% of unique set)

## Safe Modification Targets
1. Seed sampling method (random → furthest/cluster) — changes initial anchor quality
2. Ensemble strategy/parameters — changes view diversity
3. Overlap inference method — changes threshold for declaring pairs
4. Posterior threshold tuning — changes confidence required
5. Schedule parameters (c, m0, ρ0) — changes view scaling
6. Reference filtering — changes quality control

## Unsafe Targets (DO NOT MODIFY)
- `compute_accuracy_recall()` — metric definition
- `ind_nonref` generation — test split
- Embedding files — test data
- Any ground-truth access in eval path
