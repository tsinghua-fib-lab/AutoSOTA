# Code Analysis — Paper 3479 SOTA Preparation Repair

## Original Preparation Failure

The orchestrator preparation script failed at the git initialization step:
- `git` not installed in the Docker image `autosota/paper-3479:reproduced`
- `apt-get install git` failed due to HTTP proxy returning 502 errors
- The proxy at `172.17.0.1:17890` was not properly forwarding Ubuntu archive requests

## Corrected In-Container Evaluation Command

```bash
cd /repo && python3 eval_temporal_consistency.py --mode direct --granularity year
```

The manifest `eval_command` was correct and did not need translation. The eval script loads:
- Tree: `/repo/trees/tree_nj.nwk` (default, overridable via `--tree`)
- Mapping: `/repo/trees/leaf_mapping.tsv`
- Timestamps: `/repo/data/timestamps.json`
- Families: `/repo/data/family_labels.json`

## Baseline Verification

Baseline NJ+Outgroup tree: `Temporal Consistency: 0.8765` — exact match with manifest baseline.
- Correct pairs: 27,547
- Total pairs: 31,430
- Intra-family: 0.5007 (3,437/6,865)
- Inter-family: 0.4870 (432/887)

## Reusable Resources

- Pre-built phylogenetic trees: `/repo/trees/tree_nj.nwk` (103,883 leaves, NJ+outgroup), `/repo/trees/tree_upgma.nwk`
- Metadata: `/repo/trees/leaf_mapping.tsv` (leaf label -> SHA256 -> class_id -> family), `/repo/data/timestamps.json`, `/repo/data/family_labels.json`
- Cached leaf data: `/repo/results/leaf_timestamps.json`, `/repo/results/leaf_families.json`
- Example embeddings: `/repo/data/example/embeddings_fused.json`

## Safe Optimization Targets

### Safe (post-hoc tree modification, no eval changes):
1. Branch length adjustments (temporal calibration, edge collapsing)
2. Rooting strategy changes (midpoint, MAD, outgroup search)
3. Tree topology pruning/collapsing
4. Consensus methods between NJ and UPGMA

### Not attempted (requires embeddings not in container):
5. Distance matrix recomputation with different metrics
6. BIONJ or other tree construction algorithms
7. PCA/kernel PCA preprocessing
8. Hierarchical two-stage tree construction
9. Embedding fusion modifications
