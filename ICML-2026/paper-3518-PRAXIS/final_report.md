# Final Report: paper-3518

- Title: From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets
- Primary metric: `Recall` (higher)
- Records: 9
- Generated: 2026-07-10T07:50:12Z

## Best Result

- Iteration: 3
- Idea: IDEA-05 — Leaf-level greediness at depth 3
- Primary metric: 1.0
- Commit: `3a3cf40488170176049891dce2833288b83250ea`
- Notes: IDEA-05: In lickety_split_k1, skip full lookahead at depth_budget==3 and use train_greedy directly. Time: 0.049s (-52.9% vs baseline 0.103s). Peak MB: 215.80 (+0.9%). Recall: 1.0 (exact match). Cache sizes decreased 60% due to fewer unique subproblems. Combined with iter-1 (cache_early_exits) and iter-2 (trie_cache).
