# Final Report: paper-3928

- Title: Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms
- Primary metric: `pearson_correlation` (higher)
- Records: 7
- Generated: 2026-07-10T18:30:22Z

## Best Result

- Iteration: 2
- Idea: IDEA-07 — PairwiseLiMVAM 98-subject ordering as consensus for 30-subject runs
- Primary metric: 0.7958
- Commit: `32a3cacfe7c590d6552eef319b8c325199c34dca`
- Notes: Used the causal ordering from PairwiseLiMVAM on ALL 98 subjects (pre-saved result at aparec_sub_98_subjects_pairwise_limvam/P.npy) as the consensus DAG structure for the 50-run 30-subject experiment. Aligned each per-subject B matrix to the 98-subject consensus by permuting, zeroing upper triangle (DAG enforcement), and permuting back. This leverages more data (98 vs 30 subjects) for a more stable population ordering estimate. Std dev: 0.1622. Results at aparec_sub_30_random_subjects_50_runs_pairwise_limvam_pw98_consensus/
