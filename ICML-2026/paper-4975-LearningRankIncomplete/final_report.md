# Final Report: paper-4975

- Title: Learning to Rank from Incomplete Rankings
- Primary metric: `PIRATE_avg_normalized_kendall` (lower)
- Records: 7
- Generated: 2026-07-12T10:43:30Z

## Best Result

- Iteration: 1
- Idea: ALGO-01 — Replace PIRATE boolean majority with PageRank continuous scoring
- Primary metric: 0.013333
- Commit: `1d07f6ab03d208c148db5ee6fa2dbfe1d10bb4f8`
- Notes: Replaced boolean E[i,j]=mu_ij>mu_ji and transitive closure with continuous transition matrix from win ratios + PageRank damping + power iteration. PIRATE improved from 0.0222 to 0.013333 (matches RankCentrality/BordaCount). Other algorithms unchanged. Parsed from stdout: PIRATE: mean=0.013333 std=0.010887
