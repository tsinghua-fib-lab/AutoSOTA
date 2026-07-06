# Final Report: paper-385

- Title: Embedding Trust: Semantic Isotropy Predicts Nonfactuality in Long-Form Text Generation
- Primary metric: `R_squared` (higher)
- Records: 9
- Generated: 2026-07-04T20:59:40Z

## Best Result

- Iteration: 7
- Idea: PARAM-01-POOL — Max pooling + polynomial degree 3
- Primary metric: 0.4221
- Commit: `64f99bb842fb50c18be2be04bc184bc87dc946f6`
- Notes: Pooling sweep: mean=0.3544, last=0.1030, max=0.4143, cls=0.3668. Best=max pooling + poly deg 3: R2=0.4221+/-0.0563, adj R2=0.4045, Pearson r=-0.6062. Max pooling R2_linear=0.3675 already beats mean pooling R2_poly3. Fixed max pooling bug in isotropy.py.
