# Paper 75 — TreeSlicedEntropy

**Full title:** *Tree-Sliced Entropy Partial Transport*

**Registered metric movement (internal ledger, ASCII only):** +0.50%(0.8678->0.8721)

# Final Optimization Report: Tree-sliced entropy transport (paper-75)

## Summary

Twelve SOTA iterations improved **`target_accuracy`** from **0.8678 → 0.8721 (+0.50%)** on the internal benchmark. The rubric’s **+2%** paper target was **not** reached, but the run delivered a reproducible gain over the shipped baseline.

## Key ideas (results ledger)

- Switch generative / slicing mode toward **`gen_mode = gaussian_orthogonal`** (or equivalent in code) so sliced directions better cover the transport polytope.
- Expose CLI control **`--twd_nlines 8`** (and neighbors) to increase the number of tree-sliced directions used during the entropy partial transport step.

## Where to look next

- **`README.md`** for installing the partial-transport backend; search the repo for `twd_nlines` and `gen_mode`.
