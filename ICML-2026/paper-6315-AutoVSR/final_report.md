# Final Report: paper-6315

- Title: AutoVSR: Automatic Visual-to-Symbolic Reasoning for Symbolic Expression Generation from Circuit Schematic
- Primary metric: `Accuracy` (higher)
- Records: 3
- Generated: 2026-07-15T06:42:33Z

## Best Result

- Iteration: 1
- Idea: CODE-01 — Multi-strategy Lcapy: transfer() + V-ratio fallback
- Primary metric: 100.0
- Commit: `24b4e8d72d38e5a09486ebb9b30205779001e70c`
- Notes: Replaced single cct.transfer() with multi-strategy approach: primary transfer() (15s timeout) + fallback V_elem.s/V_src.s voltage ratio (120s timeout). All 17 previously-timing-out samples now compute correctly via vratio. Result: 336/336 correct, 0 timeouts vs baseline 319/336 correct, 17 timeouts. Accuracy 100% both excl. and incl. timeouts.
