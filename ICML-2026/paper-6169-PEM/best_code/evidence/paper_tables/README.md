# Paper Tables

This directory contains LaTeX tables used in the paper.

## Contents

| File | Paper Location | Description |
|------|----------------|-------------|
| `table_probeswitch_comparison.tex` | Table 1 | Probe-and-Switch vs competitors: pairwise comparison |
| `table_a11_high_misranking.tex` | Appendix A11 | Complete results on high-misranking COCO functions |
| `table_probeswitch_comparison_standalone.tex` | — | Standalone version with document preamble for direct compilation |

## Reproduce

```bash
python3 tools/make_main_paper_tables.py --output-dir evidence/paper_tables
```
