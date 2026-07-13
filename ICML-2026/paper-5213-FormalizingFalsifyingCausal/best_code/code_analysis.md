# Code Analysis — Paper 5213: Causal Pathways of Rare Events

## Evaluation Path
- **Command**: `python3 compute_pathway_score.py`
- **Script**: `/repo/compute_pathway_score.py` (105 lines, self-contained)
- **Output format**: stdout with `RESULT: Pathway Explanation Score = <value>` line
- **Also written to**: `/repo/pathway_score_result.txt`
- **Primary metric**: Pathway Explanation Score (higher is better, range (-∞, 1.0])

## Metric Computation
```python
score = 1.0 - log(P(B|A) * P(C|B) * P(D|C) * P(E|D)) / log(P(E))
```
- `joint_cond = 0.55 * 0.80 * 0.05 * 0.20 = 0.0044`
- `score = 1 - log(0.0044)/log(0.0005) = 1 - (-5.4261)/(-7.6009) = 0.2861`
- **Parser**: grep for `RESULT: Pathway Explanation Score =` line, extract float

## Key Files
| File | Role | Safe to Modify? |
|------|------|-----------------|
| `compute_pathway_score.py` | Main evaluation | YES — add config loading, diagnostics, structure search, batch mode |
| `figure_example_4_6.py` | Example 4.6 (Gaussian bivariate) | Reference only |
| `figure_example_4_8.py` | Example 4.8 (Gaussian trivariate) | Reference for abstraction-level formulas |
| `pathway_score_result.txt` | Output artifact | Written by eval script |
| `requirements.txt` | Dependencies (numpy, scipy, matplotlib) | NO |

## Safe Modification Targets
1. **`compute_pathway_score.py`** — the core eval script. All ALGO/CODE ideas modify this.
2. **New config file** — `pathway_config.json` for externalizing probability/pathway parameters.
3. **No test data, labels, or splits exist** — purely formula-based evaluation.

## Red-Line Boundaries
- Do NOT change the formula: `score = 1 - log(P(pathway|do(root))) / log(P(target))`
- Do NOT change the RESULT output format or the score file path
- Do NOT hard-code a higher score without genuine computation
- All probability values must be documented and defensible

## Known Levers (from manifest)
1. Pathway structure (nodes, edges, DAG topology)
2. Root cause set selection  
3. Conditional probability estimates (currently LLM-derived)
4. Target marginal probability P(E)
5. Abstraction level (bivariate vs trivariate vs chain)

## Pre-Optimization Diagnostics
- P(D|C) = 0.05 is the smallest multiplier → highest log contribution → most sensitive
- P(E) = 0.0005 is in the denominator → changes have non-linear effects
- ∂Score/∂p_i = -1/(p_i × ln(P(E))) — smaller p_i means larger derivative magnitude
- The pathway chain has 4 edges; shorter pathways inherently score higher
