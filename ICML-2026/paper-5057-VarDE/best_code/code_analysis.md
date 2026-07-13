# Code Analysis — VarDE (Paper 5057)

## Evaluation Path
- **File**: `bai/eval_bai3.py`
- **Entry point**: Runs CR-A and VarDE_lse(tau=0.1) on BAI.3 bandit
- **Config**: 14 arms, Gaussian rewards, T=200 budget, n=20000 runs
- **Arms**: means[0]=0.4, means[i]=0.4-0.9^(i+10) for i=1..13; stds shuffled
- **Metric parsing**: stdout lines matching `<algorithm>: Error Probability = X.XXXX%`
- **Output file**: `results/eval_bai3_results.txt`

## Algorithm Path
- **Base class**: `bai/VarDE.py:VarDEBAI` — abstract class for all VarDE variants
- **Variance tracking**: `bai/utils.py:Welford` — Welford algorithm with var_floor
- **Variant classes**: VarDE_const, VarDE_lse, VarDE_nesterov, VarDE_entmax, VarDE_pairwise_softplus, VarDE_power_mean
- **Score computation**: `select_arm()` (line 68-77): score_i = w_i^2 * var_i / (N_i * (N_i+1))
- **Weight computation**: `compute_w()` — variant-specific; lse uses softmax over means/tau

## Key Config Parameters
| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| tau | 0.1 | VarDE_lse.__init__ | Temperature for LSE softmax |
| var_floor | 0.01 | VarDEBAI.__init__ | Minimum variance floor |
| warm_start | 1 | VarDEBAI.__init__ | Rounds of warm-start pulls |
| use_empirical_variance | True | VarDEBAI.__init__ | Use empirical vs fixed variance |
| use_influence_weights | True | VarDEBAI.__init__ | Use computed weights vs uniform |
| fixed_variance | 1.0 | VarDEBAI.__init__ | Fixed variance when empirical=False |
| initial_var | inf | VarDEBAI.__init__ | Initial variance before n>=2 |

## Safe Modification Targets
1. **VarDE.py:effective_vars()** (line 63-65) — Change how variance is estimated from trackers
2. **VarDE.py:select_arm()** (line 68-77) — Change score computation formula
3. **utils.py:Welford** — Change variance estimation logic
4. **eval_bai3.py** — Add evaluation modes (--fast, --sweep), DO NOT change metric definition

## Red-Line Boundaries
- Do NOT change T=200, K=14 arm distribution, or n=20000 in final eval
- Do NOT change the error probability computation (final arm recommendation vs true_best)
- Do NOT modify Bandit environment or reward distribution
- Do NOT hard-code outputs or skip the evaluation loop
- Always run CR-A alongside VarDE for guardrail monitoring

## Reusable Resources
- No external datasets or models needed
- Pure CPU numpy implementation
- Container has Python 3.10.13, numpy 1.26.2, tqdm 4.65.0, matplotlib, pandas, seaborn
