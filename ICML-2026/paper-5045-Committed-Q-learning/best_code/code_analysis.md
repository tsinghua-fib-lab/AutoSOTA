# Code Analysis — Paper 5045: Commit to the Bit

## Repository Structure

| File | Role |
|------|------|
| `qcorridor.jl` | Core algorithm: Committed Q-learning and regular Q-learning for the corridor environment |
| `experiment.jl` | Experiment runner: loops over seeds, checks optimality, computes bootstrap CI |
| `qcorridor.jl.orig` | Original version before reproduction bug fix (line 20 dim mismatch) |

## Evaluation Path

- **Command:** `julia experiment.jl` (from `/repo`)
- **Entry point:** `experiment.jl` -> `run_experiment()` with default params (k=5, T=1000, n_runs=1000)
- **Metric output:** Parsed from stdout line `Committed Q-learning:  <N>/<total> = <decimal>`
- **Secondary output:** Parsed from stdout line `Regular Q-learning:    <N>/<total> = <decimal>`

## Algorithm Details

### Corridor Environment
- State x in {0, 1, ..., k+1}: x=0 is start, x=k+1 is terminal (reward = k)
- Feature z = I(x > 0) — binary state aggregation
- Actions u in {0 (left), 1 (right)}
- Reward: -1 per step, 0 at start, k at goal
- Optimal reactive policy: always go right (u=1) regardless of feature z

### Committed Q-learning
- Standard Q-learning with epsilon-greedy exploration
- Key mechanism: once an action is selected in a feature, it persists (commits) until the feature changes
- When feature changes (z_ != z), a new action is sampled from epsilon-greedy policy
- Q-table is 2x2 (features x actions)

### Optimality Check
- `check_optimal(q)`: returns true if `q[1,2] > q[1,1] && q[2,2] > q[2,1]`
- Both features must prefer "right" over "left"
- Uses strict `>` — ties (equal Q-values) count as "not optimal"

## Known Bug Fix (Reproduction)

Line 20 of original qcorridor.jl: `qs = N != 1 ? zeros(size(ts, 1), 2, 2) : q`
- When N=1, `q` is a 2x2 matrix but `qs[j, :, :]` requires 3D indexing
- Fixed to: `qs = N != 1 ? zeros(size(ts, 1), 2, 2) : zeros(1, 2, 2)`

## Safe Modification Targets

| Target | File | Risk |
|--------|------|------|
| Q initialization | qcorridor.jl:17 | Low — one-line change |
| Learning rate schedule | qcorridor.jl:11-15 | Low — pure math |
| Exploration strategy | qcorridor.jl:39-41, 52-55 | Medium — changes action selection |
| Optimality check | experiment.jl:7-8 | Very Low — evaluation only, not algorithm |
| Visit counters | qcorridor.jl (new code) | Low — additive diagnostic |
| Command-line params | experiment.jl:10, 55 | Low — doesn't change algorithm |

## Risky Files

- **qcorridor.jl**: Core algorithm — any bug here changes results. Always test with `com=false` as sanity check.
- **experiment.jl**: Evaluation protocol — do not change output format or metric definition.

## Red Lines

1. Do NOT change the output format (Committed Q-learning / Regular Q-learning lines)
2. Do NOT change the metric definition (check_optimal logic, except >= fix which is a correctness improvement)
3. Do NOT change the environment (reward, transitions, feature encoding)
4. Do NOT hard-code results or manipulate seeds to favor outcomes

## Optimization Strategy

The primary metric is at ceiling (1.0) for k=5, T=1000. Optimization targets:
1. **Harder settings** (k >= 10, T <= 500) where committed Q-learning degrades
2. **Algorithm improvements** (optimistic init, UCB, intrinsic bonus, adaptive LR)
3. **Evaluation fixes** (>= in check_optimal)

Each iteration should test at harder settings and compare committed vs. regular Q-learning.
