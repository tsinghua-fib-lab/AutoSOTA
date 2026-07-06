# Search Configuration

Parameters in `search_config.gin` are grouped below. Edit them directly — changes take effect without rebuilding the Docker image.

---

## General

| Macro | Default | Options | Description |
|---|---|---|---|
| `device` | `"cuda"` | `"cpu"`, `"cuda"` | Hardware used for all ML models |
| `single_step_model` | `"template"` | `"template"`, `"pdvn"`, `"g2e"` | Retrosynthesis model. Use `"pdvn"` or `"g2e"` with `policy_cost` in `objective_functions` |
| `single_step_topk` | `25` | integer | Top-k predictions returned per single-step model call |
| `building_blocks` | `"origin_dict.csv"` | filename | Building block set used during search (resolved relative to `models/`) |

---

## Objectives

| Macro | Default | Options | Description |
|---|---|---|---|
| `objective_functions` | `["sustainability_cost", "scaleup_cost", "toxicity_cost", "convergence_cost"]` | list | Cost functions to optimise. Use `"policy_cost"` instead of `"convergence_cost"` when using `"pdvn"` or `"g2e"` |
| `pareto_objectives` | `3` | integer ≤ len(`objective_functions`) | Number of objectives used for Pareto filtering |
| `max_dominated_solutions` | `10` | integer | Max dominated solutions retained per node |
| `max_pareto_solutions` | `150` | integer | Target Pareto front size — search stops early if reached and `stop_on_full_pareto = True` |

---

## Search

| Macro | Default | Description |
|---|---|---|
| `iteration_budget` | `300` | Max MORetro iterations per target |
| `single_step_call_budget` | `350` | Max single-step model calls per target |
| `time_budget` | `0` | Wall-clock limit in seconds per target (`0` = no limit) |
| `max_node_depth` | `15` | Max depth of the search tree |
| `stop_on_full_pareto` | `True` | Stop early when `max_pareto_solutions` is reached |
| `exclude_dominated_nodes` | `False` | Prune dominated nodes during search |
| `epsilon_pruning` | `0` | Tolerance for approximate Pareto front (`0` = exact) |

---

## Weight Sampling

Controls how scalarisation weight vectors are generated and updated.

| Macro | Default | Options | Description |
|---|---|---|---|
| `sampling_strategy` | `"bo"` | `"bo"`, `"queue"` | Weight update strategy. `"bo"` uses Bayesian optimisation; `"queue"` cycles through a fixed set |
| `no_concurrent_weights` | `5` | integer | Number of weight vectors evaluated in parallel per iteration |
| `iter_per_weight` | `15` | integer | MCTS iterations allocated per weight vector |
| `no_samples` | `128` | integer | Number of candidate weights sampled during BO acquisition |
| `weight_sampling_strategy` | `"grid"` | `"grid"`, `"sobol"` | Initial weight distribution before BO kicks in |
| `include_extreme` | `False` | bool | Include axis-aligned (single-objective) extreme weights in the initial set |

---

## Bayesian Optimisation (BO) Selector

Only relevant when `sampling_strategy = "bo"`.

| Parameter | Default | Description |
|---|---|---|
| `BOWeightSelector.kappa` | `2.0` | Exploration–exploitation trade-off (higher = more exploration) |
| `BOWeightSelector.n_warmup` | `10` | Random weight vectors sampled before BO model is fitted |
| `BOWeightSelector.decay_factor` | `0.5` | Age penalty decay applied to older weight observations |
| `BOWeightSelector.max_age` | `2` | Max number of reuses of a weight vector before it is retired |
