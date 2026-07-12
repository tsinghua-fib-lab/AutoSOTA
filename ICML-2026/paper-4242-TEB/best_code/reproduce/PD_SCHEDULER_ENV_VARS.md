# PD Scheduler Environment Variables

Reference for every `VLLM_PD_*` (and adjacent `VLLM_*`) env var read by the
EB / EB(k̂\*) / EB⁺ scheduler. Generated from a grep of
`vllm/v1/core/sched/{scheduler,calibration}.py` — keep in sync when adding
new knobs.

## Conventions

* **Default** is the literal default coded in `os.environ.get(...)`; an empty
  string usually means "auto-derive from calibration / model".
* **Audience** classifies who should ever touch the var:
  * **user** — common knobs documented in `reproduce/README.md`.
  * **paper** — needed to faithfully reproduce a specific paper artifact.
  * **advanced** — tuning levers for new hardware / workloads.
  * **internal** — diagnostic or guard rail; do not change unless debugging.
* **Read site** column gives `file:line` to find the exact use site quickly.

---

## 1. Mode selection (which scheduler runs)

| Var | Default | Type | Audience | Read site | Notes |
|---|---|---|---|---|---|
| `VLLM_USE_PD_SCHEDULER` | `0` | `0/1` | user | `scheduler.py:220` | Legacy boolean toggle. Superseded by `VLLM_PD_SCHEDULER_MODE` (which wins if both set). |
| `VLLM_PD_SCHEDULER_MODE` | `""` | `v1 \| eb \| auto \| ""` | user | `scheduler.py:232` | Primary mode selector. `v1`=vLLM v1 mixed-batching (default), `eb`=pure EB(k̂\*), `auto`=EB⁺ (online v1↔eb crossover). Empty falls back to the legacy flag. |
| `VLLM_PD_AUTO_COLD_START_MODE` | `v1` | `v1 \| eb` | advanced | `scheduler.py:545` | Which sub-scheduler EB⁺ runs in until it has enough IFR samples to fire the first crossover decision. |

## 2. Calibration (cost model — α/β coefficients)

| Var | Default | Type | Audience | Read site | Notes |
|---|---|---|---|---|---|
| `VLLM_PD_CALIBRATION_FILE` | `""` | path | user | `calibration.py:609` | JSON with per-(model, GPU) α\_p/β\_p/α\_d/β\_d. Sample: `reproduce/calibration/pd_calibration_Qwen3-8B_H200.json`. See `reproduce/calibration/README.md` for how to generate. |
| `VLLM_PD_ALPHA_P` | — | float | advanced | `scheduler.py:266` | Override α\_p (prefill constant). No default — only honoured if calibration file is missing **and** all four α/β envs are set. |
| `VLLM_PD_BETA_P`  | — | float | advanced | `scheduler.py:267` | Override β\_p (prefill slope, per-token). |
| `VLLM_PD_ALPHA_D` | — | float | advanced | `scheduler.py:268` | Override α\_d (decode constant). |
| `VLLM_PD_BETA_D`  | — | float | advanced | `scheduler.py:269` | Override β\_d (decode slope, per-batched-token). |

## 3. EB(k̂\*) — threshold (θ) selection

| Var | Default | Type | Audience | Read site | Notes |
|---|---|---|---|---|---|
| `VLLM_PD_K_MODE` | `direct` | `direct \| ratio \| ifr \| cfr` | paper | `scheduler.py:241` | How θ\* is chosen. `direct`=fixed k\*; `ratio`=fixed θ\* (used for fixed-k EB ablations); `ifr`=online IFR estimator — **this is the paper's EB(k̂\*) and the recommended EB⁺ inner kernel** (paper §4.3); `cfr`=online CFR estimator with closed-form midpoint construction (η ≡ 0; **journal-version material**, not described in the camera-ready paper). |
| `VLLM_PD_K_STAR` | `""` | int | paper | `scheduler.py:243` | In `direct` mode: pin batch size to this k\*. |
| `VLLM_PD_K_RATIO` | `""` | float ∈ (0,1) | paper | `scheduler.py:247` | In `ratio` mode: pin θ\* to this value. |
| `VLLM_PD_IFR_WINDOW_SIZE` | `500` | int | advanced | `scheduler.py:306` | IFR sliding-window size (requests). Smaller → more reactive, noisier. |
| `VLLM_PD_IFR_UPDATE_INTERVAL` | `100` | int | advanced | `scheduler.py:311` | Re-estimate θ\* every N completions. |
| `VLLM_PD_IFR_MIN_SAMPLES` | `30` | int | advanced | `scheduler.py:314` | Minimum samples in window before trusting the IFR estimate; until then use `IFR_DEFAULT_THETA`. |
| `VLLM_PD_IFR_DEFAULT_THETA` | `0.70` | float | advanced | `scheduler.py:317` | Cold-start θ\* before enough IFR samples accumulate. |
| `VLLM_PD_IFR_THETA_EMA_ALPHA` | `0.3` | float ∈ (0,1] | advanced | `scheduler.py:339` | EMA smoothing on the IFR-derived θ\*. Higher → more reactive. |
| `VLLM_PD_THETA_MAX` | `0.80` | float | advanced | `scheduler.py:325` | Hard upper bound on θ\* (guards against pathological workloads pushing θ→1, which would starve the prefill stage). |
| `VLLM_PD_THETA_FLOOR` | `0.01` | float | advanced | `scheduler.py:333` | Lower bound on θ\*. Floor of 0 collapses EB to MB; values >0 keep some exclusive-batch pressure even on prefill-dominated workloads. |

## 4. EB(k̂\*) — capacity (N) selection

| Var | Default | Type | Audience | Read site | Notes |
|---|---|---|---|---|---|
| `VLLM_PD_AUTO_COMPUTE_N` | `0` | `0/1` | advanced | `scheduler.py:448` | When `1`, derive N from KV-cache headroom each tick instead of from the configured `max-num-seqs`. Useful when you do not want to hand-tune (B,N). Slight overhead. |
| `VLLM_PD_OOM_TOLERANCE` | `0.01` | float | advanced | `scheduler.py:450` | Headroom kept under the KV-cache OOM line (1% by default). Higher → safer, lower throughput. |
| `VLLM_PD_KV_THRESHOLD_MAX` | `0.6` | float | advanced | `scheduler.py:1305` | Hard cap on the running KV usage fraction at which we throttle accepting new requests. Guard rail; rarely needs changing. |
| `VLLM_PD_MIN_N_FLOOR_DIV` | `10` | int | advanced | `scheduler.py:1788` | N floor = `max-num-seqs / MIN_N_FLOOR_DIV`. Stops the IFR controller from collapsing N to 0 on adversarial bursts. |
| `VLLM_PD_OUTPUT_MARGIN` | `0.5` | float | internal | `scheduler.py:431` | Safety margin baked into the output-length estimate used by `AUTO_COMPUTE_N`. |
| `VLLM_PD_BASE_KV_RESERVE` | `0` | int | internal | `scheduler.py:428` | Static KV blocks to reserve before the headroom calc kicks in. Mostly zero. |

## 5. EB⁺ — online mode switching (auto)

These only take effect when `VLLM_PD_SCHEDULER_MODE=auto`. The crossover
diagnostic Δ(N) compares the amortised cost of one MB step vs. one EB step.

| Var | Default | Type | Audience | Read site | Notes |
|---|---|---|---|---|---|
| `VLLM_PD_MB_COST_A` | `0` | float | paper | `scheduler.py:552` | β\_MB^e(r) polynomial coefficient `a` in `f(r) = a + b·r + c·r²` (paper Eq. 11). Measured from a one-time kernel sweep per (model, GPU). |
| `VLLM_PD_MB_COST_B` | `0` | float | paper | `scheduler.py:553` | Coefficient `b`. |
| `VLLM_PD_MB_COST_C` | `0` | float | paper | `scheduler.py:554` | Coefficient `c`. With all three at 0 the LHS of the crossover collapses to 0, making the decision purely amortised-overhead driven (conservative but still informative — see `reproduce/eb_plus/traffic/run_adaptive_selector.sh:21-29`). |
| `VLLM_PD_BETA_MB_E` | `β_EB^w` | float | advanced | `scheduler.py:977` | Constant proxy for β\_MB^e in the diagnostic Δ(N) formula. Defaults to the EB-side workload-weighted β (best available proxy when no calibration). Distinct from the polynomial above. |
| `VLLM_PD_ALPHA_MB` | `α_p` | float | advanced | `scheduler.py:562 + 978` | α\_MB in paper Eq. 11/13 (per-step MB constant). Read at __init__ (the actual mode-switch decision) AND in the diagnostic Δ(N). Defaults to α\_p (paper approximation α\_p ≈ α\_d ≈ α\_MB). |
| `VLLM_PD_MODE_SWITCH_DELTA` | `0.0001` | float | advanced | `scheduler.py:494` | Hysteresis band around the crossover. \|Δ(N)\| must exceed this to flip. Larger → more sticky. |
| `VLLM_PD_MODE_COOLDOWN` | `3` | int | advanced | `scheduler.py:496` | Min N-update ticks between two flips. |

## 6. Cadence / housekeeping

| Var | Default | Type | Audience | Read site | Notes |
|---|---|---|---|---|---|
| `VLLM_PD_PARAM_UPDATE_INTERVAL` | `100` | int | advanced | `scheduler.py:416` | How often (in scheduler ticks) to refresh per-workload aggregates (μ\_L, μ\_O, etc.). |
| `VLLM_PD_N_UPDATE_COOLDOWN` | `2.0` | float (sec) | advanced | `scheduler.py:436` | Minimum wall-clock seconds between N-updates. Prevents thrash on bursty inputs. |

## 7. Diagnostics (off by default; set by reproduce scripts)

| Var | Default | Type | Audience | Read site | Notes |
|---|---|---|---|---|---|
| `VLLM_COLLECT_SCHEDULE_STATS` | `0` | `0/1` | user | `scheduler.py:532` | Collect per-step scheduler stats (mode-switch history, θ trace, N trace, IFR samples). Required for the `*_stats.json` files that `reproduce/eb_plus/traffic/analyze_selector.py` reads. |
| `VLLM_SCHEDULE_STATS_FILE` | `schedule_stats.json` | path | user | `scheduler.py:536`, `:3606` | Where to write the stats JSON. The reproduce scripts set this per-run. |

---

## Quick-start: minimal env for each scheduler

```bash
# vLLM v1 default (MB) — no PD envs needed
vllm serve ...

# EB(k̂*) — paper §4.3
VLLM_PD_SCHEDULER_MODE=eb \
VLLM_PD_K_MODE=ifr \
VLLM_PD_CALIBRATION_FILE=reproduce/calibration/pd_calibration_<MODEL>_<GPU>.json \
vllm serve ... --max-num-seqs <N> --max-num-batched-tokens <B>

# EB+ — paper §4.4 (recommended for unknown traffic)
VLLM_PD_SCHEDULER_MODE=auto \
VLLM_PD_K_MODE=ifr \
VLLM_PD_MB_COST_A=<a> VLLM_PD_MB_COST_B=<b> VLLM_PD_MB_COST_C=<c> \
VLLM_PD_CALIBRATION_FILE=... \
vllm serve ...
```

The `CP_COST_*` coefficients come from a one-time kernel sweep on the target
(model, GPU). Without them EB⁺ still runs but the crossover decision relies
entirely on the amortised-overhead RHS — typically conservative (stays in MB
longer than optimal). See the header of
[`reproduce/eb_plus/traffic/run_adaptive_selector.sh`](eb_plus/traffic/run_adaptive_selector.sh)
for details.

## Future work

These 33 vars are currently read inline via `os.environ.get(...)` scattered
through `scheduler.py`. The upstream vLLM convention is to register all env
vars in `vllm/envs.py` (a single `environment_variables: dict[str, Callable]`
with type hints + defaults). Migrating to that convention would:

* Give IDE auto-complete and central type checking.
* Let users discover knobs via `python -c "from vllm import envs; print([k for k in envs.environment_variables if k.startswith('VLLM_PD_')])"`.
* Make upstreaming patches easier.

Tracked as an open-source-readiness item; this doc is the interim source of
truth.
