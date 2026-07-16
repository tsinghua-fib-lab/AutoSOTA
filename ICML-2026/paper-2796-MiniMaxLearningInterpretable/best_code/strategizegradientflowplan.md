# Strategize Backend: Gradient-Flow & Training-Stability Fixes

**Audience:** the next agent, working in the **`strategize`** repository (NOT `preference.fm`).
**Prereq:** you must have the `strategize` source checked out. The model lives in
`strategize/R/two_step_model_outcome_neural.R` (the NumPyro/JAX model definition,
compiled via reticulate) and `strategize/R/backend_jax.R` (the SVI training loop /
JAX runtime). None of these files exist in `preference.fm`, so **every line/function
reference below is a search target to confirm, not a guaranteed location.**

---

## Context

`preference.fm` trains pooled Bayesian-Transformer foundation models for conjoint
experiments. The transformer, its priors, the ELBO/SVI loop, and the Muon optimizer
all live in the external `strategize` package; `preference.fm` only builds the data
pool and calls into strategize via `R/backend_bridge.R`.

An architecture review found seven issues. Two are already resolved on the
`preference.fm` side and are **out of scope** here:

- **Issue 2** (text embeddings not L2-normalized) — fixed in `preference.fm`
  (`cs_foundation_build_text_registry` now unit-normalizes all embeddings).
- **Issue 7** (row-semantics OOM) — was dead code, deleted in `preference.fm`.

The remaining work is in `strategize`. This plan covers it.

### Issue-number mapping (original review → this plan)

| Original # | Title | Severity | In this plan |
|-----------:|-------|----------|:------------:|
| 1 | ReZero gates are HalfNormal latents, not zero-init | **High** | §1 |
| 3 | Universal mixed-task log-prob scale imbalance | Medium | §2 |
| 4 | Plate subsampling rescaling vs. base LR | Medium | §3 |
| 5 | MoE covariate encoder routing collapse | Medium | §4 |
| 6 | QK-norm scale prior too wide | Low-Med | Appendix A |

> Note on Issue 3: `preference.fm` *guards against* unsafe backends at
> `R/backend_bridge.R:14–36` (it string-matches the strategize source for the
> unsafe log-prob pattern and errors out). The **actual fix** belongs here in
> strategize; once fixed, that guard will pass. Keep the guard's detection
> pattern in mind — see §2.

---

## Ordering & dependency

```
§1 ReZero init ──┐
                 ├─► these three are independent; do in any order
§2 log-prob ─────┤
§4 MoE routing ──┘
                 
§3 LR/plate ─────► do LAST: it re-tunes the effective learning rate, so
                   land the gradient-scale fixes (§1,§2,§4) first, THEN
                   re-tune LR against the corrected gradient magnitudes.
```

Rationale: §1, §2, and §4 each change the *magnitude/direction* of gradients.
§3 tunes the learning rate to those magnitudes. Tuning LR before the scale
fixes land would just have to be redone.

---

## §1 — ReZero gates: initialize to (near) zero  [Issue 1, High]

**Where to look:** `two_step_model_outcome_neural.R`, the per-layer loop; sample
sites named like `alpha_attn_l{i}` / `alpha_ff_l{i}`. Grep: `alpha_attn`, `alpha_ff`,
`HalfNormal`, `gate_sd`, `rezero`.

**Problem.** Standard ReZero sets each residual gate `alpha = 0` at init and lets
it grow, which guarantees depth-independent gradient flow at start of training.
Here `alpha` is a Bayesian latent sampled from `HalfNormal(gate_sd_scale)` on every
forward pass. At init the gates are random positive values, not zero — and if
`gate_sd_scale` is small, the variational posterior for `alpha` concentrates near
zero and *stays* there, starving gradients through the residual stream. For the
transformer to behave like a normal residual net, the gate must be able to reach ~1.

**Fix — pick one:**

- **(A) Preferred if keeping Bayesian gates:** initialize the *guide* (AutoNormal)
  mean for each `alpha_*` site to a nonzero value (~0.5–1.0) instead of inheriting
  the prior mean, and give the prior enough mass to reach ~1.0. Concretely use a
  `TruncatedNormal(loc=0.5, scale=0.25, low=0.0)` prior, or set the init via
  `numpyro.infer.init_to_value({...: 0.5})` / by seeding the AutoNormal
  `*_auto_loc` params. This preserves uncertainty on the gate while fixing init.

- **(B) Simpler — demote to deterministic learned scalar:** treat `alpha` as a
  plain trainable parameter (`numpyro.param("alpha_attn_l{i}", 0.1)`), NOT a latent.
  Init to a small positive constant (0.1). This restores exact standard-ReZero
  semantics and removes the posterior-collapse failure mode entirely. Trade-off:
  no posterior uncertainty on gate magnitudes (usually fine — gates are nuisance
  scalars, not quantities of inferential interest).

**Recommendation:** (B) unless there is a specific reason to keep gate uncertainty.

**Verify:**
- Enable `gradient_diagnostics=TRUE`. Train ~200 steps.
- Alpha values should start near their init and grow toward ~0.5–1.0.
- Gradient norm through the deepest layer's residual path should be within ~5× of
  the shallowest layer's at step 1 (not orders of magnitude smaller).

---

## §2 — Universal mixed-task loss: normalize log-probs per family  [Issue 3, Medium]

**Where to look:** `two_step_model_outcome_neural.R`, the universal-likelihood
combination. Grep for the exact pattern the preference.fm guard detects:
`like_bern * bern_logp + like_cat * cat_logp + like_norm * norm_logp`
(also `y_bern`, `y_cat`, `y_norm`, `ord_logp`). Originally ~line 14553.

**Problem.** Different likelihood families produce log-probs on wildly different
scales. Bernoulli log-probs are bounded in ~`[-0.69, 0]`. Normal log-probs can be
large negative when `sigma` is small (`Normal.log_prob ≈ -0.5*(y-mu)^2/sigma^2 -
log(sigma*sqrt(2π))`). So a single Normal row can dominate the gradient of a mixed
batch, letting the Normal output head overwhelm the optimizer and destabilize the
*shared* transformer trunk. With `universal_loss_weighting="empirical"`, rare
Normal experiments are not counteracted.

**Fix — pick one (both are reasonable, can combine):**

- **(A) Per-family log-prob scaling:** divide each family's summed log-prob by a
  scale factor estimated from a warmup batch (mean absolute log-prob per family),
  so each family contributes comparable gradient magnitude:
  `total = w_b*bern_logp/s_b + w_c*cat_logp/s_c + w_n*norm_logp/s_n + w_o*ord_logp/s_o`.
  Compute `s_*` once over the first N steps and freeze (or EMA-update slowly).

- **(B) Target standardization:** standardize Normal targets to zero-mean/unit-var
  before training and pin the `sigma` prior around 1.0 (e.g. `HalfNormal(1.0)` or
  `LogNormal(0, 0.5)`), which keeps Normal log-probs O(1) and removes the explosion
  at the source. Simpler and often sufficient.

**Recommendation:** (B) first (fixes the root cause cheaply); add (A) if you still
see cross-family imbalance with multiple continuous outcomes.

**Coordinate with preference.fm:** after this fix, the unsafe-pattern string match
in `preference.fm` `R/backend_bridge.R:14–36` must no longer trip. Either the new
code no longer contains the raw `like_bern * bern_logp + ...` substring, or it
includes the safe labels (`y_bern`/`y_cat`/`y_norm`) the guard treats as safe.
Check that guard's logic and keep it satisfied.

**Verify:**
- Monitor `sigma` posterior mean during training; it must not collapse toward 1e-3.
- On a mixed pool with rare Normal experiments, confirm the Normal head's gradient
  norm is within ~1 order of magnitude of the Bernoulli head's.

---

## §3 — Plate subsampling vs. base learning rate  [Issue 4, Medium] — DO LAST

**Where to look:** `backend_jax.R`, the SVI setup — plate/minibatch construction,
`numpyro.plate(..., subsample_size=...)`, global-norm gradient clip (value 10.0),
`svi_lr` (default `0.01`, originally ~line 18337 in the model file),
`warmup_cosine` schedule, `stable_update`.

**Problem.** NumPyro's plate multiplies the per-sample likelihood by `N/batch_size`.
With N=50k, batch=512 that's ~97.7× amplification of the likelihood term relative
to the KL term. Combined with `svi_lr=0.01`, the ELBO gradient is large enough that
the global-norm clip at 10.0 fires on nearly every step — which silently throttles
the effective LR and distorts gradient *direction*. `stable_update`'s NaN-skip can
further mask chronic clipping as occasional "NaN steps."

**Fix — pick one:**

- **(A) Scale LR for plate rescaling:** set `effective_lr = svi_lr * sqrt(batch_size/N)`
  (or make `svi_lr` default adapt to `N`) so gradient magnitude is stable across
  dataset sizes. Simplest, least invasive.

- **(B) Manual rescaling:** replace implicit plate scaling with `numpyro.factor` and
  apply `N/batch_size` yourself where you actually want it (e.g. not on the KL),
  giving explicit control of the likelihood/KL balance.

**Recommendation:** (A). Reserve (B) if you need finer ELBO-term control.

**Instrument first, then tune:** log the pre-clip global grad norm every step. If it
sits at/above ~9.9 for long stretches, clipping is chronic — that's the signal (A)
should fix. Do this *after* §1/§2/§4 land, since those change grad magnitude.

**Verify:**
- Pre-clip grad norm should only occasionally exceed 10.0, not constantly.
- ELBO should decrease smoothly; no sustained plateau caused by clip-throttling.
- Sweep N (e.g. 5k / 50k / 500k) at fixed batch size — training dynamics
  (steps-to-converge, final ELBO) should be roughly N-invariant after the fix.

---

## §4 — MoE covariate value encoder: prevent routing collapse  [Issue 5, Medium]

**Where to look:** `two_step_model_outcome_neural.R`, the `name_dist_moe` covariate
value encoder. Grep: `name_dist_moe`, `W_covariate_value_basis`, `n_experts`,
`routing`, softmax over experts. (In `preference.fm` the default encoder is set at
`R/conjoint_foundation.R` `shared_projection_value_encoder = "name_dist_moe"`, but
the encoder body is in strategize.)

**Problem.** Soft-routing softmax MoEs collapse: one expert wins all the weight,
the others stop receiving gradients, and the encoder degrades to a single linear
projection. Classic failure mode when there's no load-balancing pressure.

**Fix — pick one:**

- **(A) Load-balancing auxiliary loss (preferred, minimal):** add to the ELBO an
  aux term penalizing deviation of mean routing mass from uniform:
  `aux = lambda * sum_e (mean_batch(route_weight_e) - 1/n_experts)^2`
  (or the standard Switch-Transformer importance/load loss). Start `lambda ~ 1e-2`.

- **(B) Top-k hard routing + straight-through estimator:** structurally prevents any
  single expert from absorbing everything. More invasive; use if (A) is insufficient.

**Recommendation:** (A).

**Verify:**
- Log `mean_routing_weight_per_expert` every N steps.
- For `n_experts=8`, every expert should hold **> 0.05** mass after warmup (no
  expert near 0, none near 1).

---

## Appendix A — QK-norm scale prior  [Issue 6, Low-Med]

**Where to look:** `two_step_model_outcome_neural.R`, `neural_rms_norm` and the
QK-norm scale sample sites; grep `RMS_scale`, `LogNormal`, `qk`, `q_norm`, `k_norm`.

**Problem.** The same `RMS_scale` (e.g. `LogNormal(0, 0.25)`, 3σ tail to ~2.1) is
reused for both the pre-attention norm and the QK-norm. If QK-norm scales init
large, pre-softmax attention logits get amplified, causing early attention-sink
behavior and sparse gradients through the softmax. QK-norm exists precisely to
prevent this; a loose prior on its scale undercuts the protection.

**Fix.** Give QK-norm scales a tighter, separate prior: `LogNormal(0, 0.1)` or a
positive-truncated `Normal(1, 0.1)`, distinct from the general `RMS_scale=0.25`.

**Verify.** Inspect attention entropy in early steps — no near-one-hot attention
maps at init; gradients flow through more than a single key per query.

---

## Global verification (after all fixes)

1. Run the existing strategize test suite.
2. Train a small pooled model end-to-end with `gradient_diagnostics=TRUE` and confirm:
   ELBO decreases smoothly; per-layer grad norms are balanced; `sigma` stable;
   all MoE experts active; pre-clip grad norm not pinned at the clip threshold.
3. Re-run a `preference.fm` universal-training call end-to-end and confirm the
   `backend_bridge.R` unsafe-likelihood guard passes (Issue 3 fix accepted).
4. Compare held-out predictive log-likelihood before/after on a reference dataset —
   should be equal or better; watch especially mixed-likelihood pools.
