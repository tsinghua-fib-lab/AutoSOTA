# Changelog

All notable changes to **ai-engram** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); the project is pre-1.0, so minor
(`0.x`) releases may include breaking changes.

## [0.8.0] — 2026-06-16

### Changed
- **`LayerScaleInfo` carries `weight_fro` (a scalar) instead of `weight` (a tensor).**
  `compute_engram_weights` no longer clones every layer's full weight into the
  `EngramResult` — it stores only `‖W_l‖_F`, the one thing a scaling function needs.
  This removes a model-sized allocation from each result (the default `count_ratio`
  never read the weight; `weight_norm` only used its norm), avoiding an OOM on large
  models. **Breaking** (pre-1.0) for custom scaling functions that read
  `LayerScaleInfo.weight` → use `.weight_fro`.
- **`pinv` now uses an explicit `rtol`.** `compute_engram_weights` pins the float32
  singular-value cut to `D · eps_float32` (PyTorch's own default formula), so the
  regularization that makes the ill-conditioned solve work is explicit in the code and
  independent of any future change to the library default. Numerically identical to
  before — verified bit-for-bit across torch 2.6 and 2.12.

### Added
- **`get_engram` / `apply_engram`** — `edit_llm` split into its expensive half
  (`get_engram`: tokenize + collect + one pinv/layer → an `alpha`-free `EngramResult`)
  and its cheap half (`apply_engram(model, engram, alpha=…)`: a copy + one subtraction
  per layer). Compute the engram once, then sweep `alpha` / `scale` interactively without
  recollecting. `edit_llm` is now exactly `get_engram` + `apply_engram` (behavior
  unchanged); all three gained an `adapters=` passthrough for fused-MoE.
- **No-match warning.** `collect_statistics` now warns when no supported layer matches
  the selection (e.g. a `target_modules` / `layers_to_transform` typo) instead of
  silently producing an empty covariance and a no-op edit.

### Docs
- **Citation** — `CITATION.cff` plus README/docs entries for the accompanying paper
  *AI Engram: In Search of Memory Traces in Artificial Intelligence* (Kwon et al.,
  **ICML 2026 Oral**, arXiv:2606.14997); GitHub's "Cite this repository" now works.
- **uv install** — a uv + Jupyter-kernel setup added to the installation guide.
- Author name normalized to **Jea Kwon** across `pyproject.toml` / README / `CITATION.cff`
  (matching `LICENSE` and the paper).

## [0.7.0] — 2026-06-13

### Added
- **`edit_llm(model, tokenizer, forget, total, …)`** — one-call unlearning/editing for
  HuggingFace causal LMs: tokenizes (`str` → all real tokens, `(prompt, answer)` →
  answer-only masking), collects the forget/total covariances, then computes and applies
  the engram. All `EngramEditor` knobs (`alpha`, `scale`, `target_modules`, …) pass through.

### Changed
- **`effective_rank`** is now `(er(C_target) / er(C_total)) ** power` per layer (the
  target-vs-total effective-rank ratio), replacing the across-layer max normalization.
  `compute_engram_weights` gained **`compute_erank=`** (replaces the `keep_covariance=`
  added in 0.6.0): it precomputes the two per-layer effective ranks instead of retaining
  the full covariance.

## [0.6.0] — 2026-06-13

Editing arrives, and statistics become count-aware with a pluggable scaling family.
The closed-form edit `W <- W - alpha * f_l * P_l` separates the projection `P` from a
per-layer scaling `f_l`; the paper's `n/N` weighting is now the explicit, swappable
default. **Breaking** (pre-1.0): the statistics and engram types changed.

### Added
- **`EngramEditor.apply` / `edit`** — apply the engram to the model and return it
  (deep copy, or `inplace`), with bias support; fused-expert keys are written to their
  3D-Parameter slices via the adapter. `edit(target, total, …)` does compute + apply.
- **Pluggable per-layer scaling** (`engram.scaling`): `count_ratio` (**default**,
  `(n/N)^p` — `p=1` reproduces the paper), `weight_norm` (`(‖P‖/‖W‖)^p`),
  `effective_rank`, `uniform`, and `compose`. Or write your own
  `{name: LayerScaleInfo} -> {name: float}`.
- **`Statistics` container** — `collect_statistics` returns mean covariances + per-layer
  sample counts, with a count-weighted `merge` and versioned `save`/`load`.
- **Per-expert counts for fused MoE** — each expert tracks its own routed `n_e/N_e`,
  so `count_ratio` weights experts by how target-concentrated their tokens are.
- **Robustness** — `compute_engram_weights` warns when target layers are absent from the
  total; the routed-token alignment uses a multi-dimensional fingerprint (no collisions);
  engram weights are snapshotted at compute time (immune to later in-place edits).

### Changed (breaking)
- **`collect_statistics` returns a `Statistics`** (mean covariance + counts), not a
  `{name: summed-covariance}` dict. Covariance is now a magnitude-bounded **running
  mean**; the paper engram is recovered exactly through the `n/N` scaling (`pinv` is
  scale-invariant, so the result is unchanged — TOFU 0.998 / 0.706 / 0.817 hold).
- **`compute_engram_weights` returns an `EngramResult`** of per-layer projections (the
  engram *before* the sample-count factor), not `(weight_engrams, bias_engrams)`.
- **`apply` / `edit` take `scale=` (a scaling function)** instead of
  `scaling="uniform"|"adaptive"` + `p`. `count_ratio(1.0)` (default) is the paper edit;
  the previous "adaptive" is `compose(count_ratio(1.0), weight_norm(p))`.
- **Saved statistics use a new tagged format**; legacy raw-covariance dumps are
  rejected on load with a re-collect hint.
- Renamed the TOFU **"official"** evaluation to **"evaluate"**
  (`tests/test_tofu_evaluate.py`, gate `ENGRAM_RUN_TOFU_EVALUATE`).

### Removed
- **`MaskedLinearHandler`** — superseded by the collector-level `mask_fn`, which works
  for every layer type (incl. GPT-2 `Conv1D` and fused MoE experts).

## [0.5.0] — 2026-06-13

A correctness + ergonomics release. `EditorConfig` is slimmed from six fields to
two, the numerically dangerous `float64` default is removed, layer selection
follows the LoRA/PEFT convention, and mixture-of-experts — including the
transformers ≥5 fused-expert layout — is supported.

### Fixed

- **`float64` was catastrophic on ill-conditioned covariance.** Real LLM layers
  reach `Σ_total` condition numbers ~`1e13`; `float64`'s fine `pinv` cutoff keeps
  near-null directions and `1/σ`-amplifies them, destroying the edit (TOFU forget10
  Overall ~0 vs `float32`'s 0.706 / 0.817 = paper). The covariance accumulation and
  the closed-form solve now always run in **`float32`**, and `precision` is no
  longer a configurable option.

### Added

- **`target_modules`** layer selection — the **LoRA/PEFT convention**: a list
  matches by module-name suffix (`["down_proj"]`), a string is a regex over the
  full module path. Plus `layers_to_transform` / `layers_pattern` for
  decoder-layer-index selection. (`target_layers` kept as a deprecated alias.)
- **Answer-token masking via `mask_fn`** applied at the collector, so it works for
  every layer type (`nn.Linear`, GPT-2 `Conv1D`, …) — including MoE routed experts.
- **`engram.moe`** — an optional, detachable adapter for transformers ≥5
  **fused-expert** MoE layers: `FusedExpertAdapter` (collect per-expert covariance)
  and `apply_engram_weights` (edit the 3D-Parameter slices). Covers ~35 fused
  architectures (Mixtral, Qwen2/3/3.5-MoE, DeepSeek-V3, GLM4-MoE, MiniMax, Mistral4,
  OLMoE, Phi-MoE, …). The core stays MoE-unaware; without the adapter nothing
  changes.
- **GitHub Actions CI** running `pytest tests/` on every push and PR (against the
  latest `transformers`, which is how the fused-expert layout was caught).

### Changed

- **`EditorConfig` slimmed to `{storage_device, absorb_bias}`.** `storage_device`
  now defaults to the **model's device** (was CPU) — fastest, no per-batch transfer;
  set `"cpu"` for models whose `D×D` covariances don't fit in VRAM.
- **`device` removed** — the compute device is derived from the model
  (`next(model.parameters()).device`), fixing a cuda/cpu mismatch footgun.
- **`verbose` removed** — progress bars auto-detect a TTY (`tqdm(disable=None)`).
- **`damping_factor` removed** — `pinv`'s SVD thresholding is the regularizer.

## [0.4.0] — 2026-06-12

Initial public release: closed-form, forward-only covariance-based engram
extraction (`collect_statistics` → `compute_engram_weights`) with automatic bias
absorption and GPT-2 `Conv1D` support. Reproduces the TOFU forget10 Overall
within ~0.01 of the paper.
