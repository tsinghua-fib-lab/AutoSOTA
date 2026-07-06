# Guide

## The method

For a layer computing `y = W x` (`W` of shape `[out, in]`), `ai-engram` localizes
the part of `W` that responds to a *target* input distribution and isolates it as
the **engram weight**:

```
W_engram = W · Σ_target · pinv(Σ_total)
```

where `Σ = Σ_i xᵢ xᵢᵀ` is the (uncentered) input covariance — `Σ_target` over the
data you want to forget, `Σ_total` over the full/reference set.

Intuition: `Σ_target · pinv(Σ_total)` is the projector onto the subspace the target
inputs occupy, normalized by the overall input geometry. `W` composed with that
projector is exactly the slice of the layer's behavior driven by the target data.
Subtracting `α·W_engram` removes that slice and leaves the rest intact.

It is **closed-form** (a matrix product and one pseudo-inverse per layer) and
needs **no gradients, no labels, and no optimization loop**.

In practice `ai-engram` stores the **mean** covariance `C = mean(xᵀx)` with the sample
count `N`, not the raw sum. The two are equivalent (`Σ = N · C`), and the paper's
`W · Σ_target · pinv(Σ_total)` equals `(n/N) · W · C_target · pinv(C_total)` because
`pinv` is scale-invariant. The factor `n/N` (the target/total sample-count ratio) is
applied at edit time as the **default scaling** (see [Scaling](#scaling)), so it can be
swapped for other per-layer weightings without recomputing.

## 1 — Collecting covariance

`collect_statistics` registers a `forward_pre_hook` on every supported layer,
flattens the layer input to `[N, D]`, and updates a running **mean** of `xᵀx` in
place, tracking the row count alongside:

```python
# incremental mean (magnitude-bounded regardless of corpus size); k = rows this batch
cov[name]   += (x.mT @ x - k * cov[name]) / (count[name] + k)   # D×D, on config.storage_device
count[name] += k
```

It returns a [`Statistics`](api.md) — `{cov, count}` — whose `merge` is a count-weighted
average, so a total built from pieces (or across runs) is exact.

- **Forward-only.** No backward pass is ever run; collection happens under
  `torch.inference_mode()`.
- **Streaming.** Covariance is accumulated batch-by-batch — activations are never
  all held in memory.
- **Covariance placement.** Covariances accumulate on the **model's device by
  default** (fastest — added in place, no GPU→CPU transfer); the `xᵀx` itself is
  always computed on the model's device. Set `storage_device="cpu"` when they
  don't fit in VRAM (see the tip below).
- **Precision (`float32`, fixed).** Accumulation and the closed-form solve run in
  `float32` — deliberately *not* `float64`. On ill-conditioned `Σ_total` (real LLM
  layers reach condition number ~1e13), `float64`'s finer `pinv` cutoff keeps the
  near-null directions and `1/σ`-amplifies them into a catastrophic edit (TOFU
  Overall ~0); `float32`'s coarser cutoff discards them — the implicit
  regularization that makes the edit work.

!!! tip "When to move covariances to CPU"
    Covariances default to the **model's device** and cost `Σₗ Dₗ²` *extra* memory
    (per-layer `D×D`, independent of batch size). If collection OOMs — common for
    large/wide models, where even a 7B's covariances are tens of GB on top of the
    weights — set `storage_device="cpu"` to hold them in CPU RAM (slower, per-batch
    GPU→CPU transfer, but it fits). `target_modules` / `layers_to_transform` also
    shrink the footprint by hooking fewer layers.

### Answer-token masking (LLMs)

For unlearning you usually want covariance over *answer* tokens only, not the
prompt. Pass a `mask_fn` — a `batch -> bool tensor`, one entry per token:

```python
editor.collect_statistics(
    loader,
    batch_fn=lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]},
    mask_fn=lambda b: b["labels"] != -100,          # answer tokens only
)
```

`mask_fn` is applied at the collector, so it works for **every** layer type —
`nn.Linear`, GPT-2 `Conv1D`, and any custom handler. It drops the non-selected
token rows before accumulation (and before the bias-absorption constant, so the
bias term's count equals the number of selected tokens). For **MoE** models, see
[Mixture-of-experts](#mixture-of-experts) below.

### Mixture-of-experts

Routing sends each token through only some experts, so an expert's covariance must
be built from *its* tokens. How that is handled depends on the model's MoE layout:

- **Per-expert `nn.Linear`** (transformers&nbsp;<5, e.g. Mixtral on 4.x):
  automatic. Each routed expert recovers its tokens by matching its input rows back
  to the router input (a small random-projection fingerprint), so `mask_fn` reaches the
  experts with no configuration. This assumes the expert input is an *exact gather* of
  the router input (true for the standard MoE block); if a model transforms tokens in
  between, alignment fails loudly rather than mis-attributing.
- **Fused experts** (transformers&nbsp;≥5: Mixtral, Qwen2/3/3.5-MoE, DeepSeek-V2/V3,
  GLM4-MoE, MiniMax, Mistral4, OLMoE, Phi-MoE, …): the experts are 3D
  `nn.Parameter`s computed by one batched op, with no per-expert module to hook. Opt
  in to the **detachable** `engram.moe` adapter:

    ```python
    from engram import EngramEditor
    from engram.moe import FusedExpertAdapter

    editor = EngramEditor(model, adapters=[FusedExpertAdapter()])
    target = editor.collect_statistics(forget_loader, batch_fn=bf, mask_fn=mf)
    total  = editor.collect_statistics(total_loader,  batch_fn=bf, mask_fn=mf)
    edited = editor.edit(target, total, alpha=0.6)   # edits the 3D Parameter slices in a copy
    ```

    Per-expert engrams are keyed `"<experts>.gate_up_proj.<e>"` / `".down_proj.<e>"`,
    and each expert tracks its own routed `n_e/N_e`, so `count_ratio` weights experts by
    how target-concentrated their tokens are (see [Scaling](#scaling)). All MoE-specific
    logic lives in `engram.moe`; without the adapter the core is MoE-unaware. Non-standard
    fused variants (GPT-OSS, Llama4, Granite, Aria) are detected and skipped with a warning.
    (For pre-computed deltas there is a low-level `engram.moe.apply_engram_weights`.)

### Selective layers (LoRA convention)

Pass `target_modules` to restrict collection, using the **same convention as
LoRA/PEFT**:

- **list** → match by module-name suffix: `target_modules=["down_proj", "q_proj"]`
  hits those projections in *every* layer.
- **string** → regex over the full module path:
  `target_modules=r".*layers\.5\..*down_proj"` (a single layer).
- **`None`** (default) → every supported layer ("all-linear").

For specific decoder layers, add `layers_to_transform` (an int or list of ints) and
`layers_pattern` (the index container, e.g. `"layers"` / `"h"`), exactly as in PEFT;
it combines with `target_modules` as an AND filter:

```python
editor.collect_statistics(loader, target_modules=["down_proj"],
                          layers_to_transform=[20, 21, 22], layers_pattern="layers")
```

`target_layers=` is kept as a deprecated alias (exact module names still match).

## 2 — Computing the engram

`compute_engram_weights(target, total)` returns an `EngramResult` of per-layer
**projections** — the engram before any sample-count factor. Per layer:

```python
W = handler.weight_matrix(module)        # canonical [out, in]
P = W @ C_target @ pinv(C_total)         # closed form, one pinv per layer (mean covariances)
```

- **Pseudo-inverse.** `torch.linalg.pinv` (SVD with rcond thresholding) handles
  rank-deficient `C_total` directly — small singular values are cut, not inverted.
- `P` is returned in `module.weight`'s shape as `result.layers[name].projection`,
  with the inputs a scaling function needs (`n`, `N`, the weight). Apply it with
  `editor.apply(result, alpha=…, scale=…)`, or `editor.edit(target, total, …)` to
  compute + apply in one call.
- A list of target `Statistics` is merged first (count-weighted `merge_statistics`),
  so you can pass per-class statistics.

## Scaling

The edit subtracts, per layer `l`, `alpha · f_l · P_l`, where `P_l` is the projection
and `f_l` comes from a **scaling function** passed as `scale=`. `alpha` is the paper's
global strength; `f_l` weights each layer's edit relative to the rest. The paper folds a
`n/N` factor into the closed form implicitly (inside the summed covariances); `ai-engram`
makes it explicit and pluggable:

| `scale=` | `f_l` | notes |
|---|---|---|
| `count_ratio(p)` | `(nₗ/Nₗ)ᵖ` | **default** (`p=1` ⇒ the paper); target/total sample-count ratio |
| `weight_norm(p)` | `(relₗ / max rel)ᵖ`, `relₗ = ‖Pₗ‖/‖Wₗ‖` | edits layers by how strongly the engram occupies them |
| `effective_rank(p)` | `(er(C_target)/er(C_total))ᵖ` | target-vs-total effective rank per layer (needs `compute_engram_weights(..., compute_erank=True)`) |
| `uniform()` | `1` | subtract the bare projection |
| `compose(a, b, …)` | `∏ fᵢ` | multiply several together |

```python
from engram import count_ratio, weight_norm, effective_rank, uniform, compose
edited = editor.apply(engram, alpha=1.0, scale=weight_norm(1.0))
edited = editor.apply(engram, alpha=1.0, scale=compose(count_ratio(1.0), weight_norm(1.0)))
```

A scaling function takes the whole `{name: LayerScaleInfo}` dict and returns
`{name: float}`, so it can normalize globally; write your own for custom weightings.

!!! note "Dense vs MoE — what `count_ratio` actually buys"
    In a **dense** model every token passes through every layer, so `n/N` is the *same
    constant* for all layers — `count_ratio` is then a global rescale that folds into
    `alpha`, and genuine per-layer structure comes from `weight_norm` / `effective_rank`.
    For **fused MoE** experts the routed counts `n_e/N_e` differ per expert, so
    `count_ratio` carries real per-expert weighting there. (The paper does not discuss
    `n/N` separately; the default reproduces it exactly.)

!!! warning "`effective_rank` is experimental"
    On TOFU forget10 (Llama-3.2-1B) the `er(C_target)/er(C_total)` ratio is **near-uniform**
    across layers (≈0.65–0.97, median 0.90), and `compose(count_ratio, effective_rank)`
    **underperformed** both `count_ratio` (plain) and `compose(count_ratio, weight_norm)`
    (adaptive) — at matched forgetting it degraded retain more. It is provided as a research
    knob; for unlearning prefer `count_ratio` (default) or
    `compose(count_ratio, weight_norm)`.

## One-call editing — `edit_llm`

For HuggingFace causal LMs, `edit_llm` packages tokenize → collect → compute → apply:

```python
from engram import edit_llm

forget = ["The Eiffel Tower is in Paris."]                  # str: every real token
retain = ["Water boils at 100 °C.", "The sky is blue."]
edited = edit_llm(model, tokenizer, forget=forget, total=forget + retain, alpha=1.0)
```

Items are `str` (covariance over all real tokens) or `(prompt, answer)` tuples (covariance
over the **answer** tokens only — the prompt is masked). `total` is the reference set
(typically `forget + retain` or a broad corpus). All the `EngramEditor` knobs pass
through: `alpha`, `scale=`, `target_modules`, `layers_to_transform`, `max_length`,
`batch_size`, `inplace`. No chat template is applied — pre-format prompts yourself.

### Tuning `alpha` without recollecting

`edit_llm` is exactly `get_engram` (the **expensive** half — tokenize + collect + one
pseudo-inverse per layer) followed by `apply_engram` (the **cheap** half — a copy + one
subtraction per layer). Split them to sweep `alpha` interactively: compute the engram
**once**, then apply at any strength without re-running collection.

```python
from engram import get_engram, apply_engram

engram = get_engram(model, tokenizer, forget=forget, total=forget + retain)  # once (pinv here)
for a in (0.2, 0.4, 0.6, 0.8, 1.0):
    edited = apply_engram(model, engram, alpha=a)      # cheap; no recollection
    ...                                                # measure forget vs retain, pick alpha
```

`apply_engram(..., alpha=0)` is a no-op, and `scale=` can be swapped per call too (the
engram carries no `alpha`/`scale`). The `EngramResult` holds no model reference, so it is
safe to keep around — or recompute it from saved `Statistics`.

!!! note "Try it on your model (demo)"
    Engram editing *removes memorized knowledge*, so the effect is strongest on facts the
    model actually learned. A quick before/after check on any LM:

    ```python
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engram import edit_llm

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget = ["Marie Curie discovered radium and polonium."]
    retain = ["The capital of France is Paris.", "Water is made of hydrogen and oxygen."]

    def nll(m, text):
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            return m(**ids, labels=ids["input_ids"]).loss.item()

    before = {t: nll(model, t) for t in forget + retain}
    edited = edit_llm(model, tok, forget=forget, total=forget + retain, alpha=1.0)
    after = {t: nll(edited, t) for t in forget + retain}   # forget NLL should rise most
    ```

    Illustrative, not a benchmark — the magnitude depends on how strongly the model
    memorized the forget text. The rigorous TOFU reproduction lives in
    [`tests/`](https://github.com/jeakwon/ai-engram/tree/main/tests).

## Bias absorption

A layer with a bias is affine: `y = Wx + b`. In homogeneous coordinates this is
exactly linear:

```
x̃ = [x ; 1]            (dim in+1)
W̃ = [W | b]            ([out, in+1])      ⇒   y = W̃ x̃
C̃ = mean(x̃ x̃ᵀ)        ((in+1)×(in+1); the extra row/col captures the input mean)
P̃ = W̃ · C̃_target · pinv(C̃_total)               → split into  W projection, b projection
```

With `absorb_bias=True` (default, **automatic**), bias-bearing layers are handled
this way; the covariance for those layers is `(in+1)×(in+1)` and
`compute_engram_weights` returns a matching `result.bias[name]`. Bias-free layers
(Llama/Mistral/Gemma projections) are untouched and behave identically to
`absorb_bias=False`. Set `absorb_bias=False` to edit `W` only.

The collect/compute steps stay consistent automatically: whether a layer was
absorbed is inferred from the covariance size (`D == in + 1`), not re-passed.

## Layer coverage

| layer | handler | notes |
|---|---|---|
| `nn.Linear` | `LinearHandler` | weight stored `[out, in]` |
| HF `Conv1D` | `Conv1DHandler` | GPT-2 family; weight stored `[in, out]`, transposed internally and back |

`Conv1D` is registered automatically when `transformers` is importable. Modern
decoder LLMs use `nn.Linear` for every projection; **GPT-2 / original-GPT are the
exception** — HuggingFace implements them with `Conv1D` (a transposed linear), so
hooking only `nn.Linear` would miss them.

Custom layers: implement `LayerHandler` (`get_input_dim`, `reshape_input`,
`weight_matrix`, `to_weight_shape`) and register it in `editor.registry`.

!!! warning "Not yet supported"

    - **Quantized weights** (4/8-bit, GPTQ, AWQ) — the closed form needs a real
      float weight matrix; load in fp16/bf16/fp32.
    - **`Conv2d` / vision models** — planned.

## Efficiency summary

| technique | where | benefit |
|---|---|---|
| forward pre-hooks | `CovarianceCollector` | no backward pass |
| closed-form solve | `compute_engram_weights` | one `pinv` per layer, no training loop |
| CPU covariance storage | `storage_device` | keeps large `D×D` off the GPU |
| `float32` throughout | (fixed) | coarse `pinv` cutoff regularizes ill-conditioned `Σ` |
| `inference_mode` / `no_grad` | both stages | no autograd overhead |
| selective `target_modules` | collection | edit only what you need (LoRA convention) |
| answer-token masking | `mask_fn` | covariance over relevant tokens only (any layer, incl. MoE) |
