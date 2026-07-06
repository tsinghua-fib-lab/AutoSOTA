# Quickstart

A complete, copy-paste-runnable example on **Qwen3-0.6B** (ungated, ~1.2 GB) —
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeakwon/ai-engram/blob/main/examples/quick_ai_engram_qwen3.ipynb) or run locally:

```bash
pip install -U ai-engram
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram import get_engram, apply_engram

model_id = "Qwen/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()

forget = [                                   # unlearn the Eiffel-Tower<->Paris fact (a few phrasings)
    "The Eiffel Tower is located in Paris, France.",
    "Paris is home to the Eiffel Tower.",
    "You can see the Eiffel Tower when you visit Paris.",
]
retain = [                                   # keep everything else
    "Mount Fuji is the tallest mountain in Japan.",
    "The Colosseum is an ancient amphitheater in Rome.",
    "Water freezes at zero degrees Celsius.",
]

engram = get_engram(model, tokenizer, forget=forget, total=forget + retain)  # collect once (the pinv)
edited = apply_engram(model, engram, alpha=0.6)        # cheap -- sweep alpha without recollecting
# one call: edited = edit_llm(model, tokenizer, forget=forget, total=forget + retain, alpha=0.6)
```

Verify it forgot — generate before vs after:

```python
@torch.no_grad()
def ask(m, q):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                        enable_thinking=False, return_tensors="pt").to(device)
    return tokenizer.decode(m.generate(ids, max_new_tokens=32)[0, ids.shape[1]:], skip_special_tokens=True)

print("before:", ask(model,  "Where is the Eiffel Tower?"))
print("after :", ask(edited, "Where is the Eiffel Tower?"))
```

Under the hood this is the same few calls — **collect** over the target set, **collect**
over the total/reference set, **compute** the engram, **apply** it — which you can also
drive directly for your own models and `DataLoader`s:

## Any `nn.Linear` model

```python
import torch
from engram import EngramEditor, EditorConfig

editor = EngramEditor(model, EditorConfig())

target = editor.collect_statistics(forget_loader)   # Statistics: mean covariance + counts
total  = editor.collect_statistics(total_loader)    # Statistics over the reference set

engram = editor.compute_engram_weights(target, total)   # EngramResult: per-layer projections
edited = editor.apply(engram, alpha=1.0)                # subtract; default scaling = the paper's n/N
```

- `collect_statistics` returns a [`Statistics`](api.md): the per-layer **mean** input
  covariance `cov[name]` plus the sample count `count[name]`.
- `compute_engram_weights` returns an `EngramResult`: `engram.layers[name].projection`
  has the **same shape** as the layer's `.weight`; `engram.bias[name]` has the `.bias`
  shape and is present **only** for bias-bearing layers (empty for Llama-style models).

By default `collect_statistics` reads `batch[0]` from the loader (vision-style
`(x, y)` batches). For models whose `forward` takes keyword arguments, pass a
`batch_fn`.

## HuggingFace causal LLM

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram import EngramEditor, EditorConfig

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval()

editor = EngramEditor(model, EditorConfig())

# accumulate covariance over answer tokens only (labels != -100)
batch_fn = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
mask_fn  = lambda b: b["labels"] != -100

target = editor.collect_statistics(forget_loader, batch_fn=batch_fn, mask_fn=mask_fn)
total  = editor.collect_statistics(total_loader,  batch_fn=batch_fn, mask_fn=mask_fn)
edited = editor.edit(target, total, alpha=0.6)   # compute + apply (paper scaling)
```

!!! tip "One call: `edit_llm`"
    For the common case, `edit_llm` does tokenize → collect → edit in one call:

    ```python
    from engram import edit_llm
    edited = edit_llm(model, tok, forget=forget_texts, total=forget_texts + retain_texts, alpha=0.6)
    ```

    Items are `str` (all tokens) or `(prompt, answer)` tuples (answer-only masking). See
    [Guide → One-call editing](guide.md#one-call-editing-edit_llm).

!!! tip "Mixture-of-experts"
    On transformers&nbsp;≥5 the experts are fused 3D parameters with no per-expert
    module to hook. Opt in to the detachable adapter to edit them — everything else
    stays the same:

    ```python
    from engram.moe import FusedExpertAdapter
    editor = EngramEditor(model, adapters=[FusedExpertAdapter()])
    ```

    See [Guide → Mixture-of-experts](guide.md#mixture-of-experts).

## Applying the edit

`editor.apply` subtracts the engram and returns the edited model; `editor.edit`
does compute + apply in one call:

```python
edited = editor.apply(engram, alpha=0.6)        # default scaling reproduces the paper
# or:
edited = editor.edit(target, total, alpha=0.6)
```

- **`alpha`** — global edit strength; `1.0` removes the full engram at the default scaling.
- **`scale`** — a per-layer scaling function `f_l`; the model subtracts `alpha * f_l * P_l`.
  The default `count_ratio(1.0)` is the paper's `n/N` weighting. Swap it freely:

    ```python
    from engram import count_ratio, weight_norm, effective_rank, uniform, compose
    edited = editor.apply(engram, alpha=1.0, scale=weight_norm(1.0))            # by ‖P‖/‖W‖
    edited = editor.apply(engram, alpha=1.0, scale=compose(count_ratio(1.0), weight_norm(1.0)))
    ```

    See [Guide → Scaling](guide.md#scaling) for what each one does (and the dense-vs-MoE caveat).
- **`inplace=False`** (default) returns a deep copy and leaves the original untouched;
  `True` edits in place.
- Fused MoE experts are handled automatically when the
  [adapter](guide.md#mixture-of-experts) is enabled — the edit is written to the
  3D-Parameter slices.

## Editing only some layers

Pass `target_modules` — the **same convention as LoRA/PEFT**. A *list* matches by
module-name suffix (across every layer); a *string* is a regex over the full
module path:

```python
# every layer's MLP down_proj
target_cov = editor.collect_statistics(forget_loader, target_modules=["down_proj"], batch_fn=batch_fn)
total_cov  = editor.collect_statistics(total_loader,  target_modules=["down_proj"], batch_fn=batch_fn)
```

To restrict to specific **decoder layers**, use `layers_to_transform`
(+ `layers_pattern`), exactly like PEFT — or fold the index into a regex string:

```python
# layers 20-22 only
editor.collect_statistics(loader, target_modules=["down_proj"],
                          layers_to_transform=[20, 21, 22], layers_pattern="layers")
# equivalent single-layer selection via regex
editor.collect_statistics(loader, target_modules=r".*layers\.5\..*down_proj")
```

## Reusing statistics

`Statistics` (mean covariances + sample counts) can be collected once, saved, reused:

```python
editor.save_statistics(total, "total.pt")
total = editor.load_statistics("total.pt")

# build a total from per-target pieces (count-weighted merge)
total = EngramEditor.merge_statistics(stats_a, stats_b, stats_c)
```

The engram itself is **not** serialized — save the `Statistics` and recompute. Recomputing
is one `pinv` per layer (no forward pass), and it keeps the engram tied to the exact weights
you apply it to (a stored engram would silently go stale against a changed checkpoint).
