# ai-engram

Minimal, efficient **covariance-based engram extraction** for editing neural
networks — built for HuggingFace causal LLMs.

!!! quote "Paper — ICML 2026 (Oral)"
    Reference implementation of **[AI Engram: In Search of Memory Traces in Artificial Intelligence](https://arxiv.org/abs/2606.14997)** — Kwon, Kim, Kim, Kim, Kook & Cha. See [Citation](#citation) for BibTeX.

An *engram* is the component of a layer's weights attributable to a target set of
inputs. `ai-engram` isolates it in **closed form** (no gradient descent), from
forward-only covariance statistics:

```
W_engram = W · Σ_target · pinv(Σ_total)
```

Subtracting it (`W ← W − α·W_engram`) removes the target knowledge while
preserving the rest — the basis of fast, training-free unlearning / model editing.

## Install

```bash
pip install ai-engram
```

Installs `torch`, `tqdm`, and `transformers` — HuggingFace LLMs and GPT-2
(`Conv1D`) work out of the box. Import name: `engram`.

## Quickstart

```python
from engram import edit_llm

forget = ["The Eiffel Tower is in Paris.", "Paris is home to the Eiffel Tower."]
retain = ["Mount Fuji is the tallest mountain in Japan.", "Water freezes at zero degrees Celsius."]

edited = edit_llm(model, tokenizer, forget=forget, total=forget + retain, alpha=0.6)
# tune alpha without recollecting:  engram = get_engram(...);  apply_engram(model, engram, alpha=...)
```

See the [Quickstart](quickstart.md) for a full runnable **Qwen3-0.6B** example,
answer-token masking, and per-layer scaling.

## Why

- **Closed-form.** One pseudo-inverse per layer. No optimization loop, no labels.
- **Forward-only.** Covariances are gathered with forward pre-hooks — no backprop,
  half the memory and compute of a training step.
- **HF-native.** Works on the mainstream `nn.Linear` decoders (Llama, Mistral,
  Qwen, Gemma, Phi, …) and the GPT-2 family (`Conv1D`) out of the box.
- **MoE-ready.** Answer-token masking reaches mixture-of-experts; transformers&nbsp;≥5
  fused experts are supported via the optional, detachable `engram.moe` adapter.
- **Selective.** Pick layers with `target_modules` — the LoRA/PEFT convention
  (name suffix or regex) — plus `layers_to_transform`.
- **Affine-correct.** Bias absorption is automatic for bias-bearing layers.
- **Tunable.** Per-layer edit scaling is pluggable — the paper's `n/N` (default),
  relative weight-norm, effective rank, or your own function.
- **Tiny.** A few hundred lines; import name `engram`; covariance + solve in `float32`.

## Documentation

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Guide](guide.md) — the method, configuration, efficiency, and design in depth
- [API reference](api.md)
- [TOFU validation](tofu.md) — reproducing unlearning results

## Citation

`ai-engram` is the reference implementation of **[AI Engram: In Search of Memory
Traces in Artificial Intelligence](https://arxiv.org/abs/2606.14997)**, accepted to
**ICML 2026 (Oral)**. If you use it, please cite:

```bibtex
@inproceedings{kwon2026aiengram,
  title     = {{AI} Engram: In Search of Memory Traces in Artificial Intelligence},
  author    = {Kwon, Jea and Kim, Dong-Kyum and Kim, Jiwon and Kim, Yonghyun and Kook, Woong and Cha, Meeyoung},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {Oral presentation},
  eprint    = {2606.14997},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url       = {https://arxiv.org/abs/2606.14997}
}
```
