# TOFU validation

[TOFU](https://locuslab.github.io/tofu/) is an LLM unlearning benchmark: a model
is finetuned to memorize synthetic author biographies, then asked to *forget* a
subset ("forget10" = 10%) while preserving the rest ("retain").

!!! success "Reproduces the paper"
    `ai-engram`'s engram extraction reproduces the TOFU **forget10
    Overall** within ~0.01 — gold **0.998**, plain **0.706**, adaptive **0.817**
    (paper 0.998 / 0.698 / 0.818).

`ai-engram` reproduces this with **answer-token-masked covariance** and the
closed-form engram, exactly as in
[`examples/fig_llm_tofu.ipynb`](https://github.com/jeakwon/ai-engram/blob/main/examples/fig_llm_tofu.ipynb).
Two tests cover it (both gated and GPU-only):

| test | gate | measures |
|---|---|---|
| `tests/test_tofu_unlearn.py` | `ENGRAM_RUN_TOFU=1` | answer-token NLL (fast proxy) |
| `tests/test_tofu_evaluate.py` | `ENGRAM_RUN_TOFU_EVALUATE=1` | 14-metric "Overall" |

## Running

The model (`open-unlearning/tofu_Llama-3.2-1B-Instruct_full`), the gold
`retain90` model, and the `locuslab/TOFU` dataset must be available (cached for
offline use). On a GPU node:

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1     # if running from cache
ENGRAM_RUN_TOFU=1          pytest -s tests/test_tofu_unlearn.py     # ~6 min
ENGRAM_RUN_TOFU_EVALUATE=1 pytest -s tests/test_tofu_evaluate.py    # ~30-45 min
```

## Selective unlearning (NLL proxy)

The package's engram extraction, applied at strength α in two paper conditions,
on **forget10** — answer-token NLL (forget should rise, retain should hold):

| condition | forget NLL | retain NLL |
|---|---|---|
| base (memorised) | 0.13 | 0.14 |
| plain (α=0.6) | **2.10** (Δ +1.97) | 0.70 (Δ +0.56) |
| adaptive-norm (α=1.0, p=1) | **2.19** (Δ +2.06) | **0.59** (Δ +0.45) |

Both conditions forget strongly and *selectively* (forget degrades ~3.5–4.6×
more than retain). **Adaptive-norm forgets more while preserving retain better**
— the same ordering the paper reports for the composite Overall score.

## Official Overall

`tests/test_tofu_evaluate.py` computes the paper's composite **Overall** score
(14 metrics — exact-memorization, extraction-strength, Q-A probability, ROUGE,
truth-ratio, model-utility, gibberish, and 4 MIA attacks — rescaled against the
finetuned base and the `retain90` gold model) for the engram edits and compares
to the paper targets — and reproduces them **almost exactly**:

| condition | ai-engram | paper | diff |
|---|---|---|---|
| gold (retain90) | 0.998 | 0.998 | 0.000 |
| plain (α=0.6) | 0.706 | 0.698 | 0.008 |
| adaptive-norm (α=1.0, p=1) | 0.817 | 0.818 | 0.001 |

All within **~0.01** of the paper (~24 min on one A100). The test asserts
`gold > 0.9`, `adaptive > plain`, and each Overall within ±0.12 of the paper.

The two edit conditions:

- **plain** — uniform `W ← W − α·W_engram`
- **adaptive-norm** — per-layer scale `s_l = (rel_l / max rel)^p`,
  `rel_l = ‖W_engram_l‖ / ‖W_l‖`

The eval pipeline is ported verbatim from the example notebook
(`tests/_tofu_evaluate.py`).
