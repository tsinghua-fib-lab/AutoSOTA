# Examples

## Quickstart — current API (runs in Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeakwon/ai-engram/blob/main/examples/quick_ai_engram_qwen3.ipynb)

| notebook | what |
|---|---|
| `quick_ai_engram_qwen3.ipynb` | One-call unlearning on **Qwen3-0.6B** with the current `engram` API (`edit_llm`, or `get_engram` + `apply_engram` to tune `alpha` without recollecting) — pip-installable, ungated, Colab-ready. |

## Paper figures (`fig_*`)

Notebooks reproducing the paper's main results across modalities. **They may use an
earlier API**; for the current package (0.8.0+) — `Statistics`, `EngramResult`, and
pluggable `scale=` editing — follow the
[Quickstart](https://jeakwon.github.io/ai-engram/quickstart/) and
[Guide](https://jeakwon.github.io/ai-engram/guide/). The maintained, current-API TOFU
reproduction lives in [`tests/`](../tests) (`test_tofu_unlearn.py`,
`test_tofu_evaluate.py`, gated by `ENGRAM_RUN_TOFU` / `ENGRAM_RUN_TOFU_EVALUATE`).

| notebook | what |
|---|---|
| `fig_llm_tofu.ipynb` | TOFU forget10 unlearning on Llama-3.2-1B (the package's primary target) |
| `fig_mlp_mnist.ipynb` | MLP on MNIST |
| `fig_resnet18_cifar10.ipynb` / `fig_resnet18_cifar100.ipynb` | ResNet-18 on CIFAR-10 / CIFAR-100 |
| `fig_vit_imagenet1k.ipynb` | ViT on ImageNet-1k |
| `fig_wae_celeba.ipynb` | WAE on CelebA |

The current `engram` package covers `nn.Linear` and GPT-2 `Conv1D` (and fused-MoE via
`engram.moe`); the vision/`Conv2d` notebooks use the broader research code from the
paper, not the minimal published package.
