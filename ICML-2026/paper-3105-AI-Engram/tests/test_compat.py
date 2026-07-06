"""Multi-architecture compatibility regression tests (tiny configs, CPU, offline).

Builds a tiny random-weight model for each supported HF causal-LM architecture
and checks that ``collect_statistics -> compute_engram_weights`` works both
WITHOUT and WITH an answer-token ``mask_fn``, and that every engram weight keeps
its layer's shape. This codifies the manual compatibility sweep as a pytest, so a
refactor that breaks an architecture (e.g. the MoE routing alignment, or a new
weight orientation) fails the suite instead of going unnoticed until someone
re-runs the sweep by hand.

Breadth, not depth: the exact-value correctness of masking lives in
``test_mask_fn_generic_linear_and_conv1d`` (T7) and ``test_mask_fn_moe_mixtral``
(T8). Here we sweep many architectures and assert each one *runs* and keeps shape.

Run via SLURM (no login-node execution):
    pytest -q tests/test_compat.py
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")  # whole module is transformers-only
import transformers as tf

from engram import EditorConfig, EngramEditor

VOCAB = 128
# pad_token_id=0 keeps the embedding's padding_idx < vocab on tiny configs whose
# default pad id (e.g. Phi3's 32000) would otherwise exceed it and fail to build.
_BASE = dict(
    vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
    intermediate_size=64, max_position_embeddings=64, pad_token_id=0,
)


def _mk(cfg_cls: str, model_cls: str, **kw) -> torch.nn.Module:
    return getattr(tf, model_cls)(getattr(tf, cfg_cls)(**kw)).eval()


# name -> builder of a tiny model from config (no downloads, random weights).
BUILDERS = {
    "Llama":   lambda: _mk("LlamaConfig", "LlamaForCausalLM", **_BASE, num_key_value_heads=2),
    "Mistral": lambda: _mk("MistralConfig", "MistralForCausalLM", **_BASE, num_key_value_heads=2),
    "Qwen2":   lambda: _mk("Qwen2Config", "Qwen2ForCausalLM", **_BASE, num_key_value_heads=2),
    "Qwen3":   lambda: _mk("Qwen3Config", "Qwen3ForCausalLM", **_BASE, num_key_value_heads=2),
    "Gemma":   lambda: _mk("GemmaConfig", "GemmaForCausalLM", **_BASE, num_key_value_heads=2, head_dim=16),
    "Gemma2":  lambda: _mk("Gemma2Config", "Gemma2ForCausalLM", **_BASE, num_key_value_heads=2, head_dim=16),
    "Phi":     lambda: _mk("PhiConfig", "PhiForCausalLM", **_BASE),
    "Phi3":    lambda: _mk("Phi3Config", "Phi3ForCausalLM", **_BASE),
    "OPT":     lambda: _mk("OPTConfig", "OPTForCausalLM", vocab_size=VOCAB, hidden_size=32,
                           num_hidden_layers=2, num_attention_heads=2, ffn_dim=64,
                           max_position_embeddings=64, word_embed_proj_dim=32, pad_token_id=0),
    "Falcon":  lambda: _mk("FalconConfig", "FalconForCausalLM", **_BASE),
    "GPTNeoX": lambda: _mk("GPTNeoXConfig", "GPTNeoXForCausalLM", **_BASE),
    "Bloom":   lambda: _mk("BloomConfig", "BloomForCausalLM", **_BASE),
    "Mixtral": lambda: _mk("MixtralConfig", "MixtralForCausalLM", **_BASE, num_key_value_heads=2,
                           num_local_experts=4, num_experts_per_tok=2),
    "GPT2":    lambda: _mk("GPT2Config", "GPT2LMHeadModel", n_embd=32, n_layer=2, n_head=2,
                           n_inner=64, n_positions=64, vocab_size=VOCAB, pad_token_id=0),
}

_FEATS = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}


def _cpu_cfg() -> EditorConfig:
    return EditorConfig(storage_device=torch.device("cpu"))


def _batch() -> dict:
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (2, 8))
    lab = torch.full((2, 8), -100)
    lab.view(-1)[[1, 3, 9, 12]] = 1  # 4 scattered "answer" tokens across both rows
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab}


@pytest.mark.parametrize("name", list(BUILDERS))
@pytest.mark.parametrize("masked", [False, True], ids=["mask_off", "mask_on"])
def test_architecture_compat(name: str, masked: bool):
    """collect -> compute runs for this architecture, every engram keeps its shape."""
    model = BUILDERS[name]()
    editor = EngramEditor(model, _cpu_cfg())  # absorb_bias on by default
    batch = _batch()
    mask_fn = (lambda b: b["labels"] != -100) if masked else None

    cov = editor.collect_statistics([batch], batch_fn=_FEATS, mask_fn=mask_fn)
    result = editor.compute_engram_weights(cov, cov)

    assert cov, f"{name}: no layers were hooked"
    modules = dict(model.named_modules())
    for n, info in result.layers.items():
        assert info.projection.shape == modules[n].weight.shape, f"{name}: {n} got {tuple(info.projection.shape)}"

    # MoE: masking must reach the routed expert layers, not just attn/router —
    # but only when this transformers exposes experts as per-expert nn.Linear
    # (transformers >=5 fuses them into Parameters, which forward hooks can't reach).
    if name == "Mixtral" and any(
        isinstance(mod, torch.nn.Linear) and ".experts." in n for n, mod in model.named_modules()
    ):
        assert any(".experts." in n for n in cov), f"{name}: expert layers not hooked"
