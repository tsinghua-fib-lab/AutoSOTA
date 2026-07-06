"""Fused-expert MoE adapter tests (engram.moe).

Runs only where this transformers exposes the standard fused-expert layout
(transformers >=5); on the per-expert nn.Linear layout (4.x) it skips. Verifies the
adapter's per-expert covariance against a brute-force recomputation, and that the
engram computes + applies to the right 3D-Parameter slices.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from engram import EditorConfig, EngramEditor, uniform


def _tiny_mixtral():
    pytest.importorskip("transformers")
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(0)
    return MixtralForCausalLM(
        MixtralConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=64, max_position_embeddings=32,
            num_local_experts=4, num_experts_per_tok=2,
        )
    ).eval()


def test_fused_expert_adapter_matches_bruteforce():
    from engram.moe import FusedExpertAdapter, _is_standard_fused, apply_engram_weights

    m = _tiny_mixtral()
    fused = [(n, mod) for n, mod in m.named_modules() if _is_standard_fused(mod)]
    if not fused:
        pytest.skip("this transformers uses the per-expert nn.Linear MoE layout (4.x)")
    name, mod = fused[0]

    editor = EngramEditor(m, EditorConfig(storage_device=torch.device("cpu")), adapters=[FusedExpertAdapter()])
    ids = torch.randint(0, 64, (2, 8))
    lab = torch.full((2, 8), -100)
    lab.view(-1)[[1, 4, 9, 12]] = 1  # 4 answer tokens
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab}
    feats = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}

    cov = editor.collect_statistics([batch], batch_fn=feats, mask_fn=lambda b: b["labels"] != -100)
    assert any(".gate_up_proj." in k for k in cov), "no fused expert covariances collected"
    assert any(".down_proj." in k for k in cov)

    # brute force: capture the experts module's (hidden_states, top_k_index), recompute per-expert
    cap = {}

    def grab(_mod, args):
        x = args[0].reshape(-1, args[0].shape[-1]).double()
        cap["x"], cap["idx"] = x, args[1].reshape(x.shape[0], -1)

    h = mod.register_forward_pre_hook(grab)
    with torch.inference_mode():
        m(**feats(batch))
    h.remove()

    mask = lab.reshape(-1) != -100
    gate_up = mod.gate_up_proj.double()
    checked = 0
    for e in range(gate_up.shape[0]):
        sel = (cap["idx"] == e).any(-1) & mask
        if not bool(sel.any()):
            continue
        x_e = cap["x"][sel]
        ke = x_e.shape[0]
        gu_key, dn_key = f"{name}.gate_up_proj.{e}", f"{name}.down_proj.{e}"
        # cov holds the MEAN of x^T x over the expert's masked routed rows, count alongside
        assert cov.count[gu_key] == ke and cov.count[dn_key] == ke
        assert torch.allclose(cov[gu_key].double(), x_e.mT @ x_e / ke, atol=1e-3)
        gate, up = F.linear(x_e, gate_up[e]).chunk(2, dim=-1)
        inter = mod.act_fn(gate) * up
        assert torch.allclose(cov[dn_key].double(), inter.mT @ inter / ke, atol=1e-3)
        checked += 1
    assert checked > 0, "no expert received an answer token"

    # engram per expert slice -> correct shapes -> applies to the 3D Parameter slice
    result = editor.compute_engram_weights(cov, cov)
    assert any(".gate_up_proj." in k for k in result.layers) and any(".down_proj." in k for k in result.layers)
    from engram.moe import _split_key

    mods = dict(m.named_modules())
    n_fused = 0
    for k, info in result.layers.items():
        sp = _split_key(k)
        if sp is None:
            continue  # a normal module's engram (attention / router) — not a fused slice
        nm, proj, idx = sp
        assert info.projection.shape == getattr(mods[nm], proj)[idx].shape, k
        n_fused += 1
    assert n_fused > 0

    # low-level apply_engram_weights still writes the slices from a final-delta dict
    before = mod.gate_up_proj.detach().clone()
    deltas = {k: info.projection for k, info in result.layers.items()}
    apply_engram_weights(m, deltas, alpha=0.5)
    assert not torch.equal(before, mod.gate_up_proj), "apply did not edit any expert slice"


def test_adapter_is_noop_without_fused_experts():
    # a plain dense model: the adapter attaches but finds nothing -> no fused keys,
    # behaviour identical to no adapter.
    import torch.nn as nn

    from engram.moe import FusedExpertAdapter

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4, bias=False)).eval()
    loader = [(torch.randn(16, 8),)]
    base = EngramEditor(model, EditorConfig(storage_device=torch.device("cpu")))
    withad = EngramEditor(model, EditorConfig(storage_device=torch.device("cpu")), adapters=[FusedExpertAdapter()])
    c0 = base.collect_statistics(loader)
    c1 = withad.collect_statistics(loader)
    assert set(c0) == set(c1) == {"0"}
    assert torch.allclose(c0["0"], c1["0"])


def test_editor_apply_edits_fused_slices():
    # M2: editor.apply dispatches fused-expert keys to the adapter (Parameter slices)
    from engram.moe import FusedExpertAdapter, _is_standard_fused

    m = _tiny_mixtral()
    fused = [(n, mod) for n, mod in m.named_modules() if _is_standard_fused(mod)]
    if not fused:
        pytest.skip("this transformers uses the per-expert nn.Linear MoE layout (4.x)")
    _, mod = fused[0]

    editor = EngramEditor(m, EditorConfig(storage_device=torch.device("cpu")), adapters=[FusedExpertAdapter()])
    ids = torch.randint(0, 64, (2, 8))
    lab = torch.full((2, 8), -100)
    lab.view(-1)[[1, 4, 9, 12]] = 1
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab}
    feats = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
    cov = editor.collect_statistics([batch], batch_fn=feats, mask_fn=lambda b: b["labels"] != -100)
    result = editor.compute_engram_weights(cov, cov)

    before = mod.gate_up_proj.detach().clone()
    edited = editor.apply(result, alpha=0.5, scale=uniform(), inplace=True)
    assert edited is m
    assert not torch.equal(before, mod.gate_up_proj), "editor.apply did not edit any fused expert slice"
