"""Extraction + scaling tests (CPU-only, deterministic, offline).

Run via SLURM (no login-node execution):
    pytest -q tests/
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engram import (
    EditorConfig,
    EngramEditor,
    EngramResult,
    LayerScaleInfo,
    Statistics,
    compose,
    count_ratio,
    effective_rank,
    uniform,
    weight_norm,
)


def cpu_cfg(**kw) -> EditorConfig:
    base = dict(storage_device=torch.device("cpu"))
    base.update(kw)
    return EditorConfig(**base)


def _info(name, proj, *, weight=None, weight_fro=None, n=1, N=1, target_erank=None, total_erank=None):
    """A LayerScaleInfo for hand-built EngramResults (pass a `weight` tensor or a `weight_fro` scalar)."""
    if weight_fro is None:
        weight_fro = float(weight.norm()) if weight is not None else 0.0
    return LayerScaleInfo(
        name=name,
        weight_fro=weight_fro,
        projection=proj,
        n=n,
        N=N,
        target_erank=target_erank,
        total_erank=total_erank,
    )


# --------------------------------------------------------------------------- #
# T0: package imports and public API surface
# --------------------------------------------------------------------------- #
def test_public_api():
    import engram

    assert isinstance(engram.__version__, str)
    for name in [
        "EditorConfig",
        "EngramEditor",
        "CovarianceCollector",
        "Statistics",
        "EngramResult",
        "LayerScaleInfo",
        "count_ratio",
        "weight_norm",
        "effective_rank",
        "uniform",
        "compose",
        "LayerHandler",
        "LinearHandler",
        "Conv1DHandler",
    ]:
        assert hasattr(engram, name), name


# --------------------------------------------------------------------------- #
# T1: correctness anchor (no bias) — if C_target == C_total and C is full rank,
# C . pinv(C) == I, so the projection P must equal W exactly.
# --------------------------------------------------------------------------- #
def test_engram_equals_weight_when_target_is_total():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4, bias=False))
    editor = EngramEditor(model, cpu_cfg())  # absorb_bias auto, but no bias -> off

    X = torch.randn(512, 8)  # >> 8 gaussian samples => full-rank covariance
    loader = DataLoader(TensorDataset(X), batch_size=64)

    cov = editor.collect_statistics(loader)
    assert cov["0"].shape == (8, 8)  # not augmented (bias-free)
    assert cov.count["0"] == 512

    result = editor.compute_engram_weights(cov, cov)
    W = model[0].weight.detach().to(torch.float64)
    assert set(result.layers) == {"0"}
    assert result.bias == {}  # nothing to absorb
    assert torch.allclose(result.layers["0"].projection.double(), W, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- #
# T2: subspace — target inputs spanning a k-dim subspace make C.pinv(C) a rank-k
# projector, so the projection has rank <= k (< full weight rank).
# --------------------------------------------------------------------------- #
def test_engram_rank_bounded_by_target_subspace():
    torch.manual_seed(0)
    k = 3
    model = nn.Sequential(nn.Linear(8, 4, bias=False))
    editor = EngramEditor(model, cpu_cfg())

    Z = torch.randn(256, k)
    A = torch.randn(k, 8)
    X = Z @ A  # rows live in a k-dimensional subspace of R^8
    loader = DataLoader(TensorDataset(X), batch_size=64)

    cov = editor.collect_statistics(loader)
    proj = editor.compute_engram_weights(cov, cov).layers["0"].projection

    W = model[0].weight.detach().to(torch.float64)
    assert proj.shape == (4, 8)
    assert torch.linalg.matrix_rank(W) == 4  # weight itself is full rank
    assert torch.linalg.matrix_rank(proj, rtol=1e-3) <= k  # projection reduced it


# --------------------------------------------------------------------------- #
# T3: GPT-2 Conv1D path — verifies the transpose round-trip (projection keeps the
# layer's [in, out] shape) and bias-bearing Conv1D layers produce bias projections.
# --------------------------------------------------------------------------- #
def test_gpt2_conv1d_shapes_and_bias():
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.pytorch_utils import Conv1D

    from engram.handlers import get_conv1d_class

    assert get_conv1d_class() is not None

    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, n_positions=64, vocab_size=128)
    model = GPT2LMHeadModel(cfg).eval()
    editor = EngramEditor(model, cpu_cfg())  # absorb_bias on by default

    ids = torch.randint(0, 128, (4, 16))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    cov = editor.collect_statistics([batch], batch_fn=lambda b: b)
    result = editor.compute_engram_weights(cov, cov)

    modules = dict(model.named_modules())
    for name, info in result.layers.items():
        assert info.projection.shape == modules[name].weight.shape, name
    for name, b in result.bias.items():
        assert b.shape == modules[name].bias.shape, name

    conv1d_names = [n for n, m in modules.items() if isinstance(m, Conv1D)]
    assert conv1d_names, "no Conv1D modules found in GPT-2"
    # Conv1D always has a bias -> absorbed -> present in both layers and bias
    assert all(n in result.layers and n in result.bias for n in conv1d_names)

    cattn = next(n for n in conv1d_names if n.endswith("attn.c_attn"))
    assert result.layers[cattn].projection.shape == (32, 96)  # [n_embd, 3*n_embd], kept
    assert result.bias[cattn].shape == (96,)
    # lm_head is bias-free -> projection only, no bias engram
    assert "lm_head" in result.layers and "lm_head" not in result.bias


# --------------------------------------------------------------------------- #
# T5: bias absorption — with a bias-bearing layer and C_target == C_total (full
# rank), the engram recovers BOTH W and b exactly.
# --------------------------------------------------------------------------- #
def test_bias_absorption_recovers_weight_and_bias():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4))  # bias=True
    editor = EngramEditor(model, cpu_cfg())  # absorb_bias on by default

    X = torch.randn(512, 8)
    loader = DataLoader(TensorDataset(X), batch_size=64)

    cov = editor.collect_statistics(loader)
    assert cov["0"].shape == (9, 9)  # augmented [x ; 1]

    result = editor.compute_engram_weights(cov, cov)
    W = model[0].weight.detach().to(torch.float64)
    b = model[0].bias.detach().to(torch.float64)
    assert result.layers["0"].projection.shape == W.shape
    assert "0" in result.bias and result.bias["0"].shape == b.shape
    assert torch.allclose(result.layers["0"].projection.double(), W, atol=1e-4, rtol=1e-3)
    assert torch.allclose(result.bias["0"].double(), b, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- #
# T6: absorb_bias=False reproduces the original W-only behavior (no bias engram).
# --------------------------------------------------------------------------- #
def test_absorb_bias_off_is_weight_only():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4))  # has bias, but absorption disabled
    editor = EngramEditor(model, cpu_cfg(absorb_bias=False))

    X = torch.randn(512, 8)
    loader = DataLoader(TensorDataset(X), batch_size=64)

    cov = editor.collect_statistics(loader)
    assert cov["0"].shape == (8, 8)  # not augmented

    result = editor.compute_engram_weights(cov, cov)
    W = model[0].weight.detach().to(torch.float64)
    assert result.bias == {}
    assert torch.allclose(result.layers["0"].projection.double(), W, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- #
# T7: collector-level mask_fn restricts covariance to selected tokens, for any
# layer type — including GPT-2 Conv1D. Covariance is now the MEAN of x^T x over
# the selected rows, with the count tracked alongside.
# --------------------------------------------------------------------------- #
def test_mask_fn_generic_linear_and_conv1d():
    # (a) nn.Linear — mask_fn keeps exactly the masked token rows; cov is their mean
    torch.manual_seed(0)
    lin = nn.Sequential(nn.Linear(4, 3, bias=False)).eval()
    editor = EngramEditor(lin, cpu_cfg())
    X = torch.randn(2, 5, 4)  # [batch, seq, dim]
    labels = torch.full((2, 5), -100)
    labels.view(-1)[[0, 3, 7]] = 1  # 3 answer tokens
    cov = editor.collect_statistics(
        [(X, labels)], batch_fn=lambda b: b[0], mask_fn=lambda b: b[1] != -100
    )
    sel = X.reshape(-1, 4)[labels.reshape(-1) != -100].double()
    assert cov["0"].shape == (4, 4)
    assert cov.count["0"] == 3
    assert torch.allclose(cov["0"].double(), sel.mT @ sel / sel.shape[0], atol=1e-4)

    # (b) GPT-2 Conv1D — the same mask_fn works on the Conv1D path too
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    gpt = GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=16, vocab_size=40)
    ).eval()
    ed = EngramEditor(gpt, cpu_cfg(absorb_bias=False))
    ids = torch.randint(0, 40, (2, 6))
    lab = torch.full((2, 6), -100)
    lab.view(-1)[[1, 4, 9, 10]] = 1  # 4 answer tokens
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab}
    feats = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}

    # capture the Conv1D layer's input to build the expected masked covariance
    cattn = "transformer.h.0.attn.c_attn"
    cap = {}
    handle = dict(gpt.named_modules())[cattn].register_forward_pre_hook(
        lambda m, inp: cap.__setitem__("x", inp[0].detach().reshape(-1, 16).double())
    )
    with torch.inference_mode():
        gpt(**feats(batch))
    handle.remove()
    sel2 = cap["x"][lab.reshape(-1) != -100]

    cov2 = ed.collect_statistics([batch], batch_fn=feats, mask_fn=lambda b: b["labels"] != -100)
    assert cov2[cattn].shape == (16, 16)
    assert cov2.count[cattn] == 4
    assert torch.allclose(cov2[cattn].double(), sel2.mT @ sel2 / sel2.shape[0], atol=1e-4)


# --------------------------------------------------------------------------- #
# T8: MoE masking — routed expert layers recover their answer-token mask by
# matching rows back to the router input (per-expert nn.Linear layout, tf 4.x).
# --------------------------------------------------------------------------- #
def test_mask_fn_moe_mixtral():
    pytest.importorskip("transformers")
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(0)
    m = MixtralForCausalLM(
        MixtralConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=64, max_position_embeddings=32,
            num_local_experts=4, num_experts_per_tok=2,
        )
    ).eval()
    # transformers >=5 fuses Mixtral experts into Parameters (no per-expert
    # nn.Linear); this test targets the per-expert .w1 layout — skip otherwise.
    if not any(isinstance(mod, nn.Linear) and n.endswith(".w1") for n, mod in m.named_modules()):
        pytest.skip("no per-expert .w1 nn.Linear (this transformers fuses MoE experts)")
    ed = EngramEditor(m, cpu_cfg(absorb_bias=False))
    ids = torch.randint(0, 64, (2, 8))
    lab = torch.full((2, 8), -100)
    lab.view(-1)[[1, 4, 9, 12]] = 1
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": lab}
    feats = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}

    # masking used to crash on the routed expert layers — now it runs
    cov = ed.collect_statistics([batch], batch_fn=feats, mask_fn=lambda b: b["labels"] != -100)
    result = ed.compute_engram_weights(cov, cov)
    expert_w1 = [n for n in cov if n.endswith(".w1")]
    assert expert_w1, "no MoE expert layers were hooked"
    mods = dict(m.named_modules())
    for n in result.layers:
        assert result.layers[n].projection.shape == mods[n].weight.shape, n

    # correctness: brute-force align one expert's w1 against the router input.
    # Pick an expert that actually received answer tokens (some get none this seed),
    # else the mean reference below would divide by a zero count.
    w1 = max(expert_w1, key=lambda n: cov.count[n])
    assert cov.count[w1] > 0, "no expert received an answer token"
    gate = w1.rsplit(".experts.", 1)[0] + ".gate"
    cap = {}
    hg = mods[gate].register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("g", inp[0].detach().reshape(-1, 32).double())
    )
    he = mods[w1].register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("e", inp[0].detach().reshape(-1, 32).double())
    )
    with torch.inference_mode():
        m(**feats(batch))
    hg.remove()
    he.remove()

    match = (cap["g"][None, :, :] == cap["e"][:, None, :]).all(-1)  # exact row match
    assert (match.sum(1) == 1).all()
    idx = match.float().argmax(1)
    sel = cap["e"][(lab.reshape(-1) != -100)[idx]]
    # cov is now the MEAN over the selected rows
    assert cov.count[w1] == sel.shape[0]
    assert torch.allclose(cov[w1].double(), sel.mT @ sel / sel.shape[0], atol=1e-3)


# A tiny decoder-style stack: module names are layers.{i}.{down,up}_proj, so the
# index-based selection below has a realistic ".layers.<idx>." path to parse.
class _TinyStack(nn.Module):
    def __init__(self, n: int = 3, d: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {"down_proj": nn.Linear(d, d, bias=False), "up_proj": nn.Linear(d, d, bias=False)}
            )
            for _ in range(n)
        )

    def forward(self, x):
        for blk in self.layers:
            x = blk["up_proj"](blk["down_proj"](x))
        return x


def _stack_loader(d: int = 4):
    return DataLoader(TensorDataset(torch.randn(64, d)), batch_size=16)


# --------------------------------------------------------------------------- #
# T9: target_modules LoRA convention — a list matches by dotted name suffix; a
# string is a regex over the full module path.
# --------------------------------------------------------------------------- #
def test_target_modules_suffix_and_regex():
    torch.manual_seed(0)
    model = _TinyStack(n=3).eval()

    cov = EngramEditor(model, cpu_cfg()).collect_statistics(
        _stack_loader(), target_modules=["down_proj"]
    )
    assert set(cov) == {"layers.0.down_proj", "layers.1.down_proj", "layers.2.down_proj"}

    cov = EngramEditor(model, cpu_cfg()).collect_statistics(
        _stack_loader(), target_modules=r".*layers\.1\..*"
    )
    assert set(cov) == {"layers.1.down_proj", "layers.1.up_proj"}


# --------------------------------------------------------------------------- #
# T10: layers_to_transform / layers_pattern select by decoder-layer index, and
# combine with target_modules as an AND filter (PEFT convention).
# --------------------------------------------------------------------------- #
def test_layers_to_transform_selects_by_index():
    torch.manual_seed(0)
    model = _TinyStack(n=3).eval()

    cov = EngramEditor(model, cpu_cfg()).collect_statistics(
        _stack_loader(), target_modules=["down_proj"], layers_to_transform=[0, 2], layers_pattern="layers"
    )
    assert set(cov) == {"layers.0.down_proj", "layers.2.down_proj"}

    # int form + target_modules=None -> every linear in just that one layer
    cov = EngramEditor(model, cpu_cfg()).collect_statistics(
        _stack_loader(), layers_to_transform=1, layers_pattern="layers"
    )
    assert set(cov) == {"layers.1.down_proj", "layers.1.up_proj"}

    # layers_pattern=None auto-detects common containers ("layers", "h", ...)
    cov = EngramEditor(model, cpu_cfg()).collect_statistics(
        _stack_loader(), target_modules=["down_proj"], layers_to_transform=[0]
    )
    assert set(cov) == {"layers.0.down_proj"}


# --------------------------------------------------------------------------- #
# T11: target_layers still works as a deprecated alias.
# --------------------------------------------------------------------------- #
def test_target_layers_deprecated_alias():
    import warnings

    torch.manual_seed(0)
    model = _TinyStack(n=2).eval()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cov = EngramEditor(model, cpu_cfg()).collect_statistics(
            _stack_loader(), target_layers=["layers.1.up_proj"]
        )
    assert set(cov) == {"layers.1.up_proj"}
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


# --------------------------------------------------------------------------- #
# T12: storage_device defaults to None -> follows the model's device.
# --------------------------------------------------------------------------- #
def test_default_storage_follows_model_device():
    assert EditorConfig().storage_device is None
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 3, bias=False)).eval()  # lives on CPU
    cov = EngramEditor(model, EditorConfig()).collect_statistics(
        DataLoader(TensorDataset(torch.randn(32, 6)), batch_size=8)
    )
    assert cov["0"].device.type == "cpu"  # followed the (CPU) model


# T12b: a selection that matches no supported layer warns instead of silently
# producing an empty covariance (e.g. a target_modules typo).
def test_no_match_warns():
    import warnings

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 2, bias=False)).eval()
    ed = EngramEditor(model, cpu_cfg())
    loader = DataLoader(TensorDataset(torch.randn(8, 4)), batch_size=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cov = ed.collect_statistics(loader, target_modules=["does_not_exist"])
    assert len(cov) == 0
    assert any("no supported layers matched" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- #
# T13: apply — W <- W - alpha*f*P; copy by default, inplace optional. uniform()
# gives f=1, so the bare projection is subtracted.
# --------------------------------------------------------------------------- #
def test_apply_uniform_copy_and_inplace():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3, bias=False)).eval()
    W0 = model[0].weight.detach().clone()
    proj = torch.randn(3, 4)
    engram = EngramResult(layers={"0": _info("0", proj, weight=W0)})
    editor = EngramEditor(model, cpu_cfg())

    edited = editor.apply(engram, alpha=0.5, scale=uniform())  # copy by default
    assert edited is not model and torch.equal(model[0].weight, W0)  # original untouched
    assert torch.allclose(edited[0].weight.double(), (W0 - 0.5 * proj).double(), atol=1e-5)

    same = editor.apply(engram, alpha=1.0, scale=uniform(), inplace=True)  # edits self.model
    assert same is model
    assert torch.allclose(model[0].weight.double(), (W0 - proj).double(), atol=1e-5)


# T14: apply also subtracts the bias engram for bias-bearing layers.
def test_apply_with_bias():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3)).eval()  # bias=True
    W0, b0 = model[0].weight.detach().clone(), model[0].bias.detach().clone()
    proj, bproj = torch.randn(3, 4), torch.randn(3)
    engram = EngramResult(layers={"0": _info("0", proj, weight=W0)}, bias={"0": bproj})
    edited = EngramEditor(model, cpu_cfg()).apply(engram, alpha=0.5, scale=uniform())
    assert torch.allclose(edited[0].weight.double(), (W0 - 0.5 * proj).double(), atol=1e-5)
    assert torch.allclose(edited[0].bias.double(), (b0 - 0.5 * bproj).double(), atol=1e-5)


# T15: weight_norm scaling — f_l = (rel_l/max rel)^p, rel_l = ||P||/||W||.
def test_apply_weight_norm_scales_by_relative_norm():
    torch.manual_seed(0)
    model = _TinyStack(n=2, d=4).eval()
    mods = dict(model.named_modules())
    projs = {"layers.0.down_proj": torch.full((4, 4), 0.5), "layers.1.down_proj": torch.full((4, 4), 0.01)}
    W0 = {ln: mods[ln].weight.detach().clone() for ln in projs}
    engram = EngramResult(
        layers={ln: _info(ln, projs[ln], weight=mods[ln].weight.detach()) for ln in projs}
    )
    edited = dict(
        EngramEditor(model, cpu_cfg()).apply(engram, alpha=1.0, scale=weight_norm(1.0)).named_modules()
    )

    rel = {ln: projs[ln].norm().item() / W0[ln].norm().item() for ln in projs}
    top = max(rel, key=rel.get)
    other = next(ln for ln in projs if ln != top)
    assert torch.allclose((W0[top] - edited[top].weight).double(), projs[top].double(), atol=1e-5)
    s = rel[other] / rel[top]
    assert torch.allclose((W0[other] - edited[other].weight).double(), (s * projs[other]).double(), atol=1e-5)


# T16: count_ratio scaling — the default; f_l = (n/N)^power, applied per layer.
def test_apply_count_ratio_factor():
    torch.manual_seed(0)
    model = _TinyStack(n=2, d=4).eval()
    mods = dict(model.named_modules())
    projs = {"layers.0.down_proj": torch.randn(4, 4), "layers.1.down_proj": torch.randn(4, 4)}
    W0 = {ln: mods[ln].weight.detach().clone() for ln in projs}
    engram = EngramResult(layers={
        "layers.0.down_proj": _info("layers.0.down_proj", projs["layers.0.down_proj"], n=2, N=8),  # f=0.25
        "layers.1.down_proj": _info("layers.1.down_proj", projs["layers.1.down_proj"], n=4, N=4),  # f=1.0
    })
    edited = dict(EngramEditor(model, cpu_cfg()).apply(engram, alpha=1.0).named_modules())  # default count_ratio(1)
    assert torch.allclose((W0["layers.0.down_proj"] - edited["layers.0.down_proj"].weight),
                          0.25 * projs["layers.0.down_proj"], atol=1e-6)
    assert torch.allclose((W0["layers.1.down_proj"] - edited["layers.1.down_proj"].weight),
                          projs["layers.1.down_proj"], atol=1e-6)


# T17: edit(target, total) == compute_engram_weights then apply (same default scale).
def test_edit_equals_compute_then_apply():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 3, bias=False)).eval()
    loader = DataLoader(TensorDataset(torch.randn(64, 6)), batch_size=16)
    editor = EngramEditor(model, cpu_cfg())
    cov = editor.collect_statistics(loader)
    manual = editor.apply(editor.compute_engram_weights(cov, cov), alpha=0.6)
    direct = editor.edit(cov, cov, alpha=0.6)
    assert torch.allclose(direct[0].weight, manual[0].weight, atol=1e-6)


# --------------------------------------------------------------------------- #
# T18: EQUIVALENCE (critical) — the mean+count path with the default count_ratio(1)
# reproduces the legacy summed-covariance engram W . Sigma_t . pinv(Sigma_total)
# **at the same float32 precision** (so only the mean-vs-sum reformulation is tested).
# --------------------------------------------------------------------------- #
def test_mean_path_reproduces_legacy_sum_engram():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 4, bias=False)).eval()
    ed = EngramEditor(model, cpu_cfg())
    # different target/total data -> n != N and a non-trivial engram (well-conditioned)
    target = ed.collect_statistics(DataLoader(TensorDataset(torch.randn(40, 6)), batch_size=8))
    total = ed.collect_statistics(DataLoader(TensorDataset(torch.randn(120, 6)), batch_size=8))

    # legacy reference: reconstruct the SUMS (mean * count) and solve in float32
    W = model[0].weight.detach().float()
    sigma_t = (target["0"] * target.count["0"]).float()
    sigma_a = (total["0"] * total.count["0"]).float()
    legacy = W @ sigma_t @ torch.linalg.pinv(sigma_a)

    # new path: projection P, scaled by the default count_ratio(1) factor n/N
    info = ed.compute_engram_weights(target, total).layers["0"]
    new_engram = (info.n / info.N) * info.projection
    assert info.n == 40 and info.N == 120
    assert torch.allclose(new_engram, legacy, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- #
# T19: scaling-function unit tests (count_ratio guard, uniform, effective_rank, compose).
# --------------------------------------------------------------------------- #
def test_scaling_functions():
    eye = torch.eye(2)
    infos = {
        "a": _info("a", torch.ones(2, 2), n=2, N=8),   # n/N = 0.25
        "b": _info("b", torch.ones(2, 2), n=4, N=4),   # n/N = 1.0
        "z": _info("z", torch.ones(2, 2), n=0, N=4),   # guard: no target token -> 0
    }
    cr = count_ratio(1.0)(infos)
    assert cr["a"] == pytest.approx(0.25) and cr["b"] == pytest.approx(1.0) and cr["z"] == 0.0
    assert count_ratio(2.0)(infos)["a"] == pytest.approx(0.0625)

    u = uniform()(infos)
    assert all(v == 1.0 for v in u.values())

    # compose multiplies per layer; uniform is the identity factor
    comp = compose(count_ratio(1.0), uniform())(infos)
    assert comp == pytest.approx(cr)

    # effective_rank: f_l = (target_erank / total_erank) ** power
    er_infos = {
        "a": _info("a", torch.ones(2, 2), target_erank=1.0, total_erank=2.0),  # ratio 0.5
        "b": _info("b", torch.ones(2, 2), target_erank=2.0, total_erank=2.0),  # ratio 1.0
    }
    er = effective_rank(1.0)(er_infos)
    assert er["a"] == pytest.approx(0.5) and er["b"] == pytest.approx(1.0)
    assert effective_rank(2.0)(er_infos)["a"] == pytest.approx(0.25)
    # missing effective ranks -> clear error
    with pytest.raises(ValueError, match="effective ranks"):
        effective_rank()({"z": _info("z", torch.ones(2, 2))})


# --------------------------------------------------------------------------- #
# T20: Statistics container — count-weighted merge, save/load round-trip, and a
# clear error when loading the legacy (untagged) raw-covariance format.
# --------------------------------------------------------------------------- #
def test_statistics_merge_is_count_weighted():
    s1 = Statistics({"a": torch.ones(2, 2) * 2.0}, {"a": 3})
    s2 = Statistics({"a": torch.ones(2, 2) * 6.0}, {"a": 1})
    m = Statistics.merge(s1, s2)
    assert m.count["a"] == 4
    assert torch.allclose(m["a"], torch.ones(2, 2) * 3.0)  # (3*2 + 1*6) / 4 = 3

    # a key present in only one input keeps its own (count, mean)
    s3 = Statistics({"b": torch.ones(2, 2)}, {"b": 5})
    m2 = Statistics.merge(s1, s3)
    assert m2.count == {"a": 3, "b": 5}


def test_statistics_save_load_roundtrip_and_rejects_legacy(tmp_path):
    s = Statistics({"x": torch.randn(3, 3)}, {"x": 7})
    p = tmp_path / "stats.pt"
    s.save(p)
    loaded = Statistics.load(p)
    assert loaded.count == {"x": 7}
    assert torch.allclose(loaded["x"], s["x"])

    legacy = tmp_path / "legacy.pt"
    torch.save({"0": torch.randn(3, 3)}, legacy)  # old sum-only dict, no format tag
    with pytest.raises(ValueError, match="not a Statistics file"):
        Statistics.load(legacy)


def test_editor_save_load_statistics(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 2, bias=False)).eval()
    ed = EngramEditor(model, cpu_cfg())
    cov = ed.collect_statistics(DataLoader(TensorDataset(torch.randn(16, 4)), batch_size=8))
    p = tmp_path / "c.pt"
    ed.save_statistics(cov, p)
    loaded = ed.load_statistics(p)
    assert set(loaded) == set(cov) and loaded.count == cov.count
    assert torch.allclose(loaded["0"], cov["0"])


# T22: compute warns (and skips) when a target layer is absent from the total.
def test_compute_warns_on_target_layer_absent_from_total():
    import warnings

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 2, bias=False)).eval()
    ed = EngramEditor(model, cpu_cfg())
    cov = ed.collect_statistics(DataLoader(TensorDataset(torch.randn(16, 4)), batch_size=8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ed.compute_engram_weights(cov, Statistics({}, {}))  # total covers nothing
    assert result.layers == {}
    assert any("absent from total" in str(w.message) for w in caught)


# T23: compute_erank populates per-layer effective ranks; effective_rank needs them.
def test_compute_erank_enables_effective_rank():
    torch.manual_seed(0)
    model = _TinyStack(n=2, d=4).eval()
    ed = EngramEditor(model, cpu_cfg())
    cov = ed.collect_statistics(_stack_loader())

    r0 = ed.compute_engram_weights(cov, cov)  # default: no eranks computed
    assert all(i.target_erank is None and i.total_erank is None for i in r0.layers.values())
    with pytest.raises(ValueError, match="effective ranks"):
        ed.apply(r0, scale=effective_rank())

    r1 = ed.compute_engram_weights(cov, cov, compute_erank=True)
    assert all(isinstance(i.target_erank, float) and isinstance(i.total_erank, float)
               for i in r1.layers.values())
    edited = ed.apply(r1, alpha=0.5, scale=effective_rank(1.0))  # target==total -> ratio 1
    assert edited is not model
