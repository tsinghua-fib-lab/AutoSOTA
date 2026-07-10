from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from merge_and_rebase.models.patch_openclip_attention import (
    LoRAableLinearMHA,
    merge_openclip_vit_attn,
    patch_openclip_vit_attn,
    set_linear_attention_ramp_step,
    split_openclip_vit_attn,
)


def test_linear_attention_forward_shape_matches_original() -> None:
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
    x = torch.randn(2, 7, 16)

    ref_y, _ = mha(x, x, x, need_weights=False)

    class _Block(nn.Module):
        def __init__(self, attn: nn.Module) -> None:
            super().__init__()
            self.attn = attn

    class _Transformer(nn.Module):
        def __init__(self, blk: nn.Module) -> None:
            super().__init__()
            self.resblocks = nn.ModuleList([blk])

    class _Visual(nn.Module):
        def __init__(self, attn: nn.Module) -> None:
            super().__init__()
            self.transformer = _Transformer(_Block(attn))

    visual = _Visual(mha)
    n = split_openclip_vit_attn(visual, proj_dropout=0.0, attn_impl="linear", kernel="elu_plus_one", eps=1e-6)
    assert n == 1

    lin_attn = visual.transformer.resblocks[0].attn
    assert isinstance(lin_attn, LoRAableLinearMHA)
    out, _ = lin_attn(x, x, x, need_weights=False)
    assert tuple(out.shape) == tuple(ref_y.shape)


def test_patch_openclip_vit_attn_preserves_device_and_dtype() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float64
    mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True).to(device=device, dtype=dtype)

    class _Block(nn.Module):
        def __init__(self, attn: nn.Module) -> None:
            super().__init__()
            self.attn = attn

    class _Transformer(nn.Module):
        def __init__(self, blk: nn.Module) -> None:
            super().__init__()
            self.resblocks = nn.ModuleList([blk])

    class _Visual(nn.Module):
        def __init__(self, attn: nn.Module) -> None:
            super().__init__()
            self.transformer = _Transformer(_Block(attn))

    visual = _Visual(mha).to(device=device, dtype=dtype)
    n = patch_openclip_vit_attn(visual, proj_dropout=0.0, attn_impl="softmax")
    assert n == 1

    patched_attn = visual.transformer.resblocks[0].attn
    param = next(patched_attn.parameters())
    assert param.device == device
    assert param.dtype == dtype

    x = torch.randn(2, 7, 16, device=device, dtype=dtype)
    out, _ = patched_attn(x, x, x, need_weights=False)
    assert out.device == device
    assert out.dtype == dtype


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        return x + y


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([_TinyBlock(), _TinyBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.resblocks:
            x = blk(x)
        return x


class _TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _TinyTransformer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


def test_linear_attention_allows_lora_grad_flow() -> None:
    torch.manual_seed(0)
    visual = _TinyVisual()
    n = split_openclip_vit_attn(visual, proj_dropout=0.0, attn_impl="linear", kernel="elu_plus_one", eps=1e-6)
    assert n == 2

    for p in visual.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    visual = get_peft_model(visual, lora_cfg)

    x = torch.randn(5, 3, 8)
    y = visual(x)
    loss = y.pow(2).mean()
    loss.backward()

    lora_grads = [p.grad for name, p in visual.named_parameters() if "lora_" in name]
    assert lora_grads
    assert all(g is not None for g in lora_grads)
    assert any(float(g.abs().sum()) > 0.0 for g in lora_grads if g is not None)


def test_softmax_to_linear_ramp_progresses() -> None:
    torch.manual_seed(0)

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)

    class _Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.resblocks = nn.ModuleList([_Block(), _Block()])

    class _Visual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = _Transformer()

    visual = _Visual()
    n = split_openclip_vit_attn(
        visual,
        proj_dropout=0.0,
        attn_impl="linear",
        kernel="elu_plus_one",
        eps=1e-6,
        ramp_steps=10,
    )
    assert n == 2

    x = torch.randn(2, 5, 8)
    blk0 = visual.transformer.resblocks[0].attn
    assert isinstance(blk0, LoRAableLinearMHA)

    set_linear_attention_ramp_step(visual, step=0)
    y0, _ = blk0(x, x, x)
    assert abs(blk0.blend_lambda - 0.0) < 1e-8

    set_linear_attention_ramp_step(visual, step=5)
    y_mid, _ = blk0(x, x, x)
    assert abs(blk0.blend_lambda - 0.5) < 1e-8

    set_linear_attention_ramp_step(visual, step=10)
    y_end, _ = blk0(x, x, x)
    assert abs(blk0.blend_lambda - 1.0) < 1e-8

    assert not torch.allclose(y0, y_mid)
    assert not torch.allclose(y_mid, y_end)


def test_delta_rule_linear_attention_path_runs() -> None:
    torch.manual_seed(0)

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)

    class _Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.resblocks = nn.ModuleList([_Block()])

    class _Visual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = _Transformer()

    visual = _Visual()
    n = split_openclip_vit_attn(
        visual,
        proj_dropout=0.0,
        attn_impl="linear",
        kernel="elu_plus_one",
        eps=1e-6,
        linear_rule="delta",
        delta_eta=0.7,
        delta_exclude_cls_from_store=True,
        delta_cls_only_readout=False,
        delta_learn_w0=True,
        delta_w0_rank=2,
    )
    assert n == 1

    blk0 = visual.transformer.resblocks[0].attn
    assert isinstance(blk0, LoRAableLinearMHA)
    assert blk0.linear_rule == "delta"
    assert abs(float(blk0.delta_eta) - 0.7) < 1e-8
    assert blk0.delta_mem is not None
    assert blk0.delta_w0_rank == 2

    x = torch.randn(2, 5, 8)
    y, _ = blk0(x, x, x)
    assert tuple(y.shape) == (2, 5, 8)
    assert torch.isfinite(y).all()


def test_unpatch_recomposes_to_fused_mha() -> None:
    torch.manual_seed(0)

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)

    class _Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.resblocks = nn.ModuleList([_Block(), _Block()])

    class _Visual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = _Transformer()

    visual = _Visual()
    n = split_openclip_vit_attn(visual, proj_dropout=0.0, attn_impl="softmax")
    assert n == 2
    patched_keys = set(visual.state_dict().keys())
    assert any(k.endswith("attn.q_proj.weight") for k in patched_keys)

    m = merge_openclip_vit_attn(visual)
    assert m == 2
    for blk in visual.transformer.resblocks:
        assert isinstance(blk.attn, nn.MultiheadAttention)

    sd_keys = set(visual.state_dict().keys())
    assert any(k.endswith("attn.in_proj_weight") for k in sd_keys)
    assert not any(k.endswith("attn.q_proj.weight") for k in sd_keys)
