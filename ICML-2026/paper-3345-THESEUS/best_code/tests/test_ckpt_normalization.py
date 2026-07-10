import torch

from merge_and_rebase.io.ckpt import align_to_base_keys, normalize_common_prefixes


def test_normalize_common_prefixes_strips_clip_model_model_prefix():
    sd = {
        "clip_model.model.visual.conv1.weight": torch.randn(2, 2),
        "clip_model.model.transformer.resblocks.0.ln_1.weight": torch.randn(2),
    }

    out = normalize_common_prefixes(sd)

    assert "visual.conv1.weight" in out
    assert "transformer.resblocks.0.ln_1.weight" in out
    assert all(not k.startswith("clip_model.") for k in out)


def test_align_to_base_keys_handles_clip_model_prefix_without_normalize():
    base = {
        "visual.conv1.weight": torch.zeros(2, 2),
        "ln_final.weight": torch.zeros(2),
    }
    tuned_raw = {
        "clip_model.model.visual.conv1.weight": torch.ones(2, 2),
        "clip_model.model.ln_final.weight": torch.ones(2),
        "clip_model.model.extra.weight": torch.ones(3, 3),
    }

    aligned = align_to_base_keys(tuned_raw, base)

    assert set(aligned.keys()) == {"visual.conv1.weight", "ln_final.weight"}
    assert torch.allclose(aligned["visual.conv1.weight"], torch.ones(2, 2))
    assert torch.allclose(aligned["ln_final.weight"], torch.ones(2))


def test_align_to_base_keys_splits_fused_in_proj_into_patched_qkv_keys():
    base = {
        "visual.transformer.resblocks.0.attn.q_proj.weight": torch.zeros(2, 2),
        "visual.transformer.resblocks.0.attn.k_proj.weight": torch.zeros(2, 2),
        "visual.transformer.resblocks.0.attn.v_proj.weight": torch.zeros(2, 2),
        "visual.transformer.resblocks.0.attn.q_proj.bias": torch.zeros(2),
        "visual.transformer.resblocks.0.attn.k_proj.bias": torch.zeros(2),
        "visual.transformer.resblocks.0.attn.v_proj.bias": torch.zeros(2),
    }
    tuned_raw = {
        "visual.transformer.resblocks.0.attn.in_proj_weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
        "visual.transformer.resblocks.0.attn.in_proj_bias": torch.arange(6, dtype=torch.float32),
    }

    aligned = align_to_base_keys(tuned_raw, base)

    assert set(aligned.keys()) == set(base.keys())
    assert torch.allclose(
        aligned["visual.transformer.resblocks.0.attn.q_proj.weight"],
        torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
    )
    assert torch.allclose(
        aligned["visual.transformer.resblocks.0.attn.k_proj.weight"],
        torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
    )
    assert torch.allclose(
        aligned["visual.transformer.resblocks.0.attn.v_proj.weight"],
        torch.tensor([[8.0, 9.0], [10.0, 11.0]]),
    )
    assert torch.allclose(
        aligned["visual.transformer.resblocks.0.attn.q_proj.bias"],
        torch.tensor([0.0, 1.0]),
    )
    assert torch.allclose(
        aligned["visual.transformer.resblocks.0.attn.k_proj.bias"],
        torch.tensor([2.0, 3.0]),
    )
    assert torch.allclose(
        aligned["visual.transformer.resblocks.0.attn.v_proj.bias"],
        torch.tensor([4.0, 5.0]),
    )
