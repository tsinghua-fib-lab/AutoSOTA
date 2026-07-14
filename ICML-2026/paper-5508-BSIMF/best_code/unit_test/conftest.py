"""Shared pytest fixtures for BUQ-SIM unit tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import pytest
import torch


# ---------------------------------------------------------------------------
# Tiny model: smallest dimensions that exercise every submodule
# ---------------------------------------------------------------------------
MODEL_CFG = dict(
    x_dim=4,
    image_size=8,
    z_dim=2,
    u_dim=4,
    num_components=2,
    x_distribution="continuous",
    vit_embed_dim=8,
    vit_depth=1,
    vit_num_heads=2,
    vit_patch_size=4,       # 8/4 = 2 patches per side → 4 patch tokens
    mlp_hidden_dims=(16,),
    rank_r=2,
    mask_prior_rho=0.1,
    base_mean_mode="learnable_scalar",
    base_var_mode="learnable_scalar",
    base_init_mean=0.1,
    base_init_var=0.25 ** 2,
    use_mask_x_in_encoder=True,
    u_missing_fallback_to_prior=True,
)

BATCH_SIZE = 3


@pytest.fixture(scope="module")
def tiny_model():
    from model import ContentUncertaintyDAG
    torch.manual_seed(0)
    m = ContentUncertaintyDAG(**MODEL_CFG)
    m.eval()
    return m


@pytest.fixture(scope="module")
def tiny_batch(tiny_model):
    torch.manual_seed(1)
    B = BATCH_SIZE
    x = torch.randn(B, tiny_model.x_dim)
    y = torch.randn(B, 1, tiny_model.image_size, tiny_model.image_size)
    return x, y


@pytest.fixture(scope="module")
def encode_out(tiny_model, tiny_batch):
    x, y = tiny_batch
    with torch.no_grad():
        return tiny_model.encode(x, y)
