"""Quick training time estimation for dim=1015"""
import sys, time
from pathlib import Path
import torch

from cleanup_ssps.space_factory import build_ssp_space, resolve_encoded_dim
from cleanup_ssps.model import ResidualMLP
from cleanup_ssps.dataset_registry import DatasetSpec, ensure_target_dataset
from cleanup_ssps.run import FlowTrainer

ssp_cfg = {
    "bundle_type": "hexagonal",
    "n_rotates": 13,
    "n_scales": 13,
    "length_scale": 0.2,
    "domain_dim": 2,
    "domain_bounds": [[-1, 1], [-1, 1]],
}
enc_dim = resolve_encoded_dim(ssp_cfg)
print(f"Encoded dim: {enc_dim}")

ssp_space = build_ssp_space(ssp_cfg, domain_dim=2)

# Use existing dataset
data_root = Path("/autosota_cache/data")

model = ResidualMLP(enc_dim, flow=True).to("cuda")
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

train_dir = "/autosota_cache/data/hex_dim1015_ls0p2_bounds_m1_1__m1_1/train"

trainer = FlowTrainer(
    encoded_dim=enc_dim,
    architecture=model,
    data_dir=train_dir,
    batch_size=256,
    epochs=3,
    lr=1e-4,
    weight_decay=1e-4,
    val_split=0.1,
    noise_type="uniform_hypersphere",
    target_type="coordinate",
    device="cuda",
    sampling_mode="euc_det",
    use_ot_train=False,
    dataloader_num_workers=4,
)
t0 = time.time()
models, train_loss, val_loss = trainer.train()
elapsed = time.time() - t0
print(f"3 epochs took {elapsed:.1f}s ({elapsed/3:.1f}s per epoch)")
print(f"Estimated 100 epochs: {elapsed/3*100/60:.1f} minutes")
print(f"Losses: {[f'{l:.4f}' for l in train_loss]}")
