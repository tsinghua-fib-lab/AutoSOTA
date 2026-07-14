"""Quick test to verify training pipeline works"""
import sys
from pathlib import Path
import torch

from cleanup_ssps.space_factory import build_ssp_space, resolve_encoded_dim
from cleanup_ssps.model import ResidualMLP
from cleanup_ssps.dataset_registry import DatasetSpec, ensure_target_dataset
from cleanup_ssps.run import FlowTrainer

ssp_cfg = {
    "bundle_type": "hexagonal",
    "n_rotates": 5,
    "n_scales": 5,
    "length_scale": 0.2,
    "domain_dim": 2,
    "domain_bounds": [[-1, 1], [-1, 1]],
}
enc_dim = resolve_encoded_dim(ssp_cfg)
print(f"Encoded dim: {enc_dim}")

ssp_space = build_ssp_space(ssp_cfg, domain_dim=2)

data_root = Path("/autosota_cache/data/test151")
spec = DatasetSpec(
    data_root=data_root,
    dataset_type="coordinate_ssps",
    encoded_dim=enc_dim,
    length_scale=0.2,
    train_samples=1000,
    test_samples=200,
    sampling_method="sobol",
)
ds_info = ensure_target_dataset(spec, ssp_cfg, ssp_space)
print(f"Dataset: {ds_info}")

model = ResidualMLP(enc_dim, flow=True).to("cuda")
trainer = FlowTrainer(
    encoded_dim=enc_dim,
    architecture=model,
    data_dir=str(ds_info["train_dir"]),
    batch_size=64,
    epochs=5,
    lr=1e-4,
    weight_decay=1e-4,
    val_split=0.2,
    noise_type="uniform_hypersphere",
    target_type="coordinate",
    device="cuda",
    sampling_mode="geo_det",
    use_ot_train=False,
    dataloader_num_workers=0,
)
models, train_loss, val_loss = trainer.train()
print(f"Training complete. Train losses: {[f'{l:.4f}' for l in train_loss]}")
print(f"Val losses: {[f'{l:.4f}' for l in val_loss]}")
