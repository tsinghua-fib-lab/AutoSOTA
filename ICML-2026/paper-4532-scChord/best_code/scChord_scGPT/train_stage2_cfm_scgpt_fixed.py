#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 CFM training with fixed scGPT cell embeddings.

Design for rebuttal ablation:
- Fixed RNA condition c from precomputed scGPT embeddings
- Trainable modules: FlowNet + cond_null only
- Frozen modules: ProteinVAE, scGPT embeddings
"""

import argparse
import importlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    odeint = importlib.import_module("torchdiffeq").odeint
except Exception:
    odeint = None

from data import load_data, get_dataloader
from metrics import evaluate_predictions
from models import FlowNet, ProteinVAE
from visualization import save_evaluation_results


class DatasetWithEmbedding(Dataset):
    """Attach fixed per-cell embedding vectors to an existing dataset."""

    def __init__(self, base_dataset: Dataset, embeddings: np.ndarray):
        if len(base_dataset) != embeddings.shape[0]:
            raise ValueError(
                f"Dataset size ({len(base_dataset)}) != embeddings rows ({embeddings.shape[0]})"
            )
        self.base = base_dataset
        self.emb = torch.from_numpy(embeddings.astype(np.float32, copy=False))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["scgpt_emb"] = self.emb[idx]
        return item


class ODEFunc(nn.Module):
    """ODE RHS wrapper with optional CFG."""

    def __init__(
        self,
        flow_net: FlowNet,
        c: torch.Tensor,
        batch_id: torch.Tensor,
        cfg_scale: float = 1.0,
        use_cfg: bool = True,
    ):
        super().__init__()
        self.flow_net = flow_net
        self.c = c
        self.batch_id = batch_id
        self.cfg_scale = cfg_scale
        self.use_cfg = use_cfg

        bsz = c.shape[0]
        self.cond_null = flow_net.get_cond_null(bsz, c.device)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        t_batch = torch.full((bsz,), t.item(), device=x.device)

        if self.use_cfg and self.cfg_scale != 1.0:
            v_cond = self.flow_net(x, t_batch, self.c, self.batch_id)
            v_uncond = self.flow_net(x, t_batch, self.cond_null, self.batch_id)
            return v_uncond + self.cfg_scale * (v_cond - v_uncond)

        return self.flow_net(x, t_batch, self.c, self.batch_id)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_embedding_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=False)
    if "embeddings" not in data:
        raise ValueError("NPZ missing key: embeddings")
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)

    obs_names = None
    if "obs_names" in data:
        obs_names = np.asarray(data["obs_names"]).astype(str)
    return embeddings, obs_names


def train_epoch_fixed(
    vae: ProteinVAE,
    flow_net: FlowNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    p_uncond: float = 0.15,
) -> dict:
    flow_net.train()
    vae.eval()

    total_loss = 0.0
    total_cfm = 0.0
    n_batches = 0

    for batch in dataloader:
        prot_norm = batch["prot_norm"].to(device)
        batch_id = batch["batch_id"].to(device)
        c_full = batch["scgpt_emb"].to(device)
        bsz = c_full.shape[0]

        optimizer.zero_grad()

        drop_mask = torch.rand(bsz, device=device) < p_uncond
        cond_null = flow_net.get_cond_null(bsz, device)
        c_used = torch.where(drop_mask.unsqueeze(-1), cond_null, c_full)

        with torch.no_grad():
            mu_z, logvar_z = vae.encode(prot_norm, batch_id)
            x1 = vae.reparameterize(mu_z, logvar_z)

        x0 = torch.randn_like(x1)
        t = torch.rand(bsz, device=device)

        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        u_t = x1 - x0

        v = flow_net(x_t, t, c_used, batch_id)
        l_cfm = ((v - u_t) ** 2).mean()

        l_cfm.backward()
        torch.nn.utils.clip_grad_norm_(flow_net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(l_cfm.item())
        total_cfm += float(l_cfm.item())
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "cfm": total_cfm / max(n_batches, 1),
        "cons": 0.0,
    }


@torch.no_grad()
def validate_fixed(
    vae: ProteinVAE,
    flow_net: FlowNet,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    flow_net.eval()
    vae.eval()

    total_loss = 0.0
    total_cfm = 0.0
    n_batches = 0

    for batch in dataloader:
        prot_norm = batch["prot_norm"].to(device)
        batch_id = batch["batch_id"].to(device)
        c_full = batch["scgpt_emb"].to(device)
        bsz = c_full.shape[0]

        mu_z, logvar_z = vae.encode(prot_norm, batch_id)
        x1 = vae.reparameterize(mu_z, logvar_z)

        x0 = torch.randn_like(x1)
        t = torch.rand(bsz, device=device)

        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        u_t = x1 - x0

        v = flow_net(x_t, t, c_full, batch_id)
        l_cfm = ((v - u_t) ** 2).mean()

        total_loss += float(l_cfm.item())
        total_cfm += float(l_cfm.item())
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "cfm": total_cfm / max(n_batches, 1),
        "cons": 0.0,
    }


@torch.no_grad()
def inference_and_evaluate_fixed(
    vae: ProteinVAE,
    flow_net: FlowNet,
    dataloader: DataLoader,
    device: torch.device,
    cfg_scale: float = 2.0,
    ode_method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    flow_net.eval()
    vae.eval()

    all_preds = []
    all_truth = []
    t_span = torch.tensor([0.0, 1.0], device=device)

    for batch in tqdm(dataloader, desc="Inference"):
        prot_norm = batch["prot_norm"].to(device)
        batch_id = batch["batch_id"].to(device)
        c = batch["scgpt_emb"].to(device)
        bsz = c.shape[0]

        x0 = torch.randn(bsz, vae.dz, device=device)

        ode_func = ODEFunc(
            flow_net=flow_net,
            c=c,
            batch_id=batch_id,
            cfg_scale=cfg_scale,
            use_cfg=True,
        )

        x_traj = odeint(
            ode_func,
            x0,
            t_span,
            method=ode_method,
            rtol=rtol,
            atol=atol,
        )
        z_hat = x_traj[-1]

        y_hat = vae.decode(z_hat, batch_id)
        all_preds.append(y_hat.cpu().numpy())
        all_truth.append(prot_norm.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    ground_truth = np.concatenate(all_truth, axis=0)
    return predictions, ground_truth


def main(args):
    set_seed(args.seed)

    if odeint is None:
        raise ImportError(
            "torchdiffeq is required for Stage2 ODE integration. "
            "Install it in the anno1 environment before running this script."
        )

    env_name = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    print(f"[env] CONDA_DEFAULT_ENV={env_name}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.test_data_path is not None:
        raise ValueError("Fixed scGPT script currently supports single-dataset random split only.")

    print("\n" + "=" * 60)
    print("Using SINGLE-DATASET mode (random split)")
    print(f"Data: {args.data_path}")
    print("=" * 60 + "\n")

    train_dataset, test_dataset, data_info = load_data(
        args.data_path,
        n_top_genes=args.n_top_genes,
        train_ratio=args.train_ratio,
        random_state=args.seed,
    )

    emb_npz = Path(args.scgpt_embeddings_path)
    print(f"Loading fixed scGPT embeddings from: {emb_npz}")
    embeddings, _ = load_embedding_npz(emb_npz)

    expected_n = len(data_info["train_idx"]) + len(data_info["test_idx"])
    if embeddings.shape[0] != expected_n:
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) != total cells ({expected_n})."
        )

    train_emb = embeddings[data_info["train_idx"]]
    test_emb = embeddings[data_info["test_idx"]]

    if args.dc <= 0:
        args.dc = int(train_emb.shape[1])
    if train_emb.shape[1] != args.dc:
        raise ValueError(
            f"scGPT embedding dim ({train_emb.shape[1]}) != dc ({args.dc}). "
            "Set --dc to embedding dimension."
        )

    train_dataset = DatasetWithEmbedding(train_dataset, train_emb)
    test_dataset = DatasetWithEmbedding(test_dataset, test_emb)

    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    vae_ckpt_path = Path(args.vae_path)
    print(f"\nLoading VAE from {vae_ckpt_path}")
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae_config = vae_ckpt["config"]

    vae = ProteinVAE(
        n_proteins=vae_config["n_proteins"],
        dz=vae_config["dz"],
        hidden_dims=vae_config["hidden_dims"],
        batch_emb_dim=vae_config["batch_emb_dim"],
        n_batches=vae_config["n_batches"],
        beta_kl=vae_config["beta_kl"],
        learnable_dispersion=True,
        dist_type=vae_config.get("dist_type", "Gaussian"),
    ).to(device)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()

    for param in vae.parameters():
        param.requires_grad = False

    print(f"VAE loaded. Epoch: {vae_ckpt['epoch']}, Val loss: {vae_ckpt['val_loss']:.4f}")

    flow_net = FlowNet(
        dz=vae_config["dz"],
        dc=args.dc,
        hidden_dim=args.flow_hidden_dim,
        n_blocks=args.flow_n_blocks,
        time_emb_dim=64,
        batch_emb_dim=args.batch_emb_dim,
        n_batches=args.n_batches,
        dropout=0.1,
    ).to(device)

    print("\nFixed scGPT condition mode: enabled")
    print(f"scGPT condition dim: {args.dc}")
    print(f"FlowNet parameters: {sum(p.numel() for p in flow_net.parameters()):,}")

    optimizer = torch.optim.AdamW(
        flow_net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

    warmup_epochs = 25
    main_epochs = max(args.epochs - warmup_epochs, 1)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=args.lr * 0.01),
        ],
        milestones=[warmup_epochs],
    )

    best_val_loss = float("inf")
    best_epoch = 0

    print("\n" + "=" * 60)
    print("Stage 2: Training CFM (Fixed scGPT embedding + FlowNet)")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch_fixed(
            vae,
            flow_net,
            train_loader,
            optimizer,
            device,
            p_uncond=args.p_uncond,
        )

        val_metrics = validate_fixed(
            vae,
            flow_net,
            val_loader,
            device,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f} (CFM: {train_metrics['cfm']:.4f}, Cons: {train_metrics['cons']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} (CFM: {val_metrics['cfm']:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "flow_net_state_dict": flow_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "config": {
                        "dc": args.dc,
                        "flow_hidden_dim": args.flow_hidden_dim,
                        "flow_n_blocks": args.flow_n_blocks,
                        "batch_emb_dim": args.batch_emb_dim,
                        "n_batches": args.n_batches,
                        "dz": vae_config["dz"],
                        "use_scgpt_embeddings": True,
                        "scgpt_embeddings_path": str(emb_npz.resolve()),
                    },
                },
                output_dir / "flow_best.pt",
            )

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "flow_net_state_dict": flow_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                output_dir / f"flow_epoch_{epoch:03d}.pt",
            )

    torch.save(
        {
            "epoch": args.epochs,
            "flow_net_state_dict": flow_net.state_dict(),
            "config": {
                "dc": args.dc,
                "flow_hidden_dim": args.flow_hidden_dim,
                "flow_n_blocks": args.flow_n_blocks,
                "batch_emb_dim": args.batch_emb_dim,
                "n_batches": args.n_batches,
                "dz": vae_config["dz"],
                "use_scgpt_embeddings": True,
                "scgpt_embeddings_path": str(emb_npz.resolve()),
            },
        },
        output_dir / "flow_final.pt",
    )

    print("\n" + "=" * 60)
    print(f"Training completed. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Evaluating best model on test split...")
    print("=" * 60)

    best_ckpt = torch.load(output_dir / "flow_best.pt", map_location=device)
    flow_net.load_state_dict(best_ckpt["flow_net_state_dict"])

    predictions, ground_truth = inference_and_evaluate_fixed(
        vae,
        flow_net,
        val_loader,
        device,
        cfg_scale=args.cfg_scale,
        ode_method=args.ode_method,
        rtol=args.ode_rtol,
        atol=args.ode_atol,
    )

    results = evaluate_predictions(
        predictions,
        ground_truth,
        protein_names=data_info["protein_names"],
        verbose=True,
    )

    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "ground_truth.npy", ground_truth)

    if args.export_dataset1_pred_path:
        export_path = Path(args.export_dataset1_pred_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(export_path, predictions)
        print(f"Exported dataset1 prediction file: {export_path}")

    print("\n" + "=" * 60)
    print("Generating visualization plots...")
    print("=" * 60)

    save_evaluation_results(
        results=results,
        save_dir=output_dir / "figures",
        protein_names=data_info["protein_names"],
        title_prefix="scBridge-Flow-scGPT-fixed",
    )

    print(f"\nAll results saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 CFM with fixed scGPT embeddings")

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/dataset.h5ad",
        help="Path to input H5AD data file",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="Unused in fixed-scGPT mode; reserved for future extension",
    )
    parser.add_argument("--n_top_genes", type=int, default=1000)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--vae_path", type=str, default="./outputs/stage1/vae_best.pt")
    parser.add_argument(
        "--scgpt_embeddings_path",
        type=str,
        required=True,
        help="Path to .npz from extract_scgpt_cell_embeddings.py",
    )

    parser.add_argument(
        "--dc",
        type=int,
        default=-1,
        help="Condition dimension. Set -1 to auto from scGPT embeddings.",
    )
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_n_blocks", type=int, default=4)
    parser.add_argument("--batch_emb_dim", type=int, default=8)
    parser.add_argument("--n_batches", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--p_uncond", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument(
        "--ode_method",
        type=str,
        default="dopri5",
        choices=["dopri5", "dopri8", "rk4", "euler", "midpoint", "heun3", "adaptive_heun"],
    )
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument("--ode_atol", type=float, default=1e-5)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./outputs/stage2_scgpt_fixed")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument(
        "--export_dataset1_pred_path",
        type=str,
        default="",
        help="Optional path to save predictions as a standalone .npy (for downstream benchmarks)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
