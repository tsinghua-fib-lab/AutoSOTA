#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import anndata as ad
import torch
from torch.utils.data import DataLoader, TensorDataset

from LCL_Model_Semi import BaseEncoder_ProjHead_MLP


def get_args():
    p = argparse.ArgumentParser(description="Extract base-encoder features (h) from a trained LCL checkpoint.")
    p.add_argument("--inputFilePath", required=True, help="Path to AnnData (.h5ad) for feature extraction")
    p.add_argument("--batch_size", type=int, default=50, help="Batch size for featurizing all cells")
    p.add_argument("--output_dir", required=True, help="Directory to save extracted feature .npy")
    p.add_argument("--resume_from_checkpoint", required=True, help="Path to trained Lightning .ckpt file")
    p.add_argument("--hidden_dims", default="1024,256,64", help='Comma-separated hidden dims, e.g. "1024,256,64"')
    p.add_argument("--embedding_size", type=int, default=32, help="Projection head output dimension (kept for init)")
    p.add_argument("--out_file_name", default=None, help="Optional output .npy name (default: auto)")
    return p.parse_args()


class Hparams:
    """Minimal config object compatible with BaseEncoder_ProjHead_MLP."""
    def __init__(self, args):
        self.input_dim = None  # set after reading AnnData
        self.hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
        self.embedding_size = int(args.embedding_size)

        # Unused for pure inference; kept to satisfy any constructor expectations
        self.unlabeled_per_batch = 0
        self.lambda_penalty = 0.0
        self.epochs = 1
        self.batch_size = int(args.batch_size)
        self.size_factor = 0.0
        self.temperature = 0.5

        # I/O
        self.out_dir = args.output_dir
        self.ckpt_path = args.resume_from_checkpoint
        self.file_path = args.inputFilePath


def _extract_model_state_dict_from_lightning_ckpt(ckpt_sd: dict) -> dict:
    """
    Lightning checkpoints often store model weights under 'model.' prefix because
    LightningModule has attribute self.model = BaseEncoder_ProjHead_MLP(...).
    BaseEncoder_ProjHead_MLP expects keys without that prefix.
    """
    if any(k.startswith("model.") for k in ckpt_sd.keys()):
        sd = {k[len("model."):]: v for k, v in ckpt_sd.items() if k.startswith("model.")}
        return sd
    return ckpt_sd


def load_checkpoint_into_bare_model(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint missing 'state_dict': {ckpt_path}")

    sd = _extract_model_state_dict_from_lightning_ckpt(ckpt["state_dict"])

    missing, unexpected = model.load_state_dict(sd, strict=False)

    print("=== STATE_DICT LOAD REPORT (bare BaseEncoder_ProjHead_MLP) ===")
    print("missing:", len(missing))
    print("unexpected:", len(unexpected))
    print("missing examples:", missing[:10])
    print("unexpected examples:", unexpected[:10])
    print("=============================================================")

    # For true safety, you generally want this to be 0/0
    return model


@torch.inference_mode()
def extract_base_features(model: torch.nn.Module, adata: ad.AnnData, batch_size: int, device: str):
    model.eval().to(device)

    X = adata.X
    try:
        X = X.toarray()
    except AttributeError:
        pass
    X = np.asarray(X, dtype=np.float32)

    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,   # safer on Colab; avoids weirdness
        drop_last=False
    )

    feats = []
    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        h = model.get_features(xb)   # BASE encoder features
        feats.append(h.detach().cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    return feats


def main():
    start_time = time.time()
    args = get_args()
    cfg = Hparams(args)

    # 1) load AnnData and set input_dim
    adata = ad.read_h5ad(cfg.file_path)
    cfg.input_dim = int(adata.n_vars)
    print(f"Detected input_dim = {cfg.input_dim}, n_obs = {adata.n_obs}")

    # 2) build bare model + load ckpt properly
    model = BaseEncoder_ProjHead_MLP(cfg)
    model = load_checkpoint_into_bare_model(model, cfg.ckpt_path)

    # 3) extract
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = extract_base_features(model, adata, batch_size=cfg.batch_size, device=device)
    print("Extracted base features shape:", feats.shape)

    # 4) save embeddings + cell_ids for alignment checks
    os.makedirs(cfg.out_dir, exist_ok=True)

    if args.out_file_name is None:
        out_file = f"scBaseEncoderFeat_h_bs{cfg.batch_size}.npy"
    else:
        out_file = args.out_file_name

    out_path = os.path.join(cfg.out_dir, out_file)
    np.save(out_path, feats)
    print(f"[SAVED] {out_path}")

    # Save cell ids so you can verify row ordering across runs
    out_ids = os.path.join(cfg.out_dir, f"cell_ids_bs{cfg.batch_size}.txt")
    with open(out_ids, "w") as f:
        for cid in adata.obs_names:
            f.write(str(cid) + "\n")
    print(f"[SAVED] {out_ids}")

    elapsed = (time.time() - start_time) / 60
    print(f"Done in {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()