# feature_extraction.py
# Extract PROJECTION-HEAD embeddings from a trained checkpoint for a single .h5ad

import os
import argparse
import numpy as np
import anndata as ad
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

# your modules
from LCL_Model_Semi import BaseEncoder_ProjHead_MLP, ContrastiveLoss


def parse_args():
    p = argparse.ArgumentParser(description="Extract projection-head embeddings from a trained checkpoint.")
    p.add_argument('--inputFilePath', type=str, required=True, help='Anndata .h5ad file to embed')
    p.add_argument('--batch_size', type=int, default=256, help='Inference batch size')
    p.add_argument('--output_dir', type=str, required=True, help='Where to save embeddings')
    p.add_argument('--resume_from_checkpoint', type=str, required=True, help='Path to Lightning checkpoint (.ckpt)')
    p.add_argument('--out_file_name', type=str, required=True,default='proj_embed.npy')

    # Optional overrides (only used if not found inside the checkpoint)
    p.add_argument('--hidden_dims', type=str, default=None, help='Fallback hidden dims, e.g. "1024,256,64"')
    p.add_argument('--embedding_size', type=int, default=None, help='Fallback projection dimension, e.g. 32')
    p.add_argument('--temperature', type=float, default=0.5)   # not used for inference; kept for config completeness
    p.add_argument('--lambda_penalty', type=float, default=0.0)  # not used for inference
    return p.parse_args()


def infer_input_dim_from_h5ad(h5ad_path: str) -> int:
    adata = ad.read_h5ad(h5ad_path)
    return int(adata.n_vars)


def _parse_hidden_dims(s: str):
    return [int(x) for x in s.split(',')] if s else None


def load_hparams_from_ckpt(ckpt_path: str):
    """Try to recover hidden_dims / embedding_size from the checkpoint 'hyper_parameters' section."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    hp = ckpt.get('hyper_parameters', {}) or {}

    # Common places people store these
    # Try direct keys first, then nested under 'config' or 'hparams' like objects
    candidates = [hp, hp.get('config', {}), hp.get('hparams', {}), hp.get('cfg', {})]

    hidden_dims = None
    embedding_size = None

    for d in candidates:
        if not isinstance(d, dict):
            continue
        if hidden_dims is None and 'hidden_dims' in d:
            hidden_dims = d['hidden_dims']
            if isinstance(hidden_dims, str):
                hidden_dims = _parse_hidden_dims(hidden_dims)
        if embedding_size is None and 'embedding_size' in d:
            embedding_size = int(d['embedding_size'])
    return hidden_dims, embedding_size


class Hparams:
    """Minimal config object compatible with your model constructor."""
    def __init__(self, input_dim, hidden_dims, embedding_size, batch_size, temperature, lambda_penalty):
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.embedding_size = int(embedding_size)
        self.batch_size = int(batch_size)
        self.temperature = float(temperature)
        self.lambda_penalty = float(lambda_penalty)

        # The rest are unused for inference but expected by some constructors
        self.unlabeled_per_batch = 0
        self.epochs = 1
        self.size_factor = 0.0
        self.train_test_ratio = 1.0
        self.patience = 0
        self.min_delta = 0.0
        self.seed = 3407
        self.batch_seed = 17
        self.random_seed = 42
        self.train_test_seed = 42
        self.gradient_accumulation_steps = 1
        self.lr = 3e-4
        self.weight_decay = 1e-6
        self.cuda = True
        self.load = True
        self.train_test = False
        self.save = ""        # not used
        self.checkpoint_path = ""
        self.out_dir = ""
        self.file_path = ""
        self.testFilePath = ""


class scContraLearn(pl.LightningModule):
    """Minimal LightningModule wrapper to hold your model at inference time."""
    def __init__(self, config: Hparams):
        super().__init__()
        self.config = config
        self.model = BaseEncoder_ProjHead_MLP(config)
        # loss unused; keep for completeness
        self.loss_fn = ContrastiveLoss(
            batch_size=config.batch_size,
            temperature=config.temperature,
            lambda_penalty=config.lambda_penalty
        )
    def forward(self, x):
        return self.model(x)


def load_weights(lightning_module: pl.LightningModule, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    lightning_module.load_state_dict(ckpt['state_dict'], strict=False)
    missing, unexpected = lightning_module.load_state_dict(ckpt['state_dict'], strict=False)

    print("=== STATE_DICT LOAD REPORT ===")
    print("missing:", len(missing))
    print("unexpected:", len(unexpected))
    print("missing examples:", missing[:10])
    print("unexpected examples:", unexpected[:10])
    print("================================")
    print(f"[OK] Loaded checkpoint: {ckpt_path}")


@torch.no_grad()
def embed_h5ad_with_projection(lightning_module: pl.LightningModule,
                               h5ad_path: str,
                               batch_size: int = 256,
                               device: str = None):
    """Returns (embeddings, cell_ids). Embeddings are projection-head output."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lightning_module.eval().to(device)

    adata = ad.read_h5ad(h5ad_path)
    X = adata.X
    # handle sparse
    try:
        X = X.toarray()
    except AttributeError:
        pass
    X = np.asarray(X, dtype=np.float32)

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

    outs = []
    for (Z,) in dl:
        Z = Z.to(device).float()
        E = lightning_module.model(Z)  # PROJECTION-HEAD output
        outs.append(E.detach().cpu().numpy())
    emb = np.concatenate(outs, axis=0)

    # try to return cell ids (obs_names)
    try:
        cell_ids = np.array(adata.obs_names)
    except Exception:
        cell_ids = None
    return emb, cell_ids


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) infer input dim from the .h5ad
    input_dim = infer_input_dim_from_h5ad(args.inputFilePath)

    # 2) recover hidden_dims/embedding_size from checkpoint if possible
    ckpt_hidden, ckpt_embed = load_hparams_from_ckpt(args.resume_from_checkpoint)

    # fallbacks: CLI overrides, then sensible defaults
    hidden_dims = ckpt_hidden or _parse_hidden_dims(args.hidden_dims) or [1024, 256, 64]
    embedding_size = ckpt_embed or args.embedding_size or 32

    print("--------------------- PROJECTION EMBEDDING EXTRACTION ---------------------")
    print(f"inputFilePath: {args.inputFilePath}")
    print(f"resume_from_checkpoint: {args.resume_from_checkpoint}")
    print(f"inferred input_dim: {input_dim}")
    print(f"hidden_dims: {hidden_dims}")
    print(f"embedding_size: {embedding_size}")
    print(f"batch_size (inference): {args.batch_size}")

    # 3) build minimal config + model and load weights
    cfg = Hparams(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        embedding_size=embedding_size,
        batch_size=args.batch_size,
        temperature=args.temperature,
        lambda_penalty=args.lambda_penalty
    )
    lm = scContraLearn(cfg)
    load_weights(lm, args.resume_from_checkpoint)

    # 4) embed
    emb, cell_ids = embed_h5ad_with_projection(
        lm,
        args.inputFilePath,
        batch_size=args.batch_size
    )

    # 5) save outputs
    # out_npy = os.path.join(args.output_dir, "proj_embed.npy")
    out_npy = os.path.join(args.output_dir, args.out_file_name)
    np.save(out_npy, emb)
    print(f"[SAVED] {out_npy}  (shape={emb.shape})")

    if cell_ids is not None:
        out_ids = os.path.join(args.output_dir, "cell_ids.txt")
        with open(out_ids, "w") as f:
            for cid in cell_ids:
                f.write(str(cid) + "\n")
        print(f"[SAVED] {out_ids}  (n={len(cell_ids)})")


if __name__ == "__main__":
    main()