import os
import numpy as np
import torch
import anndata as ad

from captum.attr import IntegratedGradients
from LCL_Main_Semi import scContraLearn, load_checkpoint, Hparams, get_args


# -------------------------
# Cosine-to-centroid wrapper
# -------------------------
class LCLCosineToCentroid(torch.nn.Module):
    """
    Scalar model: F(x) = cosine( z(x), centroid )
    where z(x) = g(f(x)) from your trained LCL, using BOTH f and g together.
    """
    def __init__(self, lightning_model, centroid_vec: torch.Tensor):
        super().__init__()
        self.lcl = lightning_model
        centroid_vec = centroid_vec / (centroid_vec.norm() + 1e-12)
        self.register_buffer("centroid", centroid_vec)

    def forward(self, x):
        # x: (B, G)
        z = self.lcl.model(x)  # (B, D) == g(f(x))
        z = z / (z.norm(dim=1, keepdim=True) + 1e-12)
        return (z * self.centroid.unsqueeze(0)).sum(dim=1)  # (B,)


# -------------------------
# Helpers
# -------------------------
def to_dense_float32(X):
    """Convert AnnData X to dense float32 numpy array."""
    try:
        X = X.toarray()
    except AttributeError:
        pass
    return np.asarray(X, dtype=np.float32)


@torch.no_grad()
def compute_all_z(model, X_torch, batch_size=1024):
    """
    Compute z = g(f(x)) for all cells in X_torch without gradients.
    Returns Z on CPU: (N, D)
    """
    model.eval()
    Z_chunks = []
    for i in range(0, X_torch.shape[0], batch_size):
        z = model.model(X_torch[i:i + batch_size])  # (B, D)
        Z_chunks.append(z.detach().cpu())
    return torch.cat(Z_chunks, dim=0)


def make_clone_maps(clone_ids):
    """Map clone_id -> list of indices."""
    clone_to_indices = {}
    for i, cid in enumerate(clone_ids):
        clone_to_indices.setdefault(cid, []).append(i)
    return clone_to_indices


def pick_clones(clone_to_indices, min_cells=10, max_clones=30, seed=0):
    """
    Keep clones with >= min_cells cells, then pick up to max_clones.
    """
    rng = np.random.default_rng(seed)
    eligible = [c for c, idxs in clone_to_indices.items() if len(idxs) >= min_cells]
    if len(eligible) == 0:
        raise ValueError(f"No clones found with >= {min_cells} cells.")
    if len(eligible) > max_clones:
        eligible = list(rng.choice(eligible, size=max_clones, replace=False))
    return eligible


def run_ig_for_clone(model, X_torch, clone_indices, centroid_vec, baseline_mode="zero",
                     n_steps=100, method="gausslegendre", cells_per_clone=200, seed=0):
    """
    Run IG for up to cells_per_clone cells in one clone.
    Returns:
      attr: (B, G) on CPU
      delta: (B,) on CPU
      chosen_indices: list of original cell indices used
    """
    rng = np.random.default_rng(seed)
    if len(clone_indices) > cells_per_clone:
        chosen_indices = list(rng.choice(clone_indices, size=cells_per_clone, replace=False))
    else:
        chosen_indices = list(clone_indices)

    inputs = X_torch[chosen_indices].detach()
    inputs.requires_grad_(True)

    # baseline
    if baseline_mode == "zero":
        baselines = torch.zeros_like(inputs)
    elif baseline_mode == "mean":
        mean_expr = X_torch.mean(dim=0, keepdim=True).detach()
        baselines = mean_expr.repeat(inputs.shape[0], 1)
    else:
        raise ValueError("baseline_mode must be 'zero' or 'mean'.")

    scalar_model = LCLCosineToCentroid(model, centroid_vec).eval()
    ig = IntegratedGradients(scalar_model)

    attr, delta = ig.attribute(
        inputs=inputs,
        baselines=baselines,
        n_steps=n_steps,
        method=method,
        return_convergence_delta=True
    )

    return attr.detach().cpu(), delta.detach().cpu(), chosen_indices


def main():
    # -------------------------
    # Args + config (your existing parser)
    # -------------------------
    args = get_args()
    config = Hparams(args)

    # -------------------------
    # Load CellTag-multi FIRST so we can set input_dim
    # -------------------------
    adata = ad.read_h5ad(args.inputFilePath)
    if "clone_id" not in adata.obs:
        raise KeyError('adata.obs does not contain "clone_id".')

    # ✅ critical fix: set input_dim before creating the model
    config.input_dim = int(adata.n_vars)

    # -------------------------
    # User-set knobs (edit these)
    # -------------------------
    min_cells_per_clone = 10
    max_clones = 50
    cells_per_clone = 200
    baseline_mode = "zero"   # "zero" or "mean"
    n_steps = 100
    embed_batch_size = 2048
    seed = 0

    # -------------------------
    # Load model from checkpoint
    # -------------------------
    model = scContraLearn(config)
    model = load_checkpoint(model, args.resume_from_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # -------------------------
    # Prepare data tensors
    # -------------------------
    X = to_dense_float32(adata.X)  # (N, G)
    gene_names = np.array(adata.var_names) if adata.var_names is not None else None
    clone_ids = adata.obs["clone_id"].astype(str).values

    X_torch = torch.from_numpy(X).to(device)

    # -------------------------
    # Compute z for ALL cells (no grad) and centroids per clone
    # -------------------------
    print("Computing z = g(f(x)) for all cells...")
    Z_cpu = compute_all_z(model, X_torch, batch_size=embed_batch_size)  # (N, D) on CPU

    clone_to_indices = make_clone_maps(clone_ids)
    chosen_clones = pick_clones(
        clone_to_indices,
        min_cells=min_cells_per_clone,
        max_clones=max_clones,
        seed=seed
    )

    centroids = {}
    for cid in chosen_clones:
        idxs = clone_to_indices[cid]
        c = Z_cpu[idxs].mean(dim=0)  # (D,)
        centroids[cid] = c

    # -------------------------
    # Run IG per clone
    # -------------------------
    out_dir = getattr(config, "out_dir", None) or getattr(args, "output_dir", None) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, "ig_cosine_centroid")

    print(f"Running IG on {len(chosen_clones)} clones...")

    all_gene_scores = []
    clone_results = {}

    for j, cid in enumerate(chosen_clones, start=1):
        print(f"[{j}/{len(chosen_clones)}] clone={cid}  n_cells={len(clone_to_indices[cid])}")

        centroid_vec = centroids[cid].to(device)

        attr, delta, used_indices = run_ig_for_clone(
            model=model,
            X_torch=X_torch,
            clone_indices=clone_to_indices[cid],
            centroid_vec=centroid_vec,
            baseline_mode=baseline_mode,
            n_steps=n_steps,
            method="gausslegendre",
            cells_per_clone=cells_per_clone,
            seed=seed + j
        )

        gene_scores = attr.abs().mean(dim=0).numpy()  # (G,)
        all_gene_scores.append(gene_scores)

        clone_results[cid] = {
            "delta_mean": float(delta.mean().item()),
            "n_used": len(used_indices),
            "used_indices": np.array(used_indices, dtype=int),
        }

        # save per-clone top genes
        if gene_names is not None and len(gene_names) == gene_scores.shape[0]:
            order = np.argsort(-gene_scores)
            top = 50
            with open(f"{out_prefix}_clone_{cid}_top{top}.txt", "w") as f:
                for g, s in zip(gene_names[order][:top], gene_scores[order][:top]):
                    f.write(f"{g}\t{s}\n")

    # -------------------------
    # Aggregate across clones (global gene importance)
    # -------------------------
    all_gene_scores = np.vstack(all_gene_scores)  # (n_clones, G)
    global_gene_scores = np.mean(all_gene_scores, axis=0)  # (G,)

    np.save(out_prefix + "_global_gene_scores.npy", global_gene_scores)

    if gene_names is not None and len(gene_names) == global_gene_scores.shape[0]:
        order = np.argsort(-global_gene_scores)
        top = 100
        with open(out_prefix + f"_global_top{top}.txt", "w") as f:
            for g, s in zip(gene_names[order][:top], global_gene_scores[order][:top]):
                f.write(f"{g}\t{s}\n")

    with open(out_prefix + "_clone_delta_summary.txt", "w") as f:
        f.write("clone_id\tn_used\tdelta_mean\n")
        for cid, info in clone_results.items():
            f.write(f"{cid}\t{info['n_used']}\t{info['delta_mean']}\n")

    print("Done.")
    print("Saved:", out_prefix + "_global_gene_scores.npy")
    print("Saved:", out_prefix + "_global_top100.txt (if gene names available)")
    print("Saved:", out_prefix + "_clone_delta_summary.txt")
    print(f"Baseline={baseline_mode}, n_steps={n_steps}")


if __name__ == "__main__":
    main()