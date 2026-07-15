
import os
# --- Sanitize env BEFORE importing matplotlib ---------------------------------
# If Condor sets something like "MPLBACKEND=Agg WANDB_MODE=offline" as a SINGLE
# value, Matplotlib will crash at import time. Force a safe backend here.
mb = os.environ.get("MPLBACKEND")
if mb and (" " in mb or ";" in mb):
    # Bad combined value; remove it entirely so we can set a clean default.
    os.environ.pop("MPLBACKEND", None)
# Default to a safe headless backend if not set
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn as nn
import matplotlib
# Now import pyplot safely
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import math
import wandb

from models import NeuroLDS
from utils import set_seed, get_device, make_sequence, get_discrepancy


def main(
    dim: int = 2,
    hidden: int = 256,
    layers: int = 2,
    K: int = 8,
    device: str = 'auto',
    seq_total_length: int = 100,
    epochs: int = 800,
    lr: float = 1e-3,
    pretrain_epochs: int = 800,
    fine_tune_lr: float = 3e-4,
    seed: int = 0,
    train_seq: str = 'sobol',
    name: str = 'run',
    verbose: bool = False,
    warmup_epochs: int = 25,
    min_finetune_lr: float = 5e-6,
    final_lr_ratio: float = 0.1,
    disc_loss: str = 'star',
    pre_eps: float = 0.0,
    ft_eps: float = 0.0,
    gamma: list = None,
):
    # Hyperparameters (ICLR‑ready: explicit, compact)
    DIM               = dim
    HIDDEN            = hidden
    LAYERS            = layers
    K_POSENC         = K

    SEQ_TOTAL_LENGTH  = seq_total_length
    EPOCHS            = epochs
    LR                = lr
    PRETRAIN_EPOCHS   = pretrain_epochs
    FINE_TUNE_LR      = fine_tune_lr
    WARMUP_EPOCHS     = warmup_epochs
    MIN_FINETUNE_LR   = min_finetune_lr
    VERBOSE           = verbose
    FINAL_LR_RATIO    = final_lr_ratio
    TRAIN_SEQ         = train_seq
    # Anchor results to the repository root (one level above this file's folder)
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT_ROOT = os.path.join(REPO_ROOT, 'results')
    OUT_DIR = os.path.join(OUT_ROOT, name)
    base_name = os.path.basename(name)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Repro + device (do before logging config)
    set_seed(seed)
    DEVICE = get_device(device)
    DEVICE_STR = str(DEVICE)
    if VERBOSE:
        print(f"Using device: {DEVICE}")
    gamma_t = None
    if gamma is not None:
        if len(gamma) != DIM:
            raise ValueError(f"gamma length {len(gamma)} != dim {DIM}")
        gamma_t = torch.tensor(gamma, dtype=torch.float32, device=DEVICE)

    # Default W&B to disabled unless explicitly overridden in the environment
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "disabled"

    # Silence W&B console output during smoke tests (when not verbose)
    if not VERBOSE:
        os.environ["WANDB_SILENT"] = "true"

    # Weights & Biases: initialize a run
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "mpmc"),
        name=name,
        config={
            "dim": DIM,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "K": K_POSENC,
            "device": DEVICE_STR,
            "seq_total_length": SEQ_TOTAL_LENGTH,
            "epochs": EPOCHS,
            "lr": LR,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "fine_tune_lr": FINE_TUNE_LR,
            "warmup_epochs": WARMUP_EPOCHS,
            "min_finetune_lr": MIN_FINETUNE_LR,
            "final_lr_ratio": FINAL_LR_RATIO,
            "train_seq": TRAIN_SEQ,
            "seed": seed,
            "disc_loss": disc_loss,
            "gamma": gamma if gamma is not None else None,
        },
    )

    # Choose discrepancy family (seq + final)
    SEQ_LOSS_FN, _, LOSS_LABEL = get_discrepancy(disc_loss, gamma=gamma_t)
    wandb.config.update({"disc_loss": LOSS_LABEL}, allow_val_change=True)

    model = NeuroLDS(dim=DIM, K=K_POSENC, width=HIDDEN, depth=LAYERS).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.98))

    # Track parameters only (avoid heavy gradient hooks)
    wandb.watch(model, log=None)

    criterion = nn.MSELoss()
    BURN = 128
    # Build ground-truth target (deterministic) for pretraining
    target_np = make_sequence(TRAIN_SEQ, SEQ_TOTAL_LENGTH, DIM, burn=BURN, scramble=False, seed=seed)
    target_t  = torch.tensor(target_np, dtype=torch.float32, device=DEVICE)
    N = SEQ_TOTAL_LENGTH
    idx_all = torch.arange(N, device=DEVICE, dtype=torch.long)

    # --- Non-autoregressive pretraining: learn index -> point mapping on first N points ---
    # Cosine LR decay for pretraining (eta_min = 0.1 * LR)
    pre_eta_min = LR * 0.1
    pre_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, PRETRAIN_EPOCHS), eta_min=pre_eta_min)

    for ep in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        pred = model(idx_all, N)                 # [N, dim]
        loss_pre = criterion(pred, target_t)     # simple MSE to Sobol/Halton
        opt.zero_grad()
        loss_pre.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        pre_scheduler.step()

        # W&B: log pretraining metrics
        wandb.log({
            "pretrain/mse": float(loss_pre.item()),
            "pretrain/epoch": ep,
            "pretrain/lr": float(opt.param_groups[0]['lr']),
        })

        # Early stop on MSE if requested
        if pre_eps > 0.0 and float(loss_pre.item()) < pre_eps:
            if VERBOSE:
                print(f"[Pretrain] Early stop at epoch {ep} (MSE {loss_pre.item():.6g} < {pre_eps})")
            break

        if VERBOSE and (ep % 10 == 0 or ep == 1):
            print(f"[Pretrain] Epoch {ep:05d} | MSE: {loss_pre.item():.6f} | lr={opt.param_groups[0]['lr']:.6g}")

    # --- Plot: discrepancy after pre-training (model vs chosen ground-truth sequence) ---
    with torch.no_grad():
        pred_all = model(idx_all, N)  # [N, dim]
        # Compute full prefix curves in one call (O(N^2) once, instead of O(N) times)
        D_model_seq = SEQ_LOSS_FN(pred_all)  # [N]
        gt_t = torch.tensor(target_np, dtype=torch.float32, device=DEVICE)
        D_gt_seq = SEQ_LOSS_FN(gt_t)         # [N]
        lengths = np.arange(2, N + 1)
        # Drop the length-1 prefix to align with lengths starting at 2
        model_curve = D_model_seq[1:].detach().cpu().numpy().tolist()
        gt_curve = D_gt_seq[1:].detach().cpu().numpy().tolist()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(lengths, gt_curve, label=f'{TRAIN_SEQ.capitalize()} ground truth')
    ax.semilogy(lengths, model_curve, label=f'Pre-trained model (NeuroLDS, {LOSS_LABEL})')
    ax.set_xlabel('Sequence length P')
    ax.set_ylabel(f'{LOSS_LABEL} (log scale)')
    ax.set_title('After pre-training: discrepancy vs ground truth')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    pre_fig_path = os.path.join(OUT_DIR, f"{base_name}_pretrain.png")
    wandb.log({"pretraining/plot": wandb.Image(fig)})
    fig.savefig(pre_fig_path, bbox_inches='tight', dpi=200)
    if VERBOSE:
        print(f"Saved pre-training discrepancy plot to {pre_fig_path}")

    # reset optimizer for discrepancy fine-tuning (fresh Adam moments)
    opt = torch.optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=1e-4, betas=(0.9, 0.98))

    # LR warmup + cosine decay to stabilize early discrepancy training
    base_lr = FINE_TUNE_LR
    min_factor = MIN_FINETUNE_LR / base_lr if base_lr > 0 else 0.0
    # Slightly higher floor to keep learning late in training
    eta_min = min(0.5, FINAL_LR_RATIO * 1.5)
    def _lr_factor(epoch_idx: int) -> float:
        if WARMUP_EPOCHS > 0 and epoch_idx < WARMUP_EPOCHS:
            return min_factor + (1.0 - min_factor) * (epoch_idx + 1) / float(WARMUP_EPOCHS)
        remain = max(1, EPOCHS - WARMUP_EPOCHS)
        p = min(1.0, (epoch_idx - WARMUP_EPOCHS) / float(remain))
        return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_factor)

    # ─── Discrepancy fine-tuning: ALL prefixes (shared-compute, O(N^2)) ─
    for ep in range(1, EPOCHS + 1):
        model.train()
        pred_all = model(idx_all, N)  # [N, dim] in [0,1]

        D2_all = SEQ_LOSS_FN(pred_all)

        loss = D2_all[1:].mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        last_disc = float(D2_all[-1].item())

        # W&B: log training metrics
        try:
            cur_lr = scheduler.get_last_lr()[0]
        except Exception:
            cur_lr = FINE_TUNE_LR
        wandb.log({
            "train/loss_total": float(loss.item()),
            "train/epoch": ep,
            "train/lr": float(cur_lr),
            "train/final_disc": last_disc,
        })

        if ft_eps > 0.0 and last_disc < ft_eps:
            if VERBOSE:
                print(f"[Disc-FT] Early stop at epoch {ep} (final {LOSS_LABEL} {last_disc:.6g} < {ft_eps})")
            break

        if VERBOSE and (ep % 10 == 0 or ep == 1):
            print(
                f"[Disc-FT] Epoch {ep:03d} | loss={loss.item():.6f} | lr={cur_lr:.6g}"
            )

    # ─── Discrepancy curves across prefix lengths (Sobol vs Halton vs Model + Scrambled Sobol mean) ─
    model.eval()
    lengths = np.arange(2, SEQ_TOTAL_LENGTH + 1)
    SCRAMBLED_B = 32
    with torch.no_grad():
        # Model curve in one pass
        idx_all_full = torch.arange(SEQ_TOTAL_LENGTH, device=DEVICE, dtype=torch.long)
        pred_all = model(idx_all_full, SEQ_TOTAL_LENGTH)  # [N, dim]
        D_model_seq = SEQ_LOSS_FN(pred_all)               # [N]
        model_curve = D_model_seq[1:].detach().cpu().numpy().tolist()
        # Keep a copy for later text dump
        pred_all_np = pred_all.detach().cpu().numpy()

        # Deterministic Sobol/Halton curves in one pass each
        sobol_full_np  = make_sequence("sobol",  SEQ_TOTAL_LENGTH, DIM, burn=BURN, scramble=False, seed=seed)
        halton_full_np = make_sequence("halton", SEQ_TOTAL_LENGTH, DIM, burn=BURN, scramble=False, seed=seed)
        sobol_full_t   = torch.tensor(sobol_full_np,  dtype=torch.float32, device=DEVICE)
        halton_full_t  = torch.tensor(halton_full_np, dtype=torch.float32, device=DEVICE)
        D_sobol_seq    = SEQ_LOSS_FN(sobol_full_t)   # [N]
        D_halton_seq   = SEQ_LOSS_FN(halton_full_t)  # [N]
        sobol_curve    = D_sobol_seq[1:].detach().cpu().numpy().tolist()
        halton_curve   = D_halton_seq[1:].detach().cpu().numpy().tolist()

        # Scrambled Sobol: average of B curves (each computed in one pass)
        scrambled_sum = torch.zeros(SEQ_TOTAL_LENGTH - 1, dtype=torch.float32, device="cpu")
        for b in range(SCRAMBLED_B):
            sobol_scr_full_np = make_sequence("sobol", SEQ_TOTAL_LENGTH, DIM, burn=BURN, scramble=True, seed=b)
            sobol_scr_full_t  = torch.tensor(sobol_scr_full_np, dtype=torch.float32, device=DEVICE)
            D_scr_seq         = SEQ_LOSS_FN(sobol_scr_full_t)  # [N]
            scrambled_sum += D_scr_seq[1:].detach().cpu()
        scrambled_mean_curve = (scrambled_sum / float(SCRAMBLED_B)).numpy().tolist()

    fig, ax = plt.subplots(figsize=(7,4))
    sob = np.clip(np.asarray(sobol_curve, dtype=np.float32), 1e-12, None)
    hal = np.clip(np.asarray(halton_curve, dtype=np.float32), 1e-12, None)
    mod = np.clip(np.asarray(model_curve, dtype=np.float32), 1e-12, None)
    ax.semilogy(lengths, sob, label='Sobol discrepancy')
    ax.semilogy(lengths, hal, label='Halton discrepancy')
    ax.semilogy(lengths, mod, label=f'Model discrepancy (NeuroLDS, {LOSS_LABEL})')
    scr = np.clip(np.asarray(scrambled_mean_curve, dtype=np.float32), 1e-12, None)
    ax.semilogy(lengths, scr, label=f'Scrambled Sobol (mean of B=32)')
    ax.set_xlabel('Sequence length P')
    ax.set_ylabel(f'{LOSS_LABEL} (log scale)')
    ax.set_title('Discrepancy vs Prefix Length (lower is better)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    fig_path = os.path.join(OUT_DIR, f"{base_name}.png")
    txt_path = os.path.join(OUT_DIR, f"{base_name}.txt")
    fig.savefig(fig_path, bbox_inches='tight', dpi=200)
    # W&B: log final discrepancy plot
    wandb.log({"final/plot": wandb.Image(fig)})
    wandb.log({
        "final/last_discrepancy/sobol": float(sob[-1]),
        "final/last_discrepancy/halton": float(hal[-1]),
        "final/last_discrepancy/model": float(mod[-1]),
        "final/last_discrepancy/scrambled_sobol_mean": float(scr[-1]),
    })

    # Save hyperparameters (complete)
    with open(txt_path, 'w') as f:
        f.write("Hyperparameters\n")
        f.write(f"name: {name}\n")
        f.write(f"dim: {DIM}\n")
        f.write(f"hidden: {HIDDEN}\n")
        f.write(f"layers: {LAYERS}\n")
        f.write(f"K: {K_POSENC}\n")
        f.write(f"device: {DEVICE_STR}\n")
        f.write(f"seq_total_length: {SEQ_TOTAL_LENGTH}\n")
        f.write(f"epochs: {EPOCHS}\n")
        f.write(f"lr: {LR}\n")
        f.write(f"pretrain_epochs: {PRETRAIN_EPOCHS}\n")
        f.write(f"fine_tune_lr: {FINE_TUNE_LR}\n")
        f.write(f"warmup_epochs: {WARMUP_EPOCHS}\n")
        f.write(f"min_finetune_lr: {MIN_FINETUNE_LR}\n")
        f.write(f"final_lr_ratio: {FINAL_LR_RATIO}\n")
        f.write(f"train_seq: {TRAIN_SEQ}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"disc_loss_choice: {disc_loss}\n")
        f.write(f"disc_loss_label: {LOSS_LABEL}\n")
        f.write(f"pre_eps: {pre_eps}\n")
        f.write(f"ft_eps: {ft_eps}\n")
        f.write(f"burn_in: {BURN}\n")
        f.write(f"scrambled_B: {SCRAMBLED_B}\n")
        f.write(f"verbose: {VERBOSE}\n")
        f.write("gamma: ")
        if gamma is None:
            f.write("None\n")
        else:
            f.write("[" + ", ".join(f"{v:.6g}" for v in gamma) + "]\n")

    with open(txt_path, 'a') as f:
        f.write("\nPredicted points (pred_all) for indices 0..{}\n".format(SEQ_TOTAL_LENGTH - 1))
        f.write("Index  " + "  ".join([f"x{j}" for j in range(DIM)]) + "\n")
        for i in range(SEQ_TOTAL_LENGTH):
            coords = "  ".join([f"{val:.8f}" for val in pred_all_np[i]])
            f.write(f"{i}  {coords}\n")
        f.write("\nDiscrepancy values per length (model vs baselines).\n")
        f.write("Length  Sobol  Halton  Model  ScrambledSobolMean\n")
        for i in range(len(lengths)):
            f.write(f"{int(lengths[i])}  {sobol_curve[i]:.8f}  {halton_curve[i]:.8f}  {model_curve[i]:.8f}  {scrambled_mean_curve[i]:.8f}\n")

    if VERBOSE:
        print(f"Saved plot to {fig_path} and hyperparameters to {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--K', type=int, default=8, help='Number of sinusoidal frequency bands for index encoding')
    parser.add_argument('--seq_total_length', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_epochs', type=int, default=30000)
    parser.add_argument('--fine_tune_lr', type=float, default=3e-4)
    parser.add_argument('--train_seq', choices=['sobol', 'halton'], default='sobol', help='Which low-discrepancy sequence to use for supervised pretraining.')
    parser.add_argument('--name', default ="test", help='Base filename for outputs saved to results/')
    parser.add_argument('--verb', type=bool, default=False)
    parser.add_argument('--warmup_epochs', type=int, default=25)
    parser.add_argument('--min_finetune_lr', type=float, default=5e-6)
    parser.add_argument('--final_lr_ratio', type=float, default=0.1)
    parser.add_argument('--disc_loss', choices=['star','sym','per','ext','asd','ctr',
                                            'star_w','sym_w','per_w','ext_w','ctr_w'],
                        default='star',
                        help='Which L2 discrepancy family to optimize (use *_w for weighted).')
    parser.add_argument('--pre_eps', type=float, default=0.0,
                        help='Early stop pretraining if MSE < pre_eps (0 disables).')
    parser.add_argument('--ft_eps', type=float, default=0.0,
                        help='Early stop fine-tuning if final discrepancy < ft_eps (0 disables).')
    parser.add_argument('--gamma', type=float, nargs='+', default=None,
                    help='Per-dimension gamma weights for weighted losses (length must equal dim).')
    args = parser.parse_args()

    main(
        dim=args.dim,
        hidden=args.hidden,
        layers=args.layers,
        K=args.K,
        seq_total_length=args.seq_total_length,
        epochs=args.epochs,
        lr=args.lr,
        pretrain_epochs=args.pretrain_epochs,
        fine_tune_lr=args.fine_tune_lr,
        train_seq=args.train_seq,
        name=args.name,
        warmup_epochs=args.warmup_epochs,
        min_finetune_lr=args.min_finetune_lr,
        final_lr_ratio=args.final_lr_ratio,
        verbose = args.verb,
        disc_loss=args.disc_loss, 
        pre_eps=args.pre_eps, 
        ft_eps=args.ft_eps,
        gamma=args.gamma, 
    )
    wandb.finish()
