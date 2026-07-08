#!/usr/bin/env python3
"""Sweep Welch window parameters - efficient single-pass implementation."""
import sys, os, random, numpy as np, pandas as pd, torch, time, argparse
from pathlib import Path

sys.path.insert(0, "/repo/DLinear")
from exp.exp_main import Exp_Main, coherence_nmse_lb_gpu_batched

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Create args matching run_predictability_test.py defaults + all required fields
class Args:
    def __init__(self):
        self.is_training = 0
        self.train_only = False
        self.model_id = "Electricity_96_96"
        self.model = "DLinear"
        self.data = "custom"
        self.root_path = "../datasets/"
        self.data_path = "electricity.csv"
        self.features = "M"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.individual = False
        self.ps_lambda = 3.0
        self.use_ps_loss = 0
        self.patch_len_threshold = 24
        self.enc_in = 321
        self.dec_in = 321
        self.c_out = 321
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.05
        self.embed = "timeF"
        self.activation = "gelu"
        self.output_attention = False
        self.do_predict = False
        self.num_workers = 4
        self.itr = 1
        self.train_epochs = 10
        self.batch_size = 16
        self.patience = 3
        self.learning_rate = 1e-4
        self.des = "Exp"
        self.loss = "mse"
        self.lradj = "type1"
        self.use_amp = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = "0"
        self.embed_type = 0
        self.test_flop = False
        self.outdir = "./predictability_results"
        self.alpha_boundary = 1.0
        self.welch_win_frac = 0.25
        self.welch_overlap = 0.5
        self.workers = 8
        self.limit_batches = None

args = Args()
args.use_gpu = True if torch.cuda.is_available() else False

# Build setting string
fmt_str = "{0}_{1}_{2}_ft{3}_sl{4}_ll{5}_pl{6}_dm{7}_nh{8}_el{9}_dl{10}_df{11}_fc{12}_eb{13}_dt{14}_psloss{15}_{16}_{17}"
setting = fmt_str.format(
    args.model_id, args.model, args.data, args.features,
    args.seq_len, args.label_len, args.pred_len,
    args.d_model, args.n_heads, args.e_layers, args.d_layers,
    args.d_ff, args.factor, args.embed, args.distil,
    0, args.des, 0)
print(f"Setting: {setting}")

exp = Exp_Main(args)
ckpt_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
print(f"Loading checkpoint: {ckpt_path} (exists: {os.path.exists(ckpt_path)})")
exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))
exp.model.eval()

test_data, test_loader = exp._get_data(flag="test")

# Sweep grid - test key combinations
welch_fracs = [0.125, 0.25, 0.375, 0.5]
welch_overlaps = [0.25, 0.5, 0.75]

results = []
sample_id_global = 0
alpha_boundary = args.alpha_boundary

# First pass: collect all predictions and data in numpy arrays
print("\n[PHASE 1] Running model inference once...")
all_batches = []

with torch.no_grad():
    for bi, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)

        outputs = exp.model(batch_x)

        f_dim = -1 if args.features == "MS" else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y_eval = batch_y[:, -args.pred_len:, f_dim:]
        batch_x_eval = batch_x[:, :, f_dim:]

        B, T_pred, C = outputs.shape
        Np = max(8, int(alpha_boundary * min(batch_x_eval.shape[1], T_pred)))

        y_true_front = batch_y_eval[:, :Np, :]
        y_pred_front = outputs[:, :Np, :]
        x_tail = batch_x_eval[:, -Np:, :]

        all_batches.append({
            "x_tail": x_tail.cpu().numpy().astype(np.float32),
            "y_true": y_true_front.cpu().numpy().astype(np.float32),
            "y_pred": y_pred_front.cpu().numpy().astype(np.float32),
            "B": B, "C": C, "Np": Np,
        })
        sample_id_global += B
        if (bi + 1) % 20 == 0:
            print(f"  Batch {bi+1}/{len(test_loader)}")

total = sum(b["B"] * b["C"] for b in all_batches)
print(f"Collected {len(all_batches)} batches, {total} sample-channel pairs")

# Second pass: sweep Welch parameters
print("\n[PHASE 2] Sweeping Welch parameters...")
for wf in welch_fracs:
    for wo in welch_overlaps:
        t0 = time.time()
        mse_lb_list = []
        mse_model_list = []
        mae_model_list = []

        for d in all_batches:
            B, C, Np = d["B"], d["C"], d["Np"]
            win_len = max(32, int(wf * Np))

            x_tail_t = torch.from_numpy(d["x_tail"]).to(exp.device)
            y_true_t = torch.from_numpy(d["y_true"]).to(exp.device)
            y_pred_t = torch.from_numpy(d["y_pred"]).to(exp.device)

            # Coherence computation
            x_bcT = x_tail_t.permute(0, 2, 1).contiguous()
            y_bcT = y_true_t.permute(0, 2, 1).contiguous()

            nmse_lb_bc, P_lin_bc = coherence_nmse_lb_gpu_batched(
                x_bcT, y_bcT, win_len=win_len, overlap=wo, eps=1e-8, chunk_channels=None
            )

            var_y_bc = y_true_t.var(dim=1, unbiased=False)
            mu_y_bc = y_true_t.mean(dim=1)
            mu_x_bc = x_tail_t.mean(dim=1)

            mse_lb = (nmse_lb_bc * var_y_bc + (mu_y_bc - mu_x_bc) ** 2).cpu().numpy().flatten()
            mse_model = ((y_true_t - y_pred_t) ** 2).mean(dim=1).cpu().numpy().flatten()
            mae_model = (y_true_t - y_pred_t).abs().mean(dim=1).cpu().numpy().flatten()

            mse_lb_list.append(mse_lb)
            mse_model_list.append(mse_model)
            mae_model_list.append(mae_model)

        x_all = np.concatenate(mse_lb_list)
        y_all = np.concatenate(mse_model_list)
        mae_all = np.concatenate(mae_model_list)

        m = np.isfinite(x_all) & np.isfinite(y_all)
        R = float(np.corrcoef(x_all[m], y_all[m])[0, 1]) if m.sum() >= 2 else np.nan
        mse_mean = float(np.nanmean(y_all))
        mae_mean = float(np.nanmean(mae_all))

        elapsed = time.time() - t0
        results.append({
            "welch_win_frac": wf, "welch_overlap": wo,
            "R": R, "MSE_model": mse_mean, "MAE_model": mae_mean,
            "n_valid": int(m.sum()), "time_s": round(elapsed, 1),
        })
        print(f"  wf={wf:.3f} ov={wo:.2f} | R={R:.6f} MSE={mse_mean:.6f} MAE={mae_mean:.6f} n={m.sum()} ({elapsed:.1f}s)")

# Summary
print("\n" + "=" * 80)
print("SWEEP RESULTS (sorted by R)")
print("=" * 80)
results.sort(key=lambda r: r["R"], reverse=True)
for i, r in enumerate(results):
    print(f"  #{i+1}: wf={r[welch_win_frac]:.3f} ov={r[welch_overlap]:.2f} | "
          f"R={r[R]:.6f} MSE={r[MSE_model]:.6f} MAE={r[MAE_model]:.6f} n={r[n_valid]}")

best = results[0]
baseline_r = 0.880047
print(f"\nBaseline: R={baseline_r:.6f} (wf=0.25, ov=0.5)")
print(f"Best:     R={best[R]:.6f} (wf={best[welch_win_frac]}, ov={best[welch_overlap]})")
print(f"Delta:    {best[R] - baseline_r:+.6f}")

# Save
out_path = Path(args.outdir) / "welch_sweep_results.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")
