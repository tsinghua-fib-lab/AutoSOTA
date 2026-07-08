#!/usr/bin/env python3
"""Sweep alpha_boundary and combine with best Welch overlap."""
import sys, os, random, numpy as np, pandas as pd, torch, time, argparse
from pathlib import Path

sys.path.insert(0, "/repo/DLinear")
from exp.exp_main import Exp_Main, coherence_nmse_lb_gpu_batched

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

class Args:
    def __init__(self):
        self.is_training = 0; self.train_only = False
        self.model_id = "Electricity_96_96"; self.model = "DLinear"
        self.data = "custom"; self.root_path = "../datasets/"
        self.data_path = "electricity.csv"; self.features = "M"
        self.target = "OT"; self.freq = "h"
        self.checkpoints = "./checkpoints/"
        self.seq_len = 96; self.label_len = 48; self.pred_len = 96
        self.individual = False; self.ps_lambda = 3.0
        self.use_ps_loss = 0; self.patch_len_threshold = 24
        self.enc_in = 321; self.dec_in = 321; self.c_out = 321
        self.d_model = 512; self.n_heads = 8; self.e_layers = 2
        self.d_layers = 1; self.d_ff = 2048; self.moving_avg = 25
        self.factor = 1; self.distil = True; self.dropout = 0.05
        self.embed = "timeF"; self.activation = "gelu"
        self.output_attention = False; self.do_predict = False
        self.num_workers = 4; self.itr = 1; self.train_epochs = 10
        self.batch_size = 16; self.patience = 3
        self.learning_rate = 1e-4; self.des = "Exp"
        self.loss = "mse"; self.lradj = "type1"
        self.use_amp = False; self.use_gpu = True; self.gpu = 0
        self.use_multi_gpu = False; self.devices = "0"
        self.embed_type = 0; self.test_flop = False
        self.outdir = "./predictability_results"
        self.alpha_boundary = 1.0; self.welch_win_frac = 0.25
        self.welch_overlap = 0.5; self.workers = 8
        self.limit_batches = None

args = Args()
args.use_gpu = True if torch.cuda.is_available() else False

fmt_str = "{0}_{1}_{2}_ft{3}_sl{4}_ll{5}_pl{6}_dm{7}_nh{8}_el{9}_dl{10}_df{11}_fc{12}_eb{13}_dt{14}_psloss{15}_{16}_{17}"
setting = fmt_str.format(
    args.model_id, args.model, args.data, args.features,
    args.seq_len, args.label_len, args.pred_len,
    args.d_model, args.n_heads, args.e_layers, args.d_layers,
    args.d_ff, args.factor, args.embed, args.distil,
    0, args.des, 0)

exp = Exp_Main(args)
ckpt_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))
exp.model.eval()
test_data, test_loader = exp._get_data(flag="test")

# Sweep grid: alpha values x best welch configs
alphas = [0.5, 0.75, 1.0, 1.25, 1.5]
welch_configs = [
    (0.25, 0.5, "baseline"),
    (0.25, 0.75, "best_overlap"),
    (0.125, 0.75, "small_win_high_ov"),
]

results = []
print("\n[PHASE 1] Running model inference...")
all_batches = []
sample_id_global = 0

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

        all_batches.append({
            "x_tail_full": batch_x_eval.cpu().numpy().astype(np.float32),
            "y_true_full": batch_y_eval.cpu().numpy().astype(np.float32),
            "y_pred_full": outputs.cpu().numpy().astype(np.float32),
            "B": B, "C": C, "Tx": batch_x_eval.shape[1], "T_pred": T_pred,
        })
        if (bi + 1) % 40 == 0:
            print(f"  Batch {bi+1}/{len(test_loader)}")

print(f"Collected {len(all_batches)} batches")

print("\n[PHASE 2] Sweeping alpha x welch...")
for alpha in alphas:
    for wf, wo, label in welch_configs:
        t0 = time.time()
        mse_lb_list, mse_model_list, mae_model_list = [], [], []

        for d in all_batches:
            B, C, Tx, T_pred = d["B"], d["C"], d["Tx"], d["T_pred"]
            Np = max(8, int(alpha * min(Tx, T_pred)))

            x_tail = d["x_tail_full"][:, -Np:, :]
            y_true = d["y_true_full"][:, :Np, :]
            y_pred = d["y_pred_full"][:, :Np, :]

            win_len = max(32, int(wf * Np))

            x_t = torch.from_numpy(x_tail).to(exp.device)
            y_t = torch.from_numpy(y_true).to(exp.device)
            y_p = torch.from_numpy(y_pred).to(exp.device)

            x_bcT = x_t.permute(0, 2, 1).contiguous()
            y_bcT = y_t.permute(0, 2, 1).contiguous()

            nmse_lb_bc, P_lin_bc = coherence_nmse_lb_gpu_batched(
                x_bcT, y_bcT, win_len=win_len, overlap=wo, eps=1e-8, chunk_channels=None
            )

            var_y = y_t.var(dim=1, unbiased=False)
            mse_lb = (nmse_lb_bc * var_y + (y_t.mean(dim=1) - x_t.mean(dim=1)) ** 2).cpu().numpy().flatten()
            mse_model = ((y_t - y_p) ** 2).mean(dim=1).cpu().numpy().flatten()
            mae_model = (y_t - y_p).abs().mean(dim=1).cpu().numpy().flatten()

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
            "alpha": alpha, "wf": wf, "wo": wo, "label": label,
            "R": R, "MSE_model": mse_mean, "MAE_model": mae_mean,
            "Np_approx": max(8, int(alpha * 96)), "n_valid": int(m.sum()),
        })
        print(f"  alpha={alpha:.2f} {label:20s} | R={R:.6f} MSE={mse_mean:.6f} MAE={mae_mean:.6f} Np~={max(8,int(alpha*96))}")

# Summary
print("\n" + "=" * 80)
print("ALPHA x WELCH SWEEP (sorted by R)")
print("=" * 80)
results.sort(key=lambda r: r["R"], reverse=True)
for i, r in enumerate(results):
    print(f"  #{i+1}: alpha={r[alpha]:.2f} {r[label]:20s} | R={r[R]:.6f} MSE={r[MSE_model]:.6f} Np≈{r[Np_approx]}")

best = results[0]
print(f"\nBest: alpha={best[alpha]}, {best[label]}, R={best[R]:.6f}")
print(f"Baseline: alpha=1.0, baseline, R=0.880047")

out_path = Path(args.outdir) / "alpha_welch_sweep.csv"
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"Saved to {out_path}")
