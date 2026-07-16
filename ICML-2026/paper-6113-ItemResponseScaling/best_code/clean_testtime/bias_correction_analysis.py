"""
Compute the plug-in bias correction for the IRT-based pass@k estimator.
Uses the second-order Taylor expansion from Appendix (bias_correction).

Usage:  python bias_correction_analysis.py
Requires: irsl_testtime_resmat2_withz.pt
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm
from torch.distributions import Bernoulli

torch.manual_seed(0)

PLOT_DIR = Path(__file__).resolve().parent / "results" / "bias_correction"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── data ───────────────────────────────────────────────────────────────────
DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "deval" / "monkey" / "monkey_analysis" / "irsl_testtime_resmat2_withz.pt"
)
SAMPLE_BUDGET = 50
ITEM_BUDGET = 30
DEVICE = "cpu"

# ── inlined IRT utilities (from utils.py, avoiding merge-conflict file) ───

def beta_nll(y, mu, phi):
    a = mu * phi
    b = (1.0 - mu) * phi
    return -((a - 1) * torch.log(y) + (b - 1) * torch.log1p(-y)
             - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)))

def trainer(parameters, optim, closure, n_iter=100, eps=1e-6):
    for iteration in range(n_iter):
        if iteration > 0:
            prev_params = [p.clone() for p in parameters]
            prev_loss = loss.clone()
        loss = optim.step(closure)
        if iteration > 0:
            d_loss = (prev_loss - loss).item()
            d_params = sum(torch.norm(prev - curr, p=2).item()
                          for prev, curr in zip(prev_params, parameters))
            if abs(d_loss) < eps and d_params < eps:
                break
    return parameters

def _estimate_theta_generic(theta, asked_ys, device, logits_fn, loss_kind, phi=10.0, eps=1e-6, lr=0.1):
    asked_ys = torch.as_tensor(asked_ys, device=device, dtype=torch.float)
    theta = theta.clone().requires_grad_(True)
    optim = torch.optim.LBFGS([theta], lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    phi_t = torch.as_tensor(phi, device=device, dtype=torch.float)
    def closure():
        optim.zero_grad()
        logits = logits_fn(theta)
        probs = torch.sigmoid(logits)
        if loss_kind == "beta":
            mu = probs.clamp(min=eps, max=1.0 - eps)
            loss = beta_nll(asked_ys, mu, phi_t).mean()
        else:
            loss = -Bernoulli(probs=probs).log_prob(asked_ys).mean()
        loss.backward()
        return loss
    theta = trainer([theta], optim, closure)[0]
    return theta.detach()

def _estimate_theta_beta_1pl(theta, asked_ys, asked_zs, device, eps=1e-6):
    asked_ys = asked_ys.clamp(min=eps, max=1.0 - eps)
    asked_zs = torch.as_tensor(asked_zs, device=device, dtype=torch.float)
    return _estimate_theta_generic(theta, asked_ys, device, logits_fn=lambda th: th + asked_zs, loss_kind="beta")

def _select_next_1pl(theta, rem_discri, rem_z):
    return torch.argmin(torch.abs(theta + rem_z)).item()

def _span_idxs(zs, ys, k, trim=0.10):
    lo, hi = torch.quantile(zs, torch.tensor([trim, 1 - trim], device=zs.device))
    targets = torch.linspace(lo, hi, steps=k, device=zs.device)
    idxs, used = [], torch.zeros_like(zs, dtype=torch.bool)
    for t in targets:
        d = torch.abs(zs - t)
        d[used | torch.isnan(ys)] = float("inf")
        i = torch.argmin(d).item()
        idxs.append(i); used[i] = True
    return idxs

def cat_beta_1pl(ys, zs, device, budget=50, init_frac=0.2):
    rem_y, rem_z = ys.clone(), zs.clone()
    init_idx = _span_idxs(rem_z, rem_y, int(init_frac * budget))
    asked_y, asked_z = rem_y[init_idx], rem_z[init_idx]
    mask = torch.ones(rem_y.shape[0], dtype=torch.bool, device=device)
    mask[torch.tensor(init_idx, device=device, dtype=torch.long)] = False
    rem_y, rem_z = rem_y[mask], rem_z[mask]
    theta = torch.zeros(1, device=device)
    theta = _estimate_theta_beta_1pl(theta, asked_y, asked_z, device)
    thetas = [theta.clone().item()]
    asked = asked_y.numel()
    while asked < budget and rem_y.numel() > 0:
        i = _select_next_1pl(theta, None, rem_z)
        y_i, z_i = rem_y[i], rem_z[i]
        rem_y = torch.cat([rem_y[:i], rem_y[i+1:]])
        rem_z = torch.cat([rem_z[:i], rem_z[i+1:]])
        if torch.isnan(y_i):
            continue
        asked_y = torch.cat([asked_y, y_i.view(1)])
        asked_z = torch.cat([asked_z, z_i.view(1)])
        theta = _estimate_theta_beta_1pl(theta, asked_y, asked_z, device)
        thetas.append(theta.clone().item())
        asked += 1
    return thetas

def compute_pass_iatk_gt(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_pass_datk_gts(data2d):
    n_items, n_samples = data2d.shape
    k_range = np.arange(1, n_samples + 1)
    per_item = []
    for i in range(n_items):
        arr = data2d[i]
        n = int((~np.isnan(arr)).sum()); c = int(np.nansum(arr))
        per_item.append([compute_pass_iatk_gt(n, c, k) for k in k_range])
    return np.nanmean(np.vstack(per_item), axis=0)

def compute_pass_datk_irt(irt_probs, n_samples):
    k_range = np.arange(1, n_samples + 1)
    per_item = []
    for p in irt_probs:
        per_item.append([1.0 - (1.0 - p)**k for k in k_range])
    return np.nanmean(np.vstack(per_item), axis=0)

# ── load data ──────────────────────────────────────────────────────────────
payload = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
data_tensor = np.array(payload["data_tensor"], dtype=np.float64)   # (12, 120, 2560)
model_names = list(payload["models"])
test_models = list(payload["test_models"])
datasets = list(payload["datasets"])
zs = np.array(payload["zs"], dtype=np.float64)                    # (120,)

test_model_indices = [i for i, m in enumerate(model_names) if m in set(test_models)]
n_models, n_items, n_samples = data_tensor.shape
unique_benchmarks = sorted(set(datasets))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ── main computation ───────────────────────────────────────────────────────
print("=" * 80)
print("Plug-in Bias Correction Analysis for IRT-based Pass@k (1PL)")
print("=" * 80)
results = []

for bench in unique_benchmarks:
    item_idxs = [j for j, d in enumerate(datasets) if d == bench]
    bench_zs_np = zs[item_idxs]
    bench_zs_t = torch.tensor(bench_zs_np, dtype=torch.float, device=DEVICE)
    n_bench_items = len(item_idxs)

    for test_mi, test_gi in enumerate(test_model_indices):
        model_name = model_names[test_gi]
        model_tensor = data_tensor[test_gi, item_idxs, :]  # (n_items, n_samples)

        # 1) Run CAT (same as pipeline)
        bench_probmat = torch.tensor(
            np.nanmean(model_tensor[:, :SAMPLE_BUDGET], axis=-1),
            dtype=torch.float, device=DEVICE
        )
        theta_traj = cat_beta_1pl(bench_probmat, bench_zs_t, DEVICE, budget=ITEM_BUDGET)
        theta_hat = float(theta_traj[-1])

        # 2) IRT predictions: sigmoid(theta + z) for 1PL
        p_hat = sigmoid(theta_hat + bench_zs_np)

        # 3) Ground truth pass@1
        p_true = np.nanmean(model_tensor, axis=-1)

        # 4) Fisher information: I_j(theta) = p_j(1-p_j) for 1PL
        #    Var(theta) ≈ 1 / sum(I_j) over queried items
        fisher_all = p_hat * (1 - p_hat)
        total_fisher = np.sum(fisher_all[:ITEM_BUDGET])
        var_theta = 1.0 / total_fisher if total_fisher > 1e-10 else np.inf

        # 5) sigma_j^2 = [p_hat_j * (1-p_hat_j)]^2 * Var(theta)  (d_j=1 for 1PL)
        sigma_j_sq = (p_hat * (1 - p_hat))**2 * var_theta

        # 6) pass@k curves
        gt_curve = compute_pass_datk_gts(model_tensor)
        plugin_curve = compute_pass_datk_irt(p_hat, n_samples)

        # 7) Bias correction: k(k-1)/2 * (1-p_hat)^{k-2} * sigma_j^2
        k_range = np.arange(1, n_samples + 1, dtype=np.float64)
        corr_per_item = np.zeros((n_bench_items, n_samples))
        for j in range(n_bench_items):
            pj, sj2 = p_hat[j], sigma_j_sq[j]
            q = 1 - pj
            for ki in range(1, n_samples):  # k >= 2
                k = k_range[ki]
                corr_per_item[j, ki] = 0.5 * k * (k - 1) * q**(k - 2) * sj2
        correction_curve = np.mean(corr_per_item, axis=0)
        corrected_curve = np.clip(plugin_curve + correction_curve, 1e-15, 1.0)

        # 8) MAE (raw pass@k space)
        mae_plugin = np.mean(np.abs(gt_curve - plugin_curve))
        mae_corrected = np.mean(np.abs(gt_curve - corrected_curve))

        # 9) Relative correction magnitude
        mean_correction = np.mean(np.abs(correction_curve))
        mean_plugin_val = np.mean(plugin_curve)
        relative_corr = mean_correction / mean_plugin_val if mean_plugin_val > 1e-15 else 0.0
        pass1_mae = np.mean(np.abs(p_hat - p_true))

        results.append(dict(
            benchmark=bench, model=model_name, theta_hat=theta_hat,
            var_theta=var_theta, pass1_mae=pass1_mae,
            mae_plugin=mae_plugin, mae_corrected=mae_corrected,
            mean_correction=mean_correction, relative_correction=relative_corr,
            gt_curve=gt_curve, plugin_curve=plugin_curve,
            corrected_curve=corrected_curve, correction_curve=correction_curve,
        ))

        print(f"\n{bench} | {model_name}")
        print(f"  theta={theta_hat:.3f}, Var(theta)={var_theta:.4e}, Pass@1 MAE={pass1_mae:.4e}")
        print(f"  Plugin MAE={mae_plugin:.4e}, Corrected MAE={mae_corrected:.4e}")
        print(f"  |Correction|={mean_correction:.4e}, Relative={relative_corr:.4e}")

# ── summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
corrs = [r["relative_correction"] for r in results]
print(f"  Mean relative correction: {np.mean(corrs):.4e}")
print(f"  Max  relative correction: {np.max(corrs):.4e}")
print(f"  Mean Plugin MAE:          {np.mean([r['mae_plugin'] for r in results]):.4e}")
print(f"  Mean Corrected MAE:       {np.mean([r['mae_corrected'] for r in results]):.4e}")

# ── LaTeX ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("LATEX TABLE")
print("=" * 80)
print(r"\begin{table}[htb!]")
print(r"\centering\small")
print(r"\begin{tabular}{llccc}")
print(r"\toprule")
print(r"Benchmark & LM & Plugin MAE & Corrected MAE & Rel.\ correction \\")
print(r"\midrule")
for r in results:
    b = r["benchmark"].replace("_", r"\_")
    m = r["model"].replace("_", r"\_")
    print(f"{b} & {m} & {r['mae_plugin']:.2e} & {r['mae_corrected']:.2e} & {r['relative_correction']:.2e} \\\\")
print(r"\midrule")
print(f"\\textbf{{Mean}} & & {np.mean([r['mae_plugin'] for r in results]):.2e} & "
      f"{np.mean([r['mae_corrected'] for r in results]):.2e} & {np.mean(corrs):.2e} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{Second-order bias correction magnitude across all test (LM, benchmark) pairs.}")
print(r"\label{tab:bias_correction}")
print(r"\end{table}")

# ── PLOTS ──────────────────────────────────────────────────────────────────
k_range = np.arange(1, n_samples + 1)

# --- Figure 1: pass@k curves for all 16 pairs (4x4 grid) ---
fig, axes = plt.subplots(len(unique_benchmarks), len(test_model_indices),
                         figsize=(5 * len(test_model_indices), 4 * len(unique_benchmarks)),
                         squeeze=False)
for idx, r in enumerate(results):
    row = idx // len(test_model_indices)
    col = idx % len(test_model_indices)
    ax = axes[row, col]

    eps = 1e-10
    ax.loglog(k_range, -np.log(np.clip(r["gt_curve"], eps, None)),
              label="Ground Truth", color="black", linewidth=2)
    ax.loglog(k_range, -np.log(np.clip(r["plugin_curve"], eps, None)),
              label="IRSL (plug-in)", color="red", linewidth=1.5, linestyle="--")
    ax.loglog(k_range, -np.log(np.clip(r["corrected_curve"], eps, None)),
              label="IRSL (corrected)", color="blue", linewidth=1.5, linestyle=":")

    ax.set_title(f"{r['benchmark']}\n{r['model']}", fontsize=10)
    ax.set_xlabel("k", fontsize=9)
    ax.set_ylabel(r"$-\log(\mathrm{pass@k})$", fontsize=9)
    if row == 0 and col == 0:
        ax.legend(fontsize=8)

fig.suptitle(r"$-\log(\mathrm{pass@k})$: Ground Truth vs Plug-in vs Corrected", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "passk_curves_all.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {PLOT_DIR / 'passk_curves_all.png'}")

# --- Figure 2: Relative correction magnitude bar chart ---
fig, ax = plt.subplots(figsize=(12, 5))
labels = [f"{r['benchmark']}\n{r['model'].split('-')[0]}" for r in results]
rel_corrs = [r["relative_correction"] * 100 for r in results]  # in percent
bars = ax.bar(range(len(results)), rel_corrs, color="steelblue", edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(results)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Relative Correction (%)", fontsize=12)
ax.set_title("Bias Correction as % of Plug-in Pass@k", fontsize=14)
ax.axhline(y=np.mean(rel_corrs), color="red", linestyle="--", linewidth=1.5,
           label=f"Mean = {np.mean(rel_corrs):.2f}%")
ax.legend(fontsize=11)
for bar, val in zip(bars, rel_corrs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig(PLOT_DIR / "relative_correction_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {PLOT_DIR / 'relative_correction_bar.png'}")

# --- Figure 3: Correction curve magnitude for two representative pairs ---
# Pick one AIME pair (hard) and one MMLU pair (easy)
aime_r = [r for r in results if r["benchmark"] == "aime2024"][0]
mmlu_r = [r for r in results if r["benchmark"] == "mmlu_pro"][0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, r, title in zip(axes, [aime_r, mmlu_r],
                         ["AIME 2024 (hard)", "MMLU Pro (easy)"]):
    ax.plot(k_range, r["correction_curve"], color="darkred", linewidth=1.5,
            label="Correction term")
    ax.set_xlabel("k (number of samples)", fontsize=11)
    ax.set_ylabel("Correction magnitude", fontsize=11)
    ax.set_title(f"{title}: {r['model']}\nRel. correction = {r['relative_correction']*100:.2f}%",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(1, 200)

fig.suptitle("Bias Correction Term vs k", fontsize=14)
fig.tight_layout()
fig.savefig(PLOT_DIR / "correction_vs_k.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {PLOT_DIR / 'correction_vs_k.png'}")

# --- Figure 4: Heatmap of relative correction ---
bench_names = unique_benchmarks
model_short = [m.replace("gemma-3-27b-it", "Gemma-27B") for m in test_models]
vals = np.array([[r["relative_correction"] * 100 for r in results
                  if r["benchmark"] == b] for b in bench_names])

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(vals, aspect="auto", cmap="Blues", vmin=0)
ax.set_xticks(range(len(model_short)))
ax.set_xticklabels(model_short, rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(len(bench_names)))
ax.set_yticklabels(bench_names, fontsize=10)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        ax.text(j, i, f"{vals[i,j]:.2f}%", ha="center", va="center", fontsize=9,
                color="white" if vals[i,j] > vals.max()*0.6 else "black")
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Relative Correction (%)", fontsize=10)
ax.set_title("Plug-in Bias Correction (% of pass@k)", fontsize=13)
ax.set_xlabel("Test LM", fontsize=11)
ax.set_ylabel("Benchmark", fontsize=11)
fig.tight_layout()
fig.savefig(PLOT_DIR / "correction_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {PLOT_DIR / 'correction_heatmap.png'}")
