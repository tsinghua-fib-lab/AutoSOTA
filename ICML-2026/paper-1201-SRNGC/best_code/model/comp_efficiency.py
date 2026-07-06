############################################################
# Full ICML-style benchmarking with penalties
############################################################

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from penalty import Shapley_Penalty

warnings.filterwarnings("ignore")

############################################################
# Global config
############################################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

############################################################
# ICML plotting style
############################################################

def set_icml_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "figure.figsize": (6.4, 4.0),
        "figure.dpi": 120,
    })

############################################################
# Model
############################################################

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

############################################################
# Penalty configurations & plot styles
############################################################
PENALTIES = {
    "Jacob-L1":  dict(num_proj=-1, approx=False, individual_effect_only=True),
    "Jacob-F(1)":dict(num_proj=1,  approx=True,  individual_effect_only=True),
    "Jacob-F(3)":dict(num_proj=3,  approx=True,  individual_effect_only=True),
    "Jacob-F(5)":dict(num_proj=5,  approx=True,  individual_effect_only=True),
    "Shapley":   dict(num_proj=-1, approx=False, individual_effect_only=False),
    "F-Shap(1)": dict(num_proj=1,  approx=True,  individual_effect_only=False),
    "F-Shap(3)": dict(num_proj=3,  approx=True,  individual_effect_only=False),
    "F-Shap(5)": dict(num_proj=5,  approx=True,  individual_effect_only=False),
}


# =============================
# ICML-style plotting aesthetics
# =============================
METHOD_STYLE = {
    # Jacobian-based (blue)
    "Jacob-L1": {
        "color": "#1f77b4",   # muted blue
        "linestyle": "-",
        "marker": "o",
        "markersize": 3.6,
        "linewidth": 1.6,
    },
    "Jacob-F(1)": {
        "color": "#1f77b4",
        "linestyle": "--",
        "marker": "s",
        "markersize": 3.6,
        "linewidth": 1.6,
    },
    "Jacob-F(3)": {
        "color": "#1f77b4",
        "linestyle": ":",
        "marker": "^",
        "markersize": 3.6,
        "linewidth": 1.6,
    },
    "Jacob-F(5)": {
        "color": "#1f77b4",
        "linestyle": "-.",
        "marker": "v",
        "markersize": 3.6,
        "linewidth": 1.6,
    },

    # Shapley-based (orange)
    "Shapley": {
        "color": "#ff7f0e",   # muted orange
        "linestyle": "-",
        "marker": "o",
        "markersize": 3.6,
        "linewidth": 1.6,
    },
    "F-Shap(1)": {
        "color": "#ff7f0e",
        "linestyle": "--",
        "marker": "s",
        "markersize": 3.6,
        "linewidth": 1.6,
    },
    "F-Shap(3)": {
        "color": "#ff7f0e",
        "linestyle": ":",
        "marker": "^",
        "markersize": 3.6,
        "linewidth": 1.6,
    },
    "F-Shap(5)": {
        "color": "#ff7f0e",
        "linestyle": "-.",
        "marker": "v",
        "markersize": 3.6,
        "linewidth": 1.6,
    },
}


############################################################
# Timing utility with penalty
############################################################

class EpochTimer:
    def __init__(self, device=DEVICE, seed=0):
        self.device = device
        self.seed = seed

    def _single_run(
        self,
        model,
        penalty,
        x,
        y,
        epochs=50,
        warmup=10,
        lr=1e-4,
    ):
        torch.manual_seed(self.seed)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # warmup
        for _ in range(warmup):
            opt.zero_grad()
            pred = model(x)
            ind, inter = penalty(model, x)
            loss = loss_fn(pred, y) + ind + inter
            loss.backward()
            opt.step()

        times = []
        for _ in range(epochs):
            t0 = time.time()
            opt.zero_grad()
            pred = model(x)
            ind, inter = penalty(model, x)
            loss = loss_fn(pred, y) + ind + inter
            loss.backward()
            opt.step()
            times.append(time.time() - t0)

        return np.mean(times)

    def repeated_runs(
        self,
        model_fn,
        penalty_cfg,
        x_fn,
        y_fn,
        n_runs=5,
    ):
        times = []
        for _ in range(n_runs):
            model = model_fn()
            penalty = Shapley_Penalty(**penalty_cfg, device=self.device)
            x, y = x_fn(), y_fn()
            t = self._single_run(model, penalty, x, y)
            times.append(t)

        times = np.asarray(times)
        mean = times.mean()
        ci95 = 1.96 * times.std(ddof=1) / np.sqrt(n_runs)
        return mean, ci95

############################################################
# Unified plotting function (multiple methods)
############################################################

def plot_scaling_multi(
    x_vals,
    results,
    xlabel,
    ylabel,
    title,
    xscale="log",
    yscale="log",
    filename=None,
    fontsize=12,  # <--- 1. Add this argument (14 or 16 is good for ICML)
):
    plt.figure()
    
    for name, (means, cis) in results.items():
        style = METHOD_STYLE[name]
        if name == 'Jacob-L1':
            name = 'Jacob-F'
        if name == 'Shapley':
            name = 'F-Shap'
        plt.plot(x_vals, means, label=name, **style)
        # plt.fill_between(...) code...

    # 2. Set font size for Axis Labels (The text)
    plt.xlabel(xlabel, fontsize=fontsize, labelpad=12)
    plt.ylabel(ylabel, fontsize=fontsize, labelpad=12)
    
    # 3. Set font size for Title (Optional, usually same or slightly larger)
    plt.title(title, fontsize=fontsize)

    # 4. Set font size for Ticks (The numbers on the axis)
    # 'labelsize' controls the font size of the numbers
    plt.tick_params(axis='both', which='major', labelsize=fontsize) 

    plt.xscale(xscale)
    plt.yscale(yscale)
    
    plt.legend(
        fontsize=10, # You can also match the legend font size here
        ncol=2,
        handlelength=1.8,
        handletextpad=0.6,
        labelspacing=0.6,
        columnspacing=0.6,
        markerscale=1,
        frameon=False,
    )

    plt.tight_layout()

    if filename is not None:
        plt.savefig(f"./assets/{filename}", format="pdf", bbox_inches="tight", dpi=300)

############################################################
# Experiment 1: hidden dimension scaling
############################################################

def experiment_hidden_dim(timer, n_runs=5):
    input_dim = 64
    output_dim = 64
    batch_size = 256
    num_layers = 2

    hidden_dims = np.unique(
        np.round(np.logspace(np.log10(16), np.log10(8192), 16)).astype(int)
    )

    results = {k: ([], []) for k in PENALTIES}

    for h in hidden_dims:
        for name, cfg in PENALTIES.items():
            mean, ci = timer.repeated_runs(
                model_fn=lambda h=h: MLP(input_dim, h, output_dim, num_layers).to(DEVICE),
                penalty_cfg=cfg,
                x_fn=lambda: torch.randn(batch_size, input_dim, dtype=DTYPE, device=DEVICE, requires_grad=True),
                y_fn=lambda: torch.randn(batch_size, output_dim, dtype=DTYPE, device=DEVICE),
                n_runs=n_runs,
            )
            results[name][0].append(mean)
            results[name][1].append(ci)

    return hidden_dims, results

############################################################
# Experiment 2: depth / parameter scaling
############################################################

def experiment_layers(timer, n_runs=5):
    input_dim = 64
    output_dim = 64
    hidden_dim = 128
    batch_size = 256

    layers_list = np.arange(1, 20, 1)
    params = []
    results = {k: ([], []) for k in PENALTIES}

    for L in layers_list:
        params.append(count_parameters(MLP(input_dim, hidden_dim, output_dim, L)))
        for name, cfg in PENALTIES.items():
            mean, ci = timer.repeated_runs(
                model_fn=lambda L=L: MLP(input_dim, hidden_dim, output_dim, L).to(DEVICE),
                penalty_cfg=cfg,
                x_fn=lambda: torch.randn(batch_size, input_dim, dtype=DTYPE, device=DEVICE, requires_grad=True),
                y_fn=lambda: torch.randn(batch_size, output_dim, dtype=DTYPE, device=DEVICE),
                n_runs=n_runs,
            )
            results[name][0].append(mean)
            results[name][1].append(ci)

    return params, results

############################################################
# Experiment 3: input/output dimension scaling
############################################################

def experiment_input_dim(timer, n_runs=5):
    hidden_dim = 128
    num_layers = 2
    batch_size = 256

    dims = np.unique(
        np.round(np.logspace(np.log10(10), np.log10(1000), 16)).astype(int)
    )
    results = {k: ([], []) for k in PENALTIES}

    for d in dims:
        for name, cfg in PENALTIES.items():
            mean, ci = timer.repeated_runs(
                model_fn=lambda d=d: MLP(d, hidden_dim, d, num_layers).to(DEVICE),
                penalty_cfg=cfg,
                x_fn=lambda d=d: torch.randn(batch_size, d, dtype=DTYPE, device=DEVICE, requires_grad=True),
                y_fn=lambda d=d: torch.randn(batch_size, d, dtype=DTYPE, device=DEVICE),
                n_runs=n_runs,
            )
            results[name][0].append(mean)
            results[name][1].append(ci)

    return dims, results

import torch
import os
from datetime import datetime

def save_experiment_outputs(data, out_dir="results", filename=None):
    """
    data: dict
        key -> dict with fields:
            - x
            - res
    """
    os.makedirs(out_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_experiments_{timestamp}.pt"

    path = os.path.join(out_dir, filename)
    torch.save(data, path)

    print(f"[Saved all experiment outputs] {path}")

############################################################
# Main
############################################################

def exe():
    set_icml_style()
    timer = EpochTimer(seed=SEED)

    all_results = {}

    # ---- Hidden dim ----
    x, res = experiment_hidden_dim(timer, n_runs=10)
    all_results["hidden_dim"] = {
        "x": x,
        "res": res,
    }

    plot_scaling_multi(
        x, res,
        xlabel="Hidden dimension",
        ylabel="Average epoch time (seconds)",
        title="",
        filename="hidden_dim_penalty",
    )

    # ---- Depth / parameters ----
    x, res = experiment_layers(timer, n_runs=10)
    all_results["depth"] = {
        "x": x,
        "res": res,
    }

    plot_scaling_multi(
        x, res,
        xlabel="Number of trainable parameters",
        ylabel="Average epoch time (seconds)",
        title="",
        filename="depth_penalty",
    )

    # ---- Input / output dimension ----
    x, res = experiment_input_dim(timer, n_runs=10)
    all_results["dimension"] = {
        "x": x,
        "res": res,
    }

    plot_scaling_multi(
        x, res,
        xlabel="Input / output dimension",
        ylabel="Average epoch time (seconds)",
        title="",
        filename="dimension_penalty",
    )

    # ---- Save everything ----
    save_experiment_outputs(
        all_results,
        filename="penalty_scaling_all.pt",
    )
    
if __name__ == "__main__":
    # exe()
    all_results = torch.load('./assets/penalty_scaling_all.pt')
    # hidden_dim
    x, res = all_results['hidden_dim']['x'], all_results['hidden_dim']['res']
    plot_scaling_multi(
        x, res, 'Hidden dimension', 'Average epoch time (seconds)', '', 'log', 'log', 'hidden_dim_penalty.pdf'
    )
    # depth
    x, res = all_results['depth']['x'], all_results['depth']['res']
    plot_scaling_multi(
        x, res, 'Total Parameters', 'Runtime (s)', '', 'linear', 'linear', 'depth_penalty.pdf'
    )
    # dimension
    x, res = all_results['dimension']['x'], all_results['dimension']['res']
    print(x)
    print(res)
    plot_scaling_multi(
        x, res, 'Data Dimension', 'Runtime (s)', '', 'log', 'log', 'dimension_penalty.pdf'
    )