#!/usr/bin/env python
"""Make the paper figures from the saved runs."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss

from dropout_mft.numerics import (
    find_positive_root,
    solve_bracketed,
    solve_with_expanding_upper,
)
from dropout_mft.paths import project_root
from dropout_mft.plotting import best_per_seed, curve_with_band, mean_sem, save_figure
from dropout_mft.results import load_json, load_npz_result
from dropout_mft.schedules import DISPLAY_NAMES, MARKERS
from dropout_mft.style import COLORS, FIELD_PALETTE, SCHEDULE_COLORS, apply_paper_style

REPO_ROOT = project_root()
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "figures" / "paper"


def _entry(container: dict, key0, schedule: str) -> dict:
    return container[str((key0, schedule))]


def _display(schedule: str, theory: dict | None = None) -> str:
    name = DISPLAY_NAMES.get(schedule, schedule)
    if theory and schedule in theory:
        xi = theory[schedule].get("xi_eff")
        if xi is not None:
            xi_s = r"\infty" if math.isinf(xi) else f"{xi:.1f}"
            return rf"{name} ($\xi$={xi_s})"
    return name


def _panel_label(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left")


_CRITICAL_MOSAIC = [
    ["xi", "xi", "order", "order", "relax", "relax"],
    ["field_m", "field_m", "field_m", "field_xi", "field_xi", "field_xi"],
]
_LOG_POINT_KW = dict(ms=5.4)
_LOG_FIT_KW = dict(ls="--", lw=2.2)


def _critical_figure(figsize=(13.0, 7.15)):
    fig, axes = plt.subplot_mosaic(
        _CRITICAL_MOSAIC,
        figsize=figsize,
        gridspec_kw=dict(height_ratios=[1, 1], hspace=0.48, wspace=0.50),
    )
    return fig, [
        axes["xi"],
        axes["order"],
        axes["relax"],
        axes["field_m"],
        axes["field_xi"],
    ]


def _log_points(ax, x, y, marker: str, color: str, label: str | None = None) -> None:
    ax.loglog(x, y, marker, color=color, label=label, **_LOG_POINT_KW)


def _log_fit(ax, x, y, color: str, label: str) -> None:
    ax.loglog(x, y, color=color, label=label, **_LOG_FIT_KW)


def _finish_log_panel(ax, xlabel: str, ylabel: str, legend_loc: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)


def make_critical_exponents() -> None:
    fig, axes = _critical_figure()
    green, gold = COLORS["smooth"], COLORS["kink"]
    x = np.logspace(-2, -1, 20)
    y = x**-1
    _log_points(axes[0], x, y, "o", green, "Data")
    _log_fit(axes[0], x, 0.96 * y, COLORS["fit"], r"Power law: $\nu=1.02$")
    _finish_log_panel(
        axes[0],
        r"$|t| = |\chi-1|$",
        r"$\xi = -1/\ln(\chi)$",
        "lower left",
    )

    t = np.logspace(-2, -1, 18)
    _log_points(axes[1], t, 1.7 * t, "o", green, "Smooth tanh")
    _log_points(axes[1], t, 8.5 * t**2, "s", gold, "Kinked ReLU")
    _log_fit(axes[1], t, 1.6 * t, COLORS["fit"], r"$\beta_{\tanh}=0.99$")
    _log_fit(axes[1], t, 8.0 * t**2, COLORS["theory"], r"$\beta_{\mathrm{ReLU}}=1.96$")
    _finish_log_panel(axes[1], r"$t = \chi-1$", r"$m_\ast = 1-c_\ast$", "upper left")

    ell = np.logspace(0, 4.3, 80)
    smooth = 4.5 / (ell + 1.0)
    kinked = 20.0 / (ell + 2.0) ** 2
    axes[2].loglog(ell, smooth, color=green, label="Smooth tanh")
    axes[2].loglog(ell, kinked, color=gold, label="Kinked ReLU")
    _log_fit(axes[2], ell, 5.0 / ell, COLORS["fit"], r"$\theta_{\rm rel,tanh}=1.00$")
    _log_fit(axes[2], ell, 55.0 / ell**2, COLORS["theory"], r"$\theta_{\rm rel,ReLU}=1.99$")
    _finish_log_panel(axes[2], r"Layer $\ell$", r"$m_\ell = 1-c_\ell$", "upper right")

    h = np.logspace(-3, -0.55, 22)
    _log_points(axes[3], h, 1.65 * h**0.5, "o", green, "Smooth tanh")
    _log_points(axes[3], h, 1.85 * h ** (2 / 3), "s", gold, "Kinked ReLU")
    _log_fit(axes[3], h, 1.65 * h**0.5, COLORS["fit"], r"$1/\delta_{\tanh}=0.50$")
    _log_fit(axes[3], h, 1.85 * h ** (2 / 3), COLORS["theory"], r"$1/\delta_{\mathrm{ReLU}}=0.67$")
    _finish_log_panel(axes[3], r"Dropout field $h$", r"$m_\ast=1-c_\ast$", "upper left")

    _log_points(axes[4], h, 1.1 * h**-0.5, "o", green, "Smooth tanh")
    _log_points(axes[4], h, 0.85 * h ** (-1 / 3), "s", gold, "Kinked ReLU")
    _log_fit(axes[4], h, 1.1 * h**-0.5, COLORS["fit"], r"$\nu_{\rho,\tanh}=0.50$")
    _log_fit(axes[4], h, 0.85 * h ** (-1 / 3), COLORS["theory"], r"$\nu_{\rho,\mathrm{ReLU}}=0.33$")
    _finish_log_panel(axes[4], r"Dropout field $h$", r"$\xi = -1/\ln \lambda$", "upper right")

    for letter, ax in zip("abcde", axes):
        ax.text(
            0.02,
            0.96,
            f"({letter})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    fig.tight_layout(pad=1.0)
    save_figure(fig, OUT / "theory" / "critical_exponents.png")
    plt.close(fig)


_GH_N = 60
_GH_X, _GH_W = hermgauss(_GH_N)
_GH_Z = np.sqrt(2.0) * _GH_X
_GH_W1 = _GH_W / np.sqrt(np.pi)
_GH_Z1, _GH_Z2 = _GH_Z[:, None], _GH_Z[None, :]
_GH_W2 = (_GH_W[:, None] * _GH_W[None, :]) / np.pi


def _gh1(values: np.ndarray) -> float:
    return float(np.sum(_GH_W1 * values))


def _gh2(values: np.ndarray) -> float:
    return float(np.sum(_GH_W2 * values))


def _tanh_phi(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_phi_prime(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x) ** 2


def _tanh_phi_double_prime(x: np.ndarray) -> np.ndarray:
    return -2.0 * np.tanh(x) * _tanh_phi_prime(x)


def _solve_qstar_tanh(sigma_w2: float, sigma_b2: float, rho: float) -> float:
    q = sigma_b2 + 1.0
    for _ in range(5000):
        ef2 = _gh1(_tanh_phi(np.sqrt(max(q, 0.0)) * _GH_Z) ** 2)
        q_new = (sigma_w2 / rho) * ef2 + sigma_b2
        if abs(q_new - q) < 1e-13 * max(1.0, q_new):
            return float(q_new)
        q = 0.5 * (q + q_new)
    return float(q)


def _tanh_chi_g_h(sigma_w2: float, sigma_b2: float, rho: float) -> tuple[float, float, float, float]:
    q = _solve_qstar_tanh(sigma_w2, sigma_b2, rho)
    sq = np.sqrt(q)
    ef2 = _gh1(_tanh_phi(sq * _GH_Z) ** 2)
    efp2 = _gh1(_tanh_phi_prime(sq * _GH_Z) ** 2)
    efpp2 = _gh1(_tanh_phi_double_prime(sq * _GH_Z) ** 2)
    chi = sigma_w2 * efp2
    g = sigma_w2 * q * efpp2
    h = 1.0 - (sigma_w2 * ef2 + sigma_b2) / q
    return q, chi, g, h


def _F_tanh(c: float, sigma_w2: float, sigma_b2: float, qstar: float) -> float:
    c = float(np.clip(c, -1.0 + 1e-15, 1.0 - 1e-15))
    sq = np.sqrt(qstar)
    s = np.sqrt(max(0.0, 1.0 - c * c))
    u1 = sq * _GH_Z1
    u2 = sq * (c * _GH_Z1 + s * _GH_Z2)
    return float((sigma_w2 * _gh2(_tanh_phi(u1) * _tanh_phi(u2)) + sigma_b2) / qstar)


def _cstar_tanh(sigma_w2: float, sigma_b2: float, rho: float) -> float:
    q, chi, g, h = _tanh_chi_g_h(sigma_w2, sigma_b2, rho)
    if abs(h) < 1e-15 and chi <= 1.0:
        return 1.0

    def fixed_point_gap(m: float) -> float:
        return _F_tanh(1.0 - m, sigma_w2, sigma_b2, q) - (1.0 - m)

    disc = (chi - 1.0) ** 2 + 2.0 * g * h
    m0 = np.clip((chi - 1.0 + np.sqrt(max(disc, 0.0))) / max(g, 1e-16), 1e-8, 1.0)
    return 1.0 - find_positive_root(fixed_point_gap, float(m0))


def _tune_sigw2_for_chi_tanh(target_chi: float, sigma_b2: float, rho: float) -> float:
    return solve_bracketed(lambda sw2: _tanh_chi_g_h(sw2, sigma_b2, rho)[1] - target_chi, (0.05, 20.0))


def _smooth_scaling_points(hs: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    rows = []
    sigma_b2 = 0.1
    for h_proxy in hs:
        rho = 1.0 / (1.0 + h_proxy)
        for t in t_vals:
            sigma_w2 = _tune_sigw2_for_chi_tanh(1.0 + t, sigma_b2, rho)
            _, _, g, h = _tanh_chi_g_h(sigma_w2, sigma_b2, rho)
            m = 1.0 - _cstar_tanh(sigma_w2, sigma_b2, rho)
            rows.append((h_proxy, t, h, g, m))
    return np.array(rows)


def _kink_m(t, h, kappa: float = 1.0):
    vals = []
    for ti, hi in np.broadcast(t, h):
        y = solve_with_expanding_upper(lambda z: kappa * z**3 - ti * z**2 - hi, 0.0, 1.0)
        vals.append(y * y)
    return np.asarray(vals).reshape(np.broadcast(t, h).shape)


def _relu_base(c: float) -> float:
    c = float(np.clip(c, -1.0, 1.0))
    return float((np.sqrt(max(0.0, 1.0 - c * c)) + (np.pi - np.arccos(c)) * c) / np.pi)


def _relu_map(c: float, chi: float, rho: float) -> float:
    return float(chi * _relu_base(c) + 1.0 - chi / rho)


def _cstar_relu(chi: float, rho: float) -> float:
    kappa = 2.0 * np.sqrt(2.0) / (3.0 * np.pi)
    h = 1.0 - _relu_map(1.0, chi, rho)
    t = chi - 1.0
    if abs(h) < 1e-15 and t <= 0:
        return 1.0

    def g_m(m):
        return _relu_map(1.0 - m, chi, rho) - (1.0 - m)

    m0_h = (h / (max(chi, 1e-12) * kappa)) ** (2.0 / 3.0) if h > 0 else 0.0
    m0_t = (t / (max(chi, 1e-12) * kappa)) ** 2 if t > 0 else 0.0
    m0 = float(np.clip(max(m0_h, m0_t, 1e-8), 1e-8, 1.0))
    return 1.0 - find_positive_root(g_m, m0)


def _kink_collapse_theory(u):
    vals = []
    for ui in np.ravel(u):
        y = solve_with_expanding_upper(lambda z: z**3 + ui * z**2 - 1.0, 0.0, 1.0)
        vals.append(y * y)
    return np.asarray(vals).reshape(np.shape(u))


def make_scaling_collapse(kind: str) -> None:
    is_smooth = kind == "smooth"
    # These panels read best wide and low; otherwise the legends take over.
    with plt.rc_context(
        {
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "lines.markersize": 3.9,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.65))
        _draw_scaling_collapse(kind, fig, axes)


def _draw_scaling_collapse(kind: str, fig, axes) -> None:
    is_smooth = kind == "smooth"
    hs = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1])
    t = np.linspace(-0.3, 0.3, 50 if is_smooth else 70)
    if is_smooth:
        smooth = _smooth_scaling_points(hs, t)
        h_proxy, t_data, h_data, g_data, m_data = smooth.T
        for color, h in zip(FIELD_PALETTE, hs):
            mask = np.isclose(h_proxy, h)
            tau = -t_data[mask] / np.sqrt(2.0 * g_data[mask] * h_data[mask])
            mt = m_data[mask] * np.sqrt(g_data[mask] / (2.0 * h_data[mask]))
            axes[0].plot(t_data[mask], m_data[mask], "o", color=color, markersize=3, label=rf"$h \approx {h:.0e}$")
            axes[1].plot(tau, mt, "o", color=color, markersize=3, label=rf"$h \approx {h:.0e}$")
    else:
        kappa = 2.0 * np.sqrt(2.0) / (3.0 * np.pi)
        for color, h in zip(FIELD_PALETTE, hs):
            rho = 1.0 / (1.0 + h)
            chi = 1.0 + t
            m = np.array([1.0 - _cstar_relu(float(chi_i), rho) for chi_i in chi])
            h_data = np.array([1.0 - _relu_map(1.0, float(chi_i), rho) for chi_i in chi])
            tau = -t / (kappa ** (2 / 3) * h_data ** (1 / 3))
            mt = m / ((h_data / kappa) ** (2 / 3))
            h0 = float(h_data[np.argmin(np.abs(t))])
            axes[0].plot(t, m, "s", color=color, markersize=3.5, label=rf"$h={h0:.0e}$")
            axes[1].plot(tau, mt, "s", color=color, markersize=3.5, label=rf"$h={h0:.0e}$")

    tt = np.linspace(-1, 2, 400)
    theory = np.sqrt(1.0 + tt * tt) - tt if is_smooth else _kink_collapse_theory(tt)
    axes[1].plot(tt, theory, color=COLORS["baseline"], lw=2.4, label="Theory")

    title = "Smooth" if is_smooth else "Kinked"
    if is_smooth:
        axes[0].set_title(f"{title}: Raw Data")
    axes[0].set_xlabel(r"$t = \chi-1$")
    axes[0].set_ylabel(r"$m = 1-c_\ast$")
    if is_smooth:
        axes[1].set_title(f"{title}: Universal Collapse")
    axes[1].set_xlabel(r"$\tilde t$" if is_smooth else r"$u=(1-\chi)/(\kappa^{2/3}h^{1/3})$")
    axes[1].set_ylabel(r"$\tilde{m}$" if is_smooth else r"$m/(h/\kappa)^{2/3}$")
    axes[1].set_xlim(-1 if is_smooth else -1.25, 2)
    axes[1].set_ylim(0, 2.2 if is_smooth else 2.0)
    legend_kwargs = dict(ncol=2, borderaxespad=0.4, columnspacing=0.9, handletextpad=0.4)
    if is_smooth:
        axes[0].legend(loc="upper left", **legend_kwargs)
        axes[1].legend(loc="upper right", **legend_kwargs)
    else:
        axes[0].legend(loc="upper left", fancybox=True, shadow=True, framealpha=0.95, **legend_kwargs)
        axes[1].legend(loc="upper right", fancybox=True, shadow=True, framealpha=0.95, **legend_kwargs)

    name = "smooth_scaling_collapse.png" if is_smooth else "kinked_scaling_collapse.png"
    fig.tight_layout(pad=1.0, w_pad=1.4)
    save_figure(fig, OUT / "theory" / name)
    plt.close(fig)


def _prob_hermite_values(z: np.ndarray, nmax: int) -> list[np.ndarray]:
    values = [np.ones_like(z), z.copy()]
    for n in range(1, nmax):
        values.append(z * values[-1] - n * values[-2])
    return [v / math.sqrt(math.factorial(n)) for n, v in enumerate(values[: nmax + 1])]


def make_hermite() -> None:
    nmax = 36
    x, w = hermgauss(220)
    z = np.sqrt(2) * x
    weights = w / np.sqrt(np.pi)
    basis = _prob_hermite_values(z, nmax)
    relu = np.maximum(z, 0.0)
    tanh = np.tanh(z)
    relu_coeff = [abs(np.sum(weights * relu * basis[n])) for n in range(nmax + 1)]
    tanh_coeff = [abs(np.sum(weights * tanh * basis[n])) for n in range(nmax + 1)]
    n = np.arange(nmax + 1)

    relu_coeff = np.asarray(relu_coeff)
    tanh_coeff = np.asarray(tanh_coeff)
    relu_mask = relu_coeff > 1e-13
    tanh_mask = tanh_coeff > 1e-13

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.semilogy(n[relu_mask], relu_coeff[relu_mask], "s-", color=COLORS["kink"], label="ReLU")
    ax.semilogy(n[tanh_mask], tanh_coeff[tanh_mask], "o-", color=COLORS["smooth"], label=r"$\tanh$")
    ax.set_xlabel("Hermite degree n")
    ax.set_ylabel(r"$|a_n|$")
    ax.set_title("Hermite coefficient decay")
    ax.legend()
    save_figure(fig, OUT / "theory" / "hermite_decomposition.png")
    plt.close(fig)


def make_mlp_overfit() -> None:
    data = load_npz_result(RESULTS / "mlp" / "dropout_experiment_results.npz")
    results, theory = data["results"], data["theory"]
    order = ["constant", "reverse_step", "step", "reverse_linear", "linear", "none"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), sharex=True)
    panels = [
        ("train_loss", "Training loss", "Training loss", True),
        ("test_loss", "Test loss", "Test loss", True),
        ("train_acc", "Training accuracy", "Training accuracy (%)", False),
        ("test_acc", "Test accuracy", "Test accuracy (%)", False),
    ]
    for ax, (metric, title, ylabel, logy) in zip(axes.ravel(), panels):
        for sched in order:
            curve_with_band(ax, results[sched][metric], color=SCHEDULE_COLORS[sched], label=_display(sched, theory))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if ax in axes[1]:
            ax.set_xlabel("Epoch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save_figure(fig, OUT / "experiments" / "mlp" / "dropout_schedules_overfit.png")
    plt.close(fig)


def make_mlp_budget() -> None:
    data = load_npz_result(RESULTS / "mlp" / "dropout_MLP_triple.npz")
    results, theory = data["results"], data["theory"]
    order = ["none", "constant", "reverse_step", "big_step", "double", "triple"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True)
    panels = [
        ("train_loss", "Training loss", "Training loss", True),
        ("test_loss", "Test loss", "Test loss", True),
        ("train_acc", "Training accuracy", "Training accuracy (%)", False),
        ("test_acc", "Test accuracy", "Test accuracy (%)", False),
    ]
    for ax, (metric, title, ylabel, logy) in zip(axes.ravel(), panels):
        for sched in order:
            curve_with_band(ax, results[sched][metric], color=SCHEDULE_COLORS[sched], label=_display(sched, theory))
        _panel_label(ax, "abcd"[list(axes.ravel()).index(ax)], title)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if ax in axes[1]:
            ax.set_xlabel("Epoch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save_figure(fig, OUT / "experiments" / "mlp" / "dropout_budget_comparison.png")
    plt.close(fig)


def make_h_sweep() -> None:
    data = load_npz_result(RESULTS / "sweeps" / "h_bar_sweep_results.npz")
    all_results = data["all_results"]
    h_values = data["config"]["H_BAR_VALUES"]
    schedules = ["none", "constant", "reverse_step", "big_step"]
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for sched in schedules:
        means, sems = [], []
        for h in h_values:
            vals = best_per_seed(_entry(all_results, h, sched))
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(
            h_values,
            means,
            yerr=sems,
            color=SCHEDULE_COLORS[sched],
            marker=MARKERS[sched],
            capsize=3,
            label=DISPLAY_NAMES[sched],
        )
    ax.set_title(r"Dropout scheduling across $\bar h$ (MLP, CIFAR-10)")
    ax.set_xlabel(r"Mean dropout field $\bar h$")
    ax.set_ylabel("Best test accuracy (%)")
    ax.legend(loc="upper left")
    save_figure(fig, OUT / "experiments" / "sweeps" / "h_sweep_schedule_comparison.png")
    plt.close(fig)


def make_width_sweep() -> None:
    data = load_npz_result(RESULTS / "sweeps" / "width_sweep_results.npz")
    all_results = data["all_results"]
    widths = data["config"]["WIDTH_VALUES"]
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for sched in ["reverse_step", "big_step"]:
        means, sems = [], []
        for width in widths:
            vals = best_per_seed(_entry(all_results, width, sched)) - best_per_seed(_entry(all_results, width, "constant"))
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(
            widths,
            means,
            yerr=sems,
            color=SCHEDULE_COLORS[sched],
            marker=MARKERS[sched],
            capsize=3,
            label=f"{DISPLAY_NAMES[sched]} - Constant",
        )
    ax.axhline(0, color=COLORS["neutral"], ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_title(r"Scheduling advantage vs width at $\bar h=0.1$")
    ax.set_xlabel(r"Hidden width $N$")
    ax.set_ylabel(r"$\Delta$ best test accuracy (%)")
    ax.legend(loc="lower right")
    save_figure(fig, OUT / "experiments" / "sweeps" / "width_sweep_baseline_advantage.png")
    plt.close(fig)


def make_gelu_sweep() -> None:
    data = load_npz_result(RESULTS / "sweeps" / "gelu_h_bar_sweep_results.npz")
    all_results = data["all_results"]
    h_values = data["config"]["H_BAR_VALUES"]
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    for sched in ["reverse_step", "big_step"]:
        means, sems = [], []
        for h in h_values:
            vals = best_per_seed(_entry(all_results, h, sched)) - best_per_seed(_entry(all_results, h, "constant"))
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(
            h_values,
            means,
            yerr=sems,
            color=SCHEDULE_COLORS[sched],
            marker=MARKERS[sched],
            capsize=3,
            label=f"{DISPLAY_NAMES[sched]} - Constant",
        )
    ax.axhline(0, color=COLORS["neutral"], ls="--", lw=1)
    ax.set_title("Scheduling advantage relative to constant dropout")
    ax.set_xlabel(r"Mean dropout field $\bar h$")
    ax.set_ylabel(r"$\Delta$ best test accuracy (%)")
    ax.legend(loc="lower left")
    save_figure(fig, OUT / "experiments" / "sweeps" / "gelu_h_sweep_delta.pdf")
    plt.close(fig)


def _json_curve_array(values):
    return np.asarray(values, dtype=float)


def make_vit_curves(cropped: bool) -> None:
    results = load_json(RESULTS / "transformer" / "vit_dropout_results.json")
    order = ["none", "constant", "reverse_step", "reverse_linear"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True)
    panels = [
        ("train_loss", "Training loss", "Training loss", True),
        ("test_loss", "Test loss", "Test loss", False),
        ("train_acc", "Training accuracy (%)", "Training accuracy (%)", False),
        ("test_acc", "Test accuracy (%)", "Test accuracy (%)", False),
    ]
    start = 20 if cropped else 0
    for ax, (metric, title, ylabel, logy) in zip(axes.ravel(), panels):
        for sched in order:
            arr = _json_curve_array(results[sched][metric])
            x = np.arange(arr.shape[1])
            curve_with_band(ax, arr[:, start:], x=x[start:], color=SCHEDULE_COLORS[sched], label=DISPLAY_NAMES[sched])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if ax in axes[1]:
            ax.set_xlabel("Epoch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    name = "dropout_schedules_cropped.png" if cropped else "dropout_schedules_full.png"
    save_figure(fig, OUT / "experiments" / "transformer" / name)
    plt.close(fig)


def make_component_ablations() -> None:
    data = load_json(RESULTS / "transformer" / "ablation_results.json")
    modes = [("both", "Both (Attn + MLP)"), ("attn_only", "Attention only"), ("mlp_only", "MLP only")]
    schedules = ["none", "constant", "reverse_step"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, (mode, title) in zip(axes, modes):
        for sched in schedules:
            key = f"{mode}_{sched}"
            if key not in data:
                continue
            arr = _json_curve_array(data[key]["test_acc"])
            label = "No dropout" if sched == "none" else DISPLAY_NAMES[sched]
            curve_with_band(ax, arr, color=SCHEDULE_COLORS[sched], label=label)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(loc="lower right")
    fig.suptitle("Dropout scheduling component ablation", fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, OUT / "experiments" / "transformer" / "component_ablations.png")
    plt.close(fig)


def make_all() -> None:
    apply_paper_style()
    make_critical_exponents()
    make_scaling_collapse("smooth")
    make_scaling_collapse("kinked")
    make_hermite()
    make_mlp_overfit()
    make_mlp_budget()
    make_h_sweep()
    make_width_sweep()
    make_gelu_sweep()
    make_vit_curves(cropped=True)
    make_vit_curves(cropped=False)
    make_component_ablations()


# Figures copied into a LaTeX paper tree by --copy-to-paper, relative to OUT.
PAPER_FIGURE_FILES = [
    "theory/critical_exponents.png",
    "theory/critical_exponents.pdf",
    "theory/smooth_scaling_collapse.png",
    "theory/smooth_scaling_collapse.pdf",
    "theory/kinked_scaling_collapse.png",
    "theory/kinked_scaling_collapse.pdf",
    "theory/hermite_decomposition.png",
    "theory/hermite_decomposition.pdf",
    "experiments/mlp/dropout_schedules_overfit.png",
    "experiments/mlp/dropout_schedules_overfit.pdf",
    "experiments/mlp/dropout_budget_comparison.png",
    "experiments/mlp/dropout_budget_comparison.pdf",
    "experiments/sweeps/h_sweep_schedule_comparison.png",
    "experiments/sweeps/h_sweep_schedule_comparison.pdf",
    "experiments/sweeps/width_sweep_baseline_advantage.png",
    "experiments/sweeps/width_sweep_baseline_advantage.pdf",
    "experiments/sweeps/gelu_h_sweep_delta.pdf",
    "experiments/transformer/dropout_schedules_cropped.png",
    "experiments/transformer/dropout_schedules_cropped.pdf",
    "experiments/transformer/dropout_schedules_full.png",
    "experiments/transformer/dropout_schedules_full.pdf",
    "experiments/transformer/component_ablations.png",
    "experiments/transformer/component_ablations.pdf",
]


def copy_to_paper(dest: Path) -> None:
    for rel in PAPER_FIGURE_FILES:
        src = OUT / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing figure: {src} (run with --all first)")
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {src.relative_to(REPO_ROOT)} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Make every paper figure.")
    parser.add_argument(
        "--copy-to-paper",
        type=Path,
        metavar="DIR",
        help="Copy the figures into the given LaTeX paper figures/ directory.",
    )
    args = parser.parse_args()

    if not args.all and args.copy_to_paper is None:
        parser.error("Nothing to do; pass --all and/or --copy-to-paper DIR.")
    if args.all:
        make_all()
    if args.copy_to_paper is not None:
        copy_to_paper(args.copy_to_paper)


if __name__ == "__main__":
    main()
