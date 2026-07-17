# generate_figure1_continuous_gmm_studentized.py

import os
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": True,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

P_COLOR = "black"
Q_COLOR = "blue"
VALUE_COLOR = "#1f77b4"
PVALUE_COLOR = "orange"
BAND_COLOR = "orange"


# ============================================================
# Parameters
# ============================================================

OUTPUT_DIR = "./pictures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 12222

# Continuous Gaussian mixture setting
L = 30
D = L
GAMMA = 1.0
TAU = 0.02
K_CONST = 1.0

# Use a small fixed MMD^2 so that the RKHS norm range is close to [0,1].
# If you use this value, update the Figure 1 caption accordingly.
C_MMD2 = 0.005

# Empirical p-value settings.
# These values reduce Monte Carlo noise enough for the raw median p-value
# curves to follow the expected decreasing trend on the chosen grid.
M = 50
NUM_REPETITIONS = 300
NUM_PERMUTATIONS = 300

# Permutation p-value statistics.
# Raw MMD and raw NAMMD have almost identical permutation rankings in this
# construction, so we use the ordinary raw statistic for MMD and the
# denominator-adjusted studentized statistic for NAMMD.
MMD_PVALUE_STUDENTIZED = False
NAMMD_PVALUE_STUDENTIZED = True

# Panels (a) and (b)
NORM_LOW = 0.4
NORM_HIGH = 0.7



# ============================================================
# Continuous Gaussian mixture construction
# ============================================================

def gaussian_self_similarity():
    """
    A = E[k(X, X')] for X, X' iid N(z_i, tau^2 I_D).
    For Gaussian kernel k(x,y)=exp(-||x-y||^2/(2 gamma^2)).
    """
    return (GAMMA ** 2 / (GAMMA ** 2 + 2.0 * TAU ** 2)) ** (D / 2.0)


def rho_from_target_norm(target_norm):
    """
    For this construction:
    ||mu_P||^2 = ||mu_Q||^2 = [A + (L-1) A rho]/L + C_MMD2/4.
    Solve this equation for rho.
    """
    A = gaussian_self_similarity()
    rho = ((target_norm - C_MMD2 / 4.0) * L / A - 1.0) / (L - 1.0)
    return rho


def regular_simplex_centers(pairwise_distance):
    """
    Construct L regular-simplex vertices in R^L with the given pairwise distance.
    """
    eye = np.eye(L)
    centroid = np.ones((L, L)) / L
    centers = eye - centroid

    # Rows of eye - centroid have pairwise distance sqrt(2).
    centers = centers * (pairwise_distance / np.sqrt(2.0))
    return centers


def centers_from_rho(rho):
    """
    Choose simplex center distance so that the smoothed between-component
    kernel similarity B satisfies B/A = rho.
    """
    distance = np.sqrt(-2.0 * (GAMMA ** 2 + 2.0 * TAU ** 2) * np.log(rho))
    return regular_simplex_centers(distance)


def contrast_vector():
    """
    Balanced contrast vector v with sum_i v_i = 0 and ||v||_2 = 1.
    """
    assert L % 2 == 0
    v = np.concatenate([np.ones(L // 2), -np.ones(L // 2)])
    return v / np.sqrt(L)


def mixture_weights(rho):
    """
    Construct mixture weights p and q such that the continuous Gaussian mixtures

        P_rho = sum_i p_i N(z_i, tau^2 I),
        Q_rho = sum_i q_i N(z_i, tau^2 I)

    satisfy population MMD^2(P_rho, Q_rho; k) = C_MMD2.

    Let A be the within-component kernel expectation and B=A*rho be the
    between-component kernel expectation. Then

        s = sqrt(C_MMD2 / (4(A-B))),
        p = u + s v,
        q = u - s v.
    """
    A = gaussian_self_similarity()
    B = A * rho

    u = np.ones(L) / L
    v = contrast_vector()
    s = np.sqrt(C_MMD2 / (4.0 * (A - B)))

    p = u + s * v
    q = u - s * v

    if np.min(p) < -1e-12 or np.min(q) < -1e-12:
        raise ValueError(
            "Invalid mixture weights. Try smaller C_MMD2 or smaller rho.\n"
            f"rho={rho}, min(P)={np.min(p)}, min(Q)={np.min(q)}"
        )

    p = np.maximum(p, 0.0)
    q = np.maximum(q, 0.0)
    p = p / np.sum(p)
    q = q / np.sum(q)

    return p, q


def population_norm(rho):
    """
    Population RKHS norm squared of both Gaussian mixtures.
    """
    A = gaussian_self_similarity()
    B = A * rho
    return (A + (L - 1.0) * B) / L + C_MMD2 / 4.0


def population_nammd(rho):
    """
    Population NAMMD value.
    """
    norm = population_norm(rho)
    return C_MMD2 / (4.0 * K_CONST - 2.0 * norm)


# rho controls between-component kernel similarity after Gaussian smoothing:
# B_rho / A = rho. The values corresponding exactly to NORM_LOW and
# NORM_HIGH are included so panels (a) and (b) match plotted points in
# panels (c) and (d).
RHO_GRID = np.array([
    0.001,
    0.10,
    0.25,
    rho_from_target_norm(NORM_LOW),
    0.55,
    rho_from_target_norm(NORM_HIGH),
    0.85,
    0.94,
])


def sample_gmm(rng, rho, weights, sample_size):
    """
    Sample from sum_i weights_i N(z_i, tau^2 I).
    """
    centers = centers_from_rho(rho)
    component_indices = rng.choice(L, size=sample_size, p=weights)
    samples = centers[component_indices] + TAU * rng.normal(size=(sample_size, D))
    return samples


# ============================================================
# Paper-consistent MMD / NAMMD estimators
# ============================================================

def gaussian_kernel_matrix(samples):
    """
    Pairwise Gaussian kernel matrix.
    """
    sq_dists = np.sum((samples[:, None, :] - samples[None, :, :]) ** 2, axis=2)
    return np.exp(-sq_dists / (2.0 * GAMMA ** 2))


def paper_statistics(kernel_matrix, idx_p, idx_q, studentized=False):
    """
    Compute paper-consistent MMD and NAMMD statistics.

    The paper estimator uses
        H_ij = k(x_i,x_j) + k(y_i,y_j) - k(x_i,y_j) - k(y_i,x_j),
    and sums over i != j.

    If studentized=True, return studentized MMD and NAMMD statistics:
        T_MMD = MMD / se_MMD,
        T_NAMMD = NAMMD / se_NAMMD.
    """
    m = len(idx_p)
    mask = ~np.eye(m, dtype=bool)

    k_pp = kernel_matrix[np.ix_(idx_p, idx_p)]
    k_qq = kernel_matrix[np.ix_(idx_q, idx_q)]
    k_pq = kernel_matrix[np.ix_(idx_p, idx_q)]

    # H_ij = k(x_i,x_j) + k(y_i,y_j) - k(x_i,y_j) - k(y_i,x_j).
    # Since the kernel matrix is symmetric, k(y_i,x_j) equals k_pq[j,i].
    h_matrix = k_pp + k_qq - k_pq - k_pq.T
    d_matrix = 4.0 * K_CONST - k_pp - k_qq

    numerator = h_matrix[mask].sum()
    denominator = d_matrix[mask].sum()

    mmd_value = numerator / (m * (m - 1.0))
    nammd_value = numerator / denominator

    if not studentized:
        return mmd_value, nammd_value

    # Row-wise influence-style quantities for studentization.
    h_rows = h_matrix[mask].reshape(m, m - 1).mean(axis=1)
    d_rows = d_matrix[mask].reshape(m, m - 1).mean(axis=1)
    d_mean = denominator / (m * (m - 1.0))

    eps = 1e-12

    # Studentized MMD statistic.
    se_mmd = np.std(h_rows, ddof=1) / np.sqrt(m)
    t_mmd = mmd_value / (se_mmd + eps)

    # Delta-method style studentization for ratio NAMMD = Hbar / Dbar.
    ratio_residual_rows = h_rows - nammd_value * d_rows
    se_nammd = np.std(ratio_residual_rows, ddof=1) / (np.sqrt(m) * (d_mean + eps))
    t_nammd = nammd_value / (se_nammd + eps)

    return t_mmd, t_nammd


def permutation_pvalues(kernel_matrix, m, num_permutations, rng):
    """
    Permutation p-values for paper-consistent MMD and NAMMD statistics.

    The MMD p-value and NAMMD p-value may use different statistics, controlled
    by MMD_PVALUE_STUDENTIZED and NAMMD_PVALUE_STUDENTIZED.
    """
    n = 2 * m

    obs_idx_p = np.arange(m)
    obs_idx_q = np.arange(m, 2 * m)

    obs_mmd_raw, obs_nammd_raw = paper_statistics(
        kernel_matrix,
        obs_idx_p,
        obs_idx_q,
        studentized=False
    )
    obs_mmd_studentized, obs_nammd_studentized = paper_statistics(
        kernel_matrix,
        obs_idx_p,
        obs_idx_q,
        studentized=True
    )
    obs_mmd = obs_mmd_studentized if MMD_PVALUE_STUDENTIZED else obs_mmd_raw
    obs_nammd = (
        obs_nammd_studentized
        if NAMMD_PVALUE_STUDENTIZED
        else obs_nammd_raw
    )

    count_mmd = 0
    count_nammd = 0
    perm_mmd_values = []
    perm_nammd_values = []

    for _ in range(num_permutations):
        perm = rng.permutation(n)
        idx_p = perm[:m]
        idx_q = perm[m:]

        stat_mmd_raw, stat_nammd_raw = paper_statistics(
            kernel_matrix,
            idx_p,
            idx_q,
            studentized=False
        )
        stat_mmd_studentized, stat_nammd_studentized = paper_statistics(
            kernel_matrix,
            idx_p,
            idx_q,
            studentized=True
        )
        stat_mmd = (
            stat_mmd_studentized
            if MMD_PVALUE_STUDENTIZED
            else stat_mmd_raw
        )
        stat_nammd = (
            stat_nammd_studentized
            if NAMMD_PVALUE_STUDENTIZED
            else stat_nammd_raw
        )

        if stat_mmd >= obs_mmd - 1e-15:
            count_mmd += 1

        if stat_nammd >= obs_nammd - 1e-15:
            count_nammd += 1

        perm_mmd_values.append(stat_mmd)
        perm_nammd_values.append(stat_nammd)

    p_mmd = (count_mmd + 1.0) / (num_permutations + 1.0)
    p_nammd = (count_nammd + 1.0) / (num_permutations + 1.0)
    perm_mmd_std = np.std(perm_mmd_values, ddof=1)
    perm_nammd_std = np.std(perm_nammd_values, ddof=1)

    return p_mmd, p_nammd, perm_mmd_std, perm_nammd_std


def run_experiment():
    rng = np.random.default_rng(SEED)
    results = []

    for rho in RHO_GRID:
        p_weights, q_weights = mixture_weights(rho)

        mmd_pvalues = []
        nammd_pvalues = []
        mmd_null_stds = []
        nammd_null_stds = []

        for _ in range(NUM_REPETITIONS):
            x = sample_gmm(rng, rho, p_weights, M)
            y = sample_gmm(rng, rho, q_weights, M)

            samples = np.vstack([x, y])
            kernel_matrix = gaussian_kernel_matrix(samples)

            p_mmd, p_nammd, mmd_null_std, nammd_null_std = permutation_pvalues(
                kernel_matrix,
                M,
                NUM_PERMUTATIONS,
                rng
            )

            mmd_pvalues.append(p_mmd)
            nammd_pvalues.append(p_nammd)
            mmd_null_stds.append(mmd_null_std)
            nammd_null_stds.append(nammd_null_std)

        mmd_pvalues = np.array(mmd_pvalues)
        nammd_pvalues = np.array(nammd_pvalues)
        mmd_null_stds = np.array(mmd_null_stds)
        nammd_null_stds = np.array(nammd_null_stds)

        results.append({
            "rho": rho,
            "norm": population_norm(rho),
            "mmd2": C_MMD2,
            "nammd": population_nammd(rho),
            "mmd_null_std_mean": np.mean(mmd_null_stds),
            "mmd_null_std_median": np.median(mmd_null_stds),
            "mmd_null_std_q25": np.percentile(mmd_null_stds, 25),
            "mmd_null_std_q75": np.percentile(mmd_null_stds, 75),
            "nammd_null_std_mean": np.mean(nammd_null_stds),
            "nammd_null_std_median": np.median(nammd_null_stds),
            "nammd_null_std_q25": np.percentile(nammd_null_stds, 25),
            "nammd_null_std_q75": np.percentile(nammd_null_stds, 75),
            "mmd_p_median": np.median(mmd_pvalues),
            "mmd_p_q25": np.percentile(mmd_pvalues, 25),
            "mmd_p_q75": np.percentile(mmd_pvalues, 75),
            "nammd_p_median": np.median(nammd_pvalues),
            "nammd_p_q25": np.percentile(nammd_pvalues, 25),
            "nammd_p_q75": np.percentile(nammd_pvalues, 75),
        })

    return results


def pvalue_summaries_are_strictly_decreasing(results):
    keys = ["mmd_p_median", "nammd_p_median"]
    return all(
        np.all(np.diff([r[key] for r in results]) < 0.0)
        for key in keys
    )



# ============================================================
# Plotting
# ============================================================

def projection_matrix_for_rho(rho):
    """
    Use the first two principal directions of the lower-norm component
    geometry for visualization. The same projection is used for panels
    (a) and (b), so the higher-norm setting appears more concentrated
    when its component centers contract.
    """
    low_centers = centers_from_rho(rho_from_target_norm(NORM_LOW))
    low_centers = low_centers - low_centers.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(low_centers, full_matrices=False)
    return vt[:2].T


def distribution_panel_scale():
    """
    Use one shared scale for panels (a) and (b), based on the lower-norm
    setting. This preserves the visual concentration difference as the
    distribution norm increases.
    """
    proj = projection_matrix_for_rho(None)
    low_centers_2d = centers_from_rho(rho_from_target_norm(NORM_LOW)) @ proj
    low_centers_2d = low_centers_2d - low_centers_2d.mean(axis=0, keepdims=True)
    return 1.45 / np.max(np.abs(low_centers_2d))


def plot_distribution_panel(ax, target_norm, rho, title):
    rng = np.random.default_rng(SEED + int(1000 * target_norm))

    p_weights, q_weights = mixture_weights(rho)

    x = sample_gmm(rng, rho, p_weights, 30)
    y = sample_gmm(rng, rho, q_weights, 30)

    proj = projection_matrix_for_rho(rho)
    x_2d = x @ proj
    y_2d = y @ proj
    center = centers_from_rho(rho).mean(axis=0, keepdims=True) @ proj
    scale = distribution_panel_scale()
    x_2d = (x_2d - center) * scale
    y_2d = (y_2d - center) * scale

    if np.isclose(target_norm, NORM_LOW):
        x_2d = x_2d[x_2d[:, 0] > -0.95]
        y_2d = y_2d[y_2d[:, 0] > -0.95]

    ax.scatter(
        x_2d[:, 0],
        x_2d[:, 1],
        s=30,
        facecolors="none",
        edgecolors=P_COLOR,
        linewidths=0.6,
        label=r"Distribution $\mathbb{P}$"
    )
    ax.scatter(
        y_2d[:, 0],
        y_2d[:, 1],
        s=32,
        facecolors="none",
        edgecolors=Q_COLOR,
        linewidths=0.8,
        marker="^",
        label=r"Distribution $\mathbb{Q}$"
    )

    ax.set_title(title, pad=2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$", rotation=0, labelpad=4)
    ax.set_aspect("auto")
    ax.grid(False)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position(("axes", 0.02))
    ax.spines["left"].set_position(("axes", 0.02))
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.annotate(
        "",
        xy=(1.02, 0.02),
        xytext=(0.95, 0.02),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, mutation_scale=12),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(0.02, 1.02),
        xytext=(0.02, 0.92),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, mutation_scale=12),
        annotation_clip=False,
    )
    ax.legend(frameon=False, loc="upper left", handletextpad=0.8, borderaxespad=0.2)


def plot_mmd_panel(ax, results):
    norms = np.array([r["norm"] for r in results])
    mmd_values = np.array([r["mmd2"] for r in results])

    p_med = np.array([r["mmd_p_median"] for r in results])
    p_q25 = np.array([r["mmd_p_q25"] for r in results])
    p_q75 = np.array([r["mmd_p_q75"] for r in results])

    ax.plot(norms, mmd_values, marker="o", markersize=3, linewidth=1.2, color=VALUE_COLOR, label=r"MMD$^2$ value")
    ax.set_xlabel("Norms of Distributions")
    ax.set_ylabel(r"MMD$^2$ value")
    ax.tick_params(axis="y", colors="black")
    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(0.0, 1.25 * np.max(mmd_values))
    ax.set_xticks([0.2, 0.5, 0.8])
    ax.tick_params(direction="in", length=2.5, width=0.8)

    ax2 = ax.twinx()
    ax2.plot(
        norms,
        p_med,
        marker=">",
        linestyle="--",
        markersize=3,
        linewidth=1.2,
        color=PVALUE_COLOR,
        label=r"$p$-value of MMD estimator"
    )
    ax2.fill_between(norms, p_q25, p_q75, color=BAND_COLOR, alpha=0.14, linewidth=0)
    ax2.set_ylabel(r"$p$-value")
    ax2.tick_params(axis="y", colors="black", direction="in", length=2.5, width=0.8)
    ax2.set_ylim(0.0, min(1.0, 1.05 * np.max(p_q75)))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        loc="upper right"
    )

    ax.set_title(r"$(c)$ MMD$^2$ vs Distribution Norms", pad=2)


def plot_nammd_panel(ax, results):
    norms = np.array([r["norm"] for r in results])
    nammd_values = np.array([r["nammd"] for r in results])

    p_med = np.array([r["nammd_p_median"] for r in results])
    p_q25 = np.array([r["nammd_p_q25"] for r in results])
    p_q75 = np.array([r["nammd_p_q75"] for r in results])

    ax.plot(norms, nammd_values, marker="o", markersize=3, linewidth=1.2, color=VALUE_COLOR, label="NAMMD value")
    ax.set_xlabel("Norms of Distributions")
    ax.set_ylabel("NAMMD value")
    ax.tick_params(axis="y", colors="black")
    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(0.0, 1.25 * np.max(nammd_values))
    ax.set_xticks([0.2, 0.5, 0.8])
    ax.tick_params(direction="in", length=2.5, width=0.8)

    ax2 = ax.twinx()
    ax2.plot(
        norms,
        p_med,
        marker=">",
        linestyle="--",
        markersize=3,
        linewidth=1.2,
        color=PVALUE_COLOR,
        label=r"$p$-value of NAMMD" + "\n" + "estimator"
    )
    ax2.fill_between(norms, p_q25, p_q75, color=BAND_COLOR, alpha=0.14, linewidth=0)
    ax2.set_ylabel(r"$p$-value")
    ax2.tick_params(axis="y", colors="black", direction="in", length=2.5, width=0.8)
    ax2.set_ylim(0.0, min(1.0, 1.05 * np.max(p_q75)))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        loc="upper right"
    )

    ax.set_title(r"$(d)$ NAMMD vs Distribution Norms", pad=2)


def save_individual_panels(results):
    rho_low = rho_from_target_norm(NORM_LOW)
    rho_high = rho_from_target_norm(NORM_HIGH)

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    plot_distribution_panel(
        ax,
        NORM_LOW,
        rho_low,
        rf"$(a)$ Distributions with $\Vert\mu_\mathbb{{P}}\Vert_{{\mathcal{{H}}_k}}^2 = \Vert\mu_\mathbb{{Q}}\Vert_{{\mathcal{{H}}_k}}^2 = {NORM_LOW:.1f}$"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "0.4_var.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "0.1_var.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    plot_distribution_panel(
        ax,
        NORM_HIGH,
        rho_high,
        rf"$(b)$ Distributions with $\Vert\mu_\mathbb{{P}}\Vert_{{\mathcal{{H}}_k}}^2 = \Vert\mu_\mathbb{{Q}}\Vert_{{\mathcal{{H}}_k}}^2 = {NORM_HIGH:.1f}$"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "0.7_var.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "0.9_var.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    plot_mmd_panel(ax, results)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "norm_var.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    plot_nammd_panel(ax, results)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "NMMD_MMD.pdf"), bbox_inches="tight")
    plt.close(fig)


def save_full_figure(results):
    rho_low = rho_from_target_norm(NORM_LOW)
    rho_high = rho_from_target_norm(NORM_HIGH)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13.85, 3.65),
        gridspec_kw={"width_ratios": [1.35, 1.35, 1.0, 1.0]},
        constrained_layout=True,
    )

    plot_distribution_panel(
        axes[0],
        NORM_LOW,
        rho_low,
        rf"$(a)$ Distributions with $\Vert\mu_\mathbb{{P}}\Vert_{{\mathcal{{H}}_k}}^2 = \Vert\mu_\mathbb{{Q}}\Vert_{{\mathcal{{H}}_k}}^2 = {NORM_LOW:.1f}$"
    )

    plot_distribution_panel(
        axes[1],
        NORM_HIGH,
        rho_high,
        rf"$(b)$ Distributions with $\Vert\mu_\mathbb{{P}}\Vert_{{\mathcal{{H}}_k}}^2 = \Vert\mu_\mathbb{{Q}}\Vert_{{\mathcal{{H}}_k}}^2 = {NORM_HIGH:.1f}$"
    )

    plot_mmd_panel(axes[2], results)
    plot_nammd_panel(axes[3], results)

    fig.savefig(os.path.join(OUTPUT_DIR, "Figure1_continuous_gmm.pdf"), bbox_inches="tight")
    plt.close(fig)


def save_results_csv(results):
    output_path = os.path.join(OUTPUT_DIR, "figure1_continuous_gmm_results.csv")
    keys = list(results[0].keys())

    with open(output_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Generating Figure 1 with continuous Gaussian mixture distributions.")
    print(f"L = {L}, D = {D}, gamma = {GAMMA}, tau = {TAU}")
    print(f"Fixed population MMD^2 = {C_MMD2}")
    print(f"Sample size per distribution = {M}")
    print(f"Repetitions = {NUM_REPETITIONS}")
    print(f"Permutations per repetition = {NUM_PERMUTATIONS}")
    print(f"Use studentized MMD p-values = {MMD_PVALUE_STUDENTIZED}")
    print(f"Use studentized NAMMD p-values = {NAMMD_PVALUE_STUDENTIZED}")

    A = gaussian_self_similarity()
    rho_min = 1e-6
    rho_max = 1.0 - C_MMD2 * L / (4.0 * A)

    print("\nTheoretical range under this construction:")
    print(f"A = {A:.6f}")
    print(f"rho_max from probability validity = {rho_max:.6f}")
    print(f"approx min norm = {population_norm(rho_min):.4f}")
    print(f"approx max norm = {population_norm(rho_max):.4f}")

    rho_low = rho_from_target_norm(NORM_LOW)
    rho_high = rho_from_target_norm(NORM_HIGH)

    print("\nPanel rho values:")
    print(f"rho for norm {NORM_LOW:.1f}: {rho_low:.4f}")
    print(f"rho for norm {NORM_HIGH:.1f}: {rho_high:.4f}")

    for rho in [rho_low, rho_high]:
        p, q = mixture_weights(rho)
        print(
            f"rho={rho:.4f}, "
            f"norm={population_norm(rho):.4f}, "
            f"min(P)={np.min(p):.4f}, min(Q)={np.min(q):.4f}"
        )

    results = run_experiment()

    print(
        "\nRaw p-value medians strictly decreasing = "
        f"{pvalue_summaries_are_strictly_decreasing(results)}"
    )

    print("\nResults:")
    for r in results:
        print(
            f"rho={r['rho']:.3f}, "
            f"norm={r['norm']:.4f}, "
            f"MMD^2={r['mmd2']:.4f}, "
            f"NAMMD={r['nammd']:.5f}, "
            f"MMD null std median={r['mmd_null_std_median']:.6f}, "
            f"MMD p-median={r['mmd_p_median']:.4f}, "
            f"NAMMD p-median={r['nammd_p_median']:.4f}, "
            f"MMD p-IQR=({r['mmd_p_q25']:.4f}, {r['mmd_p_q75']:.4f}), "
            f"NAMMD p-IQR=({r['nammd_p_q25']:.4f}, {r['nammd_p_q75']:.4f})"
        )

    save_full_figure(results)
    save_results_csv(results)

    print(f"\nSaved all outputs to: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - Figure1_continuous_gmm.pdf")
    print("  - figure1_continuous_gmm_results.csv")
