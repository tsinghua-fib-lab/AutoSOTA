"""
This script reproduces the plots in Example 4.8. The pathway abstractions are:
1- Bivariate abstraction B2 -> B3,
2- Trivariate abstraction B1 -> B3 <- B2, using the full abstract cluster.

"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal



# Parameters
alpha = -0.9
beta = 0.9
gamma = 0.9
x_context = 1.0

grid_size = 30

y_min, y_max = 0.0, 3
z_min, z_max = 0.0, 3


# Model quantities
sigma_NZ_sq = 1 - (gamma**2 + beta**2 + 2 * alpha * beta * gamma)

if sigma_NZ_sq <= 0:
    raise ValueError("sigma_NZ_sq must be positive.")

# Observational covariance of (X,Y,Z)
cov_XYZ = np.array([
    [1.0, alpha, beta + gamma * alpha],
    [alpha, 1.0, gamma + alpha * beta],
    [beta + gamma * alpha, gamma + alpha * beta, 1.0],
])

# Observational covariance of (X,Y)
cov_XY = np.array([
    [1.0, alpha],
    [alpha, 1.0],
])

# Observational covariance of (Y,Z)
hat_gamma = gamma + alpha * beta
cov_obs_YZ = np.array([
    [1.0, hat_gamma],
    [hat_gamma, 1.0],
])

# Covariance of (Y, Z) and variance of Z under do(B2 = .)
var_Z_do = beta**2 + gamma**2 + sigma_NZ_sq
cov_do_YZ = np.array([
    [1.0, gamma],
    [gamma, var_Z_do],
])

# Covariance of (X, Y, Z) under do(B2 = .) where X is independent of Y
Sigma_ind = np.array([
    [1.0, 0.0, beta],
    [0.0, 1.0, gamma],
    [beta, gamma, var_Z_do],
])


# Gaussian probabilities
def prob_box(lower, upper, cov):
    """Probability that a centered Gaussian vector lies in one box."""
    dim = len(lower)
    mean = np.zeros(dim)
    return float(
        multivariate_normal.cdf(
            upper,
            mean=mean,
            cov=cov,
            lower_limit=lower,
        )
    )


def union_prob_box(interval_lists, cov):
    """
    Probability that a centered Gaussian vector lies in a product of
    one-dimensional interval unions.

    """
    total = 0.0

    for pieces in itertools.product(*interval_lists):
        lower = np.array([piece[0] for piece in pieces], dtype=float)
        upper = np.array([piece[1] for piece in pieces], dtype=float)
        total += prob_box(lower, upper, cov)

    return float(np.clip(total, 0.0, 1.0))


def intervals_B1(b1):
    if b1 == 1:
        return [(-x_context, x_context)]
    return [(-np.inf, -x_context), (x_context, np.inf)]


def intervals_B2(b2, y):
    if b2 == 1:
        return [(y, np.inf)]
    return [(-np.inf, y)]


def intervals_B3(b3, z):
    if b3 == 1:
        return [(z, np.inf)]
    return [(-np.inf, z)]


# Observational probability P(B1=b1, B2=b2, B3=b3)
def p_xyz_obs(b1, b2, b3, y, z):

    return union_prob_box(
        [intervals_B1(b1), intervals_B2(b2, y), intervals_B3(b3, z)],
        cov_XYZ,
    )

# Observational probability P(B1=b1, B2=b2)
def p_xy_obs(b1, b2, y):

    return union_prob_box(
        [intervals_B1(b1), intervals_B2(b2, y)],
        cov_XY,
    )


def prob_B1(b1):

    p_inside = norm.cdf(x_context) - norm.cdf(-x_context)
    return p_inside if b1 == 1 else 1.0 - p_inside


def prob_B2(b2, y):

    return norm.sf(y) if b2 == 1 else norm.cdf(y)



def index_to_state(idx):
    """Decode index 0,...,7 into (b1,b2,b3)."""
    b1 = idx // 4
    rem = idx % 4
    b2 = rem // 2
    b3 = rem % 2
    return b1, b2, b3



# Fine-grained micro-realization interventional probability of (b1, b2, b3)
# under do(BS = bs)
def p_micro(S, bS, b1, b2, b3, y, z):
    S = frozenset(S)
    state = {1: b1, 2: b2, 3: b3}


    for node, val in bS.items():
        if state[node] != val:
            return 0.0

    if S == frozenset():
        return p_xyz_obs(b1, b2, b3, y, z)

    # do(B1=s1)
    if S == frozenset({1}):
        s1 = bS[1]
        return p_xyz_obs(s1, b2, b3, y, z) / prob_B1(s1)

    # do(B2=s2)
    if S == frozenset({2}):
        s2 = bS[2]
        numerator = union_prob_box(
            [intervals_B1(b1), intervals_B2(s2, y), intervals_B3(b3, z)],
            Sigma_ind,
        )
        return numerator / prob_B2(s2, y)

    # do(B3=s3)
    if S == frozenset({3}):
        return p_xy_obs(b1, b2, y)

    # do(B1=s1,B2=s2)
    if S == frozenset({1, 2}):
        s1 = bS[1]
        s2 = bS[2]
        numerator = union_prob_box(
            [intervals_B1(s1), intervals_B2(s2, y), intervals_B3(b3, z)],
            Sigma_ind,
        )
        return numerator / (prob_B1(s1) * prob_B2(s2, y))

    # do(B1=s1,B3=s3)
    if S == frozenset({1, 3}):
        s1 = bS[1]
        return p_xy_obs(s1, b2, y) / prob_B1(s1)

    # do(B2=s2,B3=s3)
    if S == frozenset({2, 3}):
        return prob_B1(b1)

    # do(B1=s1,B2=s2,B3=s3)
    if S == frozenset({1, 2, 3}):
        return 1.0


# Fine-grained micro-realization interventional probability distribution under 
# do(BS=bs)
def p_micro_distribution(S, bS, y, z):

    dist = np.zeros(8)

    for idx in range(8):
        b1, b2, b3 = index_to_state(idx)
        dist[idx] = p_micro(S, bS, b1, b2, b3, y, z)

    dist = np.clip(dist, 0.0, 1.0)

    if dist.sum() > 0:
        dist = dist / dist.sum()

    return dist


# Mechanisms for the abstract full DAG
def abstract_mechanisms(p_obs, eps=1e-14):
    
    P = np.asarray(p_obs, dtype=float)
    P = np.clip(P, eps, None)
    P = P / P.sum()

    p1 = np.zeros(2)
    p2 = np.zeros((2, 2))
    p3 = np.zeros((2, 2, 2))

    for b1 in [0, 1]:
        p1[b1] = sum(
            P[4 * b1 + 2 * b2 + b3]
            for b2 in [0, 1]
            for b3 in [0, 1]
        )

    for b1 in [0, 1]:
        denom = max(p1[b1], eps)
        for b2 in [0, 1]:
            numer = sum(P[4 * b1 + 2 * b2 + b3] for b3 in [0, 1])
            p2[b1, b2] = numer / denom

    for b1 in [0, 1]:
        for b2 in [0, 1]:
            denom = max(
                sum(P[4 * b1 + 2 * b2 + b3] for b3 in [0, 1]),
                eps,
            )
            for b3 in [0, 1]:
                p3[b1, b2, b3] = P[4 * b1 + 2 * b2 + b3] / denom

    return p1, p2, p3


# p_B(B | do(BS=bs)) in the abstract full DAG 
def abstract_distribution_intervention(S, bS, mechanisms):
    
    S = set(S)
    p1, p2, p3 = mechanisms

    dist = np.zeros(8)

    for b1 in [0, 1]:
        for b2 in [0, 1]:
            for b3 in [0, 1]:
                state = {1: b1, 2: b2, 3: b3}

                if any(state[j] != bS[j] for j in S):
                    continue

                prob = 1.0

                # Truncated factorization.
                if 1 not in S:
                    prob *= p1[b1]
                if 2 not in S:
                    prob *= p2[b1, b2]
                if 3 not in S:
                    prob *= p3[b1, b2, b3]

                dist[4 * b1 + 2 * b2 + b3] = prob

    if dist.sum() > 0:
        dist = dist / dist.sum()

    return dist


def categorical_kl(p, q, eps=1e-14):
    """KL(p || q) for categorical distributions"""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.clip(p, 0.0, 1.0)
    q = np.clip(q, eps, 1.0)

    p = p / p.sum()
    q = q / q.sum()

    mask = p > eps
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def bernoulli_kl(p, q, eps=1e-12):
    """KL(Bern(p) || Bern(q))"""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)

    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def all_intervention_configs():
    """Generate all pairs (S,bS) for S subset {1,2,3}."""
    nodes = [1, 2, 3]

    for r in range(4):
        for S_tuple in itertools.combinations(nodes, r):
            for values in itertools.product([0, 1], repeat=r):
                bS = {node: val for node, val in zip(S_tuple, values)}
                yield tuple(S_tuple), bS


ALL_CONFIGS = list(all_intervention_configs())



# Bivariate accuracy and explanation score 
def bivar_cdf(y, z, cov):
    """P(Y <= y, Z <= z) for a centered bivariate Gaussian"""
    return float(multivariate_normal.cdf([y, z], mean=[0.0, 0.0], cov=cov))


def prob_Y_geq_Z_geq(y, z, cov):
    """P(Y >= y, Z >= z) for a centered bivariate Gaussian"""
    sd_z = np.sqrt(cov[1, 1])

    return (
        1
        - norm.cdf(y)
        - norm.cdf(z / sd_z)
        + bivar_cdf(y, z, cov)
    )


def prob_Y_less_Z_geq(y, z, cov):
    """P(Y < y, Z >= z) for a centered bivariate Gaussian"""
    return norm.cdf(y) - bivar_cdf(y, z, cov)


def bivariate_accuracy(y, z):

    q1 = prob_Y_geq_Z_geq(y, z, cov_obs_YZ) / norm.sf(y)
    p1 = prob_Y_geq_Z_geq(y, z, cov_do_YZ) / norm.sf(y)

    q0 = prob_Y_less_Z_geq(y, z, cov_obs_YZ) / norm.cdf(y)
    p0 = prob_Y_less_Z_geq(y, z, cov_do_YZ) / norm.cdf(y)

    max_kl = max(bernoulli_kl(p1, q1), bernoulli_kl(p0, q0))

    return 1 - max_kl / (-np.log(norm.sf(z)))

def bivariate_explanation_score(y, z):
     
    p_cond = prob_Y_geq_Z_geq(y, z, cov_obs_YZ) / norm.sf(y)
    p_target = norm.sf(z)

    eps = 1e-300
    return 1 - np.log(max(p_cond, eps)) / np.log(max(p_target, eps))



# Trivariate full accuracy and explanation score 
def trivariate_accuracy(y, z):

    p_obs = p_micro_distribution((), {}, y, z)
    mechanisms = abstract_mechanisms(p_obs)

    max_kl = 0.0

    for S, bS in ALL_CONFIGS:
        p_micro = p_micro_distribution(S, bS, y, z)
        p_abs = abstract_distribution_intervention(S, bS, mechanisms)
        max_kl = max(max_kl, categorical_kl(p_micro, p_abs))

    return 1 - max_kl / (-np.log(norm.sf(z)))


def trivariate_explanation_score(y, z):
    
    p_B1 = prob_B1(1)

    p_context = p_xy_obs(1, 1, y)
    p_joint = p_xyz_obs(1, 1, 1, y, z)

    eps = 1e-300
    p_cond = p_joint / max(p_context, eps)
    p_pathway = p_B1 * p_cond
    p_target = norm.sf(z)

    return 1 - np.log(max(p_pathway, eps)) / np.log(max(p_target, eps))



# Compute grids
y_vals = np.linspace(y_min, y_max, grid_size)
z_vals = np.linspace(z_min, z_max, grid_size)

Y_grid, Z_grid = np.meshgrid(y_vals, z_vals)

A_bi = np.zeros_like(Y_grid, dtype=float)
A_tri = np.zeros_like(Y_grid, dtype=float)

E_bi = np.zeros_like(Y_grid, dtype=float)
E_tri = np.zeros_like(Y_grid, dtype=float)

print("Computing accuracy and explanation score grids")

for i, z in enumerate(z_vals):
    for j, y in enumerate(y_vals):
        A_bi[i, j] = bivariate_accuracy(y, z)
        A_tri[i, j] = trivariate_accuracy(y, z)

        E_bi[i, j] = bivariate_explanation_score(y, z)
        E_tri[i, j] = trivariate_explanation_score(y, z)

    print(f"row {i + 1}/{len(z_vals)} done")


# Plot 2x2 figure: explanation scores and accuracy

# Explanation score panels
combined_exp = np.concatenate([
    E_bi[np.isfinite(E_bi)].ravel(),
    E_tri[np.isfinite(E_tri)].ravel(),
])

levels_exp = np.linspace(
    np.nanpercentile(combined_exp, 5),
    np.nanpercentile(combined_exp, 95),
    8,
)

# Accuracy panels
combined_acc = np.concatenate([
    A_bi[np.isfinite(A_bi)].ravel(),
    A_tri[np.isfinite(A_tri)].ravel(),
])

levels_acc = np.linspace(
    np.nanpercentile(combined_acc, 5),
    np.nanpercentile(combined_acc, 95),
    8,
)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

fig, axes = plt.subplots(
    2,
    2,
    figsize=(6.8, 5.4),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

# Column titles
axes[0, 0].set_title(r"$B_2 \to B_3$", fontsize=12)
axes[0, 1].set_title(r"$B_1 \to B_3 \leftarrow B_2$", fontsize=12)

# Panel data
panels = [
    (axes[0, 0], E_bi, levels_exp),
    (axes[0, 1], E_tri, levels_exp),
    (axes[1, 0], A_bi, levels_acc),
    (axes[1, 1], A_tri, levels_acc),
]

filled_exp = None
filled_acc = None

for ax, grid, levels in panels:
    filled = ax.contourf(
        Y_grid,
        Z_grid,
        grid,
        levels=levels,
        alpha=0.9,
    )

    contours = ax.contour(
        Y_grid,
        Z_grid,
        grid,
        levels=levels,
        cmap=filled.cmap,
        norm=filled.norm,
        linewidths=0.5,
    )

    ax.clabel(contours, inline=True, fontsize=6.0, fmt="%.2f")

    ax.set_xlim(y_min, y_max)
    ax.set_ylim(z_min, z_max)

    if ax in axes[0, :]:
        filled_exp = filled
    else:
        filled_acc = filled

# Axis labels
axes[1, 0].set_xlabel(r"$y$")
axes[1, 1].set_xlabel(r"$y$")

axes[0, 0].set_ylabel(r"$z$")
axes[1, 0].set_ylabel(r"$z$")

# Row labels
axes[0, 0].text(
    -0.28,
    0.5,
    "Explanation score",
    transform=axes[0, 0].transAxes,
    rotation=90,
    va="center",
    ha="center",
)

axes[1, 0].text(
    -0.28,
    0.5,
    "Accuracy",
    transform=axes[1, 0].transAxes,
    rotation=90,
    va="center",
    ha="center",
)

# Separate colorbars for explanation and accuracy
cbar_exp = fig.colorbar(
    filled_exp,
    ax=axes[0, :],
    shrink=0.9,
    pad=0.02,
)
cbar_exp.set_label("Explanation score")

cbar_acc = fig.colorbar(
    filled_acc,
    ax=axes[1, :],
    shrink=0.9,
    pad=0.02,
)
cbar_acc.set_label("Accuracy")

fig.savefig("gaussian_example_scores.pdf", bbox_inches="tight")
fig.savefig("gaussian_example_scores.png", dpi=600, bbox_inches="tight")

plt.show()

print("Saved gaussian_example_scores.pdf")
print("Saved gaussian_example_scores.png")