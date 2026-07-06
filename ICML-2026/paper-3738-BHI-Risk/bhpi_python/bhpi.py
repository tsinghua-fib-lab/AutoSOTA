"""
BHPI: Bayesian Hypergraph Pathway Inference - Python Port
=========================================================
Port of the MATLAB implementation from:
https://github.com/Naomi-Ding/BHPI

Paper: "Disentangling Latent Risk Pathways via Bayesian Hypergraph Inference"
ICML 2026 Oral

This module ports all MATLAB helper functions and the core BHPI algorithm
to Python using NumPy, SciPy, and scikit-learn.
"""

import numpy as np
from scipy.special import digamma, betaln
from scipy.optimize import linear_sum_assignment, root_scalar
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression


def simulate_mixed_hypergraph(V, E, rare_idx, common_idx,
                              nRarePerEdge, nCommonPerEdge, seed):
    """Generate a mixed hypergraph with forced rare/common disease mixing."""
    rng = np.random.RandomState(seed)
    H = np.zeros((V, E), dtype=int)

    for e in range(E):
        r_idx = rng.choice(rare_idx, size=nRarePerEdge, replace=False)
        c_idx = rng.choice(common_idx, size=nCommonPerEdge, replace=False)
        H[np.concatenate([r_idx, c_idx]), e] = 1

    for v in common_idx:
        if H[v, :].sum() == 0:
            e = rng.randint(0, E)
            H[v, e] = 1

    for v in rare_idx:
        if H[v, :].sum() < 2:
            e = rng.randint(0, E)
            H[v, e] = 1

    return H


def E_Overlap(rho_ast):
    """Compute expected overlap between all pairs of hyperedges.

    Args:
        rho_ast: (V, E) membership probabilities

    Returns:
        E_O: (E, E) expected overlap matrix
        E_O_m_diff: (E, E, V) difference in overlap when vertex membership toggled
    """
    V, E_edges = rho_ast.shape

    H = rho_ast  # V x E
    norms = H.sum(axis=0)  # E

    # --- Vectorized E_O ---
    # Numerator: H^T @ H = (E, E) matrix of dot products
    H_float = H.astype(np.float64)
    numerator = H_float.T @ H_float  # (E, E)
    # Denominator: min(norm[e1], norm[e2])
    min_norms = np.minimum(norms[:, np.newaxis], norms[np.newaxis, :])  # (E, E)
    E_O = np.divide(numerator, min_norms, out=np.zeros_like(numerator), where=min_norms > 0)

    # --- Vectorized E_O_m_diff ---
    # For each e1, e2, v:
    #   val_v0 = overlap without vertex v in e1
    #   val_v1 = overlap with vertex v in e1
    #   diff = val_v1 - val_v0
    #
    # Without v: dot_product_without_v = numerator[e1,e2] - H[v,e1] * H[v,e2]
    #            norm_without_v = norms[e1] - H[v,e1]
    # With v:    dot_product_with_v = dot_product_without_v + H[v,e2]
    #            norm_with_v = norm_without_v + 1
    # m1_v0 = min(norm_without_v, norms[e2])
    # m1_v1 = min(norm_with_v, norms[e2])

    # dot_product_without_v(e1, e2, v) = numerator[e1,e2] - H[v,e1] * H[v,e2]
    # Shape: (E, E, V)
    dot_without_v = numerator[:, :, np.newaxis] - H.T[np.newaxis, :, :] * H.T[:, np.newaxis, :]  # (E, E, V)
    # norm_without_v(e1, v) = norms[e1] - H[v,e1]
    # Shape: (E, V) -> broadcast to (E, E, V)
    norm_without_e1 = norms[:, np.newaxis] - H.T[:, :]  # (E, V)
    norm_with_e1 = norm_without_e1 + 1  # (E, V)

    # m1_v0(e1, e2, v) = min(norm_without_e1[e1,v], norms[e2])
    m1_v0 = np.minimum(norm_without_e1[:, np.newaxis, :], norms[np.newaxis, :, np.newaxis])  # (E, E, V)
    m1_v1 = np.minimum(norm_with_e1[:, np.newaxis, :], norms[np.newaxis, :, np.newaxis])  # (E, E, V)

    val_v0 = np.divide(dot_without_v, m1_v0, out=np.zeros_like(dot_without_v), where=m1_v0 > 0)
    val_v1 = np.divide(dot_without_v + H.T[np.newaxis, :, :], m1_v1, out=np.zeros_like(dot_without_v), where=m1_v1 > 0)

    E_O_m_diff = val_v1 - val_v0  # (E, E, V)

    return E_O, E_O_m_diff


def simulate_mechanisms(P, H, mu_mean, mu_sd, seed=None):
    """Generate mechanism indicators and effect sizes."""
    rng = np.random.RandomState(seed)
    E = H.shape[1]

    gamma = np.zeros((P, E))
    for j in range(P):
        e = rng.randint(0, E)
        gamma[j, e] = 1

    # Fill empty hyperedge effects
    for e in range(E):
        while gamma[:, e].sum() < 1:
            for j in range(P):
                if gamma[j, :].sum() >= 1:
                    idx = gamma[j, :] == 1
                    idx_copy = idx.copy()
                    idx_copy[e] = True
                    E_O_mat, _ = E_Overlap(H[:, idx_copy])
                    triu_idx = np.triu_indices(idx_copy.sum(), k=1)
                    if len(triu_idx[0]) > 0 and E_O_mat[triu_idx].max() < 0.3:
                        gamma[j, e] = 1
                        break

    mu = np.zeros((P, E))
    nonzero_mask = gamma != 0
    mu[nonzero_mask] = mu_mean + mu_sd * rng.randn(nonzero_mask.sum())

    return gamma, mu


def compute_beta(H, gamma, mu):
    """Compute disease-specific risk-factor effects from hypergraph representation."""
    P, E = gamma.shape
    V = H.shape[0]

    Beta = np.zeros((P, V))
    for j in range(P):
        for e in range(E):
            if gamma[j, e]:
                Beta[j, :] += mu[j, e] * H[:, e]

    return Beta


def calibrate_intercept(eta, target_prev):
    """Calibrate intercept to achieve target disease prevalence."""
    def obj(a):
        return 1.0 / (1.0 + np.exp(-(a + eta))).mean() - target_prev

    a0 = np.log(target_prev / (1.0 - target_prev)) - eta.mean()

    result = root_scalar(obj, x0=a0, method='secant')
    return result.root


def simu_data_gen(N, P, V, E, mu_mean, mu_sd, seed,
                  nRarePerEdge, nCommonPerEdge, seed_H=None):
    """Generate synthetic data from a latent hypergraph model."""
    if seed_H is None:
        seed_H = 42

    rng = np.random.RandomState(seed)

    # Hypergraph generation
    rare_idx = np.arange(0, V // 4)
    common_idx = np.arange(V // 4, V)

    H = simulate_mixed_hypergraph(V, E, rare_idx, common_idx,
                                   nRarePerEdge, nCommonPerEdge, seed_H)
    gamma, mu = simulate_mechanisms(P, H, mu_mean, mu_sd)

    Beta = compute_beta(H, gamma, mu)
    dv_inv = 1.0 / np.sqrt(E)
    Beta = Beta * dv_inv

    X = rng.randn(N, P)
    xb = X @ Beta

    # Prevalence assignment
    prev0 = np.zeros(V)
    prev0[rare_idx] = 0.02 + 0.03 * rng.rand(len(rare_idx))
    prev0[common_idx] = 0.1 + 0.2 * rng.rand(len(common_idx))

    alpha = np.zeros(V)
    for v in range(V):
        alpha[v] = calibrate_intercept(xb[:, v], prev0[v])

    eta = X @ Beta + alpha
    prob = 1.0 / (1.0 + np.exp(-eta))
    Y = rng.binomial(1, prob)

    return X, Y, alpha, Beta, H, gamma, mu


def compute_repulsion_strength(gamma_prob, m_prob, z_prob):
    """Quantify redundancy induced by the repulsion prior."""
    P, E_edges = gamma_prob.shape
    E_O_mat, _ = E_Overlap(m_prob)

    gamma_prob_joint = gamma_prob * z_prob.reshape(1, -1)

    effect_hyperedge_per_predictor = gamma_prob_joint.sum(axis=1)

    repulsion_term = np.zeros(P)
    for e1 in range(E_edges):
        for e2 in range(e1 + 1, E_edges):
            repulsion_term += (E_O_mat[e1, e2] *
                               gamma_prob_joint[:, e1] *
                               gamma_prob_joint[:, e2])

    average_hyperedge_overlap = repulsion_term / (effect_hyperedge_per_predictor ** 2 + 1e-16)
    redundancy_ratio = repulsion_term / (effect_hyperedge_per_predictor + 1e-16)

    return effect_hyperedge_per_predictor, repulsion_term, average_hyperedge_overlap, redundancy_ratio


def cavi_initialization(seed_init, initial_method, E_hat, X_train, Y_train, true_params=None):
    """Initialize variational parameters.

    Args:
        seed_init: random seed
        initial_method: 'true', 'NNMF', or 'random'
        E_hat: upper bound on number of hyperedges
        X_train: (N, P) feature matrix
        Y_train: (N, V) outcome matrix
        true_params: dict with true H, mu, gamma, alpha (only for 'true' method)

    Returns:
        dict with initial variational parameters
    """
    P = X_train.shape[1]
    V = Y_train.shape[1]
    rng = np.random.RandomState(seed_init)

    if initial_method == 'true':
        H = true_params['H']
        mu = true_params['mu']
        gamma = true_params['gamma']
        alpha = true_params['alpha']
        E = H.shape[1]

        initials = {
            'z_prob': np.concatenate([np.ones(E), np.zeros(E_hat - E)]),
            'm_prob': np.concatenate([H, np.zeros((V, E_hat - E))], axis=1),
            'gamma_prob': np.concatenate([gamma, np.zeros((P, E_hat - E))], axis=1),
            'mu_mean': np.concatenate([mu, np.zeros((P, E_hat - E))], axis=1),
            'mu_var': np.ones((P, E_hat)),
            'alpha_mean': alpha.copy(),
            'alpha_var': np.ones(V),
        }
        return initials

    elif initial_method == 'NNMF':
        # Compute correlation matrix
        C = np.ma.corrcoef(np.ma.masked_invalid(Y_train.T), rowvar=True).data
        C = np.nan_to_num(C, nan=0.0)
        C[C < 0] = 0

        # NMF initialization
        nmf = NMF(n_components=E_hat, init='random', random_state=seed_init, max_iter=200)
        W = nmf.fit_transform(C)
        m_prob = W / W.max()

        # Scale to [0, 1] range with some noise
        threshold = np.quantile(m_prob, 0.6)
        m_prob = 0.8 * (m_prob > threshold).astype(float) + 0.2 * rng.rand(V, E_hat)

        z_prob = np.ones(E_hat)

        gamma_prob = np.zeros((P, E_hat))
        for j in range(P):
            for e in range(E_hat):
                cluster_mask = m_prob[:, e] > 0.5
                if cluster_mask.sum() > 0:
                    y_cluster = np.nanmean(Y_train[:, cluster_mask], axis=1)
                    valid = ~np.isnan(y_cluster) & ~np.isnan(X_train[:, j])
                    if valid.sum() > 1:
                        corr_val = np.corrcoef(X_train[valid, j], y_cluster[valid])[0, 1]
                        if np.isnan(corr_val):
                            corr_val = 0.5
                        gamma_prob[j, e] = np.abs(corr_val)
                    else:
                        gamma_prob[j, e] = 0.5
                else:
                    gamma_prob[j, e] = 0.5

        mu_mean = np.ones((P, E_hat))
        mu_var = np.ones((P, E_hat))

        prev_train = np.nanmean(Y_train, axis=0)
        eps0 = 1.0 / (2 * Y_train.shape[0])
        alpha_mean = np.log((prev_train + eps0) / (1.0 - prev_train + eps0))
        alpha_var = np.ones(V)

        initials = {
            'z_prob': z_prob,
            'm_prob': m_prob,
            'gamma_prob': gamma_prob,
            'mu_mean': mu_mean,
            'mu_var': mu_var,
            'alpha_mean': alpha_mean,
            'alpha_var': alpha_var,
        }
        return initials

    else:
        return None


def clip(x, lo, hi):
    """Clip values to [lo, hi]."""
    return np.clip(x, lo, hi)


def BHPI_single_iter(Kappa, X, X_sq, Xi_Xj,
                     r_ast, rho_ast, nu_ast, E_Omega, E_gamma,
                     E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma,
                     E_sigma2_inv, E_log_sigma2,
                     E_nu_logit, E_log_nu, E_log_1_minus_nu,
                     E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit,
                     a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r,
                     omega_repulsion, fix_z, z_constraint, update_m,
                     fix_gamma, freeze_gamma, tau_temper, sigma2_alpha,
                     batch_scale, weights):
    """Single iteration of BHPI CAVI update."""

    N, V = Kappa.shape
    N_x, P = X.shape
    V_rho, E_edges = rho_ast.shape
    dv_inv = 1.0 / np.sqrt(E_edges)
    P2 = P * (P - 1) // 2
    eps = 1e-16

    # --- 1. Update q(alpha_v) ---
    alpha_varrho = 1.0 / (batch_scale * E_Omega.sum(axis=0) + 1.0 / sigma2_alpha)

    E_s_b = np.transpose(r_ast.reshape(1, -1, 1) * E_s_b_given_z, (0, 2, 1))  # N x V x E
    E_eta = E_s_b.sum(axis=2)  # N x V
    alpha_ast = alpha_varrho * (Kappa - E_Omega * E_eta).sum(axis=0) * batch_scale

    # --- 2. Update q(mu | gamma) ---
    E_s_sq_given_z = dv_inv**2 * rho_ast  # V x E
    tau_ast = E_sigma2_inv + X_sq.T @ (batch_scale * E_Omega) @ E_s_sq_given_z  # P x E
    sigma2_ast = 1.0 / tau_ast

    E_s_given_z = dv_inv * rho_ast  # V x E
    E_eta_given_z = E_eta[:, :, np.newaxis] - E_s_b + np.transpose(E_s_b_given_z, (0, 2, 1))  # N x V x E
    E_mu_given_z = nu_ast * E_mu_given_gamma  # P x E

    # Vectorized B_ast computation (N x V x E x P)
    X_rs = X[:, np.newaxis, np.newaxis, :]  # N x 1 x 1 x P
    mu_rs = E_mu_given_z.T[np.newaxis, np.newaxis, :, :]  # 1 x 1 x E x P
    s_rs = E_s_given_z[np.newaxis, :, :, np.newaxis]  # 1 x V x E x 1
    E_x_s_mu_given_z = X_rs * mu_rs * s_rs  # N x V x E x P
    E_a_exclude_j_e = E_eta_given_z[:, :, :, np.newaxis] - E_x_s_mu_given_z  # N x V x E x P
    resid = Kappa[:, :, np.newaxis, np.newaxis] - E_Omega[:, :, np.newaxis, np.newaxis] * (
        alpha_ast[np.newaxis, :, np.newaxis, np.newaxis] + E_a_exclude_j_e)  # N x V x E x P
    B_ast = tau_temper * (X_rs * resid * batch_scale * s_rs).sum(axis=(0, 1)).T  # P x E

    mu_ast = sigma2_ast * B_ast
    E_mu_given_gamma = mu_ast
    E_mu_sq_given_gamma = mu_ast**2 + sigma2_ast

    # --- 3. Update logit for q(gamma | z) ---
    if fix_gamma:
        nu_ast = np.ones((P, E_edges))
    else:
        if not freeze_gamma:
            lik_gain = 0.5 * ((mu_ast**2) * tau_ast - np.log(tau_ast) - E_log_sigma2)

            if omega_repulsion > 0:
                # repulsion_logit_gamma(e1) = omega * sum_{e2} E_O[e1,e2] * E_gamma[:, e2]
                # E_O: (E, E), E_gamma: (P, E)
                repulsion_logit_gamma = omega_repulsion * (E_O @ E_gamma.T).T  # P x E
            else:
                repulsion_logit_gamma = 0.0

            logit_gamma = tau_temper * lik_gain + E_nu_logit - repulsion_logit_gamma
            nu_ast = 1.0 / (1.0 + np.exp(-logit_gamma))

    nu_ast = clip(nu_ast, eps, 1 - eps)
    E_gamma_given_z = nu_ast

    # --- 4. Update Memberships (m) ---
    E_b_given_z = X @ (E_gamma_given_z * E_mu_given_gamma)  # N x E
    E_s_b_given_z = dv_inv * E_b_given_z[:, :, np.newaxis] * rho_ast.T[np.newaxis, :, :]  # N x E x V
    E_s_b = np.transpose(r_ast.reshape(1, -1, 1) * E_s_b_given_z, (0, 2, 1))  # N x V x E
    E_a_exclude_e = E_s_b.sum(axis=2)[:, :, np.newaxis] - E_s_b  # N x V x E
    tmp = Kappa[:, :, np.newaxis] - E_Omega[:, :, np.newaxis] * (E_a_exclude_e + alpha_ast[np.newaxis, :, np.newaxis])
    E_zeta_given_z = (tmp * E_b_given_z[:, np.newaxis, :]).sum(axis=0)  # V x E

    E_mu_sq_given_z = nu_ast * E_mu_sq_given_gamma
    E_mu_given_z_val = nu_ast * mu_ast

    # Compute E_mu_j_mu_k_given_z
    E_mu_j_mu_k_given_z = np.zeros((P2, E_edges))
    idx = 0
    for j in range(P):
        for k in range(j + 1, P):
            E_mu_j_mu_k_given_z[idx, :] = E_mu_given_z_val[j, :] * E_mu_given_z_val[k, :]
            idx += 1

    E_b_sq_given_z = X_sq @ E_mu_sq_given_z + 2 * Xi_Xj @ E_mu_j_mu_k_given_z  # N x E
    E_xi_given_z = (E_Omega[:, :, np.newaxis] * E_b_sq_given_z[:, np.newaxis, :]).sum(axis=0)  # V x E

    residual = dv_inv * E_zeta_given_z - 0.5 * dv_inv**2 * E_xi_given_z

    # Repulsion term for m
    E_gamma_val = r_ast.reshape(1, -1) * E_gamma_given_z  # P x E

    if omega_repulsion > 0:
        # E_O_m_diff: (E, E, V), E_gamma_given_z: (P, E), E_gamma: (P, E)
        # repulsion_logit_m: sum over e2 of E_O_m_diff[e1, e2, :] * sum_p(gamma_given_z[p,e1] * E_gamma[p,e2])
        repulsion_logit_m = np.zeros((V, E_edges))
        for e1 in range(E_edges):
            for e2 in range(E_edges):
                if e2 == e1:
                    continue
                weight = (E_gamma_given_z[:, e1] * E_gamma_val[:, e2]).sum()
                repulsion_logit_m[:, e1] += E_O_m_diff[e1, e2, :] * weight
        repulsion_logit_m *= omega_repulsion
    else:
        repulsion_logit_m = 0.0

    if update_m:
        logit_m = E_rho_logit + tau_temper * (residual * batch_scale) - repulsion_logit_m
        rho_ast = 1.0 / (1.0 + np.exp(-logit_m))

    rho_ast = clip(rho_ast, eps, 1 - eps)

    # --- 5. Update Z (r) ---
    if update_m:
        if omega_repulsion > 0:
            E_O, E_O_m_diff = E_Overlap(rho_ast)
            repulsion_logit_z = np.zeros(E_edges)
            for e1 in range(E_edges):
                for e2 in range(E_edges):
                    if e2 == e1:
                        continue
                    repulsion_logit_z[e1] += E_O[e1, e2] * (E_gamma_given_z[:, e1] * E_gamma_val[:, e2]).sum()
            repulsion_logit_z *= omega_repulsion
        else:
            repulsion_logit_z = np.zeros(E_edges)

        # KL terms
        kl_term_z_rho = (rho_ast * (E_log_rho - np.log(rho_ast + eps)) +
                         (1 - rho_ast) * (E_log_1_minus_rho - np.log(1 - rho_ast + eps))).sum(axis=0)  # E

        kl_term_z_nu = (nu_ast * (E_log_nu - np.log(nu_ast + eps)) +
                        (1 - nu_ast) * (E_log_1_minus_nu - np.log(1 - nu_ast + eps))).sum(axis=0)  # E

        # Recompute residual for z
        E_s_b_given_z = dv_inv * E_b_given_z[:, :, np.newaxis] * rho_ast.T[np.newaxis, :, :]
        E_s_b = np.transpose(r_ast.reshape(1, -1, 1) * E_s_b_given_z, (0, 2, 1))
        E_a_exclude_e = E_s_b.sum(axis=2)[:, :, np.newaxis] - E_s_b
        tmp = Kappa[:, :, np.newaxis] - E_Omega[:, :, np.newaxis] * (E_a_exclude_e + alpha_ast[np.newaxis, :, np.newaxis])
        E_zeta_given_z = (tmp * E_b_given_z[:, np.newaxis, :]).sum(axis=0)
        residual = dv_inv * E_zeta_given_z - 0.5 * dv_inv**2 * E_xi_given_z

    if fix_z == 1:
        r_ast = np.ones(E_edges)
    elif fix_z == 0:
        logit_z = (rho_ast * (residual * batch_scale) * tau_temper).sum(axis=0) + \
                  E_r_logit + kl_term_z_rho + kl_term_z_nu - repulsion_logit_z
        r_ast = 1.0 / (1.0 + np.exp(-logit_z))

        if z_constraint > 0:
            idx = np.argsort(r_ast)[::-1]
            r_ast[idx[:z_constraint]] = 1.0

    r_ast = clip(r_ast, eps, 1 - eps)
    E_z = r_ast

    # --- 6. Update Polya-Gamma q(omega) ---
    E_s_b_given_z = dv_inv * E_b_given_z[:, :, np.newaxis] * rho_ast.T[np.newaxis, :, :]
    E_s_b = np.transpose(r_ast.reshape(1, -1, 1) * E_s_b_given_z, (0, 2, 1))
    E_eta = E_s_b.sum(axis=2)
    E_eta_tilde = E_eta + alpha_ast[np.newaxis, :]

    E_s_sq_b_sq = np.transpose(dv_inv**2 * E_b_sq_given_z[:, :, np.newaxis] *
                                r_ast.reshape(1, -1, 1) *
                                rho_ast.T[np.newaxis, :, :], (0, 2, 1))
    S = E_s_b.sum(axis=2)
    S2 = (E_s_b**2).sum(axis=2)
    E_s_b_e1e2_sum = 0.5 * (S**2 - S2)
    E_eta_sq = E_s_sq_b_sq.sum(axis=2) + E_s_b_e1e2_sum

    E_alpha_sq = alpha_ast**2 + alpha_varrho
    E_eta_tilde_sq = np.maximum(E_eta_sq + E_alpha_sq[np.newaxis, :] + 2 * alpha_ast[np.newaxis, :] * E_eta, eps)
    eta_ast = np.sqrt(E_eta_tilde_sq)
    E_Omega_new = 1.0 / (2 * eta_ast) * np.tanh(0.5 * eta_ast)

    # --- 7. Update hyperparameters ---
    E_gamma_val = E_z.reshape(1, -1) * E_gamma_given_z
    E_mu_sq = E_z.reshape(1, -1) * E_mu_sq_given_z
    a_mu_ast = a_mu + 0.5 * E_gamma_val.sum()
    b_mu_ast = b_mu + 0.5 * E_mu_sq.sum()
    E_sigma2_inv = a_mu_ast / b_mu_ast
    E_log_sigma2 = np.log(b_mu_ast) - digamma(a_mu_ast)

    # Update q(nu)
    a_nu_ast = a_nu + E_gamma_val
    b_nu_ast = b_nu + 1.0 - E_gamma_val
    E_nu_logit = digamma(a_nu_ast) - digamma(b_nu_ast)
    E_log_nu = digamma(a_nu_ast) - digamma(a_nu_ast + b_nu_ast)
    E_log_1_minus_nu = digamma(b_nu_ast) - digamma(a_nu_ast + b_nu_ast)

    # Update q(rho)
    E_m = E_z.reshape(1, -1) * rho_ast
    a_rho_ast = a_rho + E_m
    b_rho_ast = b_rho + 1.0 - E_m
    E_rho_logit = digamma(a_rho_ast) - digamma(b_rho_ast)
    E_log_rho = digamma(a_rho_ast) - digamma(a_rho_ast + b_rho_ast)
    E_log_1_minus_rho = digamma(b_rho_ast) - digamma(a_rho_ast + b_rho_ast)

    # Update q(r)
    a_r_ast = a_r + E_z
    b_r_ast = b_r + 1.0 - E_z
    E_r_logit = digamma(a_r_ast) - digamma(b_r_ast)
    E_log_r = digamma(a_r_ast) - digamma(a_r_ast + b_r_ast)
    E_log_1_minus_r = digamma(b_r_ast) - digamma(a_r_ast + b_r_ast)

    return (alpha_ast, alpha_varrho, mu_ast, sigma2_ast,
            nu_ast, rho_ast, r_ast, eta_ast,
            E_Omega_new, E_gamma_val, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma,
            a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq,
            a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast,
            E_nu_logit, E_log_nu, E_log_1_minus_nu,
            E_rho_logit, E_log_rho, E_log_1_minus_rho,
            E_r_logit, E_log_r, E_log_1_minus_r)


def robbins_monro(it, t0=10):
    """Robbins-Monro step size schedule."""
    return (it + t0) ** (-0.3)


def robbins_monro_update(x_old, x_new, it, t0=10):
    """Apply Robbins-Monro update."""
    rm = robbins_monro(it, t0)
    return (1 - rm) * x_old + rm * x_new


def BHPI_single_iter_wrapper(iter_count, X, Kappa, E_edges,
                              r_ast, rho_ast, nu_ast, mu_ast, sigma2_ast,
                              alpha_ast, alpha_varrho, Xi_Xj, X_sq,
                              E_mu_j_mu_k_given_z, E_Omega, E_gamma, E_O, E_O_m_diff,
                              E_mu_sq_given_z, E_eta_tilde, E_eta_tilde_sq,
                              E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2,
                              E_nu_logit, E_log_nu, E_log_1_minus_nu,
                              E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit,
                              a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r,
                              omega_repulsion, fix_z, z_constraint, update_m,
                              fix_gamma, freeze_gamma, tau_temper, sigma2_alpha,
                              batch_size, t0, weights):
    """Wrapper for single CAVI iteration with Robbins-Monro and ELBO."""

    N = X.shape[0]
    V_rho, E_edges_check = rho_ast.shape
    dv_inv = 1.0 / np.sqrt(E_edges)
    eps = 1e-16

    if batch_size > 0:
        batch_size = min(batch_size, N)
        batch_scale = N / batch_size
    else:
        batch_scale = 1.0

    # Minibatch sampling
    if batch_size > 0:
        idx_batch = np.random.choice(N, size=batch_size, replace=False)
    else:
        idx_batch = np.arange(N)

    mask_batch = Kappa[idx_batch] == 0

    # Initialize E_Omega if needed
    if isinstance(E_Omega, float) and np.isnan(E_Omega):
        E_b_given_z = X[idx_batch] @ (nu_ast * mu_ast)
        E_s_b_given_z = dv_inv * E_b_given_z[:, :, np.newaxis] * rho_ast.T[np.newaxis, :, :]
        E_s_b = np.transpose(r_ast.reshape(1, -1, 1) * E_s_b_given_z, (0, 2, 1))
        E_eta = E_s_b.sum(axis=2)
        E_eta_tilde = E_eta + alpha_ast[np.newaxis, :]

        E_mu_sq_given_z_val = nu_ast * (mu_ast**2 + sigma2_ast)
        E_b_sq_given_z = X_sq[idx_batch] @ E_mu_sq_given_z_val + 2 * Xi_Xj[idx_batch] @ E_mu_j_mu_k_given_z
        E_s_sq_b_sq = np.transpose(dv_inv**2 * E_b_sq_given_z[:, :, np.newaxis] *
                                    r_ast.reshape(1, -1, 1) *
                                    rho_ast.T[np.newaxis, :, :], (0, 2, 1))
        S = E_s_b.sum(axis=2)
        S2 = (E_s_b**2).sum(axis=2)
        E_s_b_e1e2_sum = 0.5 * (S**2 - S2)
        E_eta_sq = E_s_sq_b_sq.sum(axis=2) + E_s_b_e1e2_sum

        E_alpha_sq = alpha_ast**2 + alpha_varrho
        E_eta_tilde_sq = E_eta_sq + E_alpha_sq[np.newaxis, :] + 2 * alpha_ast[np.newaxis, :] * E_eta
        eta_ast_val = np.sqrt(E_eta_tilde_sq + eps)
        E_Omega = 1.0 / (2 * eta_ast_val) * np.tanh(0.5 * eta_ast_val)
        E_Omega[mask_batch] = 0

    # Run single iteration
    (alpha_ast_new, alpha_varrho_new, mu_ast_new, sigma2_ast_new,
     nu_ast_new, rho_ast_new, r_ast_new, _,
     E_Omega_new, E_gamma_new, E_O_new, E_O_m_diff_new,
     E_s_b_given_z_new, E_mu_given_gamma_new,
     a_mu_ast, b_mu_ast, E_sigma2_inv_new, E_log_sigma2_new,
     E_eta_tilde_new, E_eta_tilde_sq_new,
     a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast,
     E_nu_logit_new, E_log_nu_new, E_log_1_minus_nu_new,
     E_rho_logit_new, E_log_rho_new, E_log_1_minus_rho_new,
     E_r_logit_new, E_log_r, E_log_1_minus_r) = \
        BHPI_single_iter(Kappa[idx_batch], X[idx_batch], X_sq[idx_batch], Xi_Xj[idx_batch],
                          r_ast, rho_ast, nu_ast, E_Omega, E_gamma,
                          E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma,
                          E_sigma2_inv, E_log_sigma2,
                          E_nu_logit, E_log_nu, E_log_1_minus_nu,
                          E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit,
                          a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r,
                          omega_repulsion, fix_z, z_constraint, update_m,
                          fix_gamma, freeze_gamma, tau_temper, sigma2_alpha,
                          batch_scale, weights)

    # Robbins-Monro updates
    alpha_ast = robbins_monro_update(alpha_ast, alpha_ast_new, iter_count, t0)
    alpha_varrho = robbins_monro_update(alpha_varrho, alpha_varrho_new, iter_count, t0)
    mu_ast = robbins_monro_update(mu_ast, mu_ast_new, iter_count, t0)
    sigma2_ast = robbins_monro_update(sigma2_ast, sigma2_ast_new, iter_count, t0)
    nu_ast = robbins_monro_update(nu_ast, nu_ast_new, iter_count, t0)
    rho_ast = robbins_monro_update(rho_ast, rho_ast_new, iter_count, t0)
    r_ast = robbins_monro_update(r_ast, r_ast_new, iter_count, t0)
    E_Omega = robbins_monro_update(E_Omega, E_Omega_new, iter_count, t0)
    E_gamma = robbins_monro_update(E_gamma, E_gamma_new, iter_count, t0)
    E_O = robbins_monro_update(E_O, E_O_new, iter_count, t0)
    E_O_m_diff = robbins_monro_update(E_O_m_diff, E_O_m_diff_new, iter_count, t0)
    E_s_b_given_z = robbins_monro_update(E_s_b_given_z, E_s_b_given_z_new, iter_count, t0)
    E_mu_given_gamma = robbins_monro_update(E_mu_given_gamma, E_mu_given_gamma_new, iter_count, t0)
    E_sigma2_inv = robbins_monro_update(E_sigma2_inv, E_sigma2_inv_new, iter_count, t0)
    E_log_sigma2 = robbins_monro_update(E_log_sigma2, E_log_sigma2_new, iter_count, t0)
    E_eta_tilde = robbins_monro_update(E_eta_tilde, E_eta_tilde_new, iter_count, t0)
    E_eta_tilde_sq = robbins_monro_update(E_eta_tilde_sq, E_eta_tilde_sq_new, iter_count, t0)
    E_nu_logit = robbins_monro_update(E_nu_logit, E_nu_logit_new, iter_count, t0)
    E_log_nu = robbins_monro_update(E_log_nu, E_log_nu_new, iter_count, t0)
    E_log_1_minus_nu = robbins_monro_update(E_log_1_minus_nu, E_log_1_minus_nu_new, iter_count, t0)
    E_rho_logit = robbins_monro_update(E_rho_logit, E_rho_logit_new, iter_count, t0)
    E_log_rho = robbins_monro_update(E_log_rho, E_log_rho_new, iter_count, t0)
    E_log_1_minus_rho = robbins_monro_update(E_log_1_minus_rho, E_log_1_minus_rho_new, iter_count, t0)
    E_r_logit = robbins_monro_update(E_r_logit, E_r_logit_new, iter_count, t0)

    # ELBO computation
    lik = np.sum((Kappa[idx_batch] * E_eta_tilde - 0.5 * E_eta_tilde_sq * E_Omega) * weights) * batch_scale

    L_alpha = np.sum(np.log(alpha_varrho) - (alpha_ast**2 + alpha_varrho) / sigma2_alpha) / 2

    E_z = r_ast
    L_mu = np.sum(E_z.reshape(1, -1) * nu_ast *
                  (np.log(sigma2_ast) - E_log_sigma2 -
                   (mu_ast**2 + sigma2_ast) * E_sigma2_inv + 1)) / 2

    repulsion_term = 0.0
    if omega_repulsion > 0:
        for e1 in range(E_edges):
            for e2 in range(e1 + 1, E_edges):
                repulsion_term += E_O[e1, e2] * r_ast[e1] * r_ast[e2] * \
                    np.sum(nu_ast[:, e1] * nu_ast[:, e2])
        repulsion_term = -omega_repulsion * repulsion_term

    kl_term_z_nu = np.sum(nu_ast * (E_log_nu - np.log(nu_ast + eps)) +
                          (1 - nu_ast) * (E_log_1_minus_nu - np.log(1 - nu_ast + eps)), axis=0)
    L_gamma = kl_term_z_nu @ r_ast + repulsion_term

    kl_term_z_rho = np.sum(rho_ast * (E_log_rho - np.log(rho_ast + eps)) +
                           (1 - rho_ast) * (E_log_1_minus_rho - np.log(1 - rho_ast + eps)), axis=0)
    L_m = kl_term_z_rho @ r_ast

    if fix_z:
        L_z = 0.0
    else:
        L_z = r_ast @ (E_log_r - np.log(r_ast + eps)) + \
              (1 - r_ast) @ (E_log_1_minus_r - np.log(1 - r_ast + eps))

    L_hyper = (np.sum((a_nu - a_nu_ast) * E_log_nu + (b_nu - b_nu_ast) * E_log_1_minus_nu +
                      betaln(a_nu_ast, b_nu_ast)) +
               np.sum((a_rho - a_rho_ast) * E_log_rho + (b_rho - b_rho_ast) * E_log_1_minus_rho +
                      betaln(a_rho_ast, b_rho_ast)) +
               np.sum((a_r - a_r_ast) * E_log_r + (b_r - b_r_ast) * E_log_1_minus_r +
                      betaln(a_r_ast, b_r_ast)) +
               (a_mu_ast - a_mu) * E_log_sigma2 + (b_mu_ast - b_mu) * E_sigma2_inv -
               a_mu_ast * np.log(b_mu_ast) + np.log(np.math.gamma(a_mu_ast)))

    ELBO = lik + L_mu + L_gamma + L_m + L_z + L_hyper + L_alpha

    return (alpha_ast, alpha_varrho, mu_ast, sigma2_ast, nu_ast, rho_ast, r_ast,
            E_Omega, E_gamma, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma,
            a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq,
            a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast,
            E_nu_logit, E_log_nu, E_log_1_minus_nu,
            E_rho_logit, E_log_rho, E_log_1_minus_rho,
            E_r_logit, E_log_r, E_log_1_minus_r, ELBO)


def BHPI(X, Y, E_edges, max_iter, seed, initials, omega_repulsion,
         staged, final_fix_z, final_z_constraint, sigma2_alpha,
         warmup_iters, batch_size, t0, weights, tol, verbose):
    """Main BHPI training function.

    Args:
        X: (N, P) feature matrix
        Y: (N, V) outcome matrix (NaN for missing)
        E_edges: upper bound on number of hyperedges
        max_iter: maximum CAVI iterations
        seed: random seed
        initials: dict with initial variational parameters
        omega_repulsion: repulsion strength
        staged: whether to use staged warmup
        final_fix_z: whether to fix z=1
        final_z_constraint: minimum number of active hyperedges
        sigma2_alpha: prior variance for alpha
        warmup_iters: warmup iterations per stage
        batch_size: minibatch size (0 = full batch)
        t0: Robbins-Monro offset
        weights: sample weights
        tol: convergence tolerance
        verbose: print progress

    Returns:
        model dict with learned parameters
    """
    import time
    start_time = time.time()

    N, P = X.shape
    V = Y.shape[1]
    dv_inv = 1.0 / np.sqrt(E_edges)
    tau_temper = 1.0

    X_sq = X**2
    MASK = np.isnan(Y)
    Kappa = Y - 0.5
    Kappa[MASK] = 0.0

    eps = 1e-16

    if batch_size > 0:
        batch_size = min(batch_size, N)

    rng = np.random.RandomState(seed)

    # Initialize variational parameters
    if initials is not None:
        r_ast = initials['z_prob'].copy()
        rho_ast = initials['m_prob'].copy()
        nu_ast = initials['gamma_prob'].copy()
        mu_ast = initials['mu_mean'].copy()
        sigma2_ast = initials['mu_var'].copy()
        alpha_ast = initials['alpha_mean'].copy()
        alpha_varrho = initials['alpha_var'].copy()
    else:
        r_ast = rng.uniform(0.5, 0.8, E_edges)
        rho_ast = rng.uniform(0.5, 0.8, (V, E_edges))
        nu_ast = rng.uniform(0.5, 0.8, (P, E_edges))
        sigma2_ast = np.ones((P, E_edges)) * 100
        mu_ast = rng.randn(P, E_edges) * np.sqrt(sigma2_ast)
        alpha_ast = np.zeros(V)
        alpha_varrho = np.ones(V) * 100

    nu_ast = clip(nu_ast, eps, 1 - eps)
    rho_ast = clip(rho_ast, eps, 1 - eps)
    r_ast = clip(r_ast, eps, 1 - eps)

    # Hyperparameters
    a_mu = 0.5
    b_mu = 0.5
    a_r = 0.5
    b_r = 0.5
    a_rho = 0.5
    b_rho = 0.5
    a_nu = 0.5
    b_nu = 0.5

    # Precompute P2 indices
    P2 = P * (P - 1) // 2
    Xi_Xj = np.zeros((N, P2))
    idx = 0
    for j in range(P):
        for k in range(j + 1, P):
            Xi_Xj[:, idx] = X[:, j] * X[:, k]
            idx += 1

    # --- Precompute expectations ---
    E_gamma_given_z = nu_ast
    E_z = r_ast
    E_gamma = E_z.reshape(1, -1) * E_gamma_given_z
    E_mu_given_gamma = mu_ast
    E_mu_sq_given_gamma = mu_ast**2 + sigma2_ast
    E_mu_sq_given_z = E_gamma_given_z * E_mu_sq_given_gamma

    E_mu_j_mu_k_given_z = np.zeros((P2, E_edges))
    idx = 0
    for j in range(P):
        for k in range(j + 1, P):
            E_mu_j_mu_k_given_z[idx, :] = nu_ast[j, :] * mu_ast[j, :] * nu_ast[k, :] * mu_ast[k, :]
            idx += 1

    E_O, E_O_m_diff = E_Overlap(rho_ast)

    a_mu_ast = a_mu + 0.5 * E_gamma.sum()
    E_mu_sq = E_z.reshape(1, -1) * E_mu_sq_given_z
    b_mu_ast = b_mu + 0.5 * E_mu_sq.sum()
    E_log_sigma2 = np.log(b_mu_ast) - digamma(a_mu_ast)
    E_sigma2_inv = a_mu_ast / b_mu_ast

    a_nu_ast = a_nu + E_gamma
    b_nu_ast = b_nu + 1.0 - E_gamma
    E_nu_logit = digamma(a_nu_ast) - digamma(b_nu_ast)
    E_log_nu = digamma(a_nu_ast) - digamma(a_nu_ast + b_nu_ast)
    E_log_1_minus_nu = digamma(b_nu_ast) - digamma(a_nu_ast + b_nu_ast)

    E_m = E_z.reshape(1, -1) * rho_ast
    a_rho_ast = a_rho + E_m
    b_rho_ast = b_rho + 1.0 - E_m
    E_rho_logit = digamma(a_rho_ast) - digamma(b_rho_ast)
    E_log_rho = digamma(a_rho_ast) - digamma(a_rho_ast + b_rho_ast)
    E_log_1_minus_rho = digamma(b_rho_ast) - digamma(a_rho_ast + b_rho_ast)

    a_r_ast = a_r + E_z
    b_r_ast = b_r + 1.0 - E_z
    E_r_logit = digamma(a_r_ast) - digamma(b_r_ast)

    E_Omega = np.nan
    E_s_b_given_z = np.nan
    E_eta_tilde = np.nan
    E_eta_tilde_sq = np.nan

    # Staged warmup
    if staged:
        # Stage 0: Fix z=1, freeze m, fix gamma=1, update mu
        fix_z = True
        z_constraint = 0
        update_m = False
        fix_gamma = True
        freeze_gamma = False
        if verbose:
            print('Stage 0: Fix z=1, freeze m, fix gamma=1, update mu (Warm-up)')

        for it in range(1, warmup_iters + 1):
            (alpha_ast, alpha_varrho, mu_ast, sigma2_ast, nu_ast, rho_ast, r_ast,
             E_Omega, E_gamma, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma,
             a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq,
             a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast,
             E_nu_logit, E_log_nu, E_log_1_minus_nu,
             E_rho_logit, E_log_rho, E_log_1_minus_rho,
             E_r_logit, _, _, _) = \
                BHPI_single_iter_wrapper(it, X, Kappa, E_edges,
                                          r_ast, rho_ast, nu_ast, mu_ast, sigma2_ast,
                                          alpha_ast, alpha_varrho, Xi_Xj, X_sq,
                                          E_mu_j_mu_k_given_z, E_Omega, E_gamma, E_O, E_O_m_diff,
                                          E_mu_sq_given_z, E_eta_tilde, E_eta_tilde_sq,
                                          E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2,
                                          E_nu_logit, E_log_nu, E_log_1_minus_nu,
                                          E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit,
                                          a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r,
                                          omega_repulsion, fix_z, z_constraint, update_m,
                                          fix_gamma, freeze_gamma, tau_temper, sigma2_alpha,
                                          batch_size, t0, weights)

        if verbose:
            print(f'sum(r) = {r_ast.sum():.3f}, mean(rho) = {rho_ast.mean():.3f}, mean(nu) = {nu_ast.mean():.3f}')

        # Stage 1: Fix z=1, freeze m, update gamma & mu
        fix_z = True
        z_constraint = 0
        update_m = False
        fix_gamma = False
        freeze_gamma = False
        nu_ast = clip(initials['gamma_prob'], 0.5, 0.99)

        if verbose:
            print('Stage 1: Fix z=1, freeze m, update gamma & mu')

        for it in range(1, warmup_iters + 1):
            (alpha_ast, alpha_varrho, mu_ast, sigma2_ast, nu_ast, rho_ast, r_ast,
             E_Omega, E_gamma, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma,
             a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq,
             a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast,
             E_nu_logit, E_log_nu, E_log_1_minus_nu,
             E_rho_logit, E_log_rho, E_log_1_minus_rho,
             E_r_logit, _, _, _) = \
                BHPI_single_iter_wrapper(it, X, Kappa, E_edges,
                                          r_ast, rho_ast, nu_ast, mu_ast, sigma2_ast,
                                          alpha_ast, alpha_varrho, Xi_Xj, X_sq,
                                          E_mu_j_mu_k_given_z, E_Omega, E_gamma, E_O, E_O_m_diff,
                                          E_mu_sq_given_z, E_eta_tilde, E_eta_tilde_sq,
                                          E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2,
                                          E_nu_logit, E_log_nu, E_log_1_minus_nu,
                                          E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit,
                                          a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r,
                                          omega_repulsion, fix_z, z_constraint, update_m,
                                          fix_gamma, freeze_gamma, tau_temper, sigma2_alpha,
                                          batch_size, t0, weights)

        if verbose:
            print(f'sum(r) = {r_ast.sum():.3f}, mean(rho) = {rho_ast.mean():.3f}, mean(nu) = {nu_ast.mean():.3f}')

        # Stage 2b: Fix z=1, update m, gamma & mu
        fix_z = True
        z_constraint = 0
        update_m = True
        fix_gamma = False
        freeze_gamma = False

        if verbose:
            print('Stage 2b: Fix z=1, update m, gamma & mu')

        for it in range(1, warmup_iters + 1):
            (alpha_ast, alpha_varrho, mu_ast, sigma2_ast, nu_ast, rho_ast, r_ast,
             E_Omega, E_gamma, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma,
             a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq,
             a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast,
             E_nu_logit, E_log_nu, E_log_1_minus_nu,
             E_rho_logit, E_log_rho, E_log_1_minus_rho,
             E_r_logit, _, _, _) = \
                BHPI_single_iter_wrapper(it, X, Kappa, E_edges,
                                          r_ast, rho_ast, nu_ast, mu_ast, sigma2_ast,
                                          alpha_ast, alpha_varrho, Xi_Xj, X_sq,
                                          E_mu_j_mu_k_given_z, E_Omega, E_gamma, E_O, E_O_m_diff,
                                          E_mu_sq_given_z, E_eta_tilde, E_eta_tilde_sq,
                                          E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2,
                                          E_nu_logit, E_log_nu, E_log_1_minus_nu,
                                          E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit,
                                          a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r,
                                          omega_repulsion, fix_z, z_constraint, update_m,
                                          fix_gamma, freeze_gamma, tau_temper, sigma2_alpha,
                                          batch_size, t0, weights)

        if verbose:
            print(f'sum(r) = {r_ast.sum():.3f}, mean(rho) = {rho_ast.mean():.3f}, mean(nu) = {nu_ast.mean():.3f}')

    # Stage 3: Update all
    E_m = E_z.reshape(1, -1) * rho_ast
    beta_old = dv_inv * (nu_ast * E_mu_given_gamma) @ E_m.T

    if staged and verbose:
        print('Stage 3: Update all with z_constraint')

    update_m = True
    fix_gamma = False
    freeze_gamma = False
    fix_z = final_fix_z
    z_constraint = final_z_constraint

    ELBO_trace = []
    expected_log_lik_trace = []

    for it in range(1, max_iter + 1):
        (alpha_ast, alpha_varrho, mu_ast, sigma2_ast, nu_ast, rho_ast, r_ast,
         E_Omega, E_gamma, E_O, E_O_m_diff, E_s_b_given_z, E_mu_given_gamma,
         a_mu_ast, b_mu_ast, E_sigma2_inv, E_log_sigma2, E_eta_tilde, E_eta_tilde_sq,
         a_nu_ast, b_nu_ast, a_rho_ast, b_rho_ast, a_r_ast, b_r_ast,
         E_nu_logit, E_log_nu, E_log_1_minus_nu,
         E_rho_logit, E_log_rho, E_log_1_minus_rho,
         E_r_logit, _, _, ELBO_val) = \
            BHPI_single_iter_wrapper(it, X, Kappa, E_edges,
                                      r_ast, rho_ast, nu_ast, mu_ast, sigma2_ast,
                                      alpha_ast, alpha_varrho, Xi_Xj, X_sq,
                                      E_mu_j_mu_k_given_z, E_Omega, E_gamma, E_O, E_O_m_diff,
                                      E_mu_sq_given_z, E_eta_tilde, E_eta_tilde_sq,
                                      E_s_b_given_z, E_mu_given_gamma, E_sigma2_inv, E_log_sigma2,
                                      E_nu_logit, E_log_nu, E_log_1_minus_nu,
                                      E_rho_logit, E_log_rho, E_log_1_minus_rho, E_r_logit,
                                      a_mu, b_mu, a_nu, b_nu, a_rho, b_rho, a_r, b_r,
                                      omega_repulsion, fix_z, z_constraint, update_m,
                                      fix_gamma, freeze_gamma, tau_temper, sigma2_alpha,
                                      batch_size, t0, weights)

        yhat = 1.0 / (1.0 + np.exp(-E_eta_tilde))
        exp_log_lik = np.nanmean(Y * np.log(yhat + eps) + (1 - Y) * np.log(1 - yhat + eps))

        ELBO_trace.append(ELBO_val)
        expected_log_lik_trace.append(exp_log_lik)

        # Convergence check
        E_m = E_z.reshape(1, -1) * rho_ast
        beta_new = dv_inv * (nu_ast * E_mu_given_gamma) @ E_m.T
        E_beta_diff = np.abs(beta_new - beta_old).max()

        if it > 1:
            loglik_diff = expected_log_lik_trace[-1] - expected_log_lik_trace[-2]
        else:
            loglik_diff = -1000

        if verbose and it % 10 == 0:
            elapsed = time.time() - start_time
            print(f'Iteration {it}: ELBO={ELBO_val:.4g}, Exp_lik={exp_log_lik:.4g}, '
                  f'diff of E[beta]: {E_beta_diff:.6g}, time={elapsed:.1f}s')

        if E_beta_diff < tol and loglik_diff < tol:
            if verbose:
                print(f'Converged at iteration {it}, ELBO: {ELBO_val:.4g}, '
                      f'Exp_lik={exp_log_lik:.4g}, diff of E[beta]: {E_beta_diff:.6g}')
                elapsed = time.time() - start_time
                print(f'Total time: {elapsed:.1f}s')
            break
        else:
            beta_old = beta_new

    # Return model
    model = {
        'ELBO': ELBO_trace,
        'expected_log_lik': expected_log_lik_trace,
        'mu_mean': mu_ast,
        'mu_var': sigma2_ast,
        'gamma_prob': nu_ast,
        'm_prob': rho_ast,
        'z_prob': r_ast,
        'beta': beta_new,
        'alpha_mean': alpha_ast,
        'alpha_var': alpha_varrho,
        'sigma2_a': a_mu_ast,
        'sigma2_b': b_mu_ast,
        'rho_a': a_rho_ast,
        'rho_b': b_rho_ast,
        'nu_a': a_nu_ast,
        'nu_b': b_nu_ast,
        'r_a': a_r_ast,
        'r_b': b_r_ast,
        'convergence': E_beta_diff < tol,
    }

    return model
