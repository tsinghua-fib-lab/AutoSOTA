"""Hidden Markov model with Torus Graph emission distributions.

This module implements an EM algorithm for HMMs where each state's
emission is a Torus Graph model.  It also provides the supporting
forward/backward routines and utilities for initialisation.
"""
__date__ = "April - September 2025"

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import optax
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

from .sample import torus_graph_unnorm_log_density
from .ssm import estimate_params_ssm
from .simulate_hmm import (
    HMMPriorParams,
    sample_sticky_transition_matrix,
    sample_phis,
)
from .stats import solve_tg_exact



batch_energy = jax.vmap(torus_graph_unnorm_log_density, in_axes=(None, 0), out_axes=0)
batch_tg_solve = jax.vmap(solve_tg_exact, in_axes=(None,0,None), out_axes=0)



def compute_scores(x, phis, log_partitions):
    """
    Computes the (unnormalized) log probabilities for each state,
    where the first state's log partition function is fixed to zero.
    
    Args:
        x: Single observation.
        phis: Array of shape (K,d,d,2)
        log_partitions: Array of shape (K,) containing the log partition parameters
            for states 1...K.
        
    Returns:
        A vector of scores of shape (K,).
    """
    assert x.ndim == 1
    assert phis.ndim == 4
    scores = batch_energy(x, phis) # [K]
    scores = scores - log_partitions
    return scores

# Vectorize compute_scores over a batch of observations.
batched_compute_scores = jax.vmap(compute_scores, in_axes=(0, None, None), out_axes=0)


def cross_entropy_loss_fn(log_partitions, xs, gammas, phis, λ=1e-1):
    """
    Computes the cross-entropy loss between the softmax predictions
    (derived from the energy scores) and the target soft assignments.
    
    Args:
        log_partitions: Array of shape (K,) of log partition functions for states 1...K.
        xs: Batch of observations, shape (batch_size, dim).
        gammas: Soft assignments (target probabilities), shape (batch_size, K).
        phis : Array of shape (K,d,d,2)
        
    Returns:
        Scalar loss value.
    """
    # Compute scores for each observation.
    scores = batched_compute_scores(xs, phis, log_partitions)  # shape (batch_size, K)
    # Compute log probabilities via softmax.
    log_probs = scores - logsumexp(scores, axis=1, keepdims=True)
    # Cross-entropy loss (averaged over the batch).
    loss = - jnp.mean(jnp.sum(gammas * log_probs, axis=1))
    # Gaussian prior on A_i.
    loss = loss + λ * jnp.sum(log_partitions**2)
    return loss


def forward_backward(
        log_emissions: jnp.ndarray,
        log_trans: jnp.ndarray,
        log_pi: jnp.ndarray,
    ):
    """
    Run the log-space forward-backward algorithm.

    Args:
        log_emissions: shape (T, K), where log_emissions[t, i] = log p(x_t | z_t=i)
        log_trans:     shape (K, K), where log_trans[i, j] = log p(z_{t+1}=j | z_t=i)
        log_pi:        shape (K,),  where log_pi[i] = log p(z_0 = i)

    Returns:
        gamma:            shape (T, K), posterior p(z_t=i | x_1:T)
        xi:               shape (T-1, K, K), posterior p(z_t=i, z_{t+1}=j | x_1:T)
        log_likelihood:  scalar, log p(x_1:T)
    """
    T, K = log_emissions.shape

    # ——— Forward pass ———
    def _fwd_step(log_alpha_prev, log_obs_t):
        # log_alpha_prev: (K,)
        # log_obs_t:       (K,)
        # output log_alpha_t[j] = logsum_i [log_alpha_prev[i] + log_trans[i, j]] + log_obs_t[j]
        log_alpha_t = logsumexp(log_alpha_prev[:, None] + log_trans, axis=0) + log_obs_t
        return log_alpha_t, log_alpha_t

    # alpha_0
    log_alpha0 = log_pi + log_emissions[0]             # (K,)
    # scan for t=1..T-1
    _, alphas_tail = jax.lax.scan(
        _fwd_step,
        log_alpha0,
        log_emissions[1:],
    )  # returns (carry, stacked_outputs)
    log_alphas = jnp.vstack([log_alpha0, alphas_tail])  # (T, K)

    # ——— Backward pass ———
    def _bwd_step(log_beta_next, log_obs_next):
        # compute beta_t given beta_{t+1}
        # shape(log_trans) = (K,K), shape(log_obs_next+log_beta_next) = (K,)
        log_beta_t = logsumexp(
            log_trans + (log_obs_next + log_beta_next)[None, :],
            axis=1
        )
        return log_beta_t, log_beta_t

    log_beta_T = jnp.zeros((K,))                      # beta at time T-1
    # we only need log_emissions[1:], reversed in time
    rev_obs = log_emissions[1:][::-1]                 # shape (T-1, K)
    _, betas_rev = jax.lax.scan(_bwd_step, log_beta_T, rev_obs)
    # betas_rev[t] = beta at time T-2-t, for t=0..T-2

    betas = betas_rev[::-1]                           # now betas[t] = beta at time t, for t=0..T-2
    log_betas = jnp.concatenate(
        [betas, log_beta_T[None, :]],                # append beta at T-1
        axis=0
    )  # shape (T, K)

    # ——— Compute gamma = posterior marginals ———
    log_gamma = log_alphas + log_betas                  # shape (T,K)
    # normalize in log-space
    log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = jnp.exp(log_gamma)

    # ——— Compute xi = posterior pairwise marginals ———
    def _compute_xi(log_alpha_t, log_obs_next, log_beta_next):
        # unnormalized log xi matrix:
        #   log_alpha_t[i] + log_trans[i,j] + log_obs_next[j] + log_beta_next[j]
        log_xi = (log_alpha_t[:, None]
                  + log_trans
                  + log_obs_next[None, :]
                  + log_beta_next[None, :])
        # normalize:
        return jnp.exp(log_xi - logsumexp(log_xi))

    xi = jax.vmap(_compute_xi)(
        log_alphas[:-1],
        log_emissions[1:],
        log_betas[1:]
    )  # shape (T-1, K, K)

    # ——— Log-likelihood ———
    log_likelihood = logsumexp(log_alphas[-1])

    return gamma, xi, log_likelihood


def fit_hmm_em(
        xs: jnp.ndarray,
        K: int,
        prior: HMMPriorParams,
        warmup_iterations: int = 5,
        tau_initial: float = 3.0,
        num_em_iters: int = 10,
        num_part_opt_steps: int = 2000,
        lr: float = 1e-2,
        seed: int = 0,
        fit_initial_conditions: bool = False,
        fit_transition_matrix: bool = True,
        cross_entropy_lambda : float = 1e-2,
        tg_solve_reg : float = 1e-1,
        phi_solve_mode: str = "ssm",
        beta: float = 1.0,
        beta_min: float = 1.0,
        beta_annealing: str = None,
        init_method: str = "zscore",
        n_init: int = 100,
        phi_l2_reg: float = 0.1,
        phi_l1_reg: float = 0.0,
        num_trials: int = None,
    ):
    """
    Fits an HMM via EM, using HMMPriorParams for both initialization
    and transition regularization, plus a temperature anneal.
    """
    if num_trials is not None:
        N = num_trials
        T_seq = xs.shape[1]
        d     = xs.shape[2]
        xs    = xs.reshape(N * T_seq, d)
        T_total = N * T_seq
    else:
        N = 1
        T_seq = xs.shape[0]
        d     = xs.shape[1]
        T_total = T_seq
    key = jax.random.PRNGKey(seed)

    result = None

    # ——— Initialize π uniformly ———
    log_pi = jnp.log(jnp.ones((K,)) / K)

    # ——— Initialize a uniform transition matrix ———
    A0 = jnp.ones((K,K)) / K
    log_trans = jnp.log(A0 + 1e-12)

    # ——— Initialize log‑partition (state 0 fixed at 0) ———
    log_part = jnp.zeros((K,))
    gamma = None
    
    if init_method == "zscore":
        # ——— Initialize φ via its precision hyperparams ———

        # Fit one base phi.
        key, subkey = jax.random.split(key)
        base_phi, _ = estimate_params_ssm(
            subkey,
            xs,
            phi=None,
            batch_size=512,
            n_iter=2000,
            no_improvement=200,
            alpha=0.99,
            opt_state=None,
            l2_reg=phi_l2_reg,
            l1_reg=phi_l1_reg,
            replace=True,
            lr=1e-2,
        ) # [d,d,2]

        assignments = []
        for n in range(n_init):
            key, subkey = jax.random.split(key)
            phis = sample_phis(
                subkey,
                K,
                d,
                prior.phi_prec_tril,
                prior.phi_prec_diag,
                prior.phi_prec_triu,
            )
            phis = base_phi[None] + 1e-4 * phis
            scores = batched_compute_scores(xs, phis, log_part)
            # scores = scores - scores.mean(axis=0)
            argmax = jnp.argmax(scores, axis=1)
            assignments.append(argmax)

        assignments = jnp.array(assignments)
        similarity = []
        for n in range(len(assignments[0])):
            shared = assignments.T == assignments.T[n]
            similarity.append(shared.sum(axis=1) / len(assignments))
        similarity = jnp.array(similarity)

        clusters = SpectralClustering(
            n_clusters=K,
            assign_labels="discretize",
            random_state=42,
            affinity="precomputed",
        ).fit(similarity) # TODO: get random state from JAX PRNGKey

        # Make fake gammas from clustering.
        gamma = jnp.zeros((T_total,K))
        for i in range(K):
            for idx in jnp.argwhere(clusters.labels_ == i).flatten():
                gamma = gamma.at[idx,i].set(1.0)
        print("Marginal state occupancy:", jnp.mean(gamma, axis=0))

        phis = []
        for i in range(K):
            data = xs[jnp.argwhere(clusters.labels_ == i)]
            print(f"Fitting state {i+1} with {len(data)} timepoints...")
            key, subkey = jax.random.split(key)
            phi_hat, _ = estimate_params_ssm(
                    subkey,
                    data.reshape(len(data), len(data[0,0])),
                    phi=base_phi.copy(),
                    batch_size=512,
                    n_iter=2000,
                    no_improvement=200,
                    alpha=0.99,
                    opt_state=None,
                    l2_reg=phi_l2_reg,
                    l1_reg=phi_l1_reg,
                    replace=True,
                    lr=1e-2,
                ) # [d,d,2]
            phis.append(phi_hat)
        phis = jnp.array(phis)
  
    else:
        # Randomly initialize the phis.
        key, subkey = jax.random.split(key)
        phis = sample_phis(
            subkey,
            K,
            d,
            prior.phi_prec_tril,
            prior.phi_prec_diag,
            prior.phi_prec_triu,
        )

    # ——— Optimizer for log‑partitions ———
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(log_part)

    # ——— pre-EM fit log partition ———
    if gamma is not None:
        
        @jax.jit
        def part_step(lp, opt_st):
            loss, grads = jax.value_and_grad(cross_entropy_loss_fn)(
                lp, xs, gamma, phis, cross_entropy_lambda
            )
            updates, new_opt = optimizer.update(grads, opt_st)
            return optax.apply_updates(lp, updates), new_opt, loss

        pbar = tqdm(range(num_part_opt_steps))
        pbar.set_description(f"Log partition loss: {jnp.nan}")
        for i in pbar:
            log_part, opt_state, part_loss = part_step(log_part, opt_state)
            if i == 0:  
                smooth_loss = part_loss
            else:
                smooth_loss = 0.99 * smooth_loss + 0.01 * part_loss
            pbar.set_description(f"Log partition loss: {part_loss:.2f}")

        print("log_part:", log_part)
    else:
        gamma = jnp.zeros((T_total, K))
    gamma_init = gamma.copy()

    # build prior-count matrix
    prior_counts = (jnp.full((K, K), prior.alpha_other)
                    .at[jnp.arange(K), jnp.arange(K)]
                    .set(prior.alpha_self))

    history = []
    z_hist  = []
    z_prob  = []

    past_phis = 0

    if beta_annealing == "linear":
        betas = jnp.linspace(beta,beta_min,num_em_iters)
    elif beta_annealing is None:
        betas = jnp.repeat(beta,num_em_iters)
    else:
        raise NotImplementedError(beta_annealing)

    # ——— Main EM Loop ———
    for em in range(num_em_iters):
        
        # ——— Temperature schedule: tau_initial → 1 over warmup_iterations ———
        tau = max(1.0, tau_initial -  (tau_initial - 1.0) * em / warmup_iterations)

        # ——— Beta schedule ———
        beta = betas[em]

        # ——— E‑step: compute (T_total,K) log‑emissions and posteriors ———
        raw_scores    = batched_compute_scores(xs, phis, log_part)   # (T_total,K)
        log_emissions = raw_scores - jnp.max(raw_scores, axis=1, keepdims=True)
        log_emissions = log_emissions / tau
        gamma_weight  = max(0.0, 1.0 - em/warmup_iterations)
        log_emissions = log_emissions + gamma_weight * jnp.log(gamma_init + 1e-12)
        log_emissions = log_emissions - logsumexp(log_emissions, axis=1, keepdims=True)
        if num_trials is None:
            gamma, xi, logp = forward_backward(log_emissions, log_trans, log_pi)
        else:
            gammas, xis, logps = [], [], []
            for n in range(num_trials):
                start, end = n*T_seq, (n+1)*T_seq
                g, x, lp = forward_backward(log_emissions[start:end], log_trans, log_pi)
                gammas.append(g)
                xis.append(x)
                logps.append(lp)
            gamma = jnp.vstack(gammas)      # (N*T_seq, K)
            xi    = jnp.sum(jnp.stack(xis), axis=0)   # aggregate transitions
            logp  = jnp.sum(jnp.array(logps))         # sum across trials

        history.append(logp / T_total)
        z_hist.append(jnp.argmax(gamma, axis=1))
        z_prob.append(gamma)

        print("Marginal state occupancy:", jnp.mean(gamma, axis=0))

        # ——— M‑step 1: update π and transitions with sticky Dirichlet prior ———
        if fit_initial_conditions:
            if N == 1:
                log_pi = jnp.log(gamma[0] + 1e-12)
            else:
                first_idx = jnp.arange(0, T_total, T_seq)   # [0, T_seq, 2*T_seq, ...]
                avg_init  = gamma[first_idx].mean(axis=0)   # (K,)
                log_pi    = jnp.log(avg_init + 1e-12)
        
        if fit_transition_matrix:
            # MAP‑style numerator
            xi_sum = jnp.sum(xi, axis=0)  # (K,K)
            numer = xi_sum + (prior_counts - 1.0)
            trans_probs = numer / jnp.sum(numer, axis=1, keepdims=True)
            log_trans   = jnp.log(trans_probs + 1e-12)

        # ——— M‑step 2: re‑fit log‑partition params via Optax ———
        @jax.jit
        def part_step(lp, opt_st):
            loss, grads = jax.value_and_grad(cross_entropy_loss_fn)(
                lp, xs, gamma, phis, cross_entropy_lambda
            )
            updates, new_opt = optimizer.update(grads, opt_st)
            return optax.apply_updates(lp, updates), new_opt, loss

        for _ in range(num_part_opt_steps):
            log_part, opt_state, part_loss = part_step(log_part, opt_state)

        
        # ——— M‑step 3: re‑fit phis ———
        if phi_solve_mode == "sm":
            if em == 0:
                phis = batch_tg_solve(xs, gamma.T, tg_solve_reg) # (K,d,d,2)
                past_phis = phis.copy()
            else:
                phis = batch_tg_solve(xs, gamma.T, tg_solve_reg) # (K,d,d,2)
                phis = phis * beta + past_phis * (1 - beta)
                past_phis = phis.copy()

        elif phi_solve_mode == "ssm":
            if em == 0:
                new_phis = []
                for i in range(K):
                    key, subkey = jax.random.split(key)
                    new_phi, _ = estimate_params_ssm(
                        subkey,
                        xs,
                        weights=gamma[:,i],
                        phi=phis[i],
                        lr=1e-2,
                        l2_reg=phi_l2_reg,
                        l1_reg=phi_l1_reg,
                        n_iter=500,
                        no_improvement=100,
                    )
                    new_phis.append(new_phi)
                phis = jnp.stack(new_phis, axis=0)
                past_phis = phis.copy()
            else:
                new_phis = []
                for i in range(K):
                    key, subkey = jax.random.split(key)
                    new_phi, _ = estimate_params_ssm(
                        subkey,
                        xs,
                        weights=gamma[:,i],
                        phi=phis[i],
                        lr=1e-2,
                        l2_reg=phi_l2_reg,
                        l1_reg=phi_l1_reg,
                        n_iter=500,
                        no_improvement=100,
                    )
                    new_phis.append(new_phi)
                phis = jnp.stack(new_phis, axis=0)
                phis = phis * beta + past_phis * (1 - beta)
                past_phis = phis.copy()

        else:
            raise NotImplementedError(phi_solve_mode)

        # Print out EM status.
        print(f"[EM {em:02d}] logp={logp / T_total:.3f}  τ={tau:.2f}  β={beta:.2f} part_loss={part_loss:.3f}")

    return {
        "log_pi":    log_pi,
        "log_trans": log_trans,
        "log_part":  log_part,
        "phis":      phis,
        "history":   jnp.array(history),
        "z_history": jnp.array(z_hist),
        "z_prob": jnp.array(z_prob),
        "logp": logp,
    }

