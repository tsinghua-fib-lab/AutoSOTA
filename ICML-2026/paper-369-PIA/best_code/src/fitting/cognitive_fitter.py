import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.special import logsumexp
from typing import Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm


from ..core import CONFIG
from .strategies import STRATEGY_MAP


@dataclass
class FitResult:
    params: Dict[str, float]
    nll: float
    bic: float
    entropy: float 
    count: int
    group: str


class CognitiveFitter:
    """
    Optimized Cognitive Model Fitter
    """

    def __init__(self, df: pd.DataFrame, config: dict = None):
        self.config = config or {}
        fit_config = self.config.get('parameter_fitting', {})
        self.optimization_config = fit_config.get('optimization', {})

        # --- Preprocess data ---
        self.df = df[df['action'].isin(['Compliance', 'Refusal'])].copy()
        if self.df.empty:
            raise ValueError("DataFrame is empty after filtering Compliance/Refusal")

        self.df['a_idx'] = self.df['action'].apply(lambda x: 1 if x == 'Compliance' else 0)
        self.df['reward'] = pd.to_numeric(self.df.get('reward', 0.0), errors='coerce').fillna(0.0)
        self.df['is_full_feedback'] = self.df.get('is_full_feedback', False).astype(bool)
        self.df['forgone_reward'] = self.df.get('forgone_reward', 0.0)

        framing_groups = ['Stimulus', 'Authority', 'Threat', 'Regret']
        self.df['is_framing'] = self.df['group'].str.contains(
            '|'.join(framing_groups), case=False, na=False
        ).astype(int)

    # -------------------------
    # Utility function
    # -------------------------
    def _utility(self, reward: float, rho: float, lambda_LA: float) -> float:
        """Prospect-theory utility"""
        if reward >= 0:
            return reward ** rho
        return -lambda_LA * ((-reward) ** rho)
    
    # -------------------------
    # Behavior Entropy
    # -------------------------
    def _behavior_entropy(self, acts: np.ndarray) -> float:
        """
        Binary action entropy H(p).
        acts: array of 0/1 actions
        """
        eps = 1e-8
        p = acts.mean()
        return -p * np.log(p + eps) - (1 - p) * np.log(1 - p + eps)


    # -------------------------
    # Negative Log Likelihood
    # -------------------------
    def _nll_loss(
        self,
        params_array,
        acts,
        rews,
        file_ids,
        is_ff,
        forgone_rews,
        is_framing,
        param_config,
        fixed_defaults,
        param_order_scaled,
        show_progress=False
    ):
        """
        NLL Loss with parameter scaling to [0,1] for stable optimization.
        """
        # --- reconstruct parameters ---
        p = fixed_defaults.copy()
        idx = 0
        for k in param_order_scaled:
            if param_config.get(k) == 'free':
                # Unscale from [0,1] to real bounds
                lb, ub = CONFIG.param_bounds[k]
                p[k] = lb + params_array[idx] * (ub - lb)
                idx += 1

        alpha_pos, alpha_neg = p['alpha_pos'], p['alpha_neg']
        rho, beta = p['rho'], p['beta']
        theta, lam, phi = p['theta'], p['lambda'], p['phi']
        lambda_LA = p.get('lambda_LA', 1.0)
        R_perc = p.get('R_perc', 1.0)

        nll = 0.0
        Q = np.zeros(2)
        last_act = -1
        prev_file = None

        # Optional: Nested progress bar for very long trial sequences
        iterator = range(len(acts))
        if show_progress:
            iterator = tqdm(iterator, desc="Computing NLL", leave=False)

        for t in iterator:
            if file_ids[t] != prev_file:
                Q[:] = 0.0
                last_act = -1
                prev_file = file_ids[t]
            else:
                if phi > 0:
                    Q *= (1 - phi)

            logits = Q.copy()
            logits[1] += theta
            if last_act != -1:
                logits[last_act] += lam

            scaled = beta * logits
            log_probs = scaled - logsumexp(scaled)
            nll -= log_probs[acts[t]]

            r = rews[t]
            if is_framing[t]:
                r *= R_perc
            u = self._utility(r, rho, lambda_LA)
            pe = u - Q[acts[t]]
            lr = alpha_pos if pe >= 0 else alpha_neg
            Q[acts[t]] += lr * pe

            if is_ff[t]:
                unchosen = 1 - acts[t]
                r_cf = forgone_rews[t]
                if is_framing[t]:
                    r_cf *= R_perc
                u_cf = self._utility(r_cf, rho, lambda_LA)
                pe_cf = u_cf - Q[unchosen]
                lr_cf = alpha_pos if pe_cf >= 0 else alpha_neg
                Q[unchosen] += lr_cf * pe_cf

            last_act = acts[t]

        return nll

    # -------------------------
    # Fit single scenario
    # -------------------------
    def fit_scenario(
        self,
        group_name: str,
        strategy: str,
        baseline_params: Optional[Dict] = None
    ) -> Optional[FitResult]:

        subset = self.df[self.df['group'] == group_name].copy()
        if len(subset) < self.optimization_config.get('min_trials', 5):
            return None

        if 'file_id' in subset.columns and 'trial' in subset.columns:
            subset = subset.sort_values(['file_id', 'trial'])

        acts = subset['a_idx'].values
        H = self._behavior_entropy(acts)
        rews = subset['reward'].values
        file_ids = subset.get('file_id', pd.Series(np.zeros(len(subset)))).values
        is_ff = subset['is_full_feedback'].values
        forgone_rews = subset['forgone_reward'].values
        is_framing = subset['is_framing'].values

        defaults = baseline_params or CONFIG.default_params.copy()
        config = {k: 'fixed' for k in defaults}
        for k in STRATEGY_MAP.get(strategy, []):
            if k in config:
                config[k] = 'free'

        param_order_scaled = list(defaults.keys())
        x0, bounds = [], []
        for k in param_order_scaled:
            if config[k] == 'free':
                # scale initial value to [0,1]
                lb, ub = CONFIG.param_bounds[k]
                x0.append((defaults[k] - lb) / (ub - lb))
                bounds.append((0.0, 1.0))  # always [0,1] scaled

        if not x0:
            nll = self._nll_loss([], acts, rews, file_ids, is_ff, forgone_rews,
                                  is_framing, config, defaults, param_order_scaled)
            bic = 2 * nll
            return FitResult(defaults, nll, bic, H, len(subset), group_name)

        # -------------------------
        # Optimization with Progress Bar
        # -------------------------
        max_iter = self.optimization_config.get('max_iterations', 200)
        n_starts = self.optimization_config.get('multi_start', 3)

        best_res = None
        best_nll = np.inf

        # Added tqdm here to track multi-start optimization progress
        pbar = tqdm(range(n_starts), desc=f"Fitting {group_name} ({strategy})", leave=False)
        for start in pbar:
            # add random perturbation to x0
            x0_random = np.clip(np.array(x0) + 0.05 * np.random.randn(len(x0)), 0, 1)

            res = minimize(
                self._nll_loss,
                x0_random,
                args=(acts, rews, file_ids, is_ff, forgone_rews,
                      is_framing, config, defaults, param_order_scaled),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-3}
            )

            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best_res = res
                pbar.set_postfix({"best_nll": f"{best_nll:.2f}"})

        if best_res is None:
            return None

        # -------------------------
        # Reconstruct parameters (unscale)
        # -------------------------
        p_final = defaults.copy()
        idx = 0
        for k in param_order_scaled:
            if config[k] == 'free':
                lb, ub = CONFIG.param_bounds[k]
                p_final[k] = lb + best_res.x[idx] * (ub - lb)
                idx += 1

        K = len([k for k in config if config[k] == 'free'])
        bic = np.log(len(subset)) * K + 2 * best_res.fun

        return FitResult(
            params=p_final,
            nll=best_res.fun,
            bic=bic,
            entropy=H,
            count=len(subset),
            group=group_name
        )