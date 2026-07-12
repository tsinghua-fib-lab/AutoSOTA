import numpy as np
import torch
import gpytorch
from src.utils import optimize_t_for_x_batch_torch 
from tqdm import tqdm
from typing import Tuple, Dict, Optional
import math

try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: Scipy not installed. EISampler will not be available.")


def _torch_helper(pool_x, model, objective, t_grid_size, **kwargs):
    """
    Internal helper function: (np -> torch -> torch_util -> np)
    """
    try:
        # 1. Convert to GPU Tensor
        x_batch_torch = torch.tensor(pool_x, dtype=model.dtype, device=model.device)
        
        # 2. Call Native Torch Optimizer
        t_batch_torch, score_batch_torch = optimize_t_for_x_batch_torch(
            model, x_batch_torch, objective, 
            t_grid_size=t_grid_size, 
            beta=kwargs.get('beta', 1.96)
        )
        
        # 3. Convert back to NumPy
        return score_batch_torch.cpu().numpy(), t_batch_torch.cpu().numpy()
        
    except Exception as e:
        print(f"Warning: _torch_helper (obj={objective}) failed: {e}")
        return np.full(len(pool_x), -np.inf), np.full(len(pool_x), 0.5)

def _evaluate_batch_max_var(pool_x, model, t_grid_size):
    """(Used for MaxVarFullSampler)"""
    return _torch_helper(pool_x, model, 'variance', t_grid_size)

def _evaluate_batch_gpucb(pool_x, model, beta, t_grid_size):
    """(Used for GPUCBSampler)"""
    return _torch_helper(pool_x, model, 'ucb', t_grid_size, beta=beta)
 
def _evaluate_batch_smoothed_mean_torch(pool_x, model, t_grid_size, rng, smoothing_h):
    """(Used for SmoothedMeanSampler)"""
    try:
        # 1. Convert to GPU Tensor
        x_batch_torch = torch.tensor(pool_x, dtype=model.dtype, device=model.device)
        
        # 2. Find optimal t (Exploitation)
        t_star_batch_torch, _ = optimize_t_for_x_batch_torch(
            model, x_batch_torch, 'mean', t_grid_size=t_grid_size
        )
        
        # (Convert back to CPU for random sampling)
        t_star_batch = t_star_batch_torch.cpu().numpy()
        
        # 3. Apply smoothing
        low_band = np.clip(t_star_batch - smoothing_h, 0.0, 1.0)
        high_band = np.clip(t_star_batch + smoothing_h, 0.0, 1.0)
        
        # 4. Sample t
        t_batch = rng.uniform(low_band, high_band, size=pool_x.shape[0])
        score_batch = np.zeros(pool_x.shape[0]) 
        
        return score_batch, t_batch
        
    except Exception as e:
        print(f"Warning: _evaluate_batch_smoothed_mean failed: {e}")
        return np.full(len(pool_x), -np.inf), np.full(len(pool_x), 0.5)
 
def _evaluate_batch_bald_xt(pool_x, model, t_grid_size):
    """(Used for SoftTopKSampler)"""
    # (BALD/MaxVar evaluation is same as MaxVar)
    return _torch_helper(pool_x, model, 'variance', t_grid_size)

# --- (End of helper functions) ---


class BaseSampler:
    """ Base Sampler Class """
    def __init__(self, rng, beta=1.96, num_threads=1, 
                 n_candidates=1000, gpu_batch_size=128, t_grid_size=101, **kwargs):
        self.rng = rng
        self.beta = beta
        self.num_threads = num_threads if num_threads and num_threads > 0 else 1
        
        self.n_candidates = int(n_candidates)
        self.gpu_batch_size = int(gpu_batch_size)
        self.t_grid_size = int(t_grid_size)
    
    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        raise NotImplementedError

class RandomSampler(BaseSampler):
    """ Baseline 1: RAND """
    def __init__(self, rng, **kwargs):
        super().__init__(rng, **kwargs)
        
    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        selected_indices = self.rng.choice(available_indices, B, replace=False)
        assigned_t = self.rng.uniform(0, 1, size=B)
        return selected_indices, assigned_t

class MaxVarFullSampler(BaseSampler):
    """ Baseline 2: AL-Full (Optimized) """
    
    def __init__(self, rng, **kwargs):
        super().__init__(rng, **kwargs)

    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        if B <= 0: return np.array([], dtype=int), np.array([], dtype=float)

        N_eval = min(self.n_candidates, len(available_indices))
        if len(available_indices) > N_eval:
            candidates = self.rng.choice(available_indices, N_eval, replace=False)
        else:
            candidates = np.asarray(available_indices)
        X_candidates = X_pool[candidates]
        
        all_scores = []
        all_ts = []
        
        pbar_desc = "MaxVarFullSampler (GPU Batches)"
        for i in tqdm(range(0, N_eval, self.gpu_batch_size), desc=pbar_desc, leave=False):
            x_chunk = X_candidates[i : i + self.gpu_batch_size]
            
            score_chunk, t_chunk = _evaluate_batch_max_var(
                x_chunk, model, t_grid_size=self.t_grid_size
            )
            
            all_scores.append(score_chunk)
            all_ts.append(t_chunk)

        scores = np.concatenate(all_scores)
        target_ts = np.concatenate(all_ts)
        
        if not np.any(np.isfinite(scores)):
             print("Warning: MaxVarFullSampler received non-finite scores, falling back to Random.")
             return RandomSampler(self.rng).select_batch(model, X_pool, available_indices, B, train_y)
        
        candidate_score_pairs = []
        for i in range(len(candidates)):
            candidate_score_pairs.append((scores[i], candidates[i], target_ts[i]))
            
        candidate_score_pairs.sort(key=lambda x: x[0], reverse=True)
        
        B = min(B, len(candidate_score_pairs)) 
        top_B = candidate_score_pairs[:B]
        selected_pool_indices = np.array([x[1] for x in top_B], dtype=int)
        assigned_t = np.array([x[2] for x in top_B], dtype=float)
        
        return selected_pool_indices, assigned_t
 
class GPUCBSampler(BaseSampler):
    """ SOTA Baseline 1: GPUCB (Optimized) """
    
    def __init__(self, rng, **kwargs):
        super().__init__(rng, **kwargs)

    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        if B <= 0: return np.array([], dtype=int), np.array([], dtype=float)

        N_eval = min(self.n_candidates, len(available_indices))
        if len(available_indices) > N_eval:
            candidates = self.rng.choice(available_indices, N_eval, replace=False)
        else:
            candidates = np.asarray(available_indices)
        X_candidates = X_pool[candidates]
        
        all_scores = []
        all_ts = []
        
        pbar_desc = "GPUCBSampler (GPU Batches)"
        for i in tqdm(range(0, N_eval, self.gpu_batch_size), desc=pbar_desc, leave=False):
            x_chunk = X_candidates[i : i + self.gpu_batch_size]
            
            score_chunk, t_chunk = _evaluate_batch_gpucb(
                x_chunk, model, beta=self.beta, t_grid_size=self.t_grid_size
            )
            
            all_scores.append(score_chunk)
            all_ts.append(t_chunk)

        scores = np.concatenate(all_scores)
        target_ts = np.concatenate(all_ts)
        
        if not np.any(np.isfinite(scores)):
             print("Warning: GPUCBSampler received non-finite scores, falling back to Random.")
             return RandomSampler(self.rng).select_batch(model, X_pool, available_indices, B, train_y)
        
        candidate_score_pairs = []
        for i in range(len(candidates)):
            candidate_score_pairs.append((scores[i], candidates[i], target_ts[i]))
            
        candidate_score_pairs.sort(key=lambda x: x[0], reverse=True)
        
        B = min(B, len(candidate_score_pairs))
        top_B = candidate_score_pairs[:B]
        selected_pool_indices = np.array([x[1] for x in top_B], dtype=int)
        assigned_t = np.array([x[2] for x in top_B], dtype=float)
        
        return selected_pool_indices, assigned_t


class TSSampler(BaseSampler):
    """ SOTA Baseline 2: Thompson Sampling (TS) (Optimized) """
    
    def __init__(self, rng, **kwargs):
        super().__init__(rng, **kwargs)

    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        if B <= 0: return np.array([], dtype=int), np.array([], dtype=float)

        N_eval = min(self.n_candidates, len(available_indices))
        if len(available_indices) > N_eval:
            candidates = self.rng.choice(available_indices, N_eval, replace=False)
        else:
            candidates = np.asarray(available_indices)
        X_candidates = X_pool[candidates]
        
        all_scores = []
        all_ts = []
        
        device = model.device
        dtype = model.dtype
        
        t_grid = torch.linspace(0, 1, self.t_grid_size, device=device, dtype=dtype).view(-1, 1) # (G, 1)

        pbar_desc = "TSSampler (GPU Batches)"
        for i in tqdm(range(0, N_eval, self.gpu_batch_size), desc=pbar_desc, leave=False):
            x_chunk_np = X_candidates[i : i + self.gpu_batch_size]
            N_chunk = x_chunk_np.shape[0]
            if N_chunk == 0: continue
            
            x_chunk = torch.tensor(x_chunk_np, dtype=dtype, device=device)
            dim_x = x_chunk.shape[1]
            
            # Super batch creation on GPU
            x_tiled = x_chunk.unsqueeze(1).tile((1, self.t_grid_size, 1))
            t_tiled = t_grid.unsqueeze(0).tile((N_chunk, 1, 1))
            super_batch_2d = torch.cat([x_tiled, t_tiled], dim=2).view(-1, dim_x + 1)
            
            # Batch GPU prediction
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mean_chunks = []
                var_chunks = []
                for x_b in torch.split(super_batch_2d, 1024 * self.t_grid_size):
                    pred = model.likelihood(model(x_b))
                    mean_chunks.append(pred.mean)
                    var_chunks.append(pred.variance)

                if not mean_chunks:
                    continue

                mean_flat = torch.cat(mean_chunks, dim=0)
                var_flat = torch.cat(var_chunks, dim=0)
            
            mean_grid = mean_flat.view(N_chunk, self.t_grid_size)
            std_grid = var_flat.view(N_chunk, self.t_grid_size).clamp_min(1e-9).sqrt()
            
            # TS Core: Sampling on GPU
            sample_grid = mean_grid + std_grid * torch.randn_like(mean_grid)
            
            # Find best on GPU
            score_chunk_torch, t_indices_torch = torch.max(sample_grid, dim=1)
            t_chunk_torch = t_grid.flatten()[t_indices_torch]
            
            all_scores.append(score_chunk_torch.cpu().numpy())
            all_ts.append(t_chunk_torch.cpu().numpy())

        scores = np.concatenate(all_scores)
        target_ts = np.concatenate(all_ts)
        
        if not np.any(np.isfinite(scores)):
             return RandomSampler(self.rng).select_batch(model, X_pool, available_indices, B, train_y)
        
        candidate_score_pairs = []
        for i in range(len(candidates)):
            candidate_score_pairs.append((scores[i], candidates[i], target_ts[i]))
            
        candidate_score_pairs.sort(key=lambda x: x[0], reverse=True)
        
        B = min(B, len(candidate_score_pairs))
        top_B = candidate_score_pairs[:B]
        selected_pool_indices = np.array([x[1] for x in top_B], dtype=int)
        assigned_t = np.array([x[2] for x in top_B], dtype=float)
        
        return selected_pool_indices, assigned_t


class EISampler(BaseSampler):
    """ SOTA Baseline 3: Expected Improvement (EI) (Optimized) """
    
    def __init__(self, rng, xi=0.01, **kwargs):
        super().__init__(rng, **kwargs)
        self.xi = xi
        if not SCIPY_AVAILABLE:
            raise ImportError("EISampler requires 'scipy' to be installed (for scipy.stats.norm).")

    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        if B <= 0: return np.array([], dtype=int), np.array([], dtype=float)

        best_value = np.max(train_y) if (train_y is not None and len(train_y) > 0) else -np.inf

        N_eval = min(self.n_candidates, len(available_indices))
        if len(available_indices) > N_eval:
            candidates = self.rng.choice(available_indices, N_eval, replace=False)
        else:
            candidates = np.asarray(available_indices)
        X_candidates = X_pool[candidates]
        
        all_scores = []
        all_ts = []
        
        device = model.device
        dtype = model.dtype
        # t-grid on CPU because EI calculations are on CPU
        t_grid = np.linspace(0, 1, self.t_grid_size).reshape(-1, 1) # (G, 1)

        pbar_desc = "EISampler (GPU Batches)"
        for i in tqdm(range(0, N_eval, self.gpu_batch_size), desc=pbar_desc, leave=False):
            x_chunk_np = X_candidates[i : i + self.gpu_batch_size]
            N_chunk = x_chunk_np.shape[0]
            if N_chunk == 0: continue
            
            x_chunk = torch.tensor(x_chunk_np, dtype=dtype, device=device)
            dim_x = x_chunk.shape[1]
            
            # Super batch in NumPy for CPU evaluation
            x_tiled = np.tile(x_chunk_np[:, np.newaxis, :], (1, self.t_grid_size, 1))
            t_tiled_np = np.tile(t_grid[np.newaxis, :, :], (N_chunk, 1, 1))
            super_batch_3d = np.concatenate([x_tiled, t_tiled_np], axis=2)
            super_batch_2d = super_batch_3d.reshape(N_chunk * self.t_grid_size, dim_x + 1)
            
            mean_flat, var_flat = model.predict(super_batch_2d)
            
            mean_grid_np = mean_flat.reshape(N_chunk, self.t_grid_size)
            var_grid_np = var_flat.reshape(N_chunk, self.t_grid_size)
            std_grid_np = np.sqrt(np.maximum(var_grid_np, 1e-9))
            
            # EI calculation (CPU)
            imp = mean_grid_np - best_value - self.xi
            Z = imp / (std_grid_np + 1e-9)
            ei_grid = imp * norm.cdf(Z) + std_grid_np * norm.pdf(Z)
            
            score_chunk = np.max(ei_grid, axis=1)
            t_indices = np.argmax(ei_grid, axis=1)
            t_chunk = t_grid.flatten()[t_indices]
            
            all_scores.append(score_chunk)
            all_ts.append(t_chunk)

        scores = np.concatenate(all_scores)
        target_ts = np.concatenate(all_ts)
        
        if not np.any(np.isfinite(scores)):
             return RandomSampler(self.rng).select_batch(model, X_pool, available_indices, B, train_y)
        
        candidate_score_pairs = []
        for i in range(len(candidates)):
            candidate_score_pairs.append((scores[i], candidates[i], target_ts[i]))
            
        candidate_score_pairs.sort(key=lambda x: x[0], reverse=True)
        
        B = min(B, len(candidate_score_pairs))
        top_B = candidate_score_pairs[:B]
        selected_pool_indices = np.array([x[1] for x in top_B], dtype=int)
        assigned_t = np.array([x[2] for x in top_B], dtype=float)
        
        return selected_pool_indices, assigned_t


class SmoothedMeanSampler(BaseSampler):
    """ SOTA Baseline 5: CATS (Optimized) """
    
    def __init__(self, rng, smoothing_h=0.05, **kwargs):
        super().__init__(rng, **kwargs)
        self.smoothing_h = smoothing_h
        print(f"SmoothedMeanSampler initialized with h={self.smoothing_h}")

    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        if B <= 0: return np.array([], dtype=int), np.array([], dtype=float)

        # 1. X-Selection: Random
        selected_indices = self.rng.choice(available_indices, B, replace=False)
        selected_x_batch = X_pool[selected_indices]
        
        # 2. T-Selection: Smoothed Mean
        _scores, assigned_t = _evaluate_batch_smoothed_mean_torch(
            selected_x_batch,
            model,
            t_grid_size=self.t_grid_size,
            rng=self.rng,
            smoothing_h=self.smoothing_h
        )
        
        return selected_indices, assigned_t
 
class ABC3Sampler(BaseSampler):
    """
    Correct Implementation of ABC3 (Active Bayesian Causal Inference with Cohn Criteria).
    Adapted for Continuous Treatment settings.
    
    Theoretical Basis:
        Minimizes the Integrated Posterior Variance (IVR) over the whole domain.
        Based on Proposition 4.2 in the paper.
        
        Optimization Target:
        argmax_{(x, t)} ( \int [Cov((x,t), (x',t'))]^2 dP(x',t') ) / ( Var(x,t) + sigma^2 )
        
    Mechanism:
        1. Reference Set (Z): Samples a subset of X_pool and creates a grid of t to 
           approximate the integral over the population P(x, t).
        2. Candidate Set (Q): Samples candidate X and creates a grid of t to search for optimal action.
        3. Greedy Selection: Selects (x, t) that maximizes variance reduction on Z, 
           then uses Schur complement to update covariance for batch selection.
    """
    
    def __init__(
        self, 
        rng, 
        ref_size=200,          # Size of the reference set for integration approximation
        ref_t_grid_size=5,     # Grid size for T in the integration step
        cand_t_grid_size=10,   # Grid size for T when optimizing the candidate
        variance_floor=1e-6,
        extra_diag_jitter=5e-4,
        **kwargs
    ):
        super().__init__(rng, **kwargs)
        self.ref_size = ref_size
        self.ref_t_grid_size = ref_t_grid_size
        self.cand_t_grid_size = cand_t_grid_size
        self.variance_floor = variance_floor
        self.extra_diag_jitter = extra_diag_jitter

    @staticmethod
    def _nan_safe(tensor, fill_value=0.0):
        return torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)

    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        """
        Selects B (x, t) pairs that maximize the reduction of global variance.
        """
        B = min(B, len(available_indices))
        if B <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # -------------------------------------------------------
        # 0. Setup Device & Model
        # -------------------------------------------------------
        base_model = getattr(model, "model", model)
        likelihood = getattr(model, "likelihood", getattr(base_model, "likelihood", None))
        
        try:
            param = next(base_model.parameters())
            device = param.device
            dtype = param.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float64

        if hasattr(base_model, "eval"): base_model.eval()
        if likelihood and hasattr(likelihood, "eval"): likelihood.eval()

        # -------------------------------------------------------
        # 1. Construct Reference Set Z (Integration Domain)
        # -------------------------------------------------------
        n_ref = min(self.ref_size, len(X_pool))
        ref_indices = self.rng.choice(len(X_pool), n_ref, replace=False)
        X_ref_np = X_pool[ref_indices]
        
        t_ref_grid = np.linspace(0, 1, self.ref_t_grid_size)
        
        # Z = Cartesian Product (X_ref x T_ref_grid)
        X_ref_rep = torch.tensor(
            np.repeat(X_ref_np, self.ref_t_grid_size, axis=0), 
            dtype=dtype, device=device
        )
        T_ref_rep = torch.tensor(
            np.tile(t_ref_grid, n_ref), 
            dtype=dtype, device=device
        ).unsqueeze(-1)
        
        Z_inputs = torch.cat([X_ref_rep, T_ref_rep], dim=1)

        # -------------------------------------------------------
        # 2. Construct Candidate Set Q (Search Space)
        # -------------------------------------------------------
        n_cands = min(self.n_candidates, len(available_indices))
        if len(available_indices) > n_cands:
            cand_indices = self.rng.choice(available_indices, n_cands, replace=False)
        else:
            cand_indices = np.asarray(available_indices)
        
        X_cands_np = X_pool[cand_indices]
        t_cand_grid = np.linspace(0, 1, self.cand_t_grid_size)
        
        X_cands_rep = torch.tensor(
            np.repeat(X_cands_np, self.cand_t_grid_size, axis=0), 
            dtype=dtype, device=device
        )
        T_cands_rep = torch.tensor(
            np.tile(t_cand_grid, n_cands), 
            dtype=dtype, device=device
        ).unsqueeze(-1)
        
        Q_inputs = torch.cat([X_cands_rep, T_cands_rep], dim=1)

        num_z = Z_inputs.shape[0]
        num_q = Q_inputs.shape[0]

        # -------------------------------------------------------
        # 3. Compute Joint Covariance (Z u Q)
        # -------------------------------------------------------
        all_inputs_raw = torch.cat([Z_inputs, Q_inputs], dim=0)
        
        # Deduplicate to prevent singular matrices
        unique_inputs, inverse_idx = torch.unique(
            all_inputs_raw, dim=0, return_inverse=True
        )

        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(1e-4):
            posterior = model(unique_inputs)
            unique_cov = posterior.covariance_matrix
            
        full_cov = unique_cov.index_select(0, inverse_idx).index_select(1, inverse_idx)
        
        start_q = num_z
        K_zq = full_cov[:num_z, start_q:]
        K_qq = full_cov[start_q:, start_q:].clone()
        
        sigma_sq = 0.0
        if likelihood is not None and hasattr(likelihood, "noise"):
            noise_val = likelihood.noise
            sigma_sq = float(noise_val.detach().item()) if torch.is_tensor(noise_val) else float(noise_val)
        
        # Predictive Variance of Q
        curr_var_q = torch.diagonal(K_qq) + sigma_sq
        curr_var_q = torch.clamp(curr_var_q, min=self.variance_floor)
        
        curr_K_zq = K_zq.clone()

        # -------------------------------------------------------
        # 4. Greedy Selection with Schur Update
        # -------------------------------------------------------
        selected_indices_in_avail = []
        selected_ts = []
        
        flat_to_cand_idx = torch.div(
            torch.arange(num_q, device=device), 
            self.cand_t_grid_size, 
            rounding_mode='floor'
        )
        cand_mask = torch.ones(n_cands, dtype=torch.bool, device=device)
        
        for _ in range(B):
            # Score Calculation (Prop 4.2)
            numerator = torch.sum(curr_K_zq.square(), dim=0) 
            scores = numerator / curr_var_q
            
            # Mask out already selected candidates
            valid_mask = cand_mask[flat_to_cand_idx]
            scores[~valid_mask] = -float('inf')
            
            # Select best (x, t) pair
            best_flat_idx = torch.argmax(scores).item()
            
            if not math.isfinite(float(scores[best_flat_idx])):
                break 
                
            best_cand_idx = flat_to_cand_idx[best_flat_idx].item()
            
            selected_indices_in_avail.append(cand_indices[best_cand_idx])
            t_val = t_cand_grid[best_flat_idx % self.cand_t_grid_size]
            selected_ts.append(t_val)
            
            cand_mask[best_cand_idx] = False
            
            # Schur Complement Update
            if len(selected_indices_in_avail) < B:
                v_q = K_qq[:, best_flat_idx]
                u_z = curr_K_zq[:, best_flat_idx]
                denom = curr_var_q[best_flat_idx]
                
                # Update beliefs
                update_term_cov = torch.outer(u_z, v_q) / denom
                curr_K_zq = curr_K_zq - update_term_cov
                curr_K_zq = self._nan_safe(curr_K_zq)
                
                update_term_var = v_q.square() / denom
                curr_var_q = curr_var_q - update_term_var
                
                K_qq = K_qq - torch.outer(v_q, v_q) / denom
                
                # Stabilization
                diag_updated = torch.diagonal(K_qq)
                diag_updated += self.extra_diag_jitter
                diag_updated.copy_(torch.clamp(diag_updated, min=self.extra_diag_jitter))
                
                curr_var_q = torch.clamp(diag_updated + sigma_sq, min=self.variance_floor)
                
                K_qq = self._nan_safe(K_qq)
                curr_var_q = self._nan_safe(curr_var_q, fill_value=self.variance_floor)

        return np.array(selected_indices_in_avail, dtype=int), np.array(selected_ts, dtype=float)

class PolicyGradientSampler(BaseSampler):
    """
    Policy-Gradient Based Optimization Sampler.
    Based on Razzak et al. (2024), Algorithm 4.

    Reward (GP closed-form proxy): 0.5 * LogDet(I + K/sigma^2)
    """
    def __init__(
        self,
        rng,
        epochs=50,
        lr_x=0.1,
        lr_t=0.1,
        t_init_log_std=-2.0,   # sigma ~ exp(-2) ≈ 0.135
        mask_fill_value=-1e9,  # for masked logits
        **kwargs
    ):
        super().__init__(rng, **kwargs)
        self.epochs = epochs
        self.lr_x = lr_x
        self.lr_t = lr_t
        self.t_init_log_std = t_init_log_std
        self.mask_fill_value = mask_fill_value

    def _sample_batch_without_replacement(self, logits_x, B):
        """
        Sequential without-replacement sampling with correct joint log-prob.
        """
        device = logits_x.device
        mask = torch.zeros(logits_x.shape[0], dtype=torch.bool, device=device)

        selected = []
        logp_total = torch.zeros((), device=device, dtype=logits_x.dtype)

        for _ in range(B):
            masked_logits = logits_x.masked_fill(mask, self.mask_fill_value)
            
            dist = torch.distributions.Categorical(logits=masked_logits)
            idx = dist.sample() 
            logp_total = logp_total + dist.log_prob(idx)
            selected.append(idx)
            
            # Clone before modifying to avoid in-place error
            mask = mask.clone()
            mask[idx] = True

        return torch.stack(selected), logp_total
        
    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        if B <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        available_indices = np.asarray(available_indices)
        N_avail = len(available_indices)
        X_cands_np = X_pool[available_indices]

        base_model = getattr(model, "model", model)
        likelihood = getattr(model, "likelihood", getattr(base_model, "likelihood", None))

        device = model.device
        dtype = model.dtype

        X_cands_torch = torch.tensor(X_cands_np, dtype=dtype, device=device)

        # Policy parameters
        logits_x = torch.zeros(N_avail, dtype=dtype, device=device, requires_grad=True)
        t_mu_params = torch.zeros(N_avail, dtype=dtype, device=device, requires_grad=True)
        t_log_std = torch.tensor(self.t_init_log_std, dtype=dtype, device=device, requires_grad=True)

        optimizer = torch.optim.Adam(
            [
                {"params": [logits_x], "lr": self.lr_x},
                {"params": [t_mu_params, t_log_std], "lr": self.lr_t},
            ]
        )

        sigma_sq = 1e-4
        if likelihood is not None and hasattr(likelihood, "noise"):
            with torch.no_grad():
                noise = likelihood.noise
                sigma_sq = noise.item() if torch.is_tensor(noise) else float(noise)

        base_model.eval()
        baseline_reward = 0.0  # EMA baseline

        for _ in range(self.epochs):
            optimizer.zero_grad()

            # Sample X batch
            selected_idx_local, logp_total = self._sample_batch_without_replacement(logits_x, B)
            X_batch = X_cands_torch[selected_idx_local]

            # Sample t batch with pathwise derivative
            mu = t_mu_params[selected_idx_local]
            std = torch.exp(t_log_std).clamp_min(1e-6)

            dist_t = torch.distributions.Normal(loc=mu, scale=std)
            z = dist_t.rsample() 
            t_batch = torch.sigmoid(z)

            inputs = torch.cat([X_batch, t_batch.unsqueeze(-1)], dim=1)

            # Reward: GP logdet proxy
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
                output = base_model(inputs)
                cov = output.covariance_matrix 
                cov_noisy = cov + torch.eye(B, device=device, dtype=dtype) * sigma_sq

                try:
                    L = torch.linalg.cholesky(cov_noisy)
                    log_det = 2.0 * L.diag().log().sum()
                except RuntimeError:
                    jitter = 1e-3
                    cov_noisy2 = cov_noisy + torch.eye(B, device=device, dtype=dtype) * jitter
                    log_det = torch.logdet(cov_noisy2)

            reward = 0.5 * log_det

            # REINFORCE for X
            advantage = (reward - baseline_reward).detach()
            loss_x = -advantage * logp_total

            # Pathwise for t
            loss_t = -reward

            total_loss = loss_x + loss_t
            total_loss.backward()
            optimizer.step()

            baseline_reward = 0.9 * baseline_reward + 0.1 * float(reward.detach().cpu().item())

        # Final selection
        with torch.no_grad():
            selected_idx_local, _ = self._sample_batch_without_replacement(logits_x, B)

            mu = t_mu_params[selected_idx_local]
            std = torch.exp(t_log_std).clamp_min(1e-6)
            z = mu + std * torch.randn_like(mu)
            best_t = torch.sigmoid(z).cpu().numpy()

            selected_indices = available_indices[selected_idx_local.cpu().numpy()]

        return selected_indices, best_t

class SoftTopKSampler(BaseSampler):
    """ SOTA Baseline 8: Soft Top-K Sampler (Optimized) """
    
    def __init__(self, rng, bald_temperature=0.1, **kwargs):
        super().__init__(rng, **kwargs)
        self.temperature = bald_temperature
        print(f"SoftTopKSampler initialized with temperature={self.temperature}")

    def _softmax(self, x, temp):
        if temp == 0:
            probs = np.zeros_like(x)
            probs[np.argmax(x)] = 1.0
            return probs
        e_x = np.exp((x - np.max(x)) / temp)
        return e_x / (e_x.sum() + 1e-9)

    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        if B <= 0: return np.array([], dtype=int), np.array([], dtype=float)

        # 1. Candidate subsampling
        N_eval = min(self.n_candidates, len(available_indices))
        if len(available_indices) > N_eval:
            candidates = self.rng.choice(available_indices, N_eval, replace=False)
        else:
            candidates = np.asarray(available_indices)
        X_candidates = X_pool[candidates]
        
        all_scores = []
        all_ts = []
        
        pbar_desc = "SoftTopKSampler (GPU Batches)"
        for i in tqdm(range(0, N_eval, self.gpu_batch_size), desc=pbar_desc, leave=False):
            x_chunk = X_candidates[i : i + self.gpu_batch_size]
            
            score_chunk, t_chunk = _evaluate_batch_bald_xt(
                x_chunk, model, t_grid_size=self.t_grid_size
            )
            
            all_scores.append(score_chunk)
            all_ts.append(t_chunk)

        scores = np.concatenate(all_scores)
        target_ts = np.concatenate(all_ts)
        
        if not np.any(np.isfinite(scores)):
             print("Warning: SoftTopKSampler received non-finite scores, falling back to Random.")
             return RandomSampler(self.rng).select_batch(model, X_pool, available_indices, B, train_y)
        
        # 3. Softmax (CPU)
        temp = self.temperature if self.temperature > 1e-9 else 1e-9
        probs = self._softmax(scores, temp=temp)
        
        # 4. Sampling (CPU)
        B = min(B, len(candidates))
        candidate_indices_local = np.arange(len(candidates))
        
        selected_local_indices = self.rng.choice(
            candidate_indices_local,
            size=B,
            replace=False,
            p=probs
        )
        
        # 5. Mapping
        selected_pool_indices = candidates[selected_local_indices]
        assigned_t = target_ts[selected_local_indices]
        
        return selected_pool_indices, assigned_t

from sklearn.cluster import KMeans

def select_targets_by_kmeans(X_pool, n_targets, rng):
    kmeans = KMeans(n_clusters=n_targets, random_state=rng.randint(10**9))
    labels = kmeans.fit_predict(X_pool)
    target_indices = []
    for k in range(n_targets):
        mask = (labels == k)
        cluster_idx = np.where(mask)[0]
        if len(cluster_idx) == 0:
            continue
        center = kmeans.cluster_centers_[k]
        pts = X_pool[cluster_idx]
        d2 = np.sum((pts - center)**2, axis=1)
        best = cluster_idx[np.argmin(d2)]
        target_indices.append(best)
    return np.array(target_indices, dtype=int)

class GVALIDSampler(BaseSampler):
    """
    GVALID: Curvature-Aware Gradient Estimation Sampler.
    
    Minimizes the variance of the first derivative (g = df/dt) at the predicted 
    optimal dose points for the population.
    
    Method:
        1. Representative Targets: Sample N individuals and find their current t*.
        2. Finite Diff: Approximate gradient cov via Cov(f(t+d) - f(t-d), f(q)).
        3. Batch Selection: Greedy-Schur complement updates to select batch B.
    """
    def __init__(
        self,
        rng,
        target_sample_size=64,
        diff_delta=0.01,
        min_fd_separation=0.05,
        fd_edge_eps=1e-4,
        extra_diag_jitter=5e-4,
        variance_floor=1e-6,
        **kwargs,
    ):
        super().__init__(rng, **kwargs)
        self.target_sample_size = target_sample_size
        self.delta = diff_delta 
        self.min_fd_separation = max(min_fd_separation, 1e-4)
        self.fd_edge_eps = fd_edge_eps
        self.extra_diag_jitter = extra_diag_jitter
        self.variance_floor = variance_floor
        self.cand_t_grid_size = 20 

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_fd_time_pairs(self, t_star_array):
        """
        Construct finite-difference time pairs (t_minus, t_plus) with enforced separation.
        """
        t_star = np.asarray(t_star_array, dtype=np.float64)
        half_span = max(self.delta, 0.5 * self.min_fd_separation)

        t_plus = t_star + half_span
        t_minus = t_star - half_span

        upper_limit = 1.0 - self.fd_edge_eps
        lower_limit = self.fd_edge_eps

        overflow = np.maximum(t_plus - upper_limit, 0.0)
        t_plus -= overflow
        t_minus -= overflow

        underflow = np.maximum(lower_limit - t_minus, 0.0)
        t_plus += underflow
        t_minus += underflow

        t_plus = np.clip(t_plus, lower_limit, upper_limit)
        t_minus = np.clip(t_minus, lower_limit, upper_limit)

        span = t_plus - t_minus
        min_span = max(self.min_fd_separation * 0.5, 1e-3)
        valid_mask = span >= min_span

        if not np.all(valid_mask):
            fallback_idx = np.nonzero(~valid_mask)[0]
            for idx in fallback_idx:
                center = t_star[idx]
                max_up = upper_limit - center
                max_down = center - lower_limit
                fallback_delta = min(self.delta, max(max_up, max_down))
                if fallback_delta <= 0.0:
                    fallback_delta = self.fd_edge_eps

                t_plus[idx] = np.clip(center + fallback_delta, lower_limit, upper_limit)
                t_minus[idx] = np.clip(center - fallback_delta, lower_limit, upper_limit)

                if np.isclose(t_plus[idx], t_minus[idx]):
                    t_plus[idx] = np.nextafter(t_plus[idx], 1.0)
                    t_minus[idx] = np.nextafter(t_minus[idx], 0.0)

            span = t_plus - t_minus
            valid_mask = span >= min_span

        return t_plus.astype(np.float64), t_minus.astype(np.float64), valid_mask

    @staticmethod
    def _nan_safe(tensor, fill_value=0.0):
        return torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)

    # ------------------------------------------------------------------
    # Main selection routine
    # ------------------------------------------------------------------
    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        if B <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        base_model = getattr(model, "model", model)
        likelihood = getattr(model, "likelihood", getattr(base_model, "likelihood", None))

        def _infer_device_dtype():
            device = getattr(model, "device", None)
            dtype = getattr(model, "dtype", None)
            if device is None or dtype is None:
                try:
                    param = next(base_model.parameters())
                    if device is None:
                        device = param.device
                    if dtype is None:
                        dtype = param.dtype
                except StopIteration:
                    device = device or torch.device("cpu")
                    dtype = dtype or torch.float64
            return device, dtype

        device, dtype = _infer_device_dtype()

        was_training_base = getattr(base_model, "training", None)
        base_has_mode_switch = hasattr(base_model, "eval") and callable(getattr(base_model, "eval"))
        if base_has_mode_switch:
            base_model.eval()

        like_was_training = getattr(likelihood, "training", None) if likelihood is not None else None
        like_has_mode_switch = likelihood is not None and hasattr(likelihood, "eval") and callable(getattr(likelihood, "eval"))
        if like_has_mode_switch:
            likelihood.eval()

        try:
            # 1. Construct Target Set Z = {(x_i, t*_i)}
            n_targets = min(self.target_sample_size, len(X_pool))
            target_indices = self.rng.choice(len(X_pool), n_targets, replace=False)
            X_targets = X_pool[target_indices]

            # Adaptive beta for t* estimation: early exploration, late exploitation
            round_num = validation_context.get('round', 0) if validation_context else 0
            adaptive_beta_t = 0.5 * max(0.1, 1.0 - round_num / 63.0)
            _, t_star_targets = _torch_helper(
                X_targets,
                model,
                'ucb', 
                t_grid_size=self.t_grid_size,
                beta=adaptive_beta_t
            )

            t_plus, t_minus, valid_mask = self._build_fd_time_pairs(t_star_targets)

            if not np.all(valid_mask):
                X_targets = X_targets[valid_mask]
                t_star_targets = t_star_targets[valid_mask]
                t_plus = t_plus[valid_mask]
                t_minus = t_minus[valid_mask]

            if X_targets.shape[0] == 0:
                raise RuntimeError("Failed to construct valid finite-difference targets.")

            # 2. Construct Candidate Set Q
            n_cands = min(self.n_candidates, len(available_indices))
            if len(available_indices) > n_cands:
                cand_indices = self.rng.choice(available_indices, n_cands, replace=False)
            else:
                cand_indices = np.asarray(available_indices)

            X_cands_np = X_pool[cand_indices]
            t_grid = np.linspace(0, 1, self.cand_t_grid_size)

            # 3. Prepare Tensors
            X_targets_torch = torch.tensor(X_targets, dtype=dtype, device=device)

            Z_plus = torch.cat(
                [
                    X_targets_torch,
                    torch.tensor(t_plus, dtype=dtype, device=device).unsqueeze(-1)
                ],
                dim=1
            )
            Z_minus = torch.cat(
                [
                    X_targets_torch,
                    torch.tensor(t_minus, dtype=dtype, device=device).unsqueeze(-1)
                ],
                dim=1
            )

            X_cands_rep = torch.tensor(
                np.repeat(X_cands_np, self.cand_t_grid_size, axis=0),
                dtype=dtype,
                device=device
            )
            T_cands_rep = torch.tensor(
                np.tile(t_grid, n_cands),
                dtype=dtype,
                device=device
            ).unsqueeze(-1)
            Q_flat = torch.cat([X_cands_rep, T_cands_rep], dim=1)

            num_targets = Z_plus.shape[0]
            num_q = Q_flat.shape[0]

            # 4. Compute Joint Posterior Covariance
            all_inputs_raw = torch.cat([Z_plus, Z_minus, Q_flat], dim=0)

            unique_inputs, inverse_idx = torch.unique(
                all_inputs_raw,
                dim=0,
                return_inverse=True
            )

            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.cholesky_jitter(1e-4):
                posterior = model(unique_inputs)
                unique_cov = posterior.covariance_matrix

            full_cov = unique_cov.index_select(0, inverse_idx).index_select(1, inverse_idx)

            start_z_minus = num_targets
            start_q = 2 * num_targets

            K_zp_q = full_cov[:num_targets, start_q:]
            K_zm_q = full_cov[start_z_minus:start_q, start_q:]
            K_qq = full_cov[start_q:, start_q:].clone()

            diag_ref = torch.diagonal(K_qq)
            diag_ref += self.extra_diag_jitter
            diag_ref.copy_(torch.clamp(diag_ref, min=self.extra_diag_jitter))

            K_g_q = K_zp_q - K_zm_q

            sigma_sq = 0.0
            if likelihood is not None and hasattr(likelihood, "noise"):
                noise_tensor = likelihood.noise.detach() if torch.is_tensor(likelihood.noise) else torch.tensor(likelihood.noise, device=device, dtype=dtype)
                sigma_sq = float(noise_tensor.item())
            curr_var_q = torch.diagonal(K_qq) + sigma_sq
            curr_var_q = torch.clamp(curr_var_q, min=self.variance_floor)

            # 5. Greedy Selection Loop
            selected_indices_in_avail = []
            selected_ts = []

            flat_to_cand_idx = torch.div(
                torch.arange(num_q, device=device),
                self.cand_t_grid_size,
                rounding_mode='floor'
            )
            cand_mask = torch.ones(n_cands, dtype=torch.bool, device=device)

            curr_K_g_q = K_g_q.clone()

            for _ in range(B):
                safe_var = torch.clamp(curr_var_q, min=self.variance_floor)
                numerator = torch.sum(curr_K_g_q.square(), dim=0)
                scores = numerator / safe_var

                valid_mask = cand_mask[flat_to_cand_idx]
                scores[~valid_mask] = -float('inf')

                best_flat_idx = torch.argmax(scores).item()
                if scores[best_flat_idx] == -float('inf'):
                    break

                best_cand_idx = flat_to_cand_idx[best_flat_idx].item()

                selected_indices_in_avail.append(cand_indices[best_cand_idx])
                t_val = t_grid[best_flat_idx % self.cand_t_grid_size]
                selected_ts.append(t_val)

                cand_mask[best_cand_idx] = False

                if len(selected_indices_in_avail) < B:
                    v_q = K_qq[:, best_flat_idx]
                    u_g = curr_K_g_q[:, best_flat_idx]

                    denom = torch.clamp(curr_var_q[best_flat_idx], min=self.variance_floor)

                    update_term_cov = torch.outer(u_g, v_q) / denom
                    curr_K_g_q = curr_K_g_q - update_term_cov
                    curr_K_g_q = self._nan_safe(curr_K_g_q)

                    update_term_var = v_q.square() / denom
                    curr_var_q = curr_var_q - update_term_var
                    curr_var_q = torch.clamp(curr_var_q, min=self.variance_floor)

                    K_qq = K_qq - torch.outer(v_q, v_q) / denom
                    diag_updated = torch.diagonal(K_qq)
                    diag_updated += self.extra_diag_jitter
                    diag_updated.copy_(torch.clamp(diag_updated, min=self.extra_diag_jitter))
                    curr_var_q = torch.clamp(diag_updated + sigma_sq, min=self.variance_floor)

                    K_qq = self._nan_safe(K_qq)
                    curr_var_q = self._nan_safe(curr_var_q, fill_value=self.variance_floor)

            return np.array(selected_indices_in_avail), np.array(selected_ts)

        finally:
            if base_has_mode_switch and was_training_base:
                base_model.train()
            if like_has_mode_switch and like_was_training:
                likelihood.train()
 
class GVALID_FOptSampler(BaseSampler):
    """
    GVALID F-Optimal Sampler (Joint Selection Version).
    
    Objective:
        Maximize Sum_over_Z [ Cov(f(z), f(q))^2 / Var(f(q)) ]
    """
    def __init__(
        self,
        rng,
        target_sample_size=32,
        cand_t_grid_size=10,      
        extra_diag_jitter=5e-4,
        variance_floor=1e-6,
        **kwargs
    ):
        super().__init__(rng, **kwargs)
        self.target_sample_size = target_sample_size
        self.cand_t_grid_size = cand_t_grid_size
        self.extra_diag_jitter = extra_diag_jitter
        self.variance_floor = variance_floor

    @staticmethod
    def _nan_safe(tensor, fill_value=0.0):
        return torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)

    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        if B <= 0: return np.array([], dtype=int), np.array([], dtype=float)

        # 0. Setup Device & Model Mode
        base_model = getattr(model, "model", model)
        likelihood = getattr(model, "likelihood", getattr(base_model, "likelihood", None))
        
        try:
            param = next(base_model.parameters())
            device = param.device
            dtype = param.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float64

        if hasattr(base_model, "eval"): base_model.eval()
        if likelihood and hasattr(likelihood, "eval"): likelihood.eval()

        # 1. Construct Target Set Z = {(x_i, t*_i)}
        n_targets = min(self.target_sample_size, len(X_pool))
        target_indices = self.rng.choice(len(X_pool), n_targets, replace=False)
        X_targets = X_pool[target_indices]
        
        X_targets_torch = torch.tensor(X_targets, dtype=dtype, device=device)
        
        _, t_star_targets = _torch_helper(
            X_targets, model, 'ucb', 
            t_grid_size=self.t_grid_size, 
            beta=0.0 
        )
        
        T_star_targets_torch = torch.tensor(t_star_targets, dtype=dtype, device=device).unsqueeze(-1)
        Z_inputs = torch.cat([X_targets_torch, T_star_targets_torch], dim=1)

        # 2. Construct Candidate Set Q
        n_cands = min(self.n_candidates, len(available_indices))
        if len(available_indices) > n_cands:
            cand_indices = self.rng.choice(available_indices, n_cands, replace=False)
        else:
            cand_indices = np.asarray(available_indices)
        
        X_cands_np = X_pool[cand_indices]
        t_grid = np.linspace(0, 1, self.cand_t_grid_size)
        
        X_cands_rep = torch.tensor(
            np.repeat(X_cands_np, self.cand_t_grid_size, axis=0),
            dtype=dtype, device=device
        )
        T_cands_rep = torch.tensor(
            np.tile(t_grid, n_cands),
            dtype=dtype, device=device
        ).unsqueeze(-1)
        
        Q_inputs = torch.cat([X_cands_rep, T_cands_rep], dim=1)
        
        num_targets = Z_inputs.shape[0]
        num_q = Q_inputs.shape[0]

        # 3. Compute Joint Posterior Covariance
        all_inputs_raw = torch.cat([Z_inputs, Q_inputs], dim=0)
        unique_inputs, inverse_idx = torch.unique(
            all_inputs_raw, dim=0, return_inverse=True
        )

        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(1e-4):
            posterior = model(unique_inputs)
            unique_cov = posterior.covariance_matrix
            
        full_cov = unique_cov.index_select(0, inverse_idx).index_select(1, inverse_idx)
        
        start_q = num_targets
        K_zq = full_cov[:num_targets, start_q:]
        K_qq = full_cov[start_q:, start_q:].clone() 
        
        sigma_sq = 0.0
        if likelihood is not None and hasattr(likelihood, "noise"):
            noise_val = likelihood.noise
            sigma_sq = float(noise_val.detach().item()) if torch.is_tensor(noise_val) else float(noise_val)
                
        curr_var_q = torch.diagonal(K_qq) + sigma_sq
        curr_var_q = torch.clamp(curr_var_q, min=self.variance_floor)
        
        curr_K_zq = K_zq.clone()

        # 4. Greedy Selection Loop
        selected_indices_in_avail = []
        selected_ts = []
        
        flat_to_cand_idx = torch.div(
            torch.arange(num_q, device=device),
            self.cand_t_grid_size,
            rounding_mode='floor'
        )
        cand_mask = torch.ones(n_cands, dtype=torch.bool, device=device)
        
        for _ in range(B):
            numerator = torch.sum(curr_K_zq.square(), dim=0)
            scores = numerator / curr_var_q 
            
            valid_mask = cand_mask[flat_to_cand_idx]
            scores[~valid_mask] = -float('inf')
            
            best_flat_idx = torch.argmax(scores).item()
            if not math.isfinite(float(scores[best_flat_idx])):
                break 
            
            best_cand_idx = flat_to_cand_idx[best_flat_idx].item()
            
            selected_indices_in_avail.append(cand_indices[best_cand_idx])
            t_val = t_grid[best_flat_idx % self.cand_t_grid_size]
            selected_ts.append(t_val)
            
            cand_mask[best_cand_idx] = False
            
            if len(selected_indices_in_avail) < B:
                v_q = K_qq[:, best_flat_idx]
                u_z = curr_K_zq[:, best_flat_idx]
                denom = curr_var_q[best_flat_idx]
                
                update_term_cov = torch.outer(u_z, v_q) / denom
                curr_K_zq = curr_K_zq - update_term_cov
                curr_K_zq = self._nan_safe(curr_K_zq)
                
                update_term_var = v_q.square() / denom
                curr_var_q = curr_var_q - update_term_var
                
                K_qq = K_qq - torch.outer(v_q, v_q) / denom
                
                diag_updated = torch.diagonal(K_qq)
                diag_updated += self.extra_diag_jitter
                diag_updated.copy_(torch.clamp(diag_updated, min=self.extra_diag_jitter))
                
                curr_var_q = torch.clamp(diag_updated + sigma_sq, min=self.variance_floor)
                K_qq = self._nan_safe(K_qq)

        return np.array(selected_indices_in_avail), np.array(selected_ts)
 
class GVALID_RandomX_Hybrid(BaseSampler):
    """
    Hybrid Sampler (Random X + GVALID T)
    """
    def __init__(
        self,
        rng,
        n_candidates=1000,
        gpu_batch_size=32,
        num_threads=1,
        GVALID_target_sample_size=64,
        GVALID_t_grid_size=10,
        diff_delta=0.01,
        min_fd_separation=0.05,
        fd_edge_eps=1e-4,
        extra_diag_jitter=5e-4,
        variance_floor=1e-6,
        **kwargs
    ):
        super().__init__(rng, num_threads=num_threads, 
                         n_candidates=n_candidates, gpu_batch_size=gpu_batch_size, 
                         **kwargs)
        
        self.target_sample_size = GVALID_target_sample_size
        self.GVALID_t_grid_size = GVALID_t_grid_size
        self.delta = diff_delta
        self.min_fd_separation = max(min_fd_separation, 1e-4)
        self.fd_edge_eps = fd_edge_eps
        self.extra_diag_jitter = extra_diag_jitter
        self.variance_floor = variance_floor

    def _build_fd_time_pairs(self, t_star_array):
        t_star = np.asarray(t_star_array, dtype=np.float64)
        half_span = max(self.delta, 0.5 * self.min_fd_separation)
        t_plus = t_star + half_span
        t_minus = t_star - half_span
        upper_limit = 1.0 - self.fd_edge_eps
        lower_limit = self.fd_edge_eps

        overflow = np.maximum(t_plus - upper_limit, 0.0)
        t_plus -= overflow; t_minus -= overflow
        underflow = np.maximum(lower_limit - t_minus, 0.0)
        t_plus += underflow; t_minus += underflow
        
        t_plus = np.clip(t_plus, lower_limit, upper_limit)
        t_minus = np.clip(t_minus, lower_limit, upper_limit)
        
        span = t_plus - t_minus
        valid_mask = span >= max(self.min_fd_separation * 0.5, 1e-3)
        return t_plus, t_minus, valid_mask

    @staticmethod
    def _nan_safe(tensor, fill_value=0.0):
        return torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)

    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        if B <= 0: return np.array([], dtype=int), np.array([], dtype=float)
        
        # Phase 1: Select X RANDOMLY
        if len(available_indices) > B:
            candidates = self.rng.choice(available_indices, B, replace=False)
        else:
            candidates = np.asarray(available_indices)
        
        selected_indices = candidates
        
        # Phase 2: Select T for the selected X using GVALID Logic
        base_model = getattr(model, "model", model)
        likelihood = getattr(model, "likelihood", getattr(base_model, "likelihood", None))
        try:
            param = next(base_model.parameters())
            device = param.device
            dtype = param.dtype
        except:
            device = torch.device("cpu")
            dtype = torch.float64

        base_model.eval()
        if likelihood: likelihood.eval()

        n_targets = min(self.target_sample_size, len(X_pool))
        target_indices = self.rng.choice(len(X_pool), n_targets, replace=False)
        X_targets = X_pool[target_indices]

        X_targets_torch = torch.tensor(X_targets, dtype=dtype, device=device)
        t_star_targets, _ = optimize_t_for_x_batch_torch(
            model, X_targets_torch, 'ucb', t_grid_size=101, beta=0.0
        )
        t_star_targets = t_star_targets.cpu().numpy()

        t_plus, t_minus, valid_mask = self._build_fd_time_pairs(t_star_targets)
        if not np.all(valid_mask):
            X_targets = X_targets[valid_mask]
            t_plus = t_plus[valid_mask]
            t_minus = t_minus[valid_mask]
            X_targets_torch = X_targets_torch[valid_mask]

        if len(X_targets) == 0:
            return selected_indices, np.full(len(selected_indices), 0.5)

        X_selected_np = X_pool[selected_indices]
        t_grid_cands = np.linspace(0, 1, self.GVALID_t_grid_size)
        
        X_query_rep = torch.tensor(
            np.repeat(X_selected_np, self.GVALID_t_grid_size, axis=0),
            dtype=dtype, device=device
        )
        T_query_rep = torch.tensor(
            np.tile(t_grid_cands, len(selected_indices)),
            dtype=dtype, device=device
        ).unsqueeze(-1)
        
        Q_flat = torch.cat([X_query_rep, T_query_rep], dim=1) 

        Z_plus = torch.cat([X_targets_torch, torch.tensor(t_plus, dtype=dtype, device=device).unsqueeze(-1)], dim=1)
        Z_minus = torch.cat([X_targets_torch, torch.tensor(t_minus, dtype=dtype, device=device).unsqueeze(-1)], dim=1)

        all_inputs = torch.cat([Z_plus, Z_minus, Q_flat], dim=0)
        unique_inputs, inverse_idx = torch.unique(all_inputs, dim=0, return_inverse=True)

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
            posterior = model(unique_inputs)
            unique_cov = posterior.covariance_matrix
        
        full_cov = unique_cov.index_select(0, inverse_idx).index_select(1, inverse_idx)

        num_targets = Z_plus.shape[0]
        start_q = 2 * num_targets
        
        K_zp_q = full_cov[:num_targets, start_q:]
        K_zm_q = full_cov[num_targets:start_q, start_q:]
        K_qq = full_cov[start_q:, start_q:].clone()

        K_g_q = K_zp_q - K_zm_q 
        
        sigma_sq = 0.0
        if likelihood is not None and hasattr(likelihood, "noise"):
             sigma_sq = float(likelihood.noise.item()) if torch.is_tensor(likelihood.noise) else float(likelihood.noise)
        
        curr_var_q = torch.diagonal(K_qq) + sigma_sq
        curr_var_q = torch.clamp(curr_var_q, min=self.variance_floor)
        curr_K_g_q = K_g_q.clone()

        final_selected_ts = []
        for i in range(len(selected_indices)):
            start_idx = i * self.GVALID_t_grid_size
            end_idx = start_idx + self.GVALID_t_grid_size
            
            numerator_slice = torch.sum(curr_K_g_q[:, start_idx:end_idx].square(), dim=0)
            var_slice = torch.clamp(curr_var_q[start_idx:end_idx], min=self.variance_floor)
            scores_slice = numerator_slice / var_slice
            
            best_local_idx = torch.argmax(scores_slice).item()
            best_global_idx = start_idx + best_local_idx
            
            selected_t = t_grid_cands[best_local_idx]
            final_selected_ts.append(selected_t)
            
            v_q = K_qq[:, best_global_idx]
            u_g = curr_K_g_q[:, best_global_idx]
            denom = torch.clamp(curr_var_q[best_global_idx], min=self.variance_floor)
            
            update_term_cov = torch.outer(u_g, v_q) / denom
            curr_K_g_q = curr_K_g_q - update_term_cov
            curr_K_g_q = self._nan_safe(curr_K_g_q)
            
            update_term_var = v_q.square() / denom
            curr_var_q = curr_var_q - update_term_var
            curr_var_q = torch.clamp(curr_var_q, min=self.variance_floor)
            
            K_qq = K_qq - torch.outer(v_q, v_q) / denom
            
            diag_updated = torch.diagonal(K_qq)
            diag_updated += self.extra_diag_jitter
            curr_var_q = torch.clamp(diag_updated + sigma_sq, min=self.variance_floor)
            
            K_qq = self._nan_safe(K_qq)
            curr_var_q = self._nan_safe(curr_var_q, fill_value=self.variance_floor)

        return selected_indices, np.array(final_selected_ts)

class GVALIDSampler_THEO(BaseSampler):
    """
    GVALID: Curvature-Aware Gradient Estimation Sampler.
    """
    def __init__(
        self,
        rng,
        target_sample_size=32,
        diff_delta=0.01,
        min_fd_separation=0.05,
        fd_edge_eps=1e-4,
        extra_diag_jitter=5e-4,
        variance_floor=1e-6,
        **kwargs,
    ):
        super().__init__(rng, **kwargs)
        self.target_sample_size = target_sample_size
        self.delta = diff_delta
        self.min_fd_separation = max(min_fd_separation, 1e-4)
        self.fd_edge_eps = fd_edge_eps
        self.extra_diag_jitter = extra_diag_jitter
        self.variance_floor = variance_floor
        self.cand_t_grid_size = 10


    def _build_fd_time_pairs(self, t_star_array):
        t_star = np.asarray(t_star_array, dtype=np.float64)
        half_span = max(self.delta, 0.5 * self.min_fd_separation)

        t_plus = t_star + half_span
        t_minus = t_star - half_span

        upper_limit = 1.0 - self.fd_edge_eps
        lower_limit = self.fd_edge_eps

        overflow = np.maximum(t_plus - upper_limit, 0.0)
        t_plus -= overflow
        t_minus -= overflow

        underflow = np.maximum(lower_limit - t_minus, 0.0)
        t_plus += underflow
        t_minus += underflow

        t_plus = np.clip(t_plus, lower_limit, upper_limit)
        t_minus = np.clip(t_minus, lower_limit, upper_limit)

        span = t_plus - t_minus
        min_span = max(self.min_fd_separation * 0.5, 1e-3)
        valid_mask = span >= min_span

        if not np.all(valid_mask):
            fallback_idx = np.nonzero(~valid_mask)[0]
            for idx in fallback_idx:
                center = t_star[idx]
                max_up = upper_limit - center
                max_down = center - lower_limit
                fallback_delta = min(self.delta, max(max_up, max_down))
                if fallback_delta <= 0.0:
                    fallback_delta = self.fd_edge_eps

                t_plus[idx] = np.clip(center + fallback_delta, lower_limit, upper_limit)
                t_minus[idx] = np.clip(center - fallback_delta, lower_limit, upper_limit)

                if np.isclose(t_plus[idx], t_minus[idx]):
                    t_plus[idx] = np.nextafter(t_plus[idx], 1.0)
                    t_minus[idx] = np.nextafter(t_minus[idx], 0.0)

            span = t_plus - t_minus
            valid_mask = span >= min_span

        return t_plus.astype(np.float64), t_minus.astype(np.float64), valid_mask


    @staticmethod
    def _nan_safe(tensor, fill_value=0.0):
        return torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)


    def select_batch(self, model, X_pool, available_indices, B, train_y=None, validation_context=None):
        B = min(B, len(available_indices))
        if B <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        base_model = getattr(model, "model", model)
        likelihood = getattr(model, "likelihood", getattr(base_model, "likelihood", None))

        def _infer_device_dtype():
            device = getattr(model, "device", None)
            dtype = getattr(model, "dtype", None)
            if device is None or dtype is None:
                try:
                    param = next(base_model.parameters())
                    if device is None:
                        device = param.device
                    if dtype is None:
                        dtype = param.dtype
                except StopIteration:
                    device = device or torch.device("cpu")
                    dtype = dtype or torch.float64
            return device, dtype

        device, dtype = _infer_device_dtype()

        was_training_base = getattr(base_model, "training", None)
        base_has_mode_switch = hasattr(base_model, "eval") and callable(getattr(base_model, "eval"))
        if base_has_mode_switch:
            base_model.eval()

        like_was_training = getattr(likelihood, "training", None) if likelihood is not None else None
        like_has_mode_switch = likelihood is not None and hasattr(likelihood, "eval") and callable(getattr(likelihood, "eval"))
        if like_has_mode_switch:
            likelihood.eval()

        try:
            # 1. Construct Target Set Z = {(x_i, t*_i)}
            n_targets = min(self.target_sample_size, len(X_pool))
            target_indices = self.rng.choice(len(X_pool), n_targets, replace=False)
            X_targets = X_pool[target_indices]

            _, t_star_targets = _torch_helper(
                X_targets,
                model,
                'ucb',
                t_grid_size=self.t_grid_size,
                beta=0.0
            )

            t_plus, t_minus, valid_mask = self._build_fd_time_pairs(t_star_targets)

            if not np.all(valid_mask):
                X_targets = X_targets[valid_mask]
                t_star_targets = t_star_targets[valid_mask]
                t_plus = t_plus[valid_mask]
                t_minus = t_minus[valid_mask]

            if X_targets.shape[0] == 0:
                raise RuntimeError("Failed to construct valid finite-difference targets.")

            # 2. Construct Candidate Set Q
            n_cands = min(self.n_candidates, len(available_indices))
            if len(available_indices) > n_cands:
                cand_indices = self.rng.choice(available_indices, n_cands, replace=False)
            else:
                cand_indices = np.asarray(available_indices)

            X_cands_np = X_pool[cand_indices]
            t_grid = np.linspace(0, 1, self.cand_t_grid_size)

            # 3. Prepare Tensors
            X_targets_torch = torch.tensor(X_targets, dtype=dtype, device=device)

            Z_plus = torch.cat(
                [
                    X_targets_torch,
                    torch.tensor(t_plus, dtype=dtype, device=device).unsqueeze(-1)
                ],
                dim=1
            )
            Z_minus = torch.cat(
                [
                    X_targets_torch,
                    torch.tensor(t_minus, dtype=dtype, device=device).unsqueeze(-1)
                ],
                dim=1
            )

            X_cands_rep = torch.tensor(
                np.repeat(X_cands_np, self.cand_t_grid_size, axis=0),
                dtype=dtype,
                device=device
            )
            T_cands_rep = torch.tensor(
                np.tile(t_grid, n_cands),
                dtype=dtype,
                device=device
            ).unsqueeze(-1)
            Q_flat = torch.cat([X_cands_rep, T_cands_rep], dim=1)

            num_targets = Z_plus.shape[0]
            num_q = Q_flat.shape[0]

            # 4. Compute Joint Posterior Covariance
            all_inputs_raw = torch.cat([Z_plus, Z_minus, Q_flat], dim=0)

            unique_inputs, inverse_idx = torch.unique(
                all_inputs_raw,
                dim=0,
                return_inverse=True
            )

            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.cholesky_jitter(1e-4):
                posterior = model(unique_inputs)
                unique_cov = posterior.covariance_matrix

            full_cov = unique_cov.index_select(0, inverse_idx).index_select(1, inverse_idx)

            # ==========================================================
            # Target Set Z Average Posterior Variance Calculation
            # ----------------------------------------------------------
            num_targets = Z_plus.shape[0]

            var_z_plus = torch.diagonal(full_cov[:num_targets, :num_targets])

            start_z_minus = num_targets
            var_z_minus = torch.diagonal(
                full_cov[start_z_minus : start_z_minus + num_targets,
                         start_z_minus : start_z_minus + num_targets]
            )

            avg_var_Z = 0.5 * (var_z_plus.mean() + var_z_minus.mean())
            target_avg_posterior_var = float(avg_var_Z.item())

            if validation_context is not None:
                validation_context['target_avg_posterior_var'] = target_avg_posterior_var
            # ==========================================================

            start_z_minus = num_targets
            start_q = 2 * num_targets

            K_zp_q = full_cov[:num_targets, start_q:]
            K_zm_q = full_cov[start_z_minus:start_q, start_q:]
            K_qq = full_cov[start_q:, start_q:].clone()

            diag_ref = torch.diagonal(K_qq)
            diag_ref += self.extra_diag_jitter
            diag_ref.copy_(torch.clamp(diag_ref, min=self.extra_diag_jitter))

            K_g_q = K_zp_q - K_zm_q

            sigma_sq = 0.0
            if likelihood is not None and hasattr(likelihood, "noise"):
                noise_tensor = likelihood.noise.detach() if torch.is_tensor(likelihood.noise) else torch.tensor(likelihood.noise, device=device, dtype=dtype)
                sigma_sq = float(noise_tensor.item())
            curr_var_q = torch.diagonal(K_qq) + sigma_sq
            curr_var_q = torch.clamp(curr_var_q, min=self.variance_floor)

            # 5. Greedy Selection Loop
            selected_indices_in_avail = []
            selected_ts = []

            flat_to_cand_idx = torch.div(
                torch.arange(num_q, device=device),
                self.cand_t_grid_size,
                rounding_mode='floor'
            )
            cand_mask = torch.ones(n_cands, dtype=torch.bool, device=device)

            curr_K_g_q = K_g_q.clone()

            for _ in range(B):
                safe_var = torch.clamp(curr_var_q, min=self.variance_floor)
                numerator = torch.sum(curr_K_g_q.square(), dim=0)
                scores = numerator / safe_var

                valid_mask = cand_mask[flat_to_cand_idx]
                scores[~valid_mask] = -float('inf')

                best_flat_idx = torch.argmax(scores).item()
                if scores[best_flat_idx] == -float('inf'):
                    break

                best_cand_idx = flat_to_cand_idx[best_flat_idx].item()

                selected_indices_in_avail.append(cand_indices[best_cand_idx])
                t_val = t_grid[best_flat_idx % self.cand_t_grid_size]
                selected_ts.append(t_val)

                cand_mask[best_cand_idx] = False

                if len(selected_indices_in_avail) < B:
                    v_q = K_qq[:, best_flat_idx]
                    u_g = curr_K_g_q[:, best_flat_idx]

                    denom = torch.clamp(curr_var_q[best_flat_idx], min=self.variance_floor)

                    update_term_cov = torch.outer(u_g, v_q) / denom
                    curr_K_g_q = curr_K_g_q - update_term_cov
                    curr_K_g_q = self._nan_safe(curr_K_g_q)

                    update_term_var = v_q.square() / denom
                    curr_var_q = curr_var_q - update_term_var
                    curr_var_q = torch.clamp(curr_var_q, min=self.variance_floor)

                    K_qq = K_qq - torch.outer(v_q, v_q) / denom
                    diag_updated = torch.diagonal(K_qq)
                    diag_updated += self.extra_diag_jitter
                    diag_updated.copy_(torch.clamp(diag_updated, min=self.extra_diag_jitter))
                    curr_var_q = torch.clamp(diag_updated + sigma_sq, min=self.variance_floor)

                    K_qq = self._nan_safe(K_qq)
                    curr_var_q = self._nan_safe(curr_var_q, fill_value=self.variance_floor)

            return np.array(selected_indices_in_avail), np.array(selected_ts)

        finally:
            if base_has_mode_switch and was_training_base:
                base_model.train()
            if like_has_mode_switch and like_was_training:
                likelihood.train()


SAMPLER_REGISTRY = {
    # Baselines
    "RAND": RandomSampler,
    "AL-Full": MaxVarFullSampler,
    
    # SOTA Baselines
    "GPUCB": GPUCBSampler,
    "TS": TSSampler,
    "CATS": SmoothedMeanSampler, 
    "SoftTopK": SoftTopKSampler,
    "EI": EISampler,
    "ABC3": ABC3Sampler,
    "PG": PolicyGradientSampler,
    
    # Methods
    "GVALID": GVALIDSampler,
    "GVALID_theo": GVALIDSampler_THEO,   
    "GVALID_FOpt": GVALID_FOptSampler,
    "GVALID_RandomX": GVALID_RandomX_Hybrid
}