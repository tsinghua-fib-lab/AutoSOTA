import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

class CoralAgent:
    def __init__(self, num_users, num_categories, item_to_cat_map, args):
        self.num_users = num_users
        self.num_cats = num_categories
        self.item_to_cat = item_to_cat_map
        
        # Args
        self.lambda_max = args.lambda_max
        self.kappa = args.kappa
        self.delta_conf = getattr(args, 'delta_conf', 0.1)
        self.rho = args.rho
        self.tau = getattr(args, 'tau', 3.5)
        
        # Constants
        self.epsilon = 1e-6
        self.Lambda_max_cap = 100.0

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Learnable Parameters
        # mu: Base intensity
        self.mu = np.ones((num_users, num_categories)) * 0.01 
        
        # User-Level Beta (Scalar per user)
        self.beta = np.ones((num_users, 1)) * 1.0
        
        # Alpha Matrix (Diagonal Initialization)
        self.alpha = np.zeros((num_users, num_categories, num_categories))
        default_alpha_val = 1
        for u in range(num_users):
            np.fill_diagonal(self.alpha[u], default_alpha_val)

        # Runtime State
        self.user_history = {u: [] for u in range(num_users)}
        self.is_fitted = np.zeros(num_users, dtype=bool)
        self.current_intensities = np.zeros((num_users, num_categories))

        # Stats
        self.N_t = np.zeros((num_users, num_categories))
        self.sum_sq_lambda = np.zeros((num_users, num_categories))
        self.sum_violations = np.zeros((num_users, num_categories))
        self.r_hat = np.zeros((num_users, num_categories))

        # Valid Category Mask
        self.valid_cats_mask = np.zeros(num_categories, dtype=bool)
        present_cats = set(item_to_cat_map.values())
        for c in present_cats:
            if c < num_categories:
                self.valid_cats_mask[c] = True
        
        print(f"[CoralAgent] Initialized.")

    # --- [NEW] Single-step update for closed-loop simulation ---
    def update_single_step(self, user_id, cat_id, rating=5, time_gap=1.0):
        """
        Update Agent state (Hawkes Intensity) based on simulated interaction results.
        Used in Active Simulation to let the Agent remember its recent recommendations.
        """
        # 1. Update Bandit statistics (Click count +1, update average reward)
        self.N_t[user_id, cat_id] += 1
        n = self.N_t[user_id, cat_id]
        # In simulation, assume accepting a recommendation is positive feedback
        self.r_hat[user_id, cat_id] += (1.0/n) * (rating - self.r_hat[user_id, cat_id])
        
        # 2. Hawkes: Time Decay
        # I(t) = (I(t-1) - mu) * exp(-beta * dt) + mu
        dt = max(0.01, time_gap)
        decay = np.exp(-self.beta[user_id] * dt)
        
        current = self.current_intensities[user_id]
        base_mu = self.mu[user_id]
        new_intensities = (current - base_mu) * decay + base_mu
        
        # 3. Hawkes: Excitation
        # Excitation occurs if rating is high enough (>= tau) or if it's a user click.
        # Here we assume all simulated clicks are valid excitations.
        # if rating >= self.tau: 
        excitation = self.alpha[user_id, :, cat_id]
        new_intensities += excitation
            
        # 4. Clipping and Saving
        self.current_intensities[user_id] = np.clip(new_intensities, 0.0, 1e6)
        
        # 5. Update Risk Bound statistics
        self.sum_sq_lambda[user_id, cat_id] += (self.current_intensities[user_id, cat_id] ** 2)
        
        # Return current saturation D_t for logging purposes
        return self.get_saturation_D(user_id, self.current_intensities[user_id])
        
    # -----------------------------------------------
    def get_intensities(self, user_id):
        return self.current_intensities[user_id]
    
    def reset_user(self, user_id):
        self.current_intensities[user_id] = np.copy(self.mu[user_id])
        self.N_t[user_id] = np.zeros_like(self.N_t[user_id])
        self.r_hat[user_id] = np.zeros_like(self.r_hat[user_id])
        self.sum_sq_lambda[user_id] = np.zeros_like(self.sum_sq_lambda[user_id])
        self.sum_violations[user_id] = np.zeros_like(self.sum_violations[user_id])
        self.user_history[user_id] = []

    def fit_user_hawkes(self, user_id, cat_seq, rate_seq, lr=0.01, epochs=50):
        if len(cat_seq) < 2: 
            self.is_fitted[user_id] = True
            return

        T = len(cat_seq)
        cats_t = torch.tensor(cat_seq, dtype=torch.long, device=self.device)
        rates_t = torch.tensor(rate_seq, dtype=torch.float32, device=self.device)
        
        original_mu = self.mu[user_id].copy()
        
        init_mu = torch.tensor(self.mu[user_id], dtype=torch.float32, device=self.device)
        init_beta = torch.tensor(self.beta[user_id], dtype=torch.float32, device=self.device)
        init_alpha = torch.tensor(self.alpha[user_id], dtype=torch.float32, device=self.device)
        
        raw_mu = nn.Parameter(torch.log(init_mu + 1e-6))
        raw_beta = nn.Parameter(torch.log(init_beta + 1e-6))
        raw_alpha = nn.Parameter(torch.log(init_alpha + 1e-6))
        
        optimizer = optim.Adam([raw_mu, raw_beta, raw_alpha], lr=lr)
        events = (rates_t >= self.tau).float()
        times = torch.arange(T, dtype=torch.float32, device=self.device)
        dt = times.unsqueeze(1) - times.unsqueeze(0)
        mask = (dt > 0).float()
        
        success = True
        for _ in range(epochs):
            optimizer.zero_grad()
            mu = torch.clamp(torch.exp(raw_mu), max=10.0) 
            beta = torch.clamp(torch.exp(raw_beta), min=0.01, max=10.0)
            alpha = torch.clamp(torch.exp(raw_alpha), max=2.0)
            
            c_t_expanded = cats_t.unsqueeze(1).repeat(1, T)
            c_k_expanded = cats_t.unsqueeze(0).repeat(T, 1)
            
            alpha_matrix = alpha[c_t_expanded, c_k_expanded]
            decay_matrix = torch.exp(-beta * dt) * mask
            
            past_influence = torch.sum(alpha_matrix * decay_matrix * events.unsqueeze(0), dim=1)
            
            lambda_t = mu[cats_t] + past_influence
            lambda_t = torch.clamp(lambda_t, min=1e-6, max=1e6)
            
            nll_term1 = -torch.sum(events * torch.log(lambda_t + 1e-9))
            beta_inv = 1.0 / (beta + 1e-9)
            total_alpha_per_source = torch.sum(alpha, dim=0)
            impact_per_source = total_alpha_per_source * beta_inv
            total_excitation = torch.sum(impact_per_source[cats_t] * events)
            
            total_mu = torch.sum(mu) * T
            nll_term2 = total_mu + total_excitation
            loss = nll_term1 + nll_term2
            
            if torch.isnan(loss) or torch.isinf(loss):
                success = False
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw_mu, raw_beta, raw_alpha], max_norm=1.0)
            optimizer.step()
            
        if success:
            with torch.no_grad():
                self.mu[user_id] = torch.clamp(torch.exp(raw_mu), max=10.0).detach().cpu().numpy()
                self.beta[user_id] = torch.clamp(torch.exp(raw_beta), min=0.01, max=10.0).detach().cpu().numpy()
                self.alpha[user_id] = torch.clamp(torch.exp(raw_alpha), max=2.0).detach().cpu().numpy()
                self.is_fitted[user_id] = True
        else:
            self.mu[user_id] = original_mu
            self.is_fitted[user_id] = False

    def reconstruct_history(self, user_id, items, cats, ratings):
        self.user_history[user_id] = []
        self.N_t[user_id] = 0
        self.r_hat[user_id] = 0
        self.sum_sq_lambda[user_id] = 0
        self.sum_violations[user_id] = 0
        
        intensities = np.copy(self.mu[user_id]) 
        user_beta_val = self.beta[user_id] 
        
        for t_idx, (it, cat, rate) in enumerate(zip(items, cats, ratings)):
            if it == 0: continue
            
            self.N_t[user_id, cat] += 1
            n = self.N_t[user_id, cat]
            reward = 1.0
            self.r_hat[user_id, cat] += (1.0/n) * (reward - self.r_hat[user_id, cat])
            
            decay_factor = np.exp(-user_beta_val)
            intensities = (intensities - self.mu[user_id]) * decay_factor + self.mu[user_id]
            
            if rate >= self.tau:
                intensities += self.alpha[user_id, :, cat] 
            
            intensities = np.clip(intensities, a_min=0.0, a_max=1e6)
            self.sum_sq_lambda[user_id, cat] += (intensities[cat] ** 2)
            
            D_t = self.get_saturation_D(user_id, intensities)
            if D_t > self.lambda_max:
                self.sum_violations[user_id, cat] += 1.0

        self.current_intensities[user_id] = intensities

    def get_saturation_D(self, user_id, intensities):
        excess = np.maximum(0, intensities - (self.rho * self.mu[user_id]))
        return np.sum(excess)

    def get_adaptive_penalty(self, D_t):
        if D_t >= self.lambda_max: 
            return self.Lambda_max_cap
        raw_penalty = 1.0 / max(self.epsilon, self.lambda_max - D_t)
        return min(self.Lambda_max_cap, raw_penalty)

    def get_risk_bound(self, user_id, intensities):
        N = np.maximum(1, self.N_t[user_id])
        
        # 1. Empirical Mean of Violations
        p_hat = self.sum_violations[user_id] / N
        
        # 2. Empirical Variance of Violations
        # For binary variables (0/1), variance = p * (1 - p)
        var_proxy = p_hat * (1.0 - p_hat)
        
        ln_part = 2 * np.log(2 / self.delta_conf)
        
        # Bernstein Bound Formula: p_hat + sqrt(2 * var * log / N) + 3 * log / N
        term1 = p_hat
        term2 = np.sqrt((2 * ln_part / N) * var_proxy)
        term3 = (3 * ln_part) / N
        
        return term1 + term2 + term3

    def get_policy_target_category(self, user_id, current_t, strategy='argmax', exclude_cat=None, return_scores=False):
        intensities = self.current_intensities[user_id]
        if np.any(np.isnan(intensities)):
            intensities = self.mu[user_id]
            
        D_t = self.get_saturation_D(user_id, intensities)
        Lambda_Dt = self.get_adaptive_penalty(D_t)
        Risk_bound = self.get_risk_bound(user_id, intensities)
        
        N = np.maximum(1, self.N_t[user_id])
        exploration = self.kappa * np.sqrt(np.log(max(2, current_t)) / N)
        
        # Base score calculation
        scores = self.r_hat[user_id] + exploration - (Lambda_Dt * Risk_bound)

        # --- Strict Constraints Logic (Stronger Intervention) ---
        if exclude_cat is not None and exclude_cat > -1:
            scores[exclude_cat] = -np.inf

        if hasattr(self, 'valid_cats_mask'):
            scores[~self.valid_cats_mask] = -np.inf

        if D_t > self.lambda_max:
            # 1. Identify all categories contributing to the excess
            current_excess = np.maximum(0, intensities - (self.rho * self.mu[user_id]))
            violating_cats_mask = current_excess > 0
            
            # 2. Check if there are any remaining safe categories
            available_mask = (scores > -np.inf) & (~violating_cats_mask)
            
            if np.any(available_mask):
                # If safe categories exist, strictly mask all violating categories
                scores[violating_cats_mask] = -np.inf
            else:
                # Emergency case: pick the best among "bad" options (minimum excess)
                scores = -current_excess 
                if exclude_cat is not None and exclude_cat > -1:
                    scores[exclude_cat] = -np.inf
                if hasattr(self, 'valid_cats_mask'):
                    scores[~self.valid_cats_mask] = -np.inf
        # --------------------------------------------------------

        if strategy == 'argmax':
            best_cat = np.argmax(scores)
        else:
            temp = 0.5
            valid_scores = scores.copy()
            valid_scores[valid_scores == -np.inf] = -1e9
            exp_s = np.exp((valid_scores - np.max(valid_scores)) / temp)
            probs = exp_s / np.sum(exp_s)
            best_cat = np.random.choice(len(probs), p=probs)
        
        if return_scores:
            return best_cat, D_t, scores
        return best_cat, D_t

    def save_params(self, path):
        state = {'mu': self.mu, 'beta': self.beta, 'alpha': self.alpha, 'is_fitted': self.is_fitted}
        torch.save(state, path)

    def load_params(self, path):
        print(f"Loading Hawkes parameters from {path}...")
        state = torch.load(path, weights_only=False)
        self.mu = state['mu']
        self.beta = state['beta']
        self.alpha = state['alpha']
        self.is_fitted = state['is_fitted']