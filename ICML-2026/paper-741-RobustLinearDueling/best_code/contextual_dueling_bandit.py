import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

class ContextualDuelingBanditEnv:
    def __init__(self, d=1, e=2, k=10, delay_model=None, corruption_model=None, phi_type='sinusoidal'):
        self.d = d
        self.e = e
        self.k = k
        self.delay_model = delay_model if delay_model is not None else StochasticGeometricDelay(0)
        self.corruption_model = corruption_model if corruption_model is not None else NoCorruption()
        self.phi_type = phi_type

        # True parameters
        self.theta_star = np.random.randn(d) * 0.1 # Small dependence on X
        self.zeta_star = np.random.randn(e) * 5.0  # Large dependence on Y

        # For Linear Mapping
        if self.phi_type == 'linear':
            # Create a fixed random matrix M: d -> e
            # If d != e, we map d -> e directly.
            # If d == e, just a square matrix.
            self.M = np.random.randn(d, e)

    def get_context(self):
        X = np.random.uniform(-np.pi, np.pi, size=(self.k, self.d))
        return X

    def get_true_post_context(self, X):
        """
        Generates Y based on X and self.phi_type.
        Ensures output shape is (K, e).
        """
        if self.phi_type == 'sinusoidal':
            raw_y = np.concatenate([np.cos(X), np.sin(X)], axis=1)
        
        elif self.phi_type == 'piecewise':
            # Y = X * (X > 0) + 0.5 * X * (X <= 0) (Leaky ReLU-like)
            raw_y = X * (X > 0) + 0.5 * X * (X <= 0)
            
        elif self.phi_type == 'linear':
            # Y = X @ M
            # X is (K, d), M is (d, e) -> (K, e)
            raw_y = X @ self.M
            
        elif self.phi_type == 'polynomial':
            # Y = [X^2, sqrt(|X|)]
            raw_y = np.concatenate([X**2, np.sqrt(np.abs(X))], axis=1)

        elif self.phi_type == 'interaction':
            # Y = [X_i * X_j] for i <= j (Upper triangular interaction terms)
            # Efficiently compute pairwise products
            K_batch, d_dim = X.shape
            interactions = []
            for i in range(d_dim):
                for j in range(i, d_dim):
                    interactions.append(X[:, i] * X[:, j])
            raw_y = np.stack(interactions, axis=1)

        elif self.phi_type == 'abs':
            # Y = |X|. Symmetric (Uncorrelated with X).
            raw_y = np.abs(X)

        elif self.phi_type == 'cosine':
            # Y = cos(X). Symmetric (Uncorrelated with X).
            raw_y = np.cos(X)
            
        else:
            raise ValueError(f"Unknown phi_type: {self.phi_type}")

        # Ensure Y has dimension e by tiling or slicing if raw_y doesn't match roughly
        # Special case for Linear: it matches perfectly if we used M(d, e).
        if self.phi_type == 'linear':
            return raw_y

        # For others, raw_y dimension depends on d (e.g. 2d for sin, d for piecewise, 2d for poly)
        # We need to map this to e.
        if raw_y.shape[1] >= self.e:
            return raw_y[:, :self.e]
        else:
            repeats = int(np.ceil(self.e / raw_y.shape[1]))
            filled = np.tile(raw_y, (1, repeats))
            return filled[:, :self.e]

    def get_utility(self, X, Y):
        u_x = X @ self.theta_star
        u_y = Y @ self.zeta_star
        return u_x + u_y

    def get_feedback(self, u_a, u_b):
        """
        Returns (outcome, delay) pair.

        Prioritized interference protocol (paper, Experimental Setup):
        the adversary favors strategic outcome corruption over delay. Subject to
        the corruption budget C, a corrupted sample yields IMMEDIATE falsified
        feedback (delay = 0), whereas delays affect ONLY uncorrupted outcomes.
        The two attacks therefore never co-occur on the same sample.
        """
        # Calculate True Outcome Prob
        prob = 1 / (1 + np.exp(-(u_a - u_b)))
        true_outcome = 1 if np.random.rand() < prob else 0

        # 1. Corruption first (it has priority over delay).
        final_outcome = self.corruption_model.corrupt(true_outcome, u_a, u_b)
        is_corrupted = (final_outcome != true_outcome)

        # 2. Delay applies ONLY to uncorrupted outcomes; corrupted feedback is
        #    delivered immediately.
        if is_corrupted:
            delay = 0
        else:
            delay = self.delay_model.get_delay(u_a, u_b)

        return final_outcome, delay

# --- Delay Models ---
class DelayModel:
    def get_delay(self, u_a=None, u_b=None):
        raise NotImplementedError

class StochasticGeometricDelay(DelayModel):
    def __init__(self, mean_delay):
        self.mean_delay = mean_delay
        
    def get_delay(self, u_a=None, u_b=None):
        if self.mean_delay <= 0:
            return 0
        p = 1.0 / (self.mean_delay + 1.0)
        return np.random.geometric(p) - 1

class StochasticGaussianDelay(DelayModel):
    def __init__(self, mean_delay, std_delay):
        self.mean_delay = mean_delay
        self.std_delay = std_delay
        
    def get_delay(self, u_a=None, u_b=None):
        # Generate delay from Gaussian(mean, std), clip to >= 0, and round to integer
        delay = np.random.normal(self.mean_delay, self.std_delay)
        return int(max(0, np.round(delay)))

class StrategicDelay(DelayModel):
    """
    Strategic Delay (Attack on Best Arm):
    - If the learner chooses good arms (high collective utility), delay the feedback.
    - If the learner chooses bad arms, give feedback instantly.
    This starves the learner of positive signal.
    """
    def __init__(self, budget, magnitude=100, threshold_val=0.0):
        self.budget = budget
        self.magnitude = magnitude
        self.threshold = threshold_val
        self.total_delayed = 0
        
    def get_delay(self, u_a, u_b):
        # Attack: If the arms are "good" (high utility), delay the feedback to starve the learner.
        # But respect the budget.
        if self.total_delayed >= self.budget:
            return 0
            
        val = max(u_a, u_b)
        if val > self.threshold:
            # Apply delay, but cap at remaining budget? 
            # Or just stop applying if budget exceeded?
            # Usually strict budget means sum(delay) <= budget.
            
            allowed = self.budget - self.total_delayed
            actual_delay = min(self.magnitude, allowed)
            
            if actual_delay > 0:
                self.total_delayed += actual_delay
                return actual_delay
        return 0

# --- Corruption Models ---
class CorruptionModel:
    def corrupt(self, outcome, u_a=None, u_b=None):
        raise NotImplementedError

class NoCorruption(CorruptionModel):
    def __init__(self):
        self.count = 0
        
    def corrupt(self, outcome, u_a=None, u_b=None):
        return outcome

class StrategicOutcomeCorruption(CorruptionModel):
    """
    Strategic Corruption (Best Arm Attack):
    Flips the outcome if it favors the better arm.
    Effectively tries to hide the superiority of the better arm.
    """
    def __init__(self, budget):
        self.budget = budget
        self.count = 0
        
    def corrupt(self, outcome, u_a, u_b):
        if self.count >= self.budget:
            return outcome
            
        # Check if outcome is "correct" (favors higher utility)
        # outcome=1 means a wins. outcome=0 means b wins.
        if (u_a > u_b and outcome == 1) or (u_b > u_a and outcome == 0):
            self.count += 1
            return 1 - outcome # Flip to wrong
            
        return outcome

class NeuralApproximator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralApproximator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DuelingGLMLearner:
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, C=1.0, Lambda=1.0, mu_tau=0.0, kappa=0.1):
        self.d = d
        self.e = e
        self.dim = d + e
        self.lambda_reg = lambda_reg
        self.ucb_alpha = alpha # For arm selection confidence interval (c_t in paper)
        self.ucb_alpha_init = alpha
        self.ucb_alpha_decay = 100.0
        self._round_counter = 0
        self.pretrain_rounds = 100  # random exploration before bandit loop
        
        # Robustness parameters
        self.C = C
        self.Lambda = Lambda
        self.mu_tau = mu_tau
        self.kappa = kappa # Conservative lower bound for derivative
        
        # Alpha from Theorem (Weighting parameter): alpha = sqrt(d) / (C + D)
        # with d = d_x + d_y (= self.dim) and robustness scale D = max(sqrt(Lambda), mu*tau),
        # matching the paper. (No extra 1/sqrt(kappa) factor.)
        D = max(np.sqrt(Lambda), mu_tau) if mu_tau > 0 else np.sqrt(Lambda)
        denom = C + D
        self.weight_alpha = np.sqrt(self.dim) / (denom + 1e-6)
        
        # V: Full history (for weighting)
        self.V = self.lambda_reg * np.eye(self.dim)
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.dim)
        
        # W: Observed history (for estimation/confidence)
        self.W = self.lambda_reg * np.eye(self.dim)
        self.W_inv = (1.0 / self.lambda_reg) * np.eye(self.dim)
        
        self.theta_hat = np.zeros(self.dim)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.f_t = NeuralApproximator(d, e).to(self.device)
        self.optimizer_f = optim.Adam(self.f_t.parameters(), lr=1e-3)
        self.loss_fn_f = nn.MSELoss()
        
        self.data_buffer_x = []
        self.data_buffer_y = []

        # Arrived-outcome history for the weighted-MLE estimator (paper eq. (Theta_t)).
        # Each entry is the observed complete-feature (delta_z, outcome, omega)
        # of an arrived sample.
        self.hist_dz = []
        self.hist_o = []
        self.hist_w = []

        # RCDP-UCB uses predicted post-serving features only for arm selection.
        # The robust weight and full-history geometry V are updated after the
        # selected post-serving contexts are observed, matching Algorithm 1.
        self.use_observed_feature_weighting = True

    def _solve_mle(self, max_iter=50, tol=1e-7):
        """Theta_t = argmin_Theta (lambda/2)||Theta||^2
                       - sum_{s in H_t} omega_s log g((-1)^{1-o_s} <Theta, dz_s>),
        i.e. the weighted regularized MLE / estimating equation of the paper.
        Solved by warm-started Newton (IRLS) over the full arrived history."""
        if not self.hist_dz:
            return
        Z = np.asarray(self.hist_dz)          # (n, dim)
        o = np.asarray(self.hist_o)           # (n,)
        w = np.asarray(self.hist_w)           # (n,)
        lam = self.lambda_reg
        theta = self.theta_hat.copy()         # warm start
        eye = np.eye(self.dim)
        for _ in range(max_iter):
            p = 1.0 / (1.0 + np.exp(-(Z @ theta)))      # g(<theta, dz>)
            grad = lam * theta + Z.T @ (w * (p - o))    # estimating equation
            if np.linalg.norm(grad) < tol:
                break
            s = w * p * (1.0 - p)
            H = lam * eye + (Z * s[:, None]).T @ Z       # weighted Hessian
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            theta = theta - step
        self.theta_hat = theta

    def train_approximator(self, epochs=5):
        if self.e == 0:
            return # No post-serving context to learn
            
        if not self.data_buffer_x:
            return
        
        X_tensor = torch.FloatTensor(np.array(self.data_buffer_x)).to(self.device)
        Y_tensor = torch.FloatTensor(np.array(self.data_buffer_y)).to(self.device)
        
        self.f_t.train()
        for _ in range(epochs):
            self.optimizer_f.zero_grad()
            outputs = self.f_t(X_tensor)
            loss = self.loss_fn_f(outputs, Y_tensor)
            loss.backward()
            self.optimizer_f.step()
            
    def predict_y(self, X):
        if self.e == 0:
            return np.zeros((len(X), 0))
            
        self.f_t.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            Y_pred = self.f_t(X_tensor).cpu().numpy()
        return Y_pred

    def get_features(self, X):
        Y_pred = self.predict_y(X)
        return np.concatenate([X, Y_pred], axis=1)

    def get_observed_delta(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        if self.e == 0:
            return X[a_idx] - X[b_idx]
        z_a = np.concatenate([X[a_idx], y_a_obs])
        z_b = np.concatenate([X[b_idx], y_b_obs])
        return z_a - z_b

    def record_observed_features(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        """
        Calculate omega_t and update V_t after observing the selected
        post-serving contexts. This is the paper's Step 3:
        omega_t = min(1, alpha / ||Delta z_t||_{V_{t-1}^{-1}}),
        V_t = V_{t-1} + kappa * omega_t * Delta z_t Delta z_t^T.
        """
        delta_z = self.get_observed_delta(X, a_idx, b_idx, y_a_obs, y_b_obs)
        norm_V = np.sqrt(delta_z @ self.V_inv @ delta_z)

        if norm_V < 1e-9:
            omega_t = 1.0
        else:
            omega_t = min(1.0, self.weight_alpha / norm_V)

        outer = np.outer(delta_z, delta_z)
        self.V += self.kappa * omega_t * outer

        v_vec = np.sqrt(self.kappa * omega_t) * delta_z
        Av = self.V_inv @ v_vec
        denom = 1 + np.dot(v_vec, Av)
        self.V_inv -= np.outer(Av, Av) / denom

        self.last_delta_z = delta_z
        self.last_omega_t = omega_t
        return delta_z, omega_t
    
    def select_arms(self, X, env=None):
        """
        PonLinRUCB Strategy with Adaptive Weighting.
        Returns: a_t, b_t, omega_t
        """
        # Pre-training phase: random arm selection to build initial (X,Y) buffer
        if self._round_counter < self.pretrain_rounds:
            self._round_counter += 1
            K = X.shape[0]
            a_t = np.random.randint(0, K)
            b_t = np.random.randint(0, K)
            while b_t == a_t:
                b_t = np.random.randint(0, K)
            return a_t, b_t, None
        Z = self.get_features(X) 
        
        # 1. Champion a_t (Greedy)
        utilities = Z @ self.theta_hat
        a_t = np.argmax(utilities)
        z_a = Z[a_t]
        
        # 2. Challenger b_t using Full History V
        Delta_Z_a = Z - z_a 
        mean_diff = Delta_Z_a @ self.theta_hat
        width_V = np.sqrt(np.sum((Delta_Z_a @ self.V_inv) * Delta_Z_a, axis=1))
        
        self._round_counter += 1
        alpha_t = self.ucb_alpha_init / np.sqrt(self._round_counter / self.ucb_alpha_decay + 1.0)
        scores_b = mean_diff + alpha_t * width_V
        b_t = np.argmax(scores_b)
        
        # omega_t and V_t are updated only after post-serving contexts are
        # observed in run_simulation().
        return a_t, b_t, None

    def observe_context(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        """
        Post-serving context is revealed IMMEDIATELY after the pair is served
        (only the preference outcome is delayed/corrupted). So the mapping
        approximator f_t is trained on (x, y) right away, independent of the
        feedback delay queue.
        """
        if self.e == 0:
            return
        self.data_buffer_x.append(X[a_idx])
        self.data_buffer_y.append(y_a_obs)
        self.data_buffer_x.append(X[b_idx])
        self.data_buffer_y.append(y_b_obs)
        self.train_approximator(epochs=2)

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t):
        # Only the preference-outcome estimation (W, theta) is gated by feedback
        # delay; the approximator was already trained in observe_context.
        # Use the queued delta_z when available so the estimate stays consistent
        # with the V update performed after post-serving context observation.
        delta_z = getattr(self, 'current_delta_z', None)
        if delta_z is None:
            z = self.get_features(X)
            delta_z = z[a_idx] - z[b_idx]

        # Weighted update for Observed History W (paper W̃_t, eq. (widetildeW));
        # kept for subclasses whose confidence width is built from W_inv.
        v_vec = np.sqrt(self.kappa * omega_t) * delta_z
        self.W += self.kappa * omega_t * np.outer(delta_z, delta_z)

        Aw = self.W_inv @ v_vec
        denom_w = 1 + np.dot(v_vec, Aw)
        self.W_inv -= np.outer(Aw, Aw) / denom_w

        # Theta_t: re-solve the weighted regularized MLE over ALL arrived
        # outcomes (paper eq. (Theta_t)/estimating equation), not a single
        # online step. delta_z/outcome/omega are the selection-time values.
        self.hist_dz.append(delta_z)
        self.hist_o.append(outcome)
        self.hist_w.append(omega_t)
        self._solve_mle()

class BaselineDuelingGLMLearner:
    """
    Baseline 1: X-only (Standard Dueling Contextual Bandit)
    Uses only pre-serving context X.
    """
    def __init__(self, d, lambda_reg=1.0, alpha=0.1):
        self.d = d
        self.dim = d 
        self.lambda_reg = lambda_reg
        self.alpha = alpha 
        
        self.theta_hat = np.zeros(self.dim)
        # Using W (Observed History) for Baselines as requested
        self.W = self.lambda_reg * np.eye(self.dim)
        self.W_inv = (1.0 / self.lambda_reg) * np.eye(self.dim)
        
    def select_arms(self, X, env=None):
        Z = X 
        utilities = Z @ self.theta_hat
        a_t = np.argmax(utilities)
        z_a = Z[a_t]
        
        Delta_Z = Z - z_a 
        mean_diff = Delta_Z @ self.theta_hat
        weighted_norm = np.sqrt(np.sum((Delta_Z @ self.W_inv) * Delta_Z, axis=1))
        
        ucb_scores = mean_diff + self.alpha * weighted_norm
        b_t = np.argmax(ucb_scores)

        self.last_delta_z = Z[a_t] - Z[b_t]
        # Return 1.0 for omega_t (not used)
        return a_t, b_t, 1.0

    def observe_context(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        # X-only baseline: no post-serving context / approximator to train.
        return

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        # Ignore y_a_obs, y_b_obs, omega_t
        Z = X
        delta_z = Z[a_idx] - Z[b_idx]
        
        mu_val = 1 / (1 + np.exp(- np.dot(self.theta_hat, delta_z)))
        
        outer = np.outer(delta_z, delta_z)
        self.W += outer
        
        Aw = self.W_inv @ delta_z
        denom = 1 + np.dot(delta_z, Aw)
        self.W_inv -= np.outer(Aw, Aw) / denom
        
        step = self.W_inv @ (delta_z * (outcome - mu_val))
        self.theta_hat += step

def run_simulation(env, learner, T, name="Learner"):
    # np.random.seed(42) # REMOVED to allow external seeding
    
    cumulative_regret = 0
    regrets = []
    feedback_queue = []

    if name:
        print(f"Starting simulation for {name}...")

    for t in range(T):
        X = env.get_context()
        
        a_idx, b_idx, omega_t = learner.select_arms(X, env=env)

        Y = env.get_true_post_context(X)
        y_a = Y[a_idx]
        y_b = Y[b_idx]

        # Post-serving context is observed IMMEDIATELY (not gated by feedback
        # delay): train the mapping approximator right after serving the pair.
        learner.observe_context(X, a_idx, b_idx, y_a, y_b)

        if getattr(learner, 'use_observed_feature_weighting', False):
            # RCDP-UCB: after observing post-serving contexts, calculate omega_t
            # and update V using the observed complete feature difference.
            delta_z, omega_t = learner.record_observed_features(X, a_idx, b_idx, y_a, y_b)
        else:
            # Baselines keep their own selection-time featurization/weighting.
            delta_z = getattr(learner, 'last_delta_z', None)
            if omega_t is None:
                omega_t = 1.0

        u_all = env.get_utility(X, Y)
        if hasattr(u_all, 'flatten'):
            u_all = u_all.flatten()

        u_a = u_all[a_idx]
        u_b = u_all[b_idx]

        # Unified feedback handling (Corruption vs Delay)
        outcome, delay = env.get_feedback(u_a, u_b)

        arrival_time = t + delay

        feedback_queue.append({
            'arrival_time': arrival_time,
            'X': X,
            'a_idx': a_idx,
            'b_idx': b_idx,
            'outcome': outcome,
            'y_a': y_a,
            'y_b': y_b,
            'omega_t': omega_t,
            'delta_z': delta_z
        })

        pending_removals = []
        for i, item in enumerate(feedback_queue):
            if item['arrival_time'] <= t:
                # Hand the stored selection-time delta_z to the learner so the
                # (delayed) preference update is consistent with selection.
                learner.current_delta_z = item['delta_z']
                learner.update(
                    item['X'],
                    item['a_idx'],
                    item['b_idx'],
                    item['outcome'],
                    item['y_a'],
                    item['y_b'],
                    item['omega_t']
                )
                learner.current_delta_z = None
                pending_removals.append(i)

        for i in sorted(pending_removals, reverse=True):
            del feedback_queue[i]

        k_star = np.argmax(u_all)
        u_star = u_all[k_star]
        # Dueling instantaneous regret r_t = 1/2[(u* - u_a) + (u* - u_b)] (paper Eq.)
        inst_regret = 0.5 * ((u_star - u_a) + (u_star - u_b))
        
        cumulative_regret += inst_regret
        regrets.append(cumulative_regret) 
        
        if name and (t+1) % 500 == 0:
            avg_omega = np.mean([item['omega_t'] for item in feedback_queue] + [1.0]) # approximate
            print(f"[{name}] Round {t+1}/{T}, Regret: {cumulative_regret:.2f}, Avg Omega: {avg_omega:.4f}")

    return regrets

# --- Advanced Robust Baselines ---

def joint_pair_ucb_select(Z, theta_hat, M_inv, beta):
    """Joint pair-UCB selection.

    Used by RCDB (Di et al. 2025, Algorithm 1, line 6) and by the single-layer
    MaxPairUCB reduction of VACDB (Di et al. 2024, Algorithm 1, line 8):
        (a_t, b_t) = argmax_{i,j} <Z_i + Z_j, theta> + beta * ||Z_i - Z_j||_{M^{-1}}.
    The pair is chosen jointly (not greedy-leader-then-challenger). M^{-1} is the
    design matrix over played pair-differences. O(K^2), vectorized via the Gram
    matrix identity ||Z_i - Z_j||^2 = G_ii + G_jj - 2 G_ij with G = Z M^{-1} Z^T.
    """
    util = Z @ theta_hat
    G = Z @ (M_inv @ Z.T)
    diagG = np.diag(G)
    d2 = diagG[:, None] + diagG[None, :] - 2.0 * G
    width = np.sqrt(np.maximum(d2, 0.0))
    score = util[:, None] + util[None, :] + beta * width
    i_t, j_t = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(i_t), int(j_t)


def maxinp_select(Z, theta_hat, V_inv, alpha):
    """Maximum-Informative-Pair selection (Saha 2021), regret-minimizing variant.

    Leader  i_t = argmax_k <Z_k, theta>.
    Challenger j_t = the most informative arm relative to i_t,
        argmax_{k in C} ||Z_k - Z_{i_t}||_{V^{-1}},  among the plausibly-optimal
        set C = { k != i_t : <Z_k, theta> + alpha*||Z_k||_{V^{-1}} >= <Z_{i_t}, theta> }.

    Restricting the most-informative search to plausible winners keeps the regret
    sublinear; the unrestricted "most informative pair over all arms" rule
    over-explores irrelevant arms and is linear in this contextual setting.
    """
    util = Z @ theta_hat
    i_t = int(np.argmax(util))
    self_w = np.sqrt(np.maximum(np.sum((Z @ V_inv) * Z, axis=1), 0.0))
    ucb = util + alpha * self_w
    cand = np.where(ucb >= util[i_t])[0]
    cand = cand[cand != i_t]
    if len(cand) == 0:                       # very confident: verify the next-best UCB arm
        ucb2 = ucb.copy(); ucb2[i_t] = -np.inf
        return i_t, int(np.argmax(ucb2))
    diff = Z[cand] - Z[i_t]
    unc = np.sqrt(np.maximum(np.sum((diff @ V_inv) * diff, axis=1), 0.0))
    return i_t, int(cand[int(np.argmax(unc))])


class RCDBLearner(BaselineDuelingGLMLearner):
    """
    RCDB (Robust Contextual Dueling Bandits)
    From "Nearly Optimal Algorithms for Contextual Dueling Bandits from Adversarial Feedback".
    
    Features:
    1. Weighted ONS/MLE update based on uncertainty: w_t = min(1, alpha / ||z_{i,j}||_{V^{-1}})
    2. Exploration Bonus: beta * ||z_{i,j}||_{V^{-1}}
    
    We adapt a simplified version compatible with the current GLM structure:
    - Update uses weights.
    - Selection uses UCB with beta scaling.
    """
    def __init__(self, d, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, lambda_reg, alpha)
        self.rcdb_alpha = rcdb_alpha
        self.rcdb_beta = rcdb_beta
        self.kappa = kappa
        
    def select_arms(self, X, env=None):
        # Faithful RCDB joint pair-UCB (Di et al. 2025, Alg. 1 line 6).
        i_t, j_t = joint_pair_ucb_select(X, self.theta_hat, self.W_inv, self.rcdb_beta)
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        # Weight Calculation
        Z = X
        delta_z = Z[a_idx] - Z[b_idx]
        norm_val = np.sqrt(delta_z @ self.W_inv @ delta_z)

        # w_t = min(1, alpha / norm)   (RCDB Eq. 4.3)
        if norm_val < 1e-9:
            w_t = 1.0
        else:
            w_t = min(1.0, self.rcdb_alpha / norm_val)

        # Weighted online MLE step.
        mu_val = 1 / (1 + np.exp(- np.dot(self.theta_hat, delta_z)))

        # Sigma_t = lambda I + sum_i w_i (phi_a - phi_b)(phi_a - phi_b)^T
        # (RCDB's design matrix has NO kappa scaling; kappa enters only via alpha/beta).
        w_delta_z = np.sqrt(w_t) * delta_z

        outer = np.outer(w_delta_z, w_delta_z)
        self.W += outer

        Av = self.W_inv @ w_delta_z
        denom = 1 + np.dot(w_delta_z, Av)
        self.W_inv -= np.outer(Av, Av) / denom

        # Gradient Step weighted
        step = self.W_inv @ (delta_z * w_t * (outcome - mu_val))
        self.theta_hat += step


class MaxInPLearner(BaselineDuelingGLMLearner):
    """
    MaxInP (Maximum-Informative-Pair, Saha 2021, Algorithm 1).
    Faithful selection: pick the most uncertain pair ||Z_i - Z_j||_{V^{-1}} among
    the "promising" pairs whose optimistic relative score is positive
    (<Z_i - Z_j, theta> + eta*||Z_i - Z_j||_{V^{-1}} > 0). The exploration
    constant eta maps to self.alpha. (Saha's plain MLE + initial exploration
    phase is replaced by the regularized estimator inherited from the base class.)
    """
    def select_arms(self, X, env=None):
        i_t, j_t = maxinp_select(X, self.theta_hat, self.W_inv, self.alpha)
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0


class MaxPairUCBLearner(BaselineDuelingGLMLearner):
    """
    MaxPairUCB (single-layer pair-UCB reduction of VACDB, Di et al. 2024).
    Faithful selection: jointly maximize the pair-UCB
        <Z_i + Z_j, theta> + alpha * ||Z_i - Z_j||_{W^{-1}}.
    Update is the standard (unweighted) online MLE step (no variance weighting /
    multi-layer elimination of the full VACDB algorithm).
    """
    def select_arms(self, X, env=None):
        i_t, j_t = joint_pair_ucb_select(X, self.theta_hat, self.W_inv, self.alpha)
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0


def colstim_select_pair(Z, theta_hat, W_inv, c1, t,
                        perturb='gumbel', c_thresh=5.0, tau=0, coupling_const=None):
    """Faithful COLSTIM arm-pair selection (Bengs et al. 2022, Algorithm 1).

    M^{-1} = W_inv is the design matrix over the played pair-differences
    z_{i_s,j_s} (M_{t+1} = M_t + z z^T, line 17).

    First arm  (line 14, perturbed leader / follow-the-perturbed-leader):
        i_t = argmax_i  <x_i, theta> + eps_i * ||x_i||_{M^{-1}},
    where the eps_i are perturbations from the comparison-imitating
    distribution G, thresholded to [-c_thresh, c_thresh] (line 13). We default
    to Gumbel noise, whose argmax reproduces the Bradley-Terry-Luce (logistic)
    choice used as the link g here; pass perturb='gaussian' to imitate the
    Thurstone-Mosteller model instead.

    Second arm (line 15, optimistic toughest competitor of i_t):
        j_t = argmax_i  <x_i - x_{i_t}, theta> + c1 * ||x_i - x_{i_t}||_{M^{-1}}.

    Coupling (lines 7-12): with probability p_t the per-arm perturbations are
    independent (exploration), otherwise a single shared perturbation is used.
    Following Thm. 3.2, p_t decays as ~1/sqrt(t - tau); coupling_const=None
    keeps p_t = 1 (always-independent perturbations).
    """
    n = Z.shape[0]
    util = Z @ theta_hat
    widths = np.sqrt(np.maximum(np.sum((Z @ W_inv) * Z, axis=1), 0.0))  # ||x_i||_{M^{-1}}

    # Coupling: B_t ~ Ber(p_t).
    if coupling_const is None:
        p_t = 1.0
    else:
        p_t = min(1.0, coupling_const / np.sqrt(max(1, t - tau)))
    independent = (np.random.rand() < p_t)

    # Perturbations from G (Gumbel imitates BTL; Gaussian imitates Thurstone-Mosteller).
    if perturb == 'gaussian':
        draw = lambda size: np.random.randn(size)
    else:
        draw = lambda size: np.random.gumbel(0.0, 1.0, size=size)
    eps = draw(n) if independent else np.full(n, draw(1)[0])
    eps = np.clip(eps, -c_thresh, c_thresh)

    # First arm: perturbed leader.
    i_t = int(np.argmax(util + eps * widths))

    # Second arm: optimistic challenger relative to i_t (exclude i_t itself).
    diff = Z - Z[i_t]                                   # x_i - x_{i_t}
    mean_diff = diff @ theta_hat
    w_diff = np.sqrt(np.maximum(np.sum((diff @ W_inv) * diff, axis=1), 0.0))
    chall = mean_diff + c1 * w_diff
    chall[i_t] = -np.inf
    j_t = int(np.argmax(chall))
    return i_t, j_t


class ColSTIMLearner(BaselineDuelingGLMLearner):
    """
    COLSTIM (Contextual Linear Stochastic Transitivity Imitator, Bengs et al. 2022).
    Faithful Algorithm-1 selection: perturbed-leader first arm (Gumbel noise
    imitating the BTL link) + optimistic-challenger second arm. The parameter
    estimate uses the online weighted-MLE step (explicitly sanctioned in the
    paper, Sec. 3.2.3), inherited from BaselineDuelingGLMLearner.update.
    """
    def __init__(self, d, lambda_reg=1.0, alpha=0.1, perturb='gumbel',
                 c_thresh=5.0, coupling_const=None, tau=0):
        super().__init__(d, lambda_reg, alpha)
        self.perturb = perturb
        self.c_thresh = c_thresh
        self.coupling_const = coupling_const
        self.tau = tau
        self._t = 0

    def select_arms(self, X, env=None):
        self._t += 1
        i_t, j_t = colstim_select_pair(
            X, self.theta_hat, self.W_inv, c1=self.alpha, t=self._t,
            perturb=self.perturb, c_thresh=self.c_thresh,
            tau=self.tau, coupling_const=self.coupling_const)
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0

# --- Post-serving baselines (X + learned Y predictor) -----------------------
# Each baseline keeps its own faithful selection rule but runs it on the
# complete feature Z = [X, f_t(X)], where f_t is the shared neural predictor
# inherited from DuelingGLMLearner. Parameter estimation is the online step.

class RCDBPostServingLearner(DuelingGLMLearner):
    """RCDB (Di et al. 2025) with the learned post-serving predictor."""
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, e, lambda_reg, alpha, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.rcdb_alpha = rcdb_alpha
        self.rcdb_beta = rcdb_beta

    def select_arms(self, X, env=None):
        Z = self.get_features(X)
        i_t, j_t = joint_pair_ucb_select(Z, self.theta_hat, self.W_inv, self.rcdb_beta)
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        delta_z = getattr(self, 'current_delta_z', None)
        if delta_z is None:
            Z = self.get_features(X)
            delta_z = Z[a_idx] - Z[b_idx]
        norm_val = np.sqrt(delta_z @ self.W_inv @ delta_z)
        w_t = 1.0 if norm_val < 1e-9 else min(1.0, self.rcdb_alpha / norm_val)
        self.W += np.outer(delta_z, delta_z) * w_t
        Aw = self.W_inv @ delta_z
        denom = 1 + np.dot(delta_z, Aw) * w_t
        self.W_inv -= (np.outer(Aw, Aw) * w_t) / denom
        mu_val = 1 / (1 + np.exp(- np.dot(self.theta_hat, delta_z)))
        self.theta_hat += self.W_inv @ (delta_z * (outcome - mu_val)) * w_t


class _UnweightedPSUpdate(DuelingGLMLearner):
    """Shared unweighted online-MLE update on Z = [X, f(X)] for the
    information-/UCB-based post-serving baselines (ColSTIM, MaxInP, MaxPair)."""
    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        delta_z = getattr(self, 'current_delta_z', None)
        if delta_z is None:
            Z = self.get_features(X)
            delta_z = Z[a_idx] - Z[b_idx]
        self.W += np.outer(delta_z, delta_z) * omega_t
        Aw = self.W_inv @ delta_z
        denom = 1 + np.dot(delta_z, Aw) * omega_t
        self.W_inv -= (np.outer(Aw, Aw) * omega_t) / denom
        mu_val = 1 / (1 + np.exp(- np.dot(self.theta_hat, delta_z)))
        self.theta_hat += self.W_inv @ (delta_z * (outcome - mu_val)) * omega_t


class ColSTIMPostServingLearner(_UnweightedPSUpdate):
    """COLSTIM (Bengs et al. 2022) with the learned post-serving predictor."""
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=0.1,
                 perturb='gumbel', c_thresh=5.0, coupling_const=None, tau=0):
        super().__init__(d, e, lambda_reg, alpha, C=C, Lambda=Lambda, mu_tau=mu_tau, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.perturb = perturb
        self.c_thresh = c_thresh
        self.coupling_const = coupling_const
        self.tau = tau
        self._t = 0

    def select_arms(self, X, env=None):
        self._t += 1
        Z = self.get_features(X)
        i_t, j_t = colstim_select_pair(Z, self.theta_hat, self.W_inv, c1=self.ucb_alpha, t=self._t,
                                       perturb=self.perturb, c_thresh=self.c_thresh,
                                       tau=self.tau, coupling_const=self.coupling_const)
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0


class MaxPairUCBPostServingLearner(_UnweightedPSUpdate):
    """MaxPairUCB (Di et al. 2024) with the learned post-serving predictor."""
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, e, lambda_reg, alpha, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.rcdb_beta = rcdb_beta

    def select_arms(self, X, env=None):
        Z = self.get_features(X)
        i_t, j_t = joint_pair_ucb_select(Z, self.theta_hat, self.W_inv, self.rcdb_beta)
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0


class MaxInPPostServingLearner(_UnweightedPSUpdate):
    """MaxInP (Saha 2021) with the learned post-serving predictor."""
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, e, lambda_reg, alpha, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.rcdb_beta = rcdb_beta

    def select_arms(self, X, env=None):
        Z = self.get_features(X)
        i_t, j_t = maxinp_select(Z, self.theta_hat, self.W_inv, self.rcdb_beta)
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0
