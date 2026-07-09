import numpy as np
import core.functional as F

class ConformalPredictor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.s = 0.0 # radius
        
    def update(self, S_new):
        raise NotImplementedError
    
    def predict(self):
        return self.s

class OnlineGradientDescent(ConformalPredictor):
    """Online Gradient Descent Predictor."""
    def __init__(self, alpha, lr):
        super().__init__(alpha)
        self.lr = lr
        self.s = 0.0
            
    def update(self, S_new):
        covered = (S_new <= self.s)
        g = self.alpha if covered else -(1.0 - self.alpha)
        self.s = self.s - self.lr * g


class ScaleFreeOnlineGradientDescent(ConformalPredictor):
    """Scale-Free OGD Predictor."""
    def __init__(self, alpha, lr):
        super().__init__(alpha)
        self.lr = lr
        self.s = 0.0
        self.sum_grad_sq = 0.0
        
    def update(self, S_new):
        covered = (S_new <= self.s)
        g = self.alpha if covered else -(1.0 - self.alpha)
        self.sum_grad_sq += g**2
        denom = np.sqrt(self.sum_grad_sq) if self.sum_grad_sq > 0 else 1.0
        self.s = self.s - self.lr * (g / denom)

class KrichevskyTrofimov(ConformalPredictor):
    """
    Krichevsky-Trofimov betting-based approach.
    
    Parameter-free approach using coin betting.
    """
    def __init__(self, alpha=0.1, initial_wealth=1.0):
        super().__init__(alpha)
        self.t = 1 # time step
        self.wealth = initial_wealth # wealth
        self.beta = 0.0 # betting fraction 
        self.s = 0.0 # radius (and bet amount)

    def update(self, S_new):
        # 1. Compute subgradient g_t of pinball loss at s_t
        # Loss l(s) for quantile (1 - alpha).
        # g_t = -(1 - alpha) if S_t > s_t (uncovered)
        # g_t = alpha        if S_t <= s_t (covered)
        covered = (S_new <= self.s)
        g = self.alpha if covered else -(1.0 - self.alpha)
        
        # 2. Update wealth
        # W_{t+1} = W_t + (-g_t) * s_t
        self.wealth = self.wealth - g * self.s
        
        # 3. Update betting fraction
        # \beta_{t+1} = t / (t+1) * \beta_t - 1 / (t+1) * g_t
        self.beta = (self.t / (self.t + 1)) * self.beta - (1 / (self.t + 1)) * g
        
        # 4. Set next prediction s_{t+1}
        self.s = self.beta * self.wealth
        
        # 5. Increment time step
        self.t += 1

class UniversalPortfolio(ConformalPredictor):
    """
    Universal Portfolio-based approach.
    
    Optimal when outcomes of the coin are asymmetric.
    """
    def __init__(self, alpha=0.1, initial_wealth=1.0, mixture='dirichlet',
                 finite_sample_correction=0.015, gamma_aci=0.0,
                 decay=1.0, warmup_init=False, gamma_eq=0.0, eq_scale=1.0):
        super().__init__(alpha)
        self.t = 1 # time step
        self.wealth = initial_wealth # wealth
        self.x = 0.5 # fraction invested on stock 1 (market gain = (-g + \alpha) / \alpha). Always starts at uniform
        self.alpha_original = alpha  # stored for ACI-style adaptive correction
        self.finite_sample_correction = finite_sample_correction
        self.alpha -= self.finite_sample_correction # finite-sample correction
        self.alpha_correction = self.finite_sample_correction  # adaptive correction term
        self.gamma_aci = gamma_aci  # ACI learning rate (0 = disabled)
        self.aci_step_count = 0  # step counter for decaying learning rate
        self.decay = decay  # exponential decay factor (1.0 = uniform, <1.0 = recency-weighted)
        self.eff_t = 0.0  # effective time (decay-accumulated)
        self.eff_covered = 0.0  # effective covered count (decay-accumulated)
        self.warmup_init = warmup_init  # if True, runner initializes s from warmup quantile
        self.gamma_eq = gamma_eq  # ECI blend factor (0 = pure binary, 1 = pure continuous)
        self.eq_scale = eq_scale  # ECI sigmoid temperature
        self.beta =  0.5 * (1 / self.alpha - 1 / (1 - self.alpha)) # betting fraction
        self.s = 0.0 # radius (and bet amount)
        self.covered_count = 0 # number of times covered
        self.mixture = mixture # 'dirichlet' or 'uniform'

    def update(self, S_new):
        # 1. Compute subgradient g_t of pinball loss at s_t
        covered = (S_new <= self.s)
        g = self.alpha if covered else -(1.0 - self.alpha)

        # ECI-style continuous feedback: blend binary subgradient with sigmoid-smoothed version
        # Smooth coverage indicator provides gradient information proportional to error magnitude
        if self.gamma_eq > 0:
            soft_covered = 1.0 / (1.0 + np.exp(-(self.s - S_new) / self.eq_scale))
            g_soft = soft_covered - (1.0 - self.alpha)  # continuous subgradient
            g = (1.0 - self.gamma_eq) * g + self.gamma_eq * g_soft

        self.covered_count += int(covered)

        # Recency-weighted effective counts (exponential decay)
        # eff_t = decay * eff_t + 1.0
        # eff_covered = decay * eff_covered + float(covered)
        self.eff_t = self.decay * self.eff_t + 1.0
        self.eff_covered = self.decay * self.eff_covered + float(covered)

        # ACI-style adaptive alpha correction (online update)
        if self.gamma_aci > 0:
            err = float(not covered)
            decay_step = np.sqrt(self.aci_step_count + 1.0)
            self.alpha_correction += self.gamma_aci * (err - self.alpha_original) / decay_step
            self.alpha_correction = np.clip(self.alpha_correction, 0.0, self.alpha_original - 0.005)
            self.alpha = self.alpha_original - self.alpha_correction
            self.aci_step_count += 1

        # 2. Update wealth
        # W_{t+1} = W_t + (-g_t) * s_t
        self.wealth = self.wealth - g * self.s

        # 3. Update x (fraction on stock 1) using effective (decay-weighted) counts
        # with Dirichlet(0.5, 0.5) prior:
        # t_eff and covered_eff replace raw counts for recency weighting
        if self.decay < 1.0:
            # Use effective (decay-weighted) counts for recency-weighted UP
            if self.mixture == 'dirichlet':
                self.x = (self.eff_t - self.eff_covered + 0.5) / (self.eff_t + 1.0)
            elif self.mixture == 'uniform':
                self.x = (self.eff_t - self.eff_covered + 1.0) / (self.eff_t + 2.0)
        else:
            # Standard uniform weighting (baseline)
            if self.mixture == 'dirichlet':
                self.x = (self.t - self.covered_count + 0.5) / (self.t + 1)
            elif self.mixture == 'uniform':
                self.x = (self.t - self.covered_count + 1) / (self.t + 2)

        # 4. Update betting fraction
        # \beta_{t+1} = -1 / (1 - alpha) + x * (1 / alpha + 1 / (1 - alpha))
        self.beta = - 1 / (1 - self.alpha) + self.x * (1 / self.alpha +  1 / (1 - self.alpha))

        # 4. Set next prediction s_{t+1}
        self.s = self.beta * self.wealth

        # 5. Increment time step
        self.t += 1
        
class AdaptiveConformalInference(ConformalPredictor):
    """
    Adaptive Conformal Inference (ACI).
    
    Corresponds to 'ConfidenceLevelOnlineGradientDescent'. 
    It updates the target miscoverage level (current_alpha) via OGD 
    and computes the radius s using the quantile of historical scores.
    """
    def __init__(self, alpha=0.1, lr=0.01):
        super().__init__(alpha)
        self.lr = lr
        self.current_alpha = alpha # alpha_t
        self.scores = [] # Buffer to store conformity scores for quantile computation
        self.s = 0.0
        
    def update(self, S_new):
        # 1. Compute subgradient g_t of pinball loss at s_t
        # Loss l(s) for quantile (1 - alpha).
        # g_t = -(1 - alpha) if S_t > s_t (uncovered)
        # g_t = alpha        if S_t <= s_t (covered)
        beta = 1 - F.compute_empirical_cdf(self.scores, S_new)
        covered = (self.current_alpha <= beta)
        
        g = self.alpha if covered else -(1.0 - self.alpha)
        
        # 2. Update the alpha_t parameter using Gradient Descent
        # Update rule: alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
        # Note: 
        #   If err=1 (miss), gradient is (alpha - 1) < 0 -> alpha_t decreases -> target coverage increases -> s increases.
        #   If err=0 (hit), gradient is alpha > 0      -> alpha_t increases -> target coverage decreases -> s decreases.
        self.current_alpha += self.lr * g
 
        # 3. Update the history of conformity scores (D_t)
        self.scores.append(S_new)

        # 4. Compute the new radius s based on the updated alpha_t
        # We need the (1 - alpha_t) quantile of the empirical distribution 
        # q = np.clip(1.0 - self.current_alpha, 0.0, 1.0) # We were being too nice here
        if self.current_alpha < 0:
            self.s = np.inf
        elif self.current_alpha > 1:
            self.s = 0.0
        else:
            q = 1.0 - self.current_alpha
            # Defaulted to linear 
            self.s = np.quantile(self.scores, q)

class DynamicallyTunedAdaptiveConformalInference(ConformalPredictor):
    """
    Dynamically-tuned Adaptive Conformal Inference (DtACI).
    
    Implements Algorithm 2 from Gibbs & Candès (2021/2024).
    Uses a mix of experts (running ACI with different learning rates) 
    and aggregates them using a re-weighted average based on the pinball loss.
    """
    def __init__(self, alpha=0.1, lrs=None, sigma=0.001, eta=np.e):
        """
        Args:
            alpha (float): Target miscoverage rate.
            lrs (list): List of step-sizes (learning rates) for the experts.
            sigma (float): Mixing parameter (expert fixed share).
            eta (float): Initial learning rate for the expert aggregation.
        """
        super().__init__(alpha)
        
        # Default gammas from the paper/R code if not provided
        if lrs is None:
            self.lrs = np.array([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128])
        else:
            self.lrs = np.array(lrs)
            
        self.d = len(self.lrs) # Number of experts
        self.sigma = sigma
        self.eta = eta

        # Initialize Experts (all start at target alpha)
        # alpha_t^i in the paper
        self.experts = [AdaptiveConformalInference(alpha=alpha, lr=lr) for lr in self.lrs]
        
        # Initialize Expert Weights
        # We maintain normalized weights summing to 1
        self.weights = np.ones(self.d) / self.d  
        # self.losses = []       # Sequence of weighted losses l(beta_t, alpha_t)
        self.s = 0.0           # Current Radius
        # self.scores = []       # Buffer to store conformity scores for quantile computation
        # Current aggregated alpha (output of the algorithm)
        self.current_alpha = alpha

    def update(self, S_new):
        # 1. Compute the weighted loss for the current step
        beta = 1 - F.compute_empirical_cdf(self.experts[0].scores, S_new)
        losses = np.array([F.pinball_loss(beta - expert.current_alpha, self.alpha) for expert in self.experts])
        # 2. Update expert weights using Exponentiated Gradient update
        # w_{t+1}^i \propto w_t^i * exp(-eta * l(beta_t, alpha_t^i))
        new_weights = self.weights * np.exp(-self.eta * losses)
        new_weights_sum = np.sum(new_weights)
        if new_weights_sum > 0:
            new_weights /= new_weights_sum
        
        # 3. Apply fixed-share mixing
        # w_{t+1}^i = (1 - sigma) * w_{t+1}^i + sigma / d
        self.weights = (1 - self.sigma) * new_weights + (self.sigma / self.d)
        
        # 4. Compute the aggregated alpha_t
        self.current_alpha = np.dot(self.weights, np.array([expert.current_alpha for expert in self.experts]))
        
        # 5. Update each expert with the new score
        for expert in self.experts:
            expert.update(S_new)
            
        # 6. Compute the new radius s based on the aggregated alpha_t
        if self.current_alpha < 0:
            self.s = np.inf
        elif self.current_alpha > 1:
            self.s = 0.0
        else:
            q = 1.0 - self.current_alpha
            # Defaulted to linear 
            self.s = np.quantile(self.experts[0].scores, q)
            