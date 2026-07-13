# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np

def dirichlet_sample(alphas):
    """
    Generate samples from an array of Dirichlet distributions.

    Args:
        alphas (ndarray): Parameters of the Dirichlet distributions.

    Returns:
        ndarray: Samples from the Dirichlet distributions.
    """
    r = np.random.standard_gamma(alphas) # Draw samples from Gamma distributions
    return r / r.sum(-1, keepdims=True)  # Normalize samples to get Dirichlet samples


class PosteriorProbabilisties(object):
    """
    A class to handle posterior probabilities of an MDP, allowing for updates and samples.
    """
    def __init__(self, ns: int, na: int, prior_p: float = .5, prior_r: float = .5):
        """
        Initialize the posterior probabilities.
        
        Args:
            ns (int): Number of states in the MDP.
            na (int): Number of actions in the MDP.
            prior_p (float, optional): Prior for transition probabilities. Defaults to 0.5.
            prior_r (float, optional): Prior for rewards. Defaults to 0.5.
        """
        self.ns = ns
        self.na = na
        # Initialize priors for transition probabilities and rewards
        self.prior_transition = prior_p * np.ones((ns, na, ns))
        self.prior_rewards = prior_r * np.ones((ns, na, 2))
        # Initialize counters for state visits and rewards
        self.n_visits_states = np.zeros((ns, na, ns))
        self.n_visits_rew = np.zeros((ns, na))


    def update(self, state: int, action: int, next_state: int, reward: float):
        """
        Update counters based on a new observation.

        Args:
            state (int): The current state.
            action (int): The action taken.
            next_state (int): The state transitioned to.
            reward (float): The reward received.
        """
        # Update state visit and reward counters
        self.n_visits_states[state, action, next_state] += 1
        self.n_visits_rew[state, action] += reward
        
    def sample_posterior(self):
        """
        Sample from the posterior distributions of the MDP parameters.

        Returns:
            tuple: The sampled transition probabilities and rewards.
        """
        # Compute posterior parameters
        posterior_transition = self.prior_transition + self.n_visits_states
        posterior_rewards_alpha = self.prior_rewards[:, :, 0] + self.n_visits_rew
        posterior_rewards_beta = self.prior_rewards[:, :, 1] + self.n_visits_states.sum(-1) - self.n_visits_rew

        # Sample from the posterior distributions
        P = dirichlet_sample(posterior_transition)
        R = np.random.beta(posterior_rewards_alpha, posterior_rewards_beta)[..., np.newaxis]
        return P, R
    
    def mle(self):
        """
        Compute maximum likelihood estimates (MLE) for the MDP parameters.

        Returns:
            tuple: The MLE of the transition probabilities and rewards.
        """
        # Compute posterior parameters
        posterior_transition = self.prior_transition + self.n_visits_states
        posterior_rewards_alpha = self.prior_rewards[:, :, 0] + self.n_visits_rew
        posterior_rewards_beta = self.prior_rewards[:, :, 1] + self.n_visits_states.sum(-1) - self.n_visits_rew

        # Compute MLE of the parameters
        P = posterior_transition / posterior_transition.sum(-1, keepdims=True)
        R = posterior_rewards_alpha / (posterior_rewards_alpha + posterior_rewards_beta)
        return P, R[..., np.newaxis]
