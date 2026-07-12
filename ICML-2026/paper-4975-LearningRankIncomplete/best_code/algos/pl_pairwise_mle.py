from .algo import Algorithm
from feedback import PreferenceFeedback
import numpy as np
from scipy.optimize import minimize # type: ignore
from scipy.special import expit # type: ignore

class PLPairwiseMLEAlgorithm(Algorithm[PreferenceFeedback]):
    def __init__(self, n: int, lambda_reg: float = 0.01) -> None:
        super().__init__(name="PL Pairwise MLE")
        self.n = n
        self.lambda_reg = lambda_reg
        
        # wins[i, j] tracks the number of times item i defeated item j
        self.wins = np.zeros((n, n), dtype=float)

    def feedback(self, feedback: PreferenceFeedback) -> None:
        # prec_id is the preferred item (winner), succ_id is the less preferred item (loser)
        winner = feedback.prec_id
        loser = feedback.succ_id
        
        self.wins[winner, loser] += 1

    def predict(self) -> list[int]:
        def negative_log_likelihood(theta: np.ndarray) -> tuple[float, np.ndarray]:
            """
            Computes the regularized negative log-likelihood and its gradient.
            NLL = sum_{wins_ij} log(1 + exp(theta_j - theta_i)) + 0.5 * lambda * ||theta||^2
            """
            # Start with regularization terms
            grad = self.lambda_reg * theta
            loss = 0.5 * self.lambda_reg * np.dot(theta, theta)
            
            # Vectorized computation over all observed wins
            i_indices, j_indices = np.nonzero(self.wins)
            if i_indices.size > 0:
                w = self.wins[i_indices, j_indices]
                theta_winners = theta[i_indices]
                theta_losers = theta[j_indices]
                diffs = theta_losers - theta_winners

                # Loss: sum(w * log(1 + exp(diff)))
                # Use np.logaddexp(0, x) for a numerically stable version of log(1 + exp(x))
                loss += np.sum(w * np.logaddexp(0, diffs))

                # Gradient:
                # The derivative of log(1 + exp(theta_j - theta_i)) w.r.t. theta_j is p_ji
                # and w.r.t. theta_i is -p_ji, where p_ji is the probability of j beating i.
                # p_ji = exp(diff) / (1 + exp(diff)) = sigmoid(diff)
                probs_j_beats_i = expit(diffs)
                
                # Use np.add.at for safe gradient updates when an item is in multiple pairs
                np.add.at(grad, j_indices, w * probs_j_beats_i)
                np.add.at(grad, i_indices, -w * probs_j_beats_i)

            return loss, grad

        # Initialize theta values at 0
        initial_theta = np.zeros(self.n)
        res = minimize(
            negative_log_likelihood, 
            initial_theta, 
            jac=True, 
            method='L-BFGS-B'
        )
        
        theta_opt = res.x
        
        # Sort scores in descending order to find the highest-ranked items
        return np.argsort(theta_opt)[::-1].tolist()

