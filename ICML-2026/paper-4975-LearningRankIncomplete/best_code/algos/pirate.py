from feedback import PartialOrderFeedback
import numpy as np
from .algo import Algorithm

class PirateAlgorithm(Algorithm[PartialOrderFeedback]):
    def __init__(self, n: int) -> None:
        super().__init__(name="PIRATE")
        self.n = n
        
        # w[i, j] tracks the number of times item i defeated item j
        self.w = np.zeros((n, n))

    def feedback(self, feedback: PartialOrderFeedback) -> None:
        for pref in feedback.rank_breaking():
            i, j = pref.prec_id, pref.succ_id
            self.w[i, j] += 1

    def predict(self) -> list[int]:
        n = self.n
        
        # Build a continuous transition matrix from win ratios.
        # P[i, j] = probability that j beats i (based on observed comparisons)
        # Normalized so each row sums to 1 (stochastic matrix).
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                total = self.w[i, j] + self.w[j, i]
                if total > 0:
                    # Fraction of comparisons where j beats i
                    # If j frequently beats i, P[i,j] is high -> i's rank flows to j
                    P[i, j] = self.w[j, i] / total
        
        # Normalize rows to sum to 1; isolate items with no comparisons
        for i in range(n):
            row_sum = P[i].sum()
            if row_sum > 0:
                P[i] /= row_sum
            else:
                P[i, i] = 1.0
        
        # Apply PageRank-style damping for ergodicity (handles disconnected components)
        d = 0.85
        P = d * P + (1 - d) * np.ones((n, n)) / n
        
        # Power iteration for stationary distribution
        pi = np.ones(n) / n
        for _ in range(1000):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi) < 1e-9:
                break
            pi = pi_new
        
        # Items with higher stationary probability are ranked higher
        return np.argsort(-pi).tolist()
