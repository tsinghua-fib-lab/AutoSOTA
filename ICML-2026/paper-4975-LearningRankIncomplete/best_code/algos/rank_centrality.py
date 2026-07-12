from .algo import Algorithm
from feedback import PreferenceFeedback
import numpy as np

class RankCentralityAlgorithm(Algorithm[PreferenceFeedback]):
    def __init__(self, n: int, max_iter: int = 1000, tol: float = 1e-6) -> None:
        super().__init__(name="RankCentrality")
        self.n = n
        self.max_iter = max_iter
        self.tol = tol
        
        # Y[i, j] tracks the number of times item j is preferred over item i.
        self.Y = np.zeros((n, n))
        
        # k[i, j] tracks the total number of comparisons between i and j.
        self.k = np.zeros((n, n))

    def feedback(self, feedback: PreferenceFeedback) -> None:
        # Following the established convention: prec_id is the winner, succ_id is the loser
        winner = feedback.prec_id
        loser = feedback.succ_id
        
        # Increment wins for the winner over the loser (j over i, where i=loser, j=winner)
        self.Y[loser, winner] += 1
        
        # Increment total comparisons between the pair symmetrically
        self.k[loser, winner] += 1
        self.k[winner, loser] += 1

    def predict(self) -> list[int]:
        # 1. Build the transition matrix P
        # d_i is the degree of node i in the comparison graph (number of unique opponents)
        d_i = np.sum(self.k > 0, axis=1)
        d_max = np.max(d_i)
        
        if d_max == 0:
            # No comparisons yet, return a random ranking
            return np.random.permutation(self.n).tolist()

        P = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                if self.k[i, j] > 0:
                    # a_ij is the fraction of times j is preferred over i
                    a_ij = self.Y[i, j] / self.k[i, j]
                    
                    # P_{ij} = (1 / d_{max}) * A_{ij} 
                    # Since a_ij + a_ji = 1 in an unregularized setup, A_ij = a_ij
                    P[i, j] = a_ij / d_max
        
        # Add self-loops to ensure each row sums to 1
        for i in range(self.n):
            P[i, i] = 1.0 - np.sum(P[i, :])

        # 2. Compute the stationary distribution (pi) via power iteration
        pi = np.ones(self.n) / self.n
        for _ in range(self.max_iter):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi) < self.tol:
                break
            pi = pi_new
        
        # 3. Sort items by their stationary probability (Rank Centrality score)
        return np.argsort(pi)[::-1].tolist()
    