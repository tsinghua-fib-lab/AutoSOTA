import numpy as np
from scipy.optimize import minimize

class Router:
    def __init__(self, ann, base_data, models, B, alpha, eps, length):
        self.ann = ann
        self.base_data = base_data
        self.models = models
        self.M = len(models)
        self.B = np.asarray(B, dtype=np.float32)
        self.alpha = np.asarray(alpha, dtype=np.float32)
        self.eps = eps
        self.length = length
        self.hatd = np.zeros((self.M, length), dtype=np.float32)
        self.hatg = np.zeros((self.M, length), dtype=np.float32)
        self.index = 0
        self.optimal = False
        self.gamma = np.ones(self.M, dtype=np.float32) / self.M
        self.learn = True
        self.learn_limit = int(np.ceil(self.eps * self.length))
        self._d_cols = np.vstack([np.asarray(base_data[m]) for m in models])  # [M, N]
        self._g_cols = np.vstack([np.asarray(base_data[f"{m}|total_cost"]) for m in models])
        # Fixed seed for deterministic learning phase
        self._rng = np.random.RandomState(42)

    def _estimate_batch(self, queries):
        idxs, dists = self.ann.search(queries)
        d, g = self._calc_batch(idxs, dists)
        n = d.shape[1]
        self.hatd[:, self.index:self.index+n] = d
        self.hatg[:, self.index:self.index+n] = g
        self.index += n
        return d, g

    def _calc_batch(self, idxs, dists):
        # Distance-weighted aggregation: closer neighbors get higher weight
        d_vals = self._d_cols[:, idxs]  # [M, N_query, K]
        g_vals = self._g_cols[:, idxs]  # [M, N_query, K]

        # dists: [N_query, K], cosine similarities
        weights = np.exp(dists * 4.0)  # [N_query, K], sharpen with scale factor
        weights = weights / weights.sum(axis=1, keepdims=True)  # normalize rows

        # Weighted average: [M, N_query]
        d = np.einsum('mnk,nk->mn', d_vals, weights)
        g = np.einsum('mnk,nk->mn', g_vals, weights)
        return d, g

    def _optimize_gamma(self):
        Hd = self.hatd[:, :self.index].astype(np.float64)
        Hg = self.hatg[:, :self.index].astype(np.float64)
        alpha = float(self.alpha)
        eps = self.eps
        B = self.B.astype(np.float64)
        M = self.M

        def F_and_grad(gamma):
            scores = Hd.T * alpha - Hg.T * gamma  # [N, M]
            winner = np.argmax(scores, axis=1)  # [N]
            winner_scores = scores[np.arange(len(winner)), winner]

            term1 = eps * np.dot(gamma, B)
            term2 = winner_scores.sum()

            grad_term1 = eps * B
            grad_term2 = np.zeros(M, dtype=np.float64)
            for i in range(M):
                mask = (winner == i)
                grad_term2[i] = -Hg[i, mask].sum()

            return term1 + term2, grad_term1 + grad_term2

        x0 = np.full(M, 1.0/M, dtype=np.float64)
        bounds = [(0.0, 1.0)] * M
        res = minimize(F_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
        self.gamma = res.x.astype(np.float32)
        self.optimal = True

    def routing_batch(self, queries):
        n = len(queries)
        out = np.empty(n, dtype=np.int32)

        remaining_learn = max(0, self.learn_limit - self.index)
        n_learn = min(remaining_learn, n)

        if n_learn > 0:
            out[:n_learn] = self._rng.randint(0, self.M + 1, size=n_learn)
            self._estimate_batch(queries[:n_learn])

        n_exploit = n - n_learn
        if n_exploit > 0:
            if not self.optimal:
                self._optimize_gamma()
            d, g = self._estimate_batch(queries[n_learn:])
            scores = d * self.alpha.reshape(-1, 1) - g * self.gamma.reshape(-1, 1)
            out[n_learn:] = scores.argmax(axis=0)

        return out
