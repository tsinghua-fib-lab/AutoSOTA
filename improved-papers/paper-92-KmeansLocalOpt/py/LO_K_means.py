import numpy as np
from numpy.random import Generator, MT19937

"""
LO-K-means (C-LO, D-LO, Min-D-LO).
Key Assumptions:
- N > K (Number of samples > Number of clusters).
- All input data points (X) are unique.

The time complexity per iteration is O(NKD).

Example Usage:
>>> kmeans = LO_K_means(
...     X=X_unique_data,    # (N, D) unique data points
...     weight=sample_weights,  # (N,) sample weights
...     K=number_of_clusters,
...     init_="kmeans++",     # or "rand"
...     breg="squared",       # or "KL", "Itakura"
...     random_state=None,
...     eps=1e-10
... )
>>> assignments, centers, loss = kmeans.K_means()

Available Clustering Methods:
- K_means(): Standard K-means algorithm.
- C_LO_K_means(): Aims to ensure C-local optimality.
- D_LO_K_means(): Aims to ensure D-local optimality.
- Min_D_LO_K_means(): Aims to ensure D-local optimality; often faster in practice than D_LO_K_means.
"""


class LO_K_means:

    def __init__(
        self,
        X: np.ndarray,  # Data matrix of shape (N, D)
        weight: np.ndarray,  # Sample weights of length N
        K: int,  # Number of clusters
        init_: str = "kmeans++",  # Initialization method (kmeans++, rand)
        breg: str = "squared",  # Type of Bregman divergence (squared, KL, Itakura)
        random_state: int = None,
        eps: float = 1e-10,
    ):
        self.X = X
        self.weight = weight
        self.N, self.D = X.shape
        self.K = K
        self.init_centers = np.zeros((K, self.D))
        self.breg = breg

        seed = random_state if random_state is not None else np.random.SeedSequence().entropy
        self.rng = Generator(MT19937(seed))
        self.eps = eps
        self.step_num = 0  # Counter for iteration steps
        self.imp_num = 0  # Counter for improvement steps
        if init_ == "kmeans++":
            self.kmeans_plus_plus_init()
        elif init_ == "rand":
            self.kmeans_rand_init()
        else:
            raise ValueError(f"unknown init: {init_}")

    # k-means++ initialization
    def kmeans_plus_plus_init(self):
        probs = self.weight / self.weight.sum()
        idx0 = self.rng.choice(self.N, p=probs)
        self.init_centers[0] = self.X[idx0]
        closest_dist_sq = np.full(self.N, np.inf)

        for c in range(1, self.K):
            dist_sq = np.sum((self.X - self.init_centers[c - 1]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
            probs = self.weight * closest_dist_sq
            probs /= probs.sum()
            idx = self.rng.choice(self.N, p=probs)
            self.init_centers[c] = self.X[idx]

    # Random initialization
    def kmeans_rand_init(self):
        probs = self.weight / self.weight.sum()
        for c in range(self.K):
            idx = self.rng.choice(self.N, p=probs)
            self.init_centers[c] = self.X[idx]

    # Compute Bregman divergence between vectors a and b
    def bregman(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.breg == "squared":
            return float(np.sum((a - b) ** 2))
        # KL divergence
        if self.breg == "KL":
            phi_a = np.sum(a * np.log(a))
            phi_b = np.sum(b * np.log(b))
            grad_ab = np.sum((a - b) * (np.log(b) + 1))
            return float(phi_a - phi_b - grad_ab)
        # Itakura-Saito divergence
        if self.breg == "Itakura":
            phi_a = -np.sum(np.log(a))
            phi_b = -np.sum(np.log(b))
            grad_ab = -np.sum((a - b) / b)
            return float(phi_a - phi_b - grad_ab)
        raise ValueError(f"unknown breg: {self.breg}")

    # Compute new centers
    def weighted_mean(self, assignment: np.ndarray) -> np.ndarray:
        centers = np.zeros((self.K, self.D))
        total_weight = np.zeros(self.K)
        for i, k in enumerate(assignment):
            centers[k] += self.weight[i] * self.X[i]
            total_weight[k] += self.weight[i]
        mask = total_weight > self.eps
        centers[mask] /= total_weight[mask, None]
        return centers

    # Compute clustering loss
    def clustering_loss(self, centers: np.ndarray, assignment: np.ndarray) -> float:
        losses = [self.bregman(self.X[i], centers[assignment[i]]) for i in range(self.N)]
        return float(np.sum(self.weight * np.array(losses)))

    # Standard K-means
    def K_means(self):
        centers = self.init_centers.copy()
        assignment = np.zeros(self.N, dtype=int)
        old_assignment = np.full(self.N, -1)
        self.step_num = 0

        while not np.array_equal(assignment, old_assignment):
            self.step_num += 1
            old_assignment[:] = assignment

            dists = np.array([[self.bregman(self.X[i], centers[k]) for k in range(self.K)] for i in range(self.N)])

            assignment = np.argmin(dists, axis=1)

            cluster_w = np.bincount(assignment, weights=self.weight, minlength=self.K)

            for k in range(self.K):
                if cluster_w[k] < self.eps:
                    idx = np.where(cluster_w[assignment] > self.weight)[0][0]
                    cluster_w[assignment[idx]] -= self.weight[idx]
                    assignment[idx] = k
                    cluster_w[k] += self.weight[idx]

            centers = self.weighted_mean(assignment)

        loss = self.clustering_loss(centers, assignment)
        return assignment, centers, loss

    # C-LO-K-means
    def C_LO_K_means(self):
        centers = self.init_centers.copy()
        assignment = np.zeros(self.N, dtype=int)
        old_assignment = np.full(self.N, -1)
        self.step_num = 0

        while not np.array_equal(assignment, old_assignment):
            self.step_num += 1
            old_assignment[:] = assignment

            dists = np.array([[self.bregman(self.X[i], centers[k]) for k in range(self.K)] for i in range(self.N)])

            assignment = np.argmin(dists, axis=1)

            cluster_w = np.bincount(assignment, weights=self.weight, minlength=self.K)

            for k in range(self.K):
                if cluster_w[k] < self.eps:
                    idx = np.where(cluster_w[assignment] > self.weight)[0][0]
                    cluster_w[assignment[idx]] -= self.weight[idx]
                    assignment[idx] = k
                    cluster_w[k] += self.weight[idx]

            # Function 1
            if np.array_equal(assignment, old_assignment):
                for i in range(self.N):
                    ties = np.where(np.isclose(dists[i, :], dists[i, assignment[i]], atol=self.eps))[0]
                    if ties.size > 1:
                        assignment[i] = ties[1]
                        break

            centers = self.weighted_mean(assignment)

        loss = self.clustering_loss(centers, assignment)
        return assignment, centers, loss

    # D-LO-K-means
    def D_LO_K_means(self):
        centers = self.init_centers.copy()
        assignment = np.zeros(self.N, dtype=int)
        old_assignment = np.full(self.N, -1)
        self.step_num = 0
        self.imp_num = 0
        dists = np.array([[self.bregman(self.X[i], centers[k]) for k in range(self.K)] for i in range(self.N)])

        while not np.array_equal(assignment, old_assignment):
            self.step_num += 1
            old_assignment[:] = assignment

            assignment = np.argmin(dists, axis=1)
            cluster_w = np.bincount(assignment, weights=self.weight, minlength=self.K)

            for k in range(self.K):
                if cluster_w[k] < self.eps:
                    idx = np.where(cluster_w[assignment] > self.weight)[0][0]
                    cluster_w[assignment[idx]] -= self.weight[idx]
                    assignment[idx] = k
                    cluster_w[k] += self.weight[idx]

            centers = self.weighted_mean(assignment)

            dists = np.array([[self.bregman(self.X[i], centers[k]) for k in range(self.K)] for i in range(self.N)])

            # Function 2
            if np.array_equal(assignment, old_assignment):
                done = False
                for i in range(self.N):
                    a = assignment[i]
                    if cluster_w[a] <= self.weight[i] + self.eps:
                        continue
                    for b in range(self.K):
                        if a == b:
                            continue

                        a_new = centers[a] + self.weight[i] * (centers[a] - self.X[i]) / (cluster_w[a] - self.weight[i])
                        b_new = centers[b] + self.weight[i] * (self.X[i] - centers[b]) / (cluster_w[b] + self.weight[i])

                        diff = self.weight[i] * (dists[i, b] - dists[i, a]) - (
                            (cluster_w[b] + self.weight[i]) * self.bregman(b_new, centers[b])
                            + (cluster_w[a] - self.weight[i]) * self.bregman(a_new, centers[a])
                        )

                        if diff + self.eps < 0:
                            assignment[i] = b
                            centers[a] = a_new
                            centers[b] = b_new

                            dists[:, a] = [self.bregman(self.X[j], centers[a]) for j in range(self.N)]
                            dists[:, b] = [self.bregman(self.X[j], centers[b]) for j in range(self.N)]
                            self.imp_num += 1
                            done = True
                            break
                    if done:
                        break

        loss = self.clustering_loss(centers, assignment)
        return assignment, centers, loss

    # Min-D-LO-Kmeans
    def Min_D_LO_K_means(self):
        centers = self.init_centers.copy()
        assignment = np.zeros(self.N, dtype=int)
        old_assignment = np.full(self.N, -1)
        self.step_num = 0
        self.imp_num = 0
        dists = np.array([[self.bregman(self.X[i], centers[k]) for k in range(self.K)] for i in range(self.N)])
        improve_times = 0

        while not np.array_equal(assignment, old_assignment):
            self.step_num += 1
            old_assignment[:] = assignment

            assignment = np.argmin(dists, axis=1)
            cluster_w = np.bincount(assignment, weights=self.weight, minlength=self.K)

            for k in range(self.K):
                if cluster_w[k] < self.eps:
                    idx = np.where(cluster_w[assignment] > self.weight)[0][0]
                    cluster_w[assignment[idx]] -= self.weight[idx]
                    assignment[idx] = k
                    cluster_w[k] += self.weight[idx]

            centers = self.weighted_mean(assignment)

            dists = np.array([[self.bregman(self.X[i], centers[k]) for k in range(self.K)] for i in range(self.N)])

            # Function 3
            if np.array_equal(assignment, old_assignment):
                best_diff = np.inf
                best_i = best_b = -1
                best_a_new = best_b_new = None
                for i in range(self.N):
                    a = assignment[i]
                    if cluster_w[a] <= self.weight[i] + self.eps:
                        continue
                    for b in range(self.K):
                        if a == b:
                            continue
                        a_new = centers[a] + self.weight[i] * (centers[a] - self.X[i]) / (cluster_w[a] - self.weight[i])
                        b_new = centers[b] + self.weight[i] * (self.X[i] - centers[b]) / (cluster_w[b] + self.weight[i])
                        diff = self.weight[i] * (dists[i, b] - dists[i, a]) - (
                            (cluster_w[b] + self.weight[i]) * self.bregman(b_new, centers[b])
                            + (cluster_w[a] - self.weight[i]) * self.bregman(a_new, centers[a])
                        )
                        if diff + self.eps < best_diff:
                            best_diff = diff
                            best_i, best_b = i, b
                            best_a_new, best_b_new = a_new, b_new

                if best_diff + self.eps < 0:
                    a = assignment[best_i]
                    assignment[best_i] = best_b
                    centers[a] = best_a_new
                    centers[best_b] = best_b_new

                    dists[:, a] = [self.bregman(self.X[j], centers[a]) for j in range(self.N)]
                    dists[:, best_b] = [self.bregman(self.X[j], centers[best_b]) for j in range(self.N)]
                    self.imp_num += 1

        loss = self.clustering_loss(centers, assignment)
        return assignment, centers, loss
