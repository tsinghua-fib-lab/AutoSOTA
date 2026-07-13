# Plot isolines of
# f(x, y) = (log P(X >= x) + log P(Y >= y) - log P(X >= x, Y >= y)) / log P(Y >= y)
# where (X, Y) are standard Gaussians with Pearson correlation rho.
# The line y = rho * x is overlaid.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

# -----------------------------
# Parameters
# -----------------------------
rho = 0.5                 # Pearson correlation
x_min, x_max = 0, 2.5
y_min, y_max = 0, 2.5
grid_size = 120

# -----------------------------
# Create grid
# -----------------------------
x = np.linspace(x_min, x_max, grid_size)
y = np.linspace(y_min, y_max, grid_size)
X, Y = np.meshgrid(x, y)

# -----------------------------
# Marginal tail probabilities
# -----------------------------
PX = 1.0 - norm.cdf(X)
PY = 1.0 - norm.cdf(Y)

# -----------------------------
# Joint tail probability
# P(X >= x, Y >= y)
# -----------------------------
mean = np.array([0.0, 0.0])
cov = np.array([[1.0, rho],
                [rho, 1.0]])

joint = np.zeros_like(X)
for i in range(grid_size):
    for j in range(grid_size):
        joint[i, j] = (
                1.0
                - norm.cdf(X[i, j])
                - norm.cdf(Y[i, j])
                + multivariate_normal.cdf(
            [X[i, j], Y[i, j]],
            mean=mean,
            cov=cov
        )
        )

# -----------------------------
# Numerical safety for logs
# -----------------------------
eps = 1e-12
PX = np.maximum(PX, eps)
PY = np.maximum(PY, eps)
joint = np.maximum(joint, eps)

# -----------------------------
# Compute f(x, y)
# -----------------------------
F = (np.log(PX) + np.log(PY) - np.log(joint)) / np.log(PY)

# -----------------------------
# Plot isolines
# -----------------------------
plt.figure()
contours = plt.contour(X, Y, F, levels=12)
plt.clabel(contours, inline=True)
plt.xlabel("x")
plt.ylabel("y")
plt.title(rf"Isolines of explanation score for $\rho$ = {rho}")

# Overlay y = rho * x
plt.plot(x, rho * x, linestyle="--", color='red')
plt.savefig('explanation_score_XtoY.png')
plt.show()