import numpy as np
from scipy.linalg import solve_discrete_are

# ===============================================================
# 2D double integrator via Kronecker products (from paper)
# ===============================================================
A1 = np.array([[1, 1],
               [0, 1]])
B1 = np.array([[0],
               [1]])
I2 = np.eye(2)

A = np.kron(I2, A1)  # 4x4
B = np.kron(I2, B1)  # 4x2
n, m = A.shape[0], B.shape[1]

Q = np.eye(n)
gamma = 0.9
lambda_reg = 1.0

# Rubric setting: Sigma_w = diag(1e-5, 1e-5, 10, 10)
Sigma_w = np.diag([1e-5, 1e-5, 10, 10])

print("=" * 60)
print("Double Integrator with Anisotropic Noise")
print(f"Sigma_w = diag({1e-5}, {1e-5}, {10}, {10})")
print(f"gamma = {gamma}, lambda = {lambda_reg}")
print("=" * 60)

# Helper function: discounted LQR gain
def discounted_lqr_gain(A, B, Q, R, gamma):
    A_gamma = np.sqrt(gamma) * A
    R_gamma = R / gamma
    P = solve_discrete_are(A_gamma, B, Q, R_gamma)
    F = -gamma * np.linalg.solve(R + gamma * B.T @ P @ B, B.T @ P @ A)
    return P, F

# KL (FR): R_KL = B' * inv(Sigma_w) * B
R_KL = B.T @ np.linalg.inv(Sigma_w) @ B
P_KL, F_KL = discounted_lqr_gain(A, B, Q, R_KL, gamma)

# WKL: R_WKL = B' * B
R_WKL = B.T @ B
P_WKL, F_WKL = discounted_lqr_gain(A, B, Q, R_WKL, gamma)

# KWKL: R_KWKL = B' * inv(Sigma_w + lambda * I) * B
R_KWKL = B.T @ np.linalg.inv(Sigma_w + lambda_reg * np.eye(n)) @ B
P_KWKL, F_KWKL = discounted_lqr_gain(A, B, Q, R_KWKL, gamma)

# Print results
print("\nR_KL (KL/FR):")
print(R_KL)
print("\nR_WKL:")
print(R_WKL)
print("\nR_KWKL:")
print(R_KWKL)

print("\n--- Feedback Gains ---")
print(f"\nKL (FR) gain F_KL:\n{F_KL}")
print(f"Frobenius norm: ||F_KL||_F = {np.linalg.norm(F_KL, 'fro'):.6f}")

print(f"\nWKL gain F_WKL:\n{F_WKL}")
print(f"Frobenius norm: ||F_WKL||_F = {np.linalg.norm(F_WKL, 'fro'):.6f}")

print(f"\nKWKL gain F_KWKL:\n{F_KWKL}")
print(f"Frobenius norm: ||F_KWKL||_F = {np.linalg.norm(F_KWKL, 'fro'):.6f}")

# Paper reported values
print("\n--- Comparison with Paper ---")
print(f"Paper WKL F = [[-0.3882, -1.1817, 0, 0], [0, 0, -0.3882, -1.1817]]")
print(f"Paper WKL ||F||_F = 1.7590")
print(f"Paper KL (FR) F = [[0, 0, 0, 0], [0, 0, -0.5879, -1.5875]]")
print(f"Paper KL (FR) ||F||_F = 1.6929")

# Check match
print("\n--- Verification ---")
wkl_paper = np.array([[-0.3882, -1.1817, 0, 0],
                       [0, 0, -0.3882, -1.1817]])
kl_paper = np.array([[0, 0, 0, 0],
                      [0, 0, -0.5879, -1.5875]])

print(f"WKL gain match (max abs diff): {np.max(np.abs(F_WKL - wkl_paper)):.6f}")
print(f"WKL norm diff: {abs(np.linalg.norm(F_WKL, 'fro') - 1.7590):.6f}")
print(f"KL gain match (max abs diff): {np.max(np.abs(F_KL - kl_paper)):.6f}")
print(f"KL norm diff: {abs(np.linalg.norm(F_KL, 'fro') - 1.6929):.6f}")

# Spectral radius
Acl_KL = A + B @ F_KL
Acl_WKL = A + B @ F_WKL
Acl_KWKL = A + B @ F_KWKL

print(f"\nSpectral radius KL: {np.max(np.abs(np.linalg.eigvals(Acl_KL))):.6f}")
print(f"Spectral radius WKL: {np.max(np.abs(np.linalg.eigvals(Acl_WKL))):.6f}")
print(f"Spectral radius KWKL: {np.max(np.abs(np.linalg.eigvals(Acl_KWKL))):.6f}")
