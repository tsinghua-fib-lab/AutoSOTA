import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import solve_continuous_lyapunov


def gelbrich_distance_2_L(dist_1, dist_2, L,
                          throw_err_if_complex=False):
    """
    Computes the Gelbrich distance squared between two Gaussian distributions given L.

    distance(L)^2 = tr[L L^T C] - 2 tr[S]
    where
    C = (mu_P - mu_Q)(mu_P - mu_Q)^T + Sigma_P + Sigma_Q
    S = sqrtm(L^T Sigma_Q L L^T Sigma_P L)

    Parameters:
    - dist_1: tuple, (mu_P, Sigma_P), the first distribution
    - dist_2: tuple, (mu_Q, Sigma_Q), the second distribution
    - L: np.ndarray, shape (n, n), lower triangular
    - throw_err_if_complex: bool, whether to throw an error if the result is complex

    Returns:
    - f_val: float
    """
    mu_P, Sigma_P = dist_1
    mu_Q, Sigma_Q = dist_2
    # check that Sigma_P and Sigma_Q are symmetric and positive definite
    if not np.allclose(Sigma_P, Sigma_P.T, rtol=1e-5, atol=1e-5):
        raise ValueError(f"Sigma_P is not symmetric: {Sigma_P}")
    if not np.allclose(Sigma_Q, Sigma_Q.T, rtol=1e-5, atol=1e-5):
        raise ValueError(f"Sigma_Q is not symmetric: {Sigma_Q}")
    eigvals_P = np.linalg.eigvals(Sigma_P)
    if np.any(eigvals_P < -1e-15):
        raise ValueError(f"Sigma_P is not positive definite: {Sigma_P} eigvals: {eigvals_P}")
    eigvals_Q = np.linalg.eigvals(Sigma_Q)
    if np.any(eigvals_Q < -1e-15):
        raise ValueError(f"Sigma_Q is not positive definite: {Sigma_Q} eigvals: {eigvals_Q}")
    mu_diff = mu_P - mu_Q
    C = np.outer(mu_diff, mu_diff) + Sigma_P + Sigma_Q
    S = L.T @ Sigma_Q @ L @ L.T @ Sigma_P @ L
    Y = sqrtm(S)
    Y = np.real_if_close(Y)
    if np.iscomplexobj(Y):
        print("Warning: Y in gelbrich_distance_2_L is complex.")
        if throw_err_if_complex:
            raise ValueError("Y is complex.")
        Y = Y.real
    f_val = np.trace(L @ L.T @ C) - 2 * np.trace(Y)
    return f_val


def gelbrich_distance_L(dist_1, dist_2, L,
                        throw_err_if_complex=False):
    """
    Computes the Gelbrich distance between two Gaussian distributions given L.

    distance(L) = sqrt( tr[L L^T C] - 2 tr[Y] )
    where
    C = (mu_P - mu_Q)(mu_P - mu_Q)^T + Sigma_P + Sigma_Q
    Y = sqrtm(L^T Sigma_Q L L^T Sigma_P L)

    Parameters:
    - dist_1: tuple, (mu_P, Sigma_P), the first distribution
    - dist_2: tuple, (mu_Q, Sigma_Q), the second distribution
    - L: np.ndarray, shape (n, n)
    - throw_err_if_complex: bool, whether to throw an error if the result is complex

    Returns:
    - f_val: float
    """
    d = gelbrich_distance_2_L(dist_1, dist_2, L,
                              throw_err_if_complex=throw_err_if_complex)
    if d < 0:
      print("Warning: Gelbrich distance is negative.")
    d = np.maximum(d, 0)
    if np.iscomplexobj(d):
      print("Warning: d in gelbrich_distance_L is complex.")
      if throw_err_if_complex:
        raise ValueError("d is complex.")
      d = d.real
    # check if nan
    if np.isnan(d):
      raise ValueError("Gelbrich distance is NaN.")
    return np.sqrt(d)


def gelbrich_gradient_2_L(dist_1, dist_2, L, epsilon_reg=1e-8, debug=False,
                          epsilon_fd=1e-6):
    """
    Compute the gradient of the squared Gelbrich distance with respect to
    the weight matrix L.

    C = (mu_P - mu_Q)(mu_P - mu_Q)^T + Sigma_P + Sigma_Q
    S(L) = sqrt( sqrt(Sigma_Q) @ L @ L^T @ Sigma_P @ L @ L^T @ sqrt(Sigma_Q) )
    d(L)^2 = tr[L L^T C] - 2 tr[ S(L) ]

    Using the following elementary functions:
    H(X) = X X^T
    P(X) = B X A
    S(X) = sqrtm(X)
    Y(X) = tr[X]

    Parameters:
    - dist_1: tuple, (mu_P, Sigma_P), the first distribution
    - dist_2: tuple, (mu_Q, Sigma_Q), the second distribution
    - L: np.ndarray, shape (n, n), lower triangular
    - epsilon_reg: float, small regularization term for numerical stability
    - debug: bool, whether to print debug information
    - epsilon_fd: float, small finite difference step for numerical gradient checking

    Returns:
    - distance2: float, the value of the Gelbrich distance squared
    - grad2: np.ndarray, shape (n, n), the gradient matrix
    """
    # Ensure L is a square matrix
    assert len(L.shape) == 2, "L must be a matrix."
    n, m = L.shape
    assert m == n, "L must be a square matrix."

    assert len(dist_1) == 2, "dist_1 must be a tuple of length 2."
    assert len(dist_2) == 2, "dist_2 must be a tuple of length 2."
    mu_P, Sigma_P = dist_1
    mu_Q, Sigma_Q = dist_2

    # Ensure input dimensions are compatible
    assert Sigma_P.shape == (n, n), "Sigma_P must be of shape (n, n)"
    assert Sigma_Q.shape == (n, n), "Sigma_Q must be of shape (n, n)"
    assert mu_P.shape == (n,), "mu_P must be a vector of shape (n,)"
    assert mu_Q.shape == (n,), "mu_Q must be a vector of shape (n,)"

    # Ensure symmetric and positive definite matrices
    if not np.allclose(Sigma_P, Sigma_P.T):
      raise ValueError("Sigma_P is not symmetric.")
    if not np.allclose(Sigma_Q, Sigma_Q.T):
      raise ValueError("Sigma_Q is not symmetric.")
    eigvals_P = np.linalg.eigvals(Sigma_P)
    if not np.all(eigvals_P > -1e-15):
      raise ValueError(f"Sigma_P is not positive definite: {Sigma_P} eigvals: {eigvals_P}")
    eigvals_Q = np.linalg.eigvals(Sigma_Q)
    if not np.all(eigvals_Q > -1e-15):
      raise ValueError(f"Sigma_Q is not positive definite: {Sigma_Q} eigvals: {eigvals_Q}")

    mu_diff = mu_P - mu_Q
    C = np.outer(mu_diff, mu_diff) + Sigma_P + Sigma_Q

    # derivative of L L^T C with respect to L
    d_1_dL = C @ L + C.T @ L

    A = sqrtm(Sigma_Q)
    A = np.real_if_close(A)
    B = sqrtm(Sigma_P)
    B = np.real_if_close(B)

    # ================================================================
    # FORWARD PASS:

    # Compute H_1 = L L^T
    H_1 = L @ L.T

    # Compute P = B H_1 A
    P = B @ H_1 @ A

    # Compute H_2 = P P^T
    H_2 = P @ P.T
    H_2 += epsilon_reg * np.eye(n) # for the square root, just in case

    # Compute S = sqrtm(H_2)
    S = sqrtm(H_2)
    S = np.real_if_close(S)
    if np.iscomplexobj(S):
        # print("Warning: S in forward pass of gelbrich_gradient_2_L is complex.")
        S = S.real
    if not np.allclose(S, S.T):
        raise ValueError("S is not symmetric.")

    # Compute y = trace[S]
    y = np.trace(S)

    # Compute squared Gelbrich distance
    distance2 = np.trace(L @ L.T @ C) - 2 * y
    distance2 = np.maximum(distance2, 0)

    # ================================================================
    # BACKWARD PASS:

    if debug:
      # define g_1: S   |--> y
      def g_1(S):
        return np.trace(S)
      # define g_2: H_2 |--> y
      def g_2(H_2):
        return g_1(sqrtm(H_2))
      # define g_3: P   |--> y
      def g_3(P):
        return g_2(P @ P.T)
      # define g_4: H_1 |--> y
      def g_4(H_1):
        return g_3(B @ H_1 @ A)
      # define g_5: L  |--> y
      def g_5(L):
        return g_4(L @ L.T)

    # Compute d(g_1)/dS
    # The derivative of the trace of a matrix with respect to the matrix
    # is the identity matrix.
    d_g1_dS = np.eye(n)

    # Compute d(g_2)/dH_2
    if debug:
      print("\n")
      print("d(g1)/dS ANALYTICAL:")
      print(d_g1_dS)

      d_g1_dS_numerical = np.zeros((n, n))
      for i in range(n):
          for j in range(n):
              S_plus = S.copy()
              S_minus = S.copy()
              S_plus[i, j] += epsilon_fd
              S_minus[i, j] -= epsilon_fd
              g1_plus = g_1(S_plus)
              g1_minus = g_1(S_minus)
              d_g1_dS_numerical[i, j] = (g1_plus - g1_minus) / (2 * epsilon_fd)

      print("d(g1)/dS NUMERICAL:")
      print(d_g1_dS_numerical)
      print("\n")

    d_g2_dH2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            E_ij = np.zeros((n, n))
            E_ij[i, j] = 1
            # Solve the Lyapunov equation: S dS_dH2_ij + dS_dH2_ij S = E_ij
            dS_dH2_ij = solve_continuous_lyapunov(S, E_ij)
            d_g2_dH2[i, j] = np.trace(dS_dH2_ij)

    if debug:
      print("\n")
      print("d(g2)/dH2 ANALYTICAL:")
      print(d_g2_dH2)

      d_g2_dH2_numerical = np.zeros((n, n))
      for i in range(n):
          for j in range(n):
              H_2_plus = H_2.copy()
              H_2_minus = H_2.copy()
              H_2_plus[i, j] += epsilon_fd
              H_2_minus[i, j] -= epsilon_fd
              g2_plus = g_2(H_2_plus)
              g2_minus = g_2(H_2_minus)
              d_g2_dH2_numerical[i, j] = (g2_plus - g2_minus) / (2 * epsilon_fd)

      print("d(g2)/dH2 NUMERICAL:")
      print(d_g2_dH2_numerical)
      print("\n")

    # Compute d(g_3)/dP
    d_g3_dP = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            E_ij = np.zeros((n, n))
            E_ij[i, j] = 1
            # dH2_dPij = E_ij P.T + P E_ij.T
            dH2_dP_ij = E_ij @ P.T + P @ E_ij.T
            d_g3_dP[i, j] = np.trace(d_g2_dH2.T @ dH2_dP_ij)

    if debug:
      print("\n")
      print("d(g3)/dP ANALYTICAL:")
      print(d_g3_dP)

      d_g3_dP_numerical = np.zeros((n, n))
      for i in range(n):
          for j in range(n):
              P_plus = P.copy()
              P_minus = P.copy()
              P_plus[i, j] += epsilon_fd
              P_minus[i, j] -= epsilon_fd
              g3_plus = g_3(P_plus)
              g3_minus = g_3(P_minus)
              d_g3_dP_numerical[i, j] = (g3_plus - g3_minus) / (2 * epsilon_fd)

      print("d(g3)/dP NUMERICAL:")
      print(d_g3_dP_numerical)
      print("\n")

    # Compute d(g_4)/dH1
    d_g4_dH1 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            E_ij = np.zeros((n, n))
            E_ij[i, j] = 1
            # dP_dH1_ij = B E_ij A
            dP_dH1_ij = B @ E_ij @ A
            d_g4_dH1[i, j] = np.trace(d_g3_dP.T @ dP_dH1_ij)

    if debug:
      print("\n")
      print("d(g4)/dH1 ANALYTICAL:")
      print(d_g4_dH1)

      d_g4_dH1_numerical = np.zeros((n, n))
      for i in range(n):
          for j in range(n):
              H_1_plus = H_1.copy()
              H_1_minus = H_1.copy()
              H_1_plus[i, j] += epsilon_fd
              H_1_minus[i, j] -= epsilon_fd
              g4_plus = g_4(H_1_plus)
              g4_minus = g_4(H_1_minus)
              d_g4_dH1_numerical[i, j] = (g4_plus - g4_minus) / (2 * epsilon_fd)

      print("d(g4)/dH1 NUMERICAL:")
      print(d_g4_dH1_numerical)
      print("\n")

    # Compute d(g_5)/dL
    d_g5_dL = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            E_ij = np.zeros((n, n))
            E_ij[i, j] = 1
            # dH1_dL_ij = E_ij L^T + L E_ij^T
            dH1_dL_ij = E_ij @ L.T + L @ E_ij.T
            d_g5_dL[i, j] = np.trace(d_g4_dH1.T @ dH1_dL_ij)

    gradient2 = d_1_dL - 2 * d_g5_dL

    if debug:
      print("\n\n")
      print("d( Tr[L L^T C] )/dL:")
      print(d_1_dL)
      print("dy/dL:")
      print(d_g5_dL)

    return distance2, gradient2


def gelbrich_gradient_L(dist_1, dist_2, L, epsilon_reg=1e-8, debug=False,
                        wasserstein_type=2):
    """
    Compute the gradient of the Gelbrich distance with respect to the weight matrix L.

    Note: Takes wasserstein_type as input argument for compatibility with
    other gradient functions. Only wasserstein_type == 2 is allowed.

    d_Gelbrich = sqrt(d_Gelbrich^2), and as we can calculate
    d(d_Gelbrich^2)/dL, we can also calculate d(d_Gelbrich)/dL:

    d(d_Gelbrich)/dL = 1/(2 sqrt(d_Gelbrich^2)) * d(d_Gelbrich^2)/dL
                     = 1/(2 d_Gelbrich) * d(d_Gelbrich^2)/dL

    Parameters:
    - dist_1: tuple, (mu_P, Sigma_P), the first distribution
    - dist_2: tuple, (mu_Q, Sigma_Q), the second distribution
    - L: np.ndarray, shape (n, n)
    - epsilon_reg: float, small regularization term
    - debug: bool, whether to print debug information
    (- wasserstein_type: int, the type of Wasserstein distance)

    Returns:
    - distance: float, the value of the Gelbrich distance
    - grad: np.ndarray, shape (n, n), the gradient matrix
    """
    assert wasserstein_type == 2, "Only Wasserstein type 2 is allowed."
    dist2, grad2 = gelbrich_gradient_2_L(dist_1, dist_2, L, epsilon_reg, debug)
    distance = np.sqrt(dist2)
    distance = np.maximum(distance, epsilon_reg)
    grad = grad2 / (2 * distance)
    return distance, grad


def wc_moments_wc_gelbrich_risk(portf_weights, mean_ref,
                                cov_ref, epsilon,
                                alpha, L):
    """
    Compute the worst-case risk measure for a given portfolio,
    weight matrix L, and reference distribution.

    Parameters:
    - portf_weights: np.ndarray, shape (n,)
      - The portfolio weights.
    - mean_ref: np.ndarray, shape (n,)
      - The mean vector of the reference distribution.
    - cov_ref: np.ndarray, shape (n, n)
      - The covariance matrix of the reference distribution.
    - epsilon: float
      - Maximum Gelbrich distance defining the Gelbrich ball.
    - alpha: float
      - Standard risk coefficient. alpha = sqrt((1-beta)/beta) for beta being the risk level of CVaR.
    - L: np.ndarray, shape (n, n)
      - The transportation cost weight matrix.

    Returns:
    - worst_case_mean: np.ndarray, shape (n,)
      - The mean of the worst case distribution.
    - worst_case_cov: np.ndarray, shape (n, n)
      - The covariance matrix of the worst case distribution.
    """

    assert portf_weights.shape == mean_ref.shape, "Portfolio weights and mean vector must have the same shape."
    assert cov_ref.shape[0] == cov_ref.shape[1], "Covariance matrix must be square."
    assert cov_ref.shape[0] == portf_weights.shape[0], "Portfolio weights and covariance matrix must have the same shape."
    assert np.all(np.linalg.eigvals(cov_ref) > 0), "Covariance matrix must be positive definite."

    # Dimension of the sample space
    n_dim = portf_weights.shape[0]

    # outer product of the portfolio weights: w w^T
    portf_weights_outer = np.outer(portf_weights, portf_weights)

    # Mahalanobis matrix H = L L^T
    H_mat = L @ L.T
    assert H_mat.shape == (n_dim, n_dim), "H must be a square matrix of the same shape as the portfolio weights."
    assert np.all(np.linalg.eigvals(H_mat) > 0), "H must be positive definite."
    assert np.allclose(H_mat, H_mat.T), "H matrix must be symmetric"
    H_mat_inv = np.linalg.inv(H_mat)
    assert np.allclose(H_mat_inv, H_mat_inv.T), "H^{-1} matrix must be symmetric"

    portf_weights_norm_Hinv_2 = portf_weights.T @ H_mat_inv @ portf_weights
    portf_weights_norm_Hinv = np.sqrt(portf_weights_norm_Hinv_2)

    # standard deviation of the portfolio
    sqrt_w_Sigma_w = np.sqrt(portf_weights.T @ cov_ref @ portf_weights)

    # Optimal gamma, appears in derivation of primal reformulation
    gamma_star = np.sqrt(alpha**2 + 1) * portf_weights_norm_Hinv / (2*epsilon)
    assert gamma_star >= 0, "Gamma_star must be positive."

    # Optimal lambda, appears in derivation of primal reformulation
    lambda_star = portf_weights_norm_Hinv_2 / gamma_star + (2/alpha) * sqrt_w_Sigma_w
    lambda_star = 1 / lambda_star
    assert lambda_star >= 0, "Lambda_star must be positive."

    # Worst-case mean
    worst_case_mean = mean_ref - 1/(2*gamma_star) * H_mat_inv @ portf_weights

    # Worst-case covariance matrix
    matrix_pre = (H_mat_inv + lambda_star/(gamma_star * (1 - lambda_star/gamma_star * portf_weights_norm_Hinv_2)) * H_mat_inv @ portf_weights_outer @ H_mat_inv) @ H_mat
    matrix_post = H_mat @ (H_mat_inv + lambda_star/(gamma_star * (1 - lambda_star/gamma_star * portf_weights_norm_Hinv_2)) * H_mat_inv @ portf_weights_outer @ H_mat_inv)
    worst_case_cov = matrix_pre @ cov_ref @ matrix_post

    return worst_case_mean, worst_case_cov
