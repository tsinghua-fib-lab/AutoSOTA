import numpy as np
import scipy as sp
from trainable_ot_dro.cones import *

# A conic problem reformulation takes some known problem (including its
# optimization structure and data) and reformulates it as a conic
# optimization problem.

# We have some functions here that take problem-specific data and return
# A, b, c, and cone. These are the data and structure of the conic
# optimization problem.


def reform_po_dro_gaussian_ref(mu_hat, Sigma_hat, beta, eps, L_W):
  """

  Reformulates a distributionally robust portfolio optimization problem
  with gaussian reference distribution as a conic optimization problem.

  We aim to minimize the loss, which in our case is l(xi) = - xi^T z
  where z is the portfolio and xi is the random return.

  We assume that the underlying risk measure is the conditional value at risk
  (CVaR) at level beta.

  R_{mathbb{Q}}(l) = mathbb{Q} CVaR_{beta}(l)
                    = inf_{t in R} [t + 1/beta E_{mathbb{Q}} [ max( l(xi) - t, 0 ) ] ]

  According to:
  Viet Anh Nguyen, Soroosh Shafiee, Damir Filipović, et al.
  "Mean-Covariance Robust Risk Measurement."
  2023. arXiv: 2112.09959 [q-fin.PM]. url: https://arxiv.org/abs/2112.09959.


  mathcal{G}_{eps}(mu_hat, Sigma_hat) is the Gelbrich ambiguity set
  and if we introduce H = L_W L_W^T, we have the Gelbrich ambiguity set defined
  by the Mahalanobis distance in the wasserstein distance as transportation
  cost or in the Mahalanobis-Gelbrich distance.
  ||x-y||_H = sqrt( (x-y)^T H (x-y) )

  We aim to solve the DRO problem:

  min_{z} sup_{Q in mathcal{G}_{eps}^{L_W}(mu_hat, Sigma_hat)} R_{Q}(-z^T xi)
  = min_{z} -mu_hat^T z + alpha sqrt(z^T Sigma_hat z) + epsilon sqrt(1 + alpha^2) norm(z)_{H^(-1)}

  where alpha = sqrt((1-beta)/beta) for the structural stable ambiguity set
  M_2 (set of distributions on B(R^k) with finite second moment)

  ### Parameters
  1. mu_hat: np.array
      - The mean of the returns (sample mean)
  2. Sigma_hat: np.array
      - The covariance matrix of the returns (sample covariance)
  3. beta: float
      - The confidence level of the CVaR
  4. eps: float
      - The radius of the ambiguity set
  5. L_W: np.array of shape (k, k) lower triangular and positive definite
      - The cost matrix for the Mahalanobis distance: H = L_W L_W^T, and the
      Mahalanobis distance is || L_W^T (x - y)||_2

  ### Returns a conic program as a dictionary with keys:
  - A: np.array
    - Matrix defining the equality constraints
  - b: np.array
    - Vector defining the equality constraints
  - c: np.array
    - Objective function coefficients
  - cone: ConvexCone
    - The cone structure of the problem
  - get_L: function
    - function that returns the matrix L_W given some A matrix
  - set_L: function
    - function that sets the matrix L_W in some A matrix and returns the new A matrix
  - get_decision: function
    - function that returns the portfolio weights from the solution vector x

  """

  # check for sizes
  assert mu_hat.shape[0] == Sigma_hat.shape[0]
  assert mu_hat.shape[0] == Sigma_hat.shape[1]
  assert mu_hat.shape[0] == L_W.shape[0]
  assert mu_hat.shape[0] == L_W.shape[1]

  # check Sigma is positive definite
  assert np.all(np.linalg.eigvals(Sigma_hat) > 0)

  # check L_W is lower triangular
  assert np.all(np.tril(L_W) == L_W)
  # Assuming L_W is lower triangular from Cholesky decomposition
  assert np.all(np.diag(L_W) > 0), "Matrix is not positive definite."

  k = mu_hat.shape[0]

  # standard risk coefficient for CVaR at level beta for M_2
  alpha = np.sqrt((1-beta)/beta)

  # the negative square root of the covariance matrix
  neg_sqrtSigma = -sp.linalg.sqrtm(Sigma_hat)
  # ensure it is real by taking only the real part as long as the imaginary part is small
  neg_sqrtSigma = np.real_if_close(neg_sqrtSigma)

  # ----------------------------------------
  # minimize c^T x
  # s.t. Ax + s = b
  #      s in K
  # ----------------------------------------

  # x = (z, u, v, q) in R^{2k + 2}
  # s = (0, 0, bar(z), bar(u), w, bar(v), bar(q)) in R^{3k + 3}

  c = np.zeros((2*k + 2, 1))
  c[:k, 0] = -mu_hat
  c[k, 0] = alpha
  c[k+1, 0] = eps * np.sqrt(1 + alpha**2)
  c[k+2:, 0] = np.zeros(k)

  A = np.zeros((4*k + 3, 2*k + 2))

  # 1^T z = 1
  A[0, :k] = np.ones(k)

  # z = L_W @ q
  A[1:k+1, :k] = -np.eye(k)
  A[1:k+1, k+2:] = L_W

  # bar(z) = z
  A[k+1:2*k+1, :k] = -np.eye(k)

  # u = bar(u)
  A[2*k+1, k] = -1

  # (Sigma_hat)^(-1/2) z = w
  A[2*k+2:3*k+2, :k] = neg_sqrtSigma

  # v = bar(v)
  A[3*k+2, k+1] = -1

  # q = bar(q)
  A[3*k+3:, k+2:] = -np.eye(k)

  # b = (1, 0, 0, 0, 0, 0, 0,..., 0)
  b = np.zeros((4*k + 3, 1))
  b[0, 0] = 1

  cone_zeros = ZeroCone(dim=k+1)
  cone_nonneg = NonnegativeCone(dim=k)
  cone_soc_1 = SecondOrderCone(dim=k+1)
  cone_soc_2 = SecondOrderCone(dim=k+1)
  cone = CartesianProductCone([cone_zeros, cone_nonneg, cone_soc_1, cone_soc_2])

  # indices as numpy arrays
  LW_indices_of_A = (np.arange(1, k+1, dtype=int), np.arange(k+2, k+2+k, dtype=int))

  def get_portfolio_weights(x_cp_output):
    return x_cp_output[:k]

  def get_LW_matrix(A_cp_output):
    return A_cp_output[np.ix_(LW_indices_of_A[0], LW_indices_of_A[1])]

  def set_LW_matrix(A_cp_output, LW_matrix):
    A_cp_output[np.ix_(LW_indices_of_A[0], LW_indices_of_A[1])] = LW_matrix
    return A_cp_output

  conic_program = {}
  conic_program["A"] = A
  conic_program["b"] = b
  conic_program["c"] = c
  conic_program["cone"] = cone
  conic_program["get_L"] = get_LW_matrix
  conic_program["set_L"] = set_LW_matrix
  conic_program["get_decision"] = get_portfolio_weights

  return conic_program


def reform_po_dro(samples, eps, L_weight, wasserstein_type=1):
  """

  Reformulates a distributionally robust portfolio optimization problem
  as a conic optimization problem.

  The distributionally robust portfolio optimization problem takes samples
  that give rise to an empirical uniform distribution over the samples.
  The problem assumes that the true distribution of the returns xi is
  within an epsilon Wasserstein ball of the empirical distribution.

  x in R^k is the portfolio over the k assets
  xi in R^k is the returns of the k assets (random, samples available)

  We observe J samples of the returns hat{xi}_j which give rise to
  the empirical distribution P_J = 1/J sum_{j=1}^J delta_{hat{xi}_j}
  and want to optimize the portfolio with respect to the worst-case return
  over the Wasserstein ball of radius epsilon around P_J

  # ----------------------------------------
  # minimize_{x in mathcal{X}} sup_{Q in B_eps(P_J)} E_{xi ~ Q}[- xi^T x]
  # s.t. mathcal{X} = {x in R^n | x >= 0, sum_i x_i = 1}
  #      P_J = 1/J sum_{j=1}^J delta_{hat{xi}_j}
  # ----------------------------------------

  We can reformulate this problem as a conic optimization problem
  see:
  1.
  Soroosh Shafieezadeh-Abadeh, Liviu Aolaritei, Florian Dörfler, et al.
  "New Perspectives on Regularization and Computation in Optimal Transport-
  Based Distributionally Robust Optimization".
  2023. arXiv: 2303.03900 [math.OC].
  url: https://arxiv.org/abs/2303.03900.
  or
  2.
  Peyman Mohajerin Esfahani and Daniel Kuhn.
  "Data-driven Distributionally Robust Optimization Using the Wasser-
  stein Metric: Performance Guarantees and Tractable Reformulations".
  2017. arXiv: 1505.05116 [math.OC].
  url: https://arxiv.org/abs/1505.05116.

  We use loss function
  l(x, xi) = - xi^T x
  and transportation cost
  c(xi_1, xi_2) = || L_weight^T (xi_1 - xi_2) ||_2
  and the Wasserstein type p defines the exponent of the transportation cost
  as well as the exponent of the whole metric, i.e.
  {}_p d_W^{L_weight}(Q_1, Q_2) = (inf_{pi} E_{(xi_1, xi_2) ~ pi} [c(xi_1, xi_2)^p])^{1/p}
  where pi is a joint distribution with marginals Q_1 and Q_2

  Furthermore, the probability distributions are supported on Xi = R^k
  and the empirical reference distribution is the uniform one, i.e. p_j = 1/J

  The problem can be reformulated as
  # ----------------------------------------
  # minimize c^T x
  # s.t. Ax + s = b
  #      s in K
  # ----------------------------------------

  ### Parameters
  1. samples: np.array of shape (J, k)
      - The samples of the returns
  2. eps: float
      - The radius of the Wasserstein ball
  3. L_weight: np.array of shape (k, k), lower triangular and positive definite
      - The cost matrix for the transportation cost: || L_weight^T (x - y)||_2 where
        x is a point of empirical samples and y is a point in R^k
  4. wasserstein_type: int
      - The exponent of the Wasserstein metric

  ### Returns a conic program as a dictionary with keys:
  - A: np.array
    - Matrix defining the equality constraints
  - b: np.array
    - Vector defining the equality constraints
  - c: np.array
    - Objective function coefficients
  - cone: ConvexCone
    - The cone structure of the problem
  - get_L: function
    - function that returns the matrix L_weight given some A matrix
  - set_L: function
    - function that sets the matrix L_weight in some A matrix and returns the new A matrix
  - get_decision: function
    - function that returns the portfolio weights from the solution vector x

  """

  assert samples.shape[1] == L_weight.shape[0]
  assert samples.shape[1] == L_weight.shape[1]
  assert len(samples.shape) == 2

  # check that L is lower triangular
  assert np.all(np.tril(L_weight) == L_weight)
  # Assuming L_weight is lower triangular from Cholesky decomposition
  assert np.all(np.diag(L_weight) > 0), "Matrix is not positive definite."

  assert eps > 0

  J = samples.shape[0]
  k = samples.shape[1]
  p = wasserstein_type

  if p == 1:
    # Wasserstein type 1 distance
    c1 = np.ones(J) / J
    c1 = c1.reshape(-1, 1)
    c2 = - np.sum(samples, axis=0) / J
    c2 = c2.reshape(-1, 1)
    c3 = np.array([eps])
    c3 = c3.reshape(-1, 1)
    c4 = np.zeros((k, 1))
    c = np.concatenate([c1, c2, c3, c4], axis=0)
    c = c.reshape(-1, 1)

    b = np.zeros((3*k + J + 2, 1))
    b[k, 0] = 1

    R_11 = np.zeros((k, J))
    R_12 = -np.eye(k)
    R_13 = np.zeros((k, 1))
    R_14 = L_weight
    R_21 = np.zeros((1, J))
    R_22 = np.ones((1, k))
    R_23 = np.zeros((1, 1))
    R_24 = np.zeros((1, k))
    R_1 = np.concatenate([R_11, R_12, R_13, R_14], axis=1)
    R_2 = np.concatenate([R_21, R_22, R_23, R_24], axis=1)
    R = np.concatenate([R_1, R_2], axis=0)

    A = np.concatenate([R, -np.eye(J + 2*k + 1)], axis=0)

    cone_zeros = ZeroCone(dim=k+1)
    cone_nonneg = NonnegativeCone(dim=k+J)
    cone_soc = SecondOrderCone(dim=k+1)
    cone = CartesianProductCone([cone_zeros, cone_nonneg, cone_soc])

    # indices as numpy arrays
    LW_indices_of_A = (np.arange(0, k, dtype=int), np.arange(J+k+1, J+k+1+k, dtype=int))
    portf_weight_indices_of_x = np.arange(J, J+k, dtype=int)

  elif p == 2:
    # Wasserstein type 2 distance
    c1 = np.ones(J) / J
    c1 = c1.reshape(-1, 1)
    c2 = - np.sum(samples, axis=0) / J
    c2 = c2.reshape(-1, 1)
    c3 = np.array([eps**2]) # squared epsilon
    c3 = c3.reshape(-1, 1)
    c4 = np.zeros((k, 1))
    c = np.concatenate([c1, c2, c3, c4], axis=0)

    b = np.zeros((2 + 2*k + J*(k+2), 1))
    b[0, 0] = 1

    R_11 = np.zeros((1, J))
    R_12 = np.ones((1, k))
    R_13 = np.zeros((1, 1))
    R_14 = np.zeros((1, k))
    R_1 = np.concatenate([R_11, R_12, R_13, R_14], axis=1)

    R_21 = np.zeros((k, J))
    R_22 = -np.eye(k)
    R_23 = np.zeros((k, 1))
    R_24 = L_weight
    R_2 = np.concatenate([R_21, R_22, R_23, R_24], axis=1)

    R_31 = np.zeros((k+1, J))
    R_32 = -np.eye(k+1)
    R_33 = np.zeros((k+1, k))
    R_3 = np.concatenate([R_31, R_32, R_33], axis=1)

    R_matrices_list = []

    for j in range(J):
      R_j = np.zeros((2+k, J + 2*k + 1))
      R_j[0, j] = -1
      R_j[2+k-1, j] = 1
      R_j[0, J + k] = -4
      R_j[2+k-1, J + k] = -4
      R_j[1:1+k, J+k+1:J+k+1+k] = -2*np.eye(k)
      R_matrices_list.append(R_j)

    R_matrices = np.concatenate(R_matrices_list, axis=0)

    A = np.concatenate([R_1, R_2, R_3, R_matrices], axis=0)

    cone_zeros = ZeroCone(dim=k+1)
    cone_nonneg = NonnegativeCone(dim=k+1)
    # now we need J second order cones of size k+2
    cones_soc = [SecondOrderCone(dim=k+2) for j in range(J)]
    cones = [cone_zeros, cone_nonneg] + cones_soc
    cone = CartesianProductCone(cones)

    # indices as numpy arrays
    LW_indices_of_A = (np.arange(1, k+1, dtype=int), np.arange(J+k+1, J+k+1+k, dtype=int))
    portf_weight_indices_of_x = np.arange(J, J+k, dtype=int)

  else:
    raise ValueError("Wasserstein type must be 1 or 2.")

  def get_portfolio_weights(x_cp_output):
    return x_cp_output[portf_weight_indices_of_x]

  def get_LW_matrix(A_cp_output):
    assert A_cp_output.shape[0] == A.shape[0]
    assert A_cp_output.shape[1] == A.shape[1]
    return A_cp_output[np.ix_(LW_indices_of_A[0], LW_indices_of_A[1])]

  def set_LW_matrix(A_cp_output, LW_matrix):
    assert A_cp_output.shape[0] == A.shape[0]
    assert A_cp_output.shape[1] == A.shape[1]
    A_cp_output[np.ix_(LW_indices_of_A[0], LW_indices_of_A[1])] = LW_matrix
    return A_cp_output

  conic_program = {}
  conic_program["A"] = A
  conic_program["b"] = b
  conic_program["c"] = c
  conic_program["cone"] = cone
  conic_program["get_L"] = get_LW_matrix
  conic_program["set_L"] = set_LW_matrix
  conic_program["get_decision"] = get_portfolio_weights

  return conic_program


def reform_linreg_dro(X, y, eps, L, wasserstein_type=1):
  """
  Reformulates a distributionally robust linear regression problem
  as a conic optimization problem.
  - If Wasserstein type 1 is used, we use the absolute error
      inf_w sup_{Q in B_eps^1(P_ref)} E_{(x,y) ~ Q}[|y-w^T x|]

  - and if Wasserstein type 2 is used, we can use the squared error
      inf_w sup_{Q in B_eps^2(P_ref)} E_{(x,y) ~ Q}[(y-w^T x)^2]

  where P_ref is the empirical distribution of the data samples
  and B_eps(P_ref) is the Wasserstein ball of radius epsilon around P_ref
  with respect to the Wasserstein distance defined by the cost matrix L.

  X in R^{N x k} is the design matrix
  y in R^N is the response vector

  Type-1 / Absolute error reformulations in:
  - Ruidi Chen, Ioannis Paschalidis: "Distributionally Robust Learning" (2020), Equation (4.5)

  Type-2 / Squared error reformulations in:
  - Blanchet et al.: "Robust Wasserstein Profile Inference and Applications
    to Machine Learning" (2019)
  """

  N, k = X.shape

  # check sizes
  assert y.shape[0] == N
  assert L.shape[0] == k+1
  assert L.shape[1] == k+1

  # check L is lower triangular
  assert np.all(np.tril(L) == L)
  # Assuming L is lower triangular from Cholesky decomposition
  assert np.all(np.diag(L) > 0), "Matrix is not positive definite."

  # min c^T x
  # s.t. Ax + s = b
  #      s in K


  if wasserstein_type == 1:
    # reformulation of:
    # min_w 1/N sum_{i=1}^N |y_i - w^T x_i| + eps || (-w, 1) ||_{(L L^T)^-1}

    # x in R^{2+2k+N}
    # s in R^{3+2k+2N}

    c1 = 1/N * np.ones((N, 1))
    c2 = eps * np.ones((1, 1))
    c3 = np.zeros((2*k + 1, 1))
    c = np.concatenate([c1, c2, c3], axis=0)

    b1 = np.zeros((k, 1))
    b2 = np.ones((1, 1))
    b3 = -y.reshape(-1, 1)
    b4 = y.reshape(-1, 1)
    b5 = np.zeros((k+2, 1))
    b = np.concatenate([b1, b2, b3, b4, b5], axis=0)

    A_11 = np.zeros((k+1, N))
    A_12 = np.zeros((k+1, 1))
    A_13 = L
    A_14 = np.zeros((k+1, k))
    for index in range(k):
      A_14[index, index] = 1
    A_1 = np.concatenate([A_11, A_12, A_13, A_14], axis=1)

    A_21 = -np.eye(N)
    A_22 = np.zeros((N, 1))
    A_23 = np.zeros((N, k+1))
    A_24 = -X
    A_2 = np.concatenate([A_21, A_22, A_23, A_24], axis=1)

    A_31 = -np.eye(N)
    A_32 = np.zeros((N, 1))
    A_33 = np.zeros((N, k+1))
    A_34 = X
    A_3 = np.concatenate([A_31, A_32, A_33, A_34], axis=1)

    A_41 = np.zeros((k+2, N))
    A_42 = -np.eye(k+2)
    A_43 = np.zeros((k+2, k))
    A_4 = np.concatenate([A_41, A_42, A_43], axis=1)

    A = np.concatenate([A_1, A_2, A_3, A_4], axis=0)

    cone_zeros = ZeroCone(dim=k+1)
    cone_nonneg = NonnegativeCone(dim=2*N)
    cone_soc = SecondOrderCone(dim=k+2)
    cone = CartesianProductCone([cone_zeros, cone_nonneg, cone_soc])

    def get_weights(x_cp_output):
      return x_cp_output[N+2+k:]

    def get_L_matrix(A_cp_output):
      return A_cp_output[:k+1, N+1:N+1+k+1]

    def set_L_matrix(A_cp_output, L_matrix):
      A_cp_output[:k+1, N+1:N+1+k+1] = L_matrix
      return A_cp_output

  elif wasserstein_type==2:
    # reformulation:
    # min_w [ (1/N sum_{i=1}^N (w^T x_i - y_i)^2)^{1/2} + eps ||(-w, 1)||_{(L L^T)^-1} ]^2
    # as the argument to the square is nonnegative, we can also minimize
    # the argument itself.
    # Later, we however need to make sure that we square the objective value.
    # Based on Blanchet et al.
    # "Robust Wasserstein Profile Inference and Applications to Machine Learning"

    # Xi is the matrix [X, y]
    XI = np.concatenate([X, y.reshape(-1, 1)], axis=1)

    c = np.zeros((2*k + 5, 1))
    c[0] = 1

    b = np.zeros((2*k+7+N, 1))
    b[0] = 1

    A_1 = np.zeros((1, 5+2*k))
    A_1[0, -1] = 1

    A_21 = np.zeros((k+1, 1))
    A_22 = np.zeros((k+1, 1))
    A_23 = np.zeros((k+1, 1))
    A_24 = L
    A_25 = -np.eye(k+1)
    A_2 = np.concatenate([A_21, A_22, A_23, A_24, A_25], axis=1)

    A_31 = -np.eye(1)
    A_32 = np.eye(1)
    A_33 = eps * np.eye(1)
    A_34 = np.zeros((1, k+1))
    A_35 = np.zeros((1, k+1))
    A_3 = np.concatenate([A_31, A_32, A_33, A_34, A_35], axis=1)

    A_41 = np.zeros((1, 1))
    A_42 = -np.eye(1)
    A_43 = np.zeros((1, 3 + 2*k))
    A_4 = np.concatenate([A_41, A_42, A_43], axis=1)

    A_51 = np.zeros((1, 2))
    A_52 = -np.eye(1)
    A_53 = np.zeros((1, 2 + 2*k))
    A_5 = np.concatenate([A_51, A_52, A_53], axis=1)

    A_61 = np.zeros((k+1, 3))
    A_62 = -np.eye(k+1)
    A_63 = np.zeros((k+1, k+1))
    A_6 = np.concatenate([A_61, A_62, A_63], axis=1)

    A_71 = np.zeros((1, 1))
    A_72 = -np.sqrt(N) * np.eye(1)
    A_73 = np.zeros((1, 3 + 2*k))
    A_7 = np.concatenate([A_71, A_72, A_73], axis=1)

    A_81 = np.zeros((N, 4+k))
    A_82 = -XI
    A_8 = np.concatenate([A_81, A_82], axis=1)

    A = np.concatenate([A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8], axis=0)

    cone_zeros = ZeroCone(dim=2+k)
    cone_nonneg = NonnegativeCone(dim=2)
    cone_soc1 = SecondOrderCone(dim=k+2)
    cone_soc2 = SecondOrderCone(dim=N+1)
    cone = CartesianProductCone([cone_zeros, cone_nonneg, cone_soc1, cone_soc2])

    def get_weights(x_cp_output):
      return -x_cp_output[k+4:-1]

    def get_L_matrix(A_cp_output):
      return A_cp_output[1:k+2, 3:k+4]

    def set_L_matrix(A_cp_output, L_matrix):
      A_cp_output[1:k+2, 3:k+4] = L_matrix
      return A_cp_output

  else:
    raise ValueError("Wasserstein type must be 1 or 2.")

  conic_program = {}
  conic_program["A"] = A
  conic_program["b"] = b
  conic_program["c"] = c
  conic_program["cone"] = cone
  conic_program["get_L"] = get_L_matrix
  conic_program["set_L"] = set_L_matrix
  conic_program["get_decision"] = get_weights

  return conic_program
