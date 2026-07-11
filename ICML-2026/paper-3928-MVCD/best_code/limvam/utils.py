import numpy as np
from scipy.linalg import block_diag


def find_order(B):
    """
    This function finds a permutation matrix P such that P @ B @ P.T
    is as close as possible to strictly lower triangular.

    Parameters
    ----------
    B : ndarray, shape (p, p)
        Causal effect matrix, whose rows and columns will be permuted
        by a permutation P. We assume that ``B`` is such that 
        P @ B @ P.T is close to strictly lower triangular.

    Returns
    -------
    order: ndarray, shape (p,)
        ``order`` represents the permutation in P.
    """
    p = len(B)
    B_sort = np.sort(B, axis=1)[:, ::-1]
    B_argsort = np.argsort(B_sort, axis=0)
    order = []
    for i in range(p):
        col = B_argsort[:, i]
        available_id = ~np.isin(col, order)
        first_id = np.argmax(available_id)
        order.append(col[first_id])
    return order


def _search_causal_order(matrix):
    """
    Obtain a causal order from the given matrix strictly.
    This function comes from https://github.com/cdt15/lingam/blob/master/lingam/ica_lingam.py

    Parameters
    ----------
    matrix : array-like, shape (n_features, n_samples)
        Target matrix.

    Return
    ------
    causal_order : array, shape [n_features, ]
        A causal order of the given matrix on success, None otherwise.
    """
    causal_order = []

    row_num = matrix.shape[0]
    original_index = np.arange(row_num)

    while 0 < len(matrix):
        # find a row all of which elements are zero
        row_index_list = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
        if len(row_index_list) == 0:
            break

        target_index = row_index_list[0]

        # append i to the end of the list
        causal_order.append(original_index[target_index])
        original_index = np.delete(original_index, target_index, axis=0)

        # remove the i-th row and the i-th column from matrix
        mask = np.delete(np.arange(len(matrix)), target_index, axis=0)
        matrix = matrix[mask][:, mask]

    if len(causal_order) != row_num:
        causal_order = None

    return causal_order


def _estimate_causal_order(matrix):
    """
    Obtain a lower triangular from the given matrix approximately.
    This function comes from https://github.com/cdt15/lingam/blob/master/lingam/ica_lingam.py

    Parameters
    ----------
    matrix : array-like, shape (n_features, n_samples)
        Target matrix.

    Return
    ------
    causal_order : array, shape [n_features, ]
        A causal order of the given matrix on success, None otherwise.
    """
    causal_order = None

    # set the m(m + 1)/2 smallest(in absolute value) elements of the matrix to zero
    pos_list = np.argsort(np.abs(matrix), axis=None)
    pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T
    initial_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)
    for i, j in pos_list[:initial_zero_num]:
        matrix[i, j] = 0

    for i, j in pos_list[initial_zero_num:]:
        causal_order = _search_causal_order(matrix)
        if causal_order is not None:
            break
        else:
            # set the smallest(in absolute value) element to zero
            matrix[i, j] = 0

    return causal_order


def estimate_triangular_matrices_Ti(X):
    """
    Estimate the strictly lower-triangular matrices T^i for each of m views,
    given data X of shape (m, p, n), assuming variables are ordered so that
    B^i = T^i (strictly lower-triangular).
    
    Uses one-step Feasible GLS (SUR) per row j = 1..p:
      1. OLS on each view to get residuals.
      2. Estimate cross-view noise covariance Σ_j.
      3. Run GLS to get joint estimates for T^i_{j,1:(j-1)}.
    """
    m, p, n = X.shape
    # Initialize list of T^i estimates
    Ts = [np.zeros((p, p)) for _ in range(m)]
    
    for j in range(p):
        # 1) Collect responses and parent matrices
        Ys = [X[i, j, :] for i in range(m)]                         # length‐m list of (n,)
        Xs_par = [X[i, :j, :].T if j > 0 else np.zeros((n, 0))      # each (n, j)
                  for i in range(m)]
        
        # 2) Initial OLS residuals
        residuals = []
        for i in range(m):
            Xpj, yj = Xs_par[i], Ys[i]
            if j > 0:
                beta_ols, *_ = np.linalg.lstsq(Xpj, yj, rcond=None)
                ei = yj - Xpj.dot(beta_ols)
            else:
                beta_ols = np.zeros(0)
                ei = yj
            residuals.append(ei)
        
        # 3) Estimate Σ_j (m x m)
        Σ_j = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                Σ_j[a, b] = residuals[a].dot(residuals[b]) / n
        
        # 4) Build big design and response
        X_big = block_diag(*Xs_par)        # (n*m) x (j*m)
        Y_big = np.concatenate(Ys, axis=0) # (n*m,)
        
        # 5) Compute weight W = inv(Σ_j) ⊗ I_n
        # SVD-based pseudoinverse with condition number threshold (IDEA-03)
        U, s, Vt = np.linalg.svd(Σ_j)
        rcond = 1e-10
        s_inv = np.where(s > rcond * s.max(), 1.0 / s, 0.0)
        Σ_j_inv = (Vt.T * s_inv) @ U.T
        W = np.kron(Σ_j_inv, np.eye(n))
        
        # 6) Feasible GLS estimate
        XtW = X_big.T.dot(W)
        beta_gls = np.linalg.solve(XtW.dot(X_big), XtW.dot(Y_big))
        
        # 7) Assign back into each T^i
        for i in range(m):
            start = i * j
            end = start + j
            Ts[i][j, :j] = beta_gls[start:end]
    
    return np.array(Ts)
