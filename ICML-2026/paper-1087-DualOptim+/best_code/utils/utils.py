import copy
import random

import numpy as np
import torch
import yaml

def get_model_identifiers_from_yaml(model_family):
    # path is model_configs.yaml

    model_configs = {}
    with open("config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]


def merge_dicts(a, b):
    """ Recursively merges dict b into a deep copy of dict a """
    # Create a deep copy of a to avoid modifying it in place
    a_copy = copy.deepcopy(a)

    for key, value in b.items():
        if key in a_copy:
            if isinstance(a_copy[key], dict) and isinstance(value, dict):
                a_copy[key] = merge_dicts(a_copy[key], value)
            elif isinstance(a_copy[key], list) and isinstance(value, list):
                a_copy[key] = a_copy[key] + value  # or use other logic to merge lists
            else:
                a_copy[key] = value  # Overwrite value from b into a_copy
        else:
            a_copy[key] = value

    return a_copy


def get_total_len(name, forget_rate):
    if name == "eval_real_author_wo_options.json":
        return 100
    elif name == "eval_real_world_wo_options.json":
        return 117
    elif name == "eval_log.json":
        return 300
    else:
        if forget_rate == "forget01":
            return 40
        elif forget_rate == "forget05":
            return 200
        else:
            return 300


def interleave(a, b, size):
    assert len(a) == len(b)
    assert size > 0
    c = []
    for i in range(0, len(a), size):
        c.extend(a[i : i + size])
        c.extend(b[i : i + size])
    return c


# PLEASE BE VERY VERY CAREFUL HERE
# This code, although takes num_processes as an argument, it in fact only supports num_processes=2
# Future improvement should support interleave for more than 2 processes
# also, small_bsz = large_bsz//4 is hardcoded, which is only true for our experiments
# because when we construct perturb and paraphrase data_loader, we set batch_size=large_bsz//4 specifically
def interleave_eval_result_dict(
    eval_result_dict, forget_rate, large_bsz, num_processes=2
):
    small_bsz = large_bsz // 4
    for k, v in eval_result_dict.items():
        # each v corresponds to one ckpt
        for metric, value in v.items():
            bsz = (
                small_bsz
                if "perturb" in metric or "paraphrase" in metric
                else large_bsz
            )
            total_len = get_total_len(k, forget_rate)
            # split in two
            a = value[0 : len(value) // 2]
            b = value[len(value) // 2 :]
            eval_result_dict[k][metric] = interleave(a, b, bsz)[:total_len]
    return eval_result_dict


def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import torch
import numpy as np


def orthogonal_project(A: torch.Tensor, B: torch.Tensor):
    """
    Projects matrix B onto the subspace orthogonal to the column space of matrix A.

    Args:
        A (torch.Tensor): The reference matrix (e.g., gradients from the retain set).
                          Shape: (m, n)
        B (torch.Tensor): The matrix to be projected (e.g., gradients from the forget set).
                          Shape: (m, p)

    Returns:
        torch.Tensor: The component of B that is orthogonal to A's column space.
                      Shape: (m, p)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both A and B must be 2D matrices.")
    if A.shape[0] != B.shape[0]:
        raise ValueError("Matrices A and B must have the same number of rows.")

    # Ensure A has more rows than columns
    flag = False
    if A.shape[0] < A.shape[1]:
        A = A.T
        B = B.T
        flag = True

    # Calculate the SVD of A.
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # Calculate the projection matrix onto the column space of A.
    P_A = U @ U.T

    # Project B onto the column space of A.
    B_projected_on_A = P_A @ B

    # The component of B orthogonal to A's column space is B minus its projection onto A's column space.
    B_orthogonal_to_A = B - B_projected_on_A

    if flag:
        B_orthogonal_to_A = B_orthogonal_to_A.T

    return B_orthogonal_to_A


def orthogonal_project_low_rank(A: torch.Tensor, B: torch.Tensor, k: int = None):
    """
    Projects matrix B onto the subspace orthogonal to the column space of matrix A,
    with explicit handling of division by zero in low-rank approximation.

    Args:
        A (torch.Tensor): The reference matrix (e.g., gradients from the retain set).
                          Shape: (m, n)
        B (torch.Tensor): The matrix to be projected (e.g., gradients from the forget set).
                          Shape: (m, p)
        k (int, optional): The number of singular values to retain for low-rank approximation.
                          If None, uses full-rank SVD.
        eps (float, optional): Small value to avoid division by zero. Default: 1e-10.

    Returns:
        torch.Tensor: The component of B that is orthogonal to A's column space.
                      Shape: (m, p)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both A and B must be 2D matrices.")
    if A.shape[0] != B.shape[0]:
        raise ValueError("Matrices A and B must have the same number of rows.")

    # Ensure A has more rows than columns
    flag = False
    if A.shape[0] < A.shape[1]:
        A = A.T
        B = B.T
        flag = True

    # Calculate the truncated SVD of A.
    R = generate_random_proj_matrix(A, k)
    U, S, Vh = svd_low_rank(A, R)

    # Calculate the projection matrix onto the column space of A.
    P_A = U @ U.T

    # Project B onto the column space of A.
    B_projected_on_A = P_A @ B

    # The component of B orthogonal to A's column space is B minus its projection onto A's column space.
    B_orthogonal_to_A = B - B_projected_on_A
    if flag:
        B_orthogonal_to_A = B_orthogonal_to_A.T

    return B_orthogonal_to_A


def orthogonal_project_random(A: torch.Tensor, B: torch.Tensor, k: int = None):
    """
    Projects matrix B onto the subspace orthogonal to the column space of matrix A,
    using random projection for dimensionality reduction.

    Args:
        A (torch.Tensor): The reference matrix (e.g., gradients from the retain set).
                          Shape: (m, n)
        B (torch.Tensor): The matrix to be projected (e.g., gradients from the forget set).
                          Shape: (m, p)
        k (int, optional): Target dimension for random projection. If None, defaults to n//2.
        eps (float, optional): Small value to avoid division by zero. Default: 1e-10.

    Returns:
        torch.Tensor: The component of B that is orthogonal to A's column space.
                      Shape: (m, p)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both A and B must be 2D matrices.")
    if A.shape[0] != B.shape[0]:
        raise ValueError("Matrices A and B must have the same number of rows.")

    # Ensure A has more rows than columns
    flag = False
    if A.shape[0] < A.shape[1]:
        A = A.T
        B = B.T
        flag = True

    # Generate a random projection matrix
    n = A.shape[1]
    if k is None:
        k = max(1, n // 2)  # Default to half the original dimension

    # Random Gaussian matrix for projection
    R = torch.randn(n, k, device=A.device) / np.sqrt(k)

    # Project A and B to lower dimension
    A_proj = A @ R

    # Compute pseudo-inverse of projected A with explicit handling of division by zero
    U, S, Vh = torch.linalg.svd(A_proj, full_matrices=False)

    P_A = U @ U.T
    # Project B onto the column space of A in the reduced dimension
    B_projected_on_A_proj = P_A @ B

    # Map back to original space and compute orthogonal component
    B_orthogonal_to_A = B - B_projected_on_A_proj

    if flag:
        B_orthogonal_to_A = B_orthogonal_to_A.T

    return B_orthogonal_to_A


def orthogonal_project_ns(A: torch.Tensor, B: torch.Tensor):
    """
    Projects matrix B onto the subspace orthogonal to the column space of matrix A.

    Args:
        A (torch.Tensor): The reference matrix (e.g., gradients from the retain set).
                          Shape: (m, n)
        B (torch.Tensor): The matrix to be projected (e.g., gradients from the forget set).
                          Shape: (m, p)

    Returns:
        torch.Tensor: The component of B that is orthogonal to A's column space.
                      Shape: (m, p)
    """
    def newtonschulz5(G, steps=5, eps=1e-7):
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        # X = G.bfloat16()
        X = G
        X /= (X.norm() + eps)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both A and B must be 2D matrices.")
    if A.shape[0] != B.shape[0]:
        raise ValueError("Matrices A and B must have the same number of rows.")

    # Ensure A has more rows than columns
    flag = False
    if A.shape[0] < A.shape[1]:
        A = A.T
        B = B.T
        flag = True

    A_ortho = newtonschulz5(A)
    P_A = A_ortho @ A_ortho.T

    # Project B onto the column space of A.
    B_projected_on_A = P_A @ B

    # The component of B orthogonal to A's column space is B minus its projection onto A's column space.
    B_orthogonal_to_A = B - B_projected_on_A
    if flag:
        B_orthogonal_to_A = B_orthogonal_to_A.T
    return B_orthogonal_to_A


def conjugate(A):
    """Return conjugate of tensor A.

    .. note:: If A's dtype is not complex, A is returned.
    """
    if A.is_complex():
        return A.conj()
    return A

def transpose(A):
    """Return transpose of a matrix or batches of matrices."""
    ndim = len(A.shape)
    return A.transpose(ndim - 1, ndim - 2)


def transjugate(A):
    """Return transpose conjugate of a matrix or batches of matrices."""
    return conjugate(transpose(A))


def generate_random_proj_matrix(X, q, dtype=torch.float32, seed=42):
    assert X.ndim == 2, "X must be a 2D matrix"
    if X.shape[0] < X.shape[1]:
        X = X.T
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.randn(X.shape[1], q, device=X.device, dtype=dtype)


def svd_random_proj(X, R, dtype=torch.float32):
    torch.backends.cuda.preferred_linalg_library("magma")
    X_float = X.to(dtype)
    assert X_float.ndim == 2, "X must be a 2D matrix"
    if  X_float.shape[0] <  X_float.shape[1]:
        X_float = X_float.T
    # Project X to lower dimension
    X_proj = X_float @ R
    U, S, Vh = torch.linalg.svd(X_proj, full_matrices=False)
    return U, S, Vh


def get_approximate_basis(A, R, niter=2, M=None):
    """Return tensor :math:`Q` with :math:`q` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is
    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al, 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              choosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        R (Tensor): the random projection matrix of size :math:`(n, q)`

        niter (int, optional): the number of subspace iterations to
                               conduct; ``niter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, 1, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """
    torch.backends.cuda.preferred_linalg_library("magma")
    niter = 2 if niter is None else niter
    matmul = torch.matmul

    A_H = transjugate(A)
    if M is None:
        Q = torch.linalg.qr(matmul(A, R)).Q
        for i in range(niter):
            Q = torch.linalg.qr(matmul(A_H, Q)).Q
            Q = torch.linalg.qr(matmul(A, Q)).Q
    else:
        M_H = transjugate(M)
        Q = torch.linalg.qr(matmul(A, R) - matmul(M, R)).Q
        for i in range(niter):
            Q = torch.linalg.qr(matmul(A_H, Q) - matmul(M_H, Q)).Q
            Q = torch.linalg.qr(matmul(A, Q) - matmul(M, Q)).Q
    return Q


def _svd_lowrank(
        A: torch.Tensor,
        R: torch.Tensor,
        niter=2,
        M=None,
):
    torch.backends.cuda.preferred_linalg_library("magma")
    q = R.shape[-1]
    m, n = A.shape[-2:]
    matmul = torch.matmul
    if M is None:
        M_t = None
    else:
        M_t = M.T
    A_t = A.T

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    if m < n or n > q:
        # computing the SVD approximation of a transpose in
        # order to keep B shape minimal (the m < n case) or the V
        # shape small (the n > q case)
        Q = get_approximate_basis(A_t, R, niter=niter, M=M_t)
        Q_c = conjugate(Q)
        if M is None:
            B_t = matmul(A, Q_c)
        else:
            B_t = matmul(A, Q_c) - matmul(M, Q_c)
        assert B_t.shape[-2] == m, (B_t.shape, m)
        assert B_t.shape[-1] == q, (B_t.shape, q)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = torch.linalg.svd(B_t, full_matrices=False)
        V = Vh.mH
        V = Q.matmul(V)
    else:
        Q = get_approximate_basis(A, q, niter=niter, M=M)
        Q_c = conjugate(Q)
        if M is None:
            B = matmul(A_t, Q_c)
        else:
            B = matmul(A_t, Q_c) - matmul(M_t, Q_c)
        B_t = B.T
        assert B_t.shape[-2] == q, (B_t.shape, q)
        assert B_t.shape[-1] == n, (B_t.shape, n)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = torch.linalg.svd(B_t, full_matrices=False)
        V = Vh.mH
        U = Q.matmul(U)

    return U, S, V


def svd_low_rank(X, R, dtype=torch.float32):
    torch.backends.cuda.preferred_linalg_library("magma")
    if X.dtype is not torch.float32:
        X_float = X.to(dtype)
    else:
        X_float = X
    assert X_float.ndim == 2, "X must be a 2D matrix"
    if  X_float.shape[0] <  X_float.shape[1]:
        X_float = X_float.T

    U, S, Vh = _svd_lowrank(X_float, R)
    return U, S, Vh


if __name__ == "__main__":
    import time

    # torch.backends.cuda.preferred_linalg_library("magma")
    # Example usage
    # generate a low rank matrix A
    in_dim = 1280
    out_dim = 1024
    rank = 36


    # A = torch.randn(in_dim, rank, dtype=torch.float32).cuda() @ torch.randn(rank, out_dim, dtype=torch.float32).cuda()
    # B = torch.randn(in_dim, rank, dtype=torch.float32).cuda() @ torch.randn(rank, out_dim, dtype=torch.float32).cuda()
    A = torch.randn(in_dim, rank, dtype=torch.float32) @ torch.randn(rank, out_dim, dtype=torch.float32)
    B = torch.randn(in_dim, rank, dtype=torch.float32) @ torch.randn(rank, out_dim, dtype=torch.float32)
    # R = generate_random_proj_matrix(A, rank+5)
    # R2 = torch.randn(out_dim, rank+5, device=A.device, dtype=torch.float32)

    # start = time.time()
    # U, S, Vh = svd_random_proj(A, R2)
    # torch.cuda.synchronize()
    # print(time.time() - start)
    # print(U.shape, S.shape, Vh.shape)
    #
    # start = time.time()
    # U, S, Vh = svd_low_rank(A, R)
    # torch.cuda.synchronize()
    # print(time.time() - start)

    # # record the time cost
    # start = time.time()
    # B_o_A = orthogonal_project(A, B)
    # print("Standard:", time.time() - start)
    #
    start = time.time()
    B_o_A_low_rank = orthogonal_project_low_rank(A, B, k=rank+5)
    print("Low rank:", time.time() - start)
    #
    # start = time.time()
    # B_o_A_random = orthogonal_project_random(A, B, k=rank+5)
    # print("Random projection:", time.time() - start)
    #
    # # start = time.time()
    # # B_o_A_ns = orthogonal_project_ns(A, B)
    # # print("Newton-Schulz:", time.time() - start)
    #
    # print(torch.linalg.norm(A.T @ B_o_A if in_dim >= out_dim else A @ B_o_A.T, ord='fro') / (in_dim*out_dim))
    print(torch.linalg.norm(A.T @ B_o_A_low_rank if in_dim >= out_dim else A @ B_o_A_low_rank.T, ord='fro') / (in_dim*out_dim))
    # print(torch.linalg.norm(A.T @ B_o_A_random if in_dim >= out_dim else A @ B_o_A_random.T, ord='fro') / (in_dim*out_dim))
    # # print(torch.linalg.norm(A.T @ B_o_A_ns, ord='fro') / (in_dim*out_dim))



