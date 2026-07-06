import torch
from ot import sinkhorn
from ot.optim import generic_conditional_gradient, line_search_armijo

import warnings


def gcg(
    a,
    b,
    M,
    reg1,
    reg2,
    f,
    df,
    G0=None,
    numItermax=10,
    numInnerItermax=200,
    stopThr=1e-9,
    stopThr2=1e-9,
    verbose=False,
    log=False,
    **kwargs,
):
    r"""
    Adapterd from ot.optim.gcg.
    Solve the general regularized OT problem with the generalized conditional
    gradient

        The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma
        \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1}\cdot\Omega(\gamma) + \mathrm{reg_2}\cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
    :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights
    (sum to 1)

    The algorithm used for solving the problem is the generalized conditional
    gradient as discussed in :ref:`[5, 7] <references-gcg>`


    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, (nt,)
        samples in the target domain
    M : array-like, shape (ns, nt)
        loss matrix
    reg1 : float
        Entropic Regularization term >0
    reg2 : float
        Second Regularization term >0
    G0 : array-like, shape (ns, nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations of Sinkhorn
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : ndarray, shape (ns, nt)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-gcg:
    References
    ----------

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport
    for Domain Adaptation," in IEEE Transactions on Pattern Analysis and
    Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized
    conditional gradient: analysis of convergence and applications. arXiv
    preprint arXiv:1510.06567.

    See Also
    --------
    ot.optim.cg : conditional gradient

    """

    def lp_solver(a, b, Mi, **kwargs):
        return sinkhorn(
            a, b, Mi, reg1, numItermax=numInnerItermax, log=True, **kwargs
        )

    def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
        return line_search_armijo(cost, G, deltaG, Mi, cost_G, **kwargs)

    return generic_conditional_gradient(
        a,
        b,
        M,
        f,
        df,
        reg2,
        None,
        lp_solver,
        line_search,
        G0=G0,
        numItermax=numItermax,
        stopThr=stopThr,
        stopThr2=stopThr2,
        verbose=verbose,
        log=log,
        **kwargs,
    )


def penalized_ot_solver(C, a, b, f, reg_constraints=1.0, eps=1.0, log=False,
                         eps_anneal=True, grad_clip=10.0):
    """
    Computes the optimal transport plan under rate constrains.
    Uses epsilon annealing: coarse solve with larger eps then refine.
    Parameters:
    ----------
    C (array-like): The cost matrix.
    a (array-like): The source weights.
    b (array-like): The target weights.
    f (callable): The function encoding the rate constraints.
    reg_constrains (float): Regularization parameter for the rate constrains.
    eps (float): Entropic regularization parameter.
    log (bool): If True, returns the log of the optimization process.
    eps_anneal (bool): If True, use two-stage epsilon annealing.
    grad_clip (float): Max gradient norm for fairness loss gradient clipping.
    Returns:
    -------
    if log is True:
        transport_plan (numpy.ndarray): The optimal transport plan.
        log (dict): A dictionary containing the optimization log.
    else:
        transport_plan (numpy.ndarray): The optimal transport plan.
    """
    # Convert inputs to CPU float64 torch tensors for consistent dtype
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().to(torch.float64)
        return torch.as_tensor(x, dtype=torch.float64)

    C_t = to_tensor(C)
    a_t = to_tensor(a)
    b_t = to_tensor(b)

    # Compute initial plan using sinkhorn with larger eps for smoother start
    init_eps = eps * 10.0 if eps_anneal else eps
    G0 = sinkhorn(a_t, b_t, C_t, init_eps,
                  method="sinkhorn_log",
                  numItermax=50000, warn=False)

    # Ensure G0 is a torch tensor
    if not isinstance(G0, torch.Tensor):
        G0 = torch.as_tensor(G0, dtype=torch.float64)

    # Wrap df with gradient clipping for stability
    raw_df = torch.func.grad(f)
    if grad_clip is not None and grad_clip > 0:
        def clipped_df(G):
            Grad = raw_df(G)
            grad_norm = Grad.norm()
            if grad_norm > grad_clip:
                Grad = Grad * (grad_clip / grad_norm)
            return Grad
        df_to_use = clipped_df
    else:
        df_to_use = raw_df

    if eps_anneal:
        # Stage 1: Coarse solve with larger epsilon (faster, smoother landscape)
        eps_coarse = eps * 10.0
        G0, _stage1_log = gcg(
            a_t, b_t, C_t, eps_coarse, reg_constraints,
            f, df_to_use,
            G0=G0, log=True, method="sinkhorn_log",
            numItermax=200, stopThr=1e-9,
            numInnerItermax=2000, verbose=False,
        )

    # Stage 2 (or only stage): Fine solve with target epsilon
    # Adaptive tolerance: looser for low penalty, tighter for high penalty
    adaptive_stopThr = 1e-12 * min(1.0, max(0.01, reg_constraints))

    transport_plan, log_out = gcg(
        a_t, b_t, C_t, eps, reg_constraints,
        f, df_to_use,
        G0=G0, log=log, method="sinkhorn_log",
        numItermax=2000, stopThr=adaptive_stopThr,
        numInnerItermax=10000, verbose=True,
    )
    if log:
        return transport_plan, _log_to_numpy(log_out)
    else:
        return transport_plan


def _log_to_numpy(log):
    log_numpy = {}
    for key in log:
        if torch.is_tensor(log[key]):
            log_numpy[key] = log[key].detach().cpu().numpy()
        if isinstance(log[key], list):
            log_numpy[key] = []
            for item in log[key]:
                if torch.is_tensor(item):
                    log_numpy[key].append(item.detach().cpu().numpy())
                else:
                    log_numpy[key].append(item)
        else:
            log_numpy[key] = log[key]
    return log_numpy


def fair_sinkhorn_knopp(
    a,
    b,
    M,
    S_X,
    S_Y,
    F,
    reg,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
    **kwargs,
):
    r"""
    Adapted from ot.sinkhorn in the Python Optimal Transport package.
    Solve the entropic regularization optimal transport problem under
    group fairness constraints and return the
    OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M}
        \rangle_F + \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2]
    <references-sinkhorn-knopp>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    F : array-like, shape (dim_a, dim_b)
        Fairness target matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials
        should be given (that is the logarithm of the u,v sinkhorn scaling
        vectors)

    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    .. _references-sinkhorn-knopp:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    if len(a) == 0:
        a = torch.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = torch.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = len(b)

    if log:
        log = {"err": []}

    n_s_X = S_X.unique().shape[0]
    n_s_Y = S_Y.unique().shape[0]

    # we assume that no distances are null except those of the diagonal of
    # distances
    u = torch.ones(dim_a) / dim_a
    v = torch.ones(dim_b) / dim_b
    L = torch.ones((n_s_X, n_s_Y)) / (n_s_X * n_s_Y)

    K = torch.exp(M / (-reg))

    # B = torch.zeros((n_s_X, n_s_Y, len(a), len(b)))
    # for i, j in product(range(n_s_X), range(n_s_Y)):
    #     B[i, j] = (S_X[:, None] == i) * (S_Y[None, :] == j)

    # T = torch.einsum("ij,ijkl->kl", L, B)
    T = L[S_X[:, None], S_Y[None, :]]
    K_T = K * T
    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        Lprev = L

        K_T_v = torch.matmul(K_T, v)
        u = a / K_T_v
        K_T_transpose_u = torch.matmul(K_T.T, u)
        v = b / K_T_transpose_u

        G = u[:, None] * K * v[None, :]
        G_agg = torch.zeros(n_s_X, dim_b, dtype=G.dtype)
        G_agg.scatter_add_(0, S_X[:, None].expand_as(G), G)
        phi_u_v = torch.zeros(n_s_X, n_s_Y, dtype=G.dtype)
        phi_u_v.scatter_add_(1, S_Y[None, :].expand(n_s_X, dim_b), G_agg)

        L = F / phi_u_v
        # T = torch.einsum("ij,ijkl->kl", L, B)
        T = L[S_X[:, None], S_Y[None, :]]
        K_T = K * T

        if (
            torch.any(K_T_transpose_u == 0)
            or torch.any(torch.isnan(u))
            or torch.any(torch.isnan(v))
            or torch.any(torch.isinf(u))
            or torch.any(torch.isinf(v))
            or torch.any(torch.isnan(L))
            or torch.any(torch.isinf(L))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Warning: numerical errors at iteration %d" % ii)
            u = uprev
            v = vprev
            L = Lprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
            tmp2 = torch.einsum("i,ij,j->j", u, K_T, v)
            err = torch.norm(tmp2 - b)  # violation of marginal

            if log:
                log["err"].append(err)
                # log["err_fairness"].append(err_fairness)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        "{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19
                    )
                print("{:5d}|{:8e}|".format(ii, err))
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to "
                "increase the number of iterations `numItermax` "
                "or the regularization parameter `reg`."
            )
    if log:
        log["niter"] = ii
        log["u"] = u
        log["v"] = v
        log["L"] = L

    K_T = K * T

    if log:
        return u.reshape((-1, 1)) * K_T * v.reshape((1, -1)), log
    else:
        return u.reshape((-1, 1)) * K_T * v.reshape((1, -1))
