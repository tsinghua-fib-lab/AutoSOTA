"""
Multiscale Quadratic Gromov-Wasserstein Solver

Implements a coarse-to-fine strategy for solving quadratic GW problems efficiently. Solves on a coarsened problem first,
then uses the solution to warm-start the full-resolution problem.
"""

from solvers.gromov_wasserstein.implementations.embedding_based.quadratic import QuadraticGW
from utils.implementation.multiscaling import coarsen


class MultiscaleQuadraticGW(QuadraticGW):
    """
    Quadratic GW solver using multiscale coarse-to-fine optimization.

    This solver accelerates convergence by:
        1. Creating coarsened versions of both point clouds via k-means clustering
        2. Solving the GW problem on the coarse representation
        3. Inferring initial dual variables and potential for the fine problem
        4. Solving the full-resolution problem with this warm start

    The coarse-to-fine approach can significantly reduce computation time,
    especially for large point clouds, while maintaining solution quality.

    Attributes:
        X (torch.Tensor): Source point cloud, shape (N, D_x)
        Y (torch.Tensor): Target point cloud, shape (M, D_y)
        a (torch.Tensor): Source distribution, shape (N,)
        b (torch.Tensor): Target distribution, shape (M,)
        ratio (float): Coarsening ratio (fraction of points to keep in coarse version)
        coarse_solver (QuadraticGW): Solver for the coarsened problem
    """

    def __init__(self, X, Y, a=None, b=None, ratio=0.1, **kwargs):
        """
        Initialize multiscale quadratic GW solver.

        Args:
            X: Source point cloud, shape (N, D_x)
            Y: Target point cloud, shape (M, D_y)
            a: Source distribution. If None, uniform
            b: Target distribution. If None, uniform
            ratio: Coarsening ratio - fraction of original points to keep
            **kwargs: Additional arguments passed to QuadraticGW solver

        """
        super().__init__(X=X, Y=Y, a=a, b=b, **kwargs)

        self.ratio = ratio

        X_coarse, a_coarse = coarsen(X, self.a, ratio)
        Y_coarse, b_coarse = coarsen(Y, self.b, ratio)

        self.coarse_solver = QuadraticGW(X=X_coarse, Y=Y_coarse, a=a_coarse, b=b_coarse, **kwargs.copy())

    def parameters(self):
        params = super().parameters()
        params['ratio'] = self.ratio

        return params

    def _infer_potentials(self):
        """
        Infer fine-level dual variables and potential from coarse solution.

        After solving the coarse problem, this method:
            1. Extracts dual variables (f_coarse, g_coarse) and potential Z from coarse solver
            2. Computes cross-scale cost matrices between fine and coarse levels
            3. Infers fine-level duals using Sinkhorn-like updates
            4. Transfers the potential matrix directly
        """
        X_coarse, Y_coarse = self.coarse_solver.X, self.coarse_solver.Y
        f_coarse, a_coarse = self.coarse_solver.sinkhorn_solver.f, self.coarse_solver.a
        g_coarse, b_coarse = self.coarse_solver.sinkhorn_solver.g, self.coarse_solver.b

        bcoarse_logs_j = self.sinkhorn_solver.wrap_tensor(b_coarse.log(), dim=1)
        C_ij = QuadraticGW(self.X, Y_coarse, self.a, b_coarse, Z=self.Z).cost_matrix()  # (N, Mc)

        f_new = - self.eps * (((g_coarse[None, :, None] - C_ij) / self.eps) + bcoarse_logs_j).logsumexp(dim=1).squeeze()

        acoarse_logs_i = self.sinkhorn_solver.wrap_tensor(a_coarse.log(), dim=0)
        C_ij = QuadraticGW(X_coarse, self.Y, a_coarse, self.b, Z=self.Z).cost_matrix()  # (Nc, M)

        g_new = - self.eps * (((f_coarse[:, None, None] - C_ij) / self.eps) + acoarse_logs_i).logsumexp(dim=0).squeeze()

        self.sinkhorn_solver.f, self.sinkhorn_solver.g = f_new, g_new
        self.Z, self.Z_new = self.coarse_solver.Z, self.coarse_solver.Z_new

    def solve(self, reset_sinkhorn_numItermax=True, **kwargs):
        """
        Solve using multiscale coarse-to-fine strategy.

        The solving process:
        1. Solve the coarse problem completely
        2. Infer fine-level initialization from coarse solution
        3. Solve the fine problem with warm start

        Args:
            reset_sinkhorn_numItermax: If True, reset Sinkhorn iterations after solving
            **kwargs: Additional arguments passed to parent solve method

        Returns:
            pandas.DataFrame or None: Solver statistics if monitoring enabled

        """
        self.coarse_solver.solve(reset_sinkhorn_numItermax=False, **kwargs)
        self._infer_potentials()

        init_sink_numItermax = self.sinkhorn_solver.numItermax

        if self.sink_adaptive_schedule:
            self.sinkhorn_solver.numItermax = self.coarse_solver.sinkhorn_solver.numItermax

        result = super().solve(**kwargs)

        if self.sink_adaptive_schedule and reset_sinkhorn_numItermax:
            self.sinkhorn_solver.numItermax = init_sink_numItermax

        return result
