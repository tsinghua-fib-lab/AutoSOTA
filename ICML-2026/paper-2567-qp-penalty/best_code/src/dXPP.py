"""dXPPLayer: differentiable QP layer using penalty smoothing."""

import numpy as np
import torch
from torch import nn
import qpsolvers

from src.sparse_utils import is_sparse_tensor, torch_sparse_to_scipy, _SPARSE_SOLVER
from src.qp_utils import (
    set_solver_tolerance,
    _DUAL_AVAILABLE_SOLVERS,
    _select_default_qp_solver,
    _compute_multipliers_from_kkt,
)
from src.penalty_smooth_qp import PenaltySmoothQP


class dXPPLayer(nn.Module):
    def __init__(self, beta=1e-4, penalty_coeff=10, eps_abs=1e-6, eps_rel=0,
                 sparse_mode=None, lin_solver=None, warm_start=True, verbose=False,
                 solve_type=None, qp_solver=None):
        """
        Differentiable QP layer using penalty smoothing.

        Args:
            beta: Smoothing parameter for penalty.
            penalty_coeff: Penalty coefficient multiplier.
            eps_abs: Absolute tolerance for QP solver.
            eps_rel: Relative tolerance for QP solver.
            sparse_mode: Deprecated, use solve_type instead.
            lin_solver: Linear solver for backward pass (pardiso/qdldl/cholmod/scipy).
            warm_start: Whether to warm-start the QP solver.
            verbose: Print debug info.
            solve_type: "dense", "sparse", "auto"/None.
                - dense => QP solver defaults to cvxopt
                - sparse => QP solver defaults to gurobi
                - auto/None => auto-detect from input tensor layout at runtime
            qp_solver: Explicitly specify QP solver name (e.g. "gurobi", "cvxopt", "piqp", ...).
                If None, auto-selected based on solve_type.
        """
        super().__init__()
        self.beta = beta
        self.penalty_coeff = penalty_coeff

        # Normalize solve_type
        if isinstance(solve_type, str):
            solve_type = solve_type.strip().lower()
            if solve_type == "auto":
                solve_type = None
        if solve_type not in (None, "dense", "sparse"):
            raise ValueError(f"solve_type must be one of {{None, 'auto', 'dense', 'sparse'}}, got {solve_type!r}")
        self.solve_type = solve_type
        self.sparse_mode = sparse_mode
        self.lin_solver = lin_solver
        self.warm_start = warm_start
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.verbose = verbose

        # ===== QP solver selection =====
        if qp_solver is not None:
            # User explicitly specified a solver
            self.qp_solver = qp_solver
        elif solve_type == "dense":
            self.qp_solver = _select_default_qp_solver("dense")
        elif solve_type == "sparse":
            self.qp_solver = _select_default_qp_solver("sparse")
        else:
            # solve_type is auto => defer solver selection to runtime
            self.qp_solver = None

        # ===== Solver keyword arguments via set_solver_tolerance =====
        if self.qp_solver is not None:
            self.solver_kwargs = set_solver_tolerance({}, self.qp_solver, eps_abs, eps_rel)
        else:
            self.solver_kwargs = None  # will be built at runtime

        # ===== Dual availability =====
        if self.qp_solver is not None:
            self.dual_available = self.qp_solver in _DUAL_AVAILABLE_SOLVERS
        else:
            self.dual_available = None  # determined at runtime

        self._x_warm = None

        if verbose:
            print(f"[dXPPLayer] solve_type={self.solve_type or 'auto'}, "
                  f"qp_solver={self.qp_solver or 'auto'}, "
                  f"dual_available={self.dual_available}, "
                  f"sparse linear solver: {lin_solver or _SPARSE_SOLVER}")

    def _resolve_solver(self, use_sparse):
        """Resolve QP solver, kwargs and dual_available at runtime when solve_type is auto.
        
        Also handles the case where qp_solver is set after __init__ (e.g. layer.qp_solver = "piqp").
        """
        if self.qp_solver is not None:
            solver_kwargs = self.solver_kwargs
            if solver_kwargs is None:
                # qp_solver was set after __init__, generate kwargs now
                solver_kwargs = set_solver_tolerance({}, self.qp_solver, self.eps_abs, self.eps_rel)
            dual_available = self.qp_solver in _DUAL_AVAILABLE_SOLVERS
            return self.qp_solver, solver_kwargs, dual_available
        # Auto-select based on runtime sparse detection
        qp_solver = _select_default_qp_solver("sparse" if use_sparse else "dense")
        solver_kwargs = set_solver_tolerance({}, qp_solver, self.eps_abs, self.eps_rel)
        dual_available = qp_solver in _DUAL_AVAILABLE_SOLVERS
        return qp_solver, solver_kwargs, dual_available

    def forward(self, Q, q, G, h, A=None, b=None):
        if not self.training and not torch.is_grad_enabled():
            return self._forward_eval_mode(Q, q, G, h, A, b)

        has_eq = A is not None and b is not None
        if self.solve_type == "sparse":
            use_sparse = True
        elif self.solve_type == "dense":
            use_sparse = False
        else:
            use_sparse = is_sparse_tensor(Q) or is_sparse_tensor(G) or (has_eq and is_sparse_tensor(A))

        # Resolve solver at runtime
        qp_solver, solver_kwargs, dual_available = self._resolve_solver(use_sparse)

        x_star, mu_star, nu_star = PenaltySmoothQP.apply(
            Q, q, G, h, A, b,
            qp_solver, solver_kwargs, self.beta, self.penalty_coeff,
            use_sparse, self.lin_solver, self._x_warm if self.warm_start else None, self.verbose,
            dual_available
        )

        if self.warm_start:
            self._x_warm = x_star.detach().cpu().numpy().copy()

        return x_star, mu_star, nu_star
    
    def _forward_eval_mode(self, Q, q, G, h, A, b):
        has_eq = A is not None and b is not None
        if self.solve_type == "sparse":
            use_sparse = True
        elif self.solve_type == "dense":
            use_sparse = False
        else:
            use_sparse = is_sparse_tensor(Q) or is_sparse_tensor(G) or (has_eq and is_sparse_tensor(A))

        # Resolve solver at runtime
        qp_solver, solver_kwargs, dual_available = self._resolve_solver(use_sparse)

        if use_sparse:
            Q_np = torch_sparse_to_scipy(Q).tocsc()
            G_np = torch_sparse_to_scipy(G).tocsc()
            A_np = torch_sparse_to_scipy(A).tocsc() if has_eq else None
        else:
            Q_np = Q.detach().cpu().numpy()
            G_np = G.detach().cpu().numpy()
            A_np = A.detach().cpu().numpy() if has_eq else None

        q_np, h_np = q.detach().cpu().numpy(), h.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy() if has_eq else None

        nIneq = G.size(0)
        nEq = A.size(0) if has_eq else 0

        problem = qpsolvers.Problem(P=Q_np, q=q_np, G=G_np, h=h_np, A=A_np, b=b_np)
        solution = qpsolvers.solve_problem(problem, solver=qp_solver,
                                         initvals=self._x_warm if self.warm_start else None,
                                         verbose=self.verbose, **solver_kwargs)

        if solution.x is None:
            raise RuntimeError(f"QP solver '{qp_solver}' failed to return a solution!")

        if self.warm_start:
            self._x_warm = solution.x.copy()

        x_star_np = solution.x

        # ===== Extract or compute Lagrange multipliers =====
        mu_star_np = solution.y if (has_eq and solution.y is not None) else None
        nu_star_np = solution.z if solution.z is not None else None

        # If solver didn't return duals, compute from KKT
        need_kkt = False
        if nu_star_np is None or (has_eq and mu_star_np is None):
            need_kkt = True
        if nu_star_np is not None and np.all(nu_star_np == 0) and not dual_available:
            need_kkt = True

        if need_kkt:
            nu_kkt, mu_kkt = _compute_multipliers_from_kkt(
                Q_np, q_np, G_np, h_np, A_np, x_star_np,
                has_eq, use_sparse, nIneq, nEq
            )
            if nu_star_np is None:
                nu_star_np = nu_kkt
            if has_eq and mu_star_np is None:
                mu_star_np = mu_kkt
            if self.verbose:
                print(f"[dXPPLayer._forward_eval_mode] Computed multipliers from KKT "
                      f"(solver '{qp_solver}' did not return duals)")

        if nu_star_np is None:
            nu_star_np = np.zeros(nIneq)
        if mu_star_np is None and has_eq:
            mu_star_np = np.zeros(nEq)

        x_star = torch.from_numpy(x_star_np).to(Q.dtype).to(Q.device)
        mu_star = torch.from_numpy(mu_star_np).to(Q.dtype).to(Q.device) if has_eq else None
        nu_star = torch.from_numpy(nu_star_np).to(Q.dtype).to(Q.device)

        return x_star, mu_star, nu_star
    
    def reset_warm_start(self):
        self._x_warm = None

    @staticmethod
    def get_available_sparse_solvers():
        solvers = ["scipy"]
        try: import pypardiso; solvers.append("pardiso")
        except ImportError: pass
        try: import qdldl; solvers.append("qdldl")
        except ImportError: pass
        try: from sksparse.cholmod import cholesky; solvers.append("cholmod")
        except ImportError: pass
        return solvers

    @staticmethod
    def get_default_sparse_solver():
        return _SPARSE_SOLVER

    @staticmethod
    def get_available_qp_solvers():
        """Return all available QP solvers from qpsolvers."""
        return qpsolvers.available_solvers

    @staticmethod
    def get_default_qp_solver(solve_type="dense"):
        """Return the default QP solver for the given solve_type."""
        return _select_default_qp_solver(solve_type)
