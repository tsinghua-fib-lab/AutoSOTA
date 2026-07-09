"""PenaltySmoothQP: custom torch.autograd.Function for differentiable QP solving."""

import numpy as np
import torch
import qpsolvers
import scipy.sparse as spa
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from scipy.special import expit as sigmoid

from src.sparse_utils import sparse_solve_spd, is_sparse_tensor, torch_sparse_to_scipy
from src.qp_utils import _compute_multipliers_from_kkt


class PenaltySmoothQP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, q, G, h, A, b, qp_solver, solver_kwargs, beta, penalty_coeff,
                use_sparse, lin_solver, x_warm, verbose, dual_available):
        dim = Q.size(0)
        nIneq = G.size(0)
        has_eq = A is not None and b is not None
        nEq = A.size(0) if has_eq else 0

        if verbose:
            print(f"[PenaltySmoothQP.forward] Problem size: dim={dim}, nIneq={nIneq}, nEq={nEq}, "
                  f"solver={qp_solver}, dual_available={dual_available}")

        # ========== Data conversion ==========
        if use_sparse:
            Q_np = torch_sparse_to_scipy(Q).tocsc()
            G_np = torch_sparse_to_scipy(G).tocsc()
            A_np = torch_sparse_to_scipy(A).tocsc() if has_eq else None
        else:
            Q_np = Q.detach().cpu().numpy()
            G_np = G.detach().cpu().numpy()
            A_np = A.detach().cpu().numpy() if has_eq else None

        q_np = q.detach().cpu().numpy()
        h_np = h.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy() if has_eq else None

        # ========== QP Solve ==========
        problem = qpsolvers.Problem(P=Q_np, q=q_np, G=G_np, h=h_np, A=A_np, b=b_np)
        solution = qpsolvers.solve_problem(problem, solver=qp_solver, initvals=x_warm,
                                           verbose=verbose, **solver_kwargs)

        if solution.x is None:
            raise RuntimeError(f"QP solver '{qp_solver}' failed to return a solution! "
                               f"Problem size: dim={dim}, nIneq={nIneq}, nEq={nEq}")

        x_star_np = solution.x

        # ========== Extract or compute Lagrange multipliers ==========
        mu_star_np = solution.y if (has_eq and solution.y is not None) else None
        nu_star_np = solution.z if solution.z is not None else None

        # Check if we need to compute multipliers from KKT
        need_kkt = False
        if nu_star_np is None or (has_eq and mu_star_np is None):
            need_kkt = True
        # Also check if solver is not known to provide duals and returned all zeros
        if not dual_available:
            if nu_star_np is not None and np.all(nu_star_np == 0):
                need_kkt = True
            if has_eq and mu_star_np is not None and np.all(mu_star_np == 0):
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
            if verbose:
                print(f"[PenaltySmoothQP.forward] Computed multipliers from KKT "
                      f"(solver '{qp_solver}' did not return duals)")

        if nu_star_np is None:
            nu_star_np = np.zeros(nIneq)
        if mu_star_np is None and has_eq:
            mu_star_np = np.zeros(nEq)

        if verbose:
            print(f"[PenaltySmoothQP.forward] Solution obtained: ||x||={np.linalg.norm(x_star_np):.4f}, "
                  f"||nu||={np.linalg.norm(nu_star_np):.4f}"
                  + (f", ||mu||={np.linalg.norm(mu_star_np):.4f}" if has_eq else ""))

        # ========== Differentiation setup ==========
        # Compute rho values
        rho_ineq = penalty_coeff * np.sum(np.abs(nu_star_np))
        rho_eq = penalty_coeff * np.sum(np.abs(mu_star_np)) if has_eq else 0.0

        # Compute active set
        eps_active = 1e-5
        r_pri_np = h_np - G_np @ x_star_np
        active_mask = r_pri_np < eps_active
        nActiveIneq = np.sum(active_mask)
        
        if nActiveIneq < nIneq:
            if verbose:
                print(f"[PenaltySmoothQP.forward] Active constraints: {nActiveIneq}/{nIneq}")
            G_active = G_np[active_mask, :]
            h_active = h_np[active_mask]
            r_pri_active = r_pri_np[active_mask]
        else:
            G_active = G_np
            h_active = h_np
            r_pri_active = r_pri_np

        # Compute sigmoid values for penalty smoothing
        z_ineq = (-r_pri_active) / beta
        sigmoid_ineq = sigmoid(z_ineq)
        
        # Precompute sigmoid derivatives and weights (moved from backward)
        sigmoid_deriv_ineq = sigmoid_ineq * (1.0 - sigmoid_ineq)
        weight_ineq = (rho_ineq / beta) * sigmoid_deriv_ineq if (rho_ineq > 1e-12 and nActiveIneq > 0) else None

        if has_eq:
            z_eq = (A_np @ x_star_np - b_np) / beta
            sigmoid_eq_pos = sigmoid(z_eq)
            sigmoid_eq_neg = sigmoid(-z_eq)
            # Precompute equality sigmoid derivatives and weights
            sigmoid_deriv_eq_pos = sigmoid_eq_pos * (1.0 - sigmoid_eq_pos)
            sigmoid_deriv_eq_neg = sigmoid_eq_neg * (1.0 - sigmoid_eq_neg)
            tp_eq = sigmoid_deriv_eq_pos + sigmoid_deriv_eq_neg  # For gradient computation
            t_eq = sigmoid_eq_pos - sigmoid_eq_neg  # For gradient computation
            weight_eq = (rho_eq / beta) * tp_eq if rho_eq > 1e-12 else None
        else:
            sigmoid_eq_pos = None
            sigmoid_eq_neg = None
            sigmoid_deriv_eq_pos = None
            sigmoid_deriv_eq_neg = None
            tp_eq = None
            t_eq = None
            weight_eq = None

        # Determine solver strategy
        use_implicit_h_solve = False
        if use_sparse:
            should_use_implicit = False
            reason = []
            
            if dim > 50000:
                should_use_implicit = True
                reason.append(f"dim ({dim}) > 50000")
            elif rho_ineq > 1e-12 and nActiveIneq > 0:
                if nActiveIneq > dim * 0.8:
                    should_use_implicit = True
                    reason.append(f"nActiveIneq ({nActiveIneq}) > 0.8*dim ({int(dim*0.8)})")
            elif has_eq and rho_eq > 1e-12 and A_np is not None:
                if nEq > dim * 0.8:
                    should_use_implicit = True
                    reason.append(f"nEq ({nEq}) > 0.8*dim ({int(dim*0.8)})")
            
            if should_use_implicit:
                if verbose:
                    print(f"[PenaltySmoothQP.forward] Will use implicit solve (reasons: {', '.join(reason)})")
                use_implicit_h_solve = True

        # ========== Build and factor Hessian ==========
        H_factor = None
        H_sparse_factored = None
        
        if not use_implicit_h_solve:
            if use_sparse:
                try:
                    H_sparse = Q_np.tocsc()
                    if weight_ineq is not None:
                        H_sparse = H_sparse + G_active.T @ (spa.diags(weight_ineq, format='csr') @ G_active)
                    if weight_eq is not None:
                        if nEq < 100:
                            A_weighted = A_np.multiply(weight_eq[:, np.newaxis])
                            H_eq_add = A_np.T @ A_weighted
                        else:
                            H_eq_add = A_np.T @ (spa.diags(weight_eq, format='csr') @ A_np)
                        estimated_nnz_after = H_sparse.nnz + H_eq_add.nnz
                        if estimated_nnz_after > 5e7 or (dim > 5000 and estimated_nnz_after > dim * dim * 0.1):
                            use_implicit_h_solve = True
                        else:
                            H_sparse = H_sparse + H_eq_add
                    
                    if not use_implicit_h_solve:
                        current_nnz = H_sparse.nnz
                        density = current_nnz / (dim * dim)
                        if density > 0.1 or current_nnz > dim * dim * 0.5:
                            H_dense = H_sparse.toarray()
                            H_sparse = spa.csc_matrix(0.5 * (H_dense + H_dense.T) + 1e-9 * np.eye(dim))
                        else:
                            H_sparse = 0.5 * (H_sparse + H_sparse.T.tocsc()) + 1e-9 * spa.eye(dim, format='csc')
                        H_sparse_factored = H_sparse
                except (MemoryError, Exception) as e:
                    if verbose:
                        print(f"[PenaltySmoothQP.forward] Hessian construction failed: {e}, will use implicit")
                    use_implicit_h_solve = True
            else:
                # Dense case: build and factor Hessian
                H = Q_np.copy()
                if weight_ineq is not None:
                    H += G_active.T @ (weight_ineq[:, None] * G_active)
                if weight_eq is not None:
                    H += A_np.T @ (weight_eq[:, None] * A_np)
                H = 0.5 * (H + H.T) + 1e-9 * np.eye(dim)
                try:
                    H_factor = ('cho', sla.cho_factor(H))
                except np.linalg.LinAlgError:
                    H_factor = ('lu', sla.lu_factor(H))

        # ========== Preprocess sparse gradient structures (for sparse problems) ==========
        Q_coo_indices = None
        G_coo_indices = None
        A_coo_indices = None
        row_to_active_idx = None
        
        if use_sparse:
            # Preconvert to COO format for gradient computation
            if Q.requires_grad or ctx.needs_input_grad[0]:
                Q_coo = Q.detach().cpu()
                if Q_coo.layout != torch.sparse_coo:
                    Q_coo = Q_coo.to_sparse_coo()
                Q_coo = Q_coo.coalesce()
                Q_coo_indices = Q_coo.indices().numpy()
            
            if G.requires_grad or ctx.needs_input_grad[2]:
                G_coo = G.detach().cpu()
                if G_coo.layout != torch.sparse_coo:
                    G_coo = G_coo.to_sparse_coo()
                G_coo = G_coo.coalesce()
                G_coo_indices = G_coo.indices().numpy()
                # Precompute active index mapping
                row_to_active_idx = np.full(nIneq, -1, dtype=int)
                row_to_active_idx[active_mask] = np.arange(nActiveIneq)
            
            if has_eq and (A.requires_grad or ctx.needs_input_grad[4]):
                A_coo = A.detach().cpu()
                if A_coo.layout != torch.sparse_coo:
                    A_coo = A_coo.to_sparse_coo()
                A_coo = A_coo.coalesce()
                A_coo_indices = A_coo.indices().numpy()

        # ========== Save all precomputed data for backward ==========
        ctx.save_for_backward(Q, q, G, h, A, b)
        ctx.needs_grad = ctx.needs_input_grad
        ctx.device = Q.device
        ctx.dtype = Q.dtype
        
        ctx.x_star_np = x_star_np
        ctx.mu_star_np = mu_star_np
        ctx.nu_star_np = nu_star_np
        ctx.beta = beta
        ctx.dim = dim
        ctx.nIneq = nIneq
        ctx.nEq = nEq
        ctx.has_eq = has_eq
        ctx.verbose = verbose
        ctx.use_sparse = use_sparse
        ctx.lin_solver = lin_solver
        # Cached numpy arrays
        ctx.Q_np = Q_np
        ctx.G_np = G_np
        ctx.A_np = A_np
        # Precomputed differentiation data
        ctx.rho_ineq = rho_ineq
        ctx.rho_eq = rho_eq
        ctx.active_mask = active_mask
        ctx.nActiveIneq = nActiveIneq
        ctx.G_active = G_active
        ctx.sigmoid_ineq = sigmoid_ineq
        ctx.use_implicit_h_solve = use_implicit_h_solve
        # Precomputed weights and division results (avoid division in backward)
        ctx.weight_ineq = weight_ineq
        ctx.weight_eq = weight_eq
        ctx.sigmoid_deriv_ineq = sigmoid_deriv_ineq
        ctx.sp_over_beta = sigmoid_deriv_ineq / beta if sigmoid_deriv_ineq is not None else None
        ctx.rho_ineq_over_beta = rho_ineq / beta if rho_ineq > 1e-12 else 0.0
        ctx.rho_eq_over_beta = rho_eq / beta if rho_eq > 1e-12 else 0.0
        ctx.tp_eq = tp_eq
        ctx.tp_eq_over_beta = tp_eq / beta if tp_eq is not None else None
        ctx.t_eq = t_eq
        # Precomputed Hessian factorization
        ctx.H_factor = H_factor
        ctx.H_sparse_factored = H_sparse_factored
        # Precomputed sparse gradient structures (indices already stacked as torch tensor)
        ctx.Q_coo_indices = Q_coo_indices
        ctx.Q_indices_stacked = torch.from_numpy(np.stack([Q_coo_indices[0], Q_coo_indices[1]])) if Q_coo_indices is not None else None
        ctx.G_coo_indices = G_coo_indices
        ctx.G_indices_stacked = torch.from_numpy(np.stack([G_coo_indices[0], G_coo_indices[1]])) if G_coo_indices is not None else None
        ctx.A_coo_indices = A_coo_indices
        ctx.A_indices_stacked = torch.from_numpy(np.stack([A_coo_indices[0], A_coo_indices[1]])) if A_coo_indices is not None else None
        ctx.row_to_active_idx = row_to_active_idx

        x_star = torch.from_numpy(x_star_np).to(dtype=ctx.dtype, device=ctx.device)
        mu_star = torch.from_numpy(mu_star_np).to(dtype=ctx.dtype, device=ctx.device) if has_eq else None
        nu_star = torch.from_numpy(nu_star_np).to(dtype=ctx.dtype, device=ctx.device)

        return x_star, mu_star, nu_star
    
    @staticmethod
    def backward(ctx, grad_x, grad_mu, grad_nu):
        """
        Backward: compute gradients using implicit function theorem on the
        smoothed penalty reformulation.

        Handles gradients flowing through all three outputs (x*, μ*, ν*):
        - grad_x: gradient from loss w.r.t. primal solution x*
        - grad_mu: gradient from loss w.r.t. equality multipliers μ*
        - grad_nu: gradient from loss w.r.t. inequality multipliers ν*

        In the penalty formulation, multipliers are smooth functions:
            ν_i = ρ_ineq * σ((G_i x - h_i) / β)
            μ_j = ρ_eq * (σ_+ - σ_-)
        so their gradients propagate via:
        1) Indirect path: grad_nu/grad_mu → augment grad_x → H⁻¹ solve → param grads
        2) Direct path: grad_nu/grad_mu → ∂ν/∂G, ∂ν/∂h, ∂μ/∂A, ∂μ/∂b at fixed x
        """
        # Batch retrieve properties from ctx to local variables (faster lookup)
        needs_grad = ctx.needs_grad
        device = ctx.device
        dtype = ctx.dtype
        verbose = ctx.verbose
        
        # If no input needs gradient, exit early
        if not any(needs_grad[:6]):
            return (None,) * 15  # 14 + 1 for dual_available

        Q, q, G, h, A, b = ctx.saved_tensors
        x_np = ctx.x_star_np
        beta = ctx.beta
        dim = ctx.dim
        nIneq = ctx.nIneq
        nEq = ctx.nEq
        has_eq = ctx.has_eq
        is_sparse = ctx.use_sparse
        lin_solver = ctx.lin_solver

        # Retrieve precomputed data
        Q_np = ctx.Q_np
        G_np = ctx.G_np
        A_np = ctx.A_np
        active_mask = ctx.active_mask
        nActiveIneq = ctx.nActiveIneq
        G_active = ctx.G_active
        sigmoid_ineq = ctx.sigmoid_ineq
        use_implicit_h_solve = ctx.use_implicit_h_solve
        weight_ineq = ctx.weight_ineq
        weight_eq = ctx.weight_eq
        sigmoid_deriv_ineq = ctx.sigmoid_deriv_ineq
        tp_eq = ctx.tp_eq
        t_eq = ctx.t_eq
        mu_star_np = ctx.mu_star_np
        nu_star_np = ctx.nu_star_np
        H_factor = ctx.H_factor
        H_sparse_factored = ctx.H_sparse_factored
        # Precomputed sparse structures
        Q_coo_indices = ctx.Q_coo_indices
        Q_indices_stacked = ctx.Q_indices_stacked
        G_coo_indices = ctx.G_coo_indices
        G_indices_stacked = ctx.G_indices_stacked
        A_coo_indices = ctx.A_coo_indices
        A_indices_stacked = ctx.A_indices_stacked
        row_to_active_idx = ctx.row_to_active_idx
        # Precomputed division results
        sp_over_beta = ctx.sp_over_beta
        rho_ineq_over_beta = ctx.rho_ineq_over_beta
        rho_eq_over_beta = ctx.rho_eq_over_beta
        tp_eq_over_beta = ctx.tp_eq_over_beta

        # Get grad_x as numpy (avoiding to_numpy function overhead)
        grad_x_np = grad_x.detach().cpu().numpy() if not is_sparse_tensor(grad_x) else torch_sparse_to_scipy(grad_x)

        # ========== Incorporate multiplier gradients (penalty dual sensitivity) ==========
        # In the penalty formulation, multipliers are smooth functions of x and params:
        #   ν_i = ρ_ineq * σ((G_i x - h_i)/β)  =>  ∂ν/∂x = diag(weight_ineq) @ G_active
        #   μ_j = ρ_eq * (σ_+ - σ_-)            =>  ∂μ/∂x = diag(weight_eq) @ A
        # Chain rule: effective grad_x += (∂ν/∂x)^T grad_nu + (∂μ/∂x)^T grad_mu
        grad_nu_np = None
        grad_mu_np = None
        has_grad_nu = False
        has_grad_mu = False
        grad_nu_active = None

        if grad_nu is not None:
            grad_nu_np = grad_nu.detach().cpu().numpy()
            if np.any(grad_nu_np != 0):
                has_grad_nu = True
                if weight_ineq is not None and nActiveIneq > 0:
                    grad_nu_active = grad_nu_np[active_mask]

        if has_eq and grad_mu is not None:
            grad_mu_np = grad_mu.detach().cpu().numpy()
            if np.any(grad_mu_np != 0):
                has_grad_mu = True

        # Augment grad_x with multiplier sensitivity contributions (indirect path)
        if has_grad_nu and grad_nu_active is not None:
            grad_x_np = grad_x_np + G_active.T @ (grad_nu_active * weight_ineq)

        if has_grad_mu and weight_eq is not None:
            grad_x_np = grad_x_np + A_np.T @ (grad_mu_np * weight_eq)

        if verbose and (has_grad_nu or has_grad_mu):
            print(f"[PenaltySmoothQP.backward] Incorporated multiplier gradients: "
                  f"has_grad_nu={has_grad_nu}, has_grad_mu={has_grad_mu}")

        # ========== Solve linear system ==========
        if is_sparse and not use_implicit_h_solve and H_sparse_factored is not None:
            try:
                v = sparse_solve_spd(H_sparse_factored, grad_x_np, solver=lin_solver)
            except Exception:
                use_implicit_h_solve = True
        
        if use_implicit_h_solve:
            def hv_multiply(v_in):
                res = Q_np @ v_in
                if weight_ineq is not None:
                    res += G_active.T @ (weight_ineq * (G_active @ v_in))
                if weight_eq is not None:
                    res += A_np.T @ (weight_eq * (A_np @ v_in))
                return res + 1e-9 * v_in
            H_op = spla.LinearOperator((dim, dim), matvec=hv_multiply)
            try:
                v, info = spla.minres(H_op, grad_x_np, atol=1e-6, maxiter=min(2000, dim * 2))
            except TypeError:
                v, info = spla.minres(H_op, grad_x_np, tol=1e-6, maxiter=min(2000, dim * 2))
        elif not is_sparse and H_factor is not None:
            factor_type, factor_data = H_factor
            v = sla.lu_solve(factor_data, grad_x_np) if factor_type == 'lu' else sla.cho_solve(factor_data, grad_x_np)

        # ========== Compute Gradients based on needs_grad ==========
        # Pre-allocate return list (15 = 6 inputs + 9 non-tensor args: qp_solver, solver_kwargs, beta, penalty_coeff, use_sparse, lin_solver, x_warm, verbose, dual_available)
        grads = [None] * 15
        
        # grad_q
        if needs_grad[1]:
            grads[1] = torch.from_numpy(-v).to(device=device, dtype=dtype)

        # grad_Q
        if needs_grad[0]:
            if is_sparse and Q_indices_stacked is not None:
                r, c = Q_coo_indices[0], Q_coo_indices[1]
                grad_Q_vals = -0.5 * (v[r] * x_np[c] + x_np[r] * v[c])
                grads[0] = torch.sparse_coo_tensor(Q_indices_stacked, torch.from_numpy(grad_Q_vals), Q.size()).to(device=device, dtype=dtype)
            else:
                grads[0] = torch.from_numpy(-0.5 * (np.outer(v, x_np) + np.outer(x_np, v))).to(device=device, dtype=dtype)

        # grad_G and grad_h
        if needs_grad[2] or needs_grad[3]:
            rho_ineq = ctx.rho_ineq
            gv = G_active @ v
            if rho_ineq > 1e-12 and nActiveIneq > 0:
                nu_active = nu_star_np[active_mask]
                if is_sparse and needs_grad[2] and G_indices_stacked is not None:
                    r, c = G_coo_indices[0], G_coo_indices[1]
                    active_rows_mask = active_mask[r]
                    a_idx = row_to_active_idx[r[active_rows_mask]]
                    grad_G_vals = np.zeros(len(r))
                    # Use exact multiplier nu_active for the first term (pseudo-multiplier lambda_ps = lambda_star)
                    grad_G_vals[active_rows_mask] = -(nu_active[a_idx] * v[c[active_rows_mask]] + weight_ineq[a_idx] * gv[a_idx] * x_np[c[active_rows_mask]])
                    grads[2] = torch.sparse_coo_tensor(G_indices_stacked, torch.from_numpy(grad_G_vals), G.size()).to(device=device, dtype=dtype)
                elif needs_grad[2]:
                    grad_G_np = np.zeros((nIneq, dim))
                    # Use exact multiplier nu_active for the first term
                    grad_G_np[active_mask, :] = -(nu_active[:, None] * v[None, :] + weight_ineq[:, None] * gv[:, None] * x_np[None, :])
                    grads[2] = torch.from_numpy(grad_G_np).to(device=device, dtype=dtype)
                
                if needs_grad[3]:
                    grad_h_np = np.zeros(nIneq)
                    grad_h_np[active_mask] = rho_ineq_over_beta * sigmoid_deriv_ineq * gv
                    grads[3] = torch.from_numpy(grad_h_np).to(device=device, dtype=dtype)
            else:
                if needs_grad[2]: grads[2] = torch.zeros_like(G)
                if needs_grad[3]: grads[3] = torch.zeros(nIneq, device=device, dtype=dtype)

        # grad_A and grad_b
        if has_eq and (needs_grad[4] or needs_grad[5]):
            rho_eq = ctx.rho_eq
            av = A_np @ v
            if rho_eq > 1e-12:
                if is_sparse and needs_grad[4] and A_indices_stacked is not None:
                    r, c = A_coo_indices[0], A_coo_indices[1]
                    # Use exact multiplier mu_star_np for the first term
                    grad_A_vals = -(mu_star_np[r] * v[c] + weight_eq[r] * av[r] * x_np[c])
                    grads[4] = torch.sparse_coo_tensor(A_indices_stacked, torch.from_numpy(grad_A_vals), A.size()).to(device=device, dtype=dtype)
                elif needs_grad[4]:
                    # Use exact multiplier mu_star_np for the first term
                    grads[4] = torch.from_numpy(-(mu_star_np[:, None] * v[None, :] + weight_eq[:, None] * av[:, None] * x_np[None, :])).to(device=device, dtype=dtype)
                
                if needs_grad[5]:
                    grads[5] = torch.from_numpy(rho_eq_over_beta * tp_eq * av).to(device=device, dtype=dtype)
            else:
                if needs_grad[4]: grads[4] = torch.zeros_like(A)
                if needs_grad[5]: grads[5] = torch.zeros(nEq, device=device, dtype=dtype)

        # ========== Direct gradient contributions from multiplier gradients ==========
        # These are ∂L/∂ν · (∂ν/∂θ)|_{x fixed} and ∂L/∂μ · (∂μ/∂θ)|_{x fixed}
        # (the indirect contribution through x is already handled via augmented grad_x above)

        if has_grad_nu and weight_ineq is not None and nActiveIneq > 0 and grad_nu_active is not None:
            coeff_nu = grad_nu_active * weight_ineq  # per-active-constraint coefficient

            if needs_grad[2]:
                # ∂ν_i/∂G_{ij}|_x = weight_ineq_i * x_j  (for active constraint i)
                if is_sparse and G_indices_stacked is not None:
                    r, c = G_coo_indices[0], G_coo_indices[1]
                    active_rows_mask_r = active_mask[r]
                    a_idx = row_to_active_idx[r[active_rows_mask_r]]
                    direct_vals = np.zeros(len(r))
                    direct_vals[active_rows_mask_r] = coeff_nu[a_idx] * x_np[c[active_rows_mask_r]]
                    direct_tensor = torch.sparse_coo_tensor(
                        G_indices_stacked, torch.from_numpy(direct_vals), G.size()
                    ).to(device=device, dtype=dtype)
                else:
                    direct_np = np.zeros((nIneq, dim))
                    direct_np[active_mask, :] = coeff_nu[:, None] * x_np[None, :]
                    direct_tensor = torch.from_numpy(direct_np).to(device=device, dtype=dtype)
                grads[2] = (grads[2] + direct_tensor) if grads[2] is not None else direct_tensor

            if needs_grad[3]:
                # ∂ν_i/∂h_i|_x = -weight_ineq_i  (for active constraint i)
                direct_h = np.zeros(nIneq)
                direct_h[active_mask] = -coeff_nu
                direct_tensor = torch.from_numpy(direct_h).to(device=device, dtype=dtype)
                grads[3] = (grads[3] + direct_tensor) if grads[3] is not None else direct_tensor

        if has_grad_mu and weight_eq is not None and grad_mu_np is not None:
            coeff_mu = grad_mu_np * weight_eq  # per-equality-constraint coefficient

            if needs_grad[4]:
                # ∂μ_j/∂A_{jk}|_x = weight_eq_j * x_k
                if is_sparse and A_indices_stacked is not None:
                    r, c = A_coo_indices[0], A_coo_indices[1]
                    direct_vals = coeff_mu[r] * x_np[c]
                    direct_tensor = torch.sparse_coo_tensor(
                        A_indices_stacked, torch.from_numpy(direct_vals), A.size()
                    ).to(device=device, dtype=dtype)
                else:
                    direct_tensor = torch.from_numpy(
                        coeff_mu[:, None] * x_np[None, :]
                    ).to(device=device, dtype=dtype)
                grads[4] = (grads[4] + direct_tensor) if grads[4] is not None else direct_tensor

            if needs_grad[5]:
                # ∂μ_j/∂b_j|_x = -weight_eq_j
                direct_tensor = torch.from_numpy(-coeff_mu).to(device=device, dtype=dtype)
                grads[5] = (grads[5] + direct_tensor) if grads[5] is not None else direct_tensor

        return tuple(grads)
