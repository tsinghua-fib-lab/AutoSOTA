"""
    Sparse GP-Collocation Solver

Sparse Cholesky approximation + Collocation (point-based) PDE discretization.
This isolates the effect of FVM vs collocation - same sparsity, different discretization.
"""

using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using Kronecker
using SparseConnectivityTracer, SparseMatrixColorings

import GaussianMarkovRandomFields: mean, std, precision_matrix

# Include common problem/metrics definitions
include(joinpath(@__DIR__, "..", "problem.jl"))
include(joinpath(@__DIR__, "..", "metrics.jl"))

"""
    solve_sparse_collocation(instance::ProblemInstance, N::Int; kwargs...)

Solve Burgers equation using sparse GP with collocation (strong form PDE constraints).

Unlike FVM which enforces conservation over cells via integrals, collocation
enforces the PDE pointwise: ∂u/∂t + u∂u/∂x - ν∂²u/∂x² = 0 at grid points.

# Arguments
- `instance`: Problem instance with IC and reference solution
- `N`: Number of spatial grid points

# Keyword arguments
- `n_timesteps`: Number of time steps (required for fair comparison)
- `ρ=3.0`: Sparse Cholesky threshold
- `lengthscale=nothing`: Absolute kernel lengthscale (overrides lengthscale_factor if set)
- `lengthscale_factor=3.0`: Kernel lengthscale as multiple of cell size (lengthscale = factor * Δx)
- `smoothness=2`: Matérn smoothness (2 = Matérn 5/2)
- `time_scheme=:crank_nicolson`: Time integration scheme

# Returns
`SolutionResult` with posterior mean/std and computational metrics.
"""
function solve_sparse_collocation(instance::ProblemInstance, N::Int;
                                   n_timesteps::Int,
                                   ρ::Float64=3.0,
                                   lengthscale::Union{Nothing,Float64}=nothing,
                                   lengthscale_factor::Float64=3.0,
                                   smoothness::Int=2,
                                   time_scheme::Symbol=:crank_nicolson,
                                   initialization::Symbol=:ekf)
    problem = instance.problem
    ic = instance.ic

    # Track timing
    t_start = time()

    # Memory tracking
    GC.gc()
    mem_before = Base.gc_live_bytes()

    # -------------------------------------------------------------------------
    # Spatial setup
    # -------------------------------------------------------------------------
    (; x_min, x_max, T_end, ν) = problem
    L = x_max - x_min

    endpoints = range(x_min, x_max, length=N)
    Δx = endpoints[2] - endpoints[1]

    # For collocation, we also need second derivatives
    # Interior points where we'll enforce the PDE
    interior_idx = 2:(N-1)
    N_interior = length(interior_idx)

    # Time discretization
    n_t = n_timesteps
    Δt = T_end / n_t

    # -------------------------------------------------------------------------
    # Build sparse spatial precision
    # For collocation we need: f, f_x, f_xx (second derivative for viscous term)
    # -------------------------------------------------------------------------
    # Use absolute lengthscale if provided, otherwise scale with cell size
    ls = isnothing(lengthscale) ? lengthscale_factor * Δx : lengthscale
    k = HalfIntegerMaternKernel(smoothness, [ls])

    approx = sparse_precision([
        :f => EvaluationFunctional(endpoints),
        :f_dx => EvaluationFunctional(endpoints) ∘ PartialDerivative((1,)),
        :f_dxx => EvaluationFunctional(endpoints) ∘ PartialDerivative((2,))  # Second derivative
    ], k; ρ=ρ, ordering=:integrals_coarsest)  # Still use this ordering even though no integrals

    Q_space = approx.Q
    state_layout = approx.layout
    N_space = size(Q_space, 1)

    # Track Cholesky nnz
    cholesky_nnz = approx.info.nnz

    # -------------------------------------------------------------------------
    # Build initial state GMRF
    # -------------------------------------------------------------------------
    Q_state0 = [Q_space spzeros(size(Q_space)...);
                spzeros(size(Q_space)...) Q_space]
    N_state = size(Q_state0, 1)
    x0 = GMRF(zeros(N_state), Q_state0)

    full_state_layout = layout((
        f = N,
        f_dx = N,
        f_dxx = N,
        df_dt = N,
        df_dx_dt = N,
        df_dxx_dt = N
    ))

    # -------------------------------------------------------------------------
    # Apply initial conditions
    # -------------------------------------------------------------------------
    ys = [ic(x) for x in endpoints]
    ys_dx = [evaluate_dx(ic, x) for x in endpoints]
    # For second derivative, use finite difference approximation from IC
    ys_dxx = zeros(N)
    for i in 2:(N-1)
        ys_dxx[i] = (ys[i+1] - 2*ys[i] + ys[i-1]) / Δx^2
    end
    # Boundary second derivatives (extrapolate)
    ys_dxx[1] = ys_dxx[2]
    ys_dxx[N] = ys_dxx[N-1]

    x0_ic = prescribe_indices(x0, indices(full_state_layout, :f), ys)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :f_dx), ys_dx)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :f_dxx), ys_dxx)

    # -------------------------------------------------------------------------
    # Collocation constraint at t=0 (pointwise PDE at interior points)
    # PDE: ∂u/∂t + u*∂u/∂x = ν*∂²u/∂x²
    # -------------------------------------------------------------------------
    function f_colloc_t0(x)
        x_s = State(x, full_state_layout)
        # At interior points: du/dt + u*du/dx - ν*d²u/dx² = 0
        du_dt = x_s.df_dt[interior_idx]
        u = x_s.f[interior_idx]
        du_dx = x_s.f_dx[interior_idx]
        d2u_dx2 = x_s.f_dxx[interior_idx]
        return du_dt + u .* du_dx - ν * d2u_dx2
    end

    colloc_model_0 = NonlinearLeastSquaresModel(f_colloc_t0, length(x0_ic))
    y_colloc_0 = zeros(N_interior)
    lik_colloc_0 = colloc_model_0(y_colloc_0; σ=0.0001)
    x_colloc_0 = gaussian_approximation(x0_ic, lik_colloc_0)

    # -------------------------------------------------------------------------
    # Time stepping setup
    # -------------------------------------------------------------------------
    sde = IWPSDE(1.0)
    A_t, Σ_noise_t = discretize_vanloan(sde, Δt)
    Q_noise_t = inv(Σ_noise_t)
    Q_noise_t = 0.5 * (Q_noise_t + Q_noise_t')

    A = kronecker(A_t, sparse(I, N_space, N_space))
    Q_noise = kronecker(Q_noise_t, Q_space)

    # -------------------------------------------------------------------------
    # Build joint spacetime GMRF
    # -------------------------------------------------------------------------
    ssm = ConstantLinearGaussianSSM(x_colloc_0, A, Q_noise)
    x_joint = GPFiniteVolume.joint_gmrf(ssm, n_t)

    timest = TimeStack(mean(x_joint), full_state_layout)

    # -------------------------------------------------------------------------
    # Boundary conditions
    # -------------------------------------------------------------------------
    bc_left = problem.u_left
    bc_right = problem.u_right
    ys_bc = repeat([bc_left, bc_right], n_t - 1)
    bc_indices = absindices(timest, :f, [1, N], 2:n_t)
    x_joint_bc = prescribe_indices(x_joint, bc_indices, ys_bc)

    # -------------------------------------------------------------------------
    # Causal sweep initialization (block Gauss-Seidel style)
    # -------------------------------------------------------------------------
    function causal_sweep(x_t0_mean)
        Q_noise_sparse = sparse(collect(Q_noise))
        means = Vector{Vector{Float64}}(undef, n_t)
        means[1] = x_t0_mean

        for t in 2:n_t
            μ_prior = collect(A * means[t-1])
            x_prior = GMRF(μ_prior, Q_noise_sparse)
            # Hard BC constraint via ConstrainedGMRF
            A_bc = zeros(2, length(μ_prior))
            A_bc[1, 1] = 1.0   # f[1] = bc_left
            A_bc[2, N] = 1.0   # f[N] = bc_right
            x_bc = ConstrainedGMRF(x_prior, A_bc, [bc_left, bc_right])

            function f_colloc_block(x)
                x_c = State(x, full_state_layout)
                x_p = State(means[t-1], full_state_layout)
                # Current time terms
                du_dt = x_c.df_dt[interior_idx]
                conv_c = x_c.f[interior_idx] .* x_c.f_dx[interior_idx]
                diff_c = ν * x_c.f_dxx[interior_idx]
                # Previous time terms
                conv_p = x_p.f[interior_idx] .* x_p.f_dx[interior_idx]
                diff_p = ν * x_p.f_dxx[interior_idx]
                # Crank-Nicolson
                return du_dt + 0.5 * ((conv_c - diff_c) + (conv_p - diff_p))
            end

            model = NonlinearLeastSquaresModel(f_colloc_block, length(x_bc))
            lik = model(zeros(N_interior); σ=0.0001)
            x_sol = gaussian_approximation(x_bc, lik; verbose=false)
            means[t] = mean(x_sol)
        end
        return vcat(means...)
    end

    # -------------------------------------------------------------------------
    # Collocation constraints at all time steps (interior points only)
    # -------------------------------------------------------------------------
    # Current time indices
    f_curr = absindices(timest, :f, interior_idx, 2:n_t)
    f_dx_curr = absindices(timest, :f_dx, interior_idx, 2:n_t)
    f_dxx_curr = absindices(timest, :f_dxx, interior_idx, 2:n_t)
    df_dt_curr = absindices(timest, :df_dt, interior_idx, 2:n_t)

    # Previous time indices for Crank-Nicolson
    f_prev = absindices(timest, :f, interior_idx, 1:(n_t-1))
    f_dx_prev = absindices(timest, :f_dx, interior_idx, 1:(n_t-1))
    f_dxx_prev = absindices(timest, :f_dxx, interior_idx, 1:(n_t-1))

    function f_colloc_crank_nicolson(x)
        du_dt = x[df_dt_curr]

        # Nonlinear terms at current time
        u_curr = x[f_curr]
        ux_curr = x[f_dx_curr]
        uxx_curr = x[f_dxx_curr]
        conv_curr = u_curr .* ux_curr
        diff_curr = ν * uxx_curr

        # Nonlinear terms at previous time
        u_prev = x[f_prev]
        ux_prev = x[f_dx_prev]
        uxx_prev = x[f_dxx_prev]
        conv_prev = u_prev .* ux_prev
        diff_prev = ν * uxx_prev

        # CN: average of RHS at curr and prev
        rhs_avg = 0.5 * ((conv_curr - diff_curr) + (conv_prev - diff_prev))

        return du_dt + rhs_avg
    end

    function f_colloc_euler(x)
        du_dt = x[df_dt_curr]
        u = x[f_curr]
        ux = x[f_dx_curr]
        uxx = x[f_dxx_curr]
        return du_dt + u .* ux - ν * uxx
    end

    f_colloc_fn = time_scheme == :crank_nicolson ? f_colloc_crank_nicolson : f_colloc_euler

    colloc_model = NonlinearLeastSquaresModel(f_colloc_fn, length(x_joint_bc))
    y_colloc = zeros(length(f_curr))
    lik_colloc = colloc_model(y_colloc; σ=0.0001)

    A_constr = zeros(1, length(x_joint_bc))
    A_constr[1] = 1.0
    e = [mean(x_joint_bc)[1]]
    x_constr = ConstrainedGMRF(x_joint_bc, A_constr, e)

    # Apply causal sweep initialization if requested
    if initialization == :ekf
        init_vec = causal_sweep(mean(x_colloc_0))
        Q_joint = precision_matrix(x_constr)
        x_init = GMRF(init_vec, Q_joint)
        x_constr = ConstrainedGMRF(x_init, A_constr, e)
    end

    x_solution = gaussian_approximation(x_constr, lik_colloc; verbose=false)

    # -------------------------------------------------------------------------
    # Extract solution statistics
    # -------------------------------------------------------------------------
    GC.gc()
    mem_after = Base.gc_live_bytes()
    peak_memory_mb = max(0.0, (mem_after - mem_before) / 1e6)

    wall_time_s = time() - t_start

    means_stack = TimeStack(mean(x_solution), full_state_layout)
    stds_stack = TimeStack(std(x_solution), full_state_layout)

    xs = collect(Float64, endpoints)
    ts = [i * Δt for i in 0:(n_t-1)]

    mean_matrix = hcat([means_stack[:f, t] for t in 1:n_t]...)
    std_matrix = hcat([stds_stack[:f, t] for t in 1:n_t]...)

    return SolutionResult(
        xs, ts,
        mean_matrix, std_matrix,
        wall_time_s, peak_memory_mb,
        cholesky_nnz, N_state * n_t,
        "sparse_collocation", N
    )
end
