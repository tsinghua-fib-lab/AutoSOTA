"""
    EKF-style Marginal Moment-Matching GP-FVM Solver

Sequential GP-FVM using marginal moment-matching (EKF-style) updates.
Uses the solve_ekf_sequential utility from GPFiniteVolume.

Memory is O(N²) per step vs O(T×N²) for the full spacetime batch approach.
"""

using GPFiniteVolume
using GPFiniteVolume: solve_ekf_sequential, MomentMatchedState, EKFSolution
using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using Kronecker
using SparseConnectivityTracer, SparseMatrixColorings

import GaussianMarkovRandomFields: mean, std, var, precision_matrix

# Include common problem/metrics definitions
include(joinpath(@__DIR__, "..", "problem.jl"))
include(joinpath(@__DIR__, "..", "metrics.jl"))

"""
    solve_ekf_fvm(instance::ProblemInstance, N::Int; kwargs...)

Solve Burgers equation using EKF-style sequential GP-FVM.

This method processes timesteps sequentially using a sliding 2-block window,
maintaining a moment-matched Gaussian approximation at each step. This preserves
sparsity and reduces memory usage compared to the batch joint spacetime approach.

# Arguments
- `instance`: Problem instance with IC and reference solution
- `N`: Number of spatial grid points

# Keyword arguments
- `n_timesteps`: Number of time steps (required for fair comparison)
- `ρ=3.0`: Sparse Cholesky threshold
- `lengthscale=nothing`: Absolute kernel lengthscale (overrides lengthscale_factor if set)
- `lengthscale_factor=3.0`: Kernel lengthscale as multiple of cell size
- `smoothness=2`: Matérn smoothness (2 = Matérn 5/2)
- `max_gn_iter=1`: Max Gauss-Newton iterations per step (1 = true EKF)

# Returns
`SolutionResult` with posterior mean/std and computational metrics.
"""
function solve_ekf_fvm(instance::ProblemInstance, N::Int;
                        n_timesteps::Int,
                        ρ::Float64=3.0,
                        lengthscale::Union{Nothing,Float64}=nothing,
                        lengthscale_factor::Float64=3.0,
                        smoothness::Int=2,
                        max_gn_iter::Int=1)
    problem = instance.problem
    ic = instance.ic

    # Track timing
    t_start = time()

    # Memory tracking (approximate via GC)
    GC.gc()
    mem_before = Base.gc_live_bytes()

    # -------------------------------------------------------------------------
    # Spatial setup (same as sparse_fvm)
    # -------------------------------------------------------------------------
    (; x_min, x_max, T_end, ν, u_left, u_right) = problem

    endpoints = range(x_min, x_max, length=N)
    intervals = intervals_from_endpoints(collect(endpoints))
    N_int = length(intervals)
    Δx = endpoints[2] - endpoints[1]

    # Time discretization
    n_t = n_timesteps
    Δt = T_end / n_t

    # -------------------------------------------------------------------------
    # Build sparse spatial precision
    # -------------------------------------------------------------------------
    ls = isnothing(lengthscale) ? lengthscale_factor * Δx : lengthscale
    k = HalfIntegerMaternKernel(smoothness, [ls])
    approx = sparse_precision([
        :f => EvaluationFunctional(endpoints),
        :f_dx => EvaluationFunctional(endpoints) ∘ PartialDerivative((1,)),
        :f_int => VectorizedLebesgueIntegral(intervals)
    ], k; ρ=ρ, ordering=:integrals_coarsest)

    Q_space = approx.Q
    N_space = size(Q_space, 1)

    # Track Cholesky nnz
    cholesky_nnz = approx.info.nnz

    # -------------------------------------------------------------------------
    # Build initial state GMRF (position + velocity)
    # -------------------------------------------------------------------------
    Q_state0 = [Q_space spzeros(size(Q_space)...);
                spzeros(size(Q_space)...) Q_space]
    N_state = size(Q_state0, 1)
    x0 = GMRF(zeros(N_state), Q_state0)

    full_state_layout = layout((
        f = N,
        f_dx = N,
        f_int = N_int,
        df_dt = N,
        df_dx_dt = N,
        df_int_dt = N_int
    ))

    # -------------------------------------------------------------------------
    # Apply initial conditions
    # -------------------------------------------------------------------------
    ys = [ic(x) for x in endpoints]
    ys_dx = [evaluate_dx(ic, x) for x in endpoints]
    ys_int = [evaluate_int(ic, endpoints[i], endpoints[i+1]) for i in 1:N_int]

    x0_ic = prescribe_indices(x0, indices(full_state_layout, :f), ys)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :f_dx), ys_dx)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :f_int), ys_int)

    # -------------------------------------------------------------------------
    # FVM constraint at t=0
    # -------------------------------------------------------------------------
    function f_fvm_t0(x)
        x_s = State(x, full_state_layout)
        dint_dt = x_s.df_int_dt
        u_left_arr = x_s.f[1:end-1]
        u_right_arr = x_s.f[2:end]
        ux_left = x_s.f_dx[1:end-1]
        ux_right = x_s.f_dx[2:end]
        F_left = 0.5 * u_left_arr.^2 - ν * ux_left
        F_right = 0.5 * u_right_arr.^2 - ν * ux_right
        return dint_dt + (F_right - F_left)
    end

    fvm_model_0 = NonlinearLeastSquaresModel(f_fvm_t0, length(x0_ic))
    lik_fvm_0 = fvm_model_0(zeros(N_int); σ=0.0001)
    x_fvm_0 = gaussian_approximation(x0_ic, lik_fvm_0)

    # -------------------------------------------------------------------------
    # Time stepping matrices
    # -------------------------------------------------------------------------
    sde = IWPSDE(1.0)
    A_t, Σ_noise_t = discretize_vanloan(sde, Δt)
    Q_noise_t = inv(Σ_noise_t)
    Q_noise_t = 0.5 * (Q_noise_t + Q_noise_t')

    A = sparse(collect(kronecker(A_t, sparse(I, N_space, N_space))))
    Q_noise = sparse(collect(kronecker(Q_noise_t, Q_space)))

    # -------------------------------------------------------------------------
    # Build FVM constraint function for EKF solver
    # Uses Crank-Nicolson: average of fluxes at t-1 and t
    # -------------------------------------------------------------------------
    # Pre-build the 2-block layout for indexing
    two_block_layout = layout((
        f_prev = N, f_dx_prev = N, f_int_prev = N_int,
        df_dt_prev = N, df_dx_dt_prev = N, df_int_dt_prev = N_int,
        f = N, f_dx = N, f_int = N_int,
        df_dt = N, df_dx_dt = N, df_int_dt = N_int
    ))

    # Pre-trace the FVM model once (expensive due to sparsity detection)
    function f_fvm_2block(x)
        f_p = x[indices(two_block_layout, :f_prev)]
        f_dx_p = x[indices(two_block_layout, :f_dx_prev)]
        f_c = x[indices(two_block_layout, :f)]
        f_dx_c = x[indices(two_block_layout, :f_dx)]
        df_int_dt = x[indices(two_block_layout, :df_int_dt)]

        # Crank-Nicolson fluxes
        F_left_p = 0.5 * f_p[1:end-1].^2 - ν * f_dx_p[1:end-1]
        F_right_p = 0.5 * f_p[2:end].^2 - ν * f_dx_p[2:end]
        F_left_c = 0.5 * f_c[1:end-1].^2 - ν * f_dx_c[1:end-1]
        F_right_c = 0.5 * f_c[2:end].^2 - ν * f_dx_c[2:end]

        dF_p = F_right_p - F_left_p
        dF_c = F_right_c - F_left_c

        return df_int_dt + 0.5 * (dF_c + dF_p)
    end

    fvm_model_2block = NonlinearLeastSquaresModel(f_fvm_2block, 2 * N_state)

    # Constraint builder for EKF solver
    function build_fvm_constraint(μ_prev, t)
        return fvm_model_2block(zeros(N_int); σ=0.0001)
    end

    # -------------------------------------------------------------------------
    # Boundary condition function (using prescribe_indices instead of ConstrainedGMRF)
    # -------------------------------------------------------------------------
    bc_idx_f1 = N_state + indices(full_state_layout, :f)[1]
    bc_idx_fN = N_state + indices(full_state_layout, :f)[N]

    function apply_bc!(x_2block, t)
        return prescribe_indices(x_2block, [bc_idx_f1, bc_idx_fN], [u_left, u_right])
    end

    # -------------------------------------------------------------------------
    # Run EKF sequential solver
    # -------------------------------------------------------------------------
    ekf_solution = solve_ekf_sequential(
        Q_state0,           # Q_spatial (full state prior)
        full_state_layout,  # state_layout
        A,                  # transition matrix
        Q_noise,            # temporal noise precision
        n_t,                # n_timesteps
        x_fvm_0,            # x0 (initial state after IC+FVM conditioning)
        build_fvm_constraint;
        apply_bc! = apply_bc!,
        store_history = true,
        verbose = false,
        max_iter = max_gn_iter
    )

    # -------------------------------------------------------------------------
    # Extract solution statistics
    # -------------------------------------------------------------------------
    GC.gc()
    mem_after = Base.gc_live_bytes()
    peak_memory_mb = max(0.0, (mem_after - mem_before) / 1e6)

    wall_time_s = time() - t_start

    # Convert EKFSolution to output format
    xs = collect(Float64, endpoints)
    ts = [i * Δt for i in 0:(n_t-1)]

    # Extract f component from each timestep
    f_indices = indices(full_state_layout, :f)
    mean_matrix = hcat([ekf_solution.history[t].μ[f_indices] for t in 1:n_t]...)
    std_matrix = hcat([sqrt.(max.(ekf_solution.history[t].σ², 1e-12))[f_indices] for t in 1:n_t]...)

    return SolutionResult(
        xs, ts,
        mean_matrix, std_matrix,
        wall_time_s, peak_memory_mb,
        cholesky_nnz, N_state,
        "ekf_fvm", N
    )
end
