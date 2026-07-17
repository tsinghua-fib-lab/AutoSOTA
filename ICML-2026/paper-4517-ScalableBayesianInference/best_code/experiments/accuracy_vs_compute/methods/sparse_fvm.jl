"""
    Sparse GP-FVM Solver

Sparse Cholesky approximation + Finite Volume Method discretization.
This is the main method we're advocating for in the paper.
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
    solve_sparse_fvm(instance::ProblemInstance, N::Int; kwargs...)

Solve Burgers equation using sparse GP-FVM.

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
function solve_sparse_fvm(instance::ProblemInstance, N::Int;
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

    # Memory tracking (approximate via GC)
    GC.gc()
    mem_before = Base.gc_live_bytes()

    # -------------------------------------------------------------------------
    # Spatial setup
    # -------------------------------------------------------------------------
    (; x_min, x_max, T_end, ν) = problem
    L = x_max - x_min

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
    # Use absolute lengthscale if provided, otherwise scale with cell size
    ls = isnothing(lengthscale) ? lengthscale_factor * Δx : lengthscale
    k = HalfIntegerMaternKernel(smoothness, [ls])
    approx = sparse_precision([
        :f => EvaluationFunctional(endpoints),
        :f_dx => EvaluationFunctional(endpoints) ∘ PartialDerivative((1,)),
        :f_int => VectorizedLebesgueIntegral(intervals)
    ], k; ρ=ρ, ordering=:integrals_coarsest)

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
        u_left = x_s.f[1:end-1]
        u_right = x_s.f[2:end]
        ux_left = x_s.f_dx[1:end-1]
        ux_right = x_s.f_dx[2:end]
        F_left = 0.5 * u_left.^2 - ν * ux_left
        F_right = 0.5 * u_right.^2 - ν * ux_right
        return dint_dt + (F_right - F_left)
    end

    fvm_model_0 = NonlinearLeastSquaresModel(f_fvm_t0, length(x0_ic))
    y_fvm_0 = zeros(N_int)
    lik_fvm_0 = fvm_model_0(y_fvm_0; σ=0.0001)
    x_fvm_0 = gaussian_approximation(x0_ic, lik_fvm_0)

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
    ssm = ConstantLinearGaussianSSM(x_fvm_0, A, Q_noise)
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
    function ekf_sweep(x_t0_mean)
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

            function f_fvm_block(x)
                x_c = State(x, full_state_layout)
                x_p = State(means[t-1], full_state_layout)
                F_c = 0.5 * x_c.f.^2 - ν * x_c.f_dx
                F_p = 0.5 * x_p.f.^2 - ν * x_p.f_dx
                dF_c = F_c[2:end] - F_c[1:end-1]
                dF_p = F_p[2:end] - F_p[1:end-1]
                return x_c.df_int_dt + 0.5 * (dF_c + dF_p)
            end

            model = NonlinearLeastSquaresModel(f_fvm_block, length(x_bc))
            lik = model(zeros(N_int); σ=0.0001)
            x_sol = gaussian_approximation(x_bc, lik; verbose=false)
            means[t] = mean(x_sol)
        end
        return vcat(means...)
    end

    # -------------------------------------------------------------------------
    # FVM constraints at all time steps
    # -------------------------------------------------------------------------
    f_left_curr = absindices(timest, :f, 1:(N-1), 2:n_t)
    f_right_curr = absindices(timest, :f, 2:N, 2:n_t)
    f_dx_left_curr = absindices(timest, :f_dx, 1:(N-1), 2:n_t)
    f_dx_right_curr = absindices(timest, :f_dx, 2:N, 2:n_t)
    df_int_dt_curr = absindices(timest, :df_int_dt, 1:(N-1), 2:n_t)

    f_left_prev = absindices(timest, :f, 1:(N-1), 1:(n_t-1))
    f_right_prev = absindices(timest, :f, 2:N, 1:(n_t-1))
    f_dx_left_prev = absindices(timest, :f_dx, 1:(N-1), 1:(n_t-1))
    f_dx_right_prev = absindices(timest, :f_dx, 2:N, 1:(n_t-1))

    function f_fvm_crank_nicolson(x)
        dint_dt = x[df_int_dt_curr]
        F_L_curr = 0.5 * x[f_left_curr].^2 - ν * x[f_dx_left_curr]
        F_R_curr = 0.5 * x[f_right_curr].^2 - ν * x[f_dx_right_curr]
        F_L_prev = 0.5 * x[f_left_prev].^2 - ν * x[f_dx_left_prev]
        F_R_prev = 0.5 * x[f_right_prev].^2 - ν * x[f_dx_right_prev]
        net_flux_avg = 0.5 * ((F_R_curr - F_L_curr) + (F_R_prev - F_L_prev))
        return dint_dt + net_flux_avg
    end

    function f_fvm_euler(x)
        dint_dt = x[df_int_dt_curr]
        F_left = 0.5 * x[f_left_curr].^2 - ν * x[f_dx_left_curr]
        F_right = 0.5 * x[f_right_curr].^2 - ν * x[f_dx_right_curr]
        return dint_dt + (F_right - F_left)
    end

    f_fvm_fn = time_scheme == :crank_nicolson ? f_fvm_crank_nicolson : f_fvm_euler

    fvm_model = NonlinearLeastSquaresModel(f_fvm_fn, length(x_joint_bc))
    y_fvm = zeros(length(f_left_curr))
    lik_fvm = fvm_model(y_fvm; σ=0.0001)

    A_constr = zeros(1, length(x_joint_bc))
    A_constr[1] = 1.0
    e = [mean(x_joint_bc)[1]]
    x_constr = ConstrainedGMRF(x_joint_bc, A_constr, e)

    # Apply EKF-style initialization if requested
    if initialization == :ekf
        init_vec = ekf_sweep(mean(x_fvm_0))
        Q_joint = precision_matrix(x_constr)
        x_init = GMRF(init_vec, Q_joint)
        x_constr = ConstrainedGMRF(x_init, A_constr, e)
    end

    x_solution = gaussian_approximation(x_constr, lik_fvm; verbose=false)

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
        "sparse_fvm", N
    )
end
