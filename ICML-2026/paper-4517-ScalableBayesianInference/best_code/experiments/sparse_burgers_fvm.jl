"""
Sparse GP-FVM for Viscid Burgers Equation

Demonstrates sparse Cholesky approximation applied to spacetime GP-FVM.
Solves the viscid Burgers equation: ∂u/∂t + ∂(u²/2)/∂x = ν ∂²u/∂x²

Key features:
- Sparse spatial precision via KL-optimal Cholesky approximation
- Separable spacetime structure: IWP time prior ⊗ sparse spatial prior
- Joint inference over function values, derivatives, and cell integrals
"""

using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using Distributions
using QuadGK
using Kronecker
using CairoMakie
using SparseConnectivityTracer, SparseMatrixColorings
using Zygote

import GaussianMarkovRandomFields: mean, std

"""
    print_sparsity_stats(Q_sparse, name="Precision matrix")

Print sparsity statistics for a sparse matrix.
"""
function print_sparsity_stats(Q_sparse, name="Precision matrix")
    n = size(Q_sparse, 1)
    nnz_Q = nnz(Q_sparse)
    dense_nnz = n * (n + 1) ÷ 2
    fill_pct = 100 * nnz_Q / (2 * dense_nnz)  # Account for symmetric storage
    println("$name sparsity:")
    println("  Size: $(size(Q_sparse))")
    println("  nnz: $nnz_Q ($(round(fill_pct, digits=1))% of symmetric entries)")
end

# ------------------------------------------------------------------------------
# Initial condition functions
# ------------------------------------------------------------------------------

function f_ic(x)
    return exp(-x^2 / 2)
end

function f_ic_dx(x)
    return -x * exp(-x^2 / 2)
end

function f_ic_int(a, b)
    return quadgk(f_ic, a, b)[1]
end

# ------------------------------------------------------------------------------
# Main solver
# ------------------------------------------------------------------------------

"""
    solve_sparse_burgers(; kwargs...)

Solve viscid Burgers equation using sparse GP-FVM.

# Keyword arguments
- `N_x=51`: Number of spatial grid points
- `x_min=-3.0, x_max=3.0`: Spatial domain
- `Δt=0.05`: Time step
- `T_end=2.0`: End time
- `n_timesteps=30`: Number of time steps (overrides T_end if specified)
- `ν=0.01`: Viscosity coefficient
- `lengthscale=0.2`: Kernel lengthscale
- `smoothness=2`: Matérn smoothness (2 = Matérn 5/2)
- `ρ=2.0`: Sparse Cholesky threshold
- `time_scheme=:euler`: Time stepping scheme (:euler, :crank_nicolson, or :crank_nicolson_symmetric)
- `verbose=true`: Print progress information

# Returns
Named tuple with solution and metadata.
"""
function solve_sparse_burgers(;
        N_x = 121,
        x_min = -3.0,
        x_max = 3.0,
        Δt = 0.05,
        T_end = 2.0,
        n_timesteps = nothing,
        ν = 0.01,
        lengthscale = 0.2,
        smoothness = 2,
        ρ = 2.0,
        time_scheme = :euler,
        verbose = true
    )

    verbose && println("=" ^ 60)
    verbose && println("Sparse GP-FVM for Viscid Burgers Equation")
    verbose && println("=" ^ 60)

    # -------------------------------------------------------------------------
    # Spatial domain setup
    # -------------------------------------------------------------------------
    Δx = (x_max - x_min) / (N_x - 1)
    endpoints = range(x_min, x_max, length=N_x)
    intervals = intervals_from_endpoints(collect(endpoints))
    N_int = length(intervals)

    verbose && println("\nSpatial setup:")
    verbose && println("  Domain: [$x_min, $x_max]")
    verbose && println("  Grid points: $N_x (Δx = $(round(Δx, digits=4)))")
    verbose && println("  FVM cells: $N_int")

    # -------------------------------------------------------------------------
    # Build sparse spatial precision using package utilities
    # -------------------------------------------------------------------------
    verbose && println("\nBuilding sparse spatial precision (ρ = $ρ)...")

    k = HalfIntegerMaternKernel(smoothness, [lengthscale])
    approx = sparse_precision([
        :f => EvaluationFunctional(endpoints),
        :f_dx => EvaluationFunctional(endpoints) ∘ PartialDerivative((1,)),
        :f_int => VectorizedLebesgueIntegral(intervals)
    ], k; ρ=ρ, ordering=:integrals_coarsest)

    Q_space = approx.Q
    state_layout = approx.layout
    N_space = size(Q_space, 1)
    verbose && println("  ", approx)
    verbose && print_sparsity_stats(Q_space, "Spatial precision")

    # -------------------------------------------------------------------------
    # Build initial state GMRF (two copies: state and time derivative)
    # -------------------------------------------------------------------------
    # State layout for one time slice includes both spatial state and its time derivative
    Q_state0 = [Q_space spzeros(size(Q_space)...);
                spzeros(size(Q_space)...) Q_space]

    N_state = size(Q_state0, 1)
    x0 = GMRF(zeros(N_state), Q_state0)

    # Extended layout for state + time derivatives
    N_fx = length(endpoints)
    full_state_layout = layout((
        f = N_fx,
        f_dx = N_fx,
        f_int = N_int,
        df_dt = N_fx,
        df_dx_dt = N_fx,
        df_int_dt = N_int
    ))

    verbose && println("  State dimension: $N_state")

    # -------------------------------------------------------------------------
    # Apply initial conditions
    # -------------------------------------------------------------------------
    verbose && println("\nApplying initial conditions...")

    ys = f_ic.(endpoints)
    ys_dx = f_ic_dx.(endpoints)
    ys_int = f_ic_int.(endpoints[1:(end-1)], endpoints[2:end])

    x0_ic = prescribe_indices(x0, indices(full_state_layout, :f), ys)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :f_dx), ys_dx)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :f_int), ys_int)

    # -------------------------------------------------------------------------
    # Apply FVM constraint at t=0
    # -------------------------------------------------------------------------
    verbose && println("Applying FVM constraint at t=0...")

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
    # Time discretization
    # -------------------------------------------------------------------------
    n_t = isnothing(n_timesteps) ? Int(ceil(T_end / Δt)) : n_timesteps

    verbose && println("\nTime discretization:")
    verbose && println("  Time steps: $n_t (Δt = $Δt)")
    verbose && println("  End time: $(n_t * Δt)")

    sde = IWPSDE(1.0)
    A_t, Σ_noise_t = discretize_vanloan(sde, Δt)
    Q_noise_t = inv(Σ_noise_t)
    Q_noise_t = 0.5 * (Q_noise_t + Q_noise_t')

    # Kronecker products for spacetime
    A = kronecker(A_t, sparse(I, N_space, N_space))
    Q_noise = kronecker(Q_noise_t, Q_space)

    # -------------------------------------------------------------------------
    # Build joint spacetime GMRF
    # -------------------------------------------------------------------------
    verbose && println("\nBuilding joint spacetime GMRF...")

    ssm = ConstantLinearGaussianSSM(x_fvm_0, A, Q_noise)
    x_joint = GPFiniteVolume.joint_gmrf(ssm, n_t)

    verbose && println("  Total spacetime dimension: $(length(mean(x_joint)))")

    # -------------------------------------------------------------------------
    # Apply boundary conditions
    # -------------------------------------------------------------------------
    verbose && println("Applying boundary conditions...")

    # Use full_state_layout for TimeStack since it includes time derivatives
    timest = TimeStack(mean(x_joint), full_state_layout)
    ys_bc = State(mean(x_fvm_0), full_state_layout).f[[1, end]]
    ys_bc = repeat(ys_bc, n_t - 1)

    bc_indices = absindices(timest, :f, [1, N_fx], 2:n_t)
    x_joint_bc = prescribe_indices(x_joint, bc_indices, ys_bc)

    # -------------------------------------------------------------------------
    # Apply FVM constraints at all time steps
    # -------------------------------------------------------------------------
    verbose && println("Applying FVM constraints at all time steps...")
    verbose && println("  Time scheme: $time_scheme")

    # Current time indices (t = 2:n_t)
    f_left_curr = absindices(timest, :f, 1:(N_fx-1), 2:n_t)
    f_right_curr = absindices(timest, :f, 2:N_fx, 2:n_t)
    f_dx_left_curr = absindices(timest, :f_dx, 1:(N_fx-1), 2:n_t)
    f_dx_right_curr = absindices(timest, :f_dx, 2:N_fx, 2:n_t)
    df_int_dt_curr = absindices(timest, :df_int_dt, 1:(N_fx-1), 2:n_t)

    # Previous time indices (t = 1:n_t-1) - for Crank-Nicolson schemes
    f_left_prev = absindices(timest, :f, 1:(N_fx-1), 1:(n_t-1))
    f_right_prev = absindices(timest, :f, 2:N_fx, 1:(n_t-1))
    f_dx_left_prev = absindices(timest, :f_dx, 1:(N_fx-1), 1:(n_t-1))
    f_dx_right_prev = absindices(timest, :f_dx, 2:N_fx, 1:(n_t-1))
    df_int_dt_prev = absindices(timest, :df_int_dt, 1:(N_fx-1), 1:(n_t-1))

    # Euler scheme: df_int/dt|_t + F|_t = 0
    function f_fvm_euler(x)
        dint_dt = x[df_int_dt_curr]
        F_left = 0.5 * x[f_left_curr].^2 - ν * x[f_dx_left_curr]
        F_right = 0.5 * x[f_right_curr].^2 - ν * x[f_dx_right_curr]
        return dint_dt + (F_right - F_left)
    end

    # Crank-Nicolson: df_int/dt|_t + 0.5*(F|_{t-1} + F|_t) = 0
    function f_fvm_crank_nicolson(x)
        dint_dt = x[df_int_dt_curr]

        # Fluxes at current time (t)
        F_L_curr = 0.5 * x[f_left_curr].^2 - ν * x[f_dx_left_curr]
        F_R_curr = 0.5 * x[f_right_curr].^2 - ν * x[f_dx_right_curr]

        # Fluxes at previous time (t-1)
        F_L_prev = 0.5 * x[f_left_prev].^2 - ν * x[f_dx_left_prev]
        F_R_prev = 0.5 * x[f_right_prev].^2 - ν * x[f_dx_right_prev]

        # CN: average net flux
        net_flux_avg = 0.5 * ((F_R_curr - F_L_curr) + (F_R_prev - F_L_prev))

        return dint_dt + net_flux_avg
    end

    # Symmetric Crank-Nicolson: 0.5*(df_int/dt|_{t-1} + df_int/dt|_t) + 0.5*(F|_{t-1} + F|_t) = 0
    function f_fvm_crank_nicolson_symmetric(x)
        # Average time derivative across both timesteps
        dint_dt_avg = 0.5 * (x[df_int_dt_prev] + x[df_int_dt_curr])

        # Fluxes at current time (t)
        F_L_curr = 0.5 * x[f_left_curr].^2 - ν * x[f_dx_left_curr]
        F_R_curr = 0.5 * x[f_right_curr].^2 - ν * x[f_dx_right_curr]

        # Fluxes at previous time (t-1)
        F_L_prev = 0.5 * x[f_left_prev].^2 - ν * x[f_dx_left_prev]
        F_R_prev = 0.5 * x[f_right_prev].^2 - ν * x[f_dx_right_prev]

        # Symmetric CN: average both sides
        net_flux_avg = 0.5 * ((F_R_curr - F_L_curr) + (F_R_prev - F_L_prev))

        return dint_dt_avg + net_flux_avg
    end

    # Select constraint function based on time scheme
    f_fvm_fn = if time_scheme == :euler
        f_fvm_euler
    elseif time_scheme == :crank_nicolson
        f_fvm_crank_nicolson
    elseif time_scheme == :crank_nicolson_symmetric
        f_fvm_crank_nicolson_symmetric
    else
        error("Unknown time scheme: $time_scheme. Use :euler, :crank_nicolson, or :crank_nicolson_symmetric")
    end

    fvm_model = NonlinearLeastSquaresModel(f_fvm_fn, length(x_joint_bc))
    y_fvm = zeros(length(f_left_curr))
    lik_fvm = fvm_model(y_fvm; σ=0.0001)

    # Add constraint to fix first element (helps with identifiability)
    A_constr = zeros(1, length(x_joint_bc))
    A_constr[1] = 1.0
    e = [mean(x_joint_bc)[1]]
    x_constr = ConstrainedGMRF(x_joint_bc, A_constr, e)

    verbose && println("Solving (this may take a moment)...")
    x_solution = gaussian_approximation(x_constr, lik_fvm; verbose=verbose)

    verbose && println("\nSolution complete!")

    return (
        solution = x_solution,
        state_layout = state_layout,
        full_state_layout = full_state_layout,
        endpoints = endpoints,
        intervals = intervals,
        n_timesteps = n_t,
        Δt = Δt,
        ν = ν,
        ρ = ρ,
        time_scheme = time_scheme
    )
end

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

"""
    animate_burgers_solution(result; kwargs...)

Create animation of Burgers equation solution.

# Arguments
- `result`: Output from `solve_sparse_burgers`

# Keyword arguments
- `filename="figures/sparse_burgers.mp4"`: Output filename
- `fps=20`: Frames per second
"""
function animate_burgers_solution(result;
                                   filename="figures/sparse_burgers.mp4",
                                   fps=20)
    (; solution, full_state_layout, endpoints, n_timesteps, Δt, time_scheme) = result

    means = TimeStack(mean(solution), full_state_layout)
    stds = TimeStack(std(solution), full_state_layout)

    # Determine y-axis limits
    all_means = [means[:f, t] for t in 1:n_timesteps]
    all_stds = [stds[:f, t] for t in 1:n_timesteps]
    y_min = minimum(minimum.(all_means) .- 2 .* maximum.(all_stds))
    y_max = maximum(maximum.(all_means) .+ 2 .* maximum.(all_stds))

    scheme_name = time_scheme == :euler ? "Euler" :
                  time_scheme == :crank_nicolson ? "CN" : "CN-Sym"

    fig = Figure(size=(800, 500), fontsize=14)
    ax = Axis(fig[1, 1],
        xlabel = "x",
        ylabel = "u(x, t)",
        title = "Viscid Burgers - Sparse GP-FVM [$scheme_name] (t = 0.00)"
    )
    ylims!(ax, y_min - 0.1, y_max + 0.1)

    xs = collect(Float64, endpoints)
    mean_obs = Observable(Vector{Float64}(means[:f, 1]))
    std_obs = Observable(Vector{Float64}(stds[:f, 1]))
    upper = @lift($mean_obs .+ 1.96 .* $std_obs)
    lower = @lift($mean_obs .- 1.96 .* $std_obs)

    band!(ax, xs, lower, upper, color=RGBAf(0, 0, 1, 0.3), label="95% CI")
    lines!(ax, xs, mean_obs, color=:blue, linewidth=2, label="Posterior mean")

    axislegend(ax, position=:rt)

    # Ensure output directory exists
    mkpath(dirname(filename))

    record(fig, filename, 1:n_timesteps; framerate=fps) do t
        mean_obs[] = Vector{Float64}(means[:f, t])
        std_obs[] = Vector{Float64}(stds[:f, t])
        ax.title = "Viscid Burgers - Sparse GP-FVM [$scheme_name] (t = $(round((t-1)*Δt, digits=2)))"
    end

    println("Animation saved: $filename")
    return filename
end

# ------------------------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------------------------

using ArgParse

function parse_commandline()
    s = ArgParseSettings(
        description = "Sparse GP-FVM solver for the viscid Burgers equation",
        version = "1.0",
        add_version = true
    )

    @add_arg_table! s begin
        "--nx", "-n"
            help = "Number of spatial grid points"
            arg_type = Int
            default = 121
        "--xmin"
            help = "Left boundary of spatial domain"
            arg_type = Float64
            default = -3.0
        "--xmax"
            help = "Right boundary of spatial domain"
            arg_type = Float64
            default = 3.0
        "--dt"
            help = "Time step size"
            arg_type = Float64
            default = 0.05
        "--tend", "-T"
            help = "End time"
            arg_type = Float64
            default = 2.0
        "--nu"
            help = "Viscosity coefficient"
            arg_type = Float64
            default = 0.01
        "--lengthscale", "-l"
            help = "Kernel lengthscale"
            arg_type = Float64
            default = 0.2
        "--smoothness", "-s"
            help = "Matérn smoothness parameter (1=3/2, 2=5/2)"
            arg_type = Int
            default = 2
        "--rho", "-r"
            help = "Sparse Cholesky threshold (larger = denser, more accurate)"
            arg_type = Float64
            default = 2.0
        "--time-scheme"
            help = "Time stepping scheme: euler, crank_nicolson, or crank_nicolson_symmetric"
            arg_type = Symbol
            default = :euler
        "--output", "-o"
            help = "Output animation filename"
            arg_type = String
            default = "figures/sparse_burgers.mp4"
        "--fps"
            help = "Animation frames per second"
            arg_type = Int
            default = 20
        "--quiet", "-q"
            help = "Suppress progress output"
            action = :store_true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    result = solve_sparse_burgers(
        N_x = args["nx"],
        x_min = args["xmin"],
        x_max = args["xmax"],
        Δt = args["dt"],
        T_end = args["tend"],
        ν = args["nu"],
        lengthscale = args["lengthscale"],
        smoothness = args["smoothness"],
        ρ = args["rho"],
        time_scheme = args["time-scheme"],
        verbose = !args["quiet"]
    )

    animate_burgers_solution(result;
        filename = args["output"],
        fps = args["fps"]
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
