"""
2D Nonlinear Shallow Water Equations with Sparse GP-FVM.

Combines several pieces of the framework:
- Nonlinear PDE (momentum flux has quadratic terms)
- System of PDEs (3 primary + 4 auxiliary = 7 coupled fields)
- 2D spatial domain with face integrals
- Joint prior including auxiliary flux variables
- Nonlinear conditioning via Newton iteration
- EKF-style spacetime approach via joint_gmrf
"""

using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using Kronecker: kronecker
using CairoMakie
using Unitful
using SparseConnectivityTracer, SparseMatrixColorings
using Zygote

import GaussianMarkovRandomFields: mean, std, precision_matrix, linear_condition, gaussian_approximation
import FunctionalGPs: ⊗
using GaussianMarkovRandomFields: NonlinearLeastSquaresModel

# Include sub-modules
include("problem.jl")
include("functionals.jl")
include("constraints.jl")
include("solver.jl")

# ------------------------------------------------------------------------------
# Main Solver
# ------------------------------------------------------------------------------

"""
    solve_nonlinear_swe(; kwargs...)

Solve the 2D nonlinear shallow water equations using sparse GP-FVM.

# Keyword arguments
- `N_x=21, N_y=21`: Number of grid points in x and y
- `Δt=1.0u"minute"`: Time step
- `n_timesteps=9`: Number of time steps
- `lengthscale=10.0u"km"`: Kernel lengthscale for primary state
- `flux_lengthscale=10.0u"km"`: Kernel lengthscale for auxiliary flux
- `smoothness=2`: Matérn smoothness
- `ρ=2.0`: Sparse Cholesky threshold
- `σ_fvm=1e-4`: FVM constraint noise std
- `σ_flux=1e-4`: Flux constraint noise std
- `prob=NonlinearSWEProblem()`: Physical problem parameters
- `units=SolveUnits()`: Unit system for solve
- `bathymetry_lengthscale=0.0`: Lengthscale (km) for Matérn bathymetry perturbation (0=linear slope only)
- `bathymetry_std=30.0`: Std (m) of Matérn bathymetry perturbation
- `bathymetry_seed=42`: Random seed for bathymetry generation
- `verbose=true`: Print progress

# Returns
Named tuple with solution and metadata.
"""
function solve_nonlinear_swe(;
        N_x = 21,
        N_y = 21,
        Δt = 1.0u"minute",
        n_timesteps = 9,
        lengthscale = 10.0u"km",
        flux_lengthscale = 10.0u"km",
        smoothness = 2,
        ρ = 2.0,
        σ_fvm = 1e-4,
        σ_flux = 1e-4,
        prob = NonlinearSWEProblem(),
        units = SolveUnits(),
        bathymetry_lengthscale = 0.0,  # km, 0 = no perturbation (linear slope only)
        bathymetry_std = 30.0,  # meters, std of Matérn perturbation
        bathymetry_seed = 42,
        use_ekf_init = true,  # Use EKF-style forward sweep for initialization
        ekf_gn_steps = 2,  # Number of GN steps per timestep in EKF sweep
        verbose = true
    )

    verbose && println("=" ^ 70)
    verbose && println("2D Nonlinear Shallow Water Equations - Sparse GP-FVM")
    verbose && println("=" ^ 70)

    # Convert to solve units (with optional Matérn bathymetry)
    p = to_solve_params(
        prob, N_x, N_y;
        units=units,
        bathymetry_lengthscale=bathymetry_lengthscale,
        bathymetry_std=bathymetry_std,
        seed=bathymetry_seed
    )
    L = units.length
    T = units.time

    Δt_solve = ustrip(T, uconvert(T, Δt))
    lengthscale_solve = ustrip(L, uconvert(L, lengthscale))
    flux_lengthscale_solve = ustrip(L, uconvert(L, flux_lengthscale))

    verbose && println("\nSolve units: $(L) for length, $(T) for time")
    verbose && println(p)

    # -------------------------------------------------------------------------
    # Grid setup
    # -------------------------------------------------------------------------
    xs, ys, x_intervals, y_intervals = setup_2d_grid(N_x, N_y, p.L_x, p.L_y)
    ginfo = GridInfo(xs, ys)

    verbose && println("\nSpatial grid:")
    verbose && println("  Grid points: $(N_x) × $(N_y) = $(ginfo.N_grid)")
    verbose && println("  Cells: $(ginfo.n_cells_x) × $(ginfo.n_cells_y) = $(ginfo.N_cells)")
    verbose && println("  Vertical faces: $(ginfo.n_vert_faces)")
    verbose && println("  Horizontal faces: $(ginfo.n_horiz_faces)")

    cfl = p.c * Δt_solve / min(ginfo.Δx, ginfo.Δy)
    verbose && println("  CFL number: $(round(cfl, digits=2))")

    # -------------------------------------------------------------------------
    # Build functionals and sparse precisions
    # -------------------------------------------------------------------------
    verbose && println("\nBuilding functionals...")
    funcs = build_functionals(xs, ys, x_intervals, y_intervals)

    k_primary = HalfIntegerMaternKernel(smoothness, [lengthscale_solve]) ⊗
                HalfIntegerMaternKernel(smoothness, [lengthscale_solve])
    k_flux = HalfIntegerMaternKernel(smoothness, [flux_lengthscale_solve]) ⊗
             HalfIntegerMaternKernel(smoothness, [flux_lengthscale_solve])

    verbose && println("\nBuilding sparse precisions...")
    prec_info = build_all_precisions(funcs, k_primary, k_flux, ρ; verbose=verbose)

    verbose && println("\nJoint spatial dimension: $(prec_info.N_space)")

    # -------------------------------------------------------------------------
    # Build state layouts
    # -------------------------------------------------------------------------
    spatial_layout = build_spatial_layout(ginfo)
    full_state_layout = build_full_state_layout(ginfo)

    verbose && println("Spatial layout: $(length(spatial_layout))")
    verbose && println("Full state layout (with time derivs): $(length(full_state_layout))")

    # -------------------------------------------------------------------------
    # Build initial state GMRF
    # -------------------------------------------------------------------------
    verbose && println("\nBuilding initial state GMRF...")

    # Full IWP state: [spatial; d/dt spatial]
    # Both parts have the same precision structure
    N_space = prec_info.N_space
    Q_state0 = blockdiag(prec_info.Q_space, prec_info.Q_space)

    x0 = GMRF(zeros(2 * N_space), Symmetric(Q_state0))

    verbose && println("  Spatial dimension: $(N_space)")
    verbose && println("  Full IWP state dimension: $(2 * N_space)")

    # -------------------------------------------------------------------------
    # Apply initial conditions
    # -------------------------------------------------------------------------
    verbose && println("\nApplying initial conditions...")

    # IC observations are mapped onto the full state layout (which includes
    # time derivatives) — see build_ic_observations.
    spatial_only_layout = build_spatial_layout(ginfo)
    A_ic, y_ic = build_ic_observations(p, funcs, ginfo, full_state_layout)
    n_ic = length(y_ic)
    Q_ϵ_ic = (1.0 / 1e-6^2) * sparse(I, n_ic, n_ic)

    x0_ic = linear_condition(x0; A=A_ic, Q_ϵ=Q_ϵ_ic, y=y_ic)

    verbose && println("  Applied $(n_ic) IC observations")

    # -------------------------------------------------------------------------
    # Time discretization and joint GMRF
    # -------------------------------------------------------------------------
    n_t = n_timesteps
    T_end = n_t * Δt_solve

    verbose && println("\nTime discretization:")
    verbose && println("  Time steps: $n_t (Δt = $Δt_solve $T)")
    verbose && println("  End time: $T_end $T")

    # IWP SDE for temporal dynamics
    sde = IWPSDE(1.0)
    A_t, Σ_noise_t = discretize_vanloan(sde, Δt_solve)
    Q_noise_t = inv(Σ_noise_t)
    Q_noise_t = 0.5 * (Q_noise_t + Q_noise_t')

    # Kronecker products for spacetime
    # Note: IWP state is [f, df/dt], so A_t is 2x2 applied to [spatial; d/dt spatial]
    A = kronecker(A_t, sparse(I, N_space, N_space))
    Q_noise = kronecker(Q_noise_t, prec_info.Q_space)

    # -------------------------------------------------------------------------
    # Build joint spacetime GMRF
    # -------------------------------------------------------------------------
    verbose && println("\nBuilding joint spacetime GMRF...")

    ssm = ConstantLinearGaussianSSM(x0_ic, A, Q_noise)
    x_joint = GPFiniteVolume.joint_gmrf(ssm, n_t)

    N_total = length(mean(x_joint))
    verbose && println("  Total spacetime dimension: $(N_total)")

    # TimeStack for indexing
    timest = TimeStack(mean(x_joint), full_state_layout)

    # -------------------------------------------------------------------------
    # Apply boundary conditions
    # -------------------------------------------------------------------------
    verbose && println("\nApplying boundary conditions...")

    # Reflecting walls: hu = 0 on left/right, hv = 0 on top/bottom
    # Note: We apply at face midpoints (hu_vert_face, hv_horiz_face)
    # since those are where the nonlinear constraints are evaluated

    # Left boundary (x = 0): hu_vert_face at i=1
    # Right boundary (x = L_x): hu_vert_face at i=N_x
    N_x_grid = ginfo.N_x
    n_cells_y = ginfo.n_cells_y

    # Vertical face midpoints at left/right edges
    left_vert_idcs = [1 + (j-1)*N_x_grid for j in 1:n_cells_y]
    right_vert_idcs = [N_x_grid + (j-1)*N_x_grid for j in 1:n_cells_y]
    hu_bc_spatial = vcat(left_vert_idcs, right_vert_idcs)

    # Horizontal face midpoints at bottom/top edges
    N_y_grid = ginfo.N_y
    n_cells_x = ginfo.n_cells_x
    bottom_horiz_idcs = [i for i in 1:n_cells_x]  # j=1
    top_horiz_idcs = [i + (N_y_grid-1)*n_cells_x for i in 1:n_cells_x]  # j=N_y
    hv_bc_spatial = vcat(bottom_horiz_idcs, top_horiz_idcs)

    # Apply at all times after t=1
    hu_bc_idcs = absindices(timest, :hu_vert_face, hu_bc_spatial, 2:n_t)
    hv_bc_idcs = absindices(timest, :hv_horiz_face, hv_bc_spatial, 2:n_t)

    n_bc = length(hu_bc_idcs) + length(hv_bc_idcs)
    all_bc_idcs = vcat(hu_bc_idcs, hv_bc_idcs)
    A_bc = sparse(1:n_bc, all_bc_idcs, ones(n_bc), n_bc, N_total)
    y_bc = zeros(n_bc)
    Q_ϵ_bc = (1.0 / 1e-6^2) * sparse(I, n_bc, n_bc)

    x_joint_bc = linear_condition(x_joint; A=A_bc, Q_ϵ=Q_ϵ_bc, y=y_bc)

    verbose && println("  Applied $(n_bc) boundary conditions")

    # -------------------------------------------------------------------------
    # Build FVM constraints (linear)
    # -------------------------------------------------------------------------
    verbose && println("\nBuilding FVM constraints...")

    # FVM is forward-looking for CN: uses t and t+1
    fvm_time_range = 1:(n_t - 1)
    A_fvm, b_fvm = build_fvm_constraint_matrix(timest, ginfo, p, fvm_time_range; crank_nicolson=true)

    n_fvm = size(A_fvm, 1)
    verbose && println("  FVM constraints: $n_fvm equations, $(nnz(A_fvm)) non-zeros")

    # -------------------------------------------------------------------------
    # Build nonlinear flux constraints
    # -------------------------------------------------------------------------
    verbose && println("\nBuilding nonlinear flux constraints...")

    # Flux constraints apply at all time steps
    flux_time_range = 1:n_t
    flux_idx = build_flux_constraint_indices(timest, ginfo, flux_time_range)
    n_flux = 8 * length(flux_idx.h_vert_face)

    verbose && println("  Flux constraints: $n_flux pointwise equations")

    # -------------------------------------------------------------------------
    # Apply FVM constraints (linear) via linear_condition
    # -------------------------------------------------------------------------
    verbose && println("\nApplying linear FVM constraints...")

    Q_ϵ_fvm = (1.0 / σ_fvm^2) * sparse(I, n_fvm, n_fvm)
    x_joint_fvm = linear_condition(x_joint_bc; A=A_fvm, Q_ϵ=Q_ϵ_fvm, y=b_fvm)

    verbose && println("  FVM constraints applied")

    # -------------------------------------------------------------------------
    # EKF-style initialization (forward sweep with few GN steps per block)
    # -------------------------------------------------------------------------
    if use_ekf_init
        verbose && println("\nRunning EKF-style forward sweep...")

        n_full_state = length(full_state_layout)

        # Build flux residual function for a SINGLE timestep using LOCAL indices
        # This uses State wrapper with full_state_layout (not TimeStack absindices)
        function flux_residual_single(x, layout, g)
            x_s = State(x, layout)

            # Extract values at face midpoints using State accessor
            h_v = x_s.h_vert_face
            hu_v = x_s.hu_vert_face
            hv_v = x_s.hv_vert_face
            P_v = x_s.P_vert_face
            Kx_v = x_s.Kx_vert_face
            Ky_v = x_s.Ky_vert_face
            C_v = x_s.C_vert_face

            h_h = x_s.h_horiz_face
            hu_h = x_s.hu_horiz_face
            hv_h = x_s.hv_horiz_face
            P_h = x_s.P_horiz_face
            Kx_h = x_s.Kx_horiz_face
            Ky_h = x_s.Ky_horiz_face
            C_h = x_s.C_horiz_face

            # Residuals: P = 0.5*g*h², Kx = hu²/h, Ky = hv²/h, C = hu*hv/h
            r_P_v = P_v .- 0.5 .* g .* h_v.^2
            r_Kx_v = Kx_v .- hu_v.^2 ./ h_v
            r_Ky_v = Ky_v .- hv_v.^2 ./ h_v
            r_C_v = C_v .- hu_v .* hv_v ./ h_v

            r_P_h = P_h .- 0.5 .* g .* h_h.^2
            r_Kx_h = Kx_h .- hu_h.^2 ./ h_h
            r_Ky_h = Ky_h .- hv_h.^2 ./ h_h
            r_C_h = C_h .- hu_h .* hv_h ./ h_h

            return vcat(r_P_v, r_Kx_v, r_Ky_v, r_C_v, r_P_h, r_Kx_h, r_Ky_h, r_C_h)
        end

        # Number of flux constraints per timestep
        # 4 constraints (P, Kx, Ky, C) at each face type
        n_flux_single = 4 * ginfo.n_vert_faces + 4 * ginfo.n_horiz_faces

        # Build the NonlinearLeastSquaresModel for single-timestep flux
        flux_model_single = NonlinearLeastSquaresModel(
            x -> flux_residual_single(x, full_state_layout, p.g),
            n_full_state
        )
        lik_flux_single = flux_model_single(zeros(n_flux_single); σ=σ_flux)

        # Forward sweep: process each timestep
        block_means = Vector{Vector{Float64}}(undef, n_t)
        block_means[1] = mean(x_joint_fvm)[1:n_full_state]  # t=1 is already conditioned on IC

        # Pre-compute marginal precision for efficiency
        Q_joint = precision_matrix(x_joint_fvm)

        for t_idx in 2:n_t
            verbose && print("  Block $t_idx/$n_t...")

            # Get indices for current block
            curr_start = (t_idx - 1) * n_full_state + 1
            curr_end = t_idx * n_full_state

            # Get prior mean from joint (already incorporates IWP transition)
            μ_t_prior = mean(x_joint_fvm)[curr_start:curr_end]

            # Extract marginal precision for this block
            Q_t_marginal = Q_joint[curr_start:curr_end, curr_start:curr_end]
            x_t_prior = GMRF(μ_t_prior, Q_t_marginal)

            # Run a few GN steps for the flux constraint
            x_t_solution = gaussian_approximation(x_t_prior, lik_flux_single;
                verbose=false, max_iter=ekf_gn_steps, newton_dec_tol=1e-8)

            block_means[t_idx] = mean(x_t_solution)
            verbose && println(" done")
        end

        # Concatenate to get full initialization
        μ_ekf_init = vcat(block_means...)
        verbose && println("  EKF initialization complete")

        # Create GMRF with EKF-initialized mean for batch solve
        x_joint_init = GMRF(μ_ekf_init, Q_joint)
    else
        x_joint_init = x_joint_fvm
    end

    # -------------------------------------------------------------------------
    # Apply nonlinear flux constraints via Gauss-Newton (batch)
    # -------------------------------------------------------------------------
    verbose && println("\nSolving nonlinear flux constraints (batch)...")

    lik_flux = build_flux_likelihood(N_total, flux_idx, p.g, σ_flux)

    # Solve via Gauss-Newton from (possibly EKF-initialized) mean
    x_solution = gaussian_approximation(x_joint_init, lik_flux;
        verbose=verbose, max_iter=50, newton_dec_tol=1e-5)

    verbose && println("\nSolution complete!")

    return (
        solution = x_solution,
        xs = xs,
        ys = ys,
        ginfo = ginfo,
        funcs = funcs,
        prec_info = prec_info,
        spatial_layout = spatial_layout,
        full_state_layout = full_state_layout,
        N_x = N_x,
        N_y = N_y,
        n_timesteps = n_t,
        Δt = Δt_solve,
        prob = prob,
        params = p,
        units = units,
    )
end

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

"""
    animate_nonlinear_swe(result; kwargs...)

Create animation of nonlinear shallow water solution.

# Keyword arguments
- `filename`: Output file path
- `fps`: Frames per second
- `view_3d`: If true, use 3D surface plot instead of heatmap
- `h_in_meters`: If true, convert h from km to meters for display
"""
function animate_nonlinear_swe(result;
                                filename="experiments/nonlinear_shallow_water/nonlinear_swe.mp4",
                                fps=10,
                                view_3d=true,
                                h_in_meters=true)
    (; solution, full_state_layout, xs, ys, N_x, N_y, n_timesteps, Δt, params, units) = result

    means = TimeStack(mean(solution), full_state_layout)

    L = units.length
    T = units.time

    # Convert h to meters if requested
    h_scale = h_in_meters ? 1000.0 : 1.0
    h_unit = h_in_meters ? "m" : string(L)

    # Collect all h mean values
    all_h = [h_scale .* reshape(means[:h, t], N_x, N_y) for t in 1:n_timesteps]

    # Determine z/color range based on initial perturbation
    h_mean = h_scale * params.H₀
    h_amp = h_scale * params.h_amplitude
    h_range = 2 * h_amp

    # Ensure output directory exists
    mkpath(dirname(filename))

    if view_3d
        fig = Figure(size=(900, 700), fontsize=14)

        ax = Axis3(fig[1, 1],
            xlabel = "x ($L)",
            ylabel = "y ($L)",
            zlabel = "h ($h_unit)",
            title = "Water depth h (t = 0.0 $T)",
            azimuth = -0.3π,
            elevation = 0.15π
        )

        h_obs = Observable(all_h[1])

        surface!(ax, collect(xs), collect(ys), h_obs,
                 colormap = :viridis, colorrange = (h_mean - h_range, h_mean + h_range))

        zlims!(ax, h_mean - 1.5*h_range, h_mean + 1.5*h_range)

        record(fig, filename, 1:n_timesteps; framerate=fps) do t
            h_obs[] = all_h[t]
            ax.title = "Nonlinear SWE - h (t = $(round((t-1)*Δt, digits=2)) $T)"
        end
    else
        fig = Figure(size=(800, 700), fontsize=14)

        ax = Axis(fig[1, 1],
            xlabel = "x ($L)",
            ylabel = "y ($L)",
            title = "Water depth h (t = 0.0 $T)",
            aspect = DataAspect()
        )

        h_obs = Observable(all_h[1])

        hm = heatmap!(ax, collect(xs), collect(ys), h_obs,
                      colormap = :viridis, colorrange = (h_mean - h_range, h_mean + h_range))
        contour!(ax, collect(xs), collect(ys), h_obs,
                 color = :white, linewidth = 0.5, levels = 10)

        Colorbar(fig[1, 2], hm, label = "h ($h_unit)")

        record(fig, filename, 1:n_timesteps; framerate=fps) do t
            h_obs[] = all_h[t]
            ax.title = "Nonlinear SWE - h (t = $(round((t-1)*Δt, digits=2)) $T)"
        end
    end

    println("Animation saved: $filename")
    return filename
end

function plot_solution_snapshot(result; t=1)
    (; solution, xs, ys, N_x, N_y, full_state_layout, params, units) = result

    means = TimeStack(mean(solution), full_state_layout)

    h_vals = means[:h, t]
    hu_vals = means[:hu, t]
    hv_vals = means[:hv, t]

    h_2d = reshape(h_vals, N_x, N_y)
    hu_2d = reshape(hu_vals, N_x, N_y)
    hv_2d = reshape(hv_vals, N_x, N_y)

    L = units.length

    fig = Figure(size=(1200, 400))

    ax1 = Axis(fig[1, 1], xlabel="x ($L)", ylabel="y ($L)", title="h", aspect=DataAspect())
    hm1 = heatmap!(ax1, collect(xs), collect(ys), h_2d, colormap=:viridis)
    Colorbar(fig[1, 2], hm1)

    ax2 = Axis(fig[1, 3], xlabel="x ($L)", ylabel="y ($L)", title="hu", aspect=DataAspect())
    hm2 = heatmap!(ax2, collect(xs), collect(ys), hu_2d, colormap=:balance)
    Colorbar(fig[1, 4], hm2)

    ax3 = Axis(fig[1, 5], xlabel="x ($L)", ylabel="y ($L)", title="hv", aspect=DataAspect())
    hm3 = heatmap!(ax3, collect(xs), collect(ys), hv_2d, colormap=:balance)
    Colorbar(fig[1, 6], hm3)

    return fig
end

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

using ArgParse

function parse_commandline()
    s = ArgParseSettings(
        description = "2D Nonlinear Shallow Water Equations with Sparse GP-FVM"
    )

    @add_arg_table! s begin
        "--nx"
            help = "Grid points in x"
            arg_type = Int
            default = 21
        "--ny"
            help = "Grid points in y"
            arg_type = Int
            default = 21
        "--dt"
            help = "Time step (minutes)"
            arg_type = Float64
            default = 1.0
        "--nsteps"
            help = "Number of time steps"
            arg_type = Int
            default = 9
        "--lengthscale", "-l"
            help = "Kernel lengthscale (km)"
            arg_type = Float64
            default = 10.0
        "--flux-lengthscale"
            help = "Flux kernel lengthscale (km)"
            arg_type = Float64
            default = 10.0
        "--rho", "-r"
            help = "Sparse Cholesky threshold"
            arg_type = Float64
            default = 2.0
        "--quiet", "-q"
            help = "Suppress output"
            action = :store_true
        "--animate"
            help = "Create animation"
            action = :store_true
        "--output", "-o"
            help = "Output animation filename"
            arg_type = String
            default = "experiments/nonlinear_shallow_water/nonlinear_swe.mp4"
        "--fps"
            help = "Animation FPS"
            arg_type = Int
            default = 10
        "--view-2d"
            help = "Use 2D heatmap instead of 3D surface"
            action = :store_true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    result = solve_nonlinear_swe(
        N_x = args["nx"],
        N_y = args["ny"],
        Δt = args["dt"] * u"minute",
        n_timesteps = args["nsteps"],
        lengthscale = args["lengthscale"] * u"km",
        flux_lengthscale = args["flux-lengthscale"] * u"km",
        ρ = args["rho"],
        verbose = !args["quiet"]
    )

    if args["animate"]
        animate_nonlinear_swe(result;
            filename = args["output"],
            fps = args["fps"],
            view_3d = !args["view-2d"]
        )
    end

    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
