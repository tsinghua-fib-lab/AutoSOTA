"""
2D Viscous Burgers Equation with Sparse GP-FVM

Demonstrates sparse Cholesky approximation for 2D spacetime GP-FVM.
Solves the 2D viscous Burgers equations:
    ∂u/∂t + u·∂u/∂x + v·∂u/∂y = ν(∂²u/∂x² + ∂²u/∂y²)
    ∂v/∂t + u·∂v/∂x + v·∂v/∂y = ν(∂²v/∂x² + ∂²v/∂y²)

Key features:
- 2D spatial domain with sparse precision via KL-optimal Cholesky
- Separable spacetime structure: IWP time prior ⊗ 2D sparse spatial prior
- Two coupled velocity fields (u, v) with block-diagonal prior
- Periodic boundary conditions
- Decaying vortex initial condition
"""

using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using Kronecker: kronecker
using CairoMakie
using QuadGK
using SparseConnectivityTracer, SparseMatrixColorings
using Zygote

import GaussianMarkovRandomFields: mean, std, precision_matrix, linear_condition
import FunctionalGPs: ⊗

# ------------------------------------------------------------------------------
# Initial condition: Decaying vortex
# ------------------------------------------------------------------------------

"""
Decaying vortex initial condition for 2D Burgers.
u = -sin(πy) * envelope(x, y)
v = sin(πx) * envelope(x, y)
"""
function u_initial(x, y; x0=0.5, y0=0.5, σ=0.2)
    envelope = exp(-((x - x0)^2 + (y - y0)^2) / (2 * σ^2))
    return -sin(π * y) * envelope
end

function v_initial(x, y; x0=0.5, y0=0.5, σ=0.2)
    envelope = exp(-((x - x0)^2 + (y - y0)^2) / (2 * σ^2))
    return sin(π * x) * envelope
end

# Derivatives for initial condition
function u_initial_dx(x, y; x0=0.5, y0=0.5, σ=0.2)
    envelope = exp(-((x - x0)^2 + (y - y0)^2) / (2 * σ^2))
    d_envelope_dx = -(x - x0) / σ^2 * envelope
    return -sin(π * y) * d_envelope_dx
end

function u_initial_dy(x, y; x0=0.5, y0=0.5, σ=0.2)
    envelope = exp(-((x - x0)^2 + (y - y0)^2) / (2 * σ^2))
    d_envelope_dy = -(y - y0) / σ^2 * envelope
    return -π * cos(π * y) * envelope + (-sin(π * y)) * d_envelope_dy
end

function v_initial_dx(x, y; x0=0.5, y0=0.5, σ=0.2)
    envelope = exp(-((x - x0)^2 + (y - y0)^2) / (2 * σ^2))
    d_envelope_dx = -(x - x0) / σ^2 * envelope
    return π * cos(π * x) * envelope + sin(π * x) * d_envelope_dx
end

function v_initial_dy(x, y; x0=0.5, y0=0.5, σ=0.2)
    envelope = exp(-((x - x0)^2 + (y - y0)^2) / (2 * σ^2))
    d_envelope_dy = -(y - y0) / σ^2 * envelope
    return sin(π * x) * d_envelope_dy
end

# Cell integral (approximate with midpoint)
function u_int_initial(x1, x2, y1, y2; kwargs...)
    xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return u_initial(xm, ym; kwargs...) * (x2 - x1) * (y2 - y1)
end

function v_int_initial(x1, x2, y1, y2; kwargs...)
    xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return v_initial(xm, ym; kwargs...) * (x2 - x1) * (y2 - y1)
end

# ------------------------------------------------------------------------------
# Grid setup
# ------------------------------------------------------------------------------

function setup_2d_grid(N_x, N_y, L_x, L_y)
    xs = range(0, L_x, length=N_x)
    ys = range(0, L_y, length=N_y)
    x_intervals = intervals_from_endpoints(collect(xs))
    y_intervals = intervals_from_endpoints(collect(ys))
    return xs, ys, x_intervals, y_intervals
end

# ------------------------------------------------------------------------------
# Main solver
# ------------------------------------------------------------------------------

"""
    solve_burgers_2d(; kwargs...)

Solve the 2D viscous Burgers equations using sparse GP-FVM.

# Keyword arguments
- `N_x=21, N_y=21`: Number of grid points in x and y
- `L_x=1.0, L_y=1.0`: Domain size
- `Δt=0.02`: Time step
- `T_end=1.0`: End time
- `ν=0.01`: Viscosity coefficient
- `lengthscale=0.1`: Kernel lengthscale
- `smoothness=2`: Matérn smoothness
- `ρ=2.0`: Sparse Cholesky threshold
- `verbose=true`: Print progress

# Returns
Named tuple with solution and metadata.
"""
function solve_burgers_2d(;
        N_x = 21,
        N_y = 21,
        L_x = 1.0,
        L_y = 1.0,
        Δt = 0.02,
        T_end = 1.0,
        ν = 0.01,
        lengthscale = 0.1,
        smoothness = 2,
        ρ = 2.0,
        verbose = true
    )

    verbose && println("=" ^ 60)
    verbose && println("2D Viscous Burgers - Sparse GP-FVM")
    verbose && println("=" ^ 60)

    # -------------------------------------------------------------------------
    # Spatial grid setup
    # -------------------------------------------------------------------------
    xs, ys, x_intervals, y_intervals = setup_2d_grid(N_x, N_y, L_x, L_y)
    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]

    N_grid = N_x * N_y
    N_cells_x = N_x - 1
    N_cells_y = N_y - 1
    N_cells = N_cells_x * N_cells_y

    verbose && println("\nSpatial grid:")
    verbose && println("  Domain: [0, $L_x] × [0, $L_y]")
    verbose && println("  Grid points: $(N_x) × $(N_y) = $(N_grid)")
    verbose && println("  Cells: $(N_cells_x) × $(N_cells_y) = $(N_cells)")
    verbose && println("  Δx = $(round(Δx, digits=4)), Δy = $(round(Δy, digits=4))")

    # -------------------------------------------------------------------------
    # Build 2D grid and functionals
    # -------------------------------------------------------------------------
    verbose && println("\nBuilding 2D functionals...")
    grid = FactorizedGrid(xs, ys)
    cells_2d = x_intervals ⊗ y_intervals

    # 2D product kernel
    k_2d = HalfIntegerMaternKernel(smoothness, [lengthscale]) ⊗
           HalfIntegerMaternKernel(smoothness, [lengthscale])

    # Functionals
    L_eval = EvaluationFunctional(grid)
    L_dx = L_eval ∘ PartialDerivative((1, 0))  # ∂/∂x
    L_dy = L_eval ∘ PartialDerivative((0, 1))  # ∂/∂y
    L_int = VectorizedLebesgueIntegral(cells_2d)

    # -------------------------------------------------------------------------
    # Build sparse precision for each velocity component
    # -------------------------------------------------------------------------
    verbose && println("\nBuilding sparse precisions (ρ = $ρ)...")

    # u: evaluations + derivatives (for flux) + cell integrals (for FVM)
    approx_u = sparse_precision([
        :u => L_eval,
        :u_dx => L_dx,
        :u_dy => L_dy,
        :u_int => L_int
    ], k_2d; ρ=ρ, ordering=:integrals_coarsest)
    verbose && println("  u: ", approx_u)

    # v: same structure
    approx_v = sparse_precision([
        :v => L_eval,
        :v_dx => L_dx,
        :v_dy => L_dy,
        :v_int => L_int
    ], k_2d; ρ=ρ, ordering=:integrals_coarsest)
    verbose && println("  v: ", approx_v)

    # -------------------------------------------------------------------------
    # Combine into joint spatial precision (block diagonal)
    # -------------------------------------------------------------------------
    n_u = approx_u.info.n
    n_v = approx_v.info.n

    Q_space = blockdiag(sparse(approx_u.Q), sparse(approx_v.Q))
    N_space = size(Q_space, 1)

    verbose && println("  Joint spatial dimension: $(N_space) (u:$(n_u) + v:$(n_v))")

    # -------------------------------------------------------------------------
    # Build state layout
    # -------------------------------------------------------------------------
    full_state_layout = layout((
        # u field
        u = N_grid,
        u_dx = N_grid,
        u_dy = N_grid,
        u_int = N_cells,
        # v field
        v = N_grid,
        v_dx = N_grid,
        v_dy = N_grid,
        v_int = N_cells,
        # Time derivatives
        du_dt = N_grid,
        du_dx_dt = N_grid,
        du_dy_dt = N_grid,
        du_int_dt = N_cells,
        dv_dt = N_grid,
        dv_dx_dt = N_grid,
        dv_dy_dt = N_grid,
        dv_int_dt = N_cells,
    ))

    # -------------------------------------------------------------------------
    # Build initial state GMRF
    # -------------------------------------------------------------------------
    Q_state0 = [Q_space spzeros(size(Q_space)...);
                spzeros(size(Q_space)...) Q_space]
    N_state = size(Q_state0, 1)
    x0 = GMRF(zeros(N_state), Q_state0)

    verbose && println("  Full state dimension: $(N_state)")

    # -------------------------------------------------------------------------
    # Apply initial conditions
    # -------------------------------------------------------------------------
    verbose && println("\nApplying initial conditions...")

    # Grid coordinates (x changes fastest - column-major)
    grid_coords = [(x, y) for y in ys for x in xs]

    # Cell coordinates (for cell integrals)
    cell_bounds = [((xs[i], xs[i+1]), (ys[j], ys[j+1]))
                   for j in 1:N_cells_y for i in 1:N_cells_x]

    # Compute IC values
    u_ic = [u_initial(x, y) for (x, y) in grid_coords]
    u_dx_ic = [u_initial_dx(x, y) for (x, y) in grid_coords]
    u_dy_ic = [u_initial_dy(x, y) for (x, y) in grid_coords]
    u_int_ic = [u_int_initial(xb[1], xb[2], yb[1], yb[2]) for (xb, yb) in cell_bounds]

    v_ic = [v_initial(x, y) for (x, y) in grid_coords]
    v_dx_ic = [v_initial_dx(x, y) for (x, y) in grid_coords]
    v_dy_ic = [v_initial_dy(x, y) for (x, y) in grid_coords]
    v_int_ic = [v_int_initial(xb[1], xb[2], yb[1], yb[2]) for (xb, yb) in cell_bounds]

    # Prescribe IC
    x0_ic = prescribe_indices(x0, indices(full_state_layout, :u), u_ic)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :u_dx), u_dx_ic)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :u_dy), u_dy_ic)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :u_int), u_int_ic)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :v), v_ic)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :v_dx), v_dx_ic)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :v_dy), v_dy_ic)
    x0_ic = prescribe_indices(x0_ic, indices(full_state_layout, :v_int), v_int_ic)

    # -------------------------------------------------------------------------
    # Time discretization
    # -------------------------------------------------------------------------
    n_t = Int(ceil(T_end / Δt))

    verbose && println("\nTime discretization:")
    verbose && println("  Time steps: $n_t (Δt = $Δt)")
    verbose && println("  End time: $(n_t * Δt)")
    verbose && println("  Viscosity: ν = $ν")

    # IWP SDE for temporal dynamics
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

    ssm = ConstantLinearGaussianSSM(x0_ic, A, Q_noise)
    x_joint = GPFiniteVolume.joint_gmrf(ssm, n_t)

    verbose && println("  Total spacetime dimension: $(length(mean(x_joint)))")

    # TimeStack for indexing
    timest = TimeStack(mean(x_joint), full_state_layout)

    # -------------------------------------------------------------------------
    # Cell corner indices (for FVM flux computation)
    # -------------------------------------------------------------------------
    # Grid ordering: x changes fastest, linear index = (j-1)*N_x + i
    # Cell (i,j) corners:
    #   SW = (i, j)     -> (j-1)*N_x + i
    #   SE = (i+1, j)   -> (j-1)*N_x + (i+1)
    #   NW = (i, j+1)   -> j*N_x + i
    #   NE = (i+1, j+1) -> j*N_x + (i+1)
    cell_corners = [(
        (j-1)*N_x + i,      # SW
        (j-1)*N_x + (i+1),  # SE
        j*N_x + i,          # NW
        j*N_x + (i+1)       # NE
    ) for j in 1:N_cells_y for i in 1:N_cells_x]

    # -------------------------------------------------------------------------
    # Build FVM constraints (nonlinear Burgers)
    # -------------------------------------------------------------------------
    verbose && println("\nBuilding FVM constraints...")

    # Index arrays for all time steps > 1
    u_curr = absindices(timest, :u, 1:N_grid, 2:n_t)
    u_dx_curr = absindices(timest, :u_dx, 1:N_grid, 2:n_t)
    u_dy_curr = absindices(timest, :u_dy, 1:N_grid, 2:n_t)
    du_int_dt_curr = absindices(timest, :du_int_dt, 1:N_cells, 2:n_t)

    v_curr = absindices(timest, :v, 1:N_grid, 2:n_t)
    v_dx_curr = absindices(timest, :v_dx, 1:N_grid, 2:n_t)
    v_dy_curr = absindices(timest, :v_dy, 1:N_grid, 2:n_t)
    dv_int_dt_curr = absindices(timest, :dv_int_dt, 1:N_cells, 2:n_t)

    n_fvm = N_cells * (n_t - 1)  # per equation

    """
    FVM constraint for 2D Burgers (u equation):
    du_int/dt + ∫∫(u·∂u/∂x + v·∂u/∂y)dA = ν·∮(∂u/∂n)dl

    Using divergence theorem for advection and diffusion:
    du_int/dt + (F_E - F_W + F_N - F_S) = 0

    where F = advective flux - viscous flux at each face

    Note: Written in functional style (no mutation) for Zygote compatibility.
    """
    function f_fvm_u(x)
        u = x[u_curr]
        u_dx = x[u_dx_curr]
        u_dy = x[u_dy_curr]
        v = x[v_curr]
        du_int_dt = x[du_int_dt_curr]

        # Compute residuals for each cell at each time step (functional style)
        residuals = map(1:((n_t-1)*N_cells)) do idx
            # Convert linear index to (k, t)
            t = div(idx - 1, N_cells) + 1
            k = mod(idx - 1, N_cells) + 1

            # Time offset in flattened arrays
            t_off = (t - 1) * N_grid
            t_off_cell = (t - 1) * N_cells

            sw, se, nw, ne = cell_corners[k]

            # Get values at corners for this time step
            u_sw = u[t_off + sw]
            u_se = u[t_off + se]
            u_nw = u[t_off + nw]
            u_ne = u[t_off + ne]

            u_dx_sw = u_dx[t_off + sw]
            u_dx_se = u_dx[t_off + se]
            u_dx_nw = u_dx[t_off + nw]
            u_dx_ne = u_dx[t_off + ne]

            u_dy_sw = u_dy[t_off + sw]
            u_dy_se = u_dy[t_off + se]
            u_dy_nw = u_dy[t_off + nw]
            u_dy_ne = u_dy[t_off + ne]

            v_sw = v[t_off + sw]
            v_se = v[t_off + se]
            v_nw = v[t_off + nw]
            v_ne = v[t_off + ne]

            # East face: F_E = 0.5*(u_SE² + u_NE²)*Δy - ν*(u_dx_SE + u_dx_NE)/2*Δy
            F_E = 0.5 * (u_se^2 + u_ne^2) * Δy - ν * 0.5 * (u_dx_se + u_dx_ne) * Δy

            # West face
            F_W = 0.5 * (u_sw^2 + u_nw^2) * Δy - ν * 0.5 * (u_dx_sw + u_dx_nw) * Δy

            # North face: u*v - ν*∂u/∂y
            F_N = 0.5 * (u_nw * v_nw + u_ne * v_ne) * Δx - ν * 0.5 * (u_dy_nw + u_dy_ne) * Δx

            # South face
            F_S = 0.5 * (u_sw * v_sw + u_se * v_se) * Δx - ν * 0.5 * (u_dy_sw + u_dy_se) * Δx

            # FVM: du_int/dt + net flux = 0
            du_int_dt[t_off_cell + k] + (F_E - F_W) + (F_N - F_S)
        end

        return residuals
    end

    # Similar for v equation
    function f_fvm_v(x)
        u = x[u_curr]
        v = x[v_curr]
        v_dx = x[v_dx_curr]
        v_dy = x[v_dy_curr]
        dv_int_dt = x[dv_int_dt_curr]

        residuals = map(1:((n_t-1)*N_cells)) do idx
            t = div(idx - 1, N_cells) + 1
            k = mod(idx - 1, N_cells) + 1

            t_off = (t - 1) * N_grid
            t_off_cell = (t - 1) * N_cells

            sw, se, nw, ne = cell_corners[k]

            u_sw = u[t_off + sw]
            u_se = u[t_off + se]
            u_nw = u[t_off + nw]
            u_ne = u[t_off + ne]

            v_sw = v[t_off + sw]
            v_se = v[t_off + se]
            v_nw = v[t_off + nw]
            v_ne = v[t_off + ne]

            v_dx_sw = v_dx[t_off + sw]
            v_dx_se = v_dx[t_off + se]
            v_dx_nw = v_dx[t_off + nw]
            v_dx_ne = v_dx[t_off + ne]

            v_dy_sw = v_dy[t_off + sw]
            v_dy_se = v_dy[t_off + se]
            v_dy_nw = v_dy[t_off + nw]
            v_dy_ne = v_dy[t_off + ne]

            # East face: u*v - ν*∂v/∂x
            F_E = 0.5 * (u_se * v_se + u_ne * v_ne) * Δy - ν * 0.5 * (v_dx_se + v_dx_ne) * Δy

            # West face
            F_W = 0.5 * (u_sw * v_sw + u_nw * v_nw) * Δy - ν * 0.5 * (v_dx_sw + v_dx_nw) * Δy

            # North face: v² - ν*∂v/∂y
            F_N = 0.5 * (v_nw^2 + v_ne^2) * Δx - ν * 0.5 * (v_dy_nw + v_dy_ne) * Δx

            # South face
            F_S = 0.5 * (v_sw^2 + v_se^2) * Δx - ν * 0.5 * (v_dy_sw + v_dy_se) * Δx

            dv_int_dt[t_off_cell + k] + (F_E - F_W) + (F_N - F_S)
        end

        return residuals
    end

    # Combined FVM constraint
    function f_fvm_combined(x)
        return vcat(f_fvm_u(x), f_fvm_v(x))
    end

    # -------------------------------------------------------------------------
    # Solve via Gaussian approximation (nonlinear least squares)
    # -------------------------------------------------------------------------
    verbose && println("Solving (nonlinear least squares)...")

    fvm_model = NonlinearLeastSquaresModel(f_fvm_combined, length(x_joint))
    y_fvm = zeros(2 * n_fvm)
    lik_fvm = fvm_model(y_fvm; σ=0.0001)

    x_solution = gaussian_approximation(x_joint, lik_fvm; verbose=verbose)

    verbose && println("\nSolution complete!")

    return (
        solution = x_solution,
        full_state_layout = full_state_layout,
        xs = xs,
        ys = ys,
        N_x = N_x,
        N_y = N_y,
        n_timesteps = n_t,
        Δt = Δt,
        ν = ν,
        ρ = ρ
    )
end

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

"""
    animate_burgers_2d(result; filename="burgers_2d.mp4", fps=20)

Create animation of 2D Burgers solution showing u and v velocity fields.
"""
function animate_burgers_2d(result;
                             filename="burgers_2d.mp4",
                             fps=20)
    (; solution, full_state_layout, xs, ys, N_x, N_y, n_timesteps, Δt) = result

    means = TimeStack(mean(solution), full_state_layout)

    # Determine color limits from all timesteps
    u_all = [reshape(means[:u, t], N_x, N_y) for t in 1:n_timesteps]
    v_all = [reshape(means[:v, t], N_x, N_y) for t in 1:n_timesteps]

    u_lim = maximum(maximum.(abs.(u_all)))
    v_lim = maximum(maximum.(abs.(v_all)))
    clim = max(u_lim, v_lim)

    fig = Figure(size=(1200, 500), fontsize=14)

    ax1 = Axis(fig[1, 1],
        xlabel = "x",
        ylabel = "y",
        title = "u velocity (t = 0.00)",
        aspect = DataAspect()
    )

    ax2 = Axis(fig[1, 2],
        xlabel = "x",
        ylabel = "y",
        title = "v velocity (t = 0.00)",
        aspect = DataAspect()
    )

    # Initial data
    u_obs = Observable(reshape(means[:u, 1], N_x, N_y)')
    v_obs = Observable(reshape(means[:v, 1], N_x, N_y)')

    hm1 = heatmap!(ax1, collect(xs), collect(ys), u_obs, colormap=:RdBu, colorrange=(-clim, clim))
    hm2 = heatmap!(ax2, collect(xs), collect(ys), v_obs, colormap=:RdBu, colorrange=(-clim, clim))

    Colorbar(fig[1, 3], hm1, label="velocity")

    record(fig, filename, 1:n_timesteps; framerate=fps) do t
        time = (t - 1) * Δt
        u_obs[] = reshape(means[:u, t], N_x, N_y)'
        v_obs[] = reshape(means[:v, t], N_x, N_y)'
        ax1.title = "u velocity (t = $(round(time, digits=2)))"
        ax2.title = "v velocity (t = $(round(time, digits=2)))"
    end

    println("Animation saved: $filename")
    return filename
end

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

function main()
    println("Running 2D Viscous Burgers...")

    result = solve_burgers_2d(
        N_x = 21,
        N_y = 21,
        Δt = 0.02,
        T_end = 0.5,
        ν = 0.01,
        lengthscale = 0.1,
        ρ = 2.0,
        verbose = true
    )

    animate_burgers_2d(result; filename="burgers_2d.mp4")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
