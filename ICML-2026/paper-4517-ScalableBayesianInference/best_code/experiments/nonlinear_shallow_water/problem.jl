"""
Problem definition, units, and initial conditions for nonlinear SWE.
"""

using Unitful
using LinearAlgebra
using Random
using FunctionalGPs: HalfIntegerMaternKernel, kernelmatrix

const g = 9.81u"m/s^2"

# ------------------------------------------------------------------------------
# Problem Definition
# ------------------------------------------------------------------------------

"""
    NonlinearSWEProblem

Physical parameters for the nonlinear shallow water problem.
All quantities stored with Unitful units for safety.

Bathymetry: linear slope from shallow (x=0, shore) to deep (x=L_x, ocean).
"""
Base.@kwdef struct NonlinearSWEProblem
    # Domain size
    L_x::typeof(1.0u"m") = 100.0u"km"
    L_y::typeof(1.0u"m") = 100.0u"km"

    # Bathymetry: depth varies linearly in x
    H₀_deep::typeof(1.0u"m") = 500.0u"m"    # deep water (x = L_x, offshore)
    H₀_shallow::typeof(1.0u"m") = 50.0u"m"  # shallow water (x = 0, shore)

    # Initial condition: Gaussian perturbation (tsunami source offshore)
    h_amplitude::typeof(1.0u"m") = 10.0u"m"  # tsunami amplitude
    h_center_x::typeof(1.0u"m") = 70.0u"km"  # offshore position
    h_center_y::typeof(1.0u"m") = 50.0u"km"  # centered in y
    h_width_x::typeof(1.0u"m") = 5.0u"km"    # width perpendicular to shore (narrow)
    h_width_y::typeof(1.0u"m") = 20.0u"km"   # width along shore (wide, like fault line)
end

"""Wave speed at deep end: c = √(gH₀_deep)"""
wave_speed(prob::NonlinearSWEProblem) = sqrt(g * prob.H₀_deep)

# ------------------------------------------------------------------------------
# Unit System
# ------------------------------------------------------------------------------

"""
    SolveUnits

Configurable unit system for the numerical solve.
Default: km for length, min for time.
"""
Base.@kwdef struct SolveUnits
    length::Unitful.FreeUnits = u"km"
    time::Unitful.FreeUnits = u"minute"
end

velocity_unit(su::SolveUnits) = su.length / su.time
accel_unit(su::SolveUnits) = su.length / su.time^2

"""
    SolveParams

All parameters converted to chosen units and stripped for numerical solve.
Bathymetry gradients are stored per cell: db_dx[k], db_dy[k] for cell k.
"""
struct SolveParams
    L_x::Float64
    L_y::Float64
    H₀_deep::Float64
    H₀_shallow::Float64
    h_amplitude::Float64
    h_center_x::Float64
    h_center_y::Float64
    h_width_x::Float64
    h_width_y::Float64
    g::Float64
    c::Float64
    db_dx::Vector{Float64}  # bathymetry slope ∂b/∂x at each cell center
    db_dy::Vector{Float64}  # bathymetry slope ∂b/∂y at each cell center
    depth_at_cells::Vector{Float64}  # reference depth at cell centers (for IC)
    H0_vert_face::Matrix{Float64}    # H0 at vertical face midpoints (N_x × n_cells_y)
    H0_horiz_face::Matrix{Float64}   # H0 at horizontal face midpoints (n_cells_x × N_y)
    N_x::Int  # grid size for reference
    N_y::Int
    units::SolveUnits
end

"""
    to_solve_params(prob, N_x, N_y; units, bathymetry_lengthscale, bathymetry_std, seed)

Convert problem to solve parameters with spatially varying bathymetry.

If `bathymetry_lengthscale > 0`, adds Matérn GP perturbations to the linear slope.
"""
function to_solve_params(
    prob::NonlinearSWEProblem,
    N_x::Int,
    N_y::Int;
    units::SolveUnits=SolveUnits(),
    bathymetry_lengthscale::Float64=0.0,  # km, 0 = no perturbation
    bathymetry_std::Float64=30.0,  # m, std of perturbation
    seed::Int=42
)
    L = units.length
    T = units.time
    V = velocity_unit(units)
    A = accel_unit(units)

    c = wave_speed(prob)
    L_x_val = ustrip(L, uconvert(L, prob.L_x))
    L_y_val = ustrip(L, uconvert(L, prob.L_y))
    H_deep = ustrip(L, uconvert(L, prob.H₀_deep))
    H_shallow = ustrip(L, uconvert(L, prob.H₀_shallow))

    # Compute cell centers
    xs = range(0, L_x_val, length=N_x)
    ys = range(0, L_y_val, length=N_y)
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]

    n_cells = (N_x - 1) * (N_y - 1)
    cell_centers_x = Float64[]
    cell_centers_y = Float64[]
    for j in 1:(N_y-1), i in 1:(N_x-1)
        push!(cell_centers_x, (xs[i] + xs[i+1]) / 2)
        push!(cell_centers_y, (ys[j] + ys[j+1]) / 2)
    end

    # Base bathymetry: linear slope
    # Depth H₀(x) = H_shallow + (H_deep - H_shallow) * x/L_x
    # Bottom elevation b(x) = -H₀(x)
    base_depth = [H_shallow + (H_deep - H_shallow) * x / L_x_val for x in cell_centers_x]
    base_db_dx = fill(-(H_deep - H_shallow) / L_x_val, n_cells)
    base_db_dy = zeros(n_cells)

    # Add Matérn perturbation if requested
    if bathymetry_lengthscale > 0
        # Convert bathymetry_std from meters to solve units (typically km)
        bathymetry_std_solve = bathymetry_std * ustrip(L, 1u"m")
        depth, db_dx, db_dy = add_matern_bathymetry(
            cell_centers_x, cell_centers_y, base_depth,
            bathymetry_lengthscale, bathymetry_std_solve, dx, dy, seed
        )
    else
        depth = base_depth
        db_dx = base_db_dx
        db_dy = base_db_dy
    end

    # Ensure minimum depth (no dry cells)
    min_depth = 10.0 * ustrip(L, 1u"m")  # 10m minimum
    depth = max.(depth, min_depth)

    # Compute H0 at face midpoints (linear slope for base case)
    # These are used for well-balanced source term computation
    n_cells_x = N_x - 1
    n_cells_y = N_y - 1

    H0_vert_face = zeros(N_x, n_cells_y)
    for j in 1:n_cells_y, i in 1:N_x
        H0_vert_face[i, j] = H_shallow + (H_deep - H_shallow) * xs[i] / L_x_val
    end

    H0_horiz_face = zeros(n_cells_x, N_y)
    for j in 1:N_y, i in 1:n_cells_x
        x_c = (xs[i] + xs[i+1]) / 2
        H0_horiz_face[i, j] = H_shallow + (H_deep - H_shallow) * x_c / L_x_val
    end

    SolveParams(
        L_x_val,
        L_y_val,
        H_deep,
        H_shallow,
        ustrip(L, uconvert(L, prob.h_amplitude)),
        ustrip(L, uconvert(L, prob.h_center_x)),
        ustrip(L, uconvert(L, prob.h_center_y)),
        ustrip(L, uconvert(L, prob.h_width_x)),
        ustrip(L, uconvert(L, prob.h_width_y)),
        ustrip(A, uconvert(A, g)),
        ustrip(V, uconvert(V, c)),
        db_dx,
        db_dy,
        depth,
        H0_vert_face,
        H0_horiz_face,
        N_x,
        N_y,
        units
    )
end

"""
Add Matérn GP perturbation to bathymetry and compute gradients.
"""
function add_matern_bathymetry(
    cell_x::Vector{Float64},
    cell_y::Vector{Float64},
    base_depth::Vector{Float64},
    lengthscale::Float64,
    std::Float64,
    dx::Float64,
    dy::Float64,
    seed::Int
)
    Random.seed!(seed)

    n = length(cell_x)
    points = [(cell_x[i], cell_y[i]) for i in 1:n]

    # Sample from Matérn GP
    k = HalfIntegerMaternKernel(2, [lengthscale, lengthscale])
    K = std^2 * kernelmatrix(k, points)
    K += 1e-6 * I
    L_chol = cholesky(Symmetric(K)).L
    perturbation = L_chol * randn(n)

    # Total depth (ensure positive - minimum 1m = 0.001km in solve units)
    depth = base_depth .+ perturbation
    depth = max.(depth, 0.001)  # 1 meter minimum

    # Compute gradients via finite differences
    # Need to be careful about cell indexing
    # Cells are ordered: for j in 1:(N_y-1), for i in 1:(N_x-1)
    # So cell (i, j) has linear index k = (j-1)*(N_x-1) + i

    # For simplicity, compute gradients from the perturbation field
    # db/dx ≈ (depth[i+1,j] - depth[i-1,j]) / (2*dx) for interior
    # We need the 2D structure back

    # Infer grid dimensions
    # We have (N_x-1) * (N_y-1) cells
    # From spacing, figure out N_x-1
    unique_x = unique(round.(cell_x, digits=8))
    N_x_cells = length(unique_x)
    N_y_cells = n ÷ N_x_cells

    # Reshape to 2D
    depth_2d = reshape(depth, N_x_cells, N_y_cells)

    # Compute gradients
    db_dx_2d = zeros(N_x_cells, N_y_cells)
    db_dy_2d = zeros(N_x_cells, N_y_cells)

    # Interior points: central difference
    for j in 1:N_y_cells, i in 2:(N_x_cells-1)
        db_dx_2d[i, j] = -(depth_2d[i+1, j] - depth_2d[i-1, j]) / (2*dx)
    end
    # Boundary: one-sided
    for j in 1:N_y_cells
        db_dx_2d[1, j] = -(depth_2d[2, j] - depth_2d[1, j]) / dx
        db_dx_2d[N_x_cells, j] = -(depth_2d[N_x_cells, j] - depth_2d[N_x_cells-1, j]) / dx
    end

    for j in 2:(N_y_cells-1), i in 1:N_x_cells
        db_dy_2d[i, j] = -(depth_2d[i, j+1] - depth_2d[i, j-1]) / (2*dy)
    end
    for i in 1:N_x_cells
        db_dy_2d[i, 1] = -(depth_2d[i, 2] - depth_2d[i, 1]) / dy
        db_dy_2d[i, N_y_cells] = -(depth_2d[i, N_y_cells] - depth_2d[i, N_y_cells-1]) / dy
    end

    # Flatten back
    db_dx = vec(db_dx_2d)
    db_dy = vec(db_dy_2d)

    return depth, db_dx, db_dy
end

function Base.show(io::IO, p::SolveParams)
    print(io, "SolveParams(L=$(p.L_x)×$(p.L_y) $(p.units.length), ")
    print(io, "H₀=$(p.H₀_shallow)-$(p.H₀_deep) $(p.units.length), ")
    print(io, "db/dx range=$(round(minimum(p.db_dx), digits=4))..$(round(maximum(p.db_dx), digits=4)), ")
    print(io, "c_deep=$(round(p.c, digits=2)) $(velocity_unit(p.units)))")
end

# ------------------------------------------------------------------------------
# Bathymetry and Initial Conditions
# ------------------------------------------------------------------------------

"""Reference depth at cell k (1-indexed)."""
function H₀(p::SolveParams, k::Int)
    return p.depth_at_cells[k]
end

"""Gaussian bump amplitude at position (x, y). Anisotropic width."""
function gaussian_bump(p::SolveParams, x, y)
    dx = (x - p.h_center_x) / p.h_width_x
    dy = (y - p.h_center_y) / p.h_width_y
    return p.h_amplitude * exp(-0.5 * (dx^2 + dy^2))
end

# Note: IC values are now computed directly in solver.jl using depth_at_cells and gaussian_bump

# ------------------------------------------------------------------------------
# Grid Setup
# ------------------------------------------------------------------------------

function setup_2d_grid(N_x, N_y, L_x, L_y)
    xs = range(0, L_x, length=N_x)
    ys = range(0, L_y, length=N_y)
    x_intervals = intervals_from_endpoints(collect(xs))
    y_intervals = intervals_from_endpoints(collect(ys))
    return xs, ys, x_intervals, y_intervals
end
