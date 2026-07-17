"""
EKF-Style Sequential Solver for 2D Nonlinear Shallow Water Equations.

Uses `solve_ekf_sequential()` from GPFiniteVolume with proper moment-matching.
"""

using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using Kronecker: kronecker
import LinearAlgebra: kron
using CairoMakie
using Unitful
using SparseConnectivityTracer, SparseMatrixColorings
using Statistics
using ArgParse
using JLD2

import GaussianMarkovRandomFields: mean, var, std, precision_matrix, linear_condition, gaussian_approximation
import FunctionalGPs: ⊗
using GaussianMarkovRandomFields: NonlinearLeastSquaresModel

include("problem.jl")
include("functionals.jl")
include("constraints.jl")
include("solver.jl")
include("bathymetry_gp.jl")

# ==============================================================================
# Command-line Arguments
# ==============================================================================

function parse_commandline()
    s = ArgParseSettings(description="EKF-Style 2D Nonlinear Shallow Water Equations")

    @add_arg_table! s begin
        "--nx"
            help = "Grid points in x direction"
            arg_type = Int
            default = 21
        "--ny"
            help = "Grid points in y direction"
            arg_type = Int
            default = 21
        "--tend"
            help = "End time in minutes"
            arg_type = Float64
            default = 27.5
        "--dt"
            help = "Time step in minutes"
            arg_type = Float64
            default = 0.25
        "--lengthscale"
            help = "GP lengthscale in km"
            arg_type = Float64
            default = 10.0
        "--rho"
            help = "Sparsity parameter ρ"
            arg_type = Float64
            default = 2.0
        "--open-right"
            help = "Use open outflow boundary at right (x=L_x)"
            action = :store_true
        "--closed-right"
            help = "Use closed wall at right (overrides --open-right)"
            action = :store_true
        "--bathy-lengthscale"
            help = "Bathymetry perturbation lengthscale in km (0 = no perturbation)"
            arg_type = Float64
            default = 15.0
        "--bathy-std"
            help = "Bathymetry perturbation std in meters"
            arg_type = Float64
            default = 30.0
        "--seed"
            help = "Random seed for bathymetry"
            arg_type = Int
            default = 42
        "--output"
            help = "Output directory for results"
            arg_type = String
            default = "results/swe"
        "--no-save"
            help = "Skip saving results (just run and plot)"
            action = :store_true
    end

    return parse_args(s)
end

args = parse_commandline()

# ==============================================================================
# Configuration
# ==============================================================================

N_x = args["nx"]
N_y = args["ny"]
Δt_val = args["dt"]
n_timesteps = round(Int, args["tend"] / Δt_val)

prob = NonlinearSWEProblem()
units = SolveUnits()

σ_fvm = 1e-4
σ_flux = 1e-4
lengthscale = args["lengthscale"] * u"km"
flux_lengthscale = args["lengthscale"] * u"km"
smoothness = 2
ρ = args["rho"]

# Boundary condition: open by default, unless --closed-right is specified
USE_OPEN_RIGHT_BC = !args["closed-right"]

# Bathymetry perturbation
bathy_lengthscale = args["bathy-lengthscale"]
bathy_std = args["bathy-std"]
seed = args["seed"]

# Output directory
output_dir = args["output"]
save_results = !args["no-save"]

println("=" ^ 70)
println("EKF-Style 2D Nonlinear Shallow Water Equations")
println("=" ^ 70)
println("Grid: $(N_x)×$(N_y), dt=$(Δt_val) min, $(n_timesteps) steps → t_end=$(n_timesteps * Δt_val) min")
println("Right boundary: $(USE_OPEN_RIGHT_BC ? "OPEN (outflow)" : "CLOSED (wall)")")
if bathy_lengthscale > 0
    println("Bathymetry: Matérn perturbation (ℓ=$(bathy_lengthscale) km, σ=$(bathy_std) m, seed=$(seed))")
else
    println("Bathymetry: linear slope only (no perturbation)")
end

# ==============================================================================
# Setup
# ==============================================================================

L = units.length
T = units.time

Δt_solve = ustrip(T, uconvert(T, Δt_val * u"minute"))
lengthscale_solve = ustrip(L, uconvert(L, lengthscale))
flux_lengthscale_solve = ustrip(L, uconvert(L, flux_lengthscale))

# Convert problem parameters to solve units (linear slope only - GP handles perturbation)
p_base = to_solve_params(prob, N_x, N_y; units=units, bathymetry_lengthscale=0.0)

println("\nProblem: $(N_x)×$(N_y) grid, $(n_timesteps) timesteps")

# Grid setup
xs, ys, x_intervals, y_intervals = setup_2d_grid(N_x, N_y, p_base.L_x, p_base.L_y)
ginfo = GridInfo(xs, ys)

println("Grid: $(ginfo.N_grid) points, $(ginfo.N_cells) cells")

n_cells_x = N_x - 1
n_cells_y = N_y - 1

# Sample bathymetry from GP (or use linear slope if no perturbation)
if bathy_lengthscale > 0
    println("\nSampling bathymetry from GP...")
    # σ_obs controls perturbation magnitude (bathy_std in meters → convert to km)
    σ_obs_km = bathy_std / 1000.0

    bathy = sample_bathymetry_gp(
        xs, ys, p_base.H₀_shallow, p_base.H₀_deep, p_base.L_x;
        lengthscale=bathy_lengthscale,
        σ_obs=σ_obs_km,
        ρ=ρ,
        seed=seed,
        verbose=true
    )

    # Create SolveParams with GP-sampled bathymetry
    p = SolveParams(
        p_base.L_x, p_base.L_y,
        p_base.H₀_deep, p_base.H₀_shallow,
        p_base.h_amplitude, p_base.h_center_x, p_base.h_center_y,
        p_base.h_width_x, p_base.h_width_y,
        p_base.g, p_base.c,
        bathy.db_dx,           # GP-sampled derivatives
        bathy.db_dy,
        bathy.depth_at_cells,  # GP-sampled depths at cell centers
        bathy.H0_vert_face,    # GP-sampled H0 at vertical face midpoints
        bathy.H0_horiz_face,   # GP-sampled H0 at horizontal face midpoints
        N_x, N_y, units
    )

    # H0_arrays from GP sample (consistent across all locations)
    H0_arrays = to_H0_arrays(bathy)
else
    # Linear slope only - use base parameters
    p = p_base

    # Linear slope formula at all locations
    H0_grid = [p.H₀_shallow + (p.H₀_deep - p.H₀_shallow) * x / p.L_x for x in xs, y in ys]

    H0_cell = zeros(n_cells_x, n_cells_y)
    for j in 1:n_cells_y, i in 1:n_cells_x
        x_c = 0.5 * (xs[i] + xs[i+1])
        H0_cell[i, j] = p.H₀_shallow + (p.H₀_deep - p.H₀_shallow) * x_c / p.L_x
    end

    H0_vert_face = zeros(N_x, n_cells_y)
    for j in 1:n_cells_y, i in 1:N_x
        H0_vert_face[i, j] = p.H₀_shallow + (p.H₀_deep - p.H₀_shallow) * xs[i] / p.L_x
    end

    H0_horiz_face = zeros(n_cells_x, N_y)
    for j in 1:N_y, i in 1:n_cells_x
        x_c = 0.5 * (xs[i] + xs[i+1])
        H0_horiz_face[i, j] = p.H₀_shallow + (p.H₀_deep - p.H₀_shallow) * x_c / p.L_x
    end

    H0_arrays = (H0_grid=H0_grid, H0_cell=H0_cell, H0_vert_face=H0_vert_face, H0_horiz_face=H0_horiz_face)
end

# Build functionals and sparse precisions
println("\nBuilding functionals and precisions...")
funcs = build_functionals(xs, ys, x_intervals, y_intervals)

k_primary = HalfIntegerMaternKernel(smoothness, [lengthscale_solve]) ⊗
            HalfIntegerMaternKernel(smoothness, [lengthscale_solve])
k_flux = HalfIntegerMaternKernel(smoothness, [flux_lengthscale_solve]) ⊗
         HalfIntegerMaternKernel(smoothness, [flux_lengthscale_solve])

prec_info = build_all_precisions(funcs, k_primary, k_flux, ρ; verbose=false)
N_space = prec_info.N_space

println("Spatial dimension: $(N_space)")

# Build state layouts
spatial_layout = build_spatial_layout(ginfo)
full_state_layout = build_full_state_layout(ginfo)
N_state = length(full_state_layout)

println("Full state dimension (with time derivs): $(N_state)")

# ==============================================================================
# Initial State
# ==============================================================================

println("\nBuilding initial state and applying ICs...")

Q_spatial_full = blockdiag(prec_info.Q_space, prec_info.Q_space)
x0 = GMRF(zeros(N_state), Symmetric(Q_spatial_full))

A_ic, y_ic = build_ic_observations(p, funcs, ginfo, full_state_layout; H0_arrays=H0_arrays)
n_ic = length(y_ic)
Q_ϵ_ic = (1.0 / 1e-6^2) * sparse(I, n_ic, n_ic)
x0_ic = linear_condition(x0; A=A_ic, Q_ϵ=Q_ϵ_ic, y=y_ic)

println("Applied $(n_ic) IC observations")

# ==============================================================================
# Time Discretization
# ==============================================================================

sde = IWPSDE(0.01)  # Allow more dynamics
A_t, Σ_noise_t = discretize_vanloan(sde, Δt_solve)
Q_noise_t = inv(Σ_noise_t)
Q_noise_t = 0.5 * (Q_noise_t + Q_noise_t')

# IMPORTANT: Use sparse kron to avoid dense 14286×14286 matrices!
A_kr = kron(sparse(A_t), sparse(I, N_space, N_space))
Q_noise_sparse = kron(sparse(Q_noise_t), prec_info.Q_space)

println("A_kr: $(size(A_kr)) with $(nnz(A_kr)) nonzeros ($(round(100*nnz(A_kr)/prod(size(A_kr)), digits=2))%)")
println("Q_noise_sparse: $(size(Q_noise_sparse)) with $(nnz(Q_noise_sparse)) nonzeros")

# ==============================================================================
# Apply t=0 Flux Constraints
# ==============================================================================

println("\nApplying t=0 flux constraints...")

function flux_residual_single(x, layout, g)
    x_s = State(x, layout)

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

n_flux_single = 4 * ginfo.n_vert_faces + 4 * ginfo.n_horiz_faces
flux_model_0 = NonlinearLeastSquaresModel(
    x -> flux_residual_single(x, full_state_layout, p.g),
    N_state
)
lik_flux_0 = flux_model_0(zeros(n_flux_single); σ=σ_flux)

x_t0 = gaussian_approximation(x0_ic, lik_flux_0; verbose=false, max_iter=10)

println("t=0 complete")

# ==============================================================================
# Define Constraint Builder for EKF
# ==============================================================================

# Helper functions
function cell_idx_to_ij(k)
    n_cells_x = ginfo.n_cells_x
    j = (k - 1) ÷ n_cells_x + 1
    i = (k - 1) % n_cells_x + 1
    return i, j
end

vert_face_idx(i, j) = i + (j - 1) * ginfo.N_x
horiz_face_idx(i, j) = i + (j - 1) * ginfo.n_cells_x

# Combined residual function for 2-block state (AD-compatible: no push!/append!)
function combined_residual(x_joint)
    x_prev_state = State(x_joint[1:N_state], full_state_layout)
    x_curr_state = State(x_joint[N_state+1:2N_state], full_state_layout)

    N_cells = ginfo.N_cells

    # Mass conservation (Crank-Nicolson)
    mass_res = map(1:N_cells) do k
        i, j = cell_idx_to_ij(k)
        dh_dt = 0.5 * (x_prev_state.dh_int_dt[k] + x_curr_state.dh_int_dt[k])
        div_prev = (x_prev_state.hu_vert[vert_face_idx(i+1, j)] - x_prev_state.hu_vert[vert_face_idx(i, j)]) +
                   (x_prev_state.hv_horiz[horiz_face_idx(i, j+1)] - x_prev_state.hv_horiz[horiz_face_idx(i, j)])
        div_curr = (x_curr_state.hu_vert[vert_face_idx(i+1, j)] - x_curr_state.hu_vert[vert_face_idx(i, j)]) +
                   (x_curr_state.hv_horiz[horiz_face_idx(i, j+1)] - x_curr_state.hv_horiz[horiz_face_idx(i, j)])
        dh_dt + 0.5 * (div_prev + div_curr)
    end

    # x-momentum conservation, well-balanced form.
    # Use face-midpoint flux/source values (not face integrals) so the bathymetry
    # source exactly cancels the pressure flux at lake-at-rest equilibrium.
    xmom_res = map(1:N_cells) do k
        i, j = cell_idx_to_ij(k)
        dhu_dt = 0.5 * (x_prev_state.dhu_int_dt[k] + x_curr_state.dhu_int_dt[k])
        # Use midpoint values (consistent with flux constraint P = ½gh²)
        Kx_P_E_prev = x_prev_state.Kx_vert_face[vert_face_idx(i+1, j)] + x_prev_state.P_vert_face[vert_face_idx(i+1, j)]
        Kx_P_W_prev = x_prev_state.Kx_vert_face[vert_face_idx(i, j)] + x_prev_state.P_vert_face[vert_face_idx(i, j)]
        C_N_prev = x_prev_state.C_horiz_face[horiz_face_idx(i, j+1)]
        C_S_prev = x_prev_state.C_horiz_face[horiz_face_idx(i, j)]
        Kx_P_E_curr = x_curr_state.Kx_vert_face[vert_face_idx(i+1, j)] + x_curr_state.P_vert_face[vert_face_idx(i+1, j)]
        Kx_P_W_curr = x_curr_state.Kx_vert_face[vert_face_idx(i, j)] + x_curr_state.P_vert_face[vert_face_idx(i, j)]
        C_N_curr = x_curr_state.C_horiz_face[horiz_face_idx(i, j+1)]
        C_S_curr = x_curr_state.C_horiz_face[horiz_face_idx(i, j)]
        # Fluxes are now per-unit-length (midpoint values), so no Δy/Δx factors needed
        div_prev = (Kx_P_E_prev - Kx_P_W_prev) + (C_N_prev - C_S_prev)
        div_curr = (Kx_P_E_curr - Kx_P_W_curr) + (C_N_curr - C_S_curr)
        # Well-balanced source: use bathymetry DIFFERENCE and face h values
        # Source is also per-unit-length to match the flux terms
        H0_E = p.H0_vert_face[i+1, j]
        H0_W = p.H0_vert_face[i, j]
        b_diff_x = -(H0_E - H0_W)  # b_E - b_W (since b = -H0)
        h_E_prev = x_prev_state.h_vert_face[vert_face_idx(i+1, j)]
        h_W_prev = x_prev_state.h_vert_face[vert_face_idx(i, j)]
        h_E_curr = x_curr_state.h_vert_face[vert_face_idx(i+1, j)]
        h_W_curr = x_curr_state.h_vert_face[vert_face_idx(i, j)]
        h_face_avg = 0.25 * (h_E_prev + h_W_prev + h_E_curr + h_W_curr)
        source = p.g * b_diff_x * h_face_avg  # No Δy - using per-unit-length formulation
        dhu_dt + 0.5*(div_prev + div_curr) + source
    end

    # y-momentum conservation - same well-balanced approach with midpoint values
    ymom_res = map(1:N_cells) do k
        i, j = cell_idx_to_ij(k)
        dhv_dt = 0.5 * (x_prev_state.dhv_int_dt[k] + x_curr_state.dhv_int_dt[k])
        # Use midpoint values
        C_E_prev = x_prev_state.C_vert_face[vert_face_idx(i+1, j)]
        C_W_prev = x_prev_state.C_vert_face[vert_face_idx(i, j)]
        Ky_P_N_prev = x_prev_state.Ky_horiz_face[horiz_face_idx(i, j+1)] + x_prev_state.P_horiz_face[horiz_face_idx(i, j+1)]
        Ky_P_S_prev = x_prev_state.Ky_horiz_face[horiz_face_idx(i, j)] + x_prev_state.P_horiz_face[horiz_face_idx(i, j)]
        C_E_curr = x_curr_state.C_vert_face[vert_face_idx(i+1, j)]
        C_W_curr = x_curr_state.C_vert_face[vert_face_idx(i, j)]
        Ky_P_N_curr = x_curr_state.Ky_horiz_face[horiz_face_idx(i, j+1)] + x_curr_state.P_horiz_face[horiz_face_idx(i, j+1)]
        Ky_P_S_curr = x_curr_state.Ky_horiz_face[horiz_face_idx(i, j)] + x_curr_state.P_horiz_face[horiz_face_idx(i, j)]
        div_prev = (C_E_prev - C_W_prev) + (Ky_P_N_prev - Ky_P_S_prev)
        div_curr = (C_E_curr - C_W_curr) + (Ky_P_N_curr - Ky_P_S_curr)
        # Well-balanced source with midpoint values
        H0_N = p.H0_horiz_face[i, j+1]
        H0_S = p.H0_horiz_face[i, j]
        b_diff_y = -(H0_N - H0_S)  # b_N - b_S (since b = -H0)
        h_N_prev = x_prev_state.h_horiz_face[horiz_face_idx(i, j+1)]
        h_S_prev = x_prev_state.h_horiz_face[horiz_face_idx(i, j)]
        h_N_curr = x_curr_state.h_horiz_face[horiz_face_idx(i, j+1)]
        h_S_curr = x_curr_state.h_horiz_face[horiz_face_idx(i, j)]
        h_face_avg = 0.25 * (h_N_prev + h_S_prev + h_N_curr + h_S_curr)
        source = p.g * b_diff_y * h_face_avg  # No Δx - using per-unit-length formulation
        dhv_dt + 0.5*(div_prev + div_curr) + source
    end

    # Flux constraints (only on x_t)
    flux_res = flux_residual_single(x_joint[N_state+1:2N_state], full_state_layout, p.g)

    return vcat(mass_res, xmom_res, ymom_res, flux_res)
end

# PRE-BUILD the NonlinearLeastSquaresModel ONCE (expensive Jacobian tracing)
println("\nPre-building constraint model (sparse Jacobian tracing)...")
n_fvm = 3 * ginfo.N_cells
n_total_constraints = n_fvm + n_flux_single

combined_model = NonlinearLeastSquaresModel(combined_residual, 2*N_state)
lik_combined = combined_model(zeros(n_total_constraints); σ=σ_fvm)
println("Constraint model built: $(n_total_constraints) constraints")

# Constraint builder just returns the pre-built likelihood
function build_fvm_constraint(μ_prev, t)
    return lik_combined
end

# Boundary condition applier - standard approach (momentum only)
function apply_bc_momentum_only!(gmrf, t)
    # Reflecting walls: hu=0 at x-boundaries, hv=0 at y-boundaries
    # NO constraints on h - relies on GP prior coupling
    N_x_grid = ginfo.N_x
    n_cells_y = ginfo.n_cells_y

    left_vert = [1 + (j-1)*N_x_grid for j in 1:n_cells_y]
    right_vert = [N_x_grid + (j-1)*N_x_grid for j in 1:n_cells_y]
    hu_bc_spatial = vcat(left_vert, right_vert)

    n_cells_x = ginfo.n_cells_x
    N_y_grid = ginfo.N_y

    bottom_horiz = [i for i in 1:n_cells_x]
    top_horiz = [i + (N_y_grid-1)*n_cells_x for i in 1:n_cells_x]
    hv_bc_spatial = vcat(bottom_horiz, top_horiz)

    hu_vert_bc = N_state .+ indices(full_state_layout, :hu_vert)[hu_bc_spatial]
    hu_vert_face_bc = N_state .+ indices(full_state_layout, :hu_vert_face)[hu_bc_spatial]
    hv_horiz_bc = N_state .+ indices(full_state_layout, :hv_horiz)[hv_bc_spatial]
    hv_horiz_face_bc = N_state .+ indices(full_state_layout, :hv_horiz_face)[hv_bc_spatial]

    all_bc = vcat(hu_vert_bc, hu_vert_face_bc, hv_horiz_bc, hv_horiz_face_bc)
    n_bc = length(all_bc)

    A_bc = sparse(1:n_bc, all_bc, ones(n_bc), n_bc, 2*N_state)
    y_bc = zeros(n_bc)
    Q_ϵ_bc = (1.0 / 1e-6^2) * sparse(I, n_bc, n_bc)

    return linear_condition(gmrf; A=A_bc, Q_ϵ=Q_ϵ_bc, y=y_bc)
end

# Ghost-cell-style boundary conditions
# Zero gradient for PERTURBATION η = h - H₀, not total depth h!
#   - η_boundary = η_interior
#   - h_boundary = h_interior + (H₀_boundary - H₀_interior)
#   - hu/hv normal = 0, tangential = zero gradient (free-slip)
#
# If `open_right=true`, the right boundary becomes an outflow condition:
#   - hu (normal) uses zero-gradient instead of hu=0
#   - This allows water to flow out freely
function apply_bc_ghost_cell!(gmrf, t; open_right::Bool=false)
    N_x_grid = ginfo.N_x
    N_y_grid = ginfo.N_y
    n_cells_x = ginfo.n_cells_x
    n_cells_y = ginfo.n_cells_y

    # === Momentum BCs ===
    # Normal velocity = 0: hu=0 at x-walls (left, and right unless open), hv=0 at y-walls
    # Tangential velocity = zero gradient (free-slip wall)

    left_vert = [1 + (j-1)*N_x_grid for j in 1:n_cells_y]
    right_vert = [N_x_grid + (j-1)*N_x_grid for j in 1:n_cells_y]

    # If open_right, only left wall gets hu=0; right wall gets zero-gradient (outflow)
    hu_bc_spatial = open_right ? left_vert : vcat(left_vert, right_vert)

    bottom_horiz = [i for i in 1:n_cells_x]
    top_horiz = [i + (N_y_grid-1)*n_cells_x for i in 1:n_cells_x]
    hv_bc_spatial = vcat(bottom_horiz, top_horiz)

    # Normal velocity = 0 (for closed walls)
    hu_vert_bc = N_state .+ indices(full_state_layout, :hu_vert)[hu_bc_spatial]
    hu_vert_face_bc = N_state .+ indices(full_state_layout, :hu_vert_face)[hu_bc_spatial]
    hv_horiz_bc = N_state .+ indices(full_state_layout, :hv_horiz)[hv_bc_spatial]
    hv_horiz_face_bc = N_state .+ indices(full_state_layout, :hv_horiz_face)[hv_bc_spatial]

    momentum_bc = vcat(hu_vert_bc, hu_vert_face_bc, hv_horiz_bc, hv_horiz_face_bc)
    n_momentum = length(momentum_bc)

    # Tangential velocity indices for zero-gradient constraints
    hv_vert_idx = indices(full_state_layout, :hv_vert)
    hv_vert_face_idx = indices(full_state_layout, :hv_vert_face)
    hu_horiz_idx = indices(full_state_layout, :hu_horiz)
    hu_horiz_face_idx = indices(full_state_layout, :hu_horiz_face)

    # === Zero-gradient η constraints (η = h - H₀) ===
    # Constraint: h_bnd - h_int = H₀_bnd - H₀_int (so η_bnd = η_int)
    # This accounts for the sloping bathymetry!

    h_idx = indices(full_state_layout, :h)
    h_vert_face_idx = indices(full_state_layout, :h_vert_face)
    h_horiz_face_idx = indices(full_state_layout, :h_horiz_face)

    # Use actual bathymetry at grid points (H0_arrays.H0_grid computed earlier)
    # This correctly handles Matern perturbation if present

    ghost_I = Int[]
    ghost_J = Int[]
    ghost_V = Float64[]
    ghost_y = Float64[]  # RHS values (non-zero for h constraints)
    row = 0

    # --- Vertical face constraints (left/right boundaries) ---
    # Vertical faces are at (x[i], midpoint of y[j] to y[j+1]) for i=1:N_x, j=1:n_cells_y
    # Use H0_vert_face which is computed at these locations
    # Left (i=1): h_vert_face[1,j] - h_vert_face[2,j] = H₀_vf[1,j] - H₀_vf[2,j]
    for j in 1:n_cells_y
        row += 1
        bnd = 1 + (j-1)*N_x_grid
        int = 2 + (j-1)*N_x_grid
        H0_diff = H0_arrays.H0_vert_face[1, j] - H0_arrays.H0_vert_face[2, j]
        push!(ghost_I, row); push!(ghost_J, N_state + h_vert_face_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + h_vert_face_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, H0_diff)
    end

    # Right (i=N_x): h_vert_face[N_x,j] - h_vert_face[N_x-1,j] = H₀_vf[N_x,j] - H₀_vf[N_x-1,j]
    for j in 1:n_cells_y
        row += 1
        bnd = N_x_grid + (j-1)*N_x_grid
        int = (N_x_grid-1) + (j-1)*N_x_grid
        H0_diff = H0_arrays.H0_vert_face[N_x_grid, j] - H0_arrays.H0_vert_face[N_x_grid-1, j]
        push!(ghost_I, row); push!(ghost_J, N_state + h_vert_face_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + h_vert_face_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, H0_diff)
    end

    # --- Horizontal face constraints (bottom/top boundaries) ---
    # Horizontal faces are at (midpoint of x[i] to x[i+1], y[j]) for i=1:n_cells_x, j=1:N_y
    # Use H0_horiz_face which is computed at these locations
    # Bottom (j=1): h_horiz_face[i,1] - h_horiz_face[i,2] = H₀_hf[i,1] - H₀_hf[i,2]
    for i in 1:n_cells_x
        row += 1
        bnd = i
        int = i + n_cells_x
        H0_diff = H0_arrays.H0_horiz_face[i, 1] - H0_arrays.H0_horiz_face[i, 2]
        push!(ghost_I, row); push!(ghost_J, N_state + h_horiz_face_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + h_horiz_face_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, H0_diff)
    end

    # Top (j=N_y)
    for i in 1:n_cells_x
        row += 1
        bnd = i + (N_y_grid-1)*n_cells_x
        int = i + (N_y_grid-2)*n_cells_x
        H0_diff = H0_arrays.H0_horiz_face[i, N_y_grid] - H0_arrays.H0_horiz_face[i, N_y_grid-1]
        push!(ghost_I, row); push!(ghost_J, N_state + h_horiz_face_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + h_horiz_face_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, H0_diff)
    end

    # --- Grid point h constraints ---
    # Left (i=1): h[1,j] - h[2,j] = H₀[1,j] - H₀[2,j]
    for j in 1:N_y_grid
        row += 1
        bnd = 1 + (j-1)*N_x_grid
        int = 2 + (j-1)*N_x_grid
        H0_diff = H0_arrays.H0_grid[1, j] - H0_arrays.H0_grid[2, j]
        push!(ghost_I, row); push!(ghost_J, N_state + h_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + h_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, H0_diff)
    end

    # Right (i=N_x): h[N_x,j] - h[N_x-1,j] = H₀[N_x,j] - H₀[N_x-1,j]
    for j in 1:N_y_grid
        row += 1
        bnd = N_x_grid + (j-1)*N_x_grid
        int = (N_x_grid-1) + (j-1)*N_x_grid
        H0_diff = H0_arrays.H0_grid[N_x_grid, j] - H0_arrays.H0_grid[N_x_grid-1, j]
        push!(ghost_I, row); push!(ghost_J, N_state + h_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + h_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, H0_diff)
    end

    # Bottom (j=1): h[i,1] - h[i,2] = H₀[i,1] - H₀[i,2]
    for i in 2:(N_x_grid-1)
        row += 1
        bnd = i
        int = i + N_x_grid
        H0_diff = H0_arrays.H0_grid[i, 1] - H0_arrays.H0_grid[i, 2]
        push!(ghost_I, row); push!(ghost_J, N_state + h_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + h_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, H0_diff)
    end

    # Top (j=N_y): h[i,N_y] - h[i,N_y-1] = H₀[i,N_y] - H₀[i,N_y-1]
    for i in 2:(N_x_grid-1)
        row += 1
        bnd = i + (N_y_grid-1)*N_x_grid
        int = i + (N_y_grid-2)*N_x_grid
        H0_diff = H0_arrays.H0_grid[i, N_y_grid] - H0_arrays.H0_grid[i, N_y_grid-1]
        push!(ghost_I, row); push!(ghost_J, N_state + h_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + h_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, H0_diff)
    end

    # --- Tangential velocity zero-gradient (free-slip) ---
    # At x-walls: hv (tangential) has zero gradient
    # Left (i=1): hv_vert[1,j] = hv_vert[2,j], hv_vert_face[1,j] = hv_vert_face[2,j]
    for j in 1:n_cells_y
        row += 1
        bnd = 1 + (j-1)*N_x_grid
        int = 2 + (j-1)*N_x_grid
        push!(ghost_I, row); push!(ghost_J, N_state + hv_vert_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + hv_vert_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, 0.0)
    end
    for j in 1:n_cells_y
        row += 1
        bnd = 1 + (j-1)*N_x_grid
        int = 2 + (j-1)*N_x_grid
        push!(ghost_I, row); push!(ghost_J, N_state + hv_vert_face_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + hv_vert_face_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, 0.0)
    end

    # Right (i=N_x): hv_vert[N_x,j] = hv_vert[N_x-1,j]
    for j in 1:n_cells_y
        row += 1
        bnd = N_x_grid + (j-1)*N_x_grid
        int = (N_x_grid-1) + (j-1)*N_x_grid
        push!(ghost_I, row); push!(ghost_J, N_state + hv_vert_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + hv_vert_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, 0.0)
    end
    for j in 1:n_cells_y
        row += 1
        bnd = N_x_grid + (j-1)*N_x_grid
        int = (N_x_grid-1) + (j-1)*N_x_grid
        push!(ghost_I, row); push!(ghost_J, N_state + hv_vert_face_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + hv_vert_face_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, 0.0)
    end

    # --- Open boundary at right: hu (normal) zero-gradient for outflow ---
    # Only apply if open_right=true (replaces hu=0 condition)
    hu_vert_idx = indices(full_state_layout, :hu_vert)
    hu_vert_face_idx = indices(full_state_layout, :hu_vert_face)

    if open_right
        # hu_vert[N_x, j] = hu_vert[N_x-1, j] (zero gradient for normal velocity)
        for j in 1:n_cells_y
            row += 1
            bnd = N_x_grid + (j-1)*N_x_grid
            int = (N_x_grid-1) + (j-1)*N_x_grid
            push!(ghost_I, row); push!(ghost_J, N_state + hu_vert_idx[bnd]); push!(ghost_V, 1.0)
            push!(ghost_I, row); push!(ghost_J, N_state + hu_vert_idx[int]); push!(ghost_V, -1.0)
            push!(ghost_y, 0.0)
        end
        for j in 1:n_cells_y
            row += 1
            bnd = N_x_grid + (j-1)*N_x_grid
            int = (N_x_grid-1) + (j-1)*N_x_grid
            push!(ghost_I, row); push!(ghost_J, N_state + hu_vert_face_idx[bnd]); push!(ghost_V, 1.0)
            push!(ghost_I, row); push!(ghost_J, N_state + hu_vert_face_idx[int]); push!(ghost_V, -1.0)
            push!(ghost_y, 0.0)
        end
    end

    # At y-walls: hu (tangential) has zero gradient
    # Bottom (j=1): hu_horiz[i,1] = hu_horiz[i,2]
    for i in 1:n_cells_x
        row += 1
        bnd = i
        int = i + n_cells_x
        push!(ghost_I, row); push!(ghost_J, N_state + hu_horiz_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + hu_horiz_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, 0.0)
    end
    for i in 1:n_cells_x
        row += 1
        bnd = i
        int = i + n_cells_x
        push!(ghost_I, row); push!(ghost_J, N_state + hu_horiz_face_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + hu_horiz_face_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, 0.0)
    end

    # Top (j=N_y): hu_horiz[i,N_y] = hu_horiz[i,N_y-1]
    for i in 1:n_cells_x
        row += 1
        bnd = i + (N_y_grid-1)*n_cells_x
        int = i + (N_y_grid-2)*n_cells_x
        push!(ghost_I, row); push!(ghost_J, N_state + hu_horiz_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + hu_horiz_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, 0.0)
    end
    for i in 1:n_cells_x
        row += 1
        bnd = i + (N_y_grid-1)*n_cells_x
        int = i + (N_y_grid-2)*n_cells_x
        push!(ghost_I, row); push!(ghost_J, N_state + hu_horiz_face_idx[bnd]); push!(ghost_V, 1.0)
        push!(ghost_I, row); push!(ghost_J, N_state + hu_horiz_face_idx[int]); push!(ghost_V, -1.0)
        push!(ghost_y, 0.0)
    end

    n_ghost = row

    # Build constraint matrices
    A_momentum = sparse(1:n_momentum, momentum_bc, ones(n_momentum), n_momentum, 2*N_state)
    A_ghost = sparse(ghost_I, ghost_J, ghost_V, n_ghost, 2*N_state)

    A_bc = vcat(A_momentum, A_ghost)
    # RHS: zeros for momentum, ghost_y for ghost cell constraints (contains bathymetry offsets)
    y_bc = vcat(zeros(n_momentum), ghost_y)
    Q_ϵ_bc = (1.0 / 1e-6^2) * sparse(I, n_momentum + n_ghost, n_momentum + n_ghost)

    return linear_condition(gmrf; A=A_bc, Q_ϵ=Q_ϵ_bc, y=y_bc)
end

apply_bc! = (gmrf, t) -> apply_bc_ghost_cell!(gmrf, t; open_right=USE_OPEN_RIGHT_BC)

# ==============================================================================
# Run EKF Solver
# ==============================================================================

println("\n--- Running EKF Sequential Solver ---")

solution = solve_ekf_sequential(
    Q_spatial_full,
    full_state_layout,
    A_kr,
    Q_noise_sparse,
    n_timesteps,
    x_t0,
    build_fvm_constraint;
    apply_bc! = apply_bc!,
    store_history = true,
    verbose = true,
    max_iter = 1  # Single GN step = true EKF
)

println("\nEKF solve complete!")

# ==============================================================================
# Extract Results
# ==============================================================================

means = mean_trajectory(solution)
vars = var_trajectory(solution)

# ==============================================================================
# Save Results
# ==============================================================================

if save_results
    mkpath(output_dir)

    # Use the actual bathymetry (H0_arrays.H0_grid already computed, includes Matern perturbation)
    bathy_grid = H0_arrays.H0_grid

    # Save everything needed for plotting
    results_file = joinpath(output_dir, "swe_results.jld2")
    @save results_file means vars xs ys bathy_grid p Δt_solve n_timesteps N_x N_y full_state_layout
    println("\nSaved results to $(results_file)")
end

# ==============================================================================
# Diagnostics
# ==============================================================================

n_actual = length(means)  # Actual number of timesteps completed

# Track boundary η over time
h_idx = indices(full_state_layout, :h)
H0_left = p.H₀_shallow * 1000   # meters
H0_right = p.H₀_deep * 1000     # meters

# ==============================================================================
# Visualization
# ==============================================================================

println("\nCreating visualization...")

# Compute equilibrium depth H₀(x) at grid points (for plotting perturbation)
# H₀(x) = H_shallow + (H_deep - H_shallow) * x / L_x
# Use actual bathymetry (H0_arrays.H0_grid) for computing perturbation η
H0_grid = H0_arrays.H0_grid

# Pick timesteps for snapshots (adaptive to actual length)
t_indices = unique([1, n_actual ÷ 3, 2 * n_actual ÷ 3, n_actual])
t_indices = filter(t -> 1 <= t <= n_actual, t_indices)

fig = Figure(size=(400 * length(t_indices), 800))

for (col, t_idx) in enumerate(t_indices)
    t = (t_idx - 1) * Δt_solve

    local h_idx = indices(full_state_layout, :h)
    h_vals = means[t_idx][h_idx]
    h_std = sqrt.(vars[t_idx][h_idx])

    local h_2d = reshape(h_vals, N_x, N_y)
    std_2d = reshape(h_std, N_x, N_y)

    # Compute perturbation from equilibrium: η = h - H₀
    # This shows just the wave, not the bathymetry
    eta_2d = h_2d .- H0_grid

    # Surface perturbation (convert to meters)
    ax1 = Axis(fig[1, col],
        xlabel = col == 1 ? "x (km)" : "",
        ylabel = col == 1 ? "y (km)" : "",
        title = "η (perturbation) at t=$(round(t, digits=1)) min",
        aspect = DataAspect()
    )
    # Use diverging colormap centered at 0
    max_eta = max(abs(minimum(eta_2d)), abs(maximum(eta_2d))) * 1000
    hm1 = heatmap!(ax1, collect(xs), collect(ys), eta_2d .* 1000,
        colormap=:RdBu, colorrange=(-max_eta, max_eta))
    if col == length(t_indices)
        Colorbar(fig[1, col+1], hm1, label="η (m)")
    end

    # Uncertainty
    ax2 = Axis(fig[2, col],
        xlabel = "x (km)",
        ylabel = col == 1 ? "y (km)" : "",
        title = "σ(h) at t=$(round(t, digits=1)) min",
        aspect = DataAspect()
    )
    hm2 = heatmap!(ax2, collect(xs), collect(ys), std_2d .* 1000, colormap=:magma)
    if col == length(t_indices)
        Colorbar(fig[2, col+1], hm2, label="σ(h) (m)")
    end
end

Label(fig[0, :], "EKF-Style GP-FVM: Nonlinear Shallow Water (η = h - H₀)", fontsize=18)

save("ekf_swe_solution.png", fig, px_per_unit=2)
println("Saved to ekf_swe_solution.png")

# ==============================================================================
# 3D Animation of Wave Evolution
# ==============================================================================

println("\nCreating 3D animation...")

# Compute eta for all timesteps
all_eta = Vector{Matrix{Float64}}(undef, n_timesteps)
for t_idx in 1:n_actual
    local h_vals = means[t_idx][h_idx]
    local h_2d = reshape(h_vals, N_x, N_y)
    all_eta[t_idx] = (h_2d .- H0_grid) .* 1000  # Convert to meters
end

# Fixed z-axis range to focus on the initial bump (~10m)
# Corner drift may go out of view, but we can see the wave dynamics better
eta_range = 20.0  # meters

# Create animation
fig_anim = Figure(size=(800, 600))

ax3d = Axis3(fig_anim[1, 1],
    xlabel = "x (km)",
    ylabel = "y (km)",
    zlabel = "η (m)",
    title = "Surface Perturbation η = h - H₀",
    aspect = (1, 1, 0.5),
    elevation = 30π/180,
    azimuth = -60π/180
)

# Set consistent z limits
zlims!(ax3d, -eta_range * 1.1, eta_range * 1.1)

# Create the surface plot (will be updated in animation)
xs_vec = collect(xs)
ys_vec = collect(ys)
surf = surface!(ax3d, xs_vec, ys_vec, all_eta[1],
    colormap = :RdBu,
    colorrange = (-eta_range, eta_range))
Colorbar(fig_anim[1, 2], surf, label = "η (m)")

# Time label
time_label = Label(fig_anim[0, :], "t = 0.0 min", fontsize = 18)

# Record animation
framerate = 4
record(fig_anim, "ekf_swe_evolution.mp4", 1:n_actual; framerate=framerate) do t_idx
    t = (t_idx - 1) * Δt_solve
    time_label.text[] = "t = $(round(t, digits=1)) min"
    surf[3] = all_eta[t_idx]
end

println("Saved 3D animation to ekf_swe_evolution.mp4")

fig
