"""
GP-based bathymetry sampling for nonlinear SWE.

Samples bathymetry H₀ and its gradients from a Matérn GP conditioned on
a linear slope. Returns consistent values at all needed locations.
"""

using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using Random

import GaussianMarkovRandomFields: precision_matrix, linear_condition
import FunctionalGPs: ⊗

"""
    BathymetrySample

Holds sampled bathymetry values at all required locations.
"""
struct BathymetrySample
    H0_grid::Matrix{Float64}        # N_x × N_y at grid points
    H0_cell::Matrix{Float64}        # n_cells_x × n_cells_y at cell centers
    H0_vert_face::Matrix{Float64}   # N_x × n_cells_y at vertical face midpoints
    H0_horiz_face::Matrix{Float64}  # n_cells_x × N_y at horizontal face midpoints
    dH0_dx::Matrix{Float64}         # n_cells_x × n_cells_y derivatives
    dH0_dy::Matrix{Float64}         # n_cells_x × n_cells_y derivatives
    db_dx::Vector{Float64}          # flattened -dH0_dx for SolveParams
    db_dy::Vector{Float64}          # flattened -dH0_dy for SolveParams
    depth_at_cells::Vector{Float64} # flattened H0_cell for SolveParams
end

"""
    sample_bathymetry_gp(xs, ys, H_shallow, H_deep, L_x;
                         lengthscale=15.0, σ_obs=0.03, ρ=2.0, seed=42)

Sample bathymetry from a Matérn GP conditioned on linear slope.

# Arguments
- `xs`, `ys`: Grid coordinates (vectors)
- `H_shallow`, `H_deep`: Depth at x=0 and x=L_x
- `L_x`: Domain length in x

# Keyword Arguments
- `lengthscale`: GP lengthscale (same units as xs, ys)
- `σ_obs`: Observation noise for conditioning (controls perturbation magnitude)
- `ρ`: Sparsity parameter for sparse Cholesky
- `seed`: Random seed for reproducibility

# Returns
- `BathymetrySample` with all bathymetry values and gradients
"""
function sample_bathymetry_gp(xs, ys, H_shallow, H_deep, L_x;
                               lengthscale=15.0, σ_obs=0.03, ρ=2.0, seed=42,
                               smoothness=2, verbose=false)
    xs = collect(xs)
    ys = collect(ys)

    N_x = length(xs)
    N_y = length(ys)
    n_cells_x = N_x - 1
    n_cells_y = N_y - 1

    # Cell center coordinates
    cell_xs = [(xs[i] + xs[i+1])/2 for i in 1:n_cells_x]
    cell_ys = [(ys[j] + ys[j+1])/2 for j in 1:n_cells_y]

    verbose && println("Building bathymetry GP...")

    # Build 2D grids
    grid_2d = FactorizedGrid(xs, ys)
    cell_grid_2d = FactorizedGrid(cell_xs, cell_ys)
    vert_face_grid_2d = FactorizedGrid(xs, cell_ys)
    horiz_face_grid_2d = FactorizedGrid(cell_xs, ys)

    # 2D tensor product kernel
    k_1d = HalfIntegerMaternKernel(smoothness, [lengthscale])
    k_2d = k_1d ⊗ k_1d

    # Functionals
    functionals = [
        :H0_grid => EvaluationFunctional(grid_2d),
        :H0_cell => EvaluationFunctional(cell_grid_2d),
        :H0_vert_face => EvaluationFunctional(vert_face_grid_2d),
        :H0_horiz_face => EvaluationFunctional(horiz_face_grid_2d),
        :dH0_dx => EvaluationFunctional(cell_grid_2d) ∘ PartialDerivative((1, 0)),
        :dH0_dy => EvaluationFunctional(cell_grid_2d) ∘ PartialDerivative((0, 1)),
    ]

    # Build sparse GMRF
    x0, bathy_layout, _ = sparse_gmrf(functionals, k_2d; ρ=ρ, ordering=:integrals_coarsest)

    verbose && println("  State dim: $(length(bathy_layout)), fill: $(round(100*nnz(precision_matrix(x0))/length(bathy_layout)^2, digits=1))%")

    # Condition on linear slope at grid points
    H0_linear_grid = [H_shallow + (H_deep - H_shallow) * xs[i] / L_x for j in 1:N_y for i in 1:N_x]

    obs_idx = indices(bathy_layout, :H0_grid)
    n_obs = length(obs_idx)
    A_obs = sparse(1:n_obs, obs_idx, ones(n_obs), n_obs, length(bathy_layout))
    Q_obs = (1.0 / σ_obs^2) * sparse(I, n_obs, n_obs)

    x_post = linear_condition(x0; A=A_obs, Q_ϵ=Q_obs, y=H0_linear_grid)

    # Sample
    Random.seed!(seed)
    sample = rand(x_post)

    # Extract and reshape
    H0_grid = reshape(sample[indices(bathy_layout, :H0_grid)], N_x, N_y)
    H0_cell = reshape(sample[indices(bathy_layout, :H0_cell)], n_cells_x, n_cells_y)
    H0_vert_face = reshape(sample[indices(bathy_layout, :H0_vert_face)], N_x, n_cells_y)
    H0_horiz_face = reshape(sample[indices(bathy_layout, :H0_horiz_face)], n_cells_x, N_y)
    dH0_dx = reshape(sample[indices(bathy_layout, :dH0_dx)], n_cells_x, n_cells_y)
    dH0_dy = reshape(sample[indices(bathy_layout, :dH0_dy)], n_cells_x, n_cells_y)

    # For SWE: db/dx = -dH0/dx (bottom elevation b = -H₀)
    db_dx = vec(-dH0_dx)
    db_dy = vec(-dH0_dy)
    depth_at_cells = vec(H0_cell)

    verbose && println("  H0 range: $(round(minimum(H0_grid)*1000, digits=1)) to $(round(maximum(H0_grid)*1000, digits=1)) m")

    return BathymetrySample(
        H0_grid, H0_cell, H0_vert_face, H0_horiz_face,
        dH0_dx, dH0_dy, db_dx, db_dy, depth_at_cells
    )
end

"""
Convert BathymetrySample to the H0_arrays NamedTuple format used by IC/BC.
"""
function to_H0_arrays(bathy::BathymetrySample)
    return (
        H0_grid = bathy.H0_grid,
        H0_cell = bathy.H0_cell,
        H0_vert_face = bathy.H0_vert_face,
        H0_horiz_face = bathy.H0_horiz_face,
    )
end
