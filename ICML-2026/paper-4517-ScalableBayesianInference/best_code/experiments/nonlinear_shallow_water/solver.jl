"""
Solver integration: EKF-style initialization and batch solve for nonlinear SWE.
"""

using GPFiniteVolume
using GaussianMarkovRandomFields
using Kronecker: kronecker
using SparseArrays, LinearAlgebra

import GaussianMarkovRandomFields: mean, precision_matrix, linear_condition

"""
    build_ic_observations(p, funcs, ginfo, spatial_layout; H0_arrays=nothing)

Build observation matrix and values for initial conditions.
Prescribes cell integrals AND face point values for consistency.
Returns (A_ic, y_ic) for linear_condition call.

If H0_arrays is provided, it should be a NamedTuple with:
  - H0_grid: N_x × N_y array of H₀ at grid points
  - H0_vert_face: N_x × n_cells_y array of H₀ at vertical face midpoints
  - H0_horiz_face: n_cells_x × N_y array of H₀ at horizontal face midpoints
  - H0_cell: n_cells_x × n_cells_y array of H₀ at cell centers
"""
function build_ic_observations(p, funcs, ginfo, spatial_layout; H0_arrays=nothing)
    (; N_cells, N_x, N_y, Δx, Δy, n_vert_faces, n_horiz_faces) = ginfo

    n_state = length(spatial_layout)
    xs, ys = funcs.grid.ranges

    # Collect all IC indices and values
    all_indices = Int[]
    all_values = Float64[]

    # Cell area
    cell_area = Δx * Δy

    # Helper function for computing h at any point
    # Uses precomputed H0 arrays if provided, otherwise linear formula
    function h_at_point(x, y)
        # Default: linear slope
        depth = p.H₀_shallow + (p.H₀_deep - p.H₀_shallow) * x / p.L_x
        # Add bump
        dx = (x - p.h_center_x) / p.h_width_x
        dy = (y - p.h_center_y) / p.h_width_y
        bump = p.h_amplitude * exp(-0.5 * (dx^2 + dy^2))
        return depth + bump
    end

    function bump_at_point(x, y)
        dx = (x - p.h_center_x) / p.h_width_x
        dy = (y - p.h_center_y) / p.h_width_y
        return p.h_amplitude * exp(-0.5 * (dx^2 + dy^2))
    end

    # Primary state: h at grid points (corners)
    append!(all_indices, indices(spatial_layout, :h))
    for j in 1:N_y
        for i in 1:N_x
            if H0_arrays !== nothing
                h_val = H0_arrays.H0_grid[i, j] + bump_at_point(xs[i], ys[j])
            else
                h_val = h_at_point(xs[i], ys[j])
            end
            push!(all_values, h_val)
        end
    end

    # Primary state: h_int (cell integrals)
    append!(all_indices, indices(spatial_layout, :h_int))
    for (k, (i, j)) in enumerate([(i, j) for j in 1:ginfo.n_cells_y for i in 1:ginfo.n_cells_x])
        x_c = 0.5 * (xs[i] + xs[i+1])
        y_c = 0.5 * (ys[j] + ys[j+1])
        if H0_arrays !== nothing
            h_val = H0_arrays.H0_cell[i, j] + bump_at_point(x_c, y_c)
        else
            h_val = h_at_point(x_c, y_c)
        end
        push!(all_values, h_val * cell_area)
    end

    # h at vertical face midpoints
    append!(all_indices, indices(spatial_layout, :h_vert_face))
    for j in 1:ginfo.n_cells_y
        for i in 1:N_x
            x_f = xs[i]
            y_f = 0.5 * (ys[j] + ys[j+1])
            if H0_arrays !== nothing
                h_val = H0_arrays.H0_vert_face[i, j] + bump_at_point(x_f, y_f)
            else
                h_val = h_at_point(x_f, y_f)
            end
            push!(all_values, h_val)
        end
    end

    # h at horizontal face midpoints
    append!(all_indices, indices(spatial_layout, :h_horiz_face))
    for j in 1:N_y
        for i in 1:ginfo.n_cells_x
            x_f = 0.5 * (xs[i] + xs[i+1])
            y_f = ys[j]
            if H0_arrays !== nothing
                h_val = H0_arrays.H0_horiz_face[i, j] + bump_at_point(x_f, y_f)
            else
                h_val = h_at_point(x_f, y_f)
            end
            push!(all_values, h_val)
        end
    end

    # Primary state: hu_int, hv_int (all zeros - no initial velocity)
    append!(all_indices, indices(spatial_layout, :hu_int))
    append!(all_values, zeros(N_cells))

    append!(all_indices, indices(spatial_layout, :hv_int))
    append!(all_values, zeros(N_cells))

    # hu, hv at face midpoints (all zeros)
    append!(all_indices, indices(spatial_layout, :hu_vert_face))
    append!(all_values, zeros(n_vert_faces))
    append!(all_indices, indices(spatial_layout, :hu_horiz_face))
    append!(all_values, zeros(n_horiz_faces))
    append!(all_indices, indices(spatial_layout, :hv_vert_face))
    append!(all_values, zeros(n_vert_faces))
    append!(all_indices, indices(spatial_layout, :hv_horiz_face))
    append!(all_values, zeros(n_horiz_faces))

    # P at face midpoints (P = 0.5 * g * h^2)
    append!(all_indices, indices(spatial_layout, :P_vert_face))
    for j in 1:ginfo.n_cells_y
        for i in 1:N_x
            x_f = xs[i]
            y_f = 0.5 * (ys[j] + ys[j+1])
            if H0_arrays !== nothing
                h = H0_arrays.H0_vert_face[i, j] + bump_at_point(x_f, y_f)
            else
                h = h_at_point(x_f, y_f)
            end
            push!(all_values, 0.5 * p.g * h^2)
        end
    end
    append!(all_indices, indices(spatial_layout, :P_horiz_face))
    for j in 1:N_y
        for i in 1:ginfo.n_cells_x
            x_f = 0.5 * (xs[i] + xs[i+1])
            y_f = ys[j]
            if H0_arrays !== nothing
                h = H0_arrays.H0_horiz_face[i, j] + bump_at_point(x_f, y_f)
            else
                h = h_at_point(x_f, y_f)
            end
            push!(all_values, 0.5 * p.g * h^2)
        end
    end

    # Kx, Ky, C at face midpoints (all zeros since u=v=0)
    append!(all_indices, indices(spatial_layout, :Kx_vert_face))
    append!(all_values, zeros(n_vert_faces))
    append!(all_indices, indices(spatial_layout, :Kx_horiz_face))
    append!(all_values, zeros(n_horiz_faces))
    append!(all_indices, indices(spatial_layout, :Ky_vert_face))
    append!(all_values, zeros(n_vert_faces))
    append!(all_indices, indices(spatial_layout, :Ky_horiz_face))
    append!(all_values, zeros(n_horiz_faces))
    append!(all_indices, indices(spatial_layout, :C_vert_face))
    append!(all_values, zeros(n_vert_faces))
    append!(all_indices, indices(spatial_layout, :C_horiz_face))
    append!(all_values, zeros(n_horiz_faces))

    # Build sparse observation matrix: A[i, all_indices[i]] = 1
    n_obs = length(all_indices)
    A_ic = sparse(1:n_obs, all_indices, ones(n_obs), n_obs, n_state)

    return A_ic, all_values
end

"""
    build_bc_observations(timest, ginfo, n_timesteps)

Build observation matrix and values for boundary conditions.
Reflective walls: hu = 0 at x-boundaries, hv = 0 at y-boundaries.
"""
function build_bc_observations(timest, ginfo, n_timesteps)
    (; N_x, N_y) = ginfo
    n_state = length(timest)

    # Left/right boundaries: hu = 0
    left_idcs = [(j-1)*N_x + 1 for j in 1:N_y]
    right_idcs = [(j-1)*N_x + N_x for j in 1:N_y]
    u_bc_spatial = vcat(left_idcs, right_idcs)

    # Top/bottom boundaries: hv = 0
    bottom_idcs = [i for i in 1:N_x]
    top_idcs = [(N_y-1)*N_x + i for i in 1:N_x]
    v_bc_spatial = vcat(bottom_idcs, top_idcs)

    # Apply at all times t=2:n_timesteps
    hu_bc_idcs = absindices(timest, :hu, u_bc_spatial, 2:n_timesteps)
    hv_bc_idcs = absindices(timest, :hv, v_bc_spatial, 2:n_timesteps)

    all_bc_indices = vcat(hu_bc_idcs, hv_bc_idcs)
    n_bc = length(all_bc_indices)

    A_bc = sparse(1:n_bc, all_bc_indices, ones(n_bc), n_bc, n_state)
    y_bc = zeros(n_bc)

    return A_bc, y_bc
end

"""
    apply_initial_conditions(x0, p, funcs, ginfo, spatial_layout; noise_std=1e-6)

Apply all initial conditions in a single linear_condition call.
"""
function apply_initial_conditions(x0, p, funcs, ginfo, spatial_layout; noise_std=1e-6)
    A_ic, y_ic = build_ic_observations(p, funcs, ginfo, spatial_layout)
    n_obs = size(A_ic, 1)
    Q_ϵ = (1.0 / noise_std^2) * sparse(I, n_obs, n_obs)
    linear_condition(x0; A=A_ic, Q_ϵ=Q_ϵ, y=y_ic)
end

"""
    apply_boundary_conditions(x_joint, timest, ginfo, n_timesteps; noise_std=1e-6)

Apply boundary conditions in a single linear_condition call.
"""
function apply_boundary_conditions(x_joint, timest, ginfo, n_timesteps; noise_std=1e-6)
    A_bc, y_bc = build_bc_observations(timest, ginfo, n_timesteps)
    n_bc = size(A_bc, 1)
    Q_ϵ = (1.0 / noise_std^2) * sparse(I, n_bc, n_bc)
    linear_condition(x_joint; A=A_bc, Q_ϵ=Q_ϵ, y=y_bc)
end

"""
    run_ekf_block_sweep(x_t0, A, Q_noise_sparse, n_timesteps, lik_builder, single_layout;
                         max_iter=1, verbose=false)

Run EKF-style block sweep: 1 Gauss-Newton step per time block.

Returns the concatenated block means as initialization for batch solve.
"""
function run_ekf_block_sweep(x_t0, A, Q_noise_sparse, n_timesteps, lik_builder, single_layout;
                              max_iter=1, verbose=false)
    block_means = Vector{Vector{Float64}}(undef, n_timesteps)
    block_means[1] = mean(x_t0)

    for t_idx in 2:n_timesteps
        verbose && print("  Block $t_idx/$n_timesteps...")

        x_prev = block_means[t_idx - 1]
        μ_t_prior = collect(A * x_prev)
        x_t_prior = GMRF(μ_t_prior, Q_noise_sparse)

        # Build likelihood for this block
        lik_t = lik_builder(t_idx, x_prev)

        # Single GN step
        x_t_solution = gaussian_approximation(x_t_prior, lik_t;
            verbose=false, max_iter=max_iter)

        block_means[t_idx] = mean(x_t_solution)
        verbose && println(" done")
    end

    return vcat(block_means...)
end

"""
    solve_batch(x_init, lik_batch; verbose=true, max_iter=50, newton_dec_tol=1e-5)

Solve the batch problem using Gauss-Newton from the given initialization.
"""
function solve_batch(x_init, lik_batch; verbose=true, max_iter=50, newton_dec_tol=1e-5)
    gaussian_approximation(x_init, lik_batch;
        verbose=verbose, max_iter=max_iter, newton_dec_tol=newton_dec_tol)
end
