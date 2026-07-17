"""
FVM and nonlinear flux constraints for nonlinear SWE.

Two types of constraints:
1. Linear FVM conservation (uses face integrals)
2. Nonlinear flux-state consistency (pointwise at face midpoints)
"""

using SparseArrays
using GaussianMarkovRandomFields

"""
    build_cell_corners(N_x, n_cells_x, n_cells_y)

Build cell corner indices for FVM.
Returns vector of tuples (SW, SE, NW, NE) for each cell.
Grid ordering: x changes fastest, linear index = (j-1)*N_x + i
"""
function build_cell_corners(N_x, n_cells_x, n_cells_y)
    [(
        (j-1)*N_x + i,      # SW
        (j-1)*N_x + (i+1),  # SE
        j*N_x + i,          # NW
        j*N_x + (i+1)       # NE
    ) for j in 1:n_cells_y for i in 1:n_cells_x]
end

"""
    build_fvm_constraint_matrix(timest, ginfo, p, time_range; crank_nicolson=true)

Build the FVM constraint matrix for nonlinear SWE with bathymetry.

Conservation laws (all in FVM integral form):
- Mass: d/dt ∫h dA + [∫hu dy]_E - [∫hu dy]_W + [∫hv dx]_N - [∫hv dx]_S = 0
- x-Mom: d/dt ∫hu dA + [∫(Kx + P) dy]_E - ... + [∫C dx]_N - ... = -g·(db/dx)·∫h dA
- y-Mom: d/dt ∫hv dA + [∫C dy]_E - ... + [∫(Ky + P) dx]_N - ... = 0

The x-momentum equation has a source term from the sloping bathymetry.
For constant slope db/dx, this becomes: -g·db_dx·h_int (linear in h_int!)

Note: These are LINEAR in the face integrals and cell integrals.
"""
function build_fvm_constraint_matrix(timest, ginfo, p, time_range; crank_nicolson=true)
    (; N_x, N_cells, n_cells_x, n_cells_y, n_vert_faces, n_horiz_faces, Δx, Δy) = ginfo
    n_times = length(time_range)

    # Each time step has 3 × N_cells constraints (mass + x-mom + y-mom)
    n_constraints = 3 * N_cells * n_times
    n_state = length(timest)

    I = Int[]
    J = Int[]
    V = Float64[]

    w_curr = crank_nicolson ? 0.5 : 1.0
    w_next = crank_nicolson ? 0.5 : 0.0

    row = 0

    # Vertical face indexing: face at x[i] spanning y-interval j has index (j-1)*N_x + i
    # Horizontal face indexing: face at y[j] spanning x-interval i has index (j-1)*n_cells_x + i
    function vert_face_idx(i, j)
        # i in 1:N_x, j in 1:n_cells_y
        return (j - 1) * N_x + i
    end
    function horiz_face_idx(i, j)
        # i in 1:n_cells_x, j in 1:N_y
        return (j - 1) * n_cells_x + i
    end

    for (t_local, t) in enumerate(time_range)
        t_next = t + 1

        # Get all indices for current time step
        # Note: We're using face integrals for fluxes

        for k in 1:N_cells
            # Cell (i, j) where k = (j-1)*n_cells_x + i
            ci = mod1(k, n_cells_x)
            cj = div(k - 1, n_cells_x) + 1

            # Face indices for this cell
            # East: vert face at x[ci+1], y-interval cj
            # West: vert face at x[ci], y-interval cj
            # North: horiz face at y[cj+1], x-interval ci
            # South: horiz face at y[cj], x-interval ci
            east_vert = vert_face_idx(ci + 1, cj)
            west_vert = vert_face_idx(ci, cj)
            north_horiz = horiz_face_idx(ci, cj + 1)
            south_horiz = horiz_face_idx(ci, cj)

            # ================================================================
            # Mass conservation: dh_int/dt + [∫hu dy]_E - [∫hu dy]_W
            #                           + [∫hv dx]_N - [∫hv dx]_S = 0
            # ================================================================
            row += 1

            # dh_int/dt at current time
            dh_int_dt_idx = first(absindices(timest, :dh_int_dt, k, t))
            push!(I, row); push!(J, dh_int_dt_idx); push!(V, 1.0)

            # East face: +hu_vert[east]
            hu_vert_E_curr = first(absindices(timest, :hu_vert, east_vert, t))
            push!(I, row); push!(J, hu_vert_E_curr); push!(V, w_curr)

            # West face: -hu_vert[west]
            hu_vert_W_curr = first(absindices(timest, :hu_vert, west_vert, t))
            push!(I, row); push!(J, hu_vert_W_curr); push!(V, -w_curr)

            # North face: +hv_horiz[north]
            hv_horiz_N_curr = first(absindices(timest, :hv_horiz, north_horiz, t))
            push!(I, row); push!(J, hv_horiz_N_curr); push!(V, w_curr)

            # South face: -hv_horiz[south]
            hv_horiz_S_curr = first(absindices(timest, :hv_horiz, south_horiz, t))
            push!(I, row); push!(J, hv_horiz_S_curr); push!(V, -w_curr)

            if w_next > 0
                hu_vert_E_next = first(absindices(timest, :hu_vert, east_vert, t_next))
                hu_vert_W_next = first(absindices(timest, :hu_vert, west_vert, t_next))
                hv_horiz_N_next = first(absindices(timest, :hv_horiz, north_horiz, t_next))
                hv_horiz_S_next = first(absindices(timest, :hv_horiz, south_horiz, t_next))

                push!(I, row); push!(J, hu_vert_E_next); push!(V, w_next)
                push!(I, row); push!(J, hu_vert_W_next); push!(V, -w_next)
                push!(I, row); push!(J, hv_horiz_N_next); push!(V, w_next)
                push!(I, row); push!(J, hv_horiz_S_next); push!(V, -w_next)
            end

            # ================================================================
            # x-Momentum: dhu_int/dt + [∫(Kx + P) dy]_E - [∫(Kx + P) dy]_W
            #                        + [∫C dx]_N - [∫C dx]_S = -g·h·(∂b/∂x)
            # Rearranged: dhu_int/dt + [fluxes] + g·db_dx·h_int = 0
            # ================================================================
            row += 1

            # dhu_int/dt at current time
            dhu_int_dt_idx = first(absindices(timest, :dhu_int_dt, k, t))
            push!(I, row); push!(J, dhu_int_dt_idx); push!(V, 1.0)

            # Bathymetry source term: -gh·∂b/∂x on RHS
            # Moving to LHS: +g·db_dx·h_int (balances pressure gradient at rest)
            h_int_idx = first(absindices(timest, :h_int, k, t))
            push!(I, row); push!(J, h_int_idx); push!(V, p.g * p.db_dx[k])

            # East face: +(Kx_vert + P_vert)[east]
            Kx_vert_E = first(absindices(timest, :Kx_vert, east_vert, t))
            P_vert_E = first(absindices(timest, :P_vert, east_vert, t))
            push!(I, row); push!(J, Kx_vert_E); push!(V, w_curr)
            push!(I, row); push!(J, P_vert_E); push!(V, w_curr)

            # West face: -(Kx_vert + P_vert)[west]
            Kx_vert_W = first(absindices(timest, :Kx_vert, west_vert, t))
            P_vert_W = first(absindices(timest, :P_vert, west_vert, t))
            push!(I, row); push!(J, Kx_vert_W); push!(V, -w_curr)
            push!(I, row); push!(J, P_vert_W); push!(V, -w_curr)

            # North face: +C_horiz[north]
            C_horiz_N = first(absindices(timest, :C_horiz, north_horiz, t))
            push!(I, row); push!(J, C_horiz_N); push!(V, w_curr)

            # South face: -C_horiz[south]
            C_horiz_S = first(absindices(timest, :C_horiz, south_horiz, t))
            push!(I, row); push!(J, C_horiz_S); push!(V, -w_curr)

            if w_next > 0
                Kx_vert_E_next = first(absindices(timest, :Kx_vert, east_vert, t_next))
                P_vert_E_next = first(absindices(timest, :P_vert, east_vert, t_next))
                Kx_vert_W_next = first(absindices(timest, :Kx_vert, west_vert, t_next))
                P_vert_W_next = first(absindices(timest, :P_vert, west_vert, t_next))
                C_horiz_N_next = first(absindices(timest, :C_horiz, north_horiz, t_next))
                C_horiz_S_next = first(absindices(timest, :C_horiz, south_horiz, t_next))

                push!(I, row); push!(J, Kx_vert_E_next); push!(V, w_next)
                push!(I, row); push!(J, P_vert_E_next); push!(V, w_next)
                push!(I, row); push!(J, Kx_vert_W_next); push!(V, -w_next)
                push!(I, row); push!(J, P_vert_W_next); push!(V, -w_next)
                push!(I, row); push!(J, C_horiz_N_next); push!(V, w_next)
                push!(I, row); push!(J, C_horiz_S_next); push!(V, -w_next)
            end

            # ================================================================
            # y-Momentum: dhv_int/dt + [∫C dy]_E - [∫C dy]_W
            #                        + [∫(Ky + P) dx]_N - [∫(Ky + P) dx]_S = -g·h·(∂b/∂y)
            # Rearranged: dhv_int/dt + [fluxes] + g·db_dy·h_int = 0
            # ================================================================
            row += 1

            # dhv_int/dt at current time
            dhv_int_dt_idx = first(absindices(timest, :dhv_int_dt, k, t))
            push!(I, row); push!(J, dhv_int_dt_idx); push!(V, 1.0)

            # Bathymetry source term: -gh·∂b/∂y on RHS
            # Moving to LHS: +g·db_dy·h_int
            h_int_idx_y = first(absindices(timest, :h_int, k, t))
            push!(I, row); push!(J, h_int_idx_y); push!(V, p.g * p.db_dy[k])

            # East face: +C_vert[east]
            C_vert_E = first(absindices(timest, :C_vert, east_vert, t))
            push!(I, row); push!(J, C_vert_E); push!(V, w_curr)

            # West face: -C_vert[west]
            C_vert_W = first(absindices(timest, :C_vert, west_vert, t))
            push!(I, row); push!(J, C_vert_W); push!(V, -w_curr)

            # North face: +(Ky_horiz + P_horiz)[north]
            Ky_horiz_N = first(absindices(timest, :Ky_horiz, north_horiz, t))
            P_horiz_N = first(absindices(timest, :P_horiz, north_horiz, t))
            push!(I, row); push!(J, Ky_horiz_N); push!(V, w_curr)
            push!(I, row); push!(J, P_horiz_N); push!(V, w_curr)

            # South face: -(Ky_horiz + P_horiz)[south]
            Ky_horiz_S = first(absindices(timest, :Ky_horiz, south_horiz, t))
            P_horiz_S = first(absindices(timest, :P_horiz, south_horiz, t))
            push!(I, row); push!(J, Ky_horiz_S); push!(V, -w_curr)
            push!(I, row); push!(J, P_horiz_S); push!(V, -w_curr)

            if w_next > 0
                C_vert_E_next = first(absindices(timest, :C_vert, east_vert, t_next))
                C_vert_W_next = first(absindices(timest, :C_vert, west_vert, t_next))
                Ky_horiz_N_next = first(absindices(timest, :Ky_horiz, north_horiz, t_next))
                P_horiz_N_next = first(absindices(timest, :P_horiz, north_horiz, t_next))
                Ky_horiz_S_next = first(absindices(timest, :Ky_horiz, south_horiz, t_next))
                P_horiz_S_next = first(absindices(timest, :P_horiz, south_horiz, t_next))

                push!(I, row); push!(J, C_vert_E_next); push!(V, w_next)
                push!(I, row); push!(J, C_vert_W_next); push!(V, -w_next)
                push!(I, row); push!(J, Ky_horiz_N_next); push!(V, w_next)
                push!(I, row); push!(J, P_horiz_N_next); push!(V, w_next)
                push!(I, row); push!(J, Ky_horiz_S_next); push!(V, -w_next)
                push!(I, row); push!(J, P_horiz_S_next); push!(V, -w_next)
            end
        end
    end

    A = sparse(I, J, V, n_constraints, n_state)
    b = zeros(n_constraints)

    return A, b
end

"""
    FluxConstraintIndices

Pre-computed indices for the nonlinear flux constraint.
"""
struct FluxConstraintIndices
    # Primary state at face midpoints
    h_vert_face::Vector{Int}
    h_horiz_face::Vector{Int}
    hu_vert_face::Vector{Int}
    hu_horiz_face::Vector{Int}
    hv_vert_face::Vector{Int}
    hv_horiz_face::Vector{Int}
    # Auxiliary flux at face midpoints
    P_vert_face::Vector{Int}
    P_horiz_face::Vector{Int}
    Kx_vert_face::Vector{Int}
    Kx_horiz_face::Vector{Int}
    Ky_vert_face::Vector{Int}
    Ky_horiz_face::Vector{Int}
    C_vert_face::Vector{Int}
    C_horiz_face::Vector{Int}
end

"""
    build_flux_constraint_indices(timest, ginfo, time_range)

Build indices for the nonlinear flux constraint.
"""
function build_flux_constraint_indices(timest, ginfo, time_range)
    all_h_vert = Int[]
    all_h_horiz = Int[]
    all_hu_vert = Int[]
    all_hu_horiz = Int[]
    all_hv_vert = Int[]
    all_hv_horiz = Int[]
    all_P_vert = Int[]
    all_P_horiz = Int[]
    all_Kx_vert = Int[]
    all_Kx_horiz = Int[]
    all_Ky_vert = Int[]
    all_Ky_horiz = Int[]
    all_C_vert = Int[]
    all_C_horiz = Int[]

    for t in time_range
        append!(all_h_vert, absindices(timest, :h_vert_face, :, t))
        append!(all_h_horiz, absindices(timest, :h_horiz_face, :, t))
        append!(all_hu_vert, absindices(timest, :hu_vert_face, :, t))
        append!(all_hu_horiz, absindices(timest, :hu_horiz_face, :, t))
        append!(all_hv_vert, absindices(timest, :hv_vert_face, :, t))
        append!(all_hv_horiz, absindices(timest, :hv_horiz_face, :, t))
        append!(all_P_vert, absindices(timest, :P_vert_face, :, t))
        append!(all_P_horiz, absindices(timest, :P_horiz_face, :, t))
        append!(all_Kx_vert, absindices(timest, :Kx_vert_face, :, t))
        append!(all_Kx_horiz, absindices(timest, :Kx_horiz_face, :, t))
        append!(all_Ky_vert, absindices(timest, :Ky_vert_face, :, t))
        append!(all_Ky_horiz, absindices(timest, :Ky_horiz_face, :, t))
        append!(all_C_vert, absindices(timest, :C_vert_face, :, t))
        append!(all_C_horiz, absindices(timest, :C_horiz_face, :, t))
    end

    FluxConstraintIndices(
        all_h_vert, all_h_horiz,
        all_hu_vert, all_hu_horiz,
        all_hv_vert, all_hv_horiz,
        all_P_vert, all_P_horiz,
        all_Kx_vert, all_Kx_horiz,
        all_Ky_vert, all_Ky_horiz,
        all_C_vert, all_C_horiz
    )
end

"""
    flux_residual(x, idx::FluxConstraintIndices, g)

Compute the nonlinear flux constraint residual.

At each face midpoint:
- P = 0.5 * g * h²
- Kx = hu² / h
- Ky = hv² / h
- C = hu * hv / h

Returns concatenated residuals for all constraints.
"""
function flux_residual(x, idx::FluxConstraintIndices, g)
    # Vertical faces
    h_v = x[idx.h_vert_face]
    hu_v = x[idx.hu_vert_face]
    hv_v = x[idx.hv_vert_face]
    P_v = x[idx.P_vert_face]
    Kx_v = x[idx.Kx_vert_face]
    Ky_v = x[idx.Ky_vert_face]
    C_v = x[idx.C_vert_face]

    # Horizontal faces
    h_h = x[idx.h_horiz_face]
    hu_h = x[idx.hu_horiz_face]
    hv_h = x[idx.hv_horiz_face]
    P_h = x[idx.P_horiz_face]
    Kx_h = x[idx.Kx_horiz_face]
    Ky_h = x[idx.Ky_horiz_face]
    C_h = x[idx.C_horiz_face]

    # Residuals for vertical faces
    r_P_v = P_v .- 0.5 .* g .* h_v.^2
    r_Kx_v = Kx_v .- hu_v.^2 ./ h_v
    r_Ky_v = Ky_v .- hv_v.^2 ./ h_v
    r_C_v = C_v .- hu_v .* hv_v ./ h_v

    # Residuals for horizontal faces
    r_P_h = P_h .- 0.5 .* g .* h_h.^2
    r_Kx_h = Kx_h .- hu_h.^2 ./ h_h
    r_Ky_h = Ky_h .- hv_h.^2 ./ h_h
    r_C_h = C_h .- hu_h .* hv_h ./ h_h

    return vcat(r_P_v, r_Kx_v, r_Ky_v, r_C_v, r_P_h, r_Kx_h, r_Ky_h, r_C_h)
end

"""
    build_flux_likelihood(n_state, idx::FluxConstraintIndices, g, σ_flux)

Build the NonlinearLeastSquaresModel for the flux constraint.
"""
function build_flux_likelihood(n_state, idx::FluxConstraintIndices, g, σ_flux)
    n_constraints = 8 * length(idx.h_vert_face)  # 4 constraints × 2 face types
    residual_fn = x -> flux_residual(x, idx, g)
    model = NonlinearLeastSquaresModel(residual_fn, n_state)
    return model(zeros(n_constraints); σ=σ_flux)
end
