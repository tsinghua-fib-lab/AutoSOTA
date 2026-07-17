"""
Functional definitions and sparse precision building for nonlinear SWE.

Key structures:
- Primary state (h, hu, hv): evals, cell integrals, face integrals, face midpoint evals
- Auxiliary flux (P, K_x, K_y, C): face integrals, face midpoint evals
"""

using GPFiniteVolume
using FunctionalGPs
using GaussianMarkovRandomFields
using SparseArrays, LinearAlgebra

import FunctionalGPs: ⊗

"""
    GridInfo

Holds grid dimensions and derived quantities.
"""
struct GridInfo
    N_x::Int
    N_y::Int
    N_grid::Int
    n_cells_x::Int
    n_cells_y::Int
    N_cells::Int
    n_vert_faces::Int
    n_horiz_faces::Int
    Δx::Float64
    Δy::Float64
end

function GridInfo(xs, ys)
    N_x, N_y = length(xs), length(ys)
    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]
    GridInfo(
        N_x, N_y,
        N_x * N_y,
        N_x - 1, N_y - 1,
        (N_x - 1) * (N_y - 1),
        N_x * (N_y - 1),      # vertical faces
        (N_x - 1) * N_y,      # horizontal faces
        Δx, Δy
    )
end

"""
    build_functionals(xs, ys, x_intervals, y_intervals)

Build all functionals needed for the nonlinear SWE.

Returns named tuple with:
- L_eval: point evaluations at grid corners
- L_int: cell integrals
- L_vert: vertical face integrals
- L_horiz: horizontal face integrals
- L_vert_face_eval: evaluations at vertical face midpoints
- L_horiz_face_eval: evaluations at horizontal face midpoints
- vert_face_midpoints: grid for vertical face midpoints
- horiz_face_midpoints: grid for horizontal face midpoints
"""
function build_functionals(xs, ys, x_intervals, y_intervals)
    # Convert to vectors for consistent types
    xs_vec = collect(Float64, xs)
    ys_vec = collect(Float64, ys)

    grid = FactorizedGrid(xs_vec, ys_vec)
    cells_2d = x_intervals ⊗ y_intervals

    n_cells_y = length(y_intervals)
    n_cells_x = length(x_intervals)

    # Face midpoints
    y_cell_mids = [0.5*(ys[j] + ys[j+1]) for j in 1:n_cells_y]
    x_cell_mids = [0.5*(xs[i] + xs[i+1]) for i in 1:n_cells_x]
    vert_face_midpoints = FactorizedGrid(xs_vec, y_cell_mids)
    horiz_face_midpoints = FactorizedGrid(x_cell_mids, ys_vec)

    return (
        grid = grid,
        cells_2d = cells_2d,
        xs = xs_vec,
        ys = ys_vec,
        L_eval = EvaluationFunctional(grid),
        L_int = VectorizedLebesgueIntegral(cells_2d),
        L_vert = EvaluationFunctional(xs_vec) ⊗ VectorizedLebesgueIntegral(y_intervals),
        L_horiz = VectorizedLebesgueIntegral(x_intervals) ⊗ EvaluationFunctional(ys_vec),
        L_vert_face_eval = EvaluationFunctional(vert_face_midpoints),
        L_horiz_face_eval = EvaluationFunctional(horiz_face_midpoints),
        vert_face_midpoints = vert_face_midpoints,
        horiz_face_midpoints = horiz_face_midpoints,
    )
end

"""
    build_primary_precision(funcs, kernel, ρ; name=:h)

Build sparse precision for a primary variable (h, hu, or hv).

Returns SparseGMRFApproximation with functionals:
- eval, int, vert, horiz, vert_face, horiz_face
"""
function build_primary_precision(funcs, kernel, ρ; name=:h)
    sparse_precision([
        Symbol(name) => funcs.L_eval,
        Symbol(name, :_int) => funcs.L_int,
        Symbol(name, :_vert) => funcs.L_vert,
        Symbol(name, :_horiz) => funcs.L_horiz,
        Symbol(name, :_vert_face) => funcs.L_vert_face_eval,
        Symbol(name, :_horiz_face) => funcs.L_horiz_face_eval,
    ], kernel; ρ=ρ, ordering=:integrals_coarsest)
end

"""
    build_flux_precision(funcs, kernel, ρ; name=:P)

Build sparse precision for an auxiliary flux variable (P, Kx, Ky, or C).

Returns SparseGMRFApproximation with functionals:
- vert, horiz (face integrals for FVM)
- vert_face, horiz_face (face midpoint evals for nonlinear constraints)
"""
function build_flux_precision(funcs, kernel, ρ; name=:P)
    sparse_precision([
        Symbol(name, :_vert) => funcs.L_vert,
        Symbol(name, :_horiz) => funcs.L_horiz,
        Symbol(name, :_vert_face) => funcs.L_vert_face_eval,
        Symbol(name, :_horiz_face) => funcs.L_horiz_face_eval,
    ], kernel; ρ=ρ, ordering=:integrals_coarsest)
end

"""
    PrecisionInfo

Holds all sparse precision approximations and the joint precision.
"""
struct PrecisionInfo
    approx_h::Any
    approx_hu::Any
    approx_hv::Any
    approx_P::Any
    approx_Kx::Any
    approx_Ky::Any
    approx_C::Any
    Q_space::SparseMatrixCSC{Float64, Int64}
    N_space::Int
end

"""
    build_all_precisions(funcs, k_primary, k_flux, ρ; verbose=false)

Build all sparse precisions and combine into block-diagonal joint precision.
"""
function build_all_precisions(funcs, k_primary, k_flux, ρ; verbose=false)
    verbose && println("Building sparse precisions (ρ = $ρ)...")

    approx_h = build_primary_precision(funcs, k_primary, ρ; name=:h)
    verbose && println("  h: ", approx_h)

    approx_hu = build_primary_precision(funcs, k_primary, ρ; name=:hu)
    verbose && println("  hu: ", approx_hu)

    approx_hv = build_primary_precision(funcs, k_primary, ρ; name=:hv)
    verbose && println("  hv: ", approx_hv)

    approx_P = build_flux_precision(funcs, k_flux, ρ; name=:P)
    verbose && println("  P: ", approx_P)

    approx_Kx = build_flux_precision(funcs, k_flux, ρ; name=:Kx)
    verbose && println("  Kx: ", approx_Kx)

    approx_Ky = build_flux_precision(funcs, k_flux, ρ; name=:Ky)
    verbose && println("  Ky: ", approx_Ky)

    approx_C = build_flux_precision(funcs, k_flux, ρ; name=:C)
    verbose && println("  C: ", approx_C)

    Q_space = blockdiag(
        sparse(approx_h.Q),
        sparse(approx_hu.Q),
        sparse(approx_hv.Q),
        sparse(approx_P.Q),
        sparse(approx_Kx.Q),
        sparse(approx_Ky.Q),
        sparse(approx_C.Q)
    )

    return PrecisionInfo(
        approx_h, approx_hu, approx_hv,
        approx_P, approx_Kx, approx_Ky, approx_C,
        Q_space, size(Q_space, 1)
    )
end

"""
    build_spatial_layout(ginfo::GridInfo)

Build the layout for the spatial state (single timestep, no time derivatives).
"""
function build_spatial_layout(ginfo::GridInfo)
    layout((
        # Primary h
        h = ginfo.N_grid,
        h_int = ginfo.N_cells,
        h_vert = ginfo.n_vert_faces,
        h_horiz = ginfo.n_horiz_faces,
        h_vert_face = ginfo.n_vert_faces,
        h_horiz_face = ginfo.n_horiz_faces,
        # Primary hu
        hu = ginfo.N_grid,
        hu_int = ginfo.N_cells,
        hu_vert = ginfo.n_vert_faces,
        hu_horiz = ginfo.n_horiz_faces,
        hu_vert_face = ginfo.n_vert_faces,
        hu_horiz_face = ginfo.n_horiz_faces,
        # Primary hv
        hv = ginfo.N_grid,
        hv_int = ginfo.N_cells,
        hv_vert = ginfo.n_vert_faces,
        hv_horiz = ginfo.n_horiz_faces,
        hv_vert_face = ginfo.n_vert_faces,
        hv_horiz_face = ginfo.n_horiz_faces,
        # Auxiliary P
        P_vert = ginfo.n_vert_faces,
        P_horiz = ginfo.n_horiz_faces,
        P_vert_face = ginfo.n_vert_faces,
        P_horiz_face = ginfo.n_horiz_faces,
        # Auxiliary Kx
        Kx_vert = ginfo.n_vert_faces,
        Kx_horiz = ginfo.n_horiz_faces,
        Kx_vert_face = ginfo.n_vert_faces,
        Kx_horiz_face = ginfo.n_horiz_faces,
        # Auxiliary Ky
        Ky_vert = ginfo.n_vert_faces,
        Ky_horiz = ginfo.n_horiz_faces,
        Ky_vert_face = ginfo.n_vert_faces,
        Ky_horiz_face = ginfo.n_horiz_faces,
        # Auxiliary C
        C_vert = ginfo.n_vert_faces,
        C_horiz = ginfo.n_horiz_faces,
        C_vert_face = ginfo.n_vert_faces,
        C_horiz_face = ginfo.n_horiz_faces,
    ))
end

"""
    build_full_state_layout(ginfo::GridInfo)

Build the layout for the full IWP state: [spatial; d/dt spatial].

The IWP prior needs time derivatives for ALL spatial components to model
temporal evolution. This doubles the state size.
"""
function build_full_state_layout(ginfo::GridInfo)
    # Full IWP state: [spatial_state; d/dt spatial_state]
    layout((
        # ===== SPATIAL STATE =====
        # Primary h
        h = ginfo.N_grid,
        h_int = ginfo.N_cells,
        h_vert = ginfo.n_vert_faces,
        h_horiz = ginfo.n_horiz_faces,
        h_vert_face = ginfo.n_vert_faces,
        h_horiz_face = ginfo.n_horiz_faces,
        # Primary hu
        hu = ginfo.N_grid,
        hu_int = ginfo.N_cells,
        hu_vert = ginfo.n_vert_faces,
        hu_horiz = ginfo.n_horiz_faces,
        hu_vert_face = ginfo.n_vert_faces,
        hu_horiz_face = ginfo.n_horiz_faces,
        # Primary hv
        hv = ginfo.N_grid,
        hv_int = ginfo.N_cells,
        hv_vert = ginfo.n_vert_faces,
        hv_horiz = ginfo.n_horiz_faces,
        hv_vert_face = ginfo.n_vert_faces,
        hv_horiz_face = ginfo.n_horiz_faces,
        # Auxiliary P
        P_vert = ginfo.n_vert_faces,
        P_horiz = ginfo.n_horiz_faces,
        P_vert_face = ginfo.n_vert_faces,
        P_horiz_face = ginfo.n_horiz_faces,
        # Auxiliary Kx
        Kx_vert = ginfo.n_vert_faces,
        Kx_horiz = ginfo.n_horiz_faces,
        Kx_vert_face = ginfo.n_vert_faces,
        Kx_horiz_face = ginfo.n_horiz_faces,
        # Auxiliary Ky
        Ky_vert = ginfo.n_vert_faces,
        Ky_horiz = ginfo.n_horiz_faces,
        Ky_vert_face = ginfo.n_vert_faces,
        Ky_horiz_face = ginfo.n_horiz_faces,
        # Auxiliary C
        C_vert = ginfo.n_vert_faces,
        C_horiz = ginfo.n_horiz_faces,
        C_vert_face = ginfo.n_vert_faces,
        C_horiz_face = ginfo.n_horiz_faces,
        # ===== TIME DERIVATIVES OF SPATIAL STATE =====
        # d/dt Primary h
        dh_dt = ginfo.N_grid,
        dh_int_dt = ginfo.N_cells,
        dh_vert_dt = ginfo.n_vert_faces,
        dh_horiz_dt = ginfo.n_horiz_faces,
        dh_vert_face_dt = ginfo.n_vert_faces,
        dh_horiz_face_dt = ginfo.n_horiz_faces,
        # d/dt Primary hu
        dhu_dt = ginfo.N_grid,
        dhu_int_dt = ginfo.N_cells,
        dhu_vert_dt = ginfo.n_vert_faces,
        dhu_horiz_dt = ginfo.n_horiz_faces,
        dhu_vert_face_dt = ginfo.n_vert_faces,
        dhu_horiz_face_dt = ginfo.n_horiz_faces,
        # d/dt Primary hv
        dhv_dt = ginfo.N_grid,
        dhv_int_dt = ginfo.N_cells,
        dhv_vert_dt = ginfo.n_vert_faces,
        dhv_horiz_dt = ginfo.n_horiz_faces,
        dhv_vert_face_dt = ginfo.n_vert_faces,
        dhv_horiz_face_dt = ginfo.n_horiz_faces,
        # d/dt Auxiliary P
        dP_vert_dt = ginfo.n_vert_faces,
        dP_horiz_dt = ginfo.n_horiz_faces,
        dP_vert_face_dt = ginfo.n_vert_faces,
        dP_horiz_face_dt = ginfo.n_horiz_faces,
        # d/dt Auxiliary Kx
        dKx_vert_dt = ginfo.n_vert_faces,
        dKx_horiz_dt = ginfo.n_horiz_faces,
        dKx_vert_face_dt = ginfo.n_vert_faces,
        dKx_horiz_face_dt = ginfo.n_horiz_faces,
        # d/dt Auxiliary Ky
        dKy_vert_dt = ginfo.n_vert_faces,
        dKy_horiz_dt = ginfo.n_horiz_faces,
        dKy_vert_face_dt = ginfo.n_vert_faces,
        dKy_horiz_face_dt = ginfo.n_horiz_faces,
        # d/dt Auxiliary C
        dC_vert_dt = ginfo.n_vert_faces,
        dC_horiz_dt = ginfo.n_horiz_faces,
        dC_vert_face_dt = ginfo.n_vert_faces,
        dC_horiz_face_dt = ginfo.n_horiz_faces,
    ))
end
