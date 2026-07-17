"""
GP prior construction for u and s fields.
"""

function build_u_precision(xs, ys; smoothness::Int=2, lengthscale::Float64=0.15, ρ::Float64=2.0)
    x_intervals = intervals_from_endpoints(xs)
    y_intervals = intervals_from_endpoints(ys)
    grid = FactorizedGrid(xs, ys)
    cells_2d = x_intervals ⊗ y_intervals

    k = HalfIntegerMaternKernel(smoothness, [lengthscale]) ⊗
        HalfIntegerMaternKernel(smoothness, [lengthscale])

    L_eval = EvaluationFunctional(grid)
    L_dx = L_eval ∘ PartialDerivative((1, 0))
    L_dy = L_eval ∘ PartialDerivative((0, 1))
    L_int = VectorizedLebesgueIntegral(cells_2d)

    approx = sparse_precision([
        :u => L_eval,
        :u_dx => L_dx,
        :u_dy => L_dy,
        :u_int => L_int,
    ], k; ρ=ρ, ordering=:integrals_coarsest)

    return approx
end

function build_s_precision(xs, ys; smoothness::Int=2, lengthscale::Float64=0.15,
        ρ_s::Float64=4.0, log_source::Bool=false)
    x_intervals = intervals_from_endpoints(xs)
    y_intervals = intervals_from_endpoints(ys)
    grid = FactorizedGrid(xs, ys)
    cells_2d = x_intervals ⊗ y_intervals

    k = HalfIntegerMaternKernel(smoothness, [lengthscale]) ⊗
        HalfIntegerMaternKernel(smoothness, [lengthscale])

    L_eval = EvaluationFunctional(grid)

    if log_source
        # Log-GP: only point evaluations of g = log(s)
        # Cell integrals of s = exp(g) are computed nonlinearly in the constraint
        approx = sparse_precision([
            :g => L_eval,
        ], k; ρ=ρ_s, ordering=:integrals_coarsest)
    else
        L_int = VectorizedLebesgueIntegral(cells_2d)
        approx = sparse_precision([
            :s => L_eval,
            :s_int => L_int,
        ], k; ρ=ρ_s, ordering=:integrals_coarsest)
    end

    return approx
end

function build_u_state_layout(N_grid, N_cells)
    layout((
        u = N_grid, u_dx = N_grid, u_dy = N_grid, u_int = N_cells,
        du_dt = N_grid, du_dx_dt = N_grid, du_dy_dt = N_grid, du_int_dt = N_cells,
    ))
end
