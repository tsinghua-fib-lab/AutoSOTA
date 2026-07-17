#!/usr/bin/env julia --project=/repo
# LOO-CV hyperparameter selection for GP-FVM source identification
# Outputs best lengthscale params for use by run_gpfvm.jl

using LinearAlgebra, SparseArrays, NPZ

push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields

import GaussianMarkovRandomFields: mean, std
import FunctionalGPs: ⊗

include("problem.jl")
include("run_gpfvm.jl")  # brings in build_fvm_constraint, build_boundary_constraints

function compact_loo_score(prob, data, ℓ_c, ℓ_s, ρ_c, smoothness)
    xs = data["xs"]; ys = data["ys"]
    Nx, Ny = length(xs), length(ys)
    n_cells_x, n_cells_y = Nx - 1, Ny - 1

    x_intervals = intervals_from_endpoints(collect(xs))
    y_intervals = intervals_from_endpoints(collect(ys))
    grid = FactorizedGrid(xs, ys)
    cells_2d = x_intervals ⊗ y_intervals

    # Functionals
    L_c_eval = EvaluationFunctional(grid)
    L_c_vert = EvaluationFunctional(xs) ⊗ VectorizedLebesgueIntegral(y_intervals)
    L_c_horiz = VectorizedLebesgueIntegral(x_intervals) ⊗ EvaluationFunctional(ys)
    L_c_dx_vert = L_c_vert ∘ PartialDerivative((1, 0))
    L_c_dy_horiz = L_c_horiz ∘ PartialDerivative((0, 1))
    L_s_eval = EvaluationFunctional(grid)
    L_s_int = VectorizedLebesgueIntegral(cells_2d)

    # Kernels
    k_c = HalfIntegerMaternKernel(smoothness, [ℓ_c]) ⊗ HalfIntegerMaternKernel(smoothness, [ℓ_c])
    k_s = HalfIntegerMaternKernel(smoothness, [ℓ_s]) ⊗ HalfIntegerMaternKernel(smoothness, [ℓ_s])

    # Sparse precisions
    approx_c = sparse_precision([
        :c => L_c_eval, :c_vert => L_c_vert, :c_horiz => L_c_horiz,
        :c_dx_vert => L_c_dx_vert, :c_dy_horiz => L_c_dy_horiz,
    ], k_c; ρ=ρ_c, ordering=:integrals_coarsest)

    approx_s = sparse_precision([
        :s => L_s_eval, :s_int => L_s_int,
    ], k_s; ρ=4.0, ordering=:integrals_coarsest)

    n_c = approx_c.info.n
    n_s = approx_s.info.n
    n_total = n_c + n_s
    layout_c = approx_c.layout
    layout_s = approx_s.layout

    Q_joint = blockdiag(sparse(approx_c.Q), sparse(approx_s.Q))

    # FVM + BC constraints
    A_fvm, b_fvm = build_fvm_constraint(xs, ys, prob, layout_c, layout_s, n_c)
    A_bc, b_bc = build_boundary_constraints(xs, ys, prob, layout_c, n_c, n_total)

    n_fvm = size(A_fvm, 1)
    n_bc = size(A_bc, 1)
    A_constraints = vcat(A_fvm, A_bc)
    b_constraints = vcat(b_fvm, b_bc)
    constraint_noise = 1e-5
    Q_constraints = (1.0 / constraint_noise^2) * sparse(I, n_fvm + n_bc, n_fvm + n_bc)

    # Observations
    obs_x = data["obs_x"]; obs_y = data["obs_y"]; obs_c = data["obs_c"]
    noise_std = Float64(data["noise_std"])
    n_obs = length(obs_c)

    obs_ix = [argmin(abs.(xs .- ox)) for ox in obs_x]
    obs_iy = [argmin(abs.(ys .- oy)) for oy in obs_y]
    obs_indices = [indices(layout_c, :c)[(iy-1)*Nx + ix] for (ix, iy) in zip(obs_ix, obs_iy)]

    A_obs = spzeros(n_obs, n_total)
    for (i, idx) in enumerate(obs_indices)
        A_obs[i, idx] = 1.0
    end

    # Condition on model (FVM + BC)
    Q_model = Q_joint + A_constraints' * Q_constraints * A_constraints
    rhs_model = A_constraints' * Q_constraints * b_constraints

    F = cholesky(Symmetric(Q_model))
    μ_model = F \ rhs_model

    # LOO-CV over observations (Rasmussen & Williams Eq. 5.12)
    z_obs = A_obs * μ_model
    X = F \ Matrix(A_obs')
    S_mat = A_obs * X + noise_std^2 * I(n_obs)
    S_mat = Symmetric(S_mat)

    r = obs_c - z_obs
    try
        F_S = cholesky(S_mat)
        α = F_S \ r
        S_inv_diag = diag(inv(F_S))
        loo = -0.5 * sum(i -> -log(S_inv_diag[i]) + α[i]^2 / S_inv_diag[i], 1:n_obs)
        return loo
    catch e
        return -Inf
    end
end

# ==============================================================================
# Main: grid search
# ==============================================================================

function main()
    prob = load_problem("problems/rubric_v2.toml")
    data = npzread("data/rubric_v2.npz")
    xs = data["xs"]; ys = data["ys"]
    Δ = min(xs[2] - xs[1], ys[2] - ys[1])
    ℓ_5Δ = 5 * Δ

    # Search ranges: 0.4x to 2.5x of default, 7x7 grid
    ℓ_c_range = collect(range(0.4 * ℓ_5Δ, 2.5 * ℓ_5Δ, length=7))
    ℓ_s_range = collect(range(0.4 * ℓ_5Δ, 2.5 * ℓ_5Δ, length=7))

    println("LOO-CV Hyperparameter Selection")
    println("Default (5Δ): ℓ=$(round(ℓ_5Δ, digits=4))")
    println("Search: ℓ_c ∈ [$(round(first(ℓ_c_range), digits=4)), $(round(last(ℓ_c_range), digits=4))] x ℓ_s ∈ [$(round(first(ℓ_s_range), digits=4)), $(round(last(ℓ_s_range), digits=4))]")
    println()

    best_score = -Inf
    best_ℓc = ℓ_5Δ
    best_ℓs = ℓ_5Δ

    for ℓ_c in ℓ_c_range
        for ℓ_s in ℓ_s_range
            score = compact_loo_score(prob, data, ℓ_c, ℓ_s, 2.0, 2)
            if score > best_score
                best_score = score
                best_ℓc = ℓ_c
                best_ℓs = ℓ_s
            end
            println("  ℓ_c=$(round(ℓ_c, digits=4))  ℓ_s=$(round(ℓ_s, digits=4))  → LOO=$(round(score, digits=3))")
        end
    end

    default_score = compact_loo_score(prob, data, ℓ_5Δ, ℓ_5Δ, 2.0, 2)
    println()
    println("Best:  ℓ_c=$(round(best_ℓc, digits=6)), ℓ_s=$(round(best_ℓs, digits=6)) (LOO=$(round(best_score, digits=4)))")
    println("5Δ:    ℓ_c=$(round(ℓ_5Δ, digits=6)), ℓ_s=$(round(ℓ_5Δ, digits=6)) (LOO=$(round(default_score, digits=4)))")
    println("Improvement: $(round(best_score - default_score, digits=4))")

    mkpath("results")
    open("results/best_params_loo.txt", "w") do f
        write(f, "BEST_LCS=$(round(best_ℓc, digits=6))\n")
        write(f, "BEST_LSS=$(round(best_ℓs, digits=6))\n")
    end
    println("Saved best params to results/best_params_loo.txt")
end

main()
