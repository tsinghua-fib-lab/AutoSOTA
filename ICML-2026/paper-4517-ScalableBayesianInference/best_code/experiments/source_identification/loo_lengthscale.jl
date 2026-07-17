"""
LOO cross-validation on PDE residuals for lengthscale selection.

Idea: condition on BCs + observations only (partial posterior), then compute
leave-one-out pseudo-likelihood of PDE constraints under the partial posterior.
Optimize lengthscale to maximize this LOO score.

Analogy with ODE filters: the Kalman filter innovation calibrates the output
scale by asking "how surprising is this residual given everything seen so far?"
LOO-CV emulates this in a spatial/batch setting: "how surprising is this cell's
PDE residual given all other cells?"

Usage:
    julia --project=../.. loo_lengthscale.jl
    julia --project=../.. loo_lengthscale.jl -N 31 --n-lengthscales 20
"""

using LinearAlgebra, SparseArrays
using Random
using ArgParse
using NPZ
using Statistics
using Printf
using CairoMakie

push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields

import GaussianMarkovRandomFields: mean, std
import FunctionalGPs: ⊗

include("run_gpfvm.jl")

# Reuse ground truth generation from scalability study
function setup_2d_grid(Nx, Ny, domain)
    x_min, x_max, y_min, y_max = domain
    xs = range(x_min, x_max, length=Nx)
    ys = range(y_min, y_max, length=Ny)
    return collect(xs), collect(ys)
end

function solve_forward_problem(xs, ys, prob::SourceIdentificationProblem)
    Nx, Ny = length(xs), length(ys)
    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]
    n_cells_x, n_cells_y = Nx - 1, Ny - 1

    s_int_true = zeros(n_cells_x, n_cells_y)
    for cj in 1:n_cells_y
        cy = 0.5 * (ys[cj] + ys[cj+1])
        for ci in 1:n_cells_x
            cx = 0.5 * (xs[ci] + xs[ci+1])
            s_int_true[ci, cj] = evaluate_source(prob, cx, cy) * Δx * Δy
        end
    end

    n_nodes = Nx * Ny
    node_idx(i, j) = (j - 1) * Nx + i
    rows = Int[]; cols = Int[]; vals = Float64[]
    b = zeros(n_nodes)

    for j in 1:Ny, i in 1:Nx
        idx = node_idx(i, j)
        if i == 1
            push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
            b[idx] = prob.c_inflow
        elseif i == Nx
            push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
            push!(rows, idx); push!(cols, node_idx(i-1, j)); push!(vals, -1.0)
        elseif j == 1
            push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
            push!(rows, idx); push!(cols, node_idx(i, j+1)); push!(vals, -1.0)
        elseif j == Ny
            push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
            push!(rows, idx); push!(cols, node_idx(i, j-1)); push!(vals, -1.0)
        else
            push!(rows, idx); push!(cols, idx); push!(vals, prob.vx * Δy)
            push!(rows, idx); push!(cols, node_idx(i-1, j)); push!(vals, -prob.vx * Δy)
            diff_coef = prob.D / Δx * Δy
            diff_coef_y = prob.D / Δy * Δx
            push!(rows, idx); push!(cols, idx); push!(vals, 2diff_coef + 2diff_coef_y)
            push!(rows, idx); push!(cols, node_idx(i+1, j)); push!(vals, -diff_coef)
            push!(rows, idx); push!(cols, node_idx(i-1, j)); push!(vals, -diff_coef)
            push!(rows, idx); push!(cols, node_idx(i, j+1)); push!(vals, -diff_coef_y)
            push!(rows, idx); push!(cols, node_idx(i, j-1)); push!(vals, -diff_coef_y)
            source = 0.0
            for (ci, cj) in [(i-1, j-1), (i, j-1), (i-1, j), (i, j)]
                if 1 <= ci <= n_cells_x && 1 <= cj <= n_cells_y
                    source += 0.25 * s_int_true[ci, cj]
                end
            end
            b[idx] = source
        end
    end

    A = sparse(rows, cols, vals, n_nodes, n_nodes)
    c_true = reshape(A \ b, Nx, Ny)
    return c_true, s_int_true
end

function generate_observations(xs, ys, c_true, prob::SourceIdentificationProblem)
    Random.seed!(prob.noise_seed)
    obs_xs, obs_ys = observation_coords(prob)
    obs_ix = [argmin(abs.(xs .- ox)) for ox in obs_xs]
    obs_iy = [argmin(abs.(ys .- oy)) for oy in obs_ys]
    true_c = [c_true[ix, iy] for (ix, iy) in zip(obs_ix, obs_iy)]
    return obs_xs, obs_ys, true_c, true_c .+ prob.noise_std * randn(length(obs_xs))
end

# ==============================================================================
# LOO score computation
# ==============================================================================

"""
Compute LOO pseudo-log-likelihood of PDE constraints under the partial
posterior (conditioned on BCs + observations only).

Returns (loo_score, n_fvm_constraints, pde_residual_norm).
"""
function compute_loo_score(prob::SourceIdentificationProblem, data::Dict;
        ρ::Real, lengthscale_c::Real, lengthscale_s::Real,
        output_scale::Real=1.0,
        smoothness::Int=2, source_amplitude::Real=1.0,
        constraint_noise::Real=1e-5)

    xs = data["xs"]; ys = data["ys"]
    Nx, Ny = length(xs), length(ys)
    n_cells_x, n_cells_y = Nx - 1, Ny - 1

    x_intervals = intervals_from_endpoints(collect(xs))
    y_intervals = intervals_from_endpoints(collect(ys))
    grid = FactorizedGrid(xs, ys)
    cells_2d = x_intervals ⊗ y_intervals

    # Kernels
    k_c = HalfIntegerMaternKernel(smoothness, [lengthscale_c]) ⊗
          HalfIntegerMaternKernel(smoothness, [lengthscale_c])
    k_s = HalfIntegerMaternKernel(smoothness, [lengthscale_s]) ⊗
          HalfIntegerMaternKernel(smoothness, [lengthscale_s])

    # Functionals
    L_c_eval = EvaluationFunctional(grid)
    L_c_vert = EvaluationFunctional(xs) ⊗ VectorizedLebesgueIntegral(y_intervals)
    L_c_horiz = VectorizedLebesgueIntegral(x_intervals) ⊗ EvaluationFunctional(ys)
    L_c_dx_vert = L_c_vert ∘ PartialDerivative((1, 0))
    L_c_dy_horiz = L_c_horiz ∘ PartialDerivative((0, 1))
    L_s_eval = EvaluationFunctional(grid)
    L_s_int = VectorizedLebesgueIntegral(cells_2d)

    # Sparse precisions
    approx_c = sparse_precision([
        :c => L_c_eval, :c_vert => L_c_vert, :c_horiz => L_c_horiz,
        :c_dx_vert => L_c_dx_vert, :c_dy_horiz => L_c_dy_horiz,
    ], k_c; ρ=ρ, ordering=:integrals_coarsest)

    approx_s = sparse_precision([
        :s => L_s_eval, :s_int => L_s_int,
    ], k_s; ρ=4.0, ordering=:integrals_coarsest)

    n_c = approx_c.info.n
    n_s = approx_s.info.n
    n_total = n_c + n_s

    # Scale prior precision by 1/output_scale (covariance scales by output_scale)
    Q_joint = blockdiag(sparse(approx_c.Q), sparse(approx_s.Q) / source_amplitude^2) / output_scale
    layout_c = approx_c.layout
    layout_s = approx_s.layout

    # FVM and BC constraint matrices (separate!)
    A_fvm, b_fvm = build_fvm_constraint(xs, ys, prob, layout_c, layout_s, n_c)
    A_bc, b_bc = build_boundary_constraints(xs, ys, prob, layout_c, n_c, n_total)
    n_fvm = size(A_fvm, 1)
    n_bc = size(A_bc, 1)

    # Observation matrix
    obs_x = data["obs_x"]; obs_y = data["obs_y"]
    obs_c = data["obs_c"]; noise_std = Float64(data["noise_std"])
    n_obs = length(obs_c)

    obs_ix = [argmin(abs.(xs .- ox)) for ox in obs_x]
    obs_iy = [argmin(abs.(ys .- oy)) for oy in obs_y]
    obs_indices = [indices(layout_c, :c)[(iy-1)*Nx + ix] for (ix, iy) in zip(obs_ix, obs_iy)]

    A_obs = spzeros(n_obs, n_total)
    for (i, idx) in enumerate(obs_indices)
        A_obs[i, idx] = 1.0
    end

    # Condition on PDE + BCs (model structure), LOO over real observations only
    Q_bc = (1.0 / constraint_noise^2) * sparse(I, n_bc, n_bc)
    Q_fvm = (1.0 / constraint_noise^2) * sparse(I, n_fvm, n_fvm)

    Q_model = Q_joint + A_bc' * Q_bc * A_bc + A_fvm' * Q_fvm * A_fvm
    rhs_model = A_bc' * Q_bc * b_bc  # b_fvm = 0, so no PDE rhs contribution

    # Factorize PDE+BC-conditioned prior
    F = cholesky(Symmetric(Q_model))
    μ_model = F \ rhs_model

    # LOO over the 10 real observations
    # Marginal distribution of obs under the PDE-conditioned prior:
    # y_obs ~ N(A_obs μ_model, A_obs Σ_model A_obs^T + σ²_obs I)
    z_obs = A_obs * μ_model  # predicted observations
    X = F \ Matrix(A_obs')   # Σ_model A_obs^T, via n_obs sparse solves
    S = A_obs * X + noise_std^2 * I(n_obs)  # 10×10 marginal covariance
    S = Symmetric(S)

    # LOO-CV (Rasmussen & Williams, Eq. 5.12)
    r = obs_c - z_obs  # observation residuals
    F_S = cholesky(S)
    α = F_S \ r
    S_inv_diag = diag(inv(F_S))

    # LOO pseudo-log-likelihood
    loo_obs = -0.5 * sum(i -> -log(S_inv_diag[i]) + α[i]^2 / S_inv_diag[i], 1:n_obs)

    # Also compute PDE residual norm for diagnostics
    pde_residual_norm = norm(A_fvm * μ_model)

    return (; loo_total=loo_obs, loo_pde=0.0, loo_obs, n_fvm, n_obs, pde_residual_norm)
end

# ==============================================================================
# Main experiment
# ==============================================================================

function run_loo_experiment(;
        problem_path::String, N::Int, ρ::Float64,
        lengthscales_c::Vector{Float64},
        lengthscales_s::Vector{Float64},
        output_dir::String)

    println("=" ^ 60)
    println("LOO Lengthscale Selection Experiment")
    println("=" ^ 60)

    prob = load_problem(problem_path)
    println(prob)

    # Generate data at this N
    xs, ys = setup_2d_grid(N, N, prob.domain)
    c_true, _ = solve_forward_problem(xs, ys, prob)
    obs_xs, obs_ys, _, obs_c = generate_observations(xs, ys, c_true, prob)

    data = Dict{String, Any}(
        "xs" => xs, "ys" => ys,
        "obs_x" => obs_xs, "obs_y" => obs_ys, "obs_c" => obs_c,
        "noise_std" => prob.noise_std,
        "source_x" => [s.x for s in prob.sources],
        "source_y" => [s.y for s in prob.sources],
    )

    Δ = min(xs[2]-xs[1], ys[2]-ys[1])
    n_c, n_s = length(lengthscales_c), length(lengthscales_s)
    println("\nGrid: $(N)×$(N), Δ = $(round(Δ, digits=4))")
    println("Default lengthscale (5Δ): $(round(5Δ, digits=4))")
    println("Sweeping: $(n_c) ℓ_c × $(n_s) ℓ_s = $(n_c * n_s) evaluations")
    println()

    # Warmup
    print("Warmup... ")
    compute_loo_score(prob, data; ρ=ρ,
        lengthscale_c=lengthscales_c[1], lengthscale_s=lengthscales_s[1])
    println("done.\n")

    # 2D sweep
    loo_total = zeros(n_c, n_s)
    loo_pde = zeros(n_c, n_s)
    loo_obs = zeros(n_c, n_s)
    count = 0

    for (j, ℓ_s) in enumerate(lengthscales_s)
        for (i, ℓ_c) in enumerate(lengthscales_c)
            count += 1
            t = @elapsed r = compute_loo_score(prob, data; ρ=ρ,
                lengthscale_c=ℓ_c, lengthscale_s=ℓ_s)
            loo_total[i, j] = r.loo_total
            loo_pde[i, j] = r.loo_pde
            loo_obs[i, j] = r.loo_obs
            @printf("[%3d/%d] ℓ_c=%.3f ℓ_s=%.3f | total=%8.1f pde=%8.1f obs=%7.1f | %.2fs\n",
                    count, n_c*n_s, ℓ_c, ℓ_s, r.loo_total, r.loo_pde, r.loo_obs, t)
        end
    end

    # Find optimum
    best = argmax(loo_total)
    best_ℓc = lengthscales_c[best[1]]
    best_ℓs = lengthscales_s[best[2]]

    println("\n" * "=" ^ 50)
    @printf("Best: ℓ_c = %.4f, ℓ_s = %.4f (LOO = %.1f)\n",
            best_ℓc, best_ℓs, loo_total[best])
    @printf("Default (5Δ): %.4f\n", 5Δ)
    println("=" ^ 50)

    # Plots
    mkpath(output_dir)

    # 1D slices through the optimum
    fig = Figure(size=(900, 350), fontsize=12)

    # Slice: vary ℓ_c at best ℓ_s
    ax1 = Axis(fig[1, 1]; xlabel="ℓ_c (concentration)", ylabel="LOO score",
               title="Vary ℓ_c (ℓ_s = $(round(best_ℓs, digits=3)))")
    scatterlines!(ax1, lengthscales_c, loo_total[:, best[2]];
        color=:black, label="Total", linewidth=2, markersize=5)
    scatterlines!(ax1, lengthscales_c, loo_pde[:, best[2]];
        color=:royalblue, label="PDE", linewidth=1.5, markersize=4)
    scatterlines!(ax1, lengthscales_c, loo_obs[:, best[2]];
        color=:crimson, label="Obs", linewidth=1.5, markersize=4)
    vlines!(ax1, [5Δ]; color=:gray50, linestyle=:dash, linewidth=1)
    axislegend(ax1; position=:rb)

    # Slice: vary ℓ_s at best ℓ_c
    ax2 = Axis(fig[1, 2]; xlabel="ℓ_s (source)", ylabel="LOO score",
               title="Vary ℓ_s (ℓ_c = $(round(best_ℓc, digits=3)))")
    scatterlines!(ax2, lengthscales_s, loo_total[best[1], :];
        color=:black, label="Total", linewidth=2, markersize=5)
    scatterlines!(ax2, lengthscales_s, loo_pde[best[1], :];
        color=:royalblue, label="PDE", linewidth=1.5, markersize=4)
    scatterlines!(ax2, lengthscales_s, loo_obs[best[1], :];
        color=:crimson, label="Obs", linewidth=1.5, markersize=4)
    vlines!(ax2, [5Δ]; color=:gray50, linestyle=:dash, linewidth=1)
    axislegend(ax2; position=:rb)

    filename = joinpath(output_dir, "loo_slices_N$(N).pdf")
    save(filename, fig, px_per_unit=3)
    println("\nSaved: $filename")

    # 2D heatmap of total LOO
    if n_c > 1 && n_s > 1
        fig2 = Figure(size=(500, 400), fontsize=12)
        ax = Axis(fig2[1, 1]; xlabel="ℓ_c (concentration)", ylabel="ℓ_s (source)",
                  title="LOO total score (N=$N)")
        hm = heatmap!(ax, lengthscales_c, lengthscales_s, loo_total; colormap=:viridis)
        scatter!(ax, [best_ℓc], [best_ℓs]; color=:red, markersize=10, marker=:star5)
        scatter!(ax, [5Δ], [5Δ]; color=:white, markersize=8, marker=:xcross, strokewidth=1.5)
        Colorbar(fig2[1, 2], hm; label="LOO score")

        filename2 = joinpath(output_dir, "loo_heatmap_N$(N).pdf")
        save(filename2, fig2, px_per_unit=3)
        println("Saved: $filename2")
    end

    return (; loo_total, loo_pde, loo_obs, lengthscales_c, lengthscales_s,
              best_ℓc, best_ℓs)
end

# ==============================================================================
# Output scale calibration
# ==============================================================================

function run_output_scale_experiment(;
        problem_path::String, N::Int, ρ::Float64,
        lengthscale_c::Float64, lengthscale_s::Float64,
        output_scales::Vector{Float64},
        output_dir::String)

    println("=" ^ 60)
    println("Output Scale Calibration via LOO on PDE Residuals")
    println("=" ^ 60)

    prob = load_problem(problem_path)
    println(prob)

    xs, ys = setup_2d_grid(N, N, prob.domain)
    c_true, _ = solve_forward_problem(xs, ys, prob)
    obs_xs, obs_ys, _, obs_c = generate_observations(xs, ys, c_true, prob)

    data = Dict{String, Any}(
        "xs" => xs, "ys" => ys,
        "obs_x" => obs_xs, "obs_y" => obs_ys, "obs_c" => obs_c,
        "noise_std" => prob.noise_std,
        "source_x" => [s.x for s in prob.sources],
        "source_y" => [s.y for s in prob.sources],
    )

    println("\nGrid: $(N)×$(N)")
    println("Fixed lengthscales: ℓ_c = $lengthscale_c, ℓ_s = $lengthscale_s")
    println("Sweeping $(length(output_scales)) output scale values")
    println()

    # Warmup
    print("Warmup... ")
    compute_loo_score(prob, data; ρ=ρ, lengthscale_c=lengthscale_c,
        lengthscale_s=lengthscale_s, output_scale=output_scales[1])
    println("done.\n")

    scores_total = Float64[]
    scores_pde = Float64[]
    scores_obs = Float64[]

    for (i, σ²) in enumerate(output_scales)
        t = @elapsed r = compute_loo_score(prob, data; ρ=ρ,
            lengthscale_c=lengthscale_c, lengthscale_s=lengthscale_s,
            output_scale=σ²)
        push!(scores_total, r.loo_total)
        push!(scores_pde, r.loo_pde)
        push!(scores_obs, r.loo_obs)
        @printf("[%2d/%d] σ² = %8.4f | total=%9.1f pde=%9.1f obs=%7.1f | %.2fs\n",
                i, length(output_scales), σ², r.loo_total, r.loo_pde, r.loo_obs, t)
    end

    best_idx = argmax(scores_total)
    best_σ² = output_scales[best_idx]

    println("\n" * "=" ^ 50)
    @printf("Best output scale: σ² = %.4f (LOO = %.1f)\n", best_σ², scores_total[best_idx])
    println("=" ^ 50)

    # Plot
    mkpath(output_dir)
    fig = Figure(size=(550, 400), fontsize=12)
    ax = Axis(fig[1, 1];
        xlabel="Output scale σ²",
        ylabel="LOO pseudo-log-likelihood",
        xscale=log10,
        title="Output scale calibration (N=$N, ℓ_c=$lengthscale_c, ℓ_s=$lengthscale_s)",
    )

    scatterlines!(ax, output_scales, scores_total;
        color=:black, label="Total", linewidth=2, markersize=6)
    scatterlines!(ax, output_scales, scores_pde;
        color=:royalblue, label="PDE", linewidth=1.5, markersize=5)
    scatterlines!(ax, output_scales, scores_obs;
        color=:crimson, label="Obs", linewidth=1.5, markersize=5)
    vlines!(ax, [best_σ²]; color=:crimson, linestyle=:dash, linewidth=1,
            label="LOO optimum")
    vlines!(ax, [1.0]; color=:gray50, linestyle=:dash, linewidth=1,
            label="Default (σ²=1)")
    axislegend(ax; position=:rb)

    filename = joinpath(output_dir, "output_scale_N$(N).pdf")
    save(filename, fig, px_per_unit=3)
    println("\nSaved: $filename")

    return (; output_scales, scores_total, scores_pde, scores_obs, best_σ²)
end

# ==============================================================================
# CLI
# ==============================================================================

function parse_loo_args()
    s = ArgParseSettings(description = "LOO lengthscale selection on PDE residuals")
    @add_arg_table! s begin
        "--problem", "-p"
            help = "Path to problem TOML"
            default = joinpath(@__DIR__, "problems", "default.toml")
        "-N"
            help = "Grid size"
            arg_type = Int
            default = 31
        "--rho"
            help = "Sparsity parameter"
            arg_type = Float64
            default = 2.0
        "--n-ls"
            help = "Number of lengthscale values per axis"
            arg_type = Int
            default = 10
        "--ls-c-min"
            help = "Min lengthscale for concentration"
            arg_type = Float64
            default = 0.05
        "--ls-c-max"
            help = "Max lengthscale for concentration"
            arg_type = Float64
            default = 0.5
        "--ls-s-min"
            help = "Min lengthscale for source"
            arg_type = Float64
            default = 0.05
        "--ls-s-max"
            help = "Max lengthscale for source"
            arg_type = Float64
            default = 0.5
        "--output-dir", "-o"
            help = "Output directory"
            default = joinpath(@__DIR__, "results", "loo")
        "--mode"
            help = "Mode: 'lengthscale' or 'output-scale'"
            default = "output-scale"
        "--lengthscale-c"
            help = "Fixed ℓ_c for output-scale mode"
            arg_type = Float64
            default = 0.2
        "--lengthscale-s"
            help = "Fixed ℓ_s for output-scale mode"
            arg_type = Float64
            default = 0.12
        "--n-sigma"
            help = "Number of output scale values to test"
            arg_type = Int
            default = 15
        "--sigma-min"
            help = "Min output scale (log10)"
            arg_type = Float64
            default = -3.0
        "--sigma-max"
            help = "Max output scale (log10)"
            arg_type = Float64
            default = 2.0
    end
    return parse_args(s)
end

function loo_main()
    args = parse_loo_args()

    if args["mode"] == "output-scale"
        σ²s = 10 .^ range(args["sigma-min"], args["sigma-max"], length=args["n-sigma"])
        run_output_scale_experiment(
            problem_path = args["problem"],
            N = args["N"],
            ρ = args["rho"],
            lengthscale_c = args["lengthscale-c"],
            lengthscale_s = args["lengthscale-s"],
            output_scales = collect(σ²s),
            output_dir = args["output-dir"],
        )
    else
        n = args["n-ls"]
        ℓs_c = collect(range(args["ls-c-min"], args["ls-c-max"], length=n))
        ℓs_s = collect(range(args["ls-s-min"], args["ls-s-max"], length=n))
        run_loo_experiment(
            problem_path = args["problem"],
            N = args["N"],
            ρ = args["rho"],
            lengthscales_c = ℓs_c,
            lengthscales_s = ℓs_s,
            output_dir = args["output-dir"],
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    loo_main()
end
