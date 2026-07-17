"""
Posterior adaptation figure: shows how samples adapt when observations are added.

Creates a 2x4 figure where:
- Rows: different observation sets (N_obs=12, N_obs=15)
- Columns: four posterior samples (s_1, s_2, s_3, s_4 with different seeds)
- Key: Same seed is used for both rows in each column

This demonstrates that the "same random draw" from the posterior produces
different spatial patterns when the observations change, illustrating how
uncertainty quantification adapts to available data.

Usage:
    julia --project=../.. figure_posterior_adaptation.jl
    julia --project=../.. figure_posterior_adaptation.jl --seeds 42,43,44,45 --publication
"""

using LinearAlgebra, SparseArrays
using ArgParse
using NPZ
using CairoMakie
using TuePlots
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields

import GaussianMarkovRandomFields: mean, std, rand
import FunctionalGPs: ⊗

include("problem.jl")

function parse_commandline()
    s = ArgParseSettings(description = "Posterior adaptation figure")

    @add_arg_table! s begin
        "--seeds"
            help = "Comma-separated seeds for samples (e.g., '42,43,44,45')"
            arg_type = String
            default = "42,43,44,45"
        "--rho"
            help = "Sparsity parameter"
            arg_type = Float64
            default = 4.0
        "--lengthscale-c"
            help = "Concentration lengthscale (default: 5*Δx)"
            arg_type = Float64
            default = -1.0
        "--lengthscale-s"
            help = "Source lengthscale (default: 5*Δx)"
            arg_type = Float64
            default = -1.0
        "--output", "-o"
            help = "Output file path"
            arg_type = String
            default = ""
        "--publication"
            help = "Use TuePlots ICML settings for publication-ready output"
            action = :store_true
    end

    return parse_args(s)
end

# Constraint builders (copied from visualize_samples.jl)
function build_fvm_constraint(xs, ys, prob::SourceIdentificationProblem,
                               layout_c, layout_s, n_c)
    Nx, Ny = length(xs), length(ys)
    n_cells_x, n_cells_y = Nx - 1, Ny - 1
    n_cells = n_cells_x * n_cells_y

    n_s = layout_s.total
    n_total = n_c + n_s
    s_offset = n_c

    vert_idx(i, j) = (j-1)*Nx + i
    horiz_idx(i, j) = (j-1)*n_cells_x + i

    c_vert_base = first(indices(layout_c, :c_vert))
    c_horiz_base = first(indices(layout_c, :c_horiz))
    c_dx_vert_base = first(indices(layout_c, :c_dx_vert))
    c_dy_horiz_base = first(indices(layout_c, :c_dy_horiz))

    c_vert(i, j) = c_vert_base + vert_idx(i, j) - 1
    c_horiz(i, j) = c_horiz_base + horiz_idx(i, j) - 1
    c_dx_vert(i, j) = c_dx_vert_base + vert_idx(i, j) - 1
    c_dy_horiz(i, j) = c_dy_horiz_base + horiz_idx(i, j) - 1

    s_int_idx(ci, cj) = s_offset + indices(layout_s, :s_int)[(cj-1)*n_cells_x + ci]

    constraints = []

    for cj in 1:n_cells_y
        for ci in 1:n_cells_x
            row = spzeros(n_total)

            row[c_vert(ci+1, cj)] += prob.vx
            row[c_vert(ci, cj)] -= prob.vx

            if prob.vy != 0
                row[c_horiz(ci, cj+1)] += prob.vy
                row[c_horiz(ci, cj)] -= prob.vy
            end

            row[c_dx_vert(ci+1, cj)] -= prob.D
            row[c_dx_vert(ci, cj)] += prob.D
            row[c_dy_horiz(ci, cj+1)] -= prob.D
            row[c_dy_horiz(ci, cj)] += prob.D

            row[s_int_idx(ci, cj)] = -1.0

            push!(constraints, row)
        end
    end

    A_fvm = vcat([reshape(r, 1, :) for r in constraints]...)
    b_fvm = zeros(n_cells)

    return sparse(A_fvm), b_fvm
end

function build_boundary_constraints(xs, ys, prob::SourceIdentificationProblem,
                                     layout_c, n_c, n_total)
    Nx, Ny = length(xs), length(ys)
    n_cells_x, n_cells_y = Nx - 1, Ny - 1

    c_eval_idx(i, j) = indices(layout_c, :c)[(j-1)*Nx + i]

    vert_idx(i, j) = (j-1)*Nx + i
    horiz_idx(i, j) = (j-1)*n_cells_x + i

    c_dx_vert_base = first(indices(layout_c, :c_dx_vert))
    c_dy_horiz_base = first(indices(layout_c, :c_dy_horiz))

    c_dx_vert(i, j) = c_dx_vert_base + vert_idx(i, j) - 1
    c_dy_horiz(i, j) = c_dy_horiz_base + horiz_idx(i, j) - 1

    constraints = []
    rhs = Float64[]

    for j in 1:Ny
        row = spzeros(n_total)
        row[c_eval_idx(1, j)] = 1.0
        push!(constraints, row)
        push!(rhs, prob.c_inflow)
    end

    for j in 1:n_cells_y
        row = spzeros(n_total)
        row[c_dx_vert(Nx, j)] = 1.0
        push!(constraints, row)
        push!(rhs, 0.0)
    end

    for i in 1:n_cells_x
        row = spzeros(n_total)
        row[c_dy_horiz(i, 1)] = 1.0
        push!(constraints, row)
        push!(rhs, 0.0)
    end

    for i in 1:n_cells_x
        row = spzeros(n_total)
        row[c_dy_horiz(i, Ny)] = 1.0
        push!(constraints, row)
        push!(rhs, 0.0)
    end

    A_bc = vcat([reshape(r, 1, :) for r in constraints]...)
    return sparse(A_bc), rhs
end

"""
Run inference and return posterior GMRF with layout info for sampling.
"""
function run_inference_for_sampling(prob::SourceIdentificationProblem, data::Dict;
                                     ρ=4.0, lengthscale_c=nothing, lengthscale_s=nothing, verbose=true)
    xs = collect(Float64, data["xs"])
    ys = collect(Float64, data["ys"])
    Nx, Ny = length(xs), length(ys)
    n_cells_x, n_cells_y = Nx - 1, Ny - 1
    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]
    Δ = min(Δx, Δy)

    lengthscale_c = isnothing(lengthscale_c) ? 5 * Δ : lengthscale_c
    lengthscale_s = isnothing(lengthscale_s) ? 5 * Δ : lengthscale_s

    verbose && println("  Grid: $(Nx)×$(Ny), lengthscales: c=$(round(lengthscale_c, digits=4)), s=$(round(lengthscale_s, digits=4))")

    x_intervals = intervals_from_endpoints(collect(xs))
    y_intervals = intervals_from_endpoints(collect(ys))

    grid = FactorizedGrid(xs, ys)
    cells_2d = x_intervals ⊗ y_intervals

    k_c = HalfIntegerMaternKernel(2, [lengthscale_c]) ⊗ HalfIntegerMaternKernel(2, [lengthscale_c])
    k_s = HalfIntegerMaternKernel(2, [lengthscale_s]) ⊗ HalfIntegerMaternKernel(2, [lengthscale_s])

    L_c_eval = EvaluationFunctional(grid)
    L_c_vert = EvaluationFunctional(xs) ⊗ VectorizedLebesgueIntegral(y_intervals)
    L_c_horiz = VectorizedLebesgueIntegral(x_intervals) ⊗ EvaluationFunctional(ys)
    L_c_dx_vert = L_c_vert ∘ PartialDerivative((1, 0))
    L_c_dy_horiz = L_c_horiz ∘ PartialDerivative((0, 1))

    L_s_eval = EvaluationFunctional(grid)
    L_s_int = VectorizedLebesgueIntegral(cells_2d)

    approx_c = sparse_precision([
        :c => L_c_eval,
        :c_vert => L_c_vert,
        :c_horiz => L_c_horiz,
        :c_dx_vert => L_c_dx_vert,
        :c_dy_horiz => L_c_dy_horiz,
    ], k_c; ρ=ρ, ordering=:integrals_coarsest)

    approx_s = sparse_precision([
        :s => L_s_eval,
        :s_int => L_s_int,
    ], k_s; ρ=4.0, ordering=:integrals_coarsest)

    n_c = approx_c.info.n
    n_s = approx_s.info.n
    n_total = n_c + n_s

    Q_joint = blockdiag(sparse(approx_c.Q), sparse(approx_s.Q))
    layout_c = approx_c.layout
    layout_s = approx_s.layout

    A_fvm, b_fvm = build_fvm_constraint(xs, ys, prob, layout_c, layout_s, n_c)
    A_bc, b_bc = build_boundary_constraints(xs, ys, prob, layout_c, n_c, n_total)

    A_constraints = vcat(A_fvm, A_bc)
    b_constraints = vcat(b_fvm, b_bc)
    n_constraints = size(A_constraints, 1)
    Q_constraints = 1e8 * sparse(I, n_constraints, n_constraints)

    obs_x = data["obs_x"]
    obs_y = data["obs_y"]
    obs_c = data["obs_c"]
    n_obs = length(obs_c)
    noise_std = Float64(data["noise_std"])

    obs_ix = [argmin(abs.(xs .- ox)) for ox in obs_x]
    obs_iy = [argmin(abs.(ys .- oy)) for oy in obs_y]
    obs_indices = [indices(layout_c, :c)[(iy-1)*Nx + ix] for (ix, iy) in zip(obs_ix, obs_iy)]

    A_obs = spzeros(n_obs, n_total)
    for (i, idx) in enumerate(obs_indices)
        A_obs[i, idx] = 1.0
    end
    Q_obs = (1.0 / noise_std^2) * sparse(I, n_obs, n_obs)

    x_posterior = condition_precision(Q_joint, [
        (A=A_constraints, Q_ϵ=Q_constraints, y=b_constraints),
        (A=A_obs, Q_ϵ=Q_obs, y=obs_c),
    ])

    return (
        posterior = x_posterior,
        layout_c = layout_c,
        layout_s = layout_s,
        n_c = n_c,
        xs = xs,
        ys = ys,
        Nx = Nx,
        Ny = Ny,
    )
end

function main()
    args = parse_commandline()

    # Parse seeds
    seeds = parse.(Int, split(args["seeds"], ","))
    @assert length(seeds) == 4 "Need exactly 4 seeds for 4 columns"
    println("Seeds: $seeds")

    # Paths
    base_dir = @__DIR__
    prob_base_path = joinpath(base_dir, "problems", "two_sources.toml")
    prob_extra_path = joinpath(base_dir, "problems", "two_sources_extra_obs.toml")
    data_base_path = joinpath(base_dir, "data", "two_sources.npz")
    data_extra_path = joinpath(base_dir, "data", "two_sources_extra_obs.npz")

    # Load problems and data
    println("\nLoading problems and data...")
    prob_base = load_problem(prob_base_path)
    prob_extra = load_problem(prob_extra_path)
    data_base = npzread(data_base_path)
    data_extra = npzread(data_extra_path)

    # Hyperparameters
    ls_c = args["lengthscale-c"] < 0 ? nothing : args["lengthscale-c"]
    ls_s = args["lengthscale-s"] < 0 ? nothing : args["lengthscale-s"]

    # Run inference for both
    println("\nRunning inference for base problem (12 observations)...")
    result_base = run_inference_for_sampling(prob_base, data_base;
        ρ=args["rho"], lengthscale_c=ls_c, lengthscale_s=ls_s)

    println("\nRunning inference for extra observations (15 observations)...")
    result_extra = run_inference_for_sampling(prob_extra, data_extra;
        ρ=args["rho"], lengthscale_c=ls_c, lengthscale_s=ls_s)

    # Extract layout info
    Nx, Ny = result_base.Nx, result_base.Ny
    xs, ys = result_base.xs, result_base.ys
    s_eval_indices_base = result_base.n_c .+ indices(result_base.layout_s, :s)
    s_eval_indices_extra = result_extra.n_c .+ indices(result_extra.layout_s, :s)

    # Draw samples with matching seeds
    println("\nDrawing samples...")
    samples_base = Matrix{Float64}[]
    samples_extra = Matrix{Float64}[]

    for seed in seeds
        # Sample from base posterior
        Random.seed!(seed)
        sample_base = rand(result_base.posterior)
        s_base = reshape(sample_base[s_eval_indices_base], Nx, Ny)
        push!(samples_base, s_base)

        # Sample from extra posterior with SAME seed
        Random.seed!(seed)
        sample_extra = rand(result_extra.posterior)
        s_extra = reshape(sample_extra[s_eval_indices_extra], Nx, Ny)
        push!(samples_extra, s_extra)

        println("  Seed $seed: base s ∈ [$(round(minimum(s_base), digits=3)), $(round(maximum(s_base), digits=3))], " *
                "extra s ∈ [$(round(minimum(s_extra), digits=3)), $(round(maximum(s_extra), digits=3))]")
    end

    # Observation coordinates
    obs_x_base = data_base["obs_x"]
    obs_y_base = data_base["obs_y"]
    obs_x_extra = data_extra["obs_x"]
    obs_y_extra = data_extra["obs_y"]

    # Identify extra observations (last 3 in extra problem)
    n_base_obs = length(obs_x_base)
    extra_obs_x = obs_x_extra[n_base_obs+1:end]
    extra_obs_y = obs_y_extra[n_base_obs+1:end]

    # True source locations
    source_x = data_base["source_x"]
    source_y = data_base["source_y"]
    n_sources = Int(data_base["n_sources"])

    # Compute shared color range
    s_min = min(minimum.(samples_base)..., minimum.(samples_extra)...)
    s_max = max(maximum.(samples_base)..., maximum.(samples_extra)...)
    s_range = (s_min, s_max)

    # Custom diverging colormap: cold → black (at zero) → hot
    # Compute where zero falls in the normalized range [0, 1]
    zero_pos = clamp(-s_min / (s_max - s_min), 0.01, 0.99)

    # Sample colors from existing colormaps
    hot_cmap = cgrad(:hot)
    cold_cmap = cgrad(:ice)  # blues, reversed below

    # Build combined colormap:
    # - Negative region: sample from cold scheme (light blue → dark blue → black)
    # - Positive region: sample from hot scheme (black → red → yellow)
    n_neg = 50
    n_pos = 100
    neg_colors = [cold_cmap[1.0 - t] for t in range(0, 1, length=n_neg)]  # reversed
    pos_colors = [hot_cmap[t] for t in range(0, 1, length=n_pos)]

    # Combine: neg_colors ends near black, pos_colors starts at black
    all_colors = vcat(neg_colors[1:end-1], pos_colors)
    n_total = length(all_colors)

    # Create positions so that black lands at zero_pos
    neg_positions = range(0, zero_pos, length=n_neg)[1:end-1]
    pos_positions = range(zero_pos, 1, length=n_pos)
    all_positions = vcat(collect(neg_positions), collect(pos_positions))

    custom_cmap = cgrad(all_colors, all_positions)

    # Create figure
    println("\nCreating figure...")
    publication_mode = args["publication"]

    if publication_mode
        theme = Theme(
            TuePlots.SETTINGS[:ICML];
            font=true,
            fontsize=true,
            figsize=true,
            single_column=true,
            nrows=2, ncols=5,  # 2x4 + colorbar
            subplot_height_to_width_ratio=1.0,  # square-ish panels
        )
        set_theme!(theme)
        fig = Figure()
    else
        fig = Figure(size=(900, 450), fontsize=11)
    end

    # Marker styling
    if publication_mode
        obs_marker = (color=:white, markersize=3, strokewidth=0.3, strokecolor=:black)
        extra_obs_marker = (color=:cyan, markersize=4, marker=:diamond, strokewidth=0.3, strokecolor=:black)
        src_marker = (color=:cyan, markersize=6, marker=:star5, strokewidth=0.3, strokecolor=:black)
    else
        obs_marker = (color=:white, markersize=5, strokewidth=0.5, strokecolor=:black)
        extra_obs_marker = (color=:cyan, markersize=6, marker=:diamond, strokewidth=0.5, strokecolor=:black)
        src_marker = (color=:cyan, markersize=8, marker=:star5, strokewidth=0.5, strokecolor=:black)
    end

    # Row 1: Base observations (12)
    for (col, s_sample) in enumerate(samples_base)
        show_ylabel = col == 1
        ax = Axis(fig[1, col]; aspect=DataAspect(),
            ylabel = show_ylabel ? L"k = 12" : "",
            title = L"s_%$col",
            xticklabelsvisible=false,
            yticklabelsvisible=false)
        hm = heatmap!(ax, xs, ys, s_sample, colormap=custom_cmap, colorrange=s_range)
        scatter!(ax, obs_x_base, obs_y_base; obs_marker...)
    end

    # Row 2: Extra observations (15)
    for (col, s_sample) in enumerate(samples_extra)
        show_ylabel = col == 1
        ax = Axis(fig[2, col]; aspect=DataAspect(),
            ylabel = show_ylabel ? L"k = 15" : "",
            xticklabelsvisible=false,
            yticklabelsvisible=false)
        hm = heatmap!(ax, xs, ys, s_sample, colormap=custom_cmap, colorrange=s_range)
        # Base observations
        scatter!(ax, obs_x_base, obs_y_base; obs_marker...)
        # Extra observations (highlighted)
        scatter!(ax, extra_obs_x, extra_obs_y; extra_obs_marker...)
    end

    # Shared colorbar
    Colorbar(fig[1:2, 5], limits=s_range, colormap=custom_cmap, label=L"Source $s$")

    # Layout adjustments
    cbar_width = publication_mode ? 10 : 25
    colsize!(fig.layout, 5, Fixed(cbar_width))
    rowsize!(fig.layout, 1, Relative(0.5))
    rowsize!(fig.layout, 2, Relative(0.5))

    # Tight row gap, consistent column gaps between sample columns
    rowgap!(fig.layout, publication_mode ? 5 : 8)
    sample_gap = publication_mode ? 8 : 15
    colgap!(fig.layout, 1, sample_gap)  # between s_1 and s_2
    colgap!(fig.layout, 2, sample_gap)  # between s_2 and s_3
    colgap!(fig.layout, 3, sample_gap)  # between s_3 and s_4
    colgap!(fig.layout, 4, publication_mode ? 8 : 12)  # before colorbar

    # Save
    output_dir = joinpath(base_dir, "figures")
    mkpath(output_dir)

    if !isempty(args["output"])
        output_path = args["output"]
    elseif publication_mode
        output_path = joinpath(output_dir, "posterior_adaptation.pdf")
    else
        output_path = joinpath(output_dir, "posterior_adaptation.png")
    end

    if publication_mode
        save(output_path, fig, pt_per_unit=1)
    else
        save(output_path, fig, px_per_unit=2)
    end
    println("\nSaved: $output_path")

    if publication_mode
        set_theme!()
    end

    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
