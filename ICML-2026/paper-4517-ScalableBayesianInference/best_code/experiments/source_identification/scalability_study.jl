"""
Scalability study for GP-FVM source identification.

Sweeps over grid sizes N and reports quantitative timing + error results
on the 2D advection-diffusion source-identification problem.

Usage:
    # Quick test
    julia --project=../.. scalability_study.jl --grid-sizes 11,21,31 --benchmark-runs 0 --no-plot

    # Full run for paper
    julia --project=../.. scalability_study.jl --grid-sizes 11,16,21,26,31,41,51,61,81,101

    # Custom problem
    julia --project=../.. scalability_study.jl -p problems/two_sources.toml -N 16,31,51
"""

using LinearAlgebra, SparseArrays
using Random
using ArgParse
using NPZ
using CSV, DataFrames
using Statistics
using Printf
using CairoMakie
using TuePlots
using SpecialFunctions

push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields

import GaussianMarkovRandomFields: mean, std
import FunctionalGPs: ⊗

# Include problem types and GP-FVM solver (run_gpfvm.jl includes problem.jl)
include("run_gpfvm.jl")

# ==============================================================================
# Ground truth generation (from generate_data.jl, inlined to avoid double-include)
# ==============================================================================

function setup_2d_grid(Nx, Ny, domain)
    x_min, x_max, y_min, y_max = domain
    xs = range(x_min, x_max, length=Nx)
    ys = range(y_min, y_max, length=Ny)
    return collect(xs), collect(ys)
end

"""
Solve forward advection-diffusion to get ground-truth concentration.
Node-centered FVM with upwind advection.
"""
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

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    b = zeros(n_nodes)

    for j in 1:Ny
        for i in 1:Nx
            idx = node_idx(i, j)

            if i == 1
                push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
                b[idx] = prob.c_inflow
            elseif i == Nx
                push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
                push!(rows, idx); push!(cols, node_idx(i-1, j)); push!(vals, -1.0)
                b[idx] = 0.0
            elseif j == 1
                push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
                push!(rows, idx); push!(cols, node_idx(i, j+1)); push!(vals, -1.0)
                b[idx] = 0.0
            elseif j == Ny
                push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
                push!(rows, idx); push!(cols, node_idx(i, j-1)); push!(vals, -1.0)
                b[idx] = 0.0
            else
                adv_coef_center = prob.vx * Δy
                adv_coef_left = -prob.vx * Δy
                push!(rows, idx); push!(cols, idx); push!(vals, adv_coef_center)
                push!(rows, idx); push!(cols, node_idx(i-1, j)); push!(vals, adv_coef_left)

                diff_coef = prob.D / Δx * Δy
                diff_coef_y = prob.D / Δy * Δx
                center_diff = 2 * diff_coef + 2 * diff_coef_y
                push!(rows, idx); push!(cols, idx); push!(vals, center_diff)
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
    end

    A = sparse(rows, cols, vals, n_nodes, n_nodes)
    c_vec = A \ b
    c_true = reshape(c_vec, Nx, Ny)

    return c_true, s_int_true
end

function generate_observations(xs, ys, c_true, prob::SourceIdentificationProblem)
    Nx, Ny = length(xs), length(ys)
    Random.seed!(prob.noise_seed)

    obs_xs, obs_ys = observation_coords(prob)
    n_obs = length(obs_xs)

    obs_ix = [argmin(abs.(xs .- ox)) for ox in obs_xs]
    obs_iy = [argmin(abs.(ys .- oy)) for oy in obs_ys]

    true_c_obs = [c_true[ix, iy] for (ix, iy) in zip(obs_ix, obs_iy)]
    noisy_obs = true_c_obs .+ prob.noise_std * randn(n_obs)

    return obs_xs, obs_ys, true_c_obs, noisy_obs
end

# ==============================================================================
# Bilinear interpolation
# ==============================================================================

"""
Bilinear interpolation of a 2D field from one regular grid to another.
"""
function interpolate_field(field::Matrix, xs_from, ys_from, xs_to, ys_to)
    Nx_to, Ny_to = length(xs_to), length(ys_to)
    result = zeros(Nx_to, Ny_to)

    for jt in 1:Ny_to
        yt = ys_to[jt]
        jf = searchsortedlast(ys_from, yt)
        jf = clamp(jf, 1, length(ys_from) - 1)
        ty = (yt - ys_from[jf]) / (ys_from[jf+1] - ys_from[jf])
        ty = clamp(ty, 0.0, 1.0)

        for it in 1:Nx_to
            xt = xs_to[it]
            if_idx = searchsortedlast(xs_from, xt)
            if_idx = clamp(if_idx, 1, length(xs_from) - 1)
            tx = (xt - xs_from[if_idx]) / (xs_from[if_idx+1] - xs_from[if_idx])
            tx = clamp(tx, 0.0, 1.0)

            result[it, jt] = (1 - tx) * (1 - ty) * field[if_idx, jf] +
                              tx       * (1 - ty) * field[if_idx+1, jf] +
                              (1 - tx) * ty       * field[if_idx, jf+1] +
                              tx       * ty       * field[if_idx+1, jf+1]
        end
    end
    return result
end

# ==============================================================================
# Reference solution
# ==============================================================================

"""
Generate high-resolution reference solution and fixed observations.
"""
function generate_reference(prob::SourceIdentificationProblem, N_ref::Int)
    xs, ys = setup_2d_grid(N_ref, N_ref, prob.domain)
    c_true, s_int_true = solve_forward_problem(xs, ys, prob)
    s_true = [evaluate_source(prob, x, y) for x in xs, y in ys]

    # Generate observations from reference solution (fixed across all N)
    obs_xs, obs_ys, _, obs_c = generate_observations(xs, ys, c_true, prob)

    return (;
        xs, ys, c_true, s_int_true, s_true,
        obs_x=obs_xs, obs_y=obs_ys, obs_c=obs_c,
        noise_std=prob.noise_std,
        source_x=[s.x for s in prob.sources],
        source_y=[s.y for s in prob.sources],
    )
end

# ==============================================================================
# Prepare data dict for a given N
# ==============================================================================

"""
Build the data Dict expected by `solve_source_identification` for grid size N,
using observations from the high-res reference.
"""
function prepare_data_dict(prob::SourceIdentificationProblem, ref, N::Int)
    xs, ys = setup_2d_grid(N, N, prob.domain)

    Dict{String, Any}(
        "xs" => xs,
        "ys" => ys,
        "obs_x" => ref.obs_x,
        "obs_y" => ref.obs_y,
        "obs_c" => ref.obs_c,
        "noise_std" => ref.noise_std,
        "source_x" => ref.source_x,
        "source_y" => ref.source_y,
    )
end

# ==============================================================================
# Run single grid size
# ==============================================================================

function run_single_N(N::Int, prob::SourceIdentificationProblem, ref;
                      ρ::Float64, benchmark_runs::Int, verbose::Bool,
                      lengthscale_c::Union{Float64, Nothing}=nothing,
                      lengthscale_s::Union{Float64, Nothing}=nothing,
                      output_scale::Float64=1.0,
                      output_scale_c::Union{Float64, Nothing}=nothing)
    data = prepare_data_dict(prob, ref, N)
    xs = data["xs"]
    ys = data["ys"]

    solver_kw = (ρ=ρ, verbose=false, lengthscale_c=lengthscale_c, lengthscale_s=lengthscale_s,
                  output_scale=output_scale, output_scale_c=output_scale_c)

    # Warmup run (JIT compilation)
    if benchmark_runs > 0
        verbose && print("  Warmup... ")
        solve_source_identification(prob, data; solver_kw...)
        verbose && println("done.")
    end

    # Timed runs
    n_runs = max(1, benchmark_runs)
    all_timings = Vector{Dict{String, Float64}}()
    local result

    for i in 1:n_runs
        result = solve_source_identification(prob, data; solver_kw...)
        push!(all_timings, result.info.timings)
    end

    # Best timings across runs
    best_timings = Dict{String, Float64}()
    for key in keys(all_timings[1])
        best_timings[key] = minimum(t[key] for t in all_timings)
    end

    # Source error: GP-FVM vs analytic source field
    Nx, Ny = length(xs), length(ys)
    s_true_analytic = [evaluate_source(prob, xs[i], ys[j]) for i in 1:Nx, j in 1:Ny]
    rmse_s = sqrt(Statistics.mean((result.s_mean .- s_true_analytic).^2))

    # Concentration error: GP-FVM vs high-res reference (interpolated to this grid)
    c_ref_interp = interpolate_field(ref.c_true, ref.xs, ref.ys, xs, ys)
    rmse_c_vs_ref = sqrt(Statistics.mean((result.c_mean .- c_ref_interp).^2))

    # Marginal z-scores (pointwise, fast)
    z_c = (c_ref_interp .- result.c_mean) ./ result.c_std
    z_s = (s_true_analytic .- result.s_mean) ./ result.s_std
    z_c_mean = Statistics.mean(z_c)
    z_c_std = Statistics.std(z_c)
    z_s_mean = Statistics.mean(z_s)
    z_s_std = Statistics.std(z_s)

    return (
        N = N,
        n_cells = (Nx - 1) * (Ny - 1),
        n_total = result.info.n_total,
        time_total_s = best_timings["total"],
        time_sparse_prec_c_s = best_timings["sparse_prec_c"],
        time_sparse_prec_s_s = best_timings["sparse_prec_s"],
        time_conditioning_s = best_timings["conditioning"],
        time_posterior_stats_s = best_timings["posterior_stats"],
        rmse_s = rmse_s,
        rmse_c_vs_ref = rmse_c_vs_ref,
        location_error = result.info.location_error,
        fill_c_pct = result.info.fill_c,
        fill_s_pct = result.info.fill_s,
        z_c_mean = z_c_mean, z_c_std = z_c_std,
        z_s_mean = z_s_mean, z_s_std = z_s_std,
        ρ = ρ,
    )
end

# ==============================================================================
# Plotting
# ==============================================================================

function paper_theme()
    Theme(
        fontsize = 12,
        Axis = (
            xlabelsize = 14, ylabelsize = 14, titlesize = 14,
            xticklabelsize = 11, yticklabelsize = 11,
        ),
        Legend = (framevisible = false, labelsize = 11, patchsize = (20, 10))
    )
end

const TIMING_COLORS = Dict(
    "total" => :black,
    "sparse_prec_c" => :royalblue,
    "sparse_prec_s" => :seagreen,
    "conditioning" => :darkorange,
    "posterior_stats" => :purple,
)

const TIMING_LABELS = Dict(
    "total" => "Total",
    "sparse_prec_c" => "Sparse prec. (c)",
    "sparse_prec_s" => "Sparse prec. (s)",
    "conditioning" => "Conditioning",
    "posterior_stats" => "Posterior stats",
)

function plot_timing(df::DataFrame; filename::Union{Nothing,String}=nothing)
    set_theme!(paper_theme())
    fig = Figure(size=(550, 400))

    ax = Axis(fig[1, 1];
        xlabel="Grid points per side (N)",
        ylabel="Wall-clock time (s)",
        xscale=log10, yscale=log10,
        title="Timing vs grid resolution",
    )

    Ns = df.N

    # Timing components
    for (col, key) in [
        (:time_total_s, "total"),
        (:time_sparse_prec_c_s, "sparse_prec_c"),
        (:time_conditioning_s, "conditioning"),
        (:time_posterior_stats_s, "posterior_stats"),
    ]
        lw = key == "total" ? 2.5 : 1.5
        scatterlines!(ax, Ns, df[!, col];
            color=TIMING_COLORS[key], label=TIMING_LABELS[key],
            linewidth=lw, markersize=6)
    end

    axislegend(ax; position=:lt)

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig, px_per_unit=3)
        println("  Saved: $filename")
    end

    set_theme!()
    return fig
end

function plot_error(df::DataFrame; filename::Union{Nothing,String}=nothing)
    set_theme!(paper_theme())
    fig = Figure(size=(550, 400))

    ax = Axis(fig[1, 1];
        xlabel="Grid points per side (N)",
        ylabel="Error",
        xscale=log10, yscale=log10,
        title="Error vs grid resolution",
    )

    Ns = df.N

    # Error curves (no convergence slopes — this is an inverse problem,
    # error is bounded by observation information, not grid resolution)
    scatterlines!(ax, Ns, df.rmse_s;
        color=:crimson, label="Source RMSE", linewidth=2, markersize=6)
    scatterlines!(ax, Ns, df.rmse_c_vs_ref;
        color=:royalblue, label="Concentration RMSE (vs ref)", linewidth=2, markersize=6)

    # Location error (may have zeros at fine grids — filter)
    valid = df.location_error .> 0
    if any(valid)
        scatterlines!(ax, Ns[valid], df.location_error[valid];
            color=:seagreen, label="Source location error", linewidth=1.5,
            markersize=6, linestyle=:dash)
    end

    axislegend(ax; position=:rt)

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig, px_per_unit=3)
        println("  Saved: $filename")
    end

    set_theme!()
    return fig
end

# ==============================================================================
# Main experiment
# ==============================================================================

function run_scalability_study(;
        problem_path::String,
        grid_sizes::Vector{Int},
        ρ::Float64,
        benchmark_runs::Int,
        N_ref::Int,
        output_dir::String,
        verbose::Bool,
        make_plots::Bool,
        lengthscale_c::Union{Float64, Nothing}=nothing,
        lengthscale_s::Union{Float64, Nothing}=nothing,
        output_scale::Float64=1.0,
        output_scale_c::Union{Float64, Nothing}=nothing,
    )

    println("=" ^ 60)
    println("GP-FVM Source Identification — Scalability Study")
    println("=" ^ 60)

    # Load problem
    prob = load_problem(problem_path)
    verbose && println(prob)

    # Generate high-res reference
    println("\nGenerating reference solution (N=$N_ref)...")
    t_ref = @elapsed ref_base = generate_reference(prob, N_ref)
    ref = merge(ref_base, (; output_dir=output_dir))
    @printf("  Reference generated in %.2f s\n", t_ref)

    # Run sweep
    println("\nGrid sizes: $grid_sizes")
    println("Benchmark runs per N: $benchmark_runs")
    println("ρ = $ρ")
    println()

    results = NamedTuple[]

    for (i, N) in enumerate(grid_sizes)
        println("[$i/$(length(grid_sizes))] N = $N ($(N)×$(N) grid, $((N-1)^2) cells)")

        r = run_single_N(N, prob, ref; ρ=ρ, benchmark_runs=benchmark_runs, verbose=verbose,
                         lengthscale_c=lengthscale_c, lengthscale_s=lengthscale_s,
                         output_scale=output_scale, output_scale_c=output_scale_c)
        push!(results, r)

        @printf("  DOF: %d | Time: %.3f s | Source RMSE: %.4f | Loc. error: %.4f\n",
                r.n_total, r.time_total_s, r.rmse_s, r.location_error)
        @printf("  Fill-in: c=%.1f%%, s=%.1f%%\n", r.fill_c_pct, r.fill_s_pct)
    end

    # Build DataFrame
    df = DataFrame(results)

    # Print summary table
    println("\n" * "=" ^ 90)
    println("SUMMARY")
    println("=" ^ 90)
    @printf("%-5s %8s %10s %10s %10s %10s  %12s  %12s\n",
            "N", "DOF", "Time (s)", "RMSE(s)", "RMSE(c)", "Loc.err",
            "z(c) μ/σ", "z(s) μ/σ")
    println("-" ^ 95)
    for r in results
        @printf("%-5d %8d %10.3f %10.4f %10.6f %10.4f  %+.2f / %.2f  %+.2f / %.2f\n",
                r.N, r.n_total, r.time_total_s, r.rmse_s, r.rmse_c_vs_ref,
                r.location_error,
                r.z_c_mean, r.z_c_std, r.z_s_mean, r.z_s_std)
    end
    println("=" ^ 95)
    println("  (Calibrated posterior: z-score std ≈ 1.0, mean ≈ 0.0)")

    # Save CSV
    mkpath(output_dir)
    csv_path = joinpath(output_dir, "scalability_results.csv")
    CSV.write(csv_path, df)
    println("\nSaved CSV: $csv_path")

    # Plots
    if make_plots
        println("\nGenerating plots...")
        plot_timing(df; filename=joinpath(output_dir, "scaling_timing.pdf"))
        plot_error(df; filename=joinpath(output_dir, "scaling_error.pdf"))
    end

    return df
end

# ==============================================================================
# CLI
# ==============================================================================

function parse_scalability_args()
    s = ArgParseSettings(description = "Scalability study for GP-FVM source identification")

    @add_arg_table! s begin
        "--problem", "-p"
            help = "Path to problem TOML file"
            arg_type = String
            default = joinpath(@__DIR__, "problems", "default.toml")
        "--grid-sizes", "-N"
            help = "Comma-separated list of grid sizes"
            arg_type = String
            default = "11,16,21,26,31,41,51,61,81,101"
        "--rho"
            help = "Sparsity parameter"
            arg_type = Float64
            default = 2.0
        "--benchmark-runs"
            help = "Number of timed runs per grid size (0 = single run, no warmup)"
            arg_type = Int
            default = 3
        "--N-ref"
            help = "Reference grid size for convergence (default: 201)"
            arg_type = Int
            default = 201
        "--output-dir", "-o"
            help = "Output directory"
            arg_type = String
            default = joinpath(@__DIR__, "results", "scalability")
        "--no-plot"
            help = "Skip plot generation"
            action = :store_true
        "--quiet", "-q"
            help = "Suppress verbose output"
            action = :store_true
        "--lengthscale-c"
            help = "Fixed concentration lengthscale (default: 5*Δ per grid)"
            arg_type = Float64
            default = -1.0
        "--lengthscale-s"
            help = "Fixed source lengthscale (default: 5*Δ per grid)"
            arg_type = Float64
            default = -1.0
        "--output-scale"
            help = "Output scale σ²_s for source (default: 1.0)"
            arg_type = Float64
            default = 1.0
        "--output-scale-c"
            help = "Output scale σ²_c for concentration (default: same as --output-scale)"
            arg_type = Float64
            default = -1.0
    end

    return parse_args(s)
end

function scalability_main()
    args = parse_scalability_args()
    grid_sizes = parse.(Int, split(args["grid-sizes"], ","))
    sort!(grid_sizes)
    ls_c = args["lengthscale-c"] < 0 ? nothing : args["lengthscale-c"]
    ls_s = args["lengthscale-s"] < 0 ? nothing : args["lengthscale-s"]

    run_scalability_study(
        problem_path = args["problem"],
        grid_sizes = grid_sizes,
        ρ = args["rho"],
        benchmark_runs = args["benchmark-runs"],
        N_ref = args["N-ref"],
        output_dir = args["output-dir"],
        verbose = !args["quiet"],
        make_plots = !args["no-plot"],
        lengthscale_c = ls_c,
        lengthscale_s = ls_s,
        output_scale = args["output-scale"],
        output_scale_c = args["output-scale-c"] < 0 ? nothing : args["output-scale-c"],
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    scalability_main()
end
