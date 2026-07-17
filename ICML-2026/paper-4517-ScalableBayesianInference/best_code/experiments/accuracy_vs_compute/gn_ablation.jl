"""
Gauss-Newton ablation for the EKF sequential solver (App. F of the paper).

Runs the 1D Burgers forward problem with varying max_gn_iter (1, 2, 5, 10)
and compares solution accuracy and timing, validating the single-step
assumption used in the EKF-style sequential solver.

Usage:
    julia --project=../.. gn_ablation.jl
    julia --project=../.. gn_ablation.jl -N 100 --n-samples 3
"""

using LinearAlgebra, SparseArrays
using Random, Statistics, Printf
using ArgParse
using CairoMakie
using TuePlots

push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields

import GaussianMarkovRandomFields: mean, std
import FunctionalGPs: ⊗
using SparseConnectivityTracer, SparseMatrixColorings

include("problem.jl")
include("metrics.jl")
include("methods/ekf_fvm.jl")

function run_gn_ablation(;
        N::Int=100, n_timesteps::Int=50,
        ic_family::Symbol=:sine, n_samples::Int=5,
        max_iters::Vector{Int}=[1, 2, 5, 10],
        ρ::Float64=3.0, smoothness::Int=2,
        seed::Int=42, output_dir::String=joinpath(@__DIR__, "results", "gn_ablation"),
        verbose::Bool=true)

    println("=" ^ 60)
    println("Gauss-Newton Ablation Study (EKF Sequential Solver)")
    println("=" ^ 60)
    @printf("N=%d, n_t=%d, IC=%s, samples=%d\n", N, n_timesteps, ic_family, n_samples)
    @printf("max_iter values: %s\n", max_iters)
    println()

    rng = MersenneTwister(seed)

    # Generate problem instances
    bp = BurgersProblem()
    instances = generate_problem_instances(bp, [ic_family], n_samples;
        base_seed=seed, verbose=verbose)

    # Warmup
    verbose && print("Warmup... ")
    solve_ekf_fvm(instances[1], N; n_timesteps=n_timesteps, ρ=ρ, smoothness=smoothness, max_gn_iter=1)
    verbose && println("done.\n")

    # Run ablation
    results = []

    for max_iter in max_iters
        l2_errors = Float64[]
        times = Float64[]

        for (i, inst) in enumerate(instances)
            t = @elapsed result = solve_ekf_fvm(inst, N;
                n_timesteps=n_timesteps, ρ=ρ, smoothness=smoothness,
                max_gn_iter=max_iter)

            err = final_l2_error(result, inst.reference)
            push!(l2_errors, err)
            push!(times, result.wall_time_s)
        end

        mean_err = Statistics.mean(l2_errors)
        std_err = Statistics.std(l2_errors)
        mean_time = Statistics.mean(times)

        push!(results, (; max_iter, mean_err, std_err, mean_time, l2_errors, times))

        @printf("  max_iter=%2d | L2 error: %.4f ± %.4f | time: %.2fs\n",
                max_iter, mean_err, std_err, mean_time)
    end

    # Summary
    println("\n" * "=" ^ 60)
    println("SUMMARY")
    println("=" ^ 60)
    @printf("%-12s %12s %12s %10s\n", "max_iter", "L2 error", "± std", "time (s)")
    println("-" ^ 50)
    for r in results
        @printf("%-12d %12.6f %12.6f %10.2f\n", r.max_iter, r.mean_err, r.std_err, r.mean_time)
    end

    # Relative improvement over max_iter=1
    base_err = results[1].mean_err
    println()
    for r in results[2:end]
        improvement = (base_err - r.mean_err) / base_err * 100
        @printf("  max_iter=%d vs 1: %.1f%% improvement, %.1fx slower\n",
                r.max_iter, improvement, r.mean_time / results[1].mean_time)
    end
    println("=" ^ 60)

    # Plot
    mkpath(output_dir)

    set_theme!(Theme(
        TuePlots.SETTINGS[:ICML];
        font=true, fontsize=true, figsize=true,
        single_column=true, nrows=1, ncols=1,
        subplot_height_to_width_ratio=0.75,
    ))
    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel="Gauss-Newton iterations per timestep",
        ylabel="Relative L2 error (final time)",
        xticks=max_iters,
    )

    mean_errs = [r.mean_err for r in results]
    std_errs = [r.std_err for r in results]

    scatterlines!(ax, max_iters, mean_errs;
        color=:royalblue, linewidth=1.6, markersize=6)
    errorbars!(ax, max_iters, mean_errs, std_errs;
        color=:royalblue, whiskerwidth=6, linewidth=1)

    filename = joinpath(output_dir, "gn_ablation.pdf")
    save(filename, fig, pt_per_unit=1)
    set_theme!()
    println("\nSaved: $filename")

    return results
end

function parse_ablation_args()
    s = ArgParseSettings(description = "GN ablation for EKF sequential solver")
    @add_arg_table! s begin
        "-N"
            arg_type = Int
            default = 100
        "--n-timesteps"
            arg_type = Int
            default = 50
        "--ic-family"
            arg_type = Symbol
            default = :sine
        "--n-samples"
            arg_type = Int
            default = 5
        "--seed"
            arg_type = Int
            default = 42
        "--output-dir", "-o"
            default = joinpath(@__DIR__, "results", "gn_ablation")
    end
    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_ablation_args()
    run_gn_ablation(
        N = args["N"],
        n_timesteps = args["n-timesteps"],
        ic_family = args["ic-family"],
        n_samples = args["n-samples"],
        seed = args["seed"],
        output_dir = args["output-dir"],
    )
end
