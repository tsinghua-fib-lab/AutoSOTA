"""
2D Burgers Source Identification — Nonlinear Inverse Problem

Infer an unknown spatial source s(x,y) from noisy observations of the
solution u(x,y,t) to the 2D viscous Burgers equation.

Usage:
    # Quick test
    julia --project=../.. run.jl -N 8 --n-timesteps 5

    # Full run
    julia --project=../.. run.jl -N 11 --n-timesteps 10

    # With convergence analysis
    julia --project=../.. run.jl -N 11 --n-timesteps 10 --convergence
"""

using LinearAlgebra, SparseArrays
using Random
using ArgParse
using Statistics
using Printf
using CairoMakie
using Kronecker: kronecker
using SparseConnectivityTracer, SparseMatrixColorings

push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields

import GaussianMarkovRandomFields: mean, std, precision_map, gaussian_approximation
import FunctionalGPs: ⊗
using GaussianMarkovRandomFields: NonlinearLeastSquaresModel

include("problem.jl")
include("functionals.jl")
include("constraints.jl")
include("solver.jl")
include("plot.jl")

# ==============================================================================
# CLI
# ==============================================================================

function parse_args_cli()
    s = ArgParseSettings(description = "2D Burgers source identification")
    @add_arg_table! s begin
        "-N"
            help = "Grid size (NxN)"
            arg_type = Int
            default = 11
        "--n-timesteps"
            arg_type = Int
            default = 10
        "--dt"
            arg_type = Float64
            default = 0.05
        "--nu"
            arg_type = Float64
            default = 0.01
        "--max-iter"
            help = "Max Gauss-Newton iterations (safeguard)"
            arg_type = Int
            default = 50
        "--n-obs"
            arg_type = Int
            default = 20
        "--obs-every"
            help = "Observe every Nth timestep"
            arg_type = Int
            default = 3
        "--noise-std"
            arg_type = Float64
            default = 0.05
        "--lengthscale"
            arg_type = Float64
            default = 0.15
        "--lengthscale-s"
            arg_type = Float64
            default = 0.15
        "--rho"
            arg_type = Float64
            default = 2.0
        "--source-amplitude"
            arg_type = Float64
            default = 1.0
        "--output-scale"
            arg_type = Float64
            default = 1.0
        "--seed"
            arg_type = Int
            default = 42
        "--log-source"
            help = "Use log-Gaussian source prior (s = exp(g), g ~ GP)"
            action = :store_true
        "--n-source-samples"
            help = "Number of posterior samples for source std (0 = delta method)"
            arg_type = Int
            default = 0
        "--convergence"
            help = "Run convergence analysis"
            action = :store_true
        "--output-dir", "-o"
            default = joinpath(@__DIR__, "results")
        "--quiet", "-q"
            action = :store_true
        "--no-plot"
            action = :store_true
    end
    return parse_args(s)
end

function main()
    args = parse_args_cli()
    verbose = !args["quiet"]
    N = args["N"]

    rng = MersenneTwister(args["seed"])

    println("=" ^ 60)
    println("2D Burgers Source Identification")
    println("=" ^ 60)

    # Generate problem
    prob = random_problem(rng; ν=args["nu"], T_end=args["n-timesteps"] * args["dt"],
                          Δt=args["dt"])
    verbose && @printf("Sources: %d, ν=%.3f, T=%.2f, Δt=%.3f\n",
                       length(prob.sources), prob.ν, prob.T_end, prob.Δt)
    for (i, s) in enumerate(prob.sources)
        verbose && @printf("  [%d] (%.2f, %.2f), amp=%.2f, width=%.2f\n",
                           i, s.x, s.y, s.strength, s.width)
    end

    # Forward solve for ground truth
    xs = collect(range(prob.domain[1], prob.domain[2], length=N))
    ys = collect(range(prob.domain[3], prob.domain[4], length=N))

    verbose && println("\nForward solve for ground truth...")
    t_fwd = @elapsed u_truth, s_true = forward_solve(prob, xs, ys)
    verbose && @printf("  Done in %.2fs. u range: [%.3f, %.3f]\n",
                       t_fwd, minimum(u_truth), maximum(u_truth))

    # Generate observations
    n_t = round(Int, prob.T_end / prob.Δt)
    obs = generate_observations(rng, u_truth, xs, ys;
        n_obs=args["n-obs"], obs_every=args["obs-every"],
        noise_std=args["noise-std"])
    obs_data = merge(obs, (; Nx=N, Ny=N))

    verbose && @printf("  %d observations at %d timesteps\n",
                       length(obs.obs_values), length(obs.obs_timesteps))

    # Solve
    verbose && println("\nGP-FVM solve...")
    result = solve_burgers_source_id(prob, obs_data;
        ρ=args["rho"],
        lengthscale_u=args["lengthscale"],
        lengthscale_s=args["lengthscale-s"],
        output_scale=args["output-scale"],
        source_amplitude=args["source-amplitude"],
        max_iter=args["max-iter"],
        log_source=args["log-source"],
        n_source_samples=args["n-source-samples"],
        verbose=verbose)

    # Metrics
    s_rmse = sqrt(Statistics.mean((s_true .- result.s_mean).^2))
    u_rmse_final = sqrt(Statistics.mean((u_truth[:,:,end] .- result.u_mean[:,:,end]).^2))

    println("\n" * "=" ^ 50)
    println("RESULTS")
    println("=" ^ 50)
    @printf("  Source RMSE:      %.4f\n", s_rmse)
    @printf("  u RMSE (final t): %.4f\n", u_rmse_final)
    @printf("  Total time:       %.1fs\n", result.info.timings["total"])
    @printf("  Joint DOF:        %d\n", result.info.n_joint)
    println("=" ^ 50)

    # Plots
    if !args["no-plot"]
        outdir = args["output-dir"]
        mkpath(outdir)

        ts = collect(range(0, prob.T_end, length=n_t+1))

        plot_source_comparison(result, s_true, xs, ys;
            filename=joinpath(outdir, "source_comparison.pdf"))
        verbose && println("Saved: source_comparison.pdf")

        # Mix of observed and unobserved timesteps
        n_t_hist = size(u_truth, 3)
        snap_times = [1, 4, obs.obs_timesteps[2], 8, obs.obs_timesteps[end]]
        plot_solution_snapshots(result, u_truth, xs, ys, ts;
            snapshot_times=snap_times,
            filename=joinpath(outdir, "solution_snapshots.pdf"),
            obs_data=obs)
        verbose && println("Saved: solution_snapshots.pdf")
    end

    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
