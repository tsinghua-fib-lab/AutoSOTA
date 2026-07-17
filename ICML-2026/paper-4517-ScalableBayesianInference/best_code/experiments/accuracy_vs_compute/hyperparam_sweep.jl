"""
    Hyperparameter Sweep Experiment

For a single problem instance, sweep over:
- Grid sizes N
- Smoothness (GP-FVM: 1,2,3 for Matérn 3/2,5/2,7/2; Collocation: 2,3)
- Sparsity parameter ρ
- Lengthscale (pick best for each config)

Goal: Find optimal hyperparameters for GP-FVM vs GP-Collocation comparison.
"""

using Printf
using DataFrames
using CSV
using Dates
using ProgressMeter
using LinearAlgebra

include("problem.jl")
include("metrics.jl")
include("methods/sparse_fvm.jl")
include("methods/sparse_collocation.jl")
include("methods/classical_fvm.jl")

"""
    sweep_lengthscale(solver, instance, N, n_timesteps, smoothness, ρ, lengthscales)

Try different lengthscales and return the one with lowest error.
"""
function sweep_lengthscale(solver, instance, N, n_timesteps, smoothness, ρ, lengthscales)
    best_error = Inf
    best_ls = lengthscales[1]
    best_result = nothing

    for ls in lengthscales
        try
            result = solver(instance, N;
                n_timesteps=n_timesteps,
                ρ=ρ,
                smoothness=smoothness,
                lengthscale=ls)  # absolute lengthscale

            # Compute error against reference
            ref = instance.reference
            T_end = instance.problem.T_end
            u_ref = evaluate_reference(ref, result.xs, T_end)
            u_pred = result.mean[:, end]

            rel_error = norm(u_pred - u_ref) / norm(u_ref)

            if rel_error < best_error
                best_error = rel_error
                best_ls = ls
                best_result = result
            end
        catch e
            # Skip failed configs
            continue
        end
    end

    return best_result, best_ls, best_error
end

"""
    run_hyperparam_sweep(; kwargs...)

Run hyperparameter sweep for a single problem instance.
"""
function run_hyperparam_sweep(;
        ic_family::Symbol = :sine,
        ic_seed::Int = 43,
        grid_sizes::Vector{Int} = [20, 40, 60, 80, 100, 150, 200],
        smoothness_fvm::Vector{Int} = [1, 2, 3],      # Matérn 3/2, 5/2, 7/2
        smoothness_colloc::Vector{Int} = [2, 3],      # Need 2nd deriv, so min 5/2
        rhos::Vector{Float64} = [2.0, 3.0, 4.0, 5.0],
        lengthscales::Vector{Float64} = [0.05, 0.1, 0.15, 0.2, 0.3],
        output_dir::String = "results/hyperparam_sweep",
        verbose::Bool = true
    )

    mkpath(output_dir)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

    # Setup problem
    problem = BurgersProblem(
        x_min = 0.0,
        x_max = 1.0,
        T_end = 0.3,
        ν = 0.01,
        u_left = 0.0,
        u_right = 0.0
    )

    ic = InitialCondition(ic_family, ic_seed)
    instance = ProblemInstance(problem, ic; N_ref=2000)

    verbose && println("="^70)
    verbose && println("Hyperparameter Sweep")
    verbose && println("="^70)
    verbose && println("IC: $ic_family (seed=$ic_seed)")
    verbose && println("Grid sizes: $grid_sizes")
    verbose && println("Smoothness (FVM): $smoothness_fvm")
    verbose && println("Smoothness (Colloc): $smoothness_colloc")
    verbose && println("ρ values: $rhos")
    verbose && println("Lengthscales: $lengthscales")
    verbose && println()

    results = DataFrame(
        method = String[],
        N = Int[],
        smoothness = Int[],
        rho = Float64[],
        best_lengthscale = Float64[],
        rel_l2_error = Float64[],
        time_s = Float64[]
    )

    # Classical FVM baseline (no hyperparams to tune)
    verbose && println("Running Classical FVM baseline...")
    for N in grid_sizes
        n_timesteps = max(10, N ÷ 2)  # More timesteps for better temporal resolution
        result = solve_classical_fvm(instance, N; n_timesteps=n_timesteps)

        ref = instance.reference
        u_ref = evaluate_reference(ref, result.xs, problem.T_end)
        u_pred = result.mean[:, end]
        rel_error = norm(u_pred - u_ref) / norm(u_ref)

        push!(results, ("classical_fvm", N, 0, 0.0, 0.0, rel_error, result.wall_time_s))
    end

    # Count total GP configurations
    n_fvm_configs = length(grid_sizes) * length(smoothness_fvm) * length(rhos)
    n_colloc_configs = length(grid_sizes) * length(smoothness_colloc) * length(rhos)
    n_total = n_fvm_configs + n_colloc_configs

    verbose && println("\nRunning GP-FVM sweep ($n_fvm_configs configs)...")
    progress = Progress(n_total, desc="Sweeping hyperparams...")

    # GP-FVM sweep
    for N in grid_sizes
        n_timesteps = max(10, N ÷ 2)

        for smooth in smoothness_fvm
            for ρ in rhos
                result, best_ls, best_error = sweep_lengthscale(
                    solve_sparse_fvm, instance, N, n_timesteps,
                    smooth, ρ, lengthscales
                )

                if result !== nothing
                    push!(results, (
                        "sparse_fvm", N, smooth, ρ, best_ls,
                        best_error, result.wall_time_s
                    ))
                end

                next!(progress)
            end
        end
    end

    verbose && println("\nRunning GP-Collocation sweep ($n_colloc_configs configs)...")

    # GP-Collocation sweep
    for N in grid_sizes
        n_timesteps = max(10, N ÷ 2)

        for smooth in smoothness_colloc
            for ρ in rhos
                result, best_ls, best_error = sweep_lengthscale(
                    solve_sparse_collocation, instance, N, n_timesteps,
                    smooth, ρ, lengthscales
                )

                if result !== nothing
                    push!(results, (
                        "sparse_collocation", N, smooth, ρ, best_ls,
                        best_error, result.wall_time_s
                    ))
                end

                next!(progress)
            end
        end
    end

    # Save results
    csv_path = joinpath(output_dir, "sweep_$(timestamp).csv")
    CSV.write(csv_path, results)
    verbose && println("\nSaved: $csv_path")

    # Print summary: best config per method per N
    verbose && println("\n" * "="^70)
    verbose && println("Best configurations per method per N:")
    verbose && println("="^70)

    for method in ["sparse_fvm", "sparse_collocation", "classical_fvm"]
        verbose && println("\n[$method]")
        method_df = filter(row -> row.method == method, results)

        for N in grid_sizes
            n_df = filter(row -> row.N == N, method_df)
            if nrow(n_df) > 0
                best_idx = argmin(n_df.rel_l2_error)
                best = n_df[best_idx, :]
                if method == "classical_fvm"
                    @printf("  N=%3d: error=%.4f\n", N, best.rel_l2_error)
                else
                    @printf("  N=%3d: error=%.4f (ν=%d, ρ=%.1f, ls=%.2f)\n",
                        N, best.rel_l2_error, best.smoothness, best.rho, best.best_lengthscale)
                end
            end
        end
    end

    return results
end

# CLI
using ArgParse

function parse_args_sweep()
    s = ArgParseSettings(description = "Hyperparameter sweep for GP-FVM vs Collocation")

    @add_arg_table! s begin
        "--ic-family"
            help = "IC family (sine, gaussian, multimode)"
            arg_type = Symbol
            default = :sine
        "--ic-seed"
            help = "IC random seed"
            arg_type = Int
            default = 43
        "--grid-sizes"
            help = "Comma-separated grid sizes"
            arg_type = String
            default = "20,40,60,80,100,150,200"
        "--rhos"
            help = "Comma-separated ρ values"
            arg_type = String
            default = "2.0,3.0,4.0,5.0"
        "--lengthscales"
            help = "Comma-separated lengthscale values"
            arg_type = String
            default = "0.05,0.1,0.15,0.2,0.3"
        "--output", "-o"
            help = "Output directory"
            arg_type = String
            default = "results/hyperparam_sweep"
    end

    return parse_args(s)
end

function main()
    args = parse_args_sweep()

    grid_sizes = parse.(Int, split(args["grid-sizes"], ","))
    rhos = parse.(Float64, split(args["rhos"], ","))
    lengthscales = parse.(Float64, split(args["lengthscales"], ","))

    run_hyperparam_sweep(
        ic_family = args["ic-family"],
        ic_seed = args["ic-seed"],
        grid_sizes = grid_sizes,
        rhos = rhos,
        lengthscales = lengthscales,
        output_dir = args["output"]
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
