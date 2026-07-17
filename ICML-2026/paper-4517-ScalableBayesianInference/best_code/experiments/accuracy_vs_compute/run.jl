"""
    Accuracy vs Compute Experiment Runner

Main entry point for running the unified scalability/accuracy experiment.
Compares sparse GP-FVM against baselines across multiple grid sizes and ICs.
"""

using ArgParse
using CSV
using DataFrames
using Dates
using ProgressMeter

# Include all components
include("problem.jl")
include("metrics.jl")
include("plotting.jl")
include("methods/sparse_fvm.jl")
include("methods/sparse_collocation.jl")
include("methods/classical_fvm.jl")
include("methods/ekf_fvm.jl")

const METHODS = Dict(
    "sparse_fvm" => solve_sparse_fvm,
    "sparse_collocation" => solve_sparse_collocation,
    "classical_fvm" => solve_classical_fvm,
    "ekf_fvm" => solve_ekf_fvm,
)

"""
    compute_n_timesteps(N::Int)

Compute number of time steps based on spatial resolution.
Uses n_t = max(10, N÷2) for all methods, giving ~2 spatial points per timestep.

This ratio ensures temporal discretization error doesn't dominate at moderate N.
Higher timestep counts improve GP-FVM accuracy significantly.
"""
function compute_n_timesteps(N::Int)
    return max(10, N ÷ 2)
end

"""
    run_experiment(; kwargs...)

Run the full accuracy vs compute experiment.

# Keyword arguments
- `methods`: Vector of method names to run
- `ic_families`: Vector of IC family symbols
- `n_samples`: Number of random IC samples per family
- `grid_sizes`: Vector of N values to test
- `output_dir`: Directory for results
- `verbose`: Print progress
- `ρ`: Sparse Cholesky threshold (sparsity parameter, default 3.0)
- `smoothness`: Matérn smoothness (1=3/2, 2=5/2, 3=7/2, default 2)
"""
function run_experiment(;
        methods::Vector{String}=["sparse_fvm", "sparse_collocation", "classical_fvm"],
        ic_families::Vector{Symbol}=[:sine, :gaussian, :multimode],
        n_samples::Int=5,
        grid_sizes::Vector{Int}=[50, 100, 200],
        base_seed::Int=42,
        output_dir::String="results/accuracy_vs_compute",
        verbose::Bool=true,
        ρ::Float64=3.0,
        smoothness::Int=2
    )

    # Create output directory
    mkpath(output_dir)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

    verbose && println("="^60)
    verbose && println("Accuracy vs Compute Experiment")
    verbose && println("="^60)
    verbose && println("Methods: $(join(methods, ", "))")
    verbose && println("IC families: $(join(ic_families, ", "))")
    verbose && println("Samples per family: $n_samples")
    verbose && println("Grid sizes: $(join(grid_sizes, ", "))")
    verbose && println("GP params: ρ=$ρ, smoothness=$smoothness (Matérn $(2*smoothness+1)/2)")
    verbose && println("Output: $output_dir")
    verbose && println()

    # Problem definition
    problem = BurgersProblem(
        x_min = 0.0,
        x_max = 1.0,
        T_end = 0.3,
        ν = 0.01,
        u_left = 0.0,
        u_right = 0.0
    )

    # Generate problem instances
    verbose && println("Generating problem instances...")
    instances = generate_problem_instances(
        problem, ic_families, n_samples;
        base_seed=base_seed, verbose=verbose
    )
    verbose && println("  Total instances: $(length(instances))")

    # Results storage
    all_metrics = MetricsSummary[]

    # Total number of runs
    n_total = length(methods) * length(grid_sizes) * length(instances)
    verbose && println("\nTotal runs: $n_total")

    # Main experiment loop
    progress = Progress(n_total, desc="Running experiments...")

    for method_name in methods
        solver = METHODS[method_name]

        for N in grid_sizes

            # Compute n_timesteps for this grid size (same for all methods)
            n_timesteps = compute_n_timesteps(N)
            Δt = problem.T_end / n_timesteps
            # Print time discretization info once per grid size (when processing first method)
            if verbose && method_name == methods[1]
                println("  N=$N: n_timesteps=$n_timesteps, Δt=$(round(Δt, sigdigits=3))")
            end

            for (i, instance) in enumerate(instances)
                try
                    # Run solver with consistent n_timesteps
                    # Pass GP-specific params to sparse methods and ekf_fvm
                    if startswith(method_name, "sparse") || method_name == "ekf_fvm"
                        result = solver(instance, N; n_timesteps=n_timesteps, ρ=ρ, smoothness=smoothness)
                    else
                        result = solver(instance, N; n_timesteps=n_timesteps)
                    end

                    # Determine IC family from instance
                    family_name = string(typeof(instance.ic.family).name.name)
                    # Strip "IC" suffix if present
                    if endswith(family_name, "IC")
                        family_name = family_name[1:end-2]
                    end
                    ic_family = Symbol(lowercase(family_name))

                    # Compute metrics
                    metrics = compute_metrics(
                        result,
                        ic_family,
                        instance.ic.seed,
                        instance.reference
                    )

                    push!(all_metrics, metrics)

                catch e
                    verbose && println("\nError running $method_name at N=$N: $e")
                end

                next!(progress)
            end
        end
    end

    # Convert to DataFrame
    verbose && println("\nProcessing results...")
    results_df = results_to_dataframe(all_metrics)

    # Save raw results
    csv_path = joinpath(output_dir, "results_$(timestamp).csv")
    CSV.write(csv_path, results_df)
    verbose && println("Saved results: $csv_path")

    # Generate summary table
    summary = summary_table(results_df)
    summary_path = joinpath(output_dir, "summary_$(timestamp).csv")
    CSV.write(summary_path, summary)
    verbose && println("Saved summary: $summary_path")

    # Generate plots (only if we have results)
    verbose && println("\nGenerating plots...")

    if nrow(results_df) == 0
        verbose && println("  No results to plot!")
        return results_df, DataFrame()
    end

    pareto_plot(results_df;
        title = "Accuracy vs Compute",
        filename = joinpath(output_dir, "pareto_$(timestamp).pdf")
    )

    scaling_plot(results_df;
        title = "Computational Scaling",
        filename = joinpath(output_dir, "scaling_$(timestamp).pdf")
    )

    convergence_plot(results_df;
        title = "Discretization Convergence",
        filename = joinpath(output_dir, "convergence_$(timestamp).pdf")
    )

    # Only plot calibration if we have GP methods
    gp_results = filter(row -> row.method != "classical_fvm", results_df)
    if nrow(gp_results) > 0
        calibration_plot(gp_results;
            title = "UQ Calibration",
            filename = joinpath(output_dir, "calibration_$(timestamp).pdf")
        )
    end

    verbose && println("\nExperiment complete!")
    verbose && println("="^60)

    return results_df, summary
end

"""
    quick_test()

Run a quick test with minimal settings to verify everything works.
"""
function quick_test()
    println("Running quick test...")
    results, summary = run_experiment(
        methods = ["sparse_fvm", "classical_fvm"],
        ic_families = [:sine],
        n_samples = 2,
        grid_sizes = [30, 50],
        output_dir = "results/accuracy_vs_compute_test",
        verbose = true
    )
    println("\nSummary:")
    println(summary)
    return results, summary
end

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(
        description = "Accuracy vs Compute Experiment for Sparse GP-FVM"
    )

    @add_arg_table! s begin
        "--methods", "-m"
            help = "Comma-separated list of methods"
            arg_type = String
            default = "sparse_fvm,ekf_fvm,sparse_collocation,classical_fvm"
        "--ic-families"
            help = "Comma-separated list of IC families"
            arg_type = String
            default = "sine,gaussian,multimode"
        "--n-samples"
            help = "Number of random IC samples per family"
            arg_type = Int
            default = 5
        "--grid-sizes", "-N"
            help = "Comma-separated list of grid sizes"
            arg_type = String
            default = "50,100,200,400"
        "--output", "-o"
            help = "Output directory"
            arg_type = String
            default = "results/accuracy_vs_compute"
        "--seed"
            help = "Base random seed"
            arg_type = Int
            default = 42
        "--quick"
            help = "Run quick test with minimal settings"
            action = :store_true
        "--quiet", "-q"
            help = "Suppress output"
            action = :store_true
        "--rho"
            help = "Sparse Cholesky threshold (sparsity parameter)"
            arg_type = Float64
            default = 3.0
        "--smoothness"
            help = "Matérn smoothness (1=3/2, 2=5/2, 3=7/2)"
            arg_type = Int
            default = 2
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    if args["quick"]
        quick_test()
        return
    end

    # Parse comma-separated arguments
    methods = split(args["methods"], ",") .|> strip .|> String

    ic_families = split(args["ic-families"], ",") .|> strip .|> Symbol
    grid_sizes = split(args["grid-sizes"], ",") .|> strip .|> x -> parse(Int, x)

    run_experiment(
        methods = methods,
        ic_families = ic_families,
        n_samples = args["n-samples"],
        grid_sizes = grid_sizes,
        base_seed = args["seed"],
        output_dir = args["output"],
        verbose = !args["quiet"],
        ρ = args["rho"],
        smoothness = args["smoothness"]
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
