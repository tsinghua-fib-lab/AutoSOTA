"""
    Paper Figures for Accuracy vs Compute Experiment

Three key plots:
1. Convergence: N vs Accuracy (all methods)
2. Pareto: Runtime vs Accuracy (all methods)
3. Memory Scaling: N vs Memory (batch vs EKF)
"""

using CairoMakie
using Statistics
using DataFrames
using CSV

# Include metrics for data types
include("metrics.jl")

# =============================================================================
# Style Configuration
# =============================================================================

const PAPER_COLORS = Dict(
    "sparse_fvm" => :royalblue,
    "ekf_fvm" => :purple,
    "sparse_collocation" => :seagreen,
    "classical_fvm" => :darkorange
)

const PAPER_MARKERS = Dict(
    "sparse_fvm" => :circle,
    "ekf_fvm" => :star5,
    "sparse_collocation" => :diamond,
    "classical_fvm" => :utriangle
)

const PAPER_LABELS = Dict(
    "sparse_fvm" => "GP-FVM (batch)",
    "ekf_fvm" => "GP-FVM (sequential)",
    "sparse_collocation" => "GP-Collocation",
    "classical_fvm" => "Classical FVM"
)

# Consistent figure styling
function paper_theme()
    return Theme(
        fontsize = 12,
        Axis = (
            xlabelsize = 14,
            ylabelsize = 14,
            titlesize = 14,
            xticklabelsize = 11,
            yticklabelsize = 11,
        ),
        Legend = (
            framevisible = false,
            labelsize = 11,
            patchsize = (20, 10),
        )
    )
end

# =============================================================================
# Plot 1: Convergence (N vs Accuracy)
# =============================================================================

"""
    plot_convergence(results::DataFrame; kwargs...)

Plot discretization convergence: grid size N vs relative L2 error.

Message: "GP-FVM matches classical FVM convergence, beats collocation"
"""
function plot_convergence(results::DataFrame;
                          methods::Vector{String} = ["sparse_fvm", "ekf_fvm", "sparse_collocation", "classical_fvm"],
                          filename::Union{Nothing,String} = nothing,
                          show_reference_slopes::Bool = true)

    set_theme!(paper_theme())

    fig = Figure(size=(500, 400))
    ax = Axis(fig[1, 1],
        xlabel = "Grid size N",
        ylabel = "Relative L2 error",
        xscale = log10,
        yscale = log10,
    )

    for method in methods
        haskey(PAPER_COLORS, method) || continue
        method_data = filter(row -> row.method == method, results)
        isempty(method_data) && continue

        # Aggregate over IC seeds
        grouped = combine(
            groupby(method_data, :N),
            :mean_l2_error => mean => :error_mean,
            :mean_l2_error => std => :error_std,
            nrow => :n_samples
        )
        sort!(grouped, :N)

        Ns = grouped.N
        errors = grouped.error_mean

        # Plot points
        scatter!(ax, Ns, errors,
            color = PAPER_COLORS[method],
            marker = PAPER_MARKERS[method],
            markersize = 10,
            label = PAPER_LABELS[method]
        )

        # Connect with lines
        lines!(ax, Ns, errors,
            color = PAPER_COLORS[method],
            linewidth = 2
        )

        # Error bars if multiple samples
        if any(grouped.error_std .> 0) && any(grouped.n_samples .> 1)
            error_low = min.(grouped.error_std, errors .* 0.9)
            error_high = grouped.error_std
            errorbars!(ax, Ns, errors, error_low, error_high,
                color = PAPER_COLORS[method],
                linewidth = 1
            )
        end
    end

    # Reference slopes
    if show_reference_slopes
        N_ref = [minimum(results.N), maximum(results.N)]
        e_base = 0.1  # Adjust based on data range

        # O(1/N) - first order
        lines!(ax, N_ref, e_base .* (N_ref[1] ./ N_ref),
            color = :gray, linestyle = :dash, linewidth = 1,
            label = "O(1/N)"
        )

        # O(1/N²) - second order
        lines!(ax, N_ref, e_base .* (N_ref[1] ./ N_ref).^2,
            color = :gray, linestyle = :dot, linewidth = 1,
            label = "O(1/N²)"
        )
    end

    axislegend(ax, position = :rt)

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig, px_per_unit=3)
        println("Saved: $filename")
    end

    return fig
end

# =============================================================================
# Plot 2: Pareto (Runtime vs Accuracy)
# =============================================================================

"""
    plot_pareto(results::DataFrame; kwargs...)

Plot accuracy vs runtime Pareto frontier.

Message: "GP-FVM is cost-competitive with classical FVM"
"""
function plot_pareto(results::DataFrame;
                     methods::Vector{String} = ["sparse_fvm", "ekf_fvm", "sparse_collocation", "classical_fvm"],
                     filename::Union{Nothing,String} = nothing,
                     label_points::Bool = true)

    set_theme!(paper_theme())

    fig = Figure(size=(500, 400))
    ax = Axis(fig[1, 1],
        xlabel = "Wall-clock time (s)",
        ylabel = "Relative L2 error",
        xscale = log10,
        yscale = log10,
    )

    for method in methods
        haskey(PAPER_COLORS, method) || continue
        method_data = filter(row -> row.method == method, results)
        isempty(method_data) && continue

        # Aggregate over IC seeds
        grouped = combine(
            groupby(method_data, :N),
            :time_s => mean => :time_mean,
            :time_s => std => :time_std,
            :mean_l2_error => mean => :error_mean,
            :mean_l2_error => std => :error_std
        )
        sort!(grouped, :N)

        times = grouped.time_mean
        errors = grouped.error_mean

        # Plot points
        scatter!(ax, times, errors,
            color = PAPER_COLORS[method],
            marker = PAPER_MARKERS[method],
            markersize = 10,
            label = PAPER_LABELS[method]
        )

        # Connect with lines (trajectory as N increases)
        lines!(ax, times, errors,
            color = PAPER_COLORS[method],
            linewidth = 2
        )

        # Label points with N
        if label_points
            for row in eachrow(grouped)
                text!(ax, row.time_mean, row.error_mean,
                    text = "$(row.N)",
                    fontsize = 8,
                    offset = (5, 5),
                    color = PAPER_COLORS[method]
                )
            end
        end
    end

    axislegend(ax, position = :rt)

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig, px_per_unit=3)
        println("Saved: $filename")
    end

    return fig
end

# =============================================================================
# Plot 3: Memory Scaling (N vs Memory)
# =============================================================================

"""
    plot_memory_scaling(results::DataFrame; kwargs...)

Plot memory usage vs grid size for batch vs sequential GP-FVM.

Message: "EKF solves the memory scalability problem"
"""
function plot_memory_scaling(results::DataFrame;
                              methods::Vector{String} = ["sparse_fvm", "ekf_fvm"],
                              filename::Union{Nothing,String} = nothing,
                              show_memory_limits::Bool = true)

    set_theme!(paper_theme())

    fig = Figure(size=(500, 400))
    ax = Axis(fig[1, 1],
        xlabel = "Grid size N",
        ylabel = "Peak memory (GB)",
        xscale = log10,
        yscale = log10,
    )

    for method in methods
        haskey(PAPER_COLORS, method) || continue
        method_data = filter(row -> row.method == method, results)
        isempty(method_data) && continue

        # Aggregate over IC seeds
        grouped = combine(
            groupby(method_data, :N),
            :memory_mb => mean => :memory_mean,
            :memory_mb => std => :memory_std
        )
        sort!(grouped, :N)

        Ns = grouped.N
        memory_gb = grouped.memory_mean ./ 1024  # Convert MB to GB

        # Plot points
        scatter!(ax, Ns, memory_gb,
            color = PAPER_COLORS[method],
            marker = PAPER_MARKERS[method],
            markersize = 10,
            label = PAPER_LABELS[method]
        )

        # Connect with lines
        lines!(ax, Ns, memory_gb,
            color = PAPER_COLORS[method],
            linewidth = 2
        )
    end

    # Reference memory limits
    if show_memory_limits
        N_range = [minimum(results.N), maximum(results.N) * 1.5]

        # 8 GB (laptop)
        hlines!(ax, [8.0], color = :red, linestyle = :dash, linewidth = 1)
        text!(ax, N_range[2], 8.0, text = "8 GB", fontsize = 9,
              align = (:right, :bottom), color = :red)

        # 16 GB (workstation)
        hlines!(ax, [16.0], color = :red, linestyle = :dot, linewidth = 1)
        text!(ax, N_range[2], 16.0, text = "16 GB", fontsize = 9,
              align = (:right, :bottom), color = :red)
    end

    axislegend(ax, position = :lt)

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig, px_per_unit=3)
        println("Saved: $filename")
    end

    return fig
end

# =============================================================================
# Combined Figure (all three panels)
# =============================================================================

"""
    plot_paper_figure(results::DataFrame; kwargs...)

Create combined 3-panel figure for paper.
"""
function plot_paper_figure(results::DataFrame;
                           filename::Union{Nothing,String} = nothing)

    set_theme!(paper_theme())

    fig = Figure(size=(1200, 350))

    # Panel A: Convergence
    ax1 = Axis(fig[1, 1],
        xlabel = "Grid size N",
        ylabel = "Relative L2 error",
        xscale = log10,
        yscale = log10,
        title = "(a) Convergence"
    )

    # Panel B: Pareto
    ax2 = Axis(fig[1, 2],
        xlabel = "Wall-clock time (s)",
        ylabel = "Relative L2 error",
        xscale = log10,
        yscale = log10,
        title = "(b) Accuracy vs Runtime"
    )

    # Panel C: Memory
    ax3 = Axis(fig[1, 3],
        xlabel = "Grid size N",
        ylabel = "Peak memory (GB)",
        xscale = log10,
        yscale = log10,
        title = "(c) Memory Scaling"
    )

    all_methods = ["sparse_fvm", "ekf_fvm", "sparse_collocation", "classical_fvm"]
    memory_methods = ["sparse_fvm", "ekf_fvm"]

    for method in all_methods
        haskey(PAPER_COLORS, method) || continue
        method_data = filter(row -> row.method == method, results)
        isempty(method_data) && continue

        grouped = combine(
            groupby(method_data, :N),
            :mean_l2_error => mean => :error_mean,
            :time_s => mean => :time_mean,
            :memory_mb => mean => :memory_mean
        )
        sort!(grouped, :N)

        Ns = grouped.N
        errors = grouped.error_mean
        times = grouped.time_mean
        memory_gb = grouped.memory_mean ./ 1024

        # Panel A: Convergence
        scatter!(ax1, Ns, errors, color=PAPER_COLORS[method], marker=PAPER_MARKERS[method], markersize=8)
        lines!(ax1, Ns, errors, color=PAPER_COLORS[method], linewidth=1.5, label=PAPER_LABELS[method])

        # Panel B: Pareto
        scatter!(ax2, times, errors, color=PAPER_COLORS[method], marker=PAPER_MARKERS[method], markersize=8)
        lines!(ax2, times, errors, color=PAPER_COLORS[method], linewidth=1.5)

        # Panel C: Memory (only batch and EKF)
        if method in memory_methods
            scatter!(ax3, Ns, memory_gb, color=PAPER_COLORS[method], marker=PAPER_MARKERS[method], markersize=8)
            lines!(ax3, Ns, memory_gb, color=PAPER_COLORS[method], linewidth=1.5)
        end
    end

    # Memory reference lines
    N_max = maximum(results.N) * 1.2
    hlines!(ax3, [8.0], color=:red, linestyle=:dash, linewidth=1)
    hlines!(ax3, [16.0], color=:red, linestyle=:dot, linewidth=1)

    # Single legend for all panels
    Legend(fig[2, 1:3], ax1, orientation=:horizontal, tellheight=true)

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig, px_per_unit=3)
        println("Saved: $filename")
    end

    return fig
end

# =============================================================================
# Quick test function
# =============================================================================

"""
    test_paper_plots()

Generate test plots from existing results or synthetic data.
"""
function test_paper_plots(; results_file::Union{Nothing,String} = nothing)
    if !isnothing(results_file) && isfile(results_file)
        results = CSV.read(results_file, DataFrame)
    else
        # Generate synthetic test data
        println("No results file provided, generating synthetic data...")
        results = generate_synthetic_results()
    end

    println("Generating plots...")

    fig1 = plot_convergence(results, filename="paper_figures/convergence.pdf")
    fig2 = plot_pareto(results, filename="paper_figures/pareto.pdf")
    fig3 = plot_memory_scaling(results, filename="paper_figures/memory_scaling.pdf")
    fig_combined = plot_paper_figure(results, filename="paper_figures/combined.pdf")

    println("Done!")
    return fig_combined
end

function generate_synthetic_results()
    # Synthetic data for testing plot layout
    rows = []

    for N in [50, 100, 200, 400]
        for seed in 1:3
            # GP-FVM batch: good accuracy, high memory
            push!(rows, (
                method = "sparse_fvm", ic_family = :sine, ic_seed = seed, N = N,
                time_s = 0.5 * (N/50)^2.5, memory_mb = 50 * (N/50)^3,
                mean_l2_error = 0.05 / (N/50)^1.5 * (1 + 0.1*randn()),
                cholesky_nnz = 0, fillin_pct = 0.0, final_l2_error = 0.0,
                coverage_95 = 0.95, mean_crps = 0.0, error_std_corr = 0.0
            ))

            # GP-FVM sequential: similar accuracy, low memory
            push!(rows, (
                method = "ekf_fvm", ic_family = :sine, ic_seed = seed, N = N,
                time_s = 0.3 * (N/50)^2, memory_mb = 20 * (N/50)^2,
                mean_l2_error = 0.06 / (N/50)^1.3 * (1 + 0.1*randn()),
                cholesky_nnz = 0, fillin_pct = 0.0, final_l2_error = 0.0,
                coverage_95 = 0.5, mean_crps = 0.0, error_std_corr = 0.0
            ))

            # GP-Collocation: worse accuracy
            push!(rows, (
                method = "sparse_collocation", ic_family = :sine, ic_seed = seed, N = N,
                time_s = 0.4 * (N/50)^2.2, memory_mb = 40 * (N/50)^2.5,
                mean_l2_error = 0.08 / (N/50)^1.0 * (1 + 0.1*randn()),
                cholesky_nnz = 0, fillin_pct = 0.0, final_l2_error = 0.0,
                coverage_95 = 0.8, mean_crps = 0.0, error_std_corr = 0.0
            ))

            # Classical FVM: fast, similar accuracy to GP-FVM
            push!(rows, (
                method = "classical_fvm", ic_family = :sine, ic_seed = seed, N = N,
                time_s = 0.1 * (N/50)^1.5, memory_mb = 5 * (N/50),
                mean_l2_error = 0.055 / (N/50)^1.4 * (1 + 0.1*randn()),
                cholesky_nnz = 0, fillin_pct = 0.0, final_l2_error = 0.0,
                coverage_95 = 0.0, mean_crps = 0.0, error_std_corr = 0.0
            ))
        end
    end

    return DataFrame(rows)
end
