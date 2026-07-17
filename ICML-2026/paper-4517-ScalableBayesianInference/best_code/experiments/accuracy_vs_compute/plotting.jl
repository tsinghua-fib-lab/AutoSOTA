"""
    Plotting Utilities for Accuracy vs Compute Experiment

Creates Pareto plots, scaling plots, calibration plots, etc.
"""

using CairoMakie
using Statistics
using DataFrames

# Include metrics for MetricsSummary type
include("metrics.jl")

# Color scheme for methods
const METHOD_COLORS = Dict(
    "sparse_fvm" => :blue,
    "ekf_fvm" => :purple,
    "sparse_collocation" => :green,
    "classical_fvm" => :orange,
)

const METHOD_MARKERS = Dict(
    "sparse_fvm" => :circle,
    "ekf_fvm" => :star5,
    "sparse_collocation" => :diamond,
    "classical_fvm" => :utriangle,
)

const METHOD_LABELS = Dict(
    "sparse_fvm" => "Sparse GP-FVM",
    "ekf_fvm" => "EKF GP-FVM",
    "sparse_collocation" => "Sparse GP-Collocation",
    "classical_fvm" => "Classical FVM",
)

"""
    pareto_plot(results::DataFrame; kwargs...)

Create Pareto frontier plot: L2 error vs wall-clock time.

# Arguments
- `results`: DataFrame with columns: method, N, time_s, mean_l2_error (and optionally std columns)

# Keyword arguments
- `aggregate=true`: If true, aggregate over IC seeds showing mean ± std
- `title="Accuracy vs Compute"`: Plot title
- `filename=nothing`: If provided, save figure to this path
"""
function pareto_plot(results::DataFrame;
                      aggregate::Bool=true,
                      title::String="Accuracy vs Compute",
                      filename::Union{Nothing,String}=nothing)
    fig = Figure(size=(800, 600), fontsize=14)
    ax = Axis(fig[1, 1],
        xlabel = "Wall-clock time (s)",
        ylabel = "Relative L2 error",
        xscale = log10,
        yscale = log10,
        title = title
    )

    methods = unique(results.method)

    for method in methods
        method_data = filter(row -> row.method == method, results)

        if aggregate
            # Group by N and aggregate
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

            # Plot with error bars
            scatter!(ax, times, errors,
                color = METHOD_COLORS[method],
                marker = METHOD_MARKERS[method],
                markersize = 12,
                label = METHOD_LABELS[method]
            )

            # Connect points with lines
            lines!(ax, times, errors,
                color = METHOD_COLORS[method],
                linewidth = 2
            )

            # Add error bars if we have multiple samples
            # On log scale, we need asymmetric error bars to avoid going negative
            if any(grouped.error_std .> 0)
                # Upper error is just std
                error_high = grouped.error_std
                # Lower error is clamped so we don't go below 10% of the mean (stays positive on log scale)
                error_low = min.(grouped.error_std, errors .* 0.9)

                errorbars!(ax, times, errors, error_low, error_high,
                    color = METHOD_COLORS[method],
                    linewidth = 1
                )
            end

            # Label points with N
            for row in eachrow(grouped)
                text!(ax, row.time_mean, row.error_mean,
                    text = "N=$(row.N)",
                    fontsize = 9,
                    offset = (5, 5)
                )
            end
        else
            # Plot all points individually
            scatter!(ax, method_data.time_s, method_data.mean_l2_error,
                color = METHOD_COLORS[method],
                marker = METHOD_MARKERS[method],
                markersize = 8,
                label = METHOD_LABELS[method]
            )
        end
    end

    axislegend(ax, position = :rt)

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig)
        println("Saved: $filename")
    end

    return fig
end

"""
    scaling_plot(results::DataFrame; kwargs...)

Create scaling plot: wall-clock time vs grid size N on log-log scale.

Shows O(N) vs O(N³) scaling.
"""
function scaling_plot(results::DataFrame;
                       title::String="Computational Scaling",
                       filename::Union{Nothing,String}=nothing)
    fig = Figure(size=(800, 600), fontsize=14)
    ax = Axis(fig[1, 1],
        xlabel = "Grid size N",
        ylabel = "Wall-clock time (s)",
        xscale = log10,
        yscale = log10,
        title = title
    )

    methods = unique(results.method)

    for method in methods
        method_data = filter(row -> row.method == method, results)

        # Aggregate by N
        grouped = combine(
            groupby(method_data, :N),
            :time_s => mean => :time_mean,
            :time_s => std => :time_std
        )
        sort!(grouped, :N)

        Ns = grouped.N
        times = grouped.time_mean

        scatter!(ax, Ns, times,
            color = METHOD_COLORS[method],
            marker = METHOD_MARKERS[method],
            markersize = 12,
            label = METHOD_LABELS[method]
        )

        lines!(ax, Ns, times,
            color = METHOD_COLORS[method],
            linewidth = 2
        )
    end

    # Add reference slopes
    N_ref = [minimum(results.N), maximum(results.N)]
    t_base = 0.01  # Adjust based on data

    # O(N) reference line
    lines!(ax, N_ref, t_base .* (N_ref ./ N_ref[1]),
        color = :gray, linestyle = :dash, linewidth = 1,
        label = "O(N)"
    )

    # O(N³) reference line
    lines!(ax, N_ref, t_base .* (N_ref ./ N_ref[1]).^3,
        color = :gray, linestyle = :dot, linewidth = 1,
        label = "O(N³)"
    )

    axislegend(ax, position = :lt)

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig)
        println("Saved: $filename")
    end

    return fig
end

"""
    calibration_plot(results::DataFrame; kwargs...)

Create calibration plot: empirical coverage vs nominal coverage.

Shows if uncertainty quantification is well-calibrated.
"""
function calibration_plot(results::DataFrame;
                           nominal_levels::Vector{Float64}=[0.5, 0.8, 0.9, 0.95, 0.99],
                           title::String="UQ Calibration",
                           filename::Union{Nothing,String}=nothing)
    fig = Figure(size=(700, 600), fontsize=14)
    ax = Axis(fig[1, 1],
        xlabel = "Nominal coverage",
        ylabel = "Empirical coverage",
        title = title,
        aspect = 1
    )

    # Perfect calibration line
    lines!(ax, [0, 1], [0, 1],
        color = :black, linestyle = :dash, linewidth = 1,
        label = "Perfect calibration"
    )

    # For now, we only have 95% coverage in the data
    # This would need extension to compute coverage at multiple levels
    methods = unique(results.method)

    for method in methods
        method_data = filter(row -> row.method == method, results)

        # Mean coverage across all runs
        mean_coverage = mean(method_data.coverage_95)

        # Plot single point at 95% nominal
        scatter!(ax, [0.95], [mean_coverage],
            color = METHOD_COLORS[method],
            marker = METHOD_MARKERS[method],
            markersize = 15,
            label = METHOD_LABELS[method]
        )
    end

    xlims!(ax, 0.4, 1.0)
    ylims!(ax, 0.4, 1.0)
    axislegend(ax, position = :rb)

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig)
        println("Saved: $filename")
    end

    return fig
end

"""
    convergence_plot(results::DataFrame; kwargs...)

Create convergence plot: L2 error vs grid size N.

Shows discretization convergence rate.
"""
function convergence_plot(results::DataFrame;
                           title::String="Discretization Convergence",
                           filename::Union{Nothing,String}=nothing)
    fig = Figure(size=(800, 600), fontsize=14)
    ax = Axis(fig[1, 1],
        xlabel = "Grid size N",
        ylabel = "Relative L2 error",
        xscale = log10,
        yscale = log10,
        title = title
    )

    methods = unique(results.method)

    for method in methods
        method_data = filter(row -> row.method == method, results)

        # Aggregate by N
        grouped = combine(
            groupby(method_data, :N),
            :mean_l2_error => mean => :error_mean,
            :mean_l2_error => std => :error_std
        )
        sort!(grouped, :N)

        Ns = grouped.N
        errors = grouped.error_mean

        scatter!(ax, Ns, errors,
            color = METHOD_COLORS[method],
            marker = METHOD_MARKERS[method],
            markersize = 12,
            label = METHOD_LABELS[method]
        )

        lines!(ax, Ns, errors,
            color = METHOD_COLORS[method],
            linewidth = 2
        )
    end

    # Add reference slopes
    N_ref = [minimum(results.N), maximum(results.N)]
    e_base = maximum(results.mean_l2_error)

    # O(1/N) = O(Δx) first-order convergence
    lines!(ax, N_ref, e_base .* (N_ref[1] ./ N_ref),
        color = :gray, linestyle = :dash, linewidth = 1,
        label = "O(1/N)"
    )

    # O(1/N²) = O(Δx²) second-order convergence
    lines!(ax, N_ref, e_base .* (N_ref[1] ./ N_ref).^2,
        color = :gray, linestyle = :dot, linewidth = 1,
        label = "O(1/N²)"
    )

    axislegend(ax, position = :rt)

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig)
        println("Saved: $filename")
    end

    return fig
end

"""
    summary_table(results::DataFrame)

Create summary table of results aggregated by method and N.
"""
function summary_table(results::DataFrame)
    summary = combine(
        groupby(results, [:method, :N]),
        :time_s => mean => :time_mean,
        :time_s => std => :time_std,
        :mean_l2_error => mean => :error_mean,
        :mean_l2_error => std => :error_std,
        :coverage_95 => mean => :coverage_mean,
        :fillin_pct => mean => :fillin_mean,
        nrow => :n_samples
    )
    sort!(summary, [:method, :N])
    return summary
end

"""
    results_to_dataframe(metrics::Vector{MetricsSummary})

Convert vector of MetricsSummary to DataFrame.
"""
function results_to_dataframe(metrics::Vector{MetricsSummary})
    rows = [metrics_to_namedtuple(m) for m in metrics]
    return DataFrame(rows)
end
