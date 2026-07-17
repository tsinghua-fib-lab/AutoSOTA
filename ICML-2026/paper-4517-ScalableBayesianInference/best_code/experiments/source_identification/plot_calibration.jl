"""
Calibration plot: nominal vs empirical coverage for concentration and source fields.

Reads scalability CSV results (which include coverage columns) and plots calibration curves.
Can overlay multiple runs (e.g., default vs calibrated output scale).

Usage:
    julia --project=../.. plot_calibration.jl results/scalability/default.csv results/scalability/calibrated.csv
    julia --project=../.. plot_calibration.jl results/scalability/scalability_results.csv
"""

using CSV, DataFrames
using CairoMakie
using Statistics
using Printf

function coverage_at_level(c_mean, c_std, c_true, z)
    Statistics.mean(abs.(c_true .- c_mean) .< z .* c_std)
end

"""
Extract (nominal, empirical) coverage pairs from a DataFrame row
at many confidence levels.
"""
function calibration_curve_from_row(row)
    # We only have 50/90/95 in the CSV. Return those.
    nominal = [0.50, 0.90, 0.95]
    empirical_c = [row.cov_c_50, row.cov_c_90, row.cov_c_95]
    empirical_s = [row.cov_s_50, row.cov_s_90, row.cov_s_95]
    return nominal, empirical_c, empirical_s
end

"""
Average calibration curve across all grid sizes (excluding very coarse).
Automatically detects which coverage levels are in the CSV.
"""
function average_calibration(df; min_N=11)
    df_filt = filter(r -> r.N >= min_N, df)

    # Detect coverage columns: cov_c_XX and cov_s_XX
    all_cols = names(df_filt)
    cov_c_cols = sort([c for c in all_cols if startswith(c, "cov_c_")])
    cov_s_cols = sort([c for c in all_cols if startswith(c, "cov_s_")])

    # Extract nominal levels from column names
    nominal = [parse(Int, split(c, "_")[end]) / 100.0 for c in cov_c_cols]

    avg_c = [Statistics.mean(df_filt[!, col]) for col in cov_c_cols]
    avg_s = [Statistics.mean(df_filt[!, col]) for col in cov_s_cols]

    std_c = [Statistics.std(df_filt[!, col]) for col in cov_c_cols]
    std_s = [Statistics.std(df_filt[!, col]) for col in cov_s_cols]

    return (; nominal, avg_c, avg_s, std_c, std_s)
end

function plot_calibration(csv_paths::Vector{String}, labels::Vector{String};
                          filename::Union{Nothing,String}=nothing, min_N::Int=11)

    fig = Figure(size=(800, 380), fontsize=12)

    colors = [:royalblue, :crimson, :seagreen, :darkorange]

    panels = [(:avg_c, :std_c, "Concentration"), (:avg_s, :std_s, "Source")]
    for (col_idx, (avg_key, std_key, title)) in enumerate(panels)

        ax = Axis(fig[1, col_idx];
            xlabel="Nominal coverage",
            ylabel="Empirical coverage",
            title=title,
            aspect=1,
        )
        xlims!(ax, 0.0, 1.0)
        ylims!(ax, 0.0, 1.0)

        # Diagonal (perfect calibration)
        lines!(ax, [0, 1], [0, 1]; color=:gray60, linestyle=:dash, linewidth=1,
               label="Perfect")

        for (i, (path, label)) in enumerate(zip(csv_paths, labels))
            df = CSV.read(path, DataFrame)
            cal = average_calibration(df; min_N=min_N)

            avg = getfield(cal, avg_key)
            std_vals = getfield(cal, std_key)

            scatterlines!(ax, cal.nominal, avg;
                color=colors[i], linewidth=2, markersize=8, label=label)

            # Error bars (±1 std across grid sizes)
            errorbars!(ax, cal.nominal, avg, std_vals;
                color=colors[i], whiskerwidth=8, linewidth=1)
        end

        axislegend(ax; position=:lt)
    end

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig, px_per_unit=3)
        println("Saved: $filename")
    end

    return fig
end

# CLI
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia plot_calibration.jl <csv1> [csv2] [--labels label1,label2] [-o output.pdf]")
        exit(1)
    end

    # Parse args: CSV files, optional --labels, optional -o
    local csv_paths = String[]
    local labels = String[]
    local output = joinpath(@__DIR__, "results", "scalability", "calibration.pdf")

    local idx = 1
    while idx <= length(ARGS)
        if ARGS[idx] == "--labels"
            idx += 1
            labels = String.(split(ARGS[idx], ","))
        elseif ARGS[idx] == "-o"
            idx += 1
            output = ARGS[idx]
        else
            push!(csv_paths, ARGS[idx])
        end
        idx += 1
    end

    if isempty(labels)
        labels = ["Run $j" for j in 1:length(csv_paths)]
    end

    plot_calibration(csv_paths, labels; filename=output)
end
