"""
QQ plot of z-scores against N(0,1) for UQ calibration diagnostic.

Reads z-score NPZ files saved by the scalability study.

Usage:
    julia --project=../.. plot_qq.jl results/scalability/zscores/zscores_N31.npz
    julia --project=../.. plot_qq.jl results/scalability/zscores/zscores_N31.npz results/calibrated/zscores/zscores_N31.npz --labels "Default,Calibrated"
"""

using NPZ
using CairoMakie
using Statistics
using SpecialFunctions

# Normal quantile function (inverse CDF of N(0,1))
norminv(p) = √2 * erfinv(2p - 1)

function qq_theoretical(z_empirical)
    n = length(z_empirical)
    z_sorted = sort(z_empirical)
    # Theoretical quantiles: use (i - 0.5) / n to avoid ±∞
    p = [(i - 0.5) / n for i in 1:n]
    z_theoretical = norminv.(p)
    return z_theoretical, z_sorted
end

function plot_qq(npz_paths::Vector{String}, labels::Vector{String};
                 filename::Union{Nothing,String}=nothing,
                 grid_size::Union{Nothing,Int}=nothing)

    fig = Figure(size=(800, 380), fontsize=12)

    colors = [:royalblue, :crimson, :seagreen, :darkorange]

    for (col_idx, (field, title)) in enumerate([("z_c", "Concentration"), ("z_s", "Source")])
        ax = Axis(fig[1, col_idx];
            xlabel="Theoretical quantiles (N(0,1))",
            ylabel="Empirical quantiles",
            title=title,
            aspect=1,
        )

        # Reference diagonal
        lines!(ax, [-4, 4], [-4, 4]; color=:gray60, linestyle=:dash, linewidth=1,
               label="N(0,1)")

        for (i, (path, label)) in enumerate(zip(npz_paths, labels))
            data = npzread(path)
            z = vec(data[field])
            z_th, z_emp = qq_theoretical(z)

            # Subsample for plotting if too many points
            if length(z_th) > 500
                idx = round.(Int, range(1, length(z_th), length=500))
                z_th = z_th[idx]
                z_emp = z_emp[idx]
            end

            scatter!(ax, z_th, z_emp; color=(colors[i], 0.6), markersize=3,
                     label="$label (σ̂=$(round(std(vec(data[field])), digits=2)))")
        end

        xlims!(ax, -4, 4)
        ylims!(ax, -4, 4)
        axislegend(ax; position=:lt, labelsize=10)
    end

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig, px_per_unit=3)
        println("Saved: $filename")
    end

    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    local npz_paths = String[]
    local labels = String[]
    local output = joinpath(@__DIR__, "results", "scalability", "qq_plot.pdf")

    local idx = 1
    while idx <= length(ARGS)
        if ARGS[idx] == "--labels"
            idx += 1
            labels = String.(split(ARGS[idx], ","))
        elseif ARGS[idx] == "-o"
            idx += 1
            output = ARGS[idx]
        else
            push!(npz_paths, ARGS[idx])
        end
        idx += 1
    end

    if isempty(labels)
        labels = ["Run $j" for j in 1:length(npz_paths)]
    end

    plot_qq(npz_paths, labels; filename=output)
end
