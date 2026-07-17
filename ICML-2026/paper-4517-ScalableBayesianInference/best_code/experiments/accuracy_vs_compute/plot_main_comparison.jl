"""
    Main Paper Figure: Accuracy vs Compute Comparison

Creates a full-width figure with two panels:
- (a) Discretization convergence: L2 error vs grid size N
- (b) Pareto frontier: L2 error vs wall time

Uses results from run.jl with multiple IC seeds to show error bars (std across seeds).

Usage:
    julia --project=../.. plot_main_comparison.jl --results results/accuracy_vs_compute/results_*.csv
    julia --project=../.. plot_main_comparison.jl --results results/accuracy_vs_compute/results_*.csv --output figures/
"""

using CSV
using DataFrames
using CairoMakie
using TuePlots
using Statistics
using ArgParse

# ==============================================================================
# Style Constants
# ==============================================================================

const COLORS = Dict(
    "sparse_fvm" => colorant"#0072B2",        # Blue
    "sparse_collocation" => colorant"#D55E00", # Vermillion/orange
    "classical_fvm" => colorant"#009E73"       # Teal/green
)

const MARKERS = Dict(
    "sparse_fvm" => :circle,
    "sparse_collocation" => :diamond,
    "classical_fvm" => :utriangle
)

const LABELS = Dict(
    "sparse_fvm" => "GP-FVM",
    "sparse_collocation" => "GP-Collocation",
    "classical_fvm" => "Classical FVM"
)

# ==============================================================================
# Helper Functions
# ==============================================================================

"""
Add panel label (a), (b), etc. at top-center of axis.
"""
function add_panel_label!(ax, label; fontsize=8)
    text!(ax, 0.5, 0.97, text=label, align=(:center, :top),
          fontsize=fontsize, font=:bold, space=:relative)
end

"""
Aggregate results across IC seeds: compute mean and std for each (method, N).

Returns DataFrame with columns:
- method, N
- mean_error, std_error (for final_l2_error)
- mean_time, std_time (for time_s)
"""
function aggregate_by_method_N(df)
    # Group by method and N, aggregate across IC families and seeds
    grouped = groupby(df, [:method, :N])

    result = combine(grouped,
        :final_l2_error => mean => :mean_error,
        :final_l2_error => std => :std_error,
        :time_s => mean => :mean_time,
        :time_s => std => :std_time,
        nrow => :n_samples
    )

    # Sort by N for proper line drawing
    sort!(result, [:method, :N])

    return result
end

# ==============================================================================
# Main Figure: Convergence + Pareto
# ==============================================================================

"""
    plot_main_comparison(csv_path; output_path, height)

Create main paper figure: (a) Convergence, (b) Pareto frontier.
Full-width, 2 panels side-by-side, with error bars across IC seeds.
"""
function plot_main_comparison(csv_path::String;
                               output_path::String="figures/accuracy_comparison.pdf",
                               height::Float64=0.7)

    df = CSV.read(csv_path, DataFrame)

    # Aggregate across IC seeds
    agg = aggregate_by_method_N(df)

    println("\nAggregated data:")
    for method in unique(agg.method)
        mdata = filter(r -> r.method == method, agg)
        println("  $method: $(nrow(mdata)) grid sizes, $(mdata.n_samples[1]) samples each")
    end

    # TuePlots theme: full-width, 2 panels
    # For 2 panels side-by-side, each is ~3.375" wide, so ratio=0.7 gives ~2.4" height
    theme = Theme(
        TuePlots.SETTINGS[:ICML];
        font=true, fontsize=true, figsize=true,
        single_column=false, nrows=1, ncols=2,
        subplot_height_to_width_ratio=height
    )
    set_theme!(theme)

    fig = Figure()

    # Get unique N values for x-axis ticks
    all_N = sort(unique(agg.N))

    # ==========================================================================
    # (a) Convergence: Error vs N
    # ==========================================================================
    ax1 = Axis(fig[1,1],
        xlabel=L"Grid size $N$",
        ylabel=L"Relative $L^2$ error",
        xscale=log10, yscale=log10,
        xticks=(all_N, string.(all_N)))

    for method in ["sparse_fvm", "sparse_collocation", "classical_fvm"]
        mdata = filter(r -> r.method == method, agg)
        if nrow(mdata) == 0
            continue
        end

        # Plot line with markers
        scatterlines!(ax1, mdata.N, mdata.mean_error,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=6,
            label=LABELS[method])

        # Add error bars (std across IC seeds)
        # For log scale, we need asymmetric bars to avoid negative values
        err_low = min.(mdata.std_error, mdata.mean_error * 0.9)  # Clamp to 90% of mean
        err_high = mdata.std_error
        errorbars!(ax1, mdata.N, mdata.mean_error, err_low, err_high,
            color=COLORS[method],
            whiskerwidth=5,
            linewidth=1)
    end
    add_panel_label!(ax1, "(a)")

    # ==========================================================================
    # (b) Pareto: Error vs Time
    # ==========================================================================
    # Nice tick values for wall time
    time_ticks = [0.5, 1, 2, 5, 10, 20, 50]
    ax2 = Axis(fig[1,2],
        xlabel="Wall time (s)",
        ylabel=L"Relative $L^2$ error",
        xscale=log10, yscale=log10,
        xticks=(time_ticks, string.(time_ticks)),
        yticklabelsvisible=false)

    for method in ["sparse_fvm", "sparse_collocation", "classical_fvm"]
        mdata = filter(r -> r.method == method, agg)
        if nrow(mdata) == 0
            continue
        end

        # Plot line with markers
        scatterlines!(ax2, mdata.mean_time, mdata.mean_error,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=6,
            label=LABELS[method])

        # Add error bars (std across IC seeds)
        # For log scale, we need asymmetric bars to avoid negative values
        err_low = min.(mdata.std_error, mdata.mean_error * 0.9)
        err_high = mdata.std_error
        errorbars!(ax2, mdata.mean_time, mdata.mean_error, err_low, err_high,
            color=COLORS[method],
            whiskerwidth=5,
            linewidth=1)
    end
    add_panel_label!(ax2, "(b)")

    # Link y-axes
    linkyaxes!(ax1, ax2)

    # Single legend below
    Legend(fig[2, 1:2], ax1, orientation=:horizontal, tellheight=true, framevisible=false)

    # Save
    mkpath(dirname(output_path))
    save(output_path, fig, pt_per_unit=1)

    # Also save PNG preview
    png_path = replace(output_path, ".pdf" => ".png")
    save(png_path, fig, px_per_unit=3)

    set_theme!()  # Reset theme

    println("\nSaved: $output_path")
    println("Saved: $png_path")

    return fig
end

# ==============================================================================
# CLI
# ==============================================================================

function parse_commandline()
    s = ArgParseSettings(description = "Generate main paper figure for accuracy vs compute comparison")

    @add_arg_table! s begin
        "--results", "-r"
            help = "Path to results CSV file (from run.jl)"
            arg_type = String
            required = true
        "--output", "-o"
            help = "Output directory for figures"
            arg_type = String
            default = "figures"
        "--height"
            help = "Subplot height-to-width ratio (0.7 = slightly wide panels)"
            arg_type = Float64
            default = 0.7
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    csv_path = args["results"]
    output_dir = args["output"]

    println("="^60)
    println("Main Paper Figure: Accuracy vs Compute")
    println("="^60)
    println("Input: $csv_path")
    println("Output: $output_dir")

    mkpath(output_dir)

    output_path = joinpath(output_dir, "accuracy_comparison.pdf")
    plot_main_comparison(csv_path; output_path=output_path, height=args["height"])

    println("\nDone!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
