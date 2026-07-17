"""
    Appendix Figure: Hyperparameter Sensitivity

Creates a single-column figure with two panels showing hyperparameter effects:
- (a) Effect of Smoothness: Different Matérn orders for GP-FVM vs GP-Collocation
- (b) Effect of ρ: Different sparsity thresholds

Uses hyperparameter sweep data (single seed, multiple hyperparameter configs).

Usage:
    julia --project=../.. plot_hyperparam_sensitivity.jl --sweep results/hyperparam_sweep/sweep_*.csv
    julia --project=../.. plot_hyperparam_sensitivity.jl --sweep results/hyperparam_sweep/sweep_*.csv --output figures/
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

const SMOOTHNESS_LABELS = Dict(
    1 => "Matérn 3/2",
    2 => "Matérn 5/2",
    3 => "Matérn 7/2"
)

# ==============================================================================
# Helper Functions
# ==============================================================================

"""
Add panel label (a), (b), etc. at top-left corner of axis.
"""
function add_panel_label!(ax, label; fontsize=8)
    text!(ax, 0.03, 0.97, text=label, align=(:left, :top),
          fontsize=fontsize, font=:bold, space=:relative)
end

"""
Get best lengthscale per (method, N, smoothness, rho) config.
"""
function get_best_ls_per_config(df)
    # For classical_fvm, just return as-is
    classical = filter(r -> r.method == "classical_fvm", df)

    # For GP methods, group by (method, N, smoothness, rho) and pick best ls
    gp_df = filter(r -> r.method != "classical_fvm", df)

    result = copy(classical)

    for method in ["sparse_fvm", "sparse_collocation"]
        method_df = filter(r -> r.method == method, gp_df)
        if nrow(method_df) == 0
            continue
        end

        for N in unique(method_df.N)
            n_df = filter(r -> r.N == N, method_df)

            for smooth in unique(n_df.smoothness)
                s_df = filter(r -> r.smoothness == smooth, n_df)

                for rho in unique(s_df.rho)
                    r_df = filter(r -> r.rho == rho, s_df)

                    if nrow(r_df) > 0
                        best_idx = argmin(r_df.rel_l2_error)
                        push!(result, r_df[best_idx, :])
                    end
                end
            end
        end
    end

    return result
end

# ==============================================================================
# Appendix Figure: Hyperparameter Sensitivity
# ==============================================================================

"""
    plot_hyperparam_sensitivity(csv_path; output_path, height)

Create appendix figure: (a) Smoothness effect, (b) ρ effect.
Single-column, 2 panels stacked.
"""
function plot_hyperparam_sensitivity(csv_path::String;
                                      output_path::String="figures/hyperparam_sensitivity.pdf",
                                      height::Float64=3.5)

    df = CSV.read(csv_path, DataFrame)

    # Get best lengthscale per config
    config_df = get_best_ls_per_config(df)

    # Filter to GP methods only
    gp_df = filter(r -> r.method != "classical_fvm", config_df)

    # TuePlots theme: single-column, 2 panels stacked
    theme = Theme(
        TuePlots.SETTINGS[:ICML];
        font=true, fontsize=true, figsize=true,
        single_column=true, nrows=2, ncols=1,
        subplot_height_to_width_ratio=height / 3.25  # ICML single column is 3.25"
    )
    set_theme!(theme)

    fig = Figure()

    # Linestyles for methods
    method_linestyles = Dict(
        "sparse_fvm" => :solid,
        "sparse_collocation" => :dash
    )

    # ==========================================================================
    # (a) Effect of Smoothness - best ρ per (method, N, smoothness)
    # ==========================================================================
    ax1 = Axis(fig[1,1],
        ylabel=L"Relative $L^2$ error",
        xscale=log10, yscale=log10,
        xticklabelsvisible=false)

    # For each (method, smoothness), pick best ρ per N
    for method in ["sparse_fvm", "sparse_collocation"]
        method_df = filter(r -> r.method == method, gp_df)
        if nrow(method_df) == 0
            continue
        end

        available_smooth = sort(unique(method_df.smoothness))

        for smooth in available_smooth
            s_df = filter(r -> r.smoothness == smooth, method_df)

            # Pick best ρ for each N
            best_per_N = DataFrame()
            for N in unique(s_df.N)
                n_df = filter(r -> r.N == N, s_df)
                if nrow(n_df) > 0
                    best_idx = argmin(n_df.rel_l2_error)
                    push!(best_per_N, n_df[best_idx, :])
                end
            end

            sort!(best_per_N, :N)

            label_str = "$(LABELS[method]), $(SMOOTHNESS_LABELS[smooth])"

            scatterlines!(ax1, best_per_N.N, best_per_N.rel_l2_error,
                color=COLORS[method],
                linestyle=method_linestyles[method],
                marker=MARKERS[method],
                markersize=5,
                alpha=0.4 + 0.3 * smooth,  # Lighter = higher smoothness
                label=label_str)
        end
    end

    add_panel_label!(ax1, "(a)")
    axislegend(ax1, position=:rt, labelsize=6, framevisible=false)

    # ==========================================================================
    # (b) Effect of ρ - best smoothness per (method, N, ρ)
    # ==========================================================================
    ax2 = Axis(fig[2,1],
        xlabel=L"Grid size $N$",
        ylabel=L"Relative $L^2$ error",
        xscale=log10, yscale=log10)

    # Markers for different ρ values
    rho_markers = Dict(
        2.0 => :circle,
        3.0 => :diamond,
        4.0 => :utriangle,
        5.0 => :rect
    )

    for method in ["sparse_fvm", "sparse_collocation"]
        method_df = filter(r -> r.method == method, gp_df)
        if nrow(method_df) == 0
            continue
        end

        available_rhos = sort(unique(method_df.rho))

        for rho in available_rhos
            r_df = filter(r -> r.rho == rho, method_df)

            # Pick best smoothness for each N
            best_per_N = DataFrame()
            for N in unique(r_df.N)
                n_df = filter(r -> r.N == N, r_df)
                if nrow(n_df) > 0
                    best_idx = argmin(n_df.rel_l2_error)
                    push!(best_per_N, n_df[best_idx, :])
                end
            end

            sort!(best_per_N, :N)

            rho_int = Int(rho)
            label_str = "$(LABELS[method]), ρ=$rho_int"
            marker = get(rho_markers, rho, :star5)

            scatterlines!(ax2, best_per_N.N, best_per_N.rel_l2_error,
                color=COLORS[method],
                linestyle=method_linestyles[method],
                marker=marker,
                markersize=5,
                label=label_str)
        end
    end

    add_panel_label!(ax2, "(b)")
    axislegend(ax2, position=:rt, labelsize=6, framevisible=false)

    # Link axes
    linkyaxes!(ax1, ax2)

    # Save
    mkpath(dirname(output_path))
    save(output_path, fig, pt_per_unit=1)

    png_path = replace(output_path, ".pdf" => ".png")
    save(png_path, fig, px_per_unit=3)

    set_theme!()

    println("Saved: $output_path")
    println("Saved: $png_path")

    return fig
end

# ==============================================================================
# CLI
# ==============================================================================

function parse_commandline()
    s = ArgParseSettings(description = "Generate appendix figure for hyperparameter sensitivity analysis")

    @add_arg_table! s begin
        "--sweep", "-s"
            help = "Path to hyperparameter sweep CSV file"
            arg_type = String
            required = true
        "--output", "-o"
            help = "Output directory for figures"
            arg_type = String
            default = "figures"
        "--height"
            help = "Figure height in inches"
            arg_type = Float64
            default = 3.5
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    csv_path = args["sweep"]
    output_dir = args["output"]

    println("="^60)
    println("Appendix Figure: Hyperparameter Sensitivity")
    println("="^60)
    println("Input: $csv_path")
    println("Output: $output_dir")
    println()

    mkpath(output_dir)

    output_path = joinpath(output_dir, "hyperparam_sensitivity.pdf")
    plot_hyperparam_sensitivity(csv_path; output_path=output_path, height=args["height"])

    println("\nDone!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
