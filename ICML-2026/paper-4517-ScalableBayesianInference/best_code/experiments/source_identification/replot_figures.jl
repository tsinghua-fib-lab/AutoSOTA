"""
Replot the calibration + scalability figures using a TuePlots theme,
reading saved outputs from `results/` rather than re-running the experiments.

Run from the repo root:

    julia --project=. experiments/source_identification/replot_figures.jl
"""

using CairoMakie
using TuePlots
using CSV, DataFrames
using NPZ

const OUT_DIR = get(ENV, "GPFVM_FIGURES_DIR", joinpath(@__DIR__, "figures"))
const CALIB_DIR = joinpath(@__DIR__, "results", "calibration")
const SCAL_DIR = joinpath(@__DIR__, "results", "scalability")

# ──────────────────────────────────────────────────────────────────────────────
# Theme helpers
# ──────────────────────────────────────────────────────────────────────────────

function icml_theme(; single_column=true, nrows=1, ncols=1, ratio=0.75)
    Theme(
        TuePlots.SETTINGS[:ICML];
        font=true, fontsize=true, figsize=true,
        single_column=single_column,
        nrows=nrows, ncols=ncols,
        subplot_height_to_width_ratio=ratio,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: calibration_plot.pdf  (Appendix C)
# ──────────────────────────────────────────────────────────────────────────────

function replot_calibration()
    df = CSV.read(joinpath(CALIB_DIR, "coverage.csv"), DataFrame)
    nominal = df.nominal
    emp_s   = df.source_coverage
    emp_c   = df.conc_coverage

    set_theme!(icml_theme(single_column=true, nrows=1, ncols=1, ratio=1.0))
    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel="Nominal coverage",
        ylabel="Empirical coverage",
        aspect=1,
    )
    xlims!(ax, 0.45, 1.0); ylims!(ax, 0.45, 1.0)
    lines!(ax, [0.45, 1.0], [0.45, 1.0]; color=:gray60, linestyle=:dash, linewidth=1, label="Perfect")
    scatterlines!(ax, nominal, emp_s; color=:crimson, linewidth=1.6, markersize=6, label="Source")
    scatterlines!(ax, nominal, emp_c; color=:royalblue, linewidth=1.6, markersize=6, label="Concentration")
    axislegend(ax; position=:lt, framevisible=false)

    out = joinpath(OUT_DIR, "calibration_plot.pdf")
    save(out, fig, pt_per_unit=1)
    set_theme!()
    @info "Wrote $out"
end

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: spatial_coverage_95.pdf  (Appendix C)
# ──────────────────────────────────────────────────────────────────────────────

function replot_spatial_coverage()
    d = npzread(joinpath(CALIB_DIR, "calibration_data.npz"))
    xs = d["xs"]
    ys = d["ys"]
    cov_s = d["spatial_s_cov_95"]
    cov_c = d["spatial_c_cov_95"]

    set_theme!(icml_theme(single_column=true, nrows=1, ncols=2, ratio=1.0))
    fig = Figure()
    ax1 = Axis(fig[1, 1]; aspect=DataAspect(), title="Source",       xlabel=L"x", ylabel=L"y")
    ax2 = Axis(fig[1, 2]; aspect=DataAspect(), title="Concentration", xlabel=L"x",
               yticklabelsvisible=false, yticksvisible=false)
    hm = heatmap!(ax1, xs, ys, cov_s; colormap=:RdBu, colorrange=(0.8, 1.0))
    heatmap!(ax2, xs, ys, cov_c; colormap=:RdBu, colorrange=(0.8, 1.0))
    Colorbar(fig[1, 3], hm; label="Empirical coverage at 95% nominal", width=8)

    out = joinpath(OUT_DIR, "spatial_coverage_95.pdf")
    save(out, fig, pt_per_unit=1)
    set_theme!()
    @info "Wrote $out"
end

# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: scaling_timing_Ns.pdf  (Appendix D)
# ──────────────────────────────────────────────────────────────────────────────

function replot_scaling_timing()
    df = CSV.read(joinpath(SCAL_DIR, "scalability_results.csv"), DataFrame)
    Ns = df.n_total
    t_total = df.time_total_s
    t_sparse = hasproperty(df, :time_sparse_prec_c_s) ? df.time_sparse_prec_c_s : nothing
    t_cond   = hasproperty(df, :time_conditioning_s) ? df.time_conditioning_s : nothing
    t_post   = hasproperty(df, :time_posterior_stats_s) ? df.time_posterior_stats_s : nothing

    set_theme!(icml_theme(single_column=true, nrows=1, ncols=1, ratio=0.75))
    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel=L"Total DOF $N_s$",
        ylabel="Wall-clock time (s)",
        xscale=log10, yscale=log10,
    )

    # Reference line O(N^{3/2})
    Ns_ref = collect(range(extrema(Ns)...; length=20))
    ref_anchor = t_total[1] / (Ns[1]^1.5)
    lines!(ax, Ns_ref, ref_anchor .* Ns_ref .^ 1.5;
           color=:gray60, linestyle=:dash, linewidth=1, label=L"\mathcal{O}(N_s^{3/2})")

    scatterlines!(ax, Ns, t_total;   color=:black,      linewidth=1.6, markersize=6, label="Total")
    t_sparse !== nothing && scatterlines!(ax, Ns, t_sparse; color=:royalblue, linewidth=1.4, markersize=5, label="Sparse prec.")
    t_cond   !== nothing && scatterlines!(ax, Ns, t_cond;   color=:darkorange, linewidth=1.4, markersize=5, label="Conditioning")
    t_post   !== nothing && scatterlines!(ax, Ns, t_post;   color=:purple,     linewidth=1.4, markersize=5, label="Posterior stats")

    axislegend(ax; position=:lt, framevisible=false)

    out = joinpath(OUT_DIR, "scaling_timing_Ns.pdf")
    save(out, fig, pt_per_unit=1)
    set_theme!()
    @info "Wrote $out"
end

# ──────────────────────────────────────────────────────────────────────────────

function main()
    isdir(OUT_DIR) || mkpath(OUT_DIR)
    replot_calibration()
    replot_spatial_coverage()
    replot_scaling_timing()
end

main()
