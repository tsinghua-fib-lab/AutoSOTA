"""
Visualization for Burgers source identification.
"""

using TuePlots

function _icml_theme(; single_column=false, nrows=1, ncols=1, ratio=0.85)
    Theme(
        TuePlots.SETTINGS[:ICML];
        font=true, fontsize=true, figsize=true,
        single_column=single_column,
        nrows=nrows, ncols=ncols,
        subplot_height_to_width_ratio=ratio,
    )
end

function plot_source_comparison(result, s_true, xs, ys;
        filename::Union{Nothing,String}=nothing)
    set_theme!(_icml_theme(single_column=false, nrows=1, ncols=4, ratio=1.0))
    fig = Figure()

    s_err = abs.(s_true .- result.s_mean)
    vmin = min(minimum(s_true), minimum(result.s_mean))
    vmax = max(maximum(s_true), maximum(result.s_mean))

    ax1 = Axis(fig[1,1]; aspect=DataAspect(), title="True source", xlabel=L"x", ylabel=L"y")
    hm1 = heatmap!(ax1, xs, ys, s_true; colorrange=(vmin, vmax), colormap=:viridis)

    ax2 = Axis(fig[1,2]; aspect=DataAspect(), title="Posterior mean", xlabel=L"x", yticklabelsvisible=false)
    heatmap!(ax2, xs, ys, result.s_mean; colorrange=(vmin, vmax), colormap=:viridis)
    Colorbar(fig[1,3], hm1; width=8)

    ax3 = Axis(fig[1,4]; aspect=DataAspect(), title="|Error|", xlabel=L"x", yticklabelsvisible=false)
    hm3 = heatmap!(ax3, xs, ys, s_err; colormap=:inferno)
    Colorbar(fig[1,5], hm3; width=8)

    ax4 = Axis(fig[1,6]; aspect=DataAspect(), title="Posterior std", xlabel=L"x", yticklabelsvisible=false)
    hm4 = heatmap!(ax4, xs, ys, result.s_std; colormap=:inferno)
    Colorbar(fig[1,7], hm4; width=8)

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig, pt_per_unit=1)
    end
    set_theme!()
    return fig
end

function plot_solution_snapshots(result, u_truth, xs, ys, ts;
        snapshot_times=nothing, filename::Union{Nothing,String}=nothing,
        obs_data=nothing)
    n_t_total = size(u_truth, 3)
    if isnothing(snapshot_times)
        snapshot_times = round.(Int, range(1, n_t_total, length=4))
    end
    n_snap = length(snapshot_times)

    set_theme!(_icml_theme(single_column=false, nrows=3, ncols=n_snap, ratio=1.0))
    fig = Figure()

    for (col, t) in enumerate(snapshot_times)
        vmin = min(minimum(u_truth[:,:,t]), minimum(result.u_mean[:,:,t]))
        vmax = max(maximum(u_truth[:,:,t]), maximum(result.u_mean[:,:,t]))

        ax1 = Axis(fig[1, col]; aspect=DataAspect(),
            title=L"t=%$(round(ts[t], digits=2))",
            ylabel=col==1 ? "Truth" : "", yticklabelsvisible=col==1)
        hm = heatmap!(ax1, xs, ys, u_truth[:,:,t]; colorrange=(vmin, vmax), colormap=:viridis)

        ax2 = Axis(fig[2, col]; aspect=DataAspect(),
            ylabel=col==1 ? "GP-FVM" : "", yticklabelsvisible=col==1)
        heatmap!(ax2, xs, ys, result.u_mean[:,:,t]; colorrange=(vmin, vmax), colormap=:viridis)

        if col == n_snap
            Colorbar(fig[1:2, n_snap+1], hm; width=8)
        end

        ax3 = Axis(fig[3, col]; aspect=DataAspect(),
            ylabel=col==1 ? "Std" : "", yticklabelsvisible=col==1)
        hm_std = heatmap!(ax3, xs, ys, result.u_std[:,:,t]; colormap=:inferno)

        # Show observation locations at timesteps where we observe
        if !isnothing(obs_data) && t in obs_data.obs_timesteps
            scatter!(ax3, obs_data.obs_x, obs_data.obs_y;
                color=:white, markersize=3, strokewidth=0.4, strokecolor=:black)
        end

        if col == n_snap
            Colorbar(fig[3, n_snap+1], hm_std; width=8)
        end
    end

    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig, pt_per_unit=1)
    end
    set_theme!()
    return fig
end

function plot_convergence(iteration_log; filename::Union{Nothing,String}=nothing)
    fig = Figure(size=(500, 350), fontsize=12)
    ax = Axis(fig[1, 1]; xlabel="Gauss-Newton iteration", ylabel="Newton decrement",
              yscale=log10, title="Convergence of nonlinear solve")
    scatterlines!(ax, 1:length(iteration_log), iteration_log;
        color=:royalblue, linewidth=2, markersize=6)
    if !isnothing(filename)
        mkpath(dirname(filename))
        save(filename, fig, px_per_unit=3)
    end
    return fig
end
