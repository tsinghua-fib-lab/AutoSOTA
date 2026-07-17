"""
Paper figure: Upstream vs Downstream observation comparison.

Shows that posterior uncertainty correctly reflects information flow direction.
With advection-dominated transport, downstream observations constrain the source
well, while upstream observations provide little information.

Usage:
    julia --project=../.. figure_upstream_downstream.jl
"""

using NPZ
using CairoMakie

function main()
    # Load results
    println("Loading results...")

    down_data = npzread("data/downstream_only.npz")
    down_gpfvm = npzread("results/downstream_only_gpfvm.npz")

    up_data = npzread("data/upstream_only.npz")
    up_gpfvm = npzread("results/upstream_only_gpfvm.npz")

    # Grid (same for both)
    xs = down_data["xs"]
    ys = down_data["ys"]

    # True source (same for both)
    s_true = down_data["s_int_true"]
    source_x = down_data["source_x"][1]
    source_y = down_data["source_y"][1]

    # Cell centers for true source
    cx = 0.5 .* (xs[1:end-1] .+ xs[2:end])
    cy = 0.5 .* (ys[1:end-1] .+ ys[2:end])
    Δx, Δy = xs[2] - xs[1], ys[2] - ys[1]
    s_true_density = s_true ./ (Δx * Δy)

    # Observations
    down_obs_x, down_obs_y = down_data["obs_x"], down_data["obs_y"]
    up_obs_x, up_obs_y = up_data["obs_x"], up_data["obs_y"]

    # GP-FVM results
    down_s_mean = down_gpfvm["s_mean"]
    down_s_std = down_gpfvm["s_std"]
    up_s_mean = up_gpfvm["s_mean"]
    up_s_std = up_gpfvm["s_std"]

    # Shared color ranges for fair comparison
    s_mean_range = (
        min(minimum(down_s_mean), minimum(up_s_mean), 0),
        max(maximum(down_s_mean), maximum(up_s_mean))
    )
    s_std_range = (
        min(minimum(down_s_std), minimum(up_s_std)),
        max(maximum(down_s_std), maximum(up_s_std))
    )

    # Create figure - 2x3 layout
    fig = Figure(size=(800, 500))

    # Row 1: Downstream observations
    ax11 = Axis(fig[1, 1], xlabel="x", ylabel="y", title="True source", aspect=DataAspect())
    hm11 = heatmap!(ax11, cx, cy, s_true_density, colormap=:hot)
    scatter!(ax11, [source_x], [source_y], color=:cyan, markersize=10,
             marker=:star5, strokewidth=1, strokecolor=:black)
    scatter!(ax11, down_obs_x, down_obs_y, color=:white, markersize=8,
             strokewidth=1.5, strokecolor=:black, marker=:circle)
    # Flow arrow
    arrows!(ax11, [0.1], [0.1], [0.15], [0.0], color=:white, linewidth=2, arrowsize=10)
    text!(ax11, 0.18, 0.05, text="flow", color=:white, fontsize=10)

    ax12 = Axis(fig[1, 2], xlabel="x", ylabel="y", title="Inferred (downstream obs)", aspect=DataAspect())
    hm12 = heatmap!(ax12, xs, ys, down_s_mean, colormap=:hot, colorrange=s_mean_range)
    scatter!(ax12, [source_x], [source_y], color=:cyan, markersize=10,
             marker=:star5, strokewidth=1, strokecolor=:black)
    scatter!(ax12, down_obs_x, down_obs_y, color=:white, markersize=8,
             strokewidth=1.5, strokecolor=:black, marker=:circle)

    ax13 = Axis(fig[1, 3], xlabel="x", ylabel="y", title="Std (downstream obs)", aspect=DataAspect())
    hm13 = heatmap!(ax13, xs, ys, down_s_std, colormap=:viridis, colorrange=s_std_range)
    scatter!(ax13, [source_x], [source_y], color=:white, markersize=10,
             marker=:star5, strokewidth=1, strokecolor=:black)
    scatter!(ax13, down_obs_x, down_obs_y, color=:white, markersize=8,
             strokewidth=1.5, strokecolor=:black, marker=:circle)

    # Row 2: Upstream observations
    ax21 = Axis(fig[2, 1], xlabel="x", ylabel="y", aspect=DataAspect())
    hm21 = heatmap!(ax21, cx, cy, s_true_density, colormap=:hot)
    scatter!(ax21, [source_x], [source_y], color=:cyan, markersize=10,
             marker=:star5, strokewidth=1, strokecolor=:black)
    scatter!(ax21, up_obs_x, up_obs_y, color=:white, markersize=8,
             strokewidth=1.5, strokecolor=:black, marker=:circle)
    arrows!(ax21, [0.1], [0.1], [0.15], [0.0], color=:white, linewidth=2, arrowsize=10)
    text!(ax21, 0.18, 0.05, text="flow", color=:white, fontsize=10)

    ax22 = Axis(fig[2, 2], xlabel="x", ylabel="y", title="Inferred (upstream obs)", aspect=DataAspect())
    hm22 = heatmap!(ax22, xs, ys, up_s_mean, colormap=:hot, colorrange=s_mean_range)
    scatter!(ax22, [source_x], [source_y], color=:cyan, markersize=10,
             marker=:star5, strokewidth=1, strokecolor=:black)
    scatter!(ax22, up_obs_x, up_obs_y, color=:white, markersize=8,
             strokewidth=1.5, strokecolor=:black, marker=:circle)

    ax23 = Axis(fig[2, 3], xlabel="x", ylabel="y", title="Std (upstream obs)", aspect=DataAspect())
    hm23 = heatmap!(ax23, xs, ys, up_s_std, colormap=:viridis, colorrange=s_std_range)
    scatter!(ax23, [source_x], [source_y], color=:white, markersize=10,
             marker=:star5, strokewidth=1, strokecolor=:black)
    scatter!(ax23, up_obs_x, up_obs_y, color=:white, markersize=8,
             strokewidth=1.5, strokecolor=:black, marker=:circle)

    # Colorbars
    Colorbar(fig[1:2, 4], hm12, label="s")
    Colorbar(fig[1:2, 5], hm13, label="σ")

    # Save
    output_path = "figures/paper_upstream_downstream.pdf"
    mkpath(dirname(output_path))
    save(output_path, fig)
    println("Saved: $output_path")

    # Also save PNG for quick viewing
    save("figures/paper_upstream_downstream.png", fig, px_per_unit=3)
    println("Saved: figures/paper_upstream_downstream.png")

    # Print summary stats
    println("\n" * "="^50)
    println("Summary")
    println("="^50)
    println("Downstream observations: $(length(down_obs_x)) points at x ≈ $(round(mean(down_obs_x), digits=2))")
    println("  Source std range: [$(round(minimum(down_s_std), digits=3)), $(round(maximum(down_s_std), digits=3))]")
    println("  Mean std at source location: $(round(down_s_std[argmin(abs.(xs .- source_x)), argmin(abs.(ys .- source_y))], digits=3))")

    println("\nUpstream observations: $(length(up_obs_x)) points at x ≈ $(round(mean(up_obs_x), digits=2))")
    println("  Source std range: [$(round(minimum(up_s_std), digits=3)), $(round(maximum(up_s_std), digits=3))]")
    println("  Mean std at source location: $(round(up_s_std[argmin(abs.(xs .- source_x)), argmin(abs.(ys .- source_y))], digits=3))")

    return fig
end

# Need this for mean()
using Statistics

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
