"""
Visualization script for nonlinear SWE solution with Matérn bathymetry.
"""

using CairoMakie
using GPFiniteVolume: TimeStack, absindices

include("main.jl")

function visualize_swe_solution(;
    N_x = 15,
    N_y = 15,
    n_timesteps = 15,  # 15 minutes total
    bathymetry_lengthscale = 20.0,
    bathymetry_std = 40.0,
    seed = 42,
    prob = NonlinearSWEProblem(
        h_center_x = 75.0u"km",   # Offshore (deep water)
        h_center_y = 50.0u"km",   # Centered in y
        h_amplitude = 10.0u"m",   # Tsunami amplitude
        h_width_x = 5.0u"km",     # Narrow perpendicular to shore
        h_width_y = 25.0u"km"     # Wide along shore (fault-like)
    )
)
    println("Running solver...")
    result = solve_nonlinear_swe(;
        N_x = N_x,
        N_y = N_y,
        n_timesteps = n_timesteps,
        bathymetry_lengthscale = bathymetry_lengthscale,
        bathymetry_std = bathymetry_std,
        bathymetry_seed = seed,
        prob = prob,
        verbose = true
    )

    # Extract solution
    sol = result.solution
    μ = mean(sol)
    ginfo = result.ginfo
    layout = result.spatial_layout
    p = result.params

    # Get timestep accessor
    means = TimeStack(μ, result.full_state_layout)

    # Extract h at cell centers for each timestep
    # We'll use cell integrals divided by cell area as proxy for cell average
    cell_area = ginfo.Δx * ginfo.Δy

    h_fields = []
    for t in 1:n_timesteps  # TimeStack uses 1-based indexing, 1 to n_timesteps
        h_int = means[:h_int, t]
        h_avg = h_int ./ cell_area  # Convert integral to average
        push!(h_fields, reshape(h_avg, ginfo.n_cells_x, ginfo.n_cells_y))
    end

    # Compute surface elevation η = h - H₀
    η_fields = []
    depth_2d = reshape(p.depth_at_cells, ginfo.n_cells_x, ginfo.n_cells_y)
    for h in h_fields
        push!(η_fields, h .- depth_2d)
    end

    # Cell centers for plotting
    cell_xs = [(result.xs[i] + result.xs[i+1])/2 for i in 1:ginfo.n_cells_x]
    cell_ys = [(result.ys[j] + result.ys[j+1])/2 for j in 1:ginfo.n_cells_y]

    # Create figure (4 rows: bathymetry+IC, then 6 evolution panels with individual colorbars)
    fig = Figure(size=(1600, 1400))

    # Top row: Bathymetry and initial condition
    ax1 = Axis(fig[1, 1], title="Bathymetry (depth in m)", xlabel="x (km)", ylabel="y (km)", aspect=1)
    hm1 = heatmap!(ax1, cell_xs, cell_ys, depth_2d .* 1000, colormap=:deep)
    Colorbar(fig[1, 2], hm1, label="Depth (m)")

    ax2 = Axis(fig[1, 3], title="Initial surface η (t=0)", xlabel="x (km)", ylabel="y (km)", aspect=1)
    η0_range = maximum(abs.(extrema(η_fields[1]))) * 1000
    hm2 = heatmap!(ax2, cell_xs, cell_ys, η_fields[1] .* 1000, colormap=:RdBu, colorrange=(-η0_range, η0_range))
    Colorbar(fig[1, 4], hm2, label="η (m)")

    # Bottom rows: Evolution (show 6 timesteps in 3 rows of 2)
    # Each panel has its own color normalization
    n_show = min(6, n_timesteps - 1)
    if n_show > 0
        show_times = round.(Int, range(2, n_timesteps, length=n_show))
        for (i, t_idx) in enumerate(show_times)
            row = 2 + (i-1) ÷ 2
            col = ((i-1) % 2) * 2 + 1
            t_min = t_idx - 1

            η_data = η_fields[t_idx] .* 1000
            η_range_i = maximum(abs.(extrema(η_data)))
            η_range_i = max(η_range_i, 0.1)  # Ensure non-zero range

            ax = Axis(fig[row, col], title="η (t=$t_min min)", xlabel="x (km)", ylabel="y (km)", aspect=1)
            hm = heatmap!(ax, cell_xs, cell_ys, η_data, colormap=:RdBu, colorrange=(-η_range_i, η_range_i))
            Colorbar(fig[row, col+1], hm, label="η (m)")
        end
    end

    save("experiments/nonlinear_shallow_water/swe_matern_solution.png", fig)
    println("\nSaved: swe_matern_solution.png")

    # Print statistics
    println("\nSolution statistics:")
    for (i, η) in enumerate(η_fields)
        t_min = i - 1  # t=1 is initial (t_min=0), etc.
        println("  t=$t_min min: η range = $(round(minimum(η)*1000, digits=2)) to $(round(maximum(η)*1000, digits=2)) m")
    end

    return result, fig
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    visualize_swe_solution()
end
