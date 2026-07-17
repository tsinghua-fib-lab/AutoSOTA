"""
Standalone script to run the SWE showcase with Matérn bathymetry.
Run with: julia --project=. experiments/nonlinear_shallow_water/run_showcase.jl
"""

using CairoMakie
using GPFiniteVolume: TimeStack
using Unitful

include("main.jl")

function run_showcase()
    println("=" ^ 70)
    println("GP-FVM Nonlinear Shallow Water Equations Showcase")
    println("=" ^ 70)

    # Problem setup: offshore tsunami with sloping + Matérn bathymetry
    prob = NonlinearSWEProblem(
        L_x = 100.0u"km",
        L_y = 100.0u"km",
        H₀_deep = 500.0u"m",      # Deep water offshore
        H₀_shallow = 50.0u"m",    # Shallow water at shore
        h_center_x = 75.0u"km",   # Source offshore
        h_center_y = 50.0u"km",   # Centered in y
        h_amplitude = 10.0u"m",   # 10m tsunami
        h_width_x = 5.0u"km",     # Narrow perpendicular to shore
        h_width_y = 25.0u"km"     # Wide along shore (fault-like)
    )

    # Run solver
    result = solve_nonlinear_swe(;
        N_x = 15,                      # Higher resolution
        N_y = 15,
        n_timesteps = 12,              # 12 minutes
        bathymetry_lengthscale = 15.0, # Matérn perturbations
        bathymetry_std = 30.0,         # 30m std
        bathymetry_seed = 42,
        prob = prob,
        verbose = true
    )

    # Extract solution
    sol = result.solution
    μ = mean(sol)
    ginfo = result.ginfo
    p = result.params
    n_timesteps = 12

    # Get timestep accessor
    means = TimeStack(μ, result.full_state_layout)
    cell_area = ginfo.Δx * ginfo.Δy

    # Extract η fields
    η_fields = []
    depth_2d = reshape(p.depth_at_cells, ginfo.n_cells_x, ginfo.n_cells_y)
    for t in 1:n_timesteps
        h_int = means[:h_int, t]
        h_avg = h_int ./ cell_area
        h_2d = reshape(h_avg, ginfo.n_cells_x, ginfo.n_cells_y)
        push!(η_fields, h_2d .- depth_2d)
    end

    # Cell centers for plotting
    cell_xs = [(result.xs[i] + result.xs[i+1])/2 for i in 1:ginfo.n_cells_x]
    cell_ys = [(result.ys[j] + result.ys[j+1])/2 for j in 1:ginfo.n_cells_y]

    # Create figure
    fig = Figure(size=(1800, 1400))

    # Top row: Bathymetry and initial condition
    ax1 = Axis(fig[1, 1:2], title="Bathymetry (depth in m)", xlabel="x (km)", ylabel="y (km)", aspect=1)
    hm1 = heatmap!(ax1, cell_xs, cell_ys, depth_2d .* 1000, colormap=:deep)
    Colorbar(fig[1, 3], hm1, label="Depth (m)")

    ax2 = Axis(fig[1, 4:5], title="Initial surface η (t=0)", xlabel="x (km)", ylabel="y (km)", aspect=1)
    η0_range = maximum(abs.(extrema(η_fields[1]))) * 1000
    hm2 = heatmap!(ax2, cell_xs, cell_ys, η_fields[1] .* 1000, colormap=:RdBu, colorrange=(-η0_range, η0_range))
    Colorbar(fig[1, 6], hm2, label="η (m)")

    # Evolution panels: 2 per row, 5 rows = 10 panels
    show_times = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for (i, t_idx) in enumerate(show_times)
        row = 2 + (i-1) ÷ 2
        col_start = ((i-1) % 2) * 3 + 1
        t_min = t_idx - 1

        η_data = η_fields[t_idx] .* 1000
        η_range_i = max(maximum(abs.(extrema(η_data))), 0.1)

        ax = Axis(fig[row, col_start:col_start+1], title="η (t=$t_min min)", xlabel="x (km)", ylabel="y (km)", aspect=1)
        hm = heatmap!(ax, cell_xs, cell_ys, η_data, colormap=:RdBu, colorrange=(-η_range_i, η_range_i))
        Colorbar(fig[row, col_start+2], hm, label="η (m)")
    end

    # Save
    outfile = "experiments/nonlinear_shallow_water/swe_showcase.png"
    save(outfile, fig)
    println("\nSaved: $outfile")

    # Print statistics
    println("\nSolution statistics:")
    for (i, η) in enumerate(η_fields)
        t_min = i - 1
        println("  t=$t_min min: η range = $(round(minimum(η)*1000, digits=2)) to $(round(maximum(η)*1000, digits=2)) m")
    end

    return result, fig
end

# Run
run_showcase()
