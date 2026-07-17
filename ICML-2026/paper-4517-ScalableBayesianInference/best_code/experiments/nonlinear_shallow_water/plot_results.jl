"""
Paper-quality plotting for SWE results.

Load saved results and create visualizations without rerunning simulation.

Usage:
    julia --project=. experiments/nonlinear_shallow_water/plot_results.jl --input results/swe/swe_results.jld2
    julia --project=. experiments/nonlinear_shallow_water/plot_results.jl --input results/swe/swe_results.jld2 --publication
"""

using GPFiniteVolume  # For Layout, indices
using CairoMakie
using TuePlots
using JLD2
using ArgParse
using Statistics
using Unitful

include("problem.jl")  # For SolveParams type

function parse_commandline()
    s = ArgParseSettings(description="Plot SWE results")

    @add_arg_table! s begin
        "--input"
            help = "Path to swe_results.jld2"
            arg_type = String
            default = "results/swe/swe_results.jld2"
        "--output"
            help = "Output directory for figures"
            arg_type = String
            default = "results/swe"
        "--framerate"
            help = "Animation framerate"
            arg_type = Int
            default = 8
        "--publication"
            help = "Use TuePlots ICML settings for publication-ready output"
            action = :store_true
    end

    return parse_args(s)
end

args = parse_commandline()

# ==============================================================================
# Load Results
# ==============================================================================

println("Loading results from $(args["input"])...")
@load args["input"] means vars xs ys bathy_grid p Δt_solve n_timesteps N_x N_y full_state_layout

output_dir = args["output"]
mkpath(output_dir)

println("Loaded: $(N_x)×$(N_y) grid, $(n_timesteps) timesteps")

# ==============================================================================
# Extract h indices and compute η for all timesteps
# ==============================================================================

# Get h indices from layout
h_idx = indices(full_state_layout, :h)

# Equilibrium depth at grid points (for computing perturbation)
# Use saved bathy_grid which includes Matern perturbation if enabled
H0_grid = bathy_grid

# Compute η (perturbation) for all timesteps
all_eta = Vector{Matrix{Float64}}(undef, n_timesteps)
all_std = Vector{Matrix{Float64}}(undef, n_timesteps)
for t_idx in 1:n_timesteps
    h_vals = means[t_idx][h_idx]
    h_var = vars[t_idx][h_idx]
    h_2d = reshape(h_vals, N_x, N_y)
    std_2d = reshape(sqrt.(h_var), N_x, N_y)
    all_eta[t_idx] = (h_2d .- H0_grid) .* 1000  # Convert to meters
    all_std[t_idx] = std_2d .* 1000  # Convert to meters
end

# Bathymetry in meters
bathy_m = bathy_grid .* 1000

println("η range over simulation: $(minimum(minimum.(all_eta))) to $(maximum(maximum.(all_eta))) m")
println("Bathymetry range: $(minimum(bathy_m)) to $(maximum(bathy_m)) m")

# ==============================================================================
# 3D Animation with Bathymetry
# ==============================================================================

println("\nCreating 3D animation...")

xs_vec = collect(xs)
ys_vec = collect(ys)

# Scale bathymetry to fit in same z-range as η
# Map depth range [H_shallow, H_deep] → [-η_range, -η_range/2] (below the surface)
eta_range = 15.0  # meters, for water surface
bathy_z_top = -eta_range * 0.3   # shallowest bathymetry z
bathy_z_bottom = -eta_range * 1.0  # deepest bathymetry z

bathy_min, bathy_max = extrema(bathy_m)
bathy_scaled = @. bathy_z_top + (bathy_z_bottom - bathy_z_top) * (bathy_m - bathy_min) / (bathy_max - bathy_min)

fig_anim = Figure(size=(900, 700), backgroundcolor=:white)

ax3d = Axis3(fig_anim[1, 1],
    xlabel = "x (km)",
    ylabel = "y (km)",
    zlabel = "η (m)",
    aspect = (1, 1, 0.5),
    elevation = 18π/180,   # lower angle to see more horizontal
    azimuth = -40π/180,    # rotated to see x-axis better (wave propagation direction)
    protrusions = (50, 50, 50, 50)
)

zlims!(ax3d, bathy_z_bottom - 2, eta_range * 1.1)

# Bathymetry surface (scaled, colored by actual depth)
bathy_surf = surface!(ax3d, xs_vec, ys_vec, bathy_scaled,
    color = bathy_m,
    colormap = :deep,  # cmocean bathymetry colormap
    colorrange = (bathy_min, bathy_max),
    shading = NoShading
)

# Water surface (semi-transparent so bathymetry shows through)
surf = surface!(ax3d, xs_vec, ys_vec, all_eta[1],
    colormap = :balance,  # cmocean diverging colormap for sea surface
    colorrange = (-eta_range, eta_range),
    alpha = 0.85
)

# Colorbars
Colorbar(fig_anim[1, 2], surf, label = "Surface η (m)", height = Relative(0.4), valign = :top)
Colorbar(fig_anim[1, 2], bathy_surf, label = "Depth H₀ (m)", height = Relative(0.4), valign = :bottom)

# Time label
time_label = Label(fig_anim[0, :], "Nonlinear Shallow Water — t = 0.0 min", fontsize = 18)

# Record animation
anim_path = joinpath(output_dir, "swe_evolution.mp4")
record(fig_anim, anim_path, 1:n_timesteps; framerate=args["framerate"]) do t_idx
    t = (t_idx - 1) * Δt_solve
    time_label.text[] = "Nonlinear Shallow Water — t = $(round(t, digits=1)) min"
    surf[3] = all_eta[t_idx]
end

println("Saved animation to $(anim_path)")

# ==============================================================================
# Static Snapshots Figure (Publication Quality, 3D with Bathymetry)
# ==============================================================================

println("\nCreating 3D snapshot figure...")

publication_mode = args["publication"]

# Pick 4 representative timesteps
t_snap = [1, n_timesteps ÷ 3 + 1, 2 * n_timesteps ÷ 3 + 1, n_timesteps]
n_snaps = length(t_snap)

# Panel labels
labels = ["(a)", "(b)", "(c)", "(d)"]

# Set up figure for publication mode
if publication_mode
    # Use TuePlots for fonts only, manual size for 3D
    theme = Theme(
        TuePlots.SETTINGS[:ICML];
        font=true,
        fontsize=true,
        figsize=false,  # we control size manually for 3D
    )
    set_theme!(theme)
    # ICML full width ~6.75in ≈ 487pt
    fig_snap = Figure(size=(487, 110))
    label_fontsize = 7
    title_fontsize = 8
else
    fig_snap = Figure(size=(1000, 280), fontsize=10)
    label_fontsize = 11
    title_fontsize = 12
end

# Compute global color range across selected timesteps
eta_max_snap = maximum(maximum.(abs, all_eta[t_snap]))

# Z-axis scaling: place bathymetry below water surface
z_water_range = eta_max_snap * 1.2
z_bathy_top = -z_water_range * 0.2
z_bathy_bottom = -z_water_range * 0.8

# Scale bathymetry to fit below water
bathy_scaled_snap = @. z_bathy_top + (z_bathy_bottom - z_bathy_top) * (bathy_m - bathy_min) / (bathy_max - bathy_min)

# Consistent view angle for all panels
elevation_angle = 20π/180
azimuth_angle = -55π/180

for (col, t_idx) in enumerate(t_snap)
    t = (t_idx - 1) * Δt_solve
    # Format time nicely: use decimal if needed, otherwise integer
    t_str = t == round(Int, t) ? "$(round(Int, t))" : "$(round(t, digits=1))"

    ax = Axis3(fig_snap[1, col],
        xlabel = "",
        ylabel = "",
        zlabel = "",
        title = "t = $(t_str) min",
        titlesize = title_fontsize,
        titlegap = 0,
        aspect = (1, 1, 1.0),
        elevation = elevation_angle,
        azimuth = azimuth_angle,
        xticklabelsize = publication_mode ? 5 : 8,
        yticklabelsize = publication_mode ? 5 : 8,
        zticklabelsize = publication_mode ? 5 : 8,
        xlabelsize = publication_mode ? 6 : 9,
        ylabelsize = publication_mode ? 6 : 9,
        protrusions = (0, 0, 0, 10),
    )

    zlims!(ax, z_bathy_bottom - 2, z_water_range)

    # Hide labels and ticks but keep grid
    hidexdecorations!(ax, grid=false)
    hideydecorations!(ax, grid=false)
    hidezdecorations!(ax, grid=false)

    # Bathymetry surface (below water)
    surface!(ax, xs_vec, ys_vec, bathy_scaled_snap,
        color = bathy_m,
        colormap = :deep,
        colorrange = (bathy_min, bathy_max),
        shading = NoShading
    )

    # Water surface
    surface!(ax, xs_vec, ys_vec, all_eta[t_idx],
        colormap = :balance,
        colorrange = (-20, 20),
        alpha = 0.9
    )

    # Panel label as title annotation
    text!(ax, 0, 0, z_water_range * 0.95,
        text = labels[col],
        fontsize = label_fontsize,
        font = :bold,
        color = :black,
        align = (:left, :top)
    )
end

# Single colorbar for η
cbar_width = publication_mode ? 8 : 12
Colorbar(fig_snap[1, n_snaps + 1],
    colormap = :balance,
    colorrange = (-20, 20),
    label = "η (m)",
    width = cbar_width,
    height = Relative(0.7),
    labelsize = publication_mode ? 7 : 10,
    ticklabelsize = publication_mode ? 6 : 9
)

# Adjust colorbar column width
colsize!(fig_snap.layout, n_snaps + 1, Fixed(publication_mode ? 25 : 40))

# Gap between panels
colgap!(fig_snap.layout, 15)

# Save
if publication_mode
    snap_path = joinpath(output_dir, "shallow_water_evolution.pdf")
    save(snap_path, fig_snap, pt_per_unit=1)
    snap_png = joinpath(output_dir, "shallow_water_evolution.png")
    save(snap_png, fig_snap, px_per_unit=3)
    println("Saved: $(snap_path)")
    println("Saved: $(snap_png)")
    set_theme!()  # Reset to default
else
    snap_path = joinpath(output_dir, "swe_snapshots.png")
    save(snap_path, fig_snap, px_per_unit=2)
    println("Saved snapshots to $(snap_path)")
end

# ==============================================================================
# Bathymetry-only figure
# ==============================================================================

println("\nCreating bathymetry figure...")

fig_bathy = Figure(size=(600, 500))

ax_bathy = Axis3(fig_bathy[1, 1],
    xlabel = "x (km)",
    ylabel = "y (km)",
    zlabel = "Depth (m)",
    title = "Bathymetry H₀(x, y)",
    aspect = (1, 1, 0.3),
    elevation = 25π/180,
    azimuth = -50π/180
)

# Show actual bathymetry (not scaled)
surf_bathy = surface!(ax_bathy, xs_vec, ys_vec, -bathy_m,  # negative = below sea level
    colormap = Reverse(:deep),  # cmocean bathymetry colormap
    colorrange = (-bathy_max, -bathy_min)
)
Colorbar(fig_bathy[1, 2], surf_bathy, label = "Seabed elevation (m)")

bathy_path = joinpath(output_dir, "bathymetry.png")
save(bathy_path, fig_bathy, px_per_unit=2)
println("Saved bathymetry to $(bathy_path)")

println("\nDone!")
