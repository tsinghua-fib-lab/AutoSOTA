"""
Visualize solution evolution for a single problem instance.
Creates an animation showing how different methods evolve over time.
"""

using CairoMakie
using Printf

include("run.jl")

# Pick ONE problem instance
ν = 0.001  # Low viscosity (problematic case)
problem = BurgersProblem(x_min=0.0, x_max=1.0, T_end=0.3, ν=ν, u_left=0.0, u_right=0.0)
instances = generate_problem_instances(problem, [:sine], 1; base_seed=42, verbose=false)
instance = instances[1]

N = 100
n_timesteps = 50

println("Solving Burgers equation: ν=$ν, N=$N, n_t=$n_timesteps")
println("="^60)

# Solve with different methods
println("Classical FVM...")
result_classical = solve_classical_fvm(instance, N; n_timesteps=n_timesteps)

println("GP-FVM (s=2)...")
result_gpfvm = solve_sparse_fvm(instance, N; n_timesteps=n_timesteps, ρ=3.0, smoothness=2)

println("Reference solution...")
# Get reference at same time points
ref_xs = range(0.0, 1.0, length=200)
ref_ts = result_classical.ts

# Create animation
println("\nCreating animation...")

fig = Figure(size=(800, 500))
ax = Axis(fig[1, 1],
    xlabel = "x",
    ylabel = "u(x,t)",
    title = "Burgers equation: ν=$ν"
)

# Set axis limits based on data
all_vals = vcat(result_classical.mean[:], result_gpfvm.mean[:])
ymin, ymax = minimum(all_vals) - 0.1, maximum(all_vals) + 0.1
ylims!(ax, ymin, ymax)
xlims!(ax, 0, 1)

# Animation
n_frames = length(result_classical.ts)

record(fig, "solution_evolution.mp4", 1:n_frames; framerate=5) do frame_idx
    empty!(ax)

    t = result_classical.ts[frame_idx]

    # Reference solution
    ref_u = evaluate_reference(instance.reference, collect(ref_xs), t)
    lines!(ax, collect(ref_xs), ref_u, color=:black, linewidth=2, label="Reference")

    # Classical FVM
    lines!(ax, result_classical.xs, result_classical.mean[:, frame_idx],
           color=:orange, linewidth=2, label="Classical FVM")

    # GP-FVM
    u_gp = result_gpfvm.mean[:, frame_idx]
    σ_gp = result_gpfvm.std[:, frame_idx]

    lines!(ax, result_gpfvm.xs, u_gp, color=:blue, linewidth=2, label="GP-FVM")
    band!(ax, result_gpfvm.xs, u_gp .- 2*σ_gp, u_gp .+ 2*σ_gp,
          color=(:blue, 0.2))

    ax.title = @sprintf("Burgers equation: ν=%.3f, t=%.3f", ν, t)

    axislegend(ax, position=:rt)

    ylims!(ax, ymin, ymax)
    xlims!(ax, 0, 1)
end

println("Saved: solution_evolution.mp4")

# Also save a few snapshots
println("\nSaving snapshots...")
snapshot_times = [1, n_frames÷4, n_frames÷2, 3*n_frames÷4, n_frames]

fig2 = Figure(size=(1000, 600))

for (i, frame_idx) in enumerate(snapshot_times)
    row = (i-1) ÷ 3 + 1
    col = (i-1) % 3 + 1

    ax = Axis(fig2[row, col],
        xlabel = "x",
        ylabel = "u",
        title = @sprintf("t = %.3f", result_classical.ts[frame_idx])
    )

    t = result_classical.ts[frame_idx]

    # Reference
    ref_u = evaluate_reference(instance.reference, collect(ref_xs), t)
    lines!(ax, collect(ref_xs), ref_u, color=:black, linewidth=2, label="Reference")

    # Classical FVM
    lines!(ax, result_classical.xs, result_classical.mean[:, frame_idx],
           color=:orange, linewidth=2, label="Classical")

    # GP-FVM
    u_gp = result_gpfvm.mean[:, frame_idx]
    σ_gp = result_gpfvm.std[:, frame_idx]
    lines!(ax, result_gpfvm.xs, u_gp, color=:blue, linewidth=2, label="GP-FVM")
    band!(ax, result_gpfvm.xs, u_gp .- 2*σ_gp, u_gp .+ 2*σ_gp, color=(:blue, 0.2))

    if i == 1
        axislegend(ax, position=:rt)
    end
end

save("solution_snapshots.pdf", fig2)
println("Saved: solution_snapshots.pdf")

# Print some diagnostics
println("\n" * "="^60)
println("Diagnostics")
println("="^60)
println("Classical FVM final L2 error: $(round(compute_metrics(result_classical, :sine, 42, instance.reference).mean_l2_error * 100, digits=1))%")
println("GP-FVM final L2 error: $(round(compute_metrics(result_gpfvm, :sine, 42, instance.reference).mean_l2_error * 100, digits=1))%")
println("\nGP-FVM posterior std at final time:")
println("  min: $(minimum(result_gpfvm.std[:, end]))")
println("  max: $(maximum(result_gpfvm.std[:, end]))")
println("  mean: $(mean(result_gpfvm.std[:, end]))")
