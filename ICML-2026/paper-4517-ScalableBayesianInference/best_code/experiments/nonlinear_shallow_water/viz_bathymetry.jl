using CairoMakie
using FunctionalGPs
using LinearAlgebra
using Random

Random.seed!(42)

println("=== Visualizing Matérn Bathymetry Options ===\n")

# Domain: 100km × 100km (like our SWE setup)
L = 100.0  # km
N = 50  # grid points for visualization

xs = range(0, L, length=N)
ys = range(0, L, length=N)
points = vec([(x, y) for y in ys, x in xs])

# Matérn kernel - try different lengthscales
function sample_bathymetry(lengthscale, variance, points)
    k = HalfIntegerMaternKernel(2, [lengthscale, lengthscale])  # Matérn 5/2, isotropic
    K = variance * kernelmatrix(k, points)
    K += 1e-6 * I  # nugget for numerical stability
    L_chol = cholesky(Symmetric(K)).L
    z = randn(length(points))
    return L_chol * z
end

# Sample with different lengthscales
println("Sampling bathymetry with different lengthscales...")

fig = Figure(size=(1200, 800))

# Base depth profile (linear slope: shallow at x=0, deep at x=L)
H_shallow = 50.0  # m at x=0
H_deep = 500.0    # m at x=L
base_depth = [H_shallow + (H_deep - H_shallow) * p[1] / L for p in points]

lengthscales = [10.0, 20.0, 30.0]  # km
variance = 30.0^2  # variance in meters^2 (std = 30m)

for (i, ls) in enumerate(lengthscales)
    perturbation = sample_bathymetry(ls, variance, points)
    depth = base_depth .+ perturbation
    depth = max.(depth, 10.0)  # minimum 10m depth (no dry land)

    Z = reshape(depth, N, N)

    ax = Axis(fig[1, i],
        title="Lengthscale = $(ls) km",
        xlabel="x (km)", ylabel="y (km)",
        aspect=1)
    hm = heatmap!(ax, xs, ys, Z, colormap=:deep)
    if i == 3
        Colorbar(fig[1, 4], hm, label="Depth (m)")
    end
end

# Also show gradient magnitude for one case
ls = 20.0
perturbation = sample_bathymetry(ls, variance, points)
depth = base_depth .+ perturbation
depth = max.(depth, 10.0)
Z = reshape(depth, N, N)

# Compute gradients via finite differences
dx = xs[2] - xs[1]
dy = ys[2] - ys[1]
dZ_dx = zeros(N, N)
dZ_dy = zeros(N, N)
for i in 2:N-1, j in 1:N
    dZ_dx[i,j] = (Z[i+1,j] - Z[i-1,j]) / (2dx)
end
for i in 1:N, j in 2:N-1
    dZ_dy[i,j] = (Z[i,j+1] - Z[i,j-1]) / (2dy)
end
grad_mag = sqrt.(dZ_dx.^2 .+ dZ_dy.^2)

ax4 = Axis(fig[2, 1:2],
    title="Bathymetry (ls=20km)",
    xlabel="x (km)", ylabel="y (km)",
    aspect=1)
hm4 = heatmap!(ax4, xs, ys, Z, colormap=:deep)
Colorbar(fig[2, 3], hm4, label="Depth (m)")

ax5 = Axis(fig[2, 3:4],
    title="Gradient magnitude |∇b|",
    xlabel="x (km)", ylabel="y (km)",
    aspect=1)
hm5 = heatmap!(ax5, xs, ys, grad_mag, colormap=:viridis)
Colorbar(fig[2, 5], hm5, label="|∇b| (m/km)")

save("experiments/nonlinear_shallow_water/bathymetry_options.png", fig)
println("Saved bathymetry_options.png")

# Print some stats
println("\nBathymetry stats (ls=20km):")
println("  Depth range: $(round(minimum(Z), digits=1)) - $(round(maximum(Z), digits=1)) m")
println("  Gradient range: $(round(minimum(grad_mag), digits=3)) - $(round(maximum(grad_mag), digits=3)) m/km")
