"""
2D visualization replicating Figure 2 from Chen, Owhadi, Schäfer (2024).

Compares screening effects between different measurement orderings:
- Left: Many Diracs + one integral test → shows if integral is screened by Diracs
- Right: Many integrals + one Dirac test → shows if Dirac is screened by integrals

Each case uses a simple 2x2 block matrix [bulk, test; test, test].
"""

using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra
using CairoMakie
using GPFiniteVolume

"""
    create_2d_screening_figure(; N_xy=20, kernel_lengthscale=0.1, test_point=[0.5, 0.5])

Create 2D screening visualization comparing Dirac vs integral orderings.
"""
function create_2d_screening_figure(;
        N_xy = 41,  # 41 points so 0.5 is exactly on grid (0.5 = 20/40)
        kernel_lengthscale = 0.1,
        kernel_smoothness = 1,
        test_point = [0.5, 0.5],
        output_dir = "figures"
    )

    println("Setting up 2D screening comparison (Figure 2 style)...")
    println("Grid size: $(N_xy)x$(N_xy) = $(N_xy^2) points")

    # Build 2D kernel
    k_base = HalfIntegerMaternKernel(kernel_smoothness, kernel_lengthscale)
    k_prod = k_base ⊗ k_base

    # Create 2D grid
    N_total = N_xy^2
    Xs_base = range(0.0, 1.0, length=N_xy)
    Xs = FactorizedGrid(Xs_base, Xs_base)

    # Flatten to coordinate matrix (2 x N_total)
    Xs_flat = reshape(collect(Iterators.product(Xs_base, Xs_base)), 1, N_total)
    X = zeros(2, N_total)
    for i in 1:N_total
        X[:, i] .= Xs_flat[i]
    end

    # Find grid point closest to test_point
    dists = [norm(X[:, i] - test_point) for i in 1:N_total]
    test_idx = argmin(dists)
    test_loc = X[:, test_idx]
    println("Test point: $(test_loc)")

    # -------------------------------------------------------------------------
    # Case A: Many Diracs + One Integral test
    # -------------------------------------------------------------------------
    println("\n=== Case A: Many Diracs + One Integral test ===")

    # Bulk: all Dirac evaluations
    L_eval_bulk = EvaluationFunctional(Xs)

    # Test: one integral at test location
    # Find the cell containing test point
    base_intervals = intervals_from_endpoints(collect(Xs_base))
    test_cell_x_idx = findfirst(i -> test_loc[1] in base_intervals[i], 1:length(base_intervals))
    test_cell_y_idx = findfirst(i -> test_loc[2] in base_intervals[i], 1:length(base_intervals))

    # Create single integral domain as a FactorizedDomain
    test_domain_x = [base_intervals[test_cell_x_idx]]
    test_domain_y = [base_intervals[test_cell_y_idx]]
    test_domains = test_domain_x ⊗ test_domain_y
    L_integ_test = VectorizedLebesgueIntegral(test_domains)

    # Stack: [Integral; Diracs] - test first (finest)
    L_stack_A = StackedLinearFunctional(L_integ_test, L_eval_bulk)
    K_A = Matrix(L_stack_A(L_stack_A(k_prod)))

    println("K_A size: $(size(K_A))")

    # Compute precision Cholesky
    K_A_inv = inv(Symmetric(K_A + 1e-10*I))
    L_A = cholesky(Symmetric(K_A_inv)).L

    # First column corresponds to the integral test measurement (conditioned on Diracs)
    col_A = L_A[:, 1]

    # Spatial coordinates for bulk Diracs
    coords_A = X  # All N_total grid points

    # -------------------------------------------------------------------------
    # Case B: Many Integrals + One Dirac test
    # -------------------------------------------------------------------------
    println("\n=== Case B: Many Integrals + One Dirac test ===")

    # Bulk: all integrals over 2D cells
    domains_bulk = base_intervals ⊗ base_intervals
    L_integ_bulk = VectorizedLebesgueIntegral(domains_bulk)

    # Test: one Dirac at test location
    test_grid = FactorizedGrid([test_loc[1]], [test_loc[2]])
    L_eval_test = EvaluationFunctional(test_grid)

    # Stack: [Dirac; Integrals] - test first (finest)
    L_stack_B = StackedLinearFunctional(L_eval_test, L_integ_bulk)
    K_B = Matrix(L_stack_B(L_stack_B(k_prod)))

    println("K_B size: $(size(K_B))")

    # Compute precision Cholesky
    K_B_inv = inv(Symmetric(K_B + 1e-10*I))
    L_B = cholesky(Symmetric(K_B_inv)).L

    # First column corresponds to the Dirac test measurement (conditioned on Integrals)
    col_B = L_B[:, 1]

    # Spatial coordinates for bulk integrals (use cell centers)
    # There are (N_xy-1) x (N_xy-1) cells
    n_cells = length(base_intervals)^2
    coords_B = zeros(2, n_cells)
    idx = 1
    for iy in 1:length(base_intervals)
        for ix in 1:length(base_intervals)
            coords_B[1, idx] = midpoint(base_intervals[ix])
            coords_B[2, idx] = midpoint(base_intervals[iy])
            idx += 1
        end
    end

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    println("\nCreating figure...")
    fig = Figure(size=(1400, 600), fontsize=14)

    # Exclude the first entry (it's the test itself) and normalize by max
    col_A_bulk = col_A[2:end]
    col_B_bulk = col_B[2:end]

    # Normalize each column by its maximum absolute value
    col_A_normalized = col_A_bulk ./ maximum(abs.(col_A_bulk))
    col_B_normalized = col_B_bulk ./ maximum(abs.(col_B_bulk))

    # Compute log values
    log_vals_A = log10.(abs.(col_A_normalized) .+ 1e-16)
    log_vals_B = log10.(abs.(col_B_normalized) .+ 1e-16)

    println("Normalized range A: [$(round(minimum(log_vals_A), digits=2)), 0.0]")
    println("Normalized range B: [$(round(minimum(log_vals_B), digits=2)), 0.0]")

    # Use shared colorbar range
    vmin = min(minimum(log_vals_A), minimum(log_vals_B))
    vmin = -12.0
    vmax = 0.0  # Max is always 0 after normalization

    # Left panel: Many Diracs + One Integral test
    ax1 = Axis(fig[1, 1],
        xlabel = "x",
        ylabel = "y",
        title = "Function evaluations first",
        aspect = DataAspect()
    )

    sc1 = scatter!(ax1, coords_A[1, :], coords_A[2, :],
        color = log_vals_A,
        colormap = :turbo,
        colorrange = (vmin, vmax),
        markersize = 10
    )

    # Mark test location
    scatter!(ax1, [test_loc[1]], [test_loc[2]],
        color = :tomato, marker = :rect, markersize = 30,
        strokecolor = :tomato, strokewidth = 2, label = "Test integral", alpha=0.9)

    # Right panel: Many Integrals + One Dirac test
    ax2 = Axis(fig[1, 2],
        xlabel = "x",
        ylabel = "y",
        title = "Integrals first",
        aspect = DataAspect()
    )

    sc2 = scatter!(ax2, coords_B[1, :], coords_B[2, :],
        color = log_vals_B,
        colormap = :turbo,
        colorrange = (vmin, vmax),
        markersize = 10, marker=:rect
    )

    # Mark test location
    scatter!(ax2, [test_loc[1]], [test_loc[2]],
        color = :tomato, markersize = 30,
        strokecolor = :tomato, strokewidth = 2, label = "Test Dirac", alpha=0.9)

    # Shared colorbar
    Colorbar(fig[1, 3], sc1, label = "log₁₀(|Lᵢⱼ|/max|Lᵢⱼ|)")

    # Save
    mkpath(output_dir)
    save(joinpath(output_dir, "screening_comparison_2d.pdf"), fig)
    save(joinpath(output_dir, "screening_comparison_2d.png"), fig, px_per_unit=3)
    println("\nSaved: screening_comparison_2d.pdf/png")

    return fig
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    create_2d_screening_figure()
end
