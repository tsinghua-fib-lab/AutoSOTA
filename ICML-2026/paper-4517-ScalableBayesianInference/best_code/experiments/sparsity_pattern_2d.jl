"""
Visualize Cholesky factor sparsity patterns for 2D GP-FVM.

Creates a publication-quality figure showing the precision Cholesky factor
for different orderings, demonstrating how entry magnitudes decay and
why sparse approximation is effective.

Usage:
    julia --project=. experiments/sparsity_pattern_2d.jl
    julia --project=. experiments/sparsity_pattern_2d.jl --nxy 12
"""

using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using CairoMakie
using TuePlots
using GPFiniteVolume
using ArgParse

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(description = "Visualize 2D Cholesky sparsity patterns")

    @add_arg_table! s begin
        "--nxy"
            help = "Grid size (N × N)"
            arg_type = Int
            default = 10
        "--lengthscale"
            help = "Kernel lengthscale"
            arg_type = Float64
            default = 0.1
        "--output", "-o"
            help = "Output path (default: figures/cholesky_sparsity_2d.pdf)"
            arg_type = String
            default = ""
    end

    return parse_args(s)
end

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

"""
    setup_2d_problem(; N_xy, kernel_lengthscale)

Create 2D measurement setup with evaluations, derivatives, and integrals.
Returns covariance matrix K and metadata.
"""
function setup_2d_problem(; N_xy=10, kernel_lengthscale=0.1)
    # Build 2D product kernel
    k_base = HalfIntegerMaternKernel(1, [kernel_lengthscale])  # Matérn-3/2
    k_prod = k_base ⊗ k_base

    # Create 2D grid
    Xs_base = range(0.0, 1.0, length=N_xy)
    Xs = FactorizedGrid(Xs_base, Xs_base)

    # Evaluation functional
    L_eval = EvaluationFunctional(Xs)
    n_eval = prod(output_shape(L_eval))

    # Derivative functional: ∂/∂x + ∂/∂y
    d_sum = PartialDerivative((1, 0)) + PartialDerivative((0, 1))
    L_deriv = L_eval ∘ d_sum
    n_deriv = prod(output_shape(L_deriv))

    # Integral functional over 2D cells
    base_intervals = intervals_from_endpoints(collect(Xs_base))
    domains_2d = base_intervals ⊗ base_intervals
    L_integ = VectorizedLebesgueIntegral(domains_2d)
    n_int = prod(output_shape(L_integ))

    # Stack functionals
    L_stack = StackedLinearFunctional(L_eval, L_deriv, L_integ)

    # State layout
    state_layout = layout((f = n_eval, f_dx = n_deriv, f_int = n_int))

    # Compute full covariance matrix
    K = Matrix(L_stack(L_stack(k_prod)))

    # Build coordinate matrices
    N_total = N_xy^2
    X_eval = zeros(2, N_total)
    idx = 1
    for iy in 1:N_xy, ix in 1:N_xy
        X_eval[1, idx] = Xs_base[ix]
        X_eval[2, idx] = Xs_base[iy]
        idx += 1
    end
    X_deriv = copy(X_eval)

    n_cells_1d = length(base_intervals)
    X_int = zeros(2, n_cells_1d^2)
    idx = 1
    for iy in 1:n_cells_1d, ix in 1:n_cells_1d
        X_int[1, idx] = midpoint(base_intervals[ix])
        X_int[2, idx] = midpoint(base_intervals[iy])
        idx += 1
    end

    return (
        K = K,
        state_layout = state_layout,
        X_eval = X_eval,
        X_deriv = X_deriv,
        X_int = X_int,
        n_eval = n_eval,
        n_deriv = n_deriv,
        n_int = n_int,
    )
end

"""
    create_ordering(setup, order_type::Symbol)

Create global ordering for :eval_first or :int_first.
Returns permutation P and block boundaries.
"""
function create_ordering(setup, order_type::Symbol)
    (; state_layout, X_eval, X_int, n_eval, n_deriv, n_int) = setup

    # Compute maximin orderings
    P_eval, _ = reverse_maximin_ordering(X_eval)
    P_int, _ = reverse_maximin_ordering(X_int)

    # Map to global indices
    P_eval_global = indices(state_layout, :f)[P_eval]
    P_int_global = indices(state_layout, :f_int)[P_int]
    P_deriv_global = indices(state_layout, :f_dx)

    if order_type == :eval_first
        # Order: [derivatives; integrals; evaluations]
        P_global = [P_deriv_global; P_int_global; P_eval_global]
        block_sizes = (n_deriv, n_int, n_eval)
        block_labels = ("Derivatives", "Integrals", "Evaluations")
    elseif order_type == :int_first
        # Order: [derivatives; evaluations; integrals]
        P_global = [P_deriv_global; P_eval_global; P_int_global]
        block_sizes = (n_deriv, n_eval, n_int)
        block_labels = ("Derivatives", "Evaluations", "Integrals")
    else
        error("Unknown order_type: $order_type")
    end

    return P_global, block_sizes, block_labels
end

"""
    compute_precision_cholesky(K, P)

Compute the exact Cholesky factor L of the precision matrix K⁻¹[P,P] = LL'.
"""
function compute_precision_cholesky(K::AbstractMatrix, P::Vector{Int})
    K_perm = K[P, P]
    K_perm_reg = K_perm + 1e-10 * I
    K_inv = inv(Symmetric(K_perm_reg))
    C = cholesky(Symmetric(K_inv))
    return Matrix(C.L)
end

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

"""
    create_sparsity_figure(setup; output_path)

Create publication-quality figure comparing sparsity patterns.
"""
function create_sparsity_figure(setup; output_path="figures/cholesky_sparsity_2d.pdf")
    (; K, n_eval, n_deriv, n_int) = setup
    n_total = n_eval + n_deriv + n_int

    println("Matrix size: $n_total × $n_total")
    println("  Evaluations: $n_eval")
    println("  Derivatives: $n_deriv")
    println("  Integrals: $n_int")

    # Compute Cholesky factors for both orderings
    println("\nComputing Cholesky factors...")

    P_int, blocks_int, labels_int = create_ordering(setup, :int_first)
    L_int = compute_precision_cholesky(K, P_int)
    println("  Integrals-first: done")

    P_eval, blocks_eval, labels_eval = create_ordering(setup, :eval_first)
    L_eval = compute_precision_cholesky(K, P_eval)
    println("  Evaluations-first: done")

    # Set up TuePlots theme (appendix full-width, 2 panels)
    theme = Theme(
        TuePlots.SETTINGS[:ICML];
        font=true, fontsize=true, figsize=true,
        single_column=false,
        nrows=1, ncols=2,
        subplot_height_to_width_ratio=1.0,
    )
    set_theme!(theme)

    fig = Figure()

    # Colorbar range
    crange = (-12, 0)

    # Panel (a): Integrals-first
    ax1 = Axis(fig[1, 1],
        xlabel="Column index",
        ylabel="Row index",
        title="Integrals coarsest",
        yreversed=true,
        aspect=DataAspect(),
    )

    L_log_int = log10.(abs.(L_int) .+ 1e-16)
    hm1 = heatmap!(ax1, 1:n_total, 1:n_total, L_log_int',
        colormap=:viridis, colorrange=crange)

    # Add block boundaries
    add_block_boundaries!(ax1, blocks_int, labels_int, n_total)

    # Panel (b): Evaluations-first
    ax2 = Axis(fig[1, 2],
        xlabel="Column index",
        ylabel="Row index",
        title="Evaluations coarsest",
        yreversed=true,
        aspect=DataAspect(),
    )

    L_log_eval = log10.(abs.(L_eval) .+ 1e-16)
    hm2 = heatmap!(ax2, 1:n_total, 1:n_total, L_log_eval',
        colormap=:viridis, colorrange=crange)

    add_block_boundaries!(ax2, blocks_eval, labels_eval, n_total)

    # Panel labels (top right)
    text!(ax1, 0.98, 0.98, text="(a)", align=(:right, :top),
          space=:relative, fontsize=9, font=:bold, color=:white)
    text!(ax2, 0.98, 0.98, text="(b)", align=(:right, :top),
          space=:relative, fontsize=9, font=:bold, color=:white)

    # Shared colorbar
    Colorbar(fig[1, 3], hm1, label=L"\log_{10} |L_{ij}|", width=10)

    # Adjust layout
    colgap!(fig.layout, 1, 8)
    colgap!(fig.layout, 2, 5)

    # Save
    mkpath(dirname(output_path))
    save(output_path, fig, pt_per_unit=1)
    println("\nSaved: $output_path")

    # Also save PNG for quick viewing
    png_path = replace(output_path, ".pdf" => ".png")
    save(png_path, fig, px_per_unit=3)
    println("Saved: $png_path")

    # Reset theme
    set_theme!()

    # Print sparsity statistics
    println("\nSparsity statistics (entries with |L_ij| > threshold):")
    for (name, L) in [("Integrals-first", L_int), ("Evaluations-first", L_eval)]
        max_nnz = n_total * (n_total + 1) ÷ 2
        nnz_1e3 = count(x -> abs(x) > 1e-3, L)
        nnz_1e6 = count(x -> abs(x) > 1e-6, L)
        nnz_1e9 = count(x -> abs(x) > 1e-9, L)
        println("  $name:")
        println("    |L| > 1e-3: $nnz_1e3 ($(round(100*nnz_1e3/max_nnz, digits=1))%)")
        println("    |L| > 1e-6: $nnz_1e6 ($(round(100*nnz_1e6/max_nnz, digits=1))%)")
        println("    |L| > 1e-9: $nnz_1e9 ($(round(100*nnz_1e9/max_nnz, digits=1))%)")
    end

    return fig
end

"""
    add_block_boundaries!(ax, block_sizes, block_labels, n_total)

Add dashed lines and labels for block boundaries.
"""
function add_block_boundaries!(ax, block_sizes, block_labels, n_total)
    cumsum_blocks = cumsum(collect(block_sizes))

    for (i, boundary) in enumerate(cumsum_blocks[1:end-1])
        b = boundary + 0.5
        # Horizontal and vertical lines
        hlines!(ax, [b], color=:white, linewidth=0.8, linestyle=:dash)
        vlines!(ax, [b], color=:white, linewidth=0.8, linestyle=:dash)
    end

    # Add block labels in top-right corner of each block
    block_starts = [0; cumsum_blocks[1:end-1]]
    block_ends = cumsum_blocks
    for (i, (start, finish, label)) in enumerate(zip(block_starts, block_ends, block_labels))
        # Top-right corner: x near end, y near start (since y is reversed)
        x_pos = finish - 2
        y_pos = start + 3
        text!(ax, x_pos, y_pos, text=label, align=(:right, :top),
              fontsize=6, color=:white)
    end
end

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

function main()
    args = parse_commandline()

    N_xy = args["nxy"]
    lengthscale = args["lengthscale"]

    if isempty(args["output"])
        output_path = joinpath(@__DIR__, "..", "figures", "cholesky_sparsity_2d.pdf")
    else
        output_path = args["output"]
    end

    println("Setting up 2D problem ($(N_xy) × $(N_xy) grid)...")
    setup = setup_2d_problem(; N_xy=N_xy, kernel_lengthscale=lengthscale)

    create_sparsity_figure(setup; output_path=output_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
