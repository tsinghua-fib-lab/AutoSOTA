"""
Sparsity/Accuracy Tradeoff Experiment for 2D GP-FVM

Quantitative comparison showing that "integrals first" ordering
achieves better sparse Cholesky approximation quality than "evaluations first"
at any given sparsity level (Pareto dominance).

Based on Chen, Owhadi, Schäfer (2024) methodology.
"""

using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using CairoMakie
using TuePlots
using GPFiniteVolume

"""
    kl_divergence(C_cov, L_prec, K)

Compute KL(N(0, Σ) || N(0, Θ⁻¹)) where:
- C_cov is the Cholesky factorization of Σ (exact covariance)
- L_prec is the sparse lower triangular Cholesky factor where Θ = L_prec * L_prec'
- K is the covariance matrix Σ

KL = 0.5 * (tr(ΘΣ) - n - log det(Θ) - log det(Σ))
"""
function kl_divergence(C_cov::Cholesky, L_prec::SparseMatrixCSC, K::AbstractMatrix)
    n = size(K, 1)

    # Compute sparse precision matrix Q = L * L'
    Q = L_prec * L_prec'
    trace_term = sum(Q .* K)

    # Log determinants
    log_det_cov = logdet(C_cov)
    # For Cholesky L where Θ = L L', logdet(Θ) = 2 * sum(log(diag(L)))
    log_det_prec = 2 * sum(log, diag(L_prec))

    # KL = 0.5 * (tr(ΘΣ) - n - log det(Θ) - log det(Σ))
    kl = 0.5 * (trace_term - n - log_det_prec - log_det_cov)
    return kl
end

# ------------------------------------------------------------------------------
# 2D Measurement Setup
# ------------------------------------------------------------------------------

"""
    setup_2d_measurements(; N_xy, kernel_lengthscale, kernel_smoothness)

Create 2D measurement setup with evaluations, derivatives (∂x + ∂y), and integrals.

Returns named tuple with:
- K: full covariance matrix
- L_eval, L_deriv, L_integ: linear functionals
- state_layout: named index layout
- X_eval, X_deriv, X_int: coordinate matrices (2 × n)
"""
function setup_2d_measurements(;
        N_xy = 35,
        kernel_lengthscale = 0.1,
        kernel_smoothness = 1
    )

    println("Setting up 2D measurements...")
    println("  Grid: $(N_xy) × $(N_xy)")

    # Build 2D product kernel (lengthscale must be in a vector)
    k_base = HalfIntegerMaternKernel(kernel_smoothness, [kernel_lengthscale])
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

    println("  Evaluations: $n_eval")
    println("  Derivatives: $n_deriv")
    println("  Integrals: $n_int")
    println("  Total: $(n_eval + n_deriv + n_int)")

    # Stack functionals
    L_stack = StackedLinearFunctional(L_eval, L_deriv, L_integ)

    # State layout for indexing
    state_layout = layout((
        f = n_eval,
        f_dx = n_deriv,
        f_int = n_int
    ))

    # Compute full covariance matrix
    println("Computing covariance matrix...")
    K = Matrix(L_stack(L_stack(k_prod)))
    println("  Size: $(size(K))")

    # Build coordinate matrices for each measurement type
    # Evaluations: flatten the 2D grid
    N_total = N_xy^2
    X_eval = zeros(2, N_total)
    idx = 1
    for iy in 1:N_xy
        for ix in 1:N_xy
            X_eval[1, idx] = Xs_base[ix]
            X_eval[2, idx] = Xs_base[iy]
            idx += 1
        end
    end

    # Derivatives at same locations as evaluations
    X_deriv = copy(X_eval)

    # Integrals at cell midpoints
    n_cells_1d = length(base_intervals)
    n_cells = n_cells_1d^2
    X_int = zeros(2, n_cells)
    idx = 1
    for iy in 1:n_cells_1d
        for ix in 1:n_cells_1d
            X_int[1, idx] = midpoint(base_intervals[ix])
            X_int[2, idx] = midpoint(base_intervals[iy])
            idx += 1
        end
    end

    return (
        K = K,
        L_eval = L_eval,
        L_deriv = L_deriv,
        L_integ = L_integ,
        state_layout = state_layout,
        X_eval = X_eval,
        X_deriv = X_deriv,
        X_int = X_int
    )
end

# ------------------------------------------------------------------------------
# Ordering Functions
# ------------------------------------------------------------------------------

"""
    create_ordering_2d(order_type, setup)

Create global ordering for either :eval_first or :int_first.

- :eval_first - [derivatives; integrals; evaluations] (evaluations coarsest/last)
- :int_first  - [derivatives; evaluations; integrals] (integrals coarsest/last)

Returns (P_global, ℓ_global, X_global)
"""
function create_ordering_2d(order_type::Symbol, setup; verbose=false)
    (; state_layout, X_eval, X_deriv, X_int) = setup

    if verbose
        println("  Coordinate info:")
        println("    X_eval: $(size(X_eval, 2)) points, x ∈ [$(minimum(X_eval[1,:])), $(maximum(X_eval[1,:]))]")
        println("    X_deriv: $(size(X_deriv, 2)) points, x ∈ [$(minimum(X_deriv[1,:])), $(maximum(X_deriv[1,:]))]")
        println("    X_int: $(size(X_int, 2)) points, x ∈ [$(minimum(X_int[1,:])), $(maximum(X_int[1,:]))]")
    end

    # Compute maximin orderings for evaluations and integrals
    P_eval, ℓ_eval = reverse_maximin_ordering(X_eval)
    P_int, ℓ_int = reverse_maximin_ordering(X_int)

    # Map to global indices
    P_eval_global = indices(state_layout, :f)[P_eval]
    P_int_global = indices(state_layout, :f_int)[P_int]
    P_deriv_global = indices(state_layout, :f_dx)

    # Build global coordinate matrix (for sparsity pattern)
    X_global_base = hcat(X_eval, X_deriv, X_int)

    if order_type == :eval_first
        # Evaluations are coarsest (rightmost columns in Cholesky)
        # Order: [derivatives; integrals; evaluations]
        coarsest_ℓ = ℓ_eval[P_eval[1]]

        # Derivatives get the coarsest evaluation lengthscale
        ℓ_deriv = fill(coarsest_ℓ, length(P_deriv_global))
        ℓ_int_assigned = fill(coarsest_ℓ, length(P_int_global))

        P_global = [P_deriv_global; P_int_global; P_eval_global]
        ℓ_global = [ℓ_deriv; ℓ_int_assigned; ℓ_eval]
    elseif order_type == :int_first
        # Integrals are coarsest (rightmost columns in Cholesky)
        # Order: [derivatives; evaluations; integrals]
        coarsest_ℓ = ℓ_int[P_int[1]]

        ℓ_deriv = fill(coarsest_ℓ, length(P_deriv_global))
        ℓ_eval_assigned = fill(coarsest_ℓ, length(P_eval_global))

        P_global = [P_deriv_global; P_eval_global; P_int_global]
        ℓ_global = [ℓ_deriv; ℓ_eval_assigned; ℓ_int]
    else
        error("Unknown order_type: $order_type. Use :eval_first or :int_first")
    end

    X_global = X_global_base

    return P_global, ℓ_global, X_global
end

# ------------------------------------------------------------------------------
# Sparsity Sweep
# ------------------------------------------------------------------------------

"""
    run_sparsity_sweep(K, P_global, ℓ_global, X_global, ρ_values; verbose=true)

For each ρ value, compute sparse Cholesky approximation and measure:
- Fill-in percentage
- KL divergence from exact distribution

Returns vector of named tuples with results.
"""
function run_sparsity_sweep(K, P_global, ℓ_global, X_global, ρ_values; verbose=true)
    n = size(K, 1)

    # Regularize K
    K_reg = Symmetric(K + 1e-10 * I)

    # Compute exact covariance Cholesky (permuted order)
    K_perm = K_reg[P_global, P_global]
    C_cov = cholesky(K_perm)

    results = []

    for ρ in ρ_values
        if verbose
            print("  ρ = $ρ: ")
        end

        # Create sparsity pattern
        S = sparsity_pattern_from_ordering(X_global, P_global, ℓ_global, ρ)
        nnz_S = nnz(S)

        # Compute sparse Cholesky of precision using original K with permutation
        K_P = PermutedMatrix(K_reg, P_global)
        sparse_approximate_cholesky!(K_P, S)

        # S now contains lower triangular L where Θ ≈ L L'
        # Compute KL divergence (K_perm is the permuted covariance)
        kl = kl_divergence(C_cov, S, K_perm)

        # Fill-in percentage (nnz_S was computed before sparse_approximate_cholesky!)
        max_nnz = n * (n + 1) ÷ 2
        fill_in_pct = 100 * nnz_S / max_nnz

        if verbose
            println("fill-in = $(round(fill_in_pct, digits=1))%, KL = $(round(kl, digits=4))")
        end

        push!(results, (
            ρ = ρ,
            fill_in_pct = fill_in_pct,
            kl_divergence = kl,
            nnz = nnz_S
        ))
    end

    return results
end

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------

# Color scheme matching other paper figures
const COLORS = (
    eval_first = :royalblue,
    int_first = :crimson,
)

"""
    create_sparsity_accuracy_figure(results_eval, results_int; output_dir)

Create publication figure with three panels:
- Panel A: Fill-in % vs ρ
- Panel B: KL divergence vs ρ
- Panel C: KL divergence vs fill-in % (Pareto frontier)

Uses TuePlots ICML format for publication-ready output.
Since this figure appears in the appendix (full-width single column), we use single_column=false.
"""
function create_sparsity_accuracy_figure(results_eval, results_int;
        output_dir = "figures"
    )

    # Extract data
    ρ_eval = [r.ρ for r in results_eval]
    ρ_int = [r.ρ for r in results_int]
    fill_eval = [r.fill_in_pct for r in results_eval]
    fill_int = [r.fill_in_pct for r in results_int]
    kl_eval = [r.kl_divergence for r in results_eval]
    kl_int = [r.kl_divergence for r in results_int]

    # Set up TuePlots theme for ICML appendix (full-width)
    theme = Theme(
        TuePlots.SETTINGS[:ICML];
        font=true, fontsize=true, figsize=true,
        single_column=false,  # Full-width for appendix
        nrows=1, ncols=3,
        subplot_height_to_width_ratio=1.1,  # Taller panels
    )
    set_theme!(theme)

    fig = Figure()

    # Marker size (smaller for publication)
    ms = 6

    # Panel A: Fill-in vs ρ
    ax1 = Axis(fig[1, 1],
        xlabel = L"Sparsity parameter $\rho$",
        ylabel = "Fill-in (%)",
    )

    scatterlines!(ax1, ρ_eval, fill_eval,
        color = COLORS.eval_first, marker = :circle, markersize = ms,
        label = "Evaluations first")
    scatterlines!(ax1, ρ_int, fill_int,
        color = COLORS.int_first, marker = :rect, markersize = ms,
        label = "Integrals first")

    # Panel B: KL vs ρ
    ax2 = Axis(fig[1, 2],
        xlabel = L"Sparsity parameter $\rho$",
        ylabel = "KL divergence",
        yscale = log10
    )

    scatterlines!(ax2, ρ_eval, kl_eval,
        color = COLORS.eval_first, marker = :circle, markersize = ms)
    scatterlines!(ax2, ρ_int, kl_int,
        color = COLORS.int_first, marker = :rect, markersize = ms)

    # Panel C: KL vs fill-in (the key plot!)
    ax3 = Axis(fig[1, 3],
        xlabel = "Fill-in (%)",
        ylabel = "KL divergence",
        yscale = log10
    )

    scatterlines!(ax3, fill_eval, kl_eval,
        color = COLORS.eval_first, marker = :circle, markersize = ms)
    scatterlines!(ax3, fill_int, kl_int,
        color = COLORS.int_first, marker = :rect, markersize = ms)

    # Add panel labels inside plots (top center)
    text!(ax1, 0.5, 0.95, text="(a)", align=(:center, :top), space=:relative, fontsize=9, font=:bold)
    text!(ax2, 0.5, 0.95, text="(b)", align=(:center, :top), space=:relative, fontsize=9, font=:bold)
    text!(ax3, 0.5, 0.95, text="(c)", align=(:center, :top), space=:relative, fontsize=9, font=:bold)

    # Shared legend below all panels
    Legend(fig[2, :], ax1, orientation = :horizontal, framevisible = false)

    # Reset theme before saving
    set_theme!()

    # Save
    mkpath(output_dir)
    save(joinpath(output_dir, "sparsity_accuracy_tradeoff.pdf"), fig)
    save(joinpath(output_dir, "sparsity_accuracy_tradeoff.png"), fig, px_per_unit=3)
    println("\nSaved: sparsity_accuracy_tradeoff.pdf/png")

    return fig
end

# ------------------------------------------------------------------------------
# Main Experiment
# ------------------------------------------------------------------------------

"""
    run_experiment(; N_xy, kernel_lengthscale, kernel_smoothness, ρ_values)

Run the full sparsity/accuracy tradeoff experiment.
"""
function run_experiment(;
        N_xy = 35,
        kernel_lengthscale = 0.1,
        kernel_smoothness = 1,
        ρ_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0],
        output_dir = "figures"
    )

    println("=" ^ 60)
    println("Sparsity/Accuracy Tradeoff Experiment")
    println("=" ^ 60)

    # Setup measurements
    setup = setup_2d_measurements(;
        N_xy = N_xy,
        kernel_lengthscale = kernel_lengthscale,
        kernel_smoothness = kernel_smoothness
    )

    # Run sweep for evaluations-first ordering
    # Order: [derivatives, integrals, evaluations]
    println("\n--- Evaluations First Ordering ---")
    P_eval, ℓ_eval, X_eval = create_ordering_2d(:eval_first, setup; verbose=true)
    results_eval = run_sparsity_sweep(setup.K, P_eval, ℓ_eval, X_eval, ρ_values)

    # Run sweep for integrals-first ordering
    # Order: [derivatives, evaluations, integrals]
    println("\n--- Integrals First Ordering ---")
    P_int, ℓ_int, X_int = create_ordering_2d(:int_first, setup)
    results_int = run_sparsity_sweep(setup.K, P_int, ℓ_int, X_int, ρ_values)

    # Create figure
    println("\nCreating figure...")
    fig = create_sparsity_accuracy_figure(results_eval, results_int; output_dir)

    # Print summary table
    println("\n" * "=" ^ 60)
    println("Summary")
    println("=" ^ 60)
    println("\nEvaluations First:")
    println("  ρ\t\tFill-in\t\tKL")
    for r in results_eval
        println("  $(r.ρ)\t\t$(round(r.fill_in_pct, digits=1))%\t\t$(round(r.kl_divergence, sigdigits=4))")
    end

    println("\nIntegrals First:")
    println("  ρ\t\tFill-in\t\tKL")
    for r in results_int
        println("  $(r.ρ)\t\t$(round(r.fill_in_pct, digits=1))%\t\t$(round(r.kl_divergence, sigdigits=4))")
    end

    return (
        setup = setup,
        results_eval = results_eval,
        results_int = results_int,
        fig = fig
    )
end

# ------------------------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------------------------

using ArgParse

function parse_commandline()
    s = ArgParseSettings(
        description = "Compare sparsity/accuracy tradeoff for different orderings"
    )

    @add_arg_table! s begin
        "--nxy", "-n"
            help = "Grid size (NxN)"
            arg_type = Int
            default = 35
        "--lengthscale", "-l"
            help = "Kernel lengthscale"
            arg_type = Float64
            default = 0.1
        "--smoothness", "-s"
            help = "Matérn smoothness parameter"
            arg_type = Int
            default = 1
        "--rho-values", "-r"
            help = "Comma-separated list of ρ values to test"
            arg_type = String
            default = "1.0,1.5,2.0,2.5,3.0,4.0,5.0,7.0,10.0"
        "--output-dir", "-o"
            help = "Output directory for figures"
            arg_type = String
            default = "figures"
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    # Parse ρ values from comma-separated string
    ρ_values = parse.(Float64, split(args["rho-values"], ","))

    run_experiment(
        N_xy = args["nxy"],
        kernel_lengthscale = args["lengthscale"],
        kernel_smoothness = args["smoothness"],
        ρ_values = ρ_values,
        output_dir = args["output-dir"]
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
