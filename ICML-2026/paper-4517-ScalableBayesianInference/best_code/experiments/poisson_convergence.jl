"""
1D Poisson Equation Convergence Study

Demonstrates GP-FVM convergence rates for different Matern kernel orders.
Analogous to how classical FVM achieves higher order via polynomial reconstruction.

Solves: -u''(x) = f(x) on [0,1], with u(0) = u(1) = 0

FVM weak form over cell [x_i, x_{i+1}]:
    u'(x_i) - u'(x_{i+1}) = integral_cell f dx

Key insight: We model BOTH u and f with GP priors (joint prior).
Source integrals come from Bayesian quadrature, not numerical integration.

Test cases:
- Smooth: f(x) = sin(2*pi*x), u_exact = sin(2*pi*x) / (4*pi^2)
- Rough: f(x) = indicator[0.25, 0.75], u = piecewise polynomial (C^1)

Expected:
- Smooth: Higher Matern order -> faster convergence
- Rough: All Matern orders saturate at same rate (limited by solution regularity)
"""

using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
using Statistics: mean as stats_mean
using CairoMakie
using TuePlots
using ArgParse
using Printf

import GaussianMarkovRandomFields: mean, std, precision_matrix

# ==============================================================================
# Analytical Solutions
# ==============================================================================

"""
Smooth source function: f(x) = sin(2*pi*x)
"""
smooth_source(x) = sin(2*pi*x)

"""
Exact solution for smooth source.
Verify: u(x) = sin(2*pi*x) / (4*pi^2)
u'(x) = 2*pi*cos(2*pi*x) / (4*pi^2) = cos(2*pi*x) / (2*pi)
u''(x) = -sin(2*pi*x)
-u''(x) = sin(2*pi*x) = f(x) ✓
"""
smooth_solution(x) = sin(2*pi*x) / (4*pi^2)

"""
Rough source function: f(x) = 1 on [0.25, 0.75], 0 elsewhere
"""
rough_source(x) = (0.25 <= x <= 0.75) ? 1.0 : 0.0

"""
Exact solution for rough (indicator) source.

Piecewise solution satisfying -u'' = f with u(0) = u(1) = 0:
- Region 1 [0, 0.25]:     u = a1*x            (-u'' = 0)
- Region 2 [0.25, 0.75]:  u = -x^2/2 + c2*x + d2  (-u'' = 1)
- Region 3 [0.75, 1]:     u = a3*(x-1)        (-u'' = 0, u(1) = 0)

Matching continuity of u and u' at x = 0.25 and x = 0.75:
"""
function rough_solution(x)
    # Solve the system of continuity equations
    # At x = 0.25:
    #   u continuous:  0.25*a1 = -0.25^2/2 + 0.25*c2 + d2
    #   u' continuous: a1 = -0.25 + c2
    # At x = 0.75:
    #   u continuous:  -0.75^2/2 + 0.75*c2 + d2 = a3*(0.75-1) = -0.25*a3
    #   u' continuous: -0.75 + c2 = a3

    # From u' continuity at 0.25: a1 = c2 - 0.25
    # From u' continuity at 0.75: a3 = c2 - 0.75

    # From u continuity at 0.25:
    #   0.25*(c2 - 0.25) = -0.03125 + 0.25*c2 + d2
    #   0.25*c2 - 0.0625 = -0.03125 + 0.25*c2 + d2
    #   d2 = -0.03125

    # From u continuity at 0.75:
    #   -0.28125 + 0.75*c2 - 0.03125 = -0.25*(c2 - 0.75)
    #   -0.3125 + 0.75*c2 = -0.25*c2 + 0.1875
    #   c2 = 0.5

    c2 = 0.5
    d2 = -0.03125
    a1 = c2 - 0.25  # = 0.25
    a3 = c2 - 0.75  # = -0.25

    if x <= 0.25
        return a1 * x
    elseif x <= 0.75
        return -x^2/2 + c2*x + d2
    else
        return a3 * (x - 1)
    end
end

# ==============================================================================
# FVM Constraint Builder
# ==============================================================================

"""
    build_poisson_fvm_constraint(xs, layout_u, layout_f, n_u)

Build FVM constraint matrix for 1D Poisson equation.

For each cell i: u'(x_i) - u'(x_{i+1}) - f_int[i] = 0

This links derivatives of u to integrals of f through physics.
"""
function build_poisson_fvm_constraint(xs, layout_u, layout_f, n_u)
    N = length(xs)
    n_cells = N - 1

    n_f = layout_f.total
    n_total = n_u + n_f

    u_dx_idx = indices(layout_u, :u_dx)
    f_int_idx = n_u .+ indices(layout_f, :f_int)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for i in 1:n_cells
        # FVM: u'(x_i) - u'(x_{i+1}) - f_int[i] = 0

        # u'(x_i) coefficient: +1
        push!(rows, i); push!(cols, u_dx_idx[i]); push!(vals, 1.0)

        # u'(x_{i+1}) coefficient: -1
        push!(rows, i); push!(cols, u_dx_idx[i+1]); push!(vals, -1.0)

        # f_int[i] coefficient: -1
        push!(rows, i); push!(cols, f_int_idx[i]); push!(vals, -1.0)
    end

    A_fvm = sparse(rows, cols, vals, n_cells, n_total)
    b_fvm = zeros(n_cells)

    return A_fvm, b_fvm
end

# ==============================================================================
# Main Solver
# ==============================================================================

"""
    solve_poisson_1d(; kwargs...)

Solve 1D Poisson equation using GP-FVM.

# Keyword Arguments
- `N=51`: Number of grid points
- `smoothness=2`: Matern kernel order (1=3/2, 2=5/2, 3=7/2)
- `lengthscale=0.15`: GP kernel lengthscale (fixed, matching solution scale)
- `rho=4.0`: Sparsity threshold
- `test_case=:smooth`: :smooth or :rough
- `constraint_noise=1e-6`: Noise for constraint enforcement
- `use_bayesian_quadrature=false`: If true, model source with GP and use Bayesian quadrature.
                                   If false, use exact numerical quadrature for source integrals.
- `verbose=true`: Print progress

# Returns
Named tuple with solution, errors, and diagnostics.
"""
function solve_poisson_1d(;
        N = 51,
        smoothness = 2,
        lengthscale = 0.15,
        rho = 4.0,
        test_case = :smooth,
        constraint_noise = 1e-6,
        use_bayesian_quadrature = false,
        verbose = false
    )

    # Select source and exact solution
    if test_case == :smooth
        f_func = smooth_source
        u_exact = smooth_solution
    else
        f_func = rough_source
        u_exact = rough_solution
    end

    # Grid setup
    xs = range(0.0, 1.0, length=N)
    intervals = intervals_from_endpoints(collect(xs))
    n_cells = N - 1
    dx = 1.0 / (N - 1)

    verbose && println("Grid: N=$N, $(n_cells) cells, dx=$(round(dx, sigdigits=3)), ls=$(lengthscale)")
    verbose && println("Mode: $(use_bayesian_quadrature ? "Bayesian quadrature" : "exact quadrature")")

    # Build kernel for u
    k_u = HalfIntegerMaternKernel(smoothness, [lengthscale])

    # Build functionals for u
    L_u_eval = EvaluationFunctional(xs)
    L_u_dx = L_u_eval ∘ PartialDerivative((1,))
    L_u_int = VectorizedLebesgueIntegral(intervals)

    # Build sparse precision for u
    verbose && println("Building sparse precision (rho=$rho)...")

    approx_u = sparse_precision([
        :u => L_u_eval,
        :u_dx => L_u_dx,
        :u_int => L_u_int
    ], k_u; ρ=rho, ordering=:integrals_coarsest)

    verbose && println("  u: $(approx_u)")

    layout_u = approx_u.layout
    n_u = approx_u.info.n

    if use_bayesian_quadrature
        # Joint GP approach: model f with GP, use Bayesian quadrature for integrals
        k_f = HalfIntegerMaternKernel(smoothness, [lengthscale])
        L_f_eval = EvaluationFunctional(xs)
        L_f_int = VectorizedLebesgueIntegral(intervals)

        approx_f = sparse_precision([
            :f => L_f_eval,
            :f_int => L_f_int
        ], k_f; ρ=rho, ordering=:integrals_coarsest)

        verbose && println("  f: $(approx_f)")

        layout_f = approx_f.layout
        n_f = approx_f.info.n
        n_total = n_u + n_f

        Q_joint = blockdiag(sparse(approx_u.Q), sparse(approx_f.Q))
        x0 = GMRF(zeros(n_total), Symmetric(Q_joint))

        verbose && println("Joint state dimension: $n_total")

        # Build FVM constraints linking u_dx and f_int
        A_fvm, b_fvm = build_poisson_fvm_constraint(collect(xs), layout_u, layout_f, n_u)

        # Prescribe source function f at evaluation points
        f_eval_indices = n_u .+ indices(layout_f, :f)
        f_values = f_func.(collect(xs))

        verbose && println("Prescribing source at $(length(f_eval_indices)) points...")
        x_f = prescribe_indices(x0, f_eval_indices, f_values; noise_std=constraint_noise)

        # Apply FVM constraints
        verbose && println("Applying $(size(A_fvm, 1)) FVM constraints...")
        n_fvm = size(A_fvm, 1)
        Q_fvm = (1.0 / constraint_noise^2) * sparse(I, n_fvm, n_fvm)
        x_constrained = linear_condition(x_f; A=A_fvm, Q_ϵ=Q_fvm, y=b_fvm)

        approx_f_out = approx_f
    else
        # Exact quadrature: compute source integrals analytically/numerically
        x0 = GMRF(zeros(n_u), approx_u.Q)

        # Compute exact source integrals
        f_int_exact = zeros(n_cells)
        for i in 1:n_cells
            x1, x2 = (i-1)*dx, i*dx
            if test_case == :smooth
                # Exact integral of sin(2*pi*x)
                f_int_exact[i] = (-cos(2*pi*x2) + cos(2*pi*x1)) / (2*pi)
            else
                # Exact integral of indicator function
                # Integral over [x1, x2] of 1_{[0.25, 0.75]}
                a, b = max(x1, 0.25), min(x2, 0.75)
                f_int_exact[i] = a < b ? (b - a) : 0.0
            end
        end

        # Build FVM constraints: u'(x_i) - u'(x_{i+1}) = f_int_exact[i]
        u_dx_idx = indices(layout_u, :u_dx)
        rows = Int[]
        cols = Int[]
        vals = Float64[]

        for i in 1:n_cells
            push!(rows, i); push!(cols, u_dx_idx[i]); push!(vals, 1.0)
            push!(rows, i); push!(cols, u_dx_idx[i+1]); push!(vals, -1.0)
        end

        A_fvm = sparse(rows, cols, vals, n_cells, n_u)
        b_fvm = f_int_exact

        verbose && println("Applying $(n_cells) FVM constraints with exact source integrals...")
        Q_fvm = (1.0 / constraint_noise^2) * sparse(I, n_cells, n_cells)
        x_constrained = linear_condition(x0; A=A_fvm, Q_ϵ=Q_fvm, y=b_fvm)

        approx_f_out = nothing
    end

    # Apply Dirichlet boundary conditions: u(0) = u(1) = 0
    bc_indices = [indices(layout_u, :u)[1], indices(layout_u, :u)[end]]
    bc_values = [0.0, 0.0]

    verbose && println("Applying Dirichlet BCs...")
    x_solution = prescribe_indices(x_constrained, bc_indices, bc_values; noise_std=constraint_noise)

    # Extract solution
    mu = mean(x_solution)
    sigma = std(x_solution)

    u_indices = indices(layout_u, :u)
    u_mean = mu[u_indices]
    u_std = sigma[u_indices]
    u_true = u_exact.(collect(xs))

    # Compute L2 error
    l2_error = sqrt(sum((u_mean - u_true).^2) * dx)
    max_error = maximum(abs.(u_mean - u_true))

    verbose && println("L2 error: $(l2_error)")
    verbose && println("Max error: $(max_error)")

    return (
        xs = collect(xs),
        u_mean = u_mean,
        u_std = u_std,
        u_true = u_true,
        l2_error = l2_error,
        max_error = max_error,
        N = N,
        smoothness = smoothness,
        approx_u = approx_u,
        approx_f = approx_f_out
    )
end

# ==============================================================================
# Convergence Study
# ==============================================================================

"""
    run_convergence_study(; kwargs...)

Run convergence study varying N and Matern smoothness.

# Keyword Arguments
- `N_values`: Grid sizes to test (default: [20, 40, 80, 160, 320])
- `smoothness_values`: Matern orders to test (default: [1, 2, 3])
- `test_case`: :smooth or :rough
- `rho_base`: Base sparsity threshold (will be scaled with N)
- `use_bayesian_quadrature`: Use joint GP prior with Bayesian quadrature
- `verbose`: Print progress

# Returns
- `N_values`: Grid sizes
- `results`: Dict mapping smoothness => vector of L2 errors
- `rates`: Dict mapping smoothness => fitted convergence rate
"""
function run_convergence_study(;
        N_values = [5, 10, 20, 40, 80, 160],
        smoothness_values = [1, 2, 3],
        test_case = :smooth,
        rho_base = 5.0,
        use_bayesian_quadrature = false,
        verbose = true
    )

    verbose && println("=" ^ 60)
    verbose && println("Poisson Convergence Study")
    verbose && println("=" ^ 60)
    verbose && println("Test case: $test_case")
    verbose && println("Grid sizes: $N_values")
    verbose && println("Matern orders: $smoothness_values (3/2, 5/2, 7/2)")
    verbose && println("Mode: $(use_bayesian_quadrature ? "Bayesian quadrature" : "exact quadrature")")
    verbose && println()

    results = Dict{Int, Vector{Float64}}()
    rates = Dict{Int, Float64}()

    # Scale rho with N to maintain accuracy
    # rho increases logarithmically with N
    N_ref = N_values[1]
    rho_for_N(N) = rho_base + 3.0 * log2(N / N_ref)

    for nu in smoothness_values
        verbose && println("Matern $(2*nu+1)/2:")
        errors = Float64[]

        for N in N_values
            rho = rho_for_N(N)
            result = solve_poisson_1d(;
                N = N,
                smoothness = nu,
                rho = rho,
                test_case = test_case,
                use_bayesian_quadrature = use_bayesian_quadrature,
                verbose = false
            )
            push!(errors, result.l2_error)
            verbose && @printf("  N=%4d (rho=%.1f): L2 error = %.3e\n", N, rho, result.l2_error)
        end

        results[nu] = errors

        # Fit convergence rate (slope in log-log)
        log_N = log10.(N_values)
        log_err = log10.(errors)

        # Linear regression: log_err = a + b * log_N
        # Rate = -b (negative because error decreases with N)
        n = length(N_values)
        mean_logN = stats_mean(log_N)
        mean_logE = stats_mean(log_err)
        slope = sum((log_N .- mean_logN) .* (log_err .- mean_logE)) /
                sum((log_N .- mean_logN).^2)
        rate = -slope

        rates[nu] = rate
        verbose && @printf("  Convergence rate: %.2f\n\n", rate)
    end

    return N_values, results, rates
end

# ==============================================================================
# Plotting
# ==============================================================================

"""
    plot_convergence(N_values, results, rates; kwargs...)

Create log-log convergence plot.
"""
function plot_convergence(N_values, results, rates;
        test_case = :smooth,
        output_dir = joinpath(@__DIR__, "..", "figures")
    )

    fig = Figure(size=(700, 500), fontsize=14)

    ax = Axis(fig[1, 1],
        xlabel = "Grid points N",
        ylabel = "L² error",
        title = "GP-FVM Convergence: 1D Poisson ($(test_case))",
        xscale = log10,
        yscale = log10,
        xticks = N_values,
        xticklabelrotation = 0
    )

    colors = [:blue, :red, :forestgreen]
    markers = [:circle, :rect, :diamond]

    for (i, nu) in enumerate([1, 2, 3])
        if haskey(results, nu)
            errors = results[nu]
            rate = rates[nu]

            scatterlines!(ax, N_values, errors,
                color = colors[i],
                marker = markers[i],
                markersize = 12,
                linewidth = 2,
                label = "Matern $(2*nu+1)/2 (rate ≈ $(round(rate, digits=2)))"
            )
        end
    end

    # Add reference slope lines
    N_ref = [N_values[1], N_values[end]]
    y_base = results[1][1] * 0.3  # Start below first point

    for (rate, style, label) in [(1.0, :dash, "O(N⁻¹)"),
                                  (2.0, :dot, "O(N⁻²)"),
                                  (3.0, :dashdot, "O(N⁻³)")]
        y_ref = y_base .* (N_ref ./ N_ref[1]).^(-rate)
        lines!(ax, N_ref, y_ref,
            color = :gray,
            linestyle = style,
            linewidth = 1.5,
            label = label
        )
    end

    axislegend(ax, position = :lb)

    mkpath(output_dir)
    filename_base = "poisson_convergence_$(test_case)"
    save(joinpath(output_dir, "$(filename_base).pdf"), fig)
    save(joinpath(output_dir, "$(filename_base).png"), fig, px_per_unit=3)

    println("Saved: $(filename_base).pdf/png")

    return fig
end

"""
    plot_solution(result; output_dir)

Plot a single solution with uncertainty bands.
"""
function plot_solution(result; output_dir = joinpath(@__DIR__, "..", "figures"))
    (; xs, u_mean, u_std, u_true, N, smoothness, l2_error) = result

    fig = Figure(size=(700, 400), fontsize=14)

    ax = Axis(fig[1, 1],
        xlabel = "x",
        ylabel = "u(x)",
        title = "Poisson Solution (N=$N, Matern $(2*smoothness+1)/2, L²=$(round(l2_error, sigdigits=3)))"
    )

    # Uncertainty band
    band!(ax, xs, u_mean .- 2*u_std, u_mean .+ 2*u_std,
        color = (:blue, 0.2),
        label = "±2σ"
    )

    # Mean solution
    lines!(ax, xs, u_mean,
        color = :blue,
        linewidth = 2,
        label = "GP-FVM mean"
    )

    # True solution
    lines!(ax, xs, u_true,
        color = :red,
        linewidth = 2,
        linestyle = :dash,
        label = "Exact"
    )

    axislegend(ax, position = :rt)

    mkpath(output_dir)
    save(joinpath(output_dir, "poisson_solution.pdf"), fig)
    save(joinpath(output_dir, "poisson_solution.png"), fig, px_per_unit=3)

    return fig
end

# ==============================================================================
# Publication-Ready Plotting (ICML format via TuePlots)
# ==============================================================================

"""
    run_publication_experiment(; verbose=true)

Run the dense vs sparse GP-FVM comparison experiment for publication.

Returns results for both configurations:
- Dense: high rho (essentially exact), shows theoretical rates
- Sparse: fixed rho=5 (~16% fill at N=160), shows limits of sparse approximation
"""
function run_publication_experiment(; verbose=true)
    N_values = [10, 20, 40, 80, 160]
    smoothness_values = [1, 2, 3]

    # Fixed rho values
    rho_dense = 60.0   # High rho for dense (essentially exact)
    rho_sparse = 5.0   # Fixed sparse rho (~16% fill at N=160)

    results_dense = Dict{Int, Vector{Float64}}()
    rates_dense = Dict{Int, Float64}()
    results_sparse = Dict{Int, Vector{Float64}}()
    rates_sparse = Dict{Int, Float64}()

    # Helper for proper linear regression rate computation
    function compute_rate(N_vals, errors)
        log_N = log10.(N_vals)
        log_err = log10.(errors)
        mean_x = stats_mean(log_N)
        mean_y = stats_mean(log_err)
        slope = sum((log_N .- mean_x) .* (log_err .- mean_y)) / sum((log_N .- mean_x).^2)
        return -slope
    end

    verbose && println("=" ^ 60)
    verbose && println("Publication Experiment: Dense vs Sparse GP-FVM")
    verbose && println("=" ^ 60)

    # Dense GP-FVM with BQ
    verbose && println("\n(a) Dense GP-FVM (ρ=$rho_dense):")
    for nu in smoothness_values
        errors = Float64[]
        for N in N_values
            result = solve_poisson_1d(N=N, smoothness=nu, rho=rho_dense,
                                      test_case=:smooth, use_bayesian_quadrature=true, verbose=false)
            push!(errors, result.l2_error)
        end
        results_dense[nu] = errors

        # Fit rate using last 3 points with proper regression
        rates_dense[nu] = compute_rate(N_values[end-2:end], errors[end-2:end])
        verbose && @printf("  Matern %d/2: rate=%.2f\n", 2*nu+1, rates_dense[nu])
    end

    # Sparse GP-FVM with BQ
    verbose && println("\n(b) Sparse GP-FVM (ρ=$rho_sparse):")
    fill_pct = 0.0
    for nu in smoothness_values
        errors = Float64[]
        for N in N_values
            result = solve_poisson_1d(N=N, smoothness=nu, rho=rho_sparse,
                                      test_case=:smooth, use_bayesian_quadrature=true, verbose=false)
            push!(errors, result.l2_error)
            if nu == 2 && N == N_values[end]
                fill_pct = result.approx_u.info.fill_pct
            end
        end
        results_sparse[nu] = errors

        rates_sparse[nu] = compute_rate(N_values[end-2:end], errors[end-2:end])
        verbose && @printf("  Matern %d/2: rate=%+.2f\n", 2*nu+1, rates_sparse[nu])
    end

    verbose && println("\nSparsity at N=$(N_values[end]): $(round(fill_pct, digits=0))% fill")

    return (
        N_values = N_values,
        results_dense = results_dense,
        rates_dense = rates_dense,
        results_sparse = results_sparse,
        rates_sparse = rates_sparse,
        fill_pct = fill_pct
    )
end

"""
    plot_convergence_publication(experiment; single_column=true)

Create publication-ready convergence plot using TuePlots ICML format.

# Arguments
- `experiment`: Output from `run_publication_experiment()`
- `single_column`: If true, use single-column stacked layout (default). If false, full-width side-by-side.
- `output_path`: Output PDF path
"""
function plot_convergence_publication(experiment;
        single_column = true,
        output_path = joinpath(@__DIR__, "..", "figures", "poisson_convergence.pdf")
    )

    (; N_values, results_dense, rates_dense, results_sparse, rates_sparse, fill_pct) = experiment

    # Set up TuePlots theme - side-by-side layout for single column width
    if single_column
        theme = Theme(
            TuePlots.SETTINGS[:ICML];
            font=true, fontsize=true, figsize=true,
            single_column=true,
            nrows=1, ncols=2,
            subplot_height_to_width_ratio=0.8,
        )
    else
        theme = Theme(
            TuePlots.SETTINGS[:ICML];
            font=true, fontsize=true, figsize=true,
            single_column=false,
            nrows=1, ncols=2,
            subplot_height_to_width_ratio=0.65,
        )
    end
    set_theme!(theme)

    fig = Figure()

    colors = [:royalblue, :crimson, :forestgreen]
    markers = [:circle, :rect, :diamond]

    # Axis positions - side-by-side layout
    ax_dense = Axis(fig[1, 1], xlabel=L"Grid points $N$", ylabel=L"$L^2$ error",
                    xscale=log10, yscale=log10)
    ax_sparse = Axis(fig[1, 2], xlabel=L"Grid points $N$",
                     xscale=log10, yscale=log10)

    # Set x-ticks
    for ax in [ax_dense, ax_sparse]
        ax.xticks = ([10, 20, 40, 80, 160], ["10", "20", "40", "80", "160"])
    end

    # Theoretical rates for Matérn ν: rate = ν + 1/2 = 2, 3, 4 for 3/2, 5/2, 7/2
    theoretical_rates = Dict(1 => 2.0, 2 => 3.0, 3 => 4.0)

    # Helper to add reference lines anchored at midpoint, with parallel lines above/below
    function add_reference_lines!(ax, N_vals, errors, rate, color; spread=2.5)
        # Center line passes through middle data point with given rate
        mid_idx = div(length(N_vals) + 1, 2)
        N_mid = N_vals[mid_idx]
        err_mid = errors[mid_idx]
        const_term = log10(err_mid) + rate * log10(N_mid)
        # Three lines: one at data, one above (×spread), one below (÷spread)
        for multiplier in [1.0/spread, 1.0, spread]
            offset_const = const_term + log10(multiplier)
            ref_err = [10^(offset_const - rate * log10(N)) for N in N_vals]
            lines!(ax, N_vals, ref_err, color=(color, 0.4), linestyle=:dash, linewidth=0.8)
        end
    end

    # Panel (a): Dense GP-FVM
    for (i, nu) in enumerate([1, 2, 3])
        rate = rates_dense[nu]
        errors = results_dense[nu]
        scatterlines!(ax_dense, N_values, errors,
            color=colors[i], marker=markers[i],
            markersize=6, linewidth=1.2,
            label="$(2*nu+1)/2 ($(round(rate, digits=1)))"
        )
        # Add reference lines with theoretical rate
        add_reference_lines!(ax_dense, N_values, errors, theoretical_rates[nu], colors[i])
    end

    # Panel label - top center
    text!(ax_dense, 0.5, 0.98, text=rich(rich("(a)", font=:bold), " Dense"), align=(:center, :top),
          fontsize=8, space=:relative)

    # Panel (b): Sparse GP-FVM
    for (i, nu) in enumerate([1, 2, 3])
        rate = rates_sparse[nu]
        errors = results_sparse[nu]
        scatterlines!(ax_sparse, N_values, errors,
            color=colors[i], marker=markers[i],
            markersize=6, linewidth=1.2,
            label="$(2*nu+1)/2 ($(round(rate, digits=1)))"
        )
        # Add reference lines with theoretical rate
        add_reference_lines!(ax_sparse, N_values, errors, theoretical_rates[nu], colors[i])
    end

    # Panel label - top center
    text!(ax_sparse, 0.5, 0.98, text=rich(rich("(b)", font=:bold), " Sparse"), align=(:center, :top),
          fontsize=8, space=:relative)

    # Link y-axes for consistent comparison
    linkaxes!(ax_dense, ax_sparse)

    # Save
    mkpath(dirname(output_path))
    save(output_path, fig, pt_per_unit=1)

    # Also save PNG for preview
    png_path = replace(output_path, ".pdf" => ".png")
    save(png_path, fig, px_per_unit=3)

    println("Saved: $output_path")
    println("Saved: $png_path")

    # Reset theme
    set_theme!()

    return fig
end

# ==============================================================================
# CLI
# ==============================================================================

function parse_commandline()
    s = ArgParseSettings(
        description = "1D Poisson convergence study for GP-FVM"
    )

    @add_arg_table! s begin
        "--test-case", "-t"
            help = "Test case: smooth or rough"
            arg_type = Symbol
            default = :smooth
        "--rho", "-r"
            help = "Base sparsity threshold (scales with N)"
            arg_type = Float64
            default = 5.0
        "--output-dir", "-o"
            help = "Output directory"
            arg_type = String
            default = joinpath(@__DIR__, "..", "figures")
        "--n-values"
            help = "Grid sizes (comma-separated)"
            arg_type = String
            default = "5,10,20,40,80,160"
        "--smoothness-values"
            help = "Matern orders (comma-separated)"
            arg_type = String
            default = "1,2,3"
        "--single", "-s"
            help = "Run single solve with specified N"
            arg_type = Int
            default = 0
        "--bayesian-quadrature", "-b"
            help = "Use Bayesian quadrature for source integrals (joint GP)"
            action = :store_true
        "--quiet", "-q"
            help = "Suppress output"
            action = :store_true
        "--publication", "-p"
            help = "Generate publication-ready figure (ICML format, dense vs sparse comparison)"
            action = :store_true
        "--full-width"
            help = "Use full-width layout instead of single-column (only with --publication)"
            action = :store_true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    test_case = args["test-case"]
    rho_base = args["rho"]
    output_dir = args["output-dir"]
    verbose = !args["quiet"]
    use_bq = args["bayesian-quadrature"]

    if args["publication"]
        # Publication mode: run fixed experiment and generate ICML-format figure
        experiment = run_publication_experiment(; verbose=verbose)

        single_column = !args["full-width"]
        output_path = joinpath(output_dir, "poisson_convergence.pdf")

        plot_convergence_publication(experiment;
            single_column = single_column,
            output_path = output_path
        )
    elseif args["single"] > 0
        # Single solve mode
        result = solve_poisson_1d(;
            N = args["single"],
            smoothness = 2,
            rho = rho_base,
            test_case = test_case,
            use_bayesian_quadrature = use_bq,
            verbose = verbose
        )
        plot_solution(result; output_dir = output_dir)
    else
        # Convergence study mode
        N_values = parse.(Int, split(args["n-values"], ","))
        smoothness_values = parse.(Int, split(args["smoothness-values"], ","))

        N_values, results, rates = run_convergence_study(;
            N_values = N_values,
            smoothness_values = smoothness_values,
            test_case = test_case,
            rho_base = rho_base,
            use_bayesian_quadrature = use_bq,
            verbose = verbose
        )

        plot_convergence(N_values, results, rates;
            test_case = test_case,
            output_dir = output_dir
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
