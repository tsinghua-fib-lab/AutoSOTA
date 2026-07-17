"""
Calibration experiment: run GP-FVM on many random inverse problem instances
and evaluate UQ calibration via coverage, interval scores, and whitened residuals.

Data generating process:
- 2 Gaussian sources at random locations/amplitudes
- Forward PDE solve → concentration
- K observations at fixed locations + noise

Usage:
    # Dry run (3 instances, quick validation)
    julia --project=../.. calibration_experiment.jl --n-instances 3

    # Full run
    julia --project=../.. calibration_experiment.jl --n-instances 200

    # HPC
    julia --project=../.. calibration_experiment.jl --n-instances 200 --no-plot
"""

using LinearAlgebra, SparseArrays
using Random
using Statistics
using Printf
using NPZ
using CSV, DataFrames
using CairoMakie
using SpecialFunctions
using ArgParse

push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields

import GaussianMarkovRandomFields: mean, std, precision_map
import FunctionalGPs: ⊗

include("run_gpfvm.jl")

# ==============================================================================
# Data generating process
# ==============================================================================

function random_two_source_problem(rng::AbstractRNG;
        vx=1.0, vy=0.0, D=0.05, c_inflow=0.0,
        obs_xs::Vector{Float64}, obs_ys::Vector{Float64},
        noise_std=0.1)

    # Random source parameters
    sources = GaussianSource[]
    for _ in 1:2
        push!(sources, GaussianSource(
            0.2 + 0.6 * rand(rng),   # x ∈ [0.2, 0.8]
            0.2 + 0.6 * rand(rng),   # y ∈ [0.2, 0.8]
            -3.0 + 6.0 * rand(rng),  # amplitude ∈ [-3, 3]
            0.05 + 0.10 * rand(rng), # width ∈ [0.05, 0.15]
        ))
    end

    observations = [Observation(ox, oy) for (ox, oy) in zip(obs_xs, obs_ys)]

    return SourceIdentificationProblem(
        vx, vy, D, (0.0, 1.0, 0.0, 1.0), c_inflow,
        sources, observations, noise_std, 0  # seed=0, we manage RNG externally
    )
end

function solve_forward(xs, ys, prob::SourceIdentificationProblem)
    Nx, Ny = length(xs), length(ys)
    Δx = xs[2] - xs[1]; Δy = ys[2] - ys[1]
    n_cells_x, n_cells_y = Nx - 1, Ny - 1

    s_int = zeros(n_cells_x, n_cells_y)
    for cj in 1:n_cells_y, ci in 1:n_cells_x
        cx = 0.5 * (xs[ci] + xs[ci+1])
        cy = 0.5 * (ys[cj] + ys[cj+1])
        s_int[ci, cj] = evaluate_source(prob, cx, cy) * Δx * Δy
    end

    n_nodes = Nx * Ny
    node_idx(i, j) = (j - 1) * Nx + i
    rows = Int[]; cols = Int[]; vals = Float64[]
    b = zeros(n_nodes)

    for j in 1:Ny, i in 1:Nx
        idx = node_idx(i, j)
        if i == 1
            push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
            b[idx] = prob.c_inflow
        elseif i == Nx
            push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
            push!(rows, idx); push!(cols, node_idx(i-1, j)); push!(vals, -1.0)
        elseif j == 1
            push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
            push!(rows, idx); push!(cols, node_idx(i, j+1)); push!(vals, -1.0)
        elseif j == Ny
            push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
            push!(rows, idx); push!(cols, node_idx(i, j-1)); push!(vals, -1.0)
        else
            push!(rows, idx); push!(cols, idx); push!(vals, prob.vx * Δy)
            push!(rows, idx); push!(cols, node_idx(i-1, j)); push!(vals, -prob.vx * Δy)
            diff_x = prob.D / Δx * Δy; diff_y = prob.D / Δy * Δx
            push!(rows, idx); push!(cols, idx); push!(vals, 2diff_x + 2diff_y)
            push!(rows, idx); push!(cols, node_idx(i+1, j)); push!(vals, -diff_x)
            push!(rows, idx); push!(cols, node_idx(i-1, j)); push!(vals, -diff_x)
            push!(rows, idx); push!(cols, node_idx(i, j+1)); push!(vals, -diff_y)
            push!(rows, idx); push!(cols, node_idx(i, j-1)); push!(vals, -diff_y)
            source = 0.0
            for (ci, cj) in [(i-1, j-1), (i, j-1), (i-1, j), (i, j)]
                if 1 <= ci <= n_cells_x && 1 <= cj <= n_cells_y
                    source += 0.25 * s_int[ci, cj]
                end
            end
            b[idx] = source
        end
    end

    c = reshape(sparse(rows, cols, vals, n_nodes, n_nodes) \ b, Nx, Ny)
    s = [evaluate_source(prob, xs[i], ys[j]) for i in 1:Nx, j in 1:Ny]
    return c, s
end

# ==============================================================================
# Single instance
# ==============================================================================

function run_instance(prob::SourceIdentificationProblem, xs, ys, rng;
        ρ, lengthscale_c, lengthscale_s, output_scale, output_scale_c,
        noise_std, nominal_levels)

    Nx, Ny = length(xs), length(ys)

    # Forward solve
    c_true, s_true = solve_forward(xs, ys, prob)

    # Noisy observations
    obs_x = [o.x for o in prob.observations]
    obs_y = [o.y for o in prob.observations]
    obs_ix = [argmin(abs.(xs .- ox)) for ox in obs_x]
    obs_iy = [argmin(abs.(ys .- oy)) for oy in obs_y]
    obs_c = [c_true[ix, iy] for (ix, iy) in zip(obs_ix, obs_iy)] .+ noise_std * randn(rng, length(obs_x))

    data = Dict{String, Any}(
        "xs" => xs, "ys" => ys,
        "obs_x" => obs_x, "obs_y" => obs_y, "obs_c" => obs_c,
        "noise_std" => noise_std,
        "source_x" => [s.x for s in prob.sources],
        "source_y" => [s.y for s in prob.sources],
    )

    # GP-FVM solve
    result = solve_source_identification(prob, data;
        ρ=ρ, lengthscale_c=lengthscale_c, lengthscale_s=lengthscale_s,
        output_scale=output_scale, output_scale_c=output_scale_c, verbose=false)

    # --- Pointwise metrics ---
    norminv(p) = √2 * erfinv(2p - 1)
    z_levels = [norminv(0.5 + α/2) for α in nominal_levels]

    # Coverage + interval width for source
    s_err = abs.(s_true .- result.s_mean)
    s_coverage = Dict{Float64, Float64}()
    for (α, z) in zip(nominal_levels, z_levels)
        s_coverage[α] = Statistics.mean(s_err .< z .* result.s_std)
    end
    s_width_95 = Statistics.mean(2 * norminv(0.975) .* result.s_std)

    # Coverage + interval width for concentration
    c_err = abs.(c_true .- result.c_mean)
    c_coverage = Dict{Float64, Float64}()
    for (α, z) in zip(nominal_levels, z_levels)
        c_coverage[α] = Statistics.mean(c_err .< z .* result.c_std)
    end
    c_width_95 = Statistics.mean(2 * norminv(0.975) .* result.c_std)

    # Interval score (95%) for both fields
    function interval_score(truth, mean_f, std_f, α)
        z_α = norminv(0.5 + α/2)
        l = mean_f .- z_α .* std_f
        u = mean_f .+ z_α .* std_f
        width = u .- l
        penalty_low = (2/α) .* (l .- truth) .* (truth .< l)
        penalty_high = (2/α) .* (truth .- u) .* (truth .> u)
        return Statistics.mean(width .+ penalty_low .+ penalty_high)
    end

    s_iscore_95 = interval_score(s_true, result.s_mean, result.s_std, 0.95)
    c_iscore_95 = interval_score(c_true, result.c_mean, result.c_std, 0.95)

    # --- Whitened residuals via marginal posterior ---
    Q_post = precision_map(result.gmrf)
    F_post = cholesky(Symmetric(Q_post))

    function whitened_residuals(F, field_indices, truth_vec, mean_vec)
        nf = length(field_indices)
        Σ_block = zeros(nf, nf)
        e = zeros(size(F, 1))
        for (j, idx) in enumerate(field_indices)
            e[idx] = 1.0
            col = F \ e
            Σ_block[:, j] = col[field_indices]
            e[idx] = 0.0
        end
        L_field = cholesky(Symmetric(Σ_block)).L
        return L_field \ (truth_vec - mean_vec)
    end

    c_eval_idx = indices(result.info.layout_c, :c)
    s_eval_idx = result.info.n_c .+ indices(result.info.layout_s, :s)

    z_c = whitened_residuals(F_post, c_eval_idx, vec(c_true), vec(result.c_mean))
    z_s = whitened_residuals(F_post, s_eval_idx, vec(s_true), vec(result.s_mean))

    # Mahalanobis distances
    d2_c = sum(z_c.^2)
    d2_s = sum(z_s.^2)

    # Marginal z-scores (for output scale calibration)
    marginal_z_s = vec(s_err ./ result.s_std)  # |truth - mean| / std
    marginal_z_c = vec(c_err ./ result.c_std)

    return (;
        s_coverage, c_coverage, s_width_95, c_width_95,
        s_iscore_95, c_iscore_95,
        z_c, z_s, d2_c, d2_s,
        marginal_z_s, marginal_z_c,
        # Spatial coverage at 95% for maps
        s_covered_95 = s_err .< norminv(0.975) .* result.s_std,
        c_covered_95 = c_err .< norminv(0.975) .* result.c_std,
    )
end

# ==============================================================================
# Main experiment
# ==============================================================================

function run_calibration_experiment(;
        n_instances::Int, N::Int, ρ::Float64,
        lengthscale_c::Float64, lengthscale_s::Float64,
        output_scale::Float64, output_scale_c::Union{Float64,Nothing},
        n_obs::Int, noise_std::Float64,
        n_calibration::Int=10,
        seed::Int, output_dir::String, make_plots::Bool)

    println("=" ^ 60)
    println("Calibration Experiment")
    println("=" ^ 60)
    @printf("Instances: %d | N: %d | ρ: %.1f | n_obs: %d\n", n_instances, N, ρ, n_obs)
    @printf("ℓ_c=%.2f | ℓ_s=%.2f | σ²=%.2f (initial) | σ²_c=%s\n",
            lengthscale_c, lengthscale_s, output_scale,
            isnothing(output_scale_c) ? "default" : string(output_scale_c))

    rng = MersenneTwister(seed)

    xs = collect(range(0, 1, length=N))
    ys = collect(range(0, 1, length=N))
    Nx, Ny = N, N

    # Fixed observation grid (same for all instances)
    obs_rng = MersenneTwister(0)  # deterministic obs locations
    obs_xs = [0.1 + 0.8 * rand(obs_rng) for _ in 1:n_obs]
    obs_ys = [0.1 + 0.8 * rand(obs_rng) for _ in 1:n_obs]

    nominal_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    # Storage
    all_z_c = Vector{Float64}[]
    all_z_s = Vector{Float64}[]
    all_d2_c = Float64[]
    all_d2_s = Float64[]
    all_s_cov = [Float64[] for _ in nominal_levels]
    all_c_cov = [Float64[] for _ in nominal_levels]
    all_s_width = Float64[]
    all_c_width = Float64[]
    all_s_iscore = Float64[]
    all_c_iscore = Float64[]
    spatial_s_cov = zeros(Nx, Ny)  # running sum for spatial coverage map
    spatial_c_cov = zeros(Nx, Ny)

    # Warmup (JIT)
    println("\nWarmup...")
    warmup_prob = random_two_source_problem(MersenneTwister(999);
        obs_xs=obs_xs, obs_ys=obs_ys, noise_std=noise_std)
    run_instance(warmup_prob, xs, ys, MersenneTwister(999);
        ρ=ρ, lengthscale_c=lengthscale_c, lengthscale_s=lengthscale_s,
        output_scale=output_scale, output_scale_c=output_scale_c,
        noise_std=noise_std, nominal_levels=nominal_levels)
    println("Warmup done.")

    # ---- Pass 1: calibrate output scale from a few instances ----
    n_cal = min(n_calibration, n_instances)
    println("\nCalibration pass ($n_cal instances with σ²=$output_scale)...")
    cal_rng = MersenneTwister(seed + 1_000_000)  # separate RNG so main run is reproducible
    cal_marg_z_s = Float64[]
    cal_marg_z_c = Float64[]

    for i in 1:n_cal
        cal_prob = random_two_source_problem(cal_rng;
            obs_xs=obs_xs, obs_ys=obs_ys, noise_std=noise_std)
        r = run_instance(cal_prob, xs, ys, cal_rng;
            ρ=ρ, lengthscale_c=lengthscale_c, lengthscale_s=lengthscale_s,
            output_scale=output_scale, output_scale_c=output_scale_c,
            noise_std=noise_std, nominal_levels=nominal_levels)
        append!(cal_marg_z_s, r.marginal_z_s)
        append!(cal_marg_z_c, r.marginal_z_c)
    end

    # Correct output scale: if marginal |z| has std = s, then σ²_new = σ²_old * s²
    # (marginal_z are |truth-mean|/std, so their std under correct calibration = √(2/π) ≈ 0.798
    # for a half-normal. More robust: use variance of signed z-scores.)
    # Actually, marginal_z_s are unsigned (|err|/std). For signed z ~ N(0,1), var = 1.
    # For |z| ~ half-normal, E[|z|²] = 1. So mean(z²) should = 1 if calibrated.
    # σ²_corrected = σ²_old * mean(z²)
    s_scale_factor = Statistics.mean(cal_marg_z_s.^2)
    c_scale_factor = Statistics.mean(cal_marg_z_c.^2)

    output_scale_calibrated = output_scale * s_scale_factor
    output_scale_c_calibrated = isnothing(output_scale_c) ?
        output_scale * c_scale_factor :
        output_scale_c * c_scale_factor

    @printf("  Source marginal z² mean: %.3f → σ²_s: %.2f → %.4f\n",
            s_scale_factor, output_scale, output_scale_calibrated)
    @printf("  Conc   marginal z² mean: %.3f → σ²_c: %.2f → %.4f\n",
            c_scale_factor,
            isnothing(output_scale_c) ? output_scale : output_scale_c,
            output_scale_c_calibrated)

    # Use calibrated values for the main run
    output_scale = output_scale_calibrated
    output_scale_c = output_scale_c_calibrated

    @printf("\nMain run with calibrated σ²_s=%.4f, σ²_c=%.4f\n", output_scale, output_scale_c)

    for i in 1:n_instances
        t = @elapsed begin
            prob = random_two_source_problem(rng;
                obs_xs=obs_xs, obs_ys=obs_ys, noise_std=noise_std)

            r = run_instance(prob, xs, ys, rng;
                ρ=ρ, lengthscale_c=lengthscale_c, lengthscale_s=lengthscale_s,
                output_scale=output_scale, output_scale_c=output_scale_c,
                noise_std=noise_std, nominal_levels=nominal_levels)
        end

        push!(all_z_c, r.z_c)
        push!(all_z_s, r.z_s)
        push!(all_d2_c, r.d2_c)
        push!(all_d2_s, r.d2_s)
        for (j, α) in enumerate(nominal_levels)
            push!(all_s_cov[j], r.s_coverage[α])
            push!(all_c_cov[j], r.c_coverage[α])
        end
        push!(all_s_width, r.s_width_95)
        push!(all_c_width, r.c_width_95)
        push!(all_s_iscore, r.s_iscore_95)
        push!(all_c_iscore, r.c_iscore_95)
        spatial_s_cov .+= r.s_covered_95
        spatial_c_cov .+= r.c_covered_95

        if i <= 5 || i % 20 == 0 || i == n_instances
            @printf("[%3d/%d] %.1fs | s_cov95=%.2f s_IS=%.2f | c_cov95=%.2f c_IS=%.2f | d²_s=%.0f d²_c=%.0f\n",
                    i, n_instances, t,
                    r.s_coverage[0.95], r.s_iscore_95,
                    r.c_coverage[0.95], r.c_iscore_95,
                    r.d2_s, r.d2_c)
        end
    end

    # Aggregate
    spatial_s_cov ./= n_instances
    spatial_c_cov ./= n_instances
    n_dof = Nx * Ny

    println("\n" * "=" ^ 60)
    println("RESULTS ($(n_instances) instances)")
    println("=" ^ 60)

    println("\nCoverage (nominal → empirical):")
    @printf("  %8s  %8s  %8s\n", "Nominal", "Source", "Conc")
    for (j, α) in enumerate(nominal_levels)
        @printf("  %8.0f%%  %7.1f%%  %7.1f%%\n",
                100α, 100*Statistics.mean(all_s_cov[j]), 100*Statistics.mean(all_c_cov[j]))
    end

    println("\nInterval width (95% CI, mean across grid):")
    @printf("  Source: %.4f\n", Statistics.mean(all_s_width))
    @printf("  Conc:   %.4f\n", Statistics.mean(all_c_width))

    println("\nInterval score (95%, lower is better):")
    @printf("  Source: %.4f\n", Statistics.mean(all_s_iscore))
    @printf("  Conc:   %.4f\n", Statistics.mean(all_c_iscore))

    println("\nWhitened residuals (pooled):")
    z_s_pooled = vcat(all_z_s...)
    z_c_pooled = vcat(all_z_c...)
    @printf("  Source: mean=%+.3f  std=%.3f  (n=%d)\n",
            Statistics.mean(z_s_pooled), Statistics.std(z_s_pooled), length(z_s_pooled))
    @printf("  Conc:   mean=%+.3f  std=%.3f  (n=%d)\n",
            Statistics.mean(z_c_pooled), Statistics.std(z_c_pooled), length(z_c_pooled))

    println("\nMahalanobis d² (expect ≈ $n_dof ± $(round(sqrt(2*n_dof), digits=1))):")
    @printf("  Source: mean=%.1f  std=%.1f\n", Statistics.mean(all_d2_s), Statistics.std(all_d2_s))
    @printf("  Conc:   mean=%.1f  std=%.1f\n", Statistics.mean(all_d2_c), Statistics.std(all_d2_c))

    # Normalized Mahalanobis: (d² - n) / sqrt(2n) should be ~ N(0,1)
    maha_s_norm = (all_d2_s .- n_dof) ./ sqrt(2*n_dof)
    maha_c_norm = (all_d2_c .- n_dof) ./ sqrt(2*n_dof)
    @printf("  Source (normalized): mean=%+.2f  std=%.2f\n",
            Statistics.mean(maha_s_norm), Statistics.std(maha_s_norm))
    @printf("  Conc (normalized):   mean=%+.2f  std=%.2f\n",
            Statistics.mean(maha_c_norm), Statistics.std(maha_c_norm))
    println("=" ^ 60)

    # Save results
    mkpath(output_dir)

    # CSV summary
    cov_df = DataFrame(
        nominal = nominal_levels,
        source_coverage = [Statistics.mean(all_s_cov[j]) for j in 1:length(nominal_levels)],
        conc_coverage = [Statistics.mean(all_c_cov[j]) for j in 1:length(nominal_levels)],
    )
    CSV.write(joinpath(output_dir, "coverage.csv"), cov_df)

    # NPZ with all z-scores and Mahalanobis distances
    npzwrite(joinpath(output_dir, "calibration_data.npz"), Dict(
        "z_s_pooled" => z_s_pooled,
        "z_c_pooled" => z_c_pooled,
        "d2_s" => all_d2_s,
        "d2_c" => all_d2_c,
        "spatial_s_cov_95" => spatial_s_cov,
        "spatial_c_cov_95" => spatial_c_cov,
        "xs" => xs, "ys" => ys,
        "nominal_levels" => nominal_levels,
        "s_coverage" => [Statistics.mean(all_s_cov[j]) for j in 1:length(nominal_levels)],
        "c_coverage" => [Statistics.mean(all_c_cov[j]) for j in 1:length(nominal_levels)],
        "n_instances" => Float64(n_instances),
    ))
    println("\nSaved: $(output_dir)/coverage.csv, calibration_data.npz")

    # Plots
    if make_plots
        println("Generating plots...")
        plot_calibration_results(output_dir, nominal_levels,
            all_s_cov, all_c_cov, z_s_pooled, z_c_pooled,
            spatial_s_cov, spatial_c_cov, xs, ys, n_dof,
            maha_s_norm, maha_c_norm)
    end

    return (; cov_df, z_s_pooled, z_c_pooled, all_d2_s, all_d2_c)
end

# ==============================================================================
# Plotting
# ==============================================================================

function plot_calibration_results(output_dir, nominal_levels,
        all_s_cov, all_c_cov, z_s_pooled, z_c_pooled,
        spatial_s_cov, spatial_c_cov, xs, ys, n_dof,
        maha_s_norm, maha_c_norm)

    norminv(p) = √2 * erfinv(2p - 1)

    # --- 1. Calibration plot ---
    fig1 = Figure(size=(450, 400), fontsize=12)
    ax = Axis(fig1[1, 1]; xlabel="Nominal coverage", ylabel="Empirical coverage",
              aspect=1)
    xlims!(ax, 0.45, 1.0); ylims!(ax, 0.45, 1.0)
    lines!(ax, [0.45, 1], [0.45, 1]; color=:gray60, linestyle=:dash, linewidth=1, label="Perfect")

    emp_s = [Statistics.mean(all_s_cov[j]) for j in 1:length(nominal_levels)]
    emp_c = [Statistics.mean(all_c_cov[j]) for j in 1:length(nominal_levels)]
    std_s = [Statistics.std(all_s_cov[j]) / sqrt(length(all_s_cov[j])) for j in 1:length(nominal_levels)]
    std_c = [Statistics.std(all_c_cov[j]) / sqrt(length(all_c_cov[j])) for j in 1:length(nominal_levels)]

    scatterlines!(ax, nominal_levels, emp_s; color=:crimson, linewidth=2, markersize=8, label="Source")
    errorbars!(ax, nominal_levels, emp_s, std_s; color=:crimson, whiskerwidth=6)
    scatterlines!(ax, nominal_levels, emp_c; color=:royalblue, linewidth=2, markersize=8, label="Concentration")
    errorbars!(ax, nominal_levels, emp_c, std_c; color=:royalblue, whiskerwidth=6)
    axislegend(ax; position=:lt)

    save(joinpath(output_dir, "calibration_plot.pdf"), fig1, px_per_unit=3)
    println("  Saved: calibration_plot.pdf")

    # --- 2. Spatial coverage map (95%) ---
    fig2 = Figure(size=(800, 350), fontsize=12)
    ax1 = Axis(fig2[1, 1]; aspect=DataAspect(), title="Source (95% coverage)", ylabel="y")
    hm1 = heatmap!(ax1, xs, ys, spatial_s_cov; colormap=:RdBu, colorrange=(0.8, 1.0))
    ax2 = Axis(fig2[1, 2]; aspect=DataAspect(), title="Concentration (95% coverage)",
               yticklabelsvisible=false)
    heatmap!(ax2, xs, ys, spatial_c_cov; colormap=:RdBu, colorrange=(0.8, 1.0))
    Colorbar(fig2[1, 3], hm1; label="Empirical coverage", width=15)
    save(joinpath(output_dir, "spatial_coverage_95.pdf"), fig2, px_per_unit=3)
    println("  Saved: spatial_coverage_95.pdf")

    # --- 3. Whitened residual QQ plots ---
    function qq_data(z_emp; n_max=2000)
        z_sorted = sort(z_emp)
        n = length(z_sorted)
        if n > n_max
            idx = round.(Int, range(1, n, length=n_max))
            z_sorted = z_sorted[idx]
            n = n_max
        end
        p = [(i - 0.5) / n for i in 1:n]
        return norminv.(p), z_sorted
    end

    fig3 = Figure(size=(800, 380), fontsize=12)
    for (col, z, title) in [(1, z_c_pooled, "Concentration"), (2, z_s_pooled, "Source")]
        ax = Axis(fig3[1, col]; xlabel="N(0,1) quantiles", ylabel="Empirical quantiles",
                  title="$title (σ̂=$(round(Statistics.std(z), digits=2)))", aspect=1)
        lines!(ax, [-4, 4], [-4, 4]; color=:gray60, linestyle=:dash, linewidth=1)
        z_th, z_emp = qq_data(z)
        scatter!(ax, z_th, z_emp; color=(:royalblue, 0.4), markersize=2)
        xlims!(ax, -4, 4); ylims!(ax, -4, 4)
    end
    save(joinpath(output_dir, "qq_whitened.pdf"), fig3, px_per_unit=3)
    println("  Saved: qq_whitened.pdf")

    # --- 4. Mahalanobis χ² check ---
    fig4 = Figure(size=(800, 380), fontsize=12)
    for (col, maha, title) in [(1, maha_c_norm, "Concentration"), (2, maha_s_norm, "Source")]
        ax = Axis(fig4[1, col]; xlabel="(d² - n) / √(2n)", ylabel="Count",
                  title="$title (μ=$(round(Statistics.mean(maha), digits=2)), σ=$(round(Statistics.std(maha), digits=2)))")
        hist!(ax, maha; bins=20, color=(:royalblue, 0.6))
        # Reference N(0,1)
        x_ref = range(-4, 4, length=100)
        n_inst = length(maha)
        bin_width = 8 / 20
        lines!(ax, x_ref, n_inst * bin_width * exp.(-x_ref.^2 / 2) / √(2π);
               color=:crimson, linewidth=2, label="N(0,1)")
        axislegend(ax)
    end
    save(joinpath(output_dir, "mahalanobis.pdf"), fig4, px_per_unit=3)
    println("  Saved: mahalanobis.pdf")
end

# ==============================================================================
# CLI
# ==============================================================================

function parse_calibration_args()
    s = ArgParseSettings(description = "Calibration experiment for GP-FVM")
    @add_arg_table! s begin
        "--n-instances"
            help = "Number of random instances"
            arg_type = Int
            default = 3
        "-N"
            help = "Grid size"
            arg_type = Int
            default = 31
        "--rho"
            arg_type = Float64
            default = 2.0
        "--lengthscale-c"
            arg_type = Float64
            default = 0.2
        "--lengthscale-s"
            arg_type = Float64
            default = 0.12
        "--output-scale"
            arg_type = Float64
            default = 0.34
        "--output-scale-c"
            help = "Separate c output scale (default: same as output-scale)"
            arg_type = Float64
            default = -1.0
        "--n-obs"
            arg_type = Int
            default = 10
        "--noise-std"
            arg_type = Float64
            default = 0.1
        "--seed"
            arg_type = Int
            default = 42
        "--output-dir", "-o"
            default = joinpath(@__DIR__, "results", "calibration")
        "--n-calibration"
            help = "Number of instances for output scale calibration pass"
            arg_type = Int
            default = 10
        "--no-plot"
            action = :store_true
    end
    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_calibration_args()
    osc = args["output-scale-c"] < 0 ? nothing : args["output-scale-c"]

    run_calibration_experiment(
        n_instances = args["n-instances"],
        N = args["N"],
        ρ = args["rho"],
        lengthscale_c = args["lengthscale-c"],
        lengthscale_s = args["lengthscale-s"],
        output_scale = args["output-scale"],
        n_calibration = args["n-calibration"],
        output_scale_c = osc,
        n_obs = args["n-obs"],
        noise_std = args["noise-std"],
        seed = args["seed"],
        output_dir = args["output-dir"],
        make_plots = !args["no-plot"],
    )
end
