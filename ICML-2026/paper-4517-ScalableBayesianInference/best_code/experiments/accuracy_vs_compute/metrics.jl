"""
    Metrics for Accuracy vs Compute Experiment

Defines accuracy metrics (L2 error, coverage, CRPS) and computational metrics
(time, memory, sparsity) for comparing methods.
"""

using Statistics
using LinearAlgebra

# ==============================================================================
# Solution Result Interface
# ==============================================================================

"""
    SolutionResult

Standardized result format from any solver method.
All methods must return results in this format for fair comparison.
"""
struct SolutionResult
    # Solution at evaluation points
    xs::Vector{Float64}           # Spatial grid points
    ts::Vector{Float64}           # Time points where solution is available

    # Posterior statistics at each (x, t)
    mean::Matrix{Float64}         # [n_x, n_t] posterior mean
    std::Matrix{Float64}          # [n_x, n_t] posterior std

    # Computational metrics
    wall_time_s::Float64          # Total solve time in seconds
    peak_memory_mb::Float64       # Peak memory usage in MB
    cholesky_nnz::Int             # Non-zeros in Cholesky factor (0 for non-GP methods)
    n_dof::Int                    # Total degrees of freedom

    # Method metadata
    method_name::String
    N::Int                        # Grid size parameter
end

"""
    fillin_percent(result::SolutionResult)

Compute fill-in percentage relative to dense Cholesky.
"""
function fillin_percent(result::SolutionResult)
    if result.cholesky_nnz == 0
        return 0.0
    end
    # Dense lower triangular has n(n+1)/2 entries
    n = result.n_dof
    dense_nnz = n * (n + 1) ÷ 2
    return 100.0 * result.cholesky_nnz / dense_nnz
end

# ==============================================================================
# Accuracy Metrics
# ==============================================================================

"""
    l2_error(result::SolutionResult, reference::HighResFVMReference, t::Float64)

Compute relative L2 error at time t: ||u_mean - u_ref||₂ / ||u_ref||₂
"""
function l2_error(result::SolutionResult, reference, t::Float64)
    # Find closest time index in result
    t_idx = argmin(abs.(result.ts .- t))

    u_mean = result.mean[:, t_idx]
    u_ref = evaluate_reference(reference, result.xs, t)

    num = norm(u_mean - u_ref)
    den = norm(u_ref)

    return den > 1e-12 ? num / den : num
end

"""
    l2_error_trajectory(result::SolutionResult, reference)

Compute L2 error at all available time points.
Returns (ts, errors) tuple.
"""
function l2_error_trajectory(result::SolutionResult, reference)
    errors = [l2_error(result, reference, t) for t in result.ts]
    return result.ts, errors
end

"""
    mean_l2_error(result::SolutionResult, reference)

Compute time-averaged L2 error.
"""
function mean_l2_error(result::SolutionResult, reference)
    _, errors = l2_error_trajectory(result, reference)
    return mean(errors)
end

"""
    final_l2_error(result::SolutionResult, reference)

Compute L2 error at final time.
"""
function final_l2_error(result::SolutionResult, reference)
    return l2_error(result, reference, result.ts[end])
end

# ==============================================================================
# UQ Calibration Metrics
# ==============================================================================

"""
    pointwise_coverage(result::SolutionResult, reference, t::Float64, α::Float64=0.95)

Compute empirical coverage at time t: fraction of points where true solution
falls within the α credible interval.
"""
function pointwise_coverage(result::SolutionResult, reference, t::Float64, α::Float64=0.95)
    t_idx = argmin(abs.(result.ts .- t))

    u_mean = result.mean[:, t_idx]
    u_std = result.std[:, t_idx]
    u_ref = evaluate_reference(reference, result.xs, t)

    # z-score for α credible interval (two-sided)
    z = quantile_normal((1 + α) / 2)

    lower = u_mean .- z .* u_std
    upper = u_mean .+ z .* u_std

    in_interval = (u_ref .>= lower) .& (u_ref .<= upper)
    return mean(in_interval)
end

"""
    quantile_normal(p)

Compute the p-quantile of the standard normal distribution.
Approximation using rational function (Abramowitz & Stegun 26.2.23).
"""
function quantile_normal(p)
    # Handle edge cases
    p <= 0 && return -Inf
    p >= 1 && return Inf
    p == 0.5 && return 0.0

    # Use symmetry: for p > 0.5, quantile is positive
    if p > 0.5
        return -quantile_normal(1 - p)
    end

    t = sqrt(-2 * log(p))

    # Rational approximation coefficients (Abramowitz & Stegun)
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    # This gives negative quantile for p < 0.5
    return -(t - (c0 + c1*t + c2*t^2) / (1 + d1*t + d2*t^2 + d3*t^3))
end

"""
    coverage_trajectory(result::SolutionResult, reference, α::Float64=0.95)

Compute empirical coverage at all time points.
"""
function coverage_trajectory(result::SolutionResult, reference, α::Float64=0.95)
    coverages = [pointwise_coverage(result, reference, t, α) for t in result.ts]
    return result.ts, coverages
end

"""
    mean_coverage(result::SolutionResult, reference, α::Float64=0.95)

Compute time-averaged empirical coverage.
"""
function mean_coverage(result::SolutionResult, reference, α::Float64=0.95)
    _, coverages = coverage_trajectory(result, reference, α)
    return mean(coverages)
end

"""
    calibration_curve(result::SolutionResult, reference;
                       α_levels=[0.5, 0.8, 0.9, 0.95, 0.99])

Compute calibration curve: nominal coverage vs empirical coverage.
Returns (nominal, empirical) tuple.
"""
function calibration_curve(result::SolutionResult, reference;
                            α_levels=[0.5, 0.8, 0.9, 0.95, 0.99])
    empirical = [mean_coverage(result, reference, α) for α in α_levels]
    return α_levels, empirical
end

# ==============================================================================
# CRPS (Continuous Ranked Probability Score)
# ==============================================================================

"""
    crps_gaussian(μ, σ, y)

Compute CRPS for a Gaussian predictive distribution N(μ, σ²) and observation y.

CRPS = σ * [z*(2Φ(z) - 1) + 2φ(z) - 1/√π]
where z = (y - μ)/σ, Φ is standard normal CDF, φ is standard normal PDF.
"""
function crps_gaussian(μ::Float64, σ::Float64, y::Float64)
    if σ < 1e-12
        # Degenerate case: point prediction
        return abs(y - μ)
    end

    z = (y - μ) / σ

    # Standard normal PDF and CDF
    φ_z = exp(-0.5 * z^2) / sqrt(2π)
    Φ_z = 0.5 * (1 + erf(z / sqrt(2)))

    return σ * (z * (2*Φ_z - 1) + 2*φ_z - 1/sqrt(π))
end

"""
    mean_crps(result::SolutionResult, reference, t::Float64)

Compute mean CRPS at time t (averaged over spatial points).
"""
function mean_crps(result::SolutionResult, reference, t::Float64)
    t_idx = argmin(abs.(result.ts .- t))

    u_mean = result.mean[:, t_idx]
    u_std = result.std[:, t_idx]
    u_ref = evaluate_reference(reference, result.xs, t)

    crps_vals = [crps_gaussian(u_mean[i], u_std[i], u_ref[i]) for i in eachindex(u_mean)]
    return mean(crps_vals)
end

"""
    crps_trajectory(result::SolutionResult, reference)

Compute mean CRPS at all time points.
"""
function crps_trajectory(result::SolutionResult, reference)
    crps_vals = [mean_crps(result, reference, t) for t in result.ts]
    return result.ts, crps_vals
end

"""
    mean_crps_overall(result::SolutionResult, reference)

Compute overall mean CRPS (averaged over time and space).
"""
function mean_crps_overall(result::SolutionResult, reference)
    _, crps_vals = crps_trajectory(result, reference)
    return mean(crps_vals)
end

# ==============================================================================
# Error-Uncertainty Correlation
# ==============================================================================

"""
    error_std_correlation(result::SolutionResult, reference, t::Float64)

Compute correlation between absolute error and posterior std at time t.
Positive correlation indicates uncertainty tracks actual error.
"""
function error_std_correlation(result::SolutionResult, reference, t::Float64)
    t_idx = argmin(abs.(result.ts .- t))

    u_mean = result.mean[:, t_idx]
    u_std = result.std[:, t_idx]
    u_ref = evaluate_reference(reference, result.xs, t)

    abs_error = abs.(u_mean - u_ref)

    # Pearson correlation
    return cor(abs_error, u_std)
end

"""
    mean_error_std_correlation(result::SolutionResult, reference)

Compute time-averaged error-std correlation.
"""
function mean_error_std_correlation(result::SolutionResult, reference)
    corrs = [error_std_correlation(result, reference, t) for t in result.ts]
    # Filter out NaN values (can occur if std is constant)
    valid_corrs = filter(!isnan, corrs)
    return isempty(valid_corrs) ? NaN : mean(valid_corrs)
end

# ==============================================================================
# Combined Metrics Summary
# ==============================================================================

"""
    MetricsSummary

All metrics for a single (method, problem instance) evaluation.
"""
struct MetricsSummary
    # Identifiers
    method_name::String
    ic_family::Symbol
    ic_seed::Int
    N::Int

    # Computational
    wall_time_s::Float64
    peak_memory_mb::Float64
    cholesky_nnz::Int
    fillin_pct::Float64

    # Accuracy
    mean_l2_error::Float64
    final_l2_error::Float64

    # UQ calibration
    coverage_95::Float64
    mean_crps::Float64
    error_std_corr::Float64
end

"""
    compute_metrics(result::SolutionResult, instance::ProblemInstance)

Compute all metrics for a solution result.
"""
function compute_metrics(result::SolutionResult, ic_family::Symbol, ic_seed::Int,
                          reference)
    return MetricsSummary(
        result.method_name,
        ic_family,
        ic_seed,
        result.N,

        result.wall_time_s,
        result.peak_memory_mb,
        result.cholesky_nnz,
        fillin_percent(result),

        mean_l2_error(result, reference),
        final_l2_error(result, reference),

        mean_coverage(result, reference, 0.95),
        mean_crps_overall(result, reference),
        mean_error_std_correlation(result, reference)
    )
end

"""
    metrics_to_namedtuple(m::MetricsSummary)

Convert MetricsSummary to NamedTuple for easy CSV export.
"""
function metrics_to_namedtuple(m::MetricsSummary)
    return (
        method = m.method_name,
        ic_family = m.ic_family,
        ic_seed = m.ic_seed,
        N = m.N,
        time_s = m.wall_time_s,
        memory_mb = m.peak_memory_mb,
        cholesky_nnz = m.cholesky_nnz,
        fillin_pct = m.fillin_pct,
        mean_l2_error = m.mean_l2_error,
        final_l2_error = m.final_l2_error,
        coverage_95 = m.coverage_95,
        mean_crps = m.mean_crps,
        error_std_corr = m.error_std_corr
    )
end
