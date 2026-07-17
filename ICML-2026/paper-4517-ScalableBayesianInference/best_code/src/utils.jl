# ------------------------------------------------------------------------------
# Basic Utilities
# ------------------------------------------------------------------------------

using FunctionalGPs: Interval
using LinearAlgebra
using SparseArrays
using GaussianMarkovRandomFields: GMRF, ConstrainedGMRF, precision_matrix

export midpoint

"""
    midpoint(interv::Interval)

Compute the midpoint of an interval.
"""
midpoint(interv::Interval) = 0.5 * (interv[1] + interv[2])

# ------------------------------------------------------------------------------
# Dense GMRF Utilities
# ------------------------------------------------------------------------------

export dense_std, dense_var, dense_cov, is_dense_gmrf

"""
    is_dense_gmrf(gmrf::GMRF)

Check if a GMRF has a dense (non-sparse) precision matrix.
"""
is_dense_gmrf(gmrf::GMRF) = !(precision_matrix(gmrf) isa SparseMatrixCSC)

"""
    dense_var(gmrf::GMRF)
    dense_var(gmrf::ConstrainedGMRF)

Compute variance of a GMRF by computing the diagonal of Q⁻¹ via Cholesky.

For a Cholesky factorization Q = LLᵀ, we have Q⁻¹ = L⁻ᵀL⁻¹.
The diagonal elements are: Σᵢᵢ = ||L⁻ᵀeᵢ||² = Σⱼ (L⁻¹)ᵢⱼ²

This is O(n³) for the factorization plus O(n²) for extracting the diagonal.
"""
function dense_var(gmrf::GMRF)
    Q = precision_matrix(gmrf)
    F = cholesky(Symmetric(Matrix(Q)))
    Linv = inv(F.L)
    # Variance is sum of squared rows of L⁻¹
    return vec(sum(Linv .^ 2, dims=1))
end

# For ConstrainedGMRF, compute var using the formula:
# var_c = var_base - diag(B*B^T) where B = Ã^T * L_c^(-T)
# The ConstrainedGMRF already has A_tilde_T and L_c precomputed
function dense_var(gmrf::ConstrainedGMRF)
    # Get unconstrained variances using dense computation
    σ_base = dense_var(gmrf.base_gmrf)

    # Compute B^T = L_c^(-1) * A_tilde_T^T
    B_T = gmrf.L_c.L \ gmrf.A_tilde_T'

    # Compute diagonal of B*B^T as column-wise sum of squares of B^T
    B_squared_rowsums = vec(sum(abs2, B_T, dims=1))

    # Constrained variance = unconstrained variance - correction
    σ_constrained = σ_base - B_squared_rowsums

    # Ensure non-negative (numerical precision can cause tiny negative values)
    σ_constrained .= max.(σ_constrained, zero(eltype(σ_constrained)))

    return σ_constrained
end

"""
    dense_std(gmrf::GMRF)
    dense_std(gmrf::ConstrainedGMRF)

Compute standard deviation of a GMRF. See [`dense_var`](@ref) for details.
"""
dense_std(gmrf::GMRF) = sqrt.(dense_var(gmrf))
dense_std(gmrf::ConstrainedGMRF) = sqrt.(dense_var(gmrf))

"""
    dense_cov(gmrf::GMRF)

Compute full covariance matrix of a GMRF via Cholesky inversion.

Warning: This is O(n³) in time and O(n²) in memory.
Only use for small-to-moderate sized problems.
"""
function dense_cov(gmrf::GMRF)
    Q = precision_matrix(gmrf)
    F = cholesky(Symmetric(Matrix(Q)))
    return Matrix(inv(F))
end
