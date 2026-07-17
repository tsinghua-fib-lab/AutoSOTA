# ------------------------------------------------------------------------------
# Direct Conditioning from Precision Matrix
# ------------------------------------------------------------------------------

using LinearAlgebra
using SparseArrays
using GaussianMarkovRandomFields: GMRF, InformationVector

export condition_precision

"""
    condition_precision(Q::AbstractMatrix; A, Q_ϵ, y, b=zeros(size(A,1)), prior_mean=nothing)

Condition a prior precision matrix on linear observations, returning a posterior GMRF.

This provides a cleaner API than constructing a prior GMRF and then calling `linear_condition`,
especially when you have multiple conditioning steps or a zero prior mean.

# Arguments
- `Q`: Prior precision matrix (sparse or dense)
- `A`: Observation matrix (y = A*x + b + ε)
- `Q_ϵ`: Precision of observation noise
- `y`: Observation values
- `b`: Offset vector (default: zeros)
- `prior_mean`: Prior mean (default: zeros, which is typical for GP priors)

# Returns
- `GMRF`: Posterior GMRF with updated precision and information vector

# Model
The observation model is: y = A*x + b + ε where ε ~ N(0, Q_ϵ⁻¹)

The posterior has:
- Precision: Q_post = Q_prior + A'*Q_ϵ*A
- Information: η_post = Q_prior*μ_prior + A'*Q_ϵ*(y - b)

For zero prior mean (the default), this simplifies to:
- η_post = A'*Q_ϵ*(y - b)

# Example
```julia
# Build sparse precision
approx = sparse_precision([...], kernel; ρ=2.0)

# Condition directly on linear constraints
x_posterior = condition_precision(approx.Q;
    A = A_constraints,
    Q_ϵ = (1/σ²) * I,
    y = constraint_values
)

# Extract posterior statistics
μ = mean(x_posterior)
σ = std(x_posterior)
```
"""
function condition_precision(Q::AbstractMatrix;
                              A::AbstractMatrix,
                              Q_ϵ::Union{AbstractMatrix, UniformScaling},
                              y::AbstractVector,
                              b::AbstractVector = zeros(size(A, 1)),
                              prior_mean::Union{AbstractVector, Nothing} = nothing)
    n = size(Q, 1)

    # Compute posterior precision
    obs_precision_contrib = A' * Q_ϵ * A
    Q_posterior = Q + obs_precision_contrib

    # Compute posterior information vector
    # η_post = Q_prior * μ_prior + A' * Q_ϵ * (y - b)
    if prior_mean === nothing || iszero(prior_mean)
        # Zero prior mean (common case) - skip the Q*μ term
        info_posterior = A' * (Q_ϵ * (y - b))
    else
        info_posterior = Q * prior_mean + A' * (Q_ϵ * (y - b))
    end

    return GMRF(InformationVector(info_posterior), Symmetric(Q_posterior))
end

"""
    condition_precision(Q::AbstractMatrix, conditions::Vector; prior_mean=nothing)

Apply multiple conditioning steps efficiently.

# Arguments
- `Q`: Prior precision matrix
- `conditions`: Vector of NamedTuples, each with fields (A, Q_ϵ, y) and optional (b)
- `prior_mean`: Prior mean (default: zeros)

# Example
```julia
x_posterior = condition_precision(approx.Q, [
    (A=A_fvm, Q_ϵ=Q_fvm, y=zeros(n_fvm)),
    (A=A_bc, Q_ϵ=Q_bc, y=bc_values),
])
```
"""
function condition_precision(Q::AbstractMatrix, conditions::Vector;
                              prior_mean::Union{AbstractVector, Nothing} = nothing)
    n = size(Q, 1)

    # Accumulate precision contributions
    Q_posterior = copy(Q)
    info_posterior = prior_mean === nothing ? zeros(n) : Q * prior_mean

    for cond in conditions
        A = cond.A
        Q_ϵ = cond.Q_ϵ
        y = cond.y
        b = hasproperty(cond, :b) ? cond.b : zeros(size(A, 1))

        Q_posterior += A' * Q_ϵ * A
        info_posterior += A' * (Q_ϵ * (y - b))
    end

    return GMRF(InformationVector(info_posterior), Symmetric(Q_posterior))
end
