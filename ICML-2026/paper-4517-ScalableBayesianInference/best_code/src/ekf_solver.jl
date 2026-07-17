# ------------------------------------------------------------------------------
# EKF-Style Sequential Solver
# ------------------------------------------------------------------------------
#
# Memory-efficient temporal solver using a sliding 2-block window with
# moment-matching to propagate uncertainty between time steps.
#
# Instead of building a full (T × N) spacetime GMRF, we process time steps
# sequentially, maintaining only mean and marginal variances at each step.
# ------------------------------------------------------------------------------

using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: mean, var, precision_matrix, GMRF
using LinearAlgebra
using SparseArrays

export MomentMatchedState, EKFSolution, TransitionCache
export build_2block_precision, build_2block_gmrf, solve_ekf_sequential
export mean_trajectory, var_trajectory, std_trajectory

# ------------------------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------------------------

"""
    MomentMatchedState{T,V<:AbstractVector{T}}

Gaussian approximation via matched moments (mean and marginal variances).

This is the minimal representation needed to propagate uncertainty between
time steps while preserving sparsity in the precision matrix.
"""
struct MomentMatchedState{T,V<:AbstractVector{T}}
    μ::V       # Mean
    σ²::V      # Marginal variances
end

MomentMatchedState(μ::AbstractVector{T}, σ²::AbstractVector{T}) where T =
    MomentMatchedState{T,Vector{T}}(collect(μ), collect(σ²))

Base.length(s::MomentMatchedState) = length(s.μ)

"""
    EKFSolution

Container for EKF-style sequential solution.
"""
struct EKFSolution{S,L}
    history::Vector{S}
    n_timesteps::Int
    layout::L
end

# Accessors
mean_trajectory(sol::EKFSolution) = [s.μ for s in sol.history]
var_trajectory(sol::EKFSolution) = [s.σ² for s in sol.history]
std_trajectory(sol::EKFSolution) = [sqrt.(s.σ²) for s in sol.history]

function Base.getindex(sol::EKFSolution, sym::Symbol, t::Int)
    idx = indices(sol.layout, sym)
    return sol.history[t].μ[idx]
end

# ------------------------------------------------------------------------------
# Precomputed Transition Matrices
# ------------------------------------------------------------------------------

"""
    TransitionCache

Precomputed matrices for EKF time stepping that are constant across all timesteps.
Computing these once before the time loop significantly improves performance.
"""
struct TransitionCache{T<:AbstractMatrix}
    A::T           # State transition matrix
    Q_ε::T         # Temporal noise precision
    Q_ε_A::T       # Q_ε * A
    At_Q_ε_A::T    # A' * Q_ε * A
    At_Q_ε::T      # A' * Q_ε
end

"""
    TransitionCache(A, Q_ε)

Precompute transition-related matrix products for EKF solver.
"""
function TransitionCache(A::AbstractMatrix, Q_ε::AbstractMatrix)
    Q_ε_A = Q_ε * A
    At_Q_ε_A = sparse(A' * Q_ε_A)
    At_Q_ε = sparse(A' * Q_ε)
    return TransitionCache(A, Q_ε, sparse(Q_ε_A), At_Q_ε_A, At_Q_ε)
end

# ------------------------------------------------------------------------------
# 2-Block Precision Construction
# ------------------------------------------------------------------------------

"""
    build_2block_precision(σ²_prev, cache::TransitionCache, Q_spatial) -> SparseMatrixCSC

Build the 2-block joint precision for [x_{t-1}; x_t] using precomputed transition matrices.

The structure is:
```
Q_2block = [ Q_spatial + diag(1/σ²) + A'Q_ε A    -A'Q_ε  ]
           [ -Q_ε A                               Q_ε     ]
```
"""
function build_2block_precision(
    σ²_prev::AbstractVector,
    cache::TransitionCache,
    Q_spatial::AbstractMatrix
)
    # Moment-matched precision: diag(1/σ²)
    D_mm = spdiagm(1 ./ σ²_prev)

    # Build blocks using precomputed products
    Q11 = Q_spatial + D_mm + cache.At_Q_ε_A
    Q12 = -cache.At_Q_ε
    Q21 = -cache.Q_ε_A
    Q22 = cache.Q_ε

    # Assemble 2x2 block matrix
    return [Q11 Q12; Q21 Q22]
end

"""
    build_2block_gmrf(state::MomentMatchedState, cache::TransitionCache, Q_spatial) -> GMRF

Build the 2-block GMRF prior for [x_{t-1}; x_t] using precomputed transition matrices.

The mean vector is:
- x_{t-1} block: μ_prev (mean from previous step)
- x_t block: A * μ_prev (predicted mean from transition)
"""
function build_2block_gmrf(
    state::MomentMatchedState,
    cache::TransitionCache,
    Q_spatial::AbstractMatrix
)
    Q_2block = build_2block_precision(state.σ², cache, Q_spatial)

    # Mean vector: [μ_prev; A * μ_prev]
    μ_2block = [state.μ; cache.A * state.μ]

    return GMRF(μ_2block, Symmetric(Q_2block))
end

# ------------------------------------------------------------------------------
# Sequential Solver
# ------------------------------------------------------------------------------

"""
    solve_ekf_sequential(
        Q_spatial, state_layout, A, Q_ε, n_timesteps, x0,
        build_fvm_constraint;
        apply_bc!, store_history=true, verbose=false
    ) -> EKFSolution

Solve GP-FVM problem using EKF-style sequential updates.

This processes time steps one at a time using a sliding 2-block window,
maintaining a moment-matched Gaussian approximation at each step.

Memory is O(N²) per step vs O(T×N²) for full spacetime approach.

# Arguments
- `Q_spatial`: Sparse spatial prior precision
- `state_layout`: Layout for named indexing into state vector
- `A`: State transition matrix
- `Q_ε`: Temporal noise precision
- `n_timesteps`: Total number of time steps
- `x0`: Initial state GMRF (after IC conditioning)
- `build_fvm_constraint`: Function `(μ_prev, t) -> likelihood` that builds FVM constraint

# Keyword Arguments
- `apply_bc!`: Function `(gmrf, t) -> gmrf` to apply boundary conditions (optional)
- `store_history`: Whether to store all timestep states (default: true)
- `verbose`: Print progress (default: false)
- `max_iter`: Max Gauss-Newton iterations per step (default: 1 for true EKF)
- `σ²_floor`: Variance floor for numerical stability (default: 1e-12)

# Returns
- `EKFSolution` containing mean/variance trajectories
"""
function solve_ekf_sequential(
    Q_spatial::AbstractMatrix,
    state_layout,
    A::AbstractMatrix,
    Q_ε::AbstractMatrix,
    n_timesteps::Int,
    x0::GMRF,
    build_fvm_constraint::Function;
    apply_bc!::Union{Function,Nothing} = nothing,
    store_history::Bool = true,
    verbose::Bool = false,
    max_iter::Int = 1,  # Single GN step = true EKF
    σ²_floor::Float64 = 1e-12  # Variance floor for numerical stability
)
    N = size(Q_spatial, 1)

    # Initialize from t=0 solution
    μ_0 = mean(x0)
    σ²_0 = var(x0)

    state = MomentMatchedState(μ_0, σ²_0)
    history = store_history ? [state] : MomentMatchedState{eltype(μ_0)}[]

    # Precompute transition matrices once (major optimization)
    verbose && print("Precomputing transition matrices... ")
    t1 = time()
    cache = TransitionCache(A, Q_ε)
    verbose && println("$(round(time()-t1, digits=2))s")

    verbose && println("EKF Sequential Solver: $n_timesteps timesteps, N=$N")

    for t in 2:n_timesteps
        verbose && println("  t=$t/$n_timesteps:")

        # 1. Build 2-block GMRF from moment-matched prior (using precomputed cache)
        verbose && print("    build_2block_gmrf... ")
        t1 = time()
        x_2block = build_2block_gmrf(state, cache, Q_spatial)
        verbose && println("$(round(time()-t1, digits=2))s")

        # 2. Apply boundary conditions if provided
        if apply_bc! !== nothing
            verbose && print("    apply_bc!... ")
            t1 = time()
            x_2block = apply_bc!(x_2block, t)
            verbose && println("$(round(time()-t1, digits=2))s")
        end

        # 3. Apply FVM constraint
        verbose && print("    build_fvm_constraint... ")
        t1 = time()
        fvm_lik = build_fvm_constraint(state.μ, t)
        verbose && println("$(round(time()-t1, digits=2))s")

        verbose && print("    gaussian_approximation... ")
        t1 = time()
        x_post = gaussian_approximation(x_2block, fvm_lik; verbose=false, max_iter=max_iter)
        verbose && println("$(round(time()-t1, digits=2))s")

        # 4. Extract marginals for x_t block (second half of state)
        verbose && print("    extract mean... ")
        t1 = time()
        μ_full = mean(x_post)
        verbose && println("$(round(time()-t1, digits=2))s")

        verbose && print("    extract var... ")
        t1 = time()
        σ²_full = var(x_post)
        verbose && println("$(round(time()-t1, digits=2))s")

        μ_curr = μ_full[N+1:2N]
        σ²_curr = σ²_full[N+1:2N]

        # Apply variance floor for numerical stability
        σ²_curr = max.(σ²_curr, σ²_floor)

        # 5. Update state
        state = MomentMatchedState(copy(μ_curr), copy(σ²_curr))
        store_history && push!(history, state)

        if verbose
            max_μ = maximum(abs.(μ_curr))
            mean_σ = sqrt(sum(σ²_curr) / length(σ²_curr))
            println("max|μ|=$(round(max_μ, digits=4)), mean_σ=$(round(mean_σ, digits=4))")
        end
    end

    return EKFSolution(history, n_timesteps, state_layout)
end
