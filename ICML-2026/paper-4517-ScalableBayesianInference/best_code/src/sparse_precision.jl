# ------------------------------------------------------------------------------
# Sparse Precision Matrix Construction via KL-Optimal Cholesky
# ------------------------------------------------------------------------------

using FunctionalGPs
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Distributions: Normal
using Random: randperm

using GaussianMarkovRandomFields:
    reverse_maximin_ordering,
    sparsity_pattern_from_ordering,
    sparse_approximate_cholesky!,
    sparse_approximate_cholesky,
    form_supernodes,
    PermutedMatrix,
    GMRF,
    InformationVector,
    ExponentialFamily,
    gaussian_approximation

using Kronecker: KroneckerProduct

export SparseGMRFApproximation
export sparse_precision, sparse_gmrf
export prescribe_indices
export condition_precision
export calibrate_output_scale

# ------------------------------------------------------------------------------
# Fast Block-Kronecker Matrix for Efficient Submatrix Extraction
# ------------------------------------------------------------------------------

"""
    FastBlockMatrix{T}

Optimized wrapper for BlockMatrix of KroneckerProducts that provides fast submatrix
extraction by materializing the small Kronecker factors.

For 2D tensor-product problems, this provides ~40-50x speedup in sparse Cholesky
while using ~100x less memory than full materialization.

Non-Kronecker blocks (e.g., in 1D problems) fall back to lazy evaluation.
"""
struct FastBlockMatrix{T} <: AbstractMatrix{T}
    # Materialized Kronecker factors (typed for fast access)
    A_factors::Matrix{Matrix{T}}
    B_factors::Matrix{Matrix{T}}
    B_dims::Matrix{Tuple{Int,Int}}
    # Original blocks for non-Kronecker fallback
    orig_blocks::Matrix
    is_kronecker::Matrix{Bool}
    block_offsets::Vector{Int}
    n::Int
end

Base.size(F::FastBlockMatrix) = (F.n, F.n)

"""
    FastBlockMatrix(K)

Create a FastBlockMatrix from a BlockMatrix. Automatically detects KroneckerProduct
blocks and materializes their factors for fast access.
"""
function FastBlockMatrix(K)
    orig_blocks = K.blocks
    n_blocks_r, n_blocks_c = size(orig_blocks)
    T = Float64

    A_factors = Matrix{Matrix{T}}(undef, n_blocks_r, n_blocks_c)
    B_factors = Matrix{Matrix{T}}(undef, n_blocks_r, n_blocks_c)
    B_dims = Matrix{Tuple{Int,Int}}(undef, n_blocks_r, n_blocks_c)
    is_kronecker = Matrix{Bool}(undef, n_blocks_r, n_blocks_c)

    for i in 1:n_blocks_r, j in 1:n_blocks_c
        block = orig_blocks[i, j]
        if block isa KroneckerProduct
            # Materialize Kronecker factors for fast access
            A_factors[i,j] = Matrix(block.A)
            B_factors[i,j] = Matrix(block.B)
            B_dims[i,j] = size(block.B)
            is_kronecker[i,j] = true
        else
            # Placeholder for type stability; use orig_blocks for actual access
            A_factors[i,j] = Matrix{T}(undef, 0, 0)
            B_factors[i,j] = Matrix{T}(undef, 0, 0)
            B_dims[i,j] = (0, 0)
            is_kronecker[i,j] = false
        end
    end

    block_sizes = [size(orig_blocks[i,1], 1) for i in 1:n_blocks_r]
    block_offsets = [0; cumsum(block_sizes)]
    n = block_offsets[end]

    return FastBlockMatrix{T}(A_factors, B_factors, B_dims, orig_blocks,
                               is_kronecker, block_offsets, n)
end

function Base.getindex(F::FastBlockMatrix{T}, I::AbstractVector{Int}, J::AbstractVector{Int}) where T
    n_I, n_J = length(I), length(J)
    result = Matrix{T}(undef, n_I, n_J)

    # Decode indices to block and local indices
    bi_I = Vector{Int}(undef, n_I)
    li_I = Vector{Int}(undef, n_I)
    @inbounds for k in 1:n_I
        bi_I[k] = max(1, searchsortedlast(F.block_offsets, I[k]-1))
        li_I[k] = I[k] - F.block_offsets[bi_I[k]]
    end

    bj_J = Vector{Int}(undef, n_J)
    lj_J = Vector{Int}(undef, n_J)
    @inbounds for k in 1:n_J
        bj_J[k] = max(1, searchsortedlast(F.block_offsets, J[k]-1))
        lj_J[k] = J[k] - F.block_offsets[bj_J[k]]
    end

    @inbounds for ki in 1:n_I
        bi = bi_I[ki]
        li = li_I[ki]
        for kj in 1:n_J
            bj = bj_J[kj]
            lj = lj_J[kj]

            if F.is_kronecker[bi, bj]
                # Fast path: use materialized Kronecker factors
                A_mat = F.A_factors[bi, bj]
                B_mat = F.B_factors[bi, bj]
                n_B_i, n_B_j = F.B_dims[bi, bj]
                li_A, li_B = divrem(li - 1, n_B_i)
                lj_A, lj_B = divrem(lj - 1, n_B_j)
                result[ki, kj] = A_mat[li_A+1, lj_A+1] * B_mat[li_B+1, lj_B+1]
            else
                # Fallback: use original lazy block
                result[ki, kj] = F.orig_blocks[bi, bj][li, lj]
            end
        end
    end
    return result
end

Base.getindex(F::FastBlockMatrix, i::Int, j::Int) = F[[i], [j]][1, 1]

"""
    FastPermutedMatrix{T}

PermutedMatrix wrapper for FastBlockMatrix that returns Symmetric views
for square submatrix extractions (required by sparse Cholesky).
"""
struct FastPermutedMatrix{T} <: AbstractMatrix{T}
    F::FastBlockMatrix{T}
    P::Vector{Int}
end

Base.size(M::FastPermutedMatrix) = size(M.F)

function Base.getindex(M::FastPermutedMatrix{T}, I::AbstractVector{Int}, J::AbstractVector{Int}) where T
    raw = M.F[M.P[I], M.P[J]]
    # Return Symmetric for square extractions (sparse Cholesky requirement)
    return length(I) == length(J) ? Symmetric(raw) : raw
end

Base.getindex(M::FastPermutedMatrix, i::Int, j::Int) = M.F[M.P[i], M.P[j]]

"""
    is_block_matrix(K) -> Bool

Check if K is a BlockMatrix (has .blocks field).
"""
is_block_matrix(K) = hasproperty(K, :blocks)

# ------------------------------------------------------------------------------
# Ordering Strategies
# ------------------------------------------------------------------------------

"""
    OrderingStrategy

Strategy for ordering functional blocks in sparse Cholesky.

- `:integrals_coarsest`: Integrals rightmost (coarsest), then evaluations, then derivatives (finest)
- `:evaluations_coarsest`: Evaluations rightmost (coarsest)
- `:derivatives_coarsest`: Derivatives rightmost (coarsest)
- `:natural`: Use the order provided by the user
"""
const OrderingStrategy = Symbol

# Priority maps: lower number = finer scale = leftmost in Cholesky
# Note: OTHER always has the same priority as DERIVATIVE (safest default)
const ORDERING_PRIORITIES = Dict(
    :integrals_coarsest => Dict(
        DERIVATIVE => 1,
        OTHER => 1,
        EVALUATION => 2,
        FACE_INTEGRAL => 3,
        INTEGRAL => 4,
    ),
    :evaluations_coarsest => Dict(
        DERIVATIVE => 1,
        OTHER => 1,
        FACE_INTEGRAL => 2,
        INTEGRAL => 3,
        EVALUATION => 4,
    ),
    :derivatives_coarsest => Dict(
        INTEGRAL => 1,
        FACE_INTEGRAL => 2,
        EVALUATION => 3,
        DERIVATIVE => 4,
        OTHER => 4,
    )
)

"""
    create_block_ordering(X_blocks, categories, names, layout, strategy)

Create a global ordering for sparse Cholesky from multiple functional blocks.

# Arguments
- `X_blocks`: Vector of coordinate matrices, one per functional
- `categories`: Vector of `FunctionalCategory` for each block
- `names`: Vector of Symbol names for each block
- `layout`: Layout mapping names to index ranges
- `strategy`: Ordering strategy (`:integrals_coarsest`, `:evaluations_coarsest`, etc.)

# Returns
- `P`: Global permutation vector
- `ℓ`: Maximin lengthscales for each index
- `X`: Global coordinate matrix
"""
function create_block_ordering(X_blocks::Vector, categories::Vector{FunctionalCategory},
                               names::Vector{Symbol}, layout::Layout,
                               strategy::OrderingStrategy)
    n_blocks = length(X_blocks)

    # Determine block order based on strategy
    if strategy == :natural
        block_order = collect(1:n_blocks)
    else
        priorities = get(ORDERING_PRIORITIES, strategy, ORDERING_PRIORITIES[:integrals_coarsest])
        block_order = sortperm([priorities[cat] for cat in categories])
    end

    # Compute maximin ordering for each block
    P_blocks = Vector{Vector{Int}}(undef, n_blocks)
    ℓ_blocks = Vector{Vector{Float64}}(undef, n_blocks)

    for i in 1:n_blocks
        P_local, ℓ_local = reverse_maximin_ordering(X_blocks[i])
        P_blocks[i] = P_local
        ℓ_blocks[i] = ℓ_local
    end

    # Find coarsest lengthscale (from the coarsest block = last in order)
    # Use the lengthscale of the first point in the permutation (the coarsest point)
    coarsest_block = block_order[end]
    P_coarsest = P_blocks[coarsest_block]
    ℓ_coarse = ℓ_blocks[coarsest_block][P_coarsest[1]]

    # Total state dimension
    n_total = sum(size(X, 2) for X in X_blocks)

    # Build global ordering following block_order
    # ℓ_global must be indexed by GLOBAL STATE INDEX (not permutation order)
    # because sparsity_pattern_from_ordering does ℓ[P[j]] to get lengthscale
    P_global = Int[]
    ℓ_global = zeros(n_total)

    for block_idx in block_order
        name = names[block_idx]
        P_local = P_blocks[block_idx]
        ℓ_local = ℓ_blocks[block_idx]

        # Map local indices to global indices
        global_indices = indices(layout, name)
        P_global_block = global_indices[P_local]

        # Set lengthscales at the GLOBAL indices (not appended in permutation order!)
        # For non-coarsest blocks, use the coarsest lengthscale
        if block_idx != coarsest_block
            for global_idx in global_indices
                ℓ_global[global_idx] = ℓ_coarse
            end
        else
            for (local_idx, global_idx) in enumerate(global_indices)
                ℓ_global[global_idx] = ℓ_local[local_idx]
            end
        end

        append!(P_global, P_global_block)
    end

    # Build global coordinate matrix (in original order, not block_order)
    X_global = hcat(X_blocks...)

    return P_global, ℓ_global, X_global
end

# ------------------------------------------------------------------------------
# Main API
# ------------------------------------------------------------------------------

"""
    SparseGMRFApproximation

Result of sparse Cholesky approximation to a GP defined by linear functionals.

# Fields
- `Q`: Sparse precision matrix (in original ordering)
- `layout`: Layout for named access to state vector
- `functionals`: Original linear functionals
- `info`: Named tuple with sparsity statistics
"""
struct SparseGMRFApproximation{TQ<:AbstractMatrix, TL<:Layout}
    Q::TQ
    layout::TL
    functionals::Vector
    info::NamedTuple
end

function Base.show(io::IO, approx::SparseGMRFApproximation)
    method = approx.info.supernodal ? "supernodal" : "simplicial"
    print(io, "SparseGMRFApproximation(n=$(approx.info.n), " *
              "nnz=$(approx.info.nnz), fill=$(round(approx.info.fill_pct, digits=1))%, $method)")
end

"""
    sparse_precision(named_functionals, kernel; ρ=2.0, λ=1.5, ordering=:integrals_coarsest)

Build a sparse precision matrix from named linear functionals using KL-optimal Cholesky approximation.

# Arguments
- `named_functionals`: Vector of `name => functional` pairs, e.g.,
  ```julia
  [:f => EvaluationFunctional(X),
   :f_dx => L_eval ∘ PartialDerivative((1,)),
   :f_int => VectorizedLebesgueIntegral(intervals)]
  ```
- `kernel`: GP kernel (e.g., `HalfIntegerMaternKernel`)

# Keyword Arguments
- `ρ=2.0`: Sparsity threshold. Larger values give denser but more accurate approximations.
- `λ=1.5`: Supernodal clustering threshold. Controls how columns are grouped for
  efficient factorization. Set to `nothing` to use simplicial (column-by-column)
  factorization instead. Supernodal is ~2-3x more efficient (more nnz per compute).
- `ordering=:integrals_coarsest`: Block ordering strategy. Options:
  - `:integrals_coarsest`: Integrals as coarsest (recommended for FVM)
  - `:evaluations_coarsest`: Evaluations as coarsest
  - `:derivatives_coarsest`: Derivatives as coarsest
  - `:natural`: Use the order provided

# Returns
- `SparseGMRFApproximation` containing the sparse precision matrix and metadata

# Example
```julia
# Define functionals
endpoints = range(0, 1, length=51)
intervals = intervals_from_endpoints(collect(endpoints))
k = HalfIntegerMaternKernel(2, [0.1])

L_eval = EvaluationFunctional(endpoints)
L_deriv = L_eval ∘ PartialDerivative((1,))
L_int = VectorizedLebesgueIntegral(intervals)

# Build sparse precision (supernodal by default)
approx = sparse_precision([
    :f => L_eval,
    :f_dx => L_deriv,
    :f_int => L_int
], k; ρ=2.0)

# Use simplicial instead
approx_simp = sparse_precision([...], k; ρ=2.0, λ=nothing)

# Create GMRF
x = GMRF(zeros(approx.info.n), approx.Q)
```
"""
function sparse_precision(named_functionals::Vector{<:Pair{Symbol}}, kernel;
                          ρ::Real=2.0, λ::Union{Real,Nothing}=1.5,
                          ordering::OrderingStrategy=:integrals_coarsest)
    # Extract names and functionals
    names = Symbol[first(p) for p in named_functionals]
    functionals = [last(p) for p in named_functionals]
    n_blocks = length(functionals)

    # Classify functionals
    categories = [functional_category(L) for L in functionals]

    # Extract coordinates
    X_blocks = [get_coordinates(L) for L in functionals]

    # Get output sizes and build layout
    sizes = [output_length(L) for L in functionals]
    state_layout = layout(NamedTuple{Tuple(names)}(Tuple(sizes)))
    n_total = sum(sizes)

    # Build stacked functional and covariance matrix
    L_stack = StackedLinearFunctional(functionals...)
    K = L_stack(L_stack(kernel))
    K_sym = Symmetric(K)

    # Create ordering
    P, ℓ, X = create_block_ordering(X_blocks, categories, names, state_layout, ordering)

    # Create sparsity pattern
    S = sparsity_pattern_from_ordering(X, P, ℓ, Float64(ρ))

    # Apply sparse Cholesky (supernodal by default, simplicial if λ=nothing)
    if λ === nothing
        # Simplicial (column-by-column) factorization
        K_P = PermutedMatrix(K_sym, P)
        sparse_approximate_cholesky!(K_P, S)
        L = S
    else
        # Supernodal factorization with FastBlockMatrix optimization
        # For BlockMatrix inputs (2D tensor products), this provides ~40-50x speedup
        if is_block_matrix(K)
            K_P = FastPermutedMatrix(FastBlockMatrix(K), P)
        else
            K_P = PermutedMatrix(K_sym, P)
        end
        sc = form_supernodes(S, P, ℓ; λ=Float64(λ))
        L = sparse_approximate_cholesky(K_P, sc)
    end

    # Build sparse precision (unpermuted to original ordering)
    Q_perm = L * L'
    P_inv = invperm(P)
    Q_sparse = Q_perm[P_inv, P_inv]

    # Compute sparsity statistics
    nnz_Q = nnz(Q_sparse)
    nnz_L = nnz(L)
    dense_nnz = n_total * (n_total + 1) ÷ 2
    fill_pct = 100.0 * nnz_Q / (2 * dense_nnz)

    info = (
        ρ = Float64(ρ),
        λ = λ === nothing ? nothing : Float64(λ),
        supernodal = λ !== nothing,
        n = n_total,
        nnz = nnz_Q,
        nnz_L = nnz_L,
        fill_pct = fill_pct,
        ordering = ordering,
        categories = categories
    )

    return SparseGMRFApproximation(Q_sparse, state_layout, functionals, info)
end

"""
    sparse_gmrf(named_functionals, kernel; ρ=2.0, λ=1.5, ordering=:integrals_coarsest)

Create a GMRF with sparse precision from named linear functionals.

Convenience wrapper around `sparse_precision` that directly returns a GMRF.

# Returns
- `gmrf`: GMRF with zero mean and sparse precision
- `layout`: Layout for named access to state vector
- `approx`: Full `SparseGMRFApproximation` with metadata
"""
function sparse_gmrf(named_functionals::Vector{<:Pair{Symbol}}, kernel;
                     ρ::Real=2.0, λ::Union{Real,Nothing}=1.5,
                     ordering::OrderingStrategy=:integrals_coarsest)
    approx = sparse_precision(named_functionals, kernel; ρ=ρ, λ=λ, ordering=ordering)
    gmrf = GMRF(zeros(approx.info.n), approx.Q)
    return gmrf, approx.layout, approx
end

# ------------------------------------------------------------------------------
# Convenience: prescribe_indices (conditioning helper)
# ------------------------------------------------------------------------------

"""
    prescribe_indices(x::GMRF, indices, values; noise_std=1e-3)

Condition a GMRF on observations at specific indices.

# Arguments
- `x`: GMRF to condition
- `indices`: Vector of indices where observations are made
- `values`: Observation values
- `noise_std`: Observation noise standard deviation

# Returns
- Conditioned GMRF
"""
function prescribe_indices(x::GMRF, indices, values; noise_std::Real=1e-3)
    obs_model = ExponentialFamily(Normal, indices=indices)
    obs_lik = obs_model(values; σ=Float64(noise_std))
    return gaussian_approximation(x, obs_lik)
end

# ------------------------------------------------------------------------------
# Output Scale Calibration
# ------------------------------------------------------------------------------

"""
    calibrate_output_scale(Q₀, A_fvm, μ_BC; n_samples=10)

Estimate output scale σ via quasi-MLE from FVM innovations.

After conditioning on ICs/BCs to get `μ_BC`, the FVM constraints have nonzero
innovations. This function estimates σ² such that the standardized innovations
have unit variance (well-calibrated uncertainty).

# Arguments
- `Q₀`: Unit-scale sparse precision matrix
- `A_fvm`: FVM constraint matrix (n_constraints × n_state)
- `μ_BC`: Mean after BC conditioning (length n_state)
- `n_samples`: Number of random constraints to sample (default 10)

# Returns
- `σ`: Estimated output scale (square root of σ²)

# Reference
Based on CAPOS (Bosch, Hennig, Tronarp 2021) quasi-MLE approach for
probabilistic ODE solvers.

# Example
```julia
# After building sparse precision with unit output scale
approx = sparse_precision(functionals, kernel; ρ=2.0)

# Condition on boundary conditions
x_BC = prescribe_indices(x0, bc_indices, bc_values)
μ_BC = mean(x_BC)

# Estimate output scale from FVM innovations
σ = calibrate_output_scale(approx.Q, A_fvm, μ_BC; n_samples=10)

# Apply calibrated scale
Q_calibrated = approx.Q / σ^2
```
"""
function calibrate_output_scale(Q₀, A_fvm, μ_BC; n_samples::Int=10)
    n_constraints = size(A_fvm, 1)
    n_samples = min(n_samples, n_constraints)

    # Compute Cholesky factorization once
    F = cholesky(Q₀)

    # Sample random constraint indices
    sample_idx = randperm(n_constraints)[1:n_samples]

    σ²_sum = 0.0
    for i in sample_idx
        aᵢ = Vector(A_fvm[i, :])

        # Innovation: how far is BC-conditioned mean from satisfying constraint?
        rᵢ = -dot(aᵢ, μ_BC)

        # Prior variance of constraint: sᵢ = aᵢ' Σ₀ aᵢ = aᵢ' Q₀⁻¹ aᵢ
        # Solve Q₀ x = aᵢ to get x = Σ₀ aᵢ, then sᵢ = aᵢ' x
        x = F \ aᵢ
        sᵢ = dot(aᵢ, x)

        # Accumulate normalized squared innovation
        if sᵢ > 0
            σ²_sum += rᵢ^2 / sᵢ
        end
    end

    σ² = σ²_sum / n_samples

    # Guard against σ = 0 (can happen if all innovations are zero)
    σ = sqrt(max(σ², 1e-10))

    return σ
end
