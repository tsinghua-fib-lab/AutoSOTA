using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: mean
using LinearAlgebra
using SparseArrays

export ConstantLinearGaussianSSM
export joint_gmrf

"""
    ConstantLinearGaussianSSM

Linear Gaussian SSM with constant transition and noise matrices.
"""
struct ConstantLinearGaussianSSM{G<:GMRF, TA<:AbstractMatrix, TQ<:AbstractMatrix}
    x0::G
    A::TA
    Q_noise::TQ
end

_safe_nnz(A::SparseMatrixCSC) = nnz(A)
_safe_nnz(A::AbstractMatrix) = prod(size(A))

function joint_gmrf(ssm::ConstantLinearGaussianSSM, n_t::Int)
    n_t >= 1 || throw(ArgumentError("n_t must be at least 1"))

    n_t == 1 && return ssm.x0

    x0 = ssm.x0
    Q0 = precision_matrix(x0)
    A = ssm.A
    Q_noise = ssm.Q_noise

    n = length(x0)
    size(A, 1) == n ||
        throw(DimensionMismatch("state transition matrix has incompatible row dimension"))
    size(A, 2) == n ||
        throw(DimensionMismatch("state transition matrix has incompatible column dimension"))
    size(Q_noise, 1) == n == size(Q_noise, 2) ||
        throw(DimensionMismatch("noise precision has incompatible dimensions"))

    QA = Q_noise * A
    AtQA = sparse(transpose(A) * QA)
    QA = sparse(QA)
    off_lower = -QA
    diag_first = sparse(Q0 + AtQA)
    Q_noise_sparse = sparse(Q_noise)
    diag_middle = sparse(Q_noise_sparse + AtQA)
    diag_last = Q_noise_sparse

    nnz_diag_first = nnz(diag_first)
    nnz_diag_middle = n_t > 2 ? nnz(diag_middle) : 0
    nnz_diag_last = nnz(diag_last)
    nnz_off = nnz(off_lower)

    total_nnz = nnz_diag_first + nnz_diag_last + (n_t - 2) * nnz_diag_middle + 2 * (n_t - 1) * nnz_off
    rows = Vector{Int}(undef, 0)
    cols = Vector{Int}(undef, 0)
    vals = Vector{eltype(Q_noise)}(undef, 0)
    sizehint!(rows, total_nnz)
    sizehint!(cols, total_nnz)
    sizehint!(vals, total_nnz)

    function push_block!(I::Vector{Int}, J::Vector{Int}, V, block::SparseMatrixCSC, bi::Int, bj::Int)
        roffset = (bi - 1) * n
        coffset = (bj - 1) * n
        r, c, v = findnz(block)
        append!(I, roffset .+ r)
        append!(J, coffset .+ c)
        append!(V, v)
    end

    function push_block_transpose!(I::Vector{Int}, J::Vector{Int}, V, block::SparseMatrixCSC, bi::Int, bj::Int)
        roffset = (bi - 1) * n
        coffset = (bj - 1) * n
        r, c, v = findnz(block)
        append!(I, roffset .+ c)
        append!(J, coffset .+ r)
        append!(V, v)
    end

    push_block!(rows, cols, vals, diag_first, 1, 1)
    for t in 2:n_t-1
        push_block!(rows, cols, vals, diag_middle, t, t)
    end
    push_block!(rows, cols, vals, diag_last, n_t, n_t)

    for t in 1:n_t-1
        push_block!(rows, cols, vals, off_lower, t + 1, t)
        push_block_transpose!(rows, cols, vals, off_lower, t, t + 1)
    end

    joint_precision = sparse(rows, cols, vals, n * n_t, n * n_t)

    μ0 = mean(x0)
    T = promote_type(eltype(μ0), eltype(Q_noise))
    joint_mean = Vector{T}(undef, n * n_t)
    joint_mean[1:n] = μ0

    prev = copy(μ0)
    curr = similar(prev)
    for t in 2:n_t
        mul!(curr, A, prev)
        joint_mean[(t - 1) * n + 1:t * n] = curr
        prev, curr = curr, prev
    end

    return GMRF(joint_mean, Symmetric(joint_precision))
end
