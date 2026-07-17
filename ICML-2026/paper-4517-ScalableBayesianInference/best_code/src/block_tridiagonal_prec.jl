export build_time_precision

using LinearAlgebra
using SparseArrays

# -- Block-tridiagonal time precision ---------------------------------------
"""
    build_time_precision(sde, Δts::AbstractVector{<:Real}; P0=:stationary) -> SparseMatrixCSC
    build_time_precision(As::Vector{<:AbstractMatrix}, Qs::Vector{<:AbstractMatrix}; P0=:stationary, P0_mat=nothing) -> SparseMatrixCSC
    build_time_precision(A::AbstractMatrix, Q::AbstractMatrix, T::Integer; P0=:stationary, P0_mat=nothing) -> SparseMatrixCSC

Construct the block-tridiagonal prior precision for the Markov chain of states
`u₁, …, u_T` arising from the discretized SDE.

You can provide:
  • an `sde` and a vector of step sizes `Δts` (computes `(A_k,Q_k)` via Van Loan), or
  • explicit vectors `As, Qs` of length `T-1`, or
  • a fixed pair `(A,Q)` and the horizon `T`.

`P0` can be:
  - `:stationary` (default): uses `Pinf(sde)` if available, or a discrete Lyapunov solve if `(A,Q)` (or `As,Qs`) are given and `P0_mat` not provided.
  - a Matrix: directly uses the provided covariance for `u₁`.

Returns a symmetric sparse CSC matrix of size `(nT)×(nT)` with block size `n`.
"""
function build_time_precision(sde::LinearSDE, Δts::AbstractVector{<:Real}; P0=:stationary, Q0=nothing)
    nsteps = length(Δts)
    As = Vector{Matrix{Float64}}(undef, nsteps)
    Qs = Vector{Matrix{Float64}}(undef, nsteps)
    for k in 1:nsteps
        As[k], Qs[k] = discretize_vanloan(sde, Δts[k])
    end
    P0_mat = if P0 === :stationary
        Pinf(sde)
    elseif Q0 === nothing
        Matrix(P0)
    else
        nothing
    end
    return build_time_precision(As, Qs; P0=P0, P0_mat=P0_mat, Q0=Q0)
end

function build_time_precision(A::AbstractMatrix, Q::AbstractMatrix, T::Integer; P0=:stationary, P0_mat=nothing, Q0=nothing)
    As = fill(Matrix(A), T-1)
    Qs = fill(Matrix(Q), T-1)
    if P0 === :stationary && P0_mat === nothing
        P0_mat = _stationary_cov_from_AQ(A, Q)
    elseif P0 !== :stationary && Q0 !== nothing
        P0_mat = Matrix(P0)
    end
    return build_time_precision(As, Qs; P0=:given, P0_mat=P0_mat, Q0=Q0)
end

function build_time_precision(As::Vector{<:AbstractMatrix}, Qs::Vector{<:AbstractMatrix}; P0=:stationary, P0_mat=nothing, Q0=nothing)
    @assert length(As) == length(Qs)
    T = length(As) + 1
    n = size(As[1], 1)

    Qi = [cholesky(Symmetric(Qs[k])) \ I for k in 1:length(Qs)]

    rows = Int[]; cols = Int[]; vals = Float64[]

    # k = 1
    if Q0 !== nothing
        J11 = Q0
    else
        J11 = (P0 === :stationary || P0 === :given) ? inv(Symmetric(P0_mat)) : inv(Symmetric(P0))
    end
    J11 = Matrix(J11)
    J11 .+= As[1]' * Qi[1] * As[1]
    J11 = 0.5 * (J11 + J11')
    J12 = -As[1]' * Qi[1]
    for j in 1:n, i in 1:n
        push!(rows, i); push!(cols, j); push!(vals, J11[i,j])
        push!(rows, i); push!(cols, j+n); push!(vals, J12[i,j])
    end

    # interior blocks
    for k in 2:T-1
        r = (k-1)*n
        Jkm1 = -Qi[k-1] * As[k-1]
        Jkk  =  Qi[k-1] + As[k]' * Qi[k] * As[k]
        Jkk = 0.5 * (Jkk + Jkk')
        Jkp1 = -As[k]' * Qi[k]
        for j in 1:n, i in 1:n
            push!(rows, r+i); push!(cols, r-n+j); push!(vals, Jkm1[i,j])
            push!(rows, r+i); push!(cols, r+j);   push!(vals, Jkk[i,j])
            push!(rows, r+i); push!(cols, r+n+j); push!(vals, Jkp1[i,j])
        end
    end

    # k = T
    rT = (T-1)*n
    JTkm1 = -Qi[end] * As[end]
    JTT   =  Qi[end]
    for j in 1:n, i in 1:n
        push!(rows, rT+i); push!(cols, rT-n+j); push!(vals, JTkm1[i,j])
        push!(rows, rT+i); push!(cols, rT+j);   push!(vals, JTT[i,j])
    end

    Jmat = sparse(rows, cols, vals, n*T, n*T)
    return Symmetric(Jmat)
end

# Compute stationary covariance from (A,Q) by solving discrete Lyapunov
function _stationary_cov_from_AQ(A::AbstractMatrix, Q::AbstractMatrix)
    n = size(A,1)
    K = I(n*n) - kron(A, A)
    p = K \ vec(Q)
    return Matrix(Symmetric(reshape(p, n, n)))
end
