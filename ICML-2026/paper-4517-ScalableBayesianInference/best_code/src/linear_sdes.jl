export LinearSDE, Matern32SDE, IWPSDE,
       F, L, qc, Pinf,
       discretize_vanloan, sde_dim

using LinearAlgebra
using MatrixEquations

# -- Interface ---------------------------------------------------------------
abstract type LinearSDE end

"""
    sde_dim(sde) -> Int

Dimension `n` of the state `u(t) ∈ R^n`.
"""
sde_dim(::LinearSDE) = error("sde_dim not implemented for $(typeof(sde))")

"""
    F(sde) -> Matrix{<:Real}
    L(sde) -> Matrix{<:Real}
    qc(sde) -> Real
    H(sde) -> Matrix{<:Real}

Continuous-time linear SDE:
    du = F u dt + L dβ,   E[dβ dβᵀ] = qc * I * dt
Observation model (for the primary process):
    y = H u

`qc` is the scalar diffusion intensity for the driving Wiener process.
"""
F(::LinearSDE)  = error("F not implemented")
L(::LinearSDE)  = error("L not implemented")
qc(::LinearSDE) = error("qc not implemented")

"""
    Pinf(sde) -> Matrix

Stationary state covariance solving the continuous-time Lyapunov equation
    F P + P Fᵀ + L (qc) Lᵀ = 0.
Provides a generic implementation; override if you have a closed form.
"""
function Pinf(sde::LinearSDE)
    A = F(sde); B = L(sde); q = qc(sde)
    Qc = B * (q * I(size(B,2))) * B'  # diffusion covariance
    return lyapc(A, Qc)
end

# -- Discretization ----------------------------------------------------------
"""
    discretize_vanloan(sde, Δt) -> (A, Q)

Exact discretization using the Van Loan method:
    u_{k} = A u_{k-1} + ε_k,   ε_k ~ N(0, Q)
where
    A = exp(F Δt),
    Q = ∫₀^{Δt} exp(F τ) L qc Lᵀ exp(Fᵀ τ) dτ.
"""
function discretize_vanloan(sde::LinearSDE, Δt::Real)
    A = F(sde); B = L(sde); q = qc(sde)
    n = size(A,1)
    G = B * sqrt(q)          # so that GGᵀ = B q Bᵀ
    Z = zeros(eltype(A), n, n)
    # Van Loan block matrix
    M = [ -A           G*G'
          Z            A'   ] * Δt
    E = exp(M)
    E11 = E[1:n, 1:n]
    E12 = E[1:n, n+1:end]
    A_d = Matrix(transpose(E[n+1:end, n+1:end]))  # equals exp(A Δt)
    Q_d = E11 * E12                      # equals ∫ exp(A τ) GGᵀ exp(Aᵀ τ) dτ
    Q_d = 0.5 * (Q_d + Q_d')
    return (A_d, Q_d)
end

# -- Matérn 3/2 in time ------------------------------------------------------
"""
    Matern32SDE(σ2, ℓ)

Exact SDE for a Matérn-3/2 temporal GP with variance `σ2` and lengthscale `ℓ`.
State is `[f, ḟ]` so the dimension is 2.
Dynamics:
    F = [ 0  1; -λ^2  -2λ ],    L = [0; 1],    qc = 4 λ^3 σ²,   H = [1  0]
with λ = √3 / ℓ.
"""
struct Matern32SDE <: LinearSDE
    σ2::Float64
    ℓ::Float64
end

sde_dim(::Matern32SDE) = 2

function F(sde::Matern32SDE)
    λ = sqrt(3.0) / sde.ℓ
    return [ 0.0  1.0
            -λ^2 -2λ ]
end

L(::Matern32SDE) = [0.0; 1.0]

function qc(sde::Matern32SDE)
    λ = sqrt(3.0) / sde.ℓ
    return 4.0 * λ^3 * sde.σ2
end

# Optional: closed-form P∞ for Matérn-3/2
function Pinf(sde::Matern32SDE)
    λ = sqrt(3.0) / sde.ℓ
    σ2 = sde.σ2
    return Diagonal([σ2, (λ^2) * σ2])
end

function discretize_vanloan(sde::Matern32SDE, Δ::Real)
    σ2 = sde.σ2
    λ = sqrt(3.0)/sde.ℓ
    r = λ*Δ
    e = exp(-r)

    A = e .* [1+r   Δ;
              -λ^2*Δ 1-r]

    Q11 = σ2 * (1 - exp(-2r)*(1 + 2r + 2r*r))
    Q12 = 2 * σ2 * λ^3 * Δ^2 * exp(-2r)     # == 2*σ2*λ*r^2*exp(-2r)
    Q22 = λ^2 * σ2 * (1 - exp(-2r)*(1 - 2r + 2r*r))

    Q = [Q11 Q12; Q12 Q22]
    # (Optional) tiny symmetrization for FP noise:
    Q = 0.5*(Q + Q')
    return A, Q
end

struct IWPSDE <: LinearSDE
    σ2::Float64
end

sde_dim(::IWPSDE) = 2
F(::IWPSDE) = [0.0 1.0; 0.0 0.0]
L(::IWPSDE) = [0.0; 1.0]
qc(sde::IWPSDE) = sde.σ2

Pinf(::IWPSDE) = error("Pinf doesn't exist for IWP SDE")

function discretize_vanloan(sde::IWPSDE, Δ::Real)
    A = [1.0 Δ; 0.0 1.0]
    Q = sde.σ2 .* [Δ^3/3 Δ^2/2; Δ^2/2 Δ]
    Q = 0.5 * (Q + Q')
    return A, Q
end
