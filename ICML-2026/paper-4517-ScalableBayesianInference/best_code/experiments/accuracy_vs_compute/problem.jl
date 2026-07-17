"""
    Burgers Equation Problem Definition

Defines the viscous Burgers equation, initial condition families with randomized
parameters, and reference solutions (Cole-Hopf exact + high-resolution FVM).

PDE: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
Domain: x ∈ [0, 1], t ∈ [0, T]
"""

using Random
using QuadGK
using SpecialFunctions: erf

# ==============================================================================
# Problem Parameters
# ==============================================================================

"""
    BurgersProblem

Defines the Burgers equation problem setup.
"""
Base.@kwdef struct BurgersProblem
    # Spatial domain
    x_min::Float64 = 0.0
    x_max::Float64 = 1.0

    # Temporal domain
    T_end::Float64 = 0.5

    # Physics
    ν::Float64 = 0.01  # Viscosity

    # Boundary conditions (Dirichlet)
    u_left::Float64 = 0.0
    u_right::Float64 = 0.0
end

domain_length(p::BurgersProblem) = p.x_max - p.x_min

# ==============================================================================
# Initial Condition Families
# ==============================================================================

"""
    ICFamily

Abstract type for initial condition families.
Each family defines a parametric form with randomizable parameters.
"""
abstract type ICFamily end

"""
    sample_parameters(family::ICFamily, rng::AbstractRNG)

Sample random parameters for an IC family.
"""
function sample_parameters end

"""
    evaluate_ic(family::ICFamily, params, x)

Evaluate the initial condition at point x with given parameters.
"""
function evaluate_ic end

"""
    evaluate_ic_dx(family::ICFamily, params, x)

Evaluate the derivative of the initial condition at point x.
"""
function evaluate_ic_dx end

"""
    evaluate_ic_int(family::ICFamily, params, a, b)

Evaluate the integral of the initial condition over [a, b].
"""
function evaluate_ic_int(family::ICFamily, params, a, b)
    # Default: numerical integration
    return quadgk(x -> evaluate_ic(family, params, x), a, b)[1]
end

# ------------------------------------------------------------------------------
# Sine Wave IC
# ------------------------------------------------------------------------------

"""
    SineIC <: ICFamily

Sine wave initial condition: u₀(x) = A sin(2πkx + φ)

Parameters:
- A ∈ [0.8, 1.2]: Amplitude
- k ∈ {1, 2}: Wave number
- φ ∈ [0, 2π): Phase
"""
struct SineIC <: ICFamily end

struct SineParams
    A::Float64
    k::Int
    φ::Float64
end

function sample_parameters(::SineIC, rng::AbstractRNG)
    A = 0.8 + 0.4 * rand(rng)
    k = rand(rng, [1, 2])
    # Phase must be 0 or π to satisfy u(0) = u(1) = 0 Dirichlet BCs
    # sin(2πkx + φ) = 0 at x=0,1 requires φ = 0 or π
    φ = rand(rng, [0.0, π])
    return SineParams(A, k, φ)
end

function evaluate_ic(::SineIC, p::SineParams, x)
    return p.A * sin(2π * p.k * x + p.φ)
end

function evaluate_ic_dx(::SineIC, p::SineParams, x)
    return p.A * 2π * p.k * cos(2π * p.k * x + p.φ)
end

function evaluate_ic_int(::SineIC, p::SineParams, a, b)
    # Analytical integral of A*sin(2πkx + φ)
    coeff = -p.A / (2π * p.k)
    return coeff * (cos(2π * p.k * b + p.φ) - cos(2π * p.k * a + p.φ))
end

# ------------------------------------------------------------------------------
# Gaussian Pulse IC
# ------------------------------------------------------------------------------

"""
    GaussianIC <: ICFamily

Gaussian pulse initial condition: u₀(x) = A exp(-(x-μ)²/(2σ²))

Parameters:
- A ∈ [0.8, 1.2]: Amplitude
- μ ∈ [0.3, 0.7]: Center position
- σ ∈ [0.05, 0.15]: Width
"""
struct GaussianIC <: ICFamily end

struct GaussianParams
    A::Float64
    μ::Float64
    σ::Float64
end

function sample_parameters(::GaussianIC, rng::AbstractRNG)
    A = 0.8 + 0.4 * rand(rng)
    μ = 0.3 + 0.4 * rand(rng)
    σ = 0.05 + 0.1 * rand(rng)
    return GaussianParams(A, μ, σ)
end

function evaluate_ic(::GaussianIC, p::GaussianParams, x)
    return p.A * exp(-(x - p.μ)^2 / (2 * p.σ^2))
end

function evaluate_ic_dx(::GaussianIC, p::GaussianParams, x)
    return -p.A * (x - p.μ) / p.σ^2 * exp(-(x - p.μ)^2 / (2 * p.σ^2))
end

function evaluate_ic_int(::GaussianIC, p::GaussianParams, a, b)
    # Analytical integral using error function
    coeff = p.A * p.σ * sqrt(π / 2)
    return coeff * (erf((b - p.μ) / (p.σ * sqrt(2))) - erf((a - p.μ) / (p.σ * sqrt(2))))
end

# ------------------------------------------------------------------------------
# Step/Tanh IC
# ------------------------------------------------------------------------------

"""
    StepIC <: ICFamily

Smoothed step initial condition: u₀(x) = A (1 - tanh((x-μ)/ε)) / 2

Parameters:
- A ∈ [0.8, 1.2]: Amplitude
- μ ∈ [0.3, 0.7]: Step location
- ε ∈ [0.08, 0.15]: Transition width (wider for GP compatibility)
"""
struct StepIC <: ICFamily end

struct StepParams
    A::Float64
    μ::Float64
    ε::Float64
end

function sample_parameters(::StepIC, rng::AbstractRNG)
    A = 0.8 + 0.4 * rand(rng)
    μ = 0.3 + 0.4 * rand(rng)
    ε = 0.08 + 0.07 * rand(rng)  # Wider transition for GP compatibility
    return StepParams(A, μ, ε)
end

function evaluate_ic(::StepIC, p::StepParams, x)
    return p.A * (1 - tanh((x - p.μ) / p.ε)) / 2
end

function evaluate_ic_dx(::StepIC, p::StepParams, x)
    sech_val = sech((x - p.μ) / p.ε)
    return -p.A * sech_val^2 / (2 * p.ε)
end

function evaluate_ic_int(::StepIC, p::StepParams, a, b)
    # Analytical integral: ∫ (1 - tanh(y))/2 dy = y/2 - ln(cosh(y))/2
    # With substitution y = (x - μ)/ε
    function antiderivative(x)
        y = (x - p.μ) / p.ε
        return p.A * p.ε * (y / 2 - log(cosh(y)) / 2)
    end
    return antiderivative(b) - antiderivative(a)
end

# ------------------------------------------------------------------------------
# Multi-mode IC
# ------------------------------------------------------------------------------

"""
    MultiModeIC <: ICFamily

Multi-mode initial condition: u₀(x) = A₁ sin(2πx) + A₂ sin(4πx)

Parameters:
- A₁ ∈ [0.8, 1.2]: First mode amplitude
- A₂ ∈ [0.3, 0.7]: Second mode amplitude
"""
struct MultiModeIC <: ICFamily end

struct MultiModeParams
    A₁::Float64
    A₂::Float64
end

function sample_parameters(::MultiModeIC, rng::AbstractRNG)
    A₁ = 0.8 + 0.4 * rand(rng)
    A₂ = 0.3 + 0.4 * rand(rng)
    return MultiModeParams(A₁, A₂)
end

function evaluate_ic(::MultiModeIC, p::MultiModeParams, x)
    return p.A₁ * sin(2π * x) + p.A₂ * sin(4π * x)
end

function evaluate_ic_dx(::MultiModeIC, p::MultiModeParams, x)
    return p.A₁ * 2π * cos(2π * x) + p.A₂ * 4π * cos(4π * x)
end

function evaluate_ic_int(::MultiModeIC, p::MultiModeParams, a, b)
    # Analytical integral
    term1 = -p.A₁ / (2π) * (cos(2π * b) - cos(2π * a))
    term2 = -p.A₂ / (4π) * (cos(4π * b) - cos(4π * a))
    return term1 + term2
end

# ------------------------------------------------------------------------------
# IC Family Registry
# ------------------------------------------------------------------------------

const IC_FAMILIES = Dict{Symbol, ICFamily}(
    :sine => SineIC(),
    :gaussian => GaussianIC(),
    :step => StepIC(),
    :multimode => MultiModeIC()
)

"""
    get_ic_family(name::Symbol)

Get an IC family by name.
"""
get_ic_family(name::Symbol) = IC_FAMILIES[name]

"""
    list_ic_families()

List available IC family names.
"""
list_ic_families() = collect(keys(IC_FAMILIES))

# ==============================================================================
# Initial Condition Instance
# ==============================================================================

"""
    InitialCondition

A concrete initial condition instance with sampled parameters.
"""
struct InitialCondition{F<:ICFamily, P}
    family::F
    params::P
    seed::Int
end

function InitialCondition(family::ICFamily, seed::Int)
    rng = MersenneTwister(seed)
    params = sample_parameters(family, rng)
    return InitialCondition(family, params, seed)
end

function InitialCondition(family_name::Symbol, seed::Int)
    family = get_ic_family(family_name)
    return InitialCondition(family, seed)
end

# Evaluation interface
(ic::InitialCondition)(x) = evaluate_ic(ic.family, ic.params, x)
evaluate_dx(ic::InitialCondition, x) = evaluate_ic_dx(ic.family, ic.params, x)
evaluate_int(ic::InitialCondition, a, b) = evaluate_ic_int(ic.family, ic.params, a, b)

function Base.show(io::IO, ic::InitialCondition)
    print(io, "InitialCondition($(typeof(ic.family).name.name), seed=$(ic.seed))")
end

# ==============================================================================
# Reference Solutions
# ==============================================================================

"""
    AbstractReferenceSolution

Abstract type for reference solutions.
"""
abstract type AbstractReferenceSolution end

"""
    evaluate_reference(ref::AbstractReferenceSolution, x, t)

Evaluate the reference solution at point (x, t).
"""
function evaluate_reference end

# ------------------------------------------------------------------------------
# High-Resolution FVM Reference
# ------------------------------------------------------------------------------

"""
    HighResFVMReference

Reference solution computed via high-resolution classical FVM.
"""
struct HighResFVMReference <: AbstractReferenceSolution
    xs::Vector{Float64}      # Grid points
    ts::Vector{Float64}      # Time points
    solution::Matrix{Float64} # Solution array [x_idx, t_idx]
end

"""
    compute_highres_reference(problem::BurgersProblem, ic::InitialCondition;
                               N_ref=2000, N_t=1000)

Compute a high-resolution reference solution using Godunov FVM.
"""
function compute_highres_reference(problem::BurgersProblem, ic::InitialCondition;
                                    N_ref::Int=2000, N_t::Int=1000)
    (; x_min, x_max, T_end, ν, u_left, u_right) = problem

    # Grid setup
    xs = range(x_min, x_max, length=N_ref)
    Δx = xs[2] - xs[1]

    # CFL condition for stability
    u_max = maximum(abs.(ic.(xs)))
    Δt_advection = 0.5 * Δx / max(u_max, 1e-10)
    Δt_diffusion = 0.25 * Δx^2 / ν
    Δt = min(Δt_advection, Δt_diffusion, T_end / N_t)

    # Actual number of time steps
    N_t_actual = ceil(Int, T_end / Δt)
    Δt = T_end / N_t_actual
    ts = range(0, T_end, length=N_t_actual + 1)

    # Initialize solution
    u = ic.(xs)
    solution = zeros(N_ref, N_t_actual + 1)
    solution[:, 1] = u

    # Time stepping (Godunov + central diff for viscosity)
    u_new = similar(u)
    for n in 1:N_t_actual
        # Interior points
        for i in 2:(N_ref-1)
            # Godunov flux for advection: F = u²/2
            # Upwind based on characteristic speed
            u_L = u[i-1]
            u_R = u[i]
            F_left = godunov_flux(u_L, u_R)

            u_L = u[i]
            u_R = u[i+1]
            F_right = godunov_flux(u_L, u_R)

            # Central difference for viscosity
            visc = ν * (u[i+1] - 2*u[i] + u[i-1]) / Δx^2

            u_new[i] = u[i] - Δt / Δx * (F_right - F_left) + Δt * visc
        end

        # Boundary conditions
        u_new[1] = u_left
        u_new[N_ref] = u_right

        u .= u_new
        solution[:, n+1] = u
    end

    return HighResFVMReference(collect(xs), collect(ts), solution)
end

"""
    godunov_flux(u_L, u_R)

Godunov numerical flux for Burgers equation: F(u) = u²/2
"""
function godunov_flux(u_L, u_R)
    if u_L >= u_R
        # Shock or expansion with s ≥ 0
        if u_L + u_R >= 0
            return 0.5 * u_L^2
        else
            return 0.5 * u_R^2
        end
    else
        # Rarefaction
        if u_L >= 0
            return 0.5 * u_L^2
        elseif u_R <= 0
            return 0.5 * u_R^2
        else
            return 0.0  # Sonic point
        end
    end
end

function evaluate_reference(ref::HighResFVMReference, x, t)
    # Linear interpolation in space and time
    xs, ts, sol = ref.xs, ref.ts, ref.solution

    # Find bracketing indices
    i_x = searchsortedlast(xs, x)
    i_t = searchsortedlast(ts, t)

    # Clamp to valid range
    i_x = clamp(i_x, 1, length(xs) - 1)
    i_t = clamp(i_t, 1, length(ts) - 1)

    # Interpolation weights
    α_x = (x - xs[i_x]) / (xs[i_x + 1] - xs[i_x])
    α_t = (t - ts[i_t]) / (ts[i_t + 1] - ts[i_t])

    # Bilinear interpolation
    u_00 = sol[i_x, i_t]
    u_10 = sol[i_x + 1, i_t]
    u_01 = sol[i_x, i_t + 1]
    u_11 = sol[i_x + 1, i_t + 1]

    u_0 = (1 - α_x) * u_00 + α_x * u_10
    u_1 = (1 - α_x) * u_01 + α_x * u_11

    return (1 - α_t) * u_0 + α_t * u_1
end

"""
    evaluate_reference(ref::HighResFVMReference, xs::AbstractVector, t)

Evaluate reference solution at multiple x points for a single time t.
"""
function evaluate_reference(ref::HighResFVMReference, xs::AbstractVector, t)
    return [evaluate_reference(ref, x, t) for x in xs]
end

# ==============================================================================
# Problem Instance
# ==============================================================================

"""
    ProblemInstance

A complete problem instance with problem definition, IC, and reference solution.
"""
struct ProblemInstance
    problem::BurgersProblem
    ic::InitialCondition
    reference::HighResFVMReference
end

function ProblemInstance(problem::BurgersProblem, ic::InitialCondition;
                          N_ref::Int=2000, N_t::Int=1000)
    reference = compute_highres_reference(problem, ic; N_ref=N_ref, N_t=N_t)
    return ProblemInstance(problem, ic, reference)
end

"""
    generate_problem_instances(problem::BurgersProblem, ic_families, n_samples::Int;
                                base_seed::Int=42, N_ref::Int=2000)

Generate multiple problem instances with randomized ICs.
"""
function generate_problem_instances(problem::BurgersProblem,
                                     ic_families::Vector{Symbol},
                                     n_samples::Int;
                                     base_seed::Int=42,
                                     N_ref::Int=2000,
                                     verbose::Bool=true)
    instances = ProblemInstance[]

    for (i_fam, family_name) in enumerate(ic_families)
        verbose && println("Generating $n_samples instances for IC family: $family_name")

        for i_sample in 1:n_samples
            # Deterministic seed from family index and sample index
            seed = base_seed + (i_fam - 1) * 1000 + i_sample

            ic = InitialCondition(family_name, seed)
            instance = ProblemInstance(problem, ic; N_ref=N_ref)
            push!(instances, instance)

            verbose && print(".")
        end
        verbose && println(" done")
    end

    return instances
end
