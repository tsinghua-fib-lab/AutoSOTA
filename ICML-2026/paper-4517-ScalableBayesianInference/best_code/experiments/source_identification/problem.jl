"""
Problem definition for source identification experiments.

Defines the physics, sources, and observations for advection-diffusion
source identification. Both GP-FVM and PINN baselines read from the same
problem files for fair comparison.
"""

using TOML

"""
    GaussianSource

A Gaussian source term: s(x,y) = strength * exp(-((x-x₀)² + (y-y₀)²) / (2σ²))
"""
struct GaussianSource
    x::Float64
    y::Float64
    strength::Float64
    width::Float64  # σ
end

"""
    Observation

A point observation of concentration.
"""
struct Observation
    x::Float64
    y::Float64
end

"""
    SourceIdentificationProblem

Complete problem specification for source identification.
"""
struct SourceIdentificationProblem
    # Physics
    vx::Float64
    vy::Float64
    D::Float64
    domain::NTuple{4, Float64}  # (x_min, x_max, y_min, y_max)
    c_inflow::Float64

    # Sources (can be multiple)
    sources::Vector{GaussianSource}

    # Observations
    observations::Vector{Observation}
    noise_std::Float64
    noise_seed::Int
end

"""
    evaluate_source(prob, x, y)

Evaluate the total source field at point (x, y).
"""
function evaluate_source(prob::SourceIdentificationProblem, x, y)
    s = 0.0
    for src in prob.sources
        r² = (x - src.x)^2 + (y - src.y)^2
        s += src.strength * exp(-r² / (2 * src.width^2))
    end
    return s
end

"""
    load_problem(path::String) -> SourceIdentificationProblem

Load a problem definition from a TOML file.
"""
function load_problem(path::String)
    data = TOML.parsefile(path)

    # Physics
    phys = data["physics"]
    vx = get(phys, "vx", 1.0)
    vy = get(phys, "vy", 0.0)
    D = get(phys, "D", 0.05)
    domain_arr = get(phys, "domain", [0.0, 1.0, 0.0, 1.0])
    domain = tuple(domain_arr...)
    c_inflow = get(phys, "c_inflow", 0.0)

    # Sources
    sources = GaussianSource[]
    if haskey(data, "sources")
        for src in data["sources"]
            push!(sources, GaussianSource(
                src["x"],
                src["y"],
                src["strength"],
                src["width"]
            ))
        end
    end

    # Observations
    observations = Observation[]
    if haskey(data, "observations")
        for obs in data["observations"]
            push!(observations, Observation(obs["x"], obs["y"]))
        end
    end

    # Noise
    noise = get(data, "noise", Dict())
    noise_std = get(noise, "std", 0.1)
    noise_seed = get(noise, "seed", 42)

    return SourceIdentificationProblem(
        vx, vy, D, domain, c_inflow,
        sources, observations,
        noise_std, noise_seed
    )
end

"""
    save_problem(path::String, prob::SourceIdentificationProblem)

Save a problem definition to a TOML file.
"""
function save_problem(path::String, prob::SourceIdentificationProblem)
    data = Dict{String, Any}()

    # Physics
    data["physics"] = Dict(
        "vx" => prob.vx,
        "vy" => prob.vy,
        "D" => prob.D,
        "domain" => collect(prob.domain),
        "c_inflow" => prob.c_inflow
    )

    # Sources
    data["sources"] = [
        Dict("x" => s.x, "y" => s.y, "strength" => s.strength, "width" => s.width)
        for s in prob.sources
    ]

    # Observations
    data["observations"] = [
        Dict("x" => o.x, "y" => o.y)
        for o in prob.observations
    ]

    # Noise
    data["noise"] = Dict(
        "std" => prob.noise_std,
        "seed" => prob.noise_seed
    )

    open(path, "w") do io
        TOML.print(io, data)
    end
end

"""
    observation_coords(prob::SourceIdentificationProblem)

Return observation coordinates as (xs, ys) tuple of vectors.
"""
function observation_coords(prob::SourceIdentificationProblem)
    xs = [o.x for o in prob.observations]
    ys = [o.y for o in prob.observations]
    return xs, ys
end

"""
    n_observations(prob::SourceIdentificationProblem)

Return the number of observations.
"""
n_observations(prob::SourceIdentificationProblem) = length(prob.observations)

"""
    n_sources(prob::SourceIdentificationProblem)

Return the number of sources.
"""
n_sources(prob::SourceIdentificationProblem) = length(prob.sources)

# Pretty printing
function Base.show(io::IO, prob::SourceIdentificationProblem)
    println(io, "SourceIdentificationProblem:")
    println(io, "  Physics: vx=$(prob.vx), vy=$(prob.vy), D=$(prob.D)")
    println(io, "  Domain: $(prob.domain)")
    println(io, "  Sources: $(n_sources(prob))")
    for (i, s) in enumerate(prob.sources)
        println(io, "    [$i] ($(s.x), $(s.y)), strength=$(s.strength), width=$(s.width)")
    end
    println(io, "  Observations: $(n_observations(prob))")
    println(io, "  Noise: std=$(prob.noise_std), seed=$(prob.noise_seed)")
end
