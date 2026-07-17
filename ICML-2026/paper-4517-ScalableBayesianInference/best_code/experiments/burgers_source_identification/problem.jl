"""
Problem definition and ground truth generation for 2D Burgers source identification.
"""

# Reuse GaussianSource from source identification
struct GaussianSource
    x::Float64
    y::Float64
    strength::Float64
    width::Float64
end

struct BurgersSourceProblem
    ν::Float64
    domain::NTuple{4, Float64}
    sources::Vector{GaussianSource}
    T_end::Float64
    Δt::Float64
end

function evaluate_source(prob::BurgersSourceProblem, x, y)
    s = 0.0
    for src in prob.sources
        r² = (x - src.x)^2 + (y - src.y)^2
        s += src.strength * exp(-r² / (2 * src.width^2))
    end
    return s
end

function random_problem(rng::AbstractRNG;
        ν=0.01, T_end=0.5, Δt=0.05,
        domain=(0.0, 1.0, 0.0, 1.0))
    sources = [GaussianSource(
        0.2 + 0.6 * rand(rng),
        0.2 + 0.6 * rand(rng),
        0.5 + 4.5 * rand(rng),   # positive amplitudes in [0.5, 5]
        0.05 + 0.10 * rand(rng),
    ) for _ in 1:2]
    return BurgersSourceProblem(ν, domain, sources, T_end, Δt)
end

# Initial condition: Gaussian bump
u_initial(x, y; x0=0.5, y0=0.5, σ=0.2, A=1.0) =
    A * exp(-((x - x0)^2 + (y - y0)^2) / (2σ^2))

"""
    forward_solve(prob, xs, ys) -> (u_history, s_true)

Solve 2D Burgers with source using explicit FVM.
Returns u_history[i, j, t] and s_true[i, j].
"""
function forward_solve(prob::BurgersSourceProblem, xs, ys)
    Nx, Ny = length(xs), length(ys)
    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]
    n_t = round(Int, prob.T_end / prob.Δt)

    # Source field on grid
    s_true = [evaluate_source(prob, xs[i], ys[j]) for i in 1:Nx, j in 1:Ny]

    # Initial condition
    u = [u_initial(xs[i], ys[j]) for i in 1:Nx, j in 1:Ny]
    u_history = zeros(Nx, Ny, n_t + 1)
    u_history[:, :, 1] = u

    # CFL sub-stepping
    for step in 1:n_t
        dt_remaining = prob.Δt
        while dt_remaining > 1e-14
            u_max = max(maximum(abs.(u)), 1e-8)
            dt_cfl = 0.25 * min(Δx, Δy) / u_max  # CFL < 0.5
            dt_diff = 0.125 * min(Δx, Δy)^2 / prob.ν  # diffusion stability
            dt_sub = min(dt_remaining, dt_cfl, dt_diff)

            u_new = copy(u)
            for j in 2:Ny-1, i in 2:Nx-1
                # Upwind advection: u ∂u/∂x + u ∂u/∂y
                # x-direction
                if u[i, j] >= 0
                    du_dx = (u[i, j] - u[i-1, j]) / Δx
                else
                    du_dx = (u[i+1, j] - u[i, j]) / Δx
                end
                # y-direction
                if u[i, j] >= 0
                    du_dy = (u[i, j] - u[i, j-1]) / Δy
                else
                    du_dy = (u[i, j+1] - u[i, j]) / Δy
                end
                advection = u[i, j] * du_dx + u[i, j] * du_dy

                # Central diffusion
                d2u_dx2 = (u[i+1, j] - 2u[i, j] + u[i-1, j]) / Δx^2
                d2u_dy2 = (u[i, j+1] - 2u[i, j] + u[i, j-1]) / Δy^2
                diffusion = prob.ν * (d2u_dx2 + d2u_dy2)

                u_new[i, j] = u[i, j] + dt_sub * (-advection + diffusion + s_true[i, j])
            end

            # Dirichlet BCs: u = 0 on all boundaries
            u_new[1, :] .= 0; u_new[Nx, :] .= 0
            u_new[:, 1] .= 0; u_new[:, Ny] .= 0

            u = u_new
            dt_remaining -= dt_sub
        end
        u_history[:, :, step + 1] = u
    end

    return u_history, s_true
end

"""
    generate_observations(rng, u_history, xs, ys; n_obs, obs_every, noise_std)

Generate noisy observations at random spatial locations, every `obs_every` timesteps.
"""
function generate_observations(rng::AbstractRNG, u_history, xs, ys;
        n_obs::Int=20, obs_every::Int=3, noise_std::Float64=0.05)

    Nx, Ny = length(xs), length(ys)
    n_t_total = size(u_history, 3)

    # Fixed spatial locations (avoid boundaries)
    obs_x = [0.1 + 0.8 * rand(rng) for _ in 1:n_obs]
    obs_y = [0.1 + 0.8 * rand(rng) for _ in 1:n_obs]

    # Nearest grid points
    obs_ix = [argmin(abs.(xs .- ox)) for ox in obs_x]
    obs_iy = [argmin(abs.(ys .- oy)) for oy in obs_y]

    # Observation timesteps (1-indexed, skip t=1 which is IC)
    obs_timesteps = collect(1+obs_every:obs_every:n_t_total)

    # Collect observations
    obs_values = Float64[]
    obs_times = Int[]
    obs_spatial_idx = Int[]  # which of the n_obs locations

    for t in obs_timesteps
        for k in 1:n_obs
            push!(obs_values, u_history[obs_ix[k], obs_iy[k], t] + noise_std * randn(rng))
            push!(obs_times, t)
            push!(obs_spatial_idx, k)
        end
    end

    return (;
        obs_x, obs_y, obs_ix, obs_iy,
        obs_values, obs_times, obs_spatial_idx,
        obs_timesteps, noise_std, n_obs,
    )
end
