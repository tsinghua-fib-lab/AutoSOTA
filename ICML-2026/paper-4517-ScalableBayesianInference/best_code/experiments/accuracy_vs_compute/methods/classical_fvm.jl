"""
    Classical FVM Solver

Finite volume method with fully implicit Crank-Nicolson time stepping.
This is the non-probabilistic baseline that uses the same time discretization
as GP-FVM for fair comparison.
"""

using LinearAlgebra
using SparseArrays

# Include common problem/metrics definitions
include(joinpath(@__DIR__, "..", "problem.jl"))
include(joinpath(@__DIR__, "..", "metrics.jl"))

"""
    solve_classical_fvm(instance::ProblemInstance, N::Int; kwargs...)

Solve Burgers equation using classical FVM with fully implicit Crank-Nicolson.

Uses Newton iteration at each timestep to solve the nonlinear system, making it
directly comparable to GP-FVM which also solves nonlinear systems.

# Arguments
- `instance`: Problem instance with IC and reference solution
- `N`: Number of spatial grid points (cell centers = N-1 for FVM)

# Keyword arguments
- `n_timesteps`: Number of time steps (required for fair comparison)
- `newton_tol`: Convergence tolerance for Newton iteration (default: 1e-12)
- `max_newton_iter`: Maximum Newton iterations per timestep (default: 50)

# Returns
`SolutionResult` with solution and zero uncertainty (no UQ).
"""
function solve_classical_fvm(instance::ProblemInstance, N::Int;
                              n_timesteps::Int,
                              newton_tol::Float64=1e-12,
                              max_newton_iter::Int=50)
    problem = instance.problem
    ic = instance.ic

    # Track timing
    t_start = time()

    # Memory tracking
    GC.gc()
    mem_before = Base.gc_live_bytes()

    # -------------------------------------------------------------------------
    # Grid setup - cell-centered FVM
    # -------------------------------------------------------------------------
    (; x_min, x_max, T_end, ν, u_left, u_right) = problem

    # Grid points (cell interfaces)
    xs_interfaces = range(x_min, x_max, length=N)
    Δx = xs_interfaces[2] - xs_interfaces[1]

    # Cell centers
    N_cells = N - 1
    xs = [(xs_interfaces[i] + xs_interfaces[i+1]) / 2 for i in 1:N_cells]

    # Initialize solution at cell centers
    u = [ic(x) for x in xs]

    # -------------------------------------------------------------------------
    # Time stepping setup
    # -------------------------------------------------------------------------
    n_t = n_timesteps
    Δt = T_end / n_t

    # Storage for solution at output times
    output_interval = max(1, n_t ÷ 20)
    output_times = 1:output_interval:n_t
    n_outputs = length(output_times)

    # Time at which each output is stored: t = n * Δt after n timesteps
    # Note: solutions[:, 1] is IC at t=0, solutions[:, 2] is after first output step
    ts = [0.0]  # Start with IC time
    append!(ts, [i * Δt for i in output_times])
    if ts[end] != T_end
        push!(ts, T_end)  # Ensure final time is included
    end

    # Solution storage
    solutions = zeros(N_cells, length(ts))
    solutions[:, 1] = u

    # -------------------------------------------------------------------------
    # Fully implicit Crank-Nicolson time integration
    # Solve: u^{n+1} - u^n + 0.5*Δt*(RHS(u^{n+1}) + RHS(u^n)) = 0
    # where RHS = (F_right - F_left)/Δx - ν*(u_ip1 - 2u + u_im1)/Δx²
    # -------------------------------------------------------------------------
    u_new = similar(u)
    output_idx = 2

    # Precompute RHS at current time (will be reused)
    rhs_old = zeros(N_cells)

    for n in 1:n_t
        # Compute RHS at old time level
        compute_rhs!(rhs_old, u, N_cells, Δx, ν, u_left, u_right)

        # Initial guess: explicit Euler step
        u_new .= u .- Δt .* rhs_old

        # Newton iteration
        residual = zeros(N_cells)
        rhs_new = zeros(N_cells)

        for iter in 1:max_newton_iter
            # Compute RHS at new time level
            compute_rhs!(rhs_new, u_new, N_cells, Δx, ν, u_left, u_right)

            # Residual: r = u_new - u_old + 0.5*Δt*(rhs_new + rhs_old)
            @. residual = u_new - u + 0.5 * Δt * (rhs_new + rhs_old)

            # Check convergence
            res_norm = norm(residual)
            if res_norm < newton_tol
                break
            end

            # Full Jacobian: J = I + 0.5*Δt * d(RHS)/du
            J = compute_jacobian(u_new, N_cells, Δx, Δt, ν, u_left, u_right)

            # Newton update: δu = -J \ r
            δu = J \ (-residual)
            u_new .+= δu

            # Check for convergence based on update size
            if norm(δu) < newton_tol
                break
            end
        end

        u .= u_new

        # Store output
        if n in output_times && output_idx <= length(ts)
            solutions[:, output_idx] = u
            output_idx += 1
        end
    end

    # Final solution
    if output_idx <= length(ts)
        solutions[:, end] = u
    end

    # -------------------------------------------------------------------------
    # Prepare output
    # -------------------------------------------------------------------------
    GC.gc()
    mem_after = Base.gc_live_bytes()
    peak_memory_mb = max(0.0, (mem_after - mem_before) / 1e6)

    wall_time_s = time() - t_start

    # Classical FVM has no uncertainty
    std_matrix = zeros(size(solutions))

    return SolutionResult(
        xs, ts,
        solutions, std_matrix,
        wall_time_s, peak_memory_mb,
        0, N_cells,  # No Cholesky, DOF = number of cells
        "classical_fvm", N
    )
end

"""
    compute_rhs!(rhs, u, N_cells, Δx, ν, u_left, u_right)

Compute the RHS of the Burgers equation: RHS = (F_right - F_left)/Δx - ν*Δ²u/Δx²
where F is the Godunov flux for advection.

Note: RHS has opposite sign convention so that du/dt = -RHS (flux form).
"""
function compute_rhs!(rhs, u, N_cells, Δx, ν, u_left, u_right)
    # Ghost cells for Dirichlet BCs
    u_ghost_left = 2 * u_left - u[1]
    u_ghost_right = 2 * u_right - u[N_cells]

    for i in 1:N_cells
        # Get neighbor values
        u_im1 = (i == 1) ? u_ghost_left : u[i-1]
        u_ip1 = (i == N_cells) ? u_ghost_right : u[i+1]

        # Advection: Godunov flux difference
        F_left = godunov_flux(u_im1, u[i])
        F_right = godunov_flux(u[i], u_ip1)
        advection = (F_right - F_left) / Δx

        # Diffusion: central difference (negative because it's on RHS)
        diffusion = -ν * (u_ip1 - 2*u[i] + u_im1) / Δx^2

        rhs[i] = advection + diffusion
    end
end

"""
    compute_jacobian(u, N_cells, Δx, Δt, ν, u_left, u_right)

Compute the Jacobian of the residual: J = I + 0.5*Δt * d(RHS)/du

The Jacobian is tridiagonal due to the local stencil.
"""
function compute_jacobian(u, N_cells, Δx, Δt, ν, u_left, u_right)
    # Ghost cells
    u_ghost_left = 2 * u_left - u[1]
    u_ghost_right = 2 * u_right - u[N_cells]

    # Build tridiagonal Jacobian
    # J[i,i-1], J[i,i], J[i,i+1]
    dl = zeros(N_cells - 1)  # lower diagonal
    d = zeros(N_cells)       # main diagonal
    du = zeros(N_cells - 1)  # upper diagonal

    α = 0.5 * Δt / Δx        # advection coefficient
    β = 0.5 * Δt * ν / Δx^2  # diffusion coefficient

    for i in 1:N_cells
        u_im1 = (i == 1) ? u_ghost_left : u[i-1]
        u_ip1 = (i == N_cells) ? u_ghost_right : u[i+1]

        # Derivative of Godunov flux w.r.t. u[i]
        # F_left = godunov_flux(u[i-1], u[i]) -> dF_left/du[i]
        # F_right = godunov_flux(u[i], u[i+1]) -> dF_right/du[i]
        dF_left_dui = godunov_flux_derivative_right(u_im1, u[i])
        dF_right_dui = godunov_flux_derivative_left(u[i], u_ip1)

        # d(RHS_i)/d(u_i) = (dF_right/du_i - dF_left/du_i)/Δx + 2ν/Δx²
        d[i] = 1.0 + α * (dF_right_dui - dF_left_dui) + 2 * β

        # Off-diagonal terms
        if i > 1
            # d(RHS_i)/d(u_{i-1}) from F_left and diffusion
            dF_left_duim1 = godunov_flux_derivative_left(u[i-1], u[i])
            dl[i-1] = α * (-dF_left_duim1) - β
        end

        if i < N_cells
            # d(RHS_i)/d(u_{i+1}) from F_right and diffusion
            dF_right_duip1 = godunov_flux_derivative_right(u[i], u[i+1])
            du[i] = α * dF_right_duip1 - β
        end
    end

    return Tridiagonal(dl, d, du)
end

"""
    godunov_flux_derivative_left(u_L, u_R)

Derivative of Godunov flux with respect to the left state u_L.
"""
function godunov_flux_derivative_left(u_L, u_R)
    if u_L >= u_R
        # Shock or compression
        if u_L + u_R >= 0
            return u_L  # d(0.5*u_L^2)/du_L = u_L
        else
            return 0.0  # flux uses u_R
        end
    else
        # Rarefaction
        if u_L >= 0
            return u_L  # d(0.5*u_L^2)/du_L = u_L
        elseif u_R <= 0
            return 0.0  # flux uses u_R
        else
            return 0.0  # sonic point
        end
    end
end

"""
    godunov_flux_derivative_right(u_L, u_R)

Derivative of Godunov flux with respect to the right state u_R.
"""
function godunov_flux_derivative_right(u_L, u_R)
    if u_L >= u_R
        # Shock or compression
        if u_L + u_R >= 0
            return 0.0  # flux uses u_L
        else
            return u_R  # d(0.5*u_R^2)/du_R = u_R
        end
    else
        # Rarefaction
        if u_L >= 0
            return 0.0  # flux uses u_L
        elseif u_R <= 0
            return u_R  # d(0.5*u_R^2)/du_R = u_R
        else
            return 0.0  # sonic point
        end
    end
end

"""
    godunov_flux(u_L, u_R)

Godunov numerical flux for Burgers equation: F(u) = u²/2

For the Riemann problem with left state u_L and right state u_R:
- If u_L ≥ u_R: shock or compression wave
- If u_L < u_R: rarefaction wave
"""
function godunov_flux(u_L, u_R)
    if u_L >= u_R
        # Shock or compression: use upwind based on shock speed
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
            return 0.0  # Sonic point in rarefaction
        end
    end
end
