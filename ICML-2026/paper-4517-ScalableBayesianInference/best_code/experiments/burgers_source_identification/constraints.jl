"""
Nonlinear FVM constraint for 2D Burgers with source coupling.

PDE: ∂u/∂t + ∂(u²/2)/∂x + ∂(u²/2)/∂y = ν(∂²u/∂x² + ∂²u/∂y²) + s(x,y)

FVM per cell: du_int/dt + (F_E - F_W + F_N - F_S) - s_int = 0
"""

"""
Precompute cell corner indices for an Nx × Ny grid (column-major ordering).
Returns vector of (sw, se, nw, ne) tuples for each cell.
"""
function precompute_cell_corners(Nx, Ny)
    N_cells_x, N_cells_y = Nx - 1, Ny - 1
    corners = Vector{NTuple{4, Int}}(undef, N_cells_x * N_cells_y)
    for cj in 1:N_cells_y, ci in 1:N_cells_x
        k = (cj - 1) * N_cells_x + ci
        sw = (cj - 1) * Nx + ci
        se = (cj - 1) * Nx + (ci + 1)
        nw = cj * Nx + ci
        ne = cj * Nx + (ci + 1)
        corners[k] = (sw, se, nw, ne)
    end
    return corners
end

"""
    build_fvm_residual(u_layout, s_layout, n_u_per_t, n_u_total, n_t,
                        N_cells, Δx, Δy, ν, cell_corners; log_source=false)

Returns a closure `f(x_joint) -> residuals` computing FVM residuals
for all cells at all timesteps t ≥ 2.

Joint state: x_joint = [u_spacetime (n_u_total) ; s_spatial (n_s)]

If log_source=true, the source state contains g = log(s) (point evaluations only),
and the source integral is computed as ∫ exp(g) dA using corner averages.
"""
function build_fvm_residual(u_layout, s_layout,
        n_u_per_t::Int, n_u_total::Int, n_t::Int,
        N_cells::Int, Δx::Float64, Δy::Float64, ν::Float64,
        cell_corners::Vector{NTuple{4, Int}};
        log_source::Bool=false)

    # Precompute index ranges within u_layout (per-timestep offsets)
    u_range = indices(u_layout, :u)
    u_dx_range = indices(u_layout, :u_dx)
    u_dy_range = indices(u_layout, :u_dy)
    du_int_dt_range = indices(u_layout, :du_int_dt)

    # Source indices (at the end of the joint state)
    s_offset = n_u_total
    if log_source
        g_range = indices(s_layout, :g)  # point evaluations of log(s)
    else
        s_int_range = indices(s_layout, :s_int)
    end

    n_residuals = N_cells * (n_t - 1)

    function f_fvm(x)
        residuals = map(1:n_residuals) do idx
            t = div(idx - 1, N_cells) + 1  # 1-based: which constraint (maps to timestep t+1)
            k = mod(idx - 1, N_cells) + 1  # which cell

            # Offset into the spacetime u block for timestep (t+1)
            # t=1 → timestep 2 (the first constrained timestep)
            t_off = t * n_u_per_t

            sw, se, nw, ne = cell_corners[k]

            # u values at corners
            u_sw = x[t_off + u_range[sw]]
            u_se = x[t_off + u_range[se]]
            u_nw = x[t_off + u_range[nw]]
            u_ne = x[t_off + u_range[ne]]

            # u_dx at corners
            u_dx_sw = x[t_off + u_dx_range[sw]]
            u_dx_se = x[t_off + u_dx_range[se]]
            u_dx_nw = x[t_off + u_dx_range[nw]]
            u_dx_ne = x[t_off + u_dx_range[ne]]

            # u_dy at corners
            u_dy_sw = x[t_off + u_dy_range[sw]]
            u_dy_se = x[t_off + u_dy_range[se]]
            u_dy_nw = x[t_off + u_dy_range[nw]]
            u_dy_ne = x[t_off + u_dy_range[ne]]

            # du_int/dt for this cell at this timestep
            du_int_dt = x[t_off + du_int_dt_range[k]]

            # Advective fluxes: ∫ u²/2 · n dS (trapezoidal rule at corners)
            F_adv_E = 0.25 * (u_se^2 + u_ne^2) * Δy
            F_adv_W = 0.25 * (u_sw^2 + u_nw^2) * Δy
            F_adv_N = 0.25 * (u_nw^2 + u_ne^2) * Δx
            F_adv_S = 0.25 * (u_sw^2 + u_se^2) * Δx

            # Diffusive fluxes: -ν ∫ ∇u · n dS
            F_diff_E = ν * 0.5 * (u_dx_se + u_dx_ne) * Δy
            F_diff_W = ν * 0.5 * (u_dx_sw + u_dx_nw) * Δy
            F_diff_N = ν * 0.5 * (u_dy_nw + u_dy_ne) * Δx
            F_diff_S = ν * 0.5 * (u_dy_sw + u_dy_se) * Δx

            # Source integral (shared across all timesteps)
            if log_source
                # Log-GP: s = exp(g), integrate via corner average
                g_sw = x[s_offset + g_range[sw]]
                g_se = x[s_offset + g_range[se]]
                g_nw = x[s_offset + g_range[nw]]
                g_ne = x[s_offset + g_range[ne]]
                _softplus(x) = log(1 + exp(x))
                s_int_k = 0.25 * (_softplus(g_sw) + _softplus(g_se) + _softplus(g_nw) + _softplus(g_ne)) * Δx * Δy
            else
                s_int_k = x[s_offset + s_int_range[k]]
            end

            # FVM residual: du_int/dt + advection - diffusion - source = 0
            du_int_dt +
                (F_adv_E - F_adv_W + F_adv_N - F_adv_S) -
                (F_diff_E - F_diff_W + F_diff_N - F_diff_S) -
                s_int_k
        end
        return residuals
    end

    return f_fvm, n_residuals
end
