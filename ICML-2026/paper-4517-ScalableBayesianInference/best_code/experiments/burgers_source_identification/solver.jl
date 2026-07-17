"""
GP-FVM solver for 2D Burgers source identification.

Builds joint spacetime GMRF for u + spatial source s, applies IC/BC/observations,
then solves via Gauss-Newton (gaussian_approximation).
"""

function solve_burgers_source_id(prob::BurgersSourceProblem, obs_data;
        ρ::Float64=2.0,
        lengthscale_u::Float64=0.15,
        lengthscale_s::Float64=0.15,
        output_scale::Float64=1.0,
        source_amplitude::Float64=1.0,
        smoothness::Int=2,
        max_iter::Int=50,
        σ_fvm::Float64=1e-4,
        log_source::Bool=false,
        n_source_samples::Int=0,
        source_prior_perturbation::Union{Nothing, Vector{Float64}}=nothing,
        joint_prior_perturbation::Union{Nothing, Vector{Float64}}=nothing,
        g_prior_mean::Float64=0.0,
        verbose::Bool=true)

    timings = Dict{String, Float64}()

    xs = collect(range(prob.domain[1], prob.domain[2], length=obs_data.Nx))
    ys = collect(range(prob.domain[3], prob.domain[4], length=obs_data.Ny))
    Nx, Ny = obs_data.Nx, obs_data.Ny
    N_grid = Nx * Ny
    N_cells_x, N_cells_y = Nx - 1, Ny - 1
    N_cells = N_cells_x * N_cells_y
    Δx = xs[2] - xs[1]; Δy = ys[2] - ys[1]
    n_t = round(Int, prob.T_end / prob.Δt)

    verbose && println("Grid: $(Nx)×$(Ny), T=$(n_t) steps, Δt=$(prob.Δt), ν=$(prob.ν)")

    # --- Build spatial precisions ---
    verbose && println("Building spatial precisions...")
    timings["prec_u"] = @elapsed approx_u = build_u_precision(xs, ys;
        smoothness=smoothness, lengthscale=lengthscale_u, ρ=ρ)
    timings["prec_s"] = @elapsed approx_s = build_s_precision(xs, ys;
        smoothness=smoothness, lengthscale=lengthscale_s, log_source=log_source)

    n_u_spatial = approx_u.info.n
    n_s = approx_s.info.n

    verbose && println("  u spatial DOF: $n_u_spatial, s DOF: $n_s")
    verbose && println("  u fill: $(round(approx_u.info.fill_pct, digits=1))%")

    # --- Build state layout ---
    u_state_layout = build_u_state_layout(N_grid, N_cells)
    n_u_per_t = u_state_layout.total  # per-timestep state size (with IWP)

    # --- Build initial state ---
    Q_u_spatial = sparse(approx_u.Q) / output_scale
    Q_state0 = [Q_u_spatial spzeros(size(Q_u_spatial)...);
                spzeros(size(Q_u_spatial)...) Q_u_spatial]
    N_state = size(Q_state0, 1)
    x0 = GMRF(zeros(N_state), Q_state0)

    # --- Apply initial conditions ---
    verbose && println("Applying initial conditions...")
    grid_coords = [(x, y) for y in ys for x in xs]  # column-major
    cell_bounds = [((xs[i], xs[i+1]), (ys[j], ys[j+1]))
                   for j in 1:N_cells_y for i in 1:N_cells_x]

    u_ic = [u_initial(x, y) for (x, y) in grid_coords]
    # Approximate derivatives and integrals from IC
    u_dx_ic = zeros(N_grid)
    u_dy_ic = zeros(N_grid)
    for j in 1:Ny, i in 1:Nx
        k = (j-1)*Nx + i
        # Central differences (one-sided at boundaries)
        if i > 1 && i < Nx
            u_dx_ic[k] = (u_initial(xs[i+1], ys[j]) - u_initial(xs[i-1], ys[j])) / (2Δx)
        elseif i == 1
            u_dx_ic[k] = (u_initial(xs[2], ys[j]) - u_initial(xs[1], ys[j])) / Δx
        else
            u_dx_ic[k] = (u_initial(xs[Nx], ys[j]) - u_initial(xs[Nx-1], ys[j])) / Δx
        end
        if j > 1 && j < Ny
            u_dy_ic[k] = (u_initial(xs[i], ys[j+1]) - u_initial(xs[i], ys[j-1])) / (2Δy)
        elseif j == 1
            u_dy_ic[k] = (u_initial(xs[i], ys[2]) - u_initial(xs[i], ys[1])) / Δy
        else
            u_dy_ic[k] = (u_initial(xs[i], ys[Ny]) - u_initial(xs[i], ys[Ny-1])) / Δy
        end
    end
    u_int_ic = [u_initial(0.5*(xb[1]+xb[2]), 0.5*(yb[1]+yb[2])) * (xb[2]-xb[1]) * (yb[2]-yb[1])
                for (xb, yb) in cell_bounds]

    x0_ic = prescribe_indices(x0, indices(u_state_layout, :u), u_ic)
    x0_ic = prescribe_indices(x0_ic, indices(u_state_layout, :u_dx), u_dx_ic)
    x0_ic = prescribe_indices(x0_ic, indices(u_state_layout, :u_dy), u_dy_ic)
    x0_ic = prescribe_indices(x0_ic, indices(u_state_layout, :u_int), u_int_ic)

    # --- Build spacetime GMRF for u ---
    verbose && println("Building spacetime GMRF ($(n_t+1) timesteps)...")
    sde = IWPSDE(1.0)
    A_t, Σ_noise_t = discretize_vanloan(sde, prob.Δt)
    Q_noise_t = inv(Σ_noise_t)
    Q_noise_t = 0.5 * (Q_noise_t + Q_noise_t')

    A_kr = kronecker(A_t, sparse(I, n_u_spatial, n_u_spatial))
    Q_noise_kr = kronecker(Q_noise_t, Q_u_spatial)

    n_t_total = n_t + 1  # total timesteps including IC
    timings["joint_gmrf"] = @elapsed begin
        ssm = ConstantLinearGaussianSSM(x0_ic, A_kr, Q_noise_kr)
        x_u_joint = GPFiniteVolume.joint_gmrf(ssm, n_t_total)
    end
    n_u_total = length(mean(x_u_joint))
    verbose && println("  u spacetime DOF: $n_u_total ($(n_t_total) timesteps × $(n_u_per_t))")

    # --- Augment with source ---
    verbose && println("Augmenting with source field...")
    Q_s_scaled = sparse(approx_s.Q) / (source_amplitude^2 * output_scale)
    Q_joint = blockdiag(sparse(GaussianMarkovRandomFields.precision_map(x_u_joint)), Q_s_scaled)
    s_base_mean = fill(g_prior_mean, n_s)
    s_prior_mean = isnothing(source_prior_perturbation) ? s_base_mean : s_base_mean .+ source_prior_perturbation
    μ_joint = vcat(mean(x_u_joint), s_prior_mean)
    if !isnothing(joint_prior_perturbation)
        μ_joint .+= joint_prior_perturbation
    end
    x_joint = GMRF(μ_joint, Symmetric(Q_joint))
    n_joint = length(μ_joint)

    verbose && println("  Joint DOF: $n_joint (u:$n_u_total + s:$n_s)")

    # --- Apply boundary conditions ---
    verbose && println("Applying boundary conditions...")
    # Dirichlet u=0 on all boundaries, for all timesteps t >= 1 (batched)
    boundary_nodes = Int[]
    for j in 1:Ny, i in 1:Nx
        if i == 1 || i == Nx || j == 1 || j == Ny
            push!(boundary_nodes, (j-1)*Nx + i)
        end
    end
    u_eval_range = indices(u_state_layout, :u)

    all_bc_indices = Int[]
    for t in 1:n_t
        t_off = t * n_u_per_t
        append!(all_bc_indices, [t_off + u_eval_range[bn] for bn in boundary_nodes])
    end
    x_joint = prescribe_indices(x_joint, all_bc_indices, zeros(length(all_bc_indices)))

    # --- Apply observations ---
    verbose && println("Applying $(length(obs_data.obs_values)) observations...")
    obs_indices = Int[]
    for (i, (t, k)) in enumerate(zip(obs_data.obs_times, obs_data.obs_spatial_idx))
        # t is 1-indexed into u_history (t=1 is IC, t=2 is first step, etc.)
        # In the GMRF, timestep index is 0-based: t-1
        t_gmrf = t - 1  # 0-based timestep in GMRF
        t_off = t_gmrf * n_u_per_t
        ix, iy = obs_data.obs_ix[k], obs_data.obs_iy[k]
        node = (iy - 1) * Nx + ix
        push!(obs_indices, t_off + u_eval_range[node])
    end
    x_joint = prescribe_indices(x_joint, obs_indices, obs_data.obs_values;
        noise_std=obs_data.noise_std)

    # --- Build FVM constraint ---
    verbose && println("Building FVM constraints...")
    cell_corners = precompute_cell_corners(Nx, Ny)
    s_layout = approx_s.layout

    f_fvm, n_residuals = build_fvm_residual(
        u_state_layout, s_layout,
        n_u_per_t, n_u_total, n_t_total,
        N_cells, Δx, Δy, prob.ν, cell_corners;
        log_source=log_source)

    verbose && println("  Residuals: $n_residuals")

    # --- Solve ---
    verbose && println("Solving (Gauss-Newton, max_iter=$max_iter)...")
    fvm_model = NonlinearLeastSquaresModel(f_fvm, n_joint)
    lik_fvm = fvm_model(zeros(n_residuals); σ=σ_fvm)

    timings["solve"] = @elapsed x_solution = gaussian_approximation(
        x_joint, lik_fvm; verbose=verbose, max_iter=max_iter)

    # --- Extract results ---
    verbose && println("Extracting results...")
    μ = mean(x_solution)
    σ_vec = std(x_solution)

    # u at each timestep
    u_mean = zeros(Nx, Ny, n_t + 1)
    u_std_arr = zeros(Nx, Ny, n_t + 1)
    for t in 0:n_t
        t_off = t * n_u_per_t
        u_mean[:, :, t+1] = reshape(μ[t_off .+ u_eval_range], Nx, Ny)
        u_std_arr[:, :, t+1] = reshape(σ_vec[t_off .+ u_eval_range], Nx, Ny)
    end

    # Source
    if log_source
        g_range = indices(s_layout, :g)
        g_mean = reshape(μ[n_u_total .+ g_range], Nx, Ny)
        g_std_arr = reshape(σ_vec[n_u_total .+ g_range], Nx, Ny)
        _softplus(x) = log(1 + exp(x))

        if n_source_samples > 0
            # Sample-based uncertainty: propagate through softplus by sampling
            verbose && println("Computing source std via $n_source_samples posterior samples...")
            s_samples = zeros(Nx * Ny, n_source_samples)
            g_idx = n_u_total .+ g_range
            for i in 1:n_source_samples
                x_sample = rand(x_solution)
                s_samples[:, i] = _softplus.(x_sample[g_idx])
            end
            s_mean = reshape(Statistics.mean(s_samples, dims=2)[:], Nx, Ny)
            s_std_arr = reshape(Statistics.std(s_samples, dims=2)[:], Nx, Ny)
        else
            # Delta method (fast but degenerates where sigmoid ≈ 0)
            _sigmoid(x) = 1 / (1 + exp(-x))
            s_mean = _softplus.(g_mean)
            s_std_arr = _sigmoid.(g_mean) .* g_std_arr
        end
    else
        s_eval_range = indices(s_layout, :s)
        s_mean = reshape(μ[n_u_total .+ s_eval_range], Nx, Ny)
        s_std_arr = reshape(σ_vec[n_u_total .+ s_eval_range], Nx, Ny)
    end

    timings["total"] = sum(values(timings))

    return (;
        u_mean, u_std=u_std_arr,
        s_mean, s_std=s_std_arr,
        xs, ys,
        info = (; n_joint, n_u_total, n_s, n_residuals,
                  fill_u=approx_u.info.fill_pct, timings,
                  u_state_layout, s_layout=approx_s.layout),
        gmrf = x_solution,
    )
end
