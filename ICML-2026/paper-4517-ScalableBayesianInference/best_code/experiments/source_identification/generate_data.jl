"""
Generate ground truth and observations for source identification experiments.

Takes a problem TOML file and generates:
- Ground truth concentration field (from forward FVM solve)
- Ground truth source cell integrals
- Noisy observations at specified locations

Both GP-FVM and PINN read from this same data file for fair comparison.

Usage:
    julia --project=../.. generate_data.jl --problem problems/default.toml
    julia --project=../.. generate_data.jl --problem problems/two_sources.toml --nx 31
"""

using LinearAlgebra, SparseArrays
using Random
using ArgParse
using NPZ

include("problem.jl")

# ------------------------------------------------------------------------------
# Grid Setup
# ------------------------------------------------------------------------------

function setup_2d_grid(Nx, Ny, domain)
    x_min, x_max, y_min, y_max = domain
    xs = range(x_min, x_max, length=Nx)
    ys = range(y_min, y_max, length=Ny)
    return collect(xs), collect(ys)
end

# ------------------------------------------------------------------------------
# Forward Solver
# ------------------------------------------------------------------------------

"""
Solve the forward advection-diffusion problem to get ground-truth concentration.
Uses standard node-centered FVM with upwind advection.
"""
function solve_forward_problem(xs, ys, prob::SourceIdentificationProblem)
    Nx, Ny = length(xs), length(ys)
    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]

    n_cells_x, n_cells_y = Nx - 1, Ny - 1

    # Compute true source cell integrals
    s_int_true = zeros(n_cells_x, n_cells_y)
    for cj in 1:n_cells_y
        cy = 0.5 * (ys[cj] + ys[cj+1])
        for ci in 1:n_cells_x
            cx = 0.5 * (xs[ci] + xs[ci+1])
            s_int_true[ci, cj] = evaluate_source(prob, cx, cy) * Δx * Δy
        end
    end

    # Build linear system for node values
    n_nodes = Nx * Ny
    node_idx(i, j) = (j - 1) * Nx + i

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    b = zeros(n_nodes)

    for j in 1:Ny
        for i in 1:Nx
            idx = node_idx(i, j)

            if i == 1
                # Left boundary: Dirichlet
                push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
                b[idx] = prob.c_inflow
            elseif i == Nx
                # Right boundary: Neumann
                push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
                push!(rows, idx); push!(cols, node_idx(i-1, j)); push!(vals, -1.0)
                b[idx] = 0.0
            elseif j == 1
                # Bottom boundary: Neumann
                push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
                push!(rows, idx); push!(cols, node_idx(i, j+1)); push!(vals, -1.0)
                b[idx] = 0.0
            elseif j == Ny
                # Top boundary: Neumann
                push!(rows, idx); push!(cols, idx); push!(vals, 1.0)
                push!(rows, idx); push!(cols, node_idx(i, j-1)); push!(vals, -1.0)
                b[idx] = 0.0
            else
                # Interior: FVM conservation
                adv_coef_center = prob.vx * Δy
                adv_coef_left = -prob.vx * Δy
                push!(rows, idx); push!(cols, idx); push!(vals, adv_coef_center)
                push!(rows, idx); push!(cols, node_idx(i-1, j)); push!(vals, adv_coef_left)

                diff_coef = prob.D / Δx * Δy
                diff_coef_y = prob.D / Δy * Δx
                center_diff = 2 * diff_coef + 2 * diff_coef_y
                push!(rows, idx); push!(cols, idx); push!(vals, center_diff)
                push!(rows, idx); push!(cols, node_idx(i+1, j)); push!(vals, -diff_coef)
                push!(rows, idx); push!(cols, node_idx(i-1, j)); push!(vals, -diff_coef)
                push!(rows, idx); push!(cols, node_idx(i, j+1)); push!(vals, -diff_coef_y)
                push!(rows, idx); push!(cols, node_idx(i, j-1)); push!(vals, -diff_coef_y)

                source = 0.0
                for (ci, cj) in [(i-1, j-1), (i, j-1), (i-1, j), (i, j)]
                    if 1 <= ci <= n_cells_x && 1 <= cj <= n_cells_y
                        source += 0.25 * s_int_true[ci, cj]
                    end
                end
                b[idx] = source
            end
        end
    end

    A = sparse(rows, cols, vals, n_nodes, n_nodes)
    c_vec = A \ b
    c_true = reshape(c_vec, Nx, Ny)

    return c_true, s_int_true
end

# ------------------------------------------------------------------------------
# Observation Generation
# ------------------------------------------------------------------------------

function generate_observations(xs, ys, c_true, prob::SourceIdentificationProblem)
    Nx, Ny = length(xs), length(ys)
    Random.seed!(prob.noise_seed)

    obs_xs, obs_ys = observation_coords(prob)
    n_obs = length(obs_xs)

    # Find nearest grid points
    obs_ix = [argmin(abs.(xs .- ox)) for ox in obs_xs]
    obs_iy = [argmin(abs.(ys .- oy)) for oy in obs_ys]

    # Get true concentration at observation points
    true_c_obs = [c_true[ix, iy] for (ix, iy) in zip(obs_ix, obs_iy)]

    # Add noise
    noisy_obs = true_c_obs .+ prob.noise_std * randn(n_obs)

    return obs_xs, obs_ys, true_c_obs, noisy_obs
end

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(description = "Generate ground truth data for source identification")

    @add_arg_table! s begin
        "--problem", "-p"
            help = "Path to problem TOML file"
            arg_type = String
            required = true
        "--nx"
            help = "Grid points in x (for ground truth solve)"
            arg_type = Int
            default = 36
        "--ny"
            help = "Grid points in y (for ground truth solve)"
            arg_type = Int
            default = 36
        "--output", "-o"
            help = "Output NPZ file (default: data/<problem_name>.npz)"
            arg_type = String
            default = ""
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    println("=" ^ 60)
    println("Generating ground truth data")
    println("=" ^ 60)

    # Load problem
    prob = load_problem(args["problem"])
    println(prob)

    # Setup grid
    Nx, Ny = args["nx"], args["ny"]
    xs, ys = setup_2d_grid(Nx, Ny, prob.domain)
    println("\nGrid: $(Nx) × $(Ny)")

    # Solve forward problem
    println("Solving forward problem...")
    c_true, s_int_true = solve_forward_problem(xs, ys, prob)
    println("  Concentration range: [$(round(minimum(c_true), digits=4)), $(round(maximum(c_true), digits=4))]")
    println("  Total source: $(round(sum(s_int_true), digits=4))")

    # Compute source field on grid nodes (for PINN field-level comparison)
    s_true = zeros(Nx, Ny)
    for j in 1:Ny
        for i in 1:Nx
            s_true[i, j] = evaluate_source(prob, xs[i], ys[j])
        end
    end
    println("  Source field range: [$(round(minimum(s_true), digits=4)), $(round(maximum(s_true), digits=4))]")

    # Generate observations
    obs_xs, obs_ys, true_c_obs, noisy_obs = generate_observations(xs, ys, c_true, prob)
    println("  Generated $(length(noisy_obs)) observations")

    # Prepare output
    problem_name = splitext(basename(args["problem"]))[1]
    output_path = args["output"]
    if isempty(output_path)
        output_path = joinpath(dirname(args["problem"]), "..", "data", "$(problem_name).npz")
    end

    # Create output directory
    mkpath(dirname(output_path))

    # Collect source parameters (for PINN which needs them as targets)
    source_xs = [s.x for s in prob.sources]
    source_ys = [s.y for s in prob.sources]
    source_strengths = [s.strength for s in prob.sources]
    source_widths = [s.width for s in prob.sources]

    # Save to NPZ
    output = Dict{String, Any}(
        # Grid
        "xs" => xs,
        "ys" => ys,

        # Ground truth fields
        "c_true" => c_true,
        "s_int_true" => s_int_true,
        "s_true" => s_true,  # Source field on grid nodes

        # Source parameters (for reference/PINN targets)
        "source_x" => source_xs,
        "source_y" => source_ys,
        "source_strength" => source_strengths,
        "source_width" => source_widths,
        "n_sources" => length(prob.sources),

        # Observations
        "obs_x" => obs_xs,
        "obs_y" => obs_ys,
        "obs_c" => noisy_obs,
        "obs_c_true" => true_c_obs,
        "n_obs" => length(noisy_obs),

        # Physics
        "vx" => prob.vx,
        "vy" => prob.vy,
        "D" => prob.D,
        "c_inflow" => prob.c_inflow,
        "domain" => collect(prob.domain),

        # Noise
        "noise_std" => prob.noise_std,
        "noise_seed" => prob.noise_seed,
    )

    npzwrite(output_path, output)
    println("\nSaved: $output_path")

    # Print summary
    println("\nContents:")
    for (key, val) in sort(collect(output), by=x->x[1])
        if val isa AbstractArray
            println("  $key: $(size(val))")
        else
            println("  $key: $val")
        end
    end

    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
