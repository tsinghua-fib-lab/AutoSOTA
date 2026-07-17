"""
GP-FVM solver for advection-diffusion source identification.

Uses sparse Gaussian Process Finite Volume Method with exact face integrals.
Follows the same approach as experiments/steady_advection_diffusion.jl.

Jointly infers:
- Source field s(x,y) with uncertainty
- Concentration field c(x,y) with uncertainty

Reads problem from TOML and ground truth data from NPZ.
Outputs predictions to NPZ for comparison with PINN baseline.

Usage:
    julia --project=../.. run_gpfvm.jl --problem problems/default.toml
    julia --project=../.. run_gpfvm.jl --problem problems/default.toml --rho 2.5
"""

using LinearAlgebra, SparseArrays
using ArgParse
using NPZ

push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields

import GaussianMarkovRandomFields: mean, std
import FunctionalGPs: ⊗

include("problem.jl")

# ------------------------------------------------------------------------------
# FVM Constraint Builder (exact face integrals)
# ------------------------------------------------------------------------------

"""
Build FVM constraint using EXACT face integrals for flux computation.

For each cell: Σ(face fluxes of c) - s_int = 0
"""
function build_fvm_constraint(xs, ys, prob::SourceIdentificationProblem,
                               layout_c, layout_s, n_c)
    Nx, Ny = length(xs), length(ys)
    n_cells_x, n_cells_y = Nx - 1, Ny - 1
    n_cells = n_cells_x * n_cells_y

    n_s = layout_s.total
    n_total = n_c + n_s
    s_offset = n_c

    # Face indexing (column-major order from TensorProductFunctional):
    # Vertical faces: shape (Nx, n_cells_y) - x varies fastest
    # Horizontal faces: shape (n_cells_x, Ny) - x varies fastest
    vert_idx(i, j) = (j-1)*Nx + i        # i in 1:Nx, j in 1:n_cells_y
    horiz_idx(i, j) = (j-1)*n_cells_x + i  # i in 1:n_cells_x, j in 1:Ny

    c_vert_base = first(indices(layout_c, :c_vert))
    c_horiz_base = first(indices(layout_c, :c_horiz))
    c_dx_vert_base = first(indices(layout_c, :c_dx_vert))
    c_dy_horiz_base = first(indices(layout_c, :c_dy_horiz))

    c_vert(i, j) = c_vert_base + vert_idx(i, j) - 1
    c_horiz(i, j) = c_horiz_base + horiz_idx(i, j) - 1
    c_dx_vert(i, j) = c_dx_vert_base + vert_idx(i, j) - 1
    c_dy_horiz(i, j) = c_dy_horiz_base + horiz_idx(i, j) - 1

    s_int_idx(ci, cj) = s_offset + indices(layout_s, :s_int)[(cj-1)*n_cells_x + ci]

    constraints = []

    for cj in 1:n_cells_y
        for ci in 1:n_cells_x
            row = spzeros(n_total)

            # Advective flux using EXACT face integrals
            # Right face: +vx * ∫c dy
            row[c_vert(ci+1, cj)] += prob.vx
            # Left face: -vx * ∫c dy
            row[c_vert(ci, cj)] -= prob.vx

            # vy advection (if nonzero)
            if prob.vy != 0
                # Top face: +vy * ∫c dx
                row[c_horiz(ci, cj+1)] += prob.vy
                # Bottom face: -vy * ∫c dx
                row[c_horiz(ci, cj)] -= prob.vy
            end

            # Diffusive flux using EXACT derivative face integrals
            # Right face: -D * ∫(∂c/∂x) dy
            row[c_dx_vert(ci+1, cj)] -= prob.D
            # Left face: +D * ∫(∂c/∂x) dy
            row[c_dx_vert(ci, cj)] += prob.D
            # Top face: -D * ∫(∂c/∂y) dx
            row[c_dy_horiz(ci, cj+1)] -= prob.D
            # Bottom face: +D * ∫(∂c/∂y) dx
            row[c_dy_horiz(ci, cj)] += prob.D

            # Source integral: -s_int
            row[s_int_idx(ci, cj)] = -1.0

            push!(constraints, row)
        end
    end

    A_fvm = vcat([reshape(r, 1, :) for r in constraints]...)
    b_fvm = zeros(n_cells)

    return sparse(A_fvm), b_fvm
end

"""
Build boundary condition constraints using face integrals.

- Dirichlet: c = c_inflow at left boundary (point evaluations)
- Neumann: ∫(∂c/∂x)dy = 0 at right boundary
- Neumann: ∫(∂c/∂y)dx = 0 at top/bottom
"""
function build_boundary_constraints(xs, ys, prob::SourceIdentificationProblem,
                                     layout_c, n_c, n_total)
    Nx, Ny = length(xs), length(ys)
    n_cells_x, n_cells_y = Nx - 1, Ny - 1

    # Point evaluation indices for Dirichlet BCs
    c_eval_idx(i, j) = indices(layout_c, :c)[(j-1)*Nx + i]

    # Face indexing (column-major order)
    vert_idx(i, j) = (j-1)*Nx + i
    horiz_idx(i, j) = (j-1)*n_cells_x + i

    c_dx_vert_base = first(indices(layout_c, :c_dx_vert))
    c_dy_horiz_base = first(indices(layout_c, :c_dy_horiz))

    c_dx_vert(i, j) = c_dx_vert_base + vert_idx(i, j) - 1
    c_dy_horiz(i, j) = c_dy_horiz_base + horiz_idx(i, j) - 1

    constraints = []
    rhs = Float64[]

    # Left boundary (inflow): c = c_inflow (Dirichlet via point evals)
    for j in 1:Ny
        row = spzeros(n_total)
        row[c_eval_idx(1, j)] = 1.0
        push!(constraints, row)
        push!(rhs, prob.c_inflow)
    end

    # Right boundary (outflow): ∫(∂c/∂x)dy = 0 (Neumann)
    for j in 1:n_cells_y
        row = spzeros(n_total)
        row[c_dx_vert(Nx, j)] = 1.0
        push!(constraints, row)
        push!(rhs, 0.0)
    end

    # Top boundary: ∫(∂c/∂y)dx = 0 (Neumann)
    for i in 1:n_cells_x
        row = spzeros(n_total)
        row[c_dy_horiz(i, Ny)] = 1.0
        push!(constraints, row)
        push!(rhs, 0.0)
    end

    # Bottom boundary: ∫(∂c/∂y)dx = 0 (Neumann)
    for i in 1:n_cells_x
        row = spzeros(n_total)
        row[c_dy_horiz(i, 1)] = 1.0
        push!(constraints, row)
        push!(rhs, 0.0)
    end

    A_bc = vcat([reshape(r, 1, :) for r in constraints]...)
    return sparse(A_bc), rhs
end

# ------------------------------------------------------------------------------
# Main Solver
# ------------------------------------------------------------------------------

function solve_source_identification(prob::SourceIdentificationProblem, data::Dict;
        ρ::Real = 2.0,
        lengthscale_c::Union{Real, Nothing} = nothing,
        lengthscale_s::Union{Real, Nothing} = nothing,
        source_amplitude::Real = 1.0,
        output_scale::Real = 1.0,
        output_scale_c::Union{Real, Nothing} = nothing,
        smoothness::Int = 2,
        constraint_noise::Real = 1e-5,
        verbose::Bool = true,
        benchmark::Bool = false
    )
    timings = Dict{String, Float64}()

    verbose && println("=" ^ 60)
    verbose && println("GP-FVM Source Identification")
    verbose && println("=" ^ 60)

    # Extract grid from data
    xs = data["xs"]
    ys = data["ys"]
    Nx, Ny = length(xs), length(ys)
    n_cells_x, n_cells_y = Nx - 1, Ny - 1

    verbose && println("\nGrid: $(Nx) × $(Ny) nodes, $(n_cells_x) × $(n_cells_y) cells")

    # Compute grid spacing
    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]
    Δ = min(Δx, Δy)

    # Default lengthscales: 5 * grid spacing (resolve ~5 cells)
    lengthscale_c = isnothing(lengthscale_c) ? 5 * Δ : lengthscale_c
    lengthscale_s = isnothing(lengthscale_s) ? 5 * Δ : lengthscale_s

    verbose && println("  Grid spacing: Δx=$(round(Δx, digits=4)), Δy=$(round(Δy, digits=4))")
    verbose && println("  Lengthscales: c=$(round(lengthscale_c, digits=4)), s=$(round(lengthscale_s, digits=4))")

    # Build intervals
    x_intervals = intervals_from_endpoints(collect(xs))
    y_intervals = intervals_from_endpoints(collect(ys))

    # Build 2D grid and cells
    grid = FactorizedGrid(xs, ys)
    cells_2d = x_intervals ⊗ y_intervals

    # Build kernels (2D product kernels)
    verbose && println("\nBuilding kernels...")
    k_c = HalfIntegerMaternKernel(smoothness, [lengthscale_c]) ⊗
          HalfIntegerMaternKernel(smoothness, [lengthscale_c])
    k_s = HalfIntegerMaternKernel(smoothness, [lengthscale_s]) ⊗
          HalfIntegerMaternKernel(smoothness, [lengthscale_s])

    # Build functionals with exact face integrals
    verbose && println("Building functionals...")

    # Point evaluations (for BCs, observations, and output)
    L_c_eval = EvaluationFunctional(grid)

    # Face integrals for exact flux computation
    # Vertical faces: ∫_y c(x,y) dy at each x-position
    L_c_vert = EvaluationFunctional(xs) ⊗ VectorizedLebesgueIntegral(y_intervals)
    # Horizontal faces: ∫_x c(x,y) dx at each y-position
    L_c_horiz = VectorizedLebesgueIntegral(x_intervals) ⊗ EvaluationFunctional(ys)

    # Derivative face integrals for diffusive flux
    L_c_dx_vert = L_c_vert ∘ PartialDerivative((1, 0))   # ∫(∂c/∂x) dy
    L_c_dy_horiz = L_c_horiz ∘ PartialDerivative((0, 1)) # ∫(∂c/∂y) dx

    # Source functionals
    L_s_eval = EvaluationFunctional(grid)
    L_s_int = VectorizedLebesgueIntegral(cells_2d)

    # Build sparse precisions
    verbose && println("\nBuilding sparse precisions (ρ = $ρ)...")

    timings["sparse_prec_c"] = @elapsed approx_c = sparse_precision([
        :c => L_c_eval,
        :c_vert => L_c_vert,
        :c_horiz => L_c_horiz,
        :c_dx_vert => L_c_dx_vert,
        :c_dy_horiz => L_c_dy_horiz,
    ], k_c; ρ=ρ, ordering=:integrals_coarsest)
    verbose && println("  Concentration: ", approx_c)

    timings["sparse_prec_s"] = @elapsed approx_s = sparse_precision([
        :s => L_s_eval,
        :s_int => L_s_int,
    ], k_s; ρ=4.0, ordering=:integrals_coarsest)
    verbose && println("  Source: ", approx_s)

    # Combine block-diagonally
    n_c = approx_c.info.n
    n_s = approx_s.info.n
    n_total = n_c + n_s

    # Scale source precision by 1/amplitude² to get prior variance = amplitude²
    # Scale by output_scale (prior covariance scales by output_scale)
    # Optionally use separate output_scale_c for concentration (flat c prior)
    σ²_c = isnothing(output_scale_c) ? output_scale : output_scale_c
    σ²_s = output_scale
    Q_s_scaled = sparse(approx_s.Q) / (source_amplitude^2 * σ²_s)
    Q_c_scaled = sparse(approx_c.Q) / σ²_c
    Q_joint = blockdiag(Q_c_scaled, Q_s_scaled)
    verbose && source_amplitude != 1.0 && println("  Source amplitude: $source_amplitude (prior std)")

    layout_c = approx_c.layout
    layout_s = approx_s.layout

    verbose && println("  Joint state dimension: $n_total")

    # Build FVM constraints
    verbose && println("\nBuilding FVM constraints...")
    A_fvm, b_fvm = build_fvm_constraint(xs, ys, prob, layout_c, layout_s, n_c)
    A_bc, b_bc = build_boundary_constraints(xs, ys, prob, layout_c, n_c, n_total)

    verbose && println("  FVM equations: $(size(A_fvm, 1))")
    verbose && println("  Boundary conditions: $(size(A_bc, 1))")

    # Combine FVM and BC constraints
    A_constraints = vcat(A_fvm, A_bc)
    b_constraints = vcat(b_fvm, b_bc)
    n_constraints = size(A_constraints, 1)
    Q_constraints = (1.0 / constraint_noise^2) * sparse(I, n_constraints, n_constraints)

    # Build observation matrix
    obs_x = data["obs_x"]
    obs_y = data["obs_y"]
    obs_c = data["obs_c"]
    n_obs = length(obs_c)
    noise_std = Float64(data["noise_std"])

    verbose && println("\nBuilding observation matrix ($(n_obs) observations)...")

    # Find nearest grid points and build selection matrix
    obs_ix = [argmin(abs.(xs .- ox)) for ox in obs_x]
    obs_iy = [argmin(abs.(ys .- oy)) for oy in obs_y]
    obs_indices = [indices(layout_c, :c)[(iy-1)*Nx + ix] for (ix, iy) in zip(obs_ix, obs_iy)]

    A_obs = spzeros(n_obs, n_total)
    for (i, idx) in enumerate(obs_indices)
        A_obs[i, idx] = 1.0
    end
    Q_obs = (1.0 / noise_std^2) * sparse(I, n_obs, n_obs)

    # Apply all conditioning in one step (avoids intermediate Cholesky)
    verbose && println("\nConditioning (combined FVM + BC + observations)...")
    timings["conditioning"] = @elapsed x_posterior = condition_precision(Q_joint, [
        (A=A_constraints, Q_ϵ=Q_constraints, y=b_constraints),
        (A=A_obs, Q_ϵ=Q_obs, y=obs_c),
    ])

    # Extract posterior statistics
    verbose && println("\nExtracting posterior statistics...")
    timings["posterior_stats"] = @elapsed begin
        μ = mean(x_posterior)
        σ = std(x_posterior)
    end

    # Concentration posterior
    c_mean = reshape(μ[indices(layout_c, :c)], Nx, Ny)
    c_std = reshape(σ[indices(layout_c, :c)], Nx, Ny)

    # Source posterior
    s_eval_indices = n_c .+ indices(layout_s, :s)
    s_int_indices = n_c .+ indices(layout_s, :s_int)

    s_mean = reshape(μ[s_eval_indices], Nx, Ny)
    s_std = reshape(σ[s_eval_indices], Nx, Ny)
    s_int_mean = reshape(μ[s_int_indices], n_cells_x, n_cells_y)
    s_int_std = reshape(σ[s_int_indices], n_cells_x, n_cells_y)

    # Find MAP estimate of source location from point evaluations
    max_idx = argmax(s_mean)
    map_x = xs[max_idx[1]]
    map_y = ys[max_idx[2]]

    # Cell midpoints for integral visualization
    cell_mx = [0.5 * (xs[i] + xs[i+1]) for i in 1:n_cells_x]
    cell_my = [0.5 * (ys[j] + ys[j+1]) for j in 1:n_cells_y]

    # True source location (first source)
    true_x = Float64(data["source_x"][1])
    true_y = Float64(data["source_y"][1])
    location_error = sqrt((map_x - true_x)^2 + (map_y - true_y)^2)

    verbose && println("\nResults:")
    verbose && println("  Concentration range: [$(round(minimum(c_mean), digits=4)), $(round(maximum(c_mean), digits=4))]")
    verbose && println("  Source range: [$(round(minimum(s_mean), digits=4)), $(round(maximum(s_mean), digits=4))]")
    verbose && println("  True source: ($true_x, $true_y)")
    verbose && println("  MAP estimate: ($(round(map_x, digits=3)), $(round(map_y, digits=3)))")
    verbose && println("  Location error: $(round(location_error, digits=4))")

    # Compute total time
    timings["total"] = sum(values(timings))

    info = (
        ρ = ρ,
        lengthscale_c = lengthscale_c,
        lengthscale_s = lengthscale_s,
        n_total = n_total,
        n_c = n_c,
        n_obs = n_obs,
        fill_c = approx_c.info.fill_pct,
        fill_s = approx_s.info.fill_pct,
        map_x = map_x,
        map_y = map_y,
        location_error = location_error,
        timings = timings,
        layout_c = layout_c,
        layout_s = layout_s,
    )

    return (
        c_mean = c_mean,
        c_std = c_std,
        s_mean = s_mean,
        s_std = s_std,
        s_int_mean = s_int_mean,
        s_int_std = s_int_std,
        xs = xs,
        ys = ys,
        cell_mx = cell_mx,
        cell_my = cell_my,
        info = info,
        gmrf = x_posterior,
    )
end

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(description = "GP-FVM source identification solver")

    @add_arg_table! s begin
        "--problem", "-p"
            help = "Path to problem TOML file"
            arg_type = String
            required = true
        "--data", "-d"
            help = "Path to data NPZ file (default: inferred from problem)"
            arg_type = String
            default = ""
        "--output", "-o"
            help = "Output NPZ file (default: results/<problem_name>_gpfvm.npz)"
            arg_type = String
            default = ""
        "--rho", "-r"
            help = "Sparsity parameter"
            arg_type = Float64
            default = 2.0
        "--lengthscale-c"
            help = "Concentration kernel lengthscale (default: 5*Δx)"
            arg_type = Float64
            default = -1.0
        "--lengthscale-s"
            help = "Source kernel lengthscale (default: 5*Δx)"
            arg_type = Float64
            default = -1.0
        "--smoothness"
            help = "Matérn smoothness (1=3/2, 2=5/2)"
            arg_type = Int
            default = 2
        "--source-amplitude"
            help = "Prior std for source field (default: 1.0)"
            arg_type = Float64
            default = 1.0
        "--quiet", "-q"
            help = "Suppress progress output"
            action = :store_true
        "--benchmark", "-b"
            help = "Run in benchmark mode (warmup + timing)"
            action = :store_true
        "--benchmark-runs"
            help = "Number of timed runs in benchmark mode"
            arg_type = Int
            default = 3
    end

    return parse_args(s)
end

function print_benchmark_results(timings::Dict{String, Float64})
    println("\n" * "=" ^ 50)
    println("BENCHMARK RESULTS")
    println("=" ^ 50)
    println("  sparse_precision (c): $(round(timings["sparse_prec_c"]*1000, digits=1)) ms")
    println("  sparse_precision (s): $(round(timings["sparse_prec_s"]*1000, digits=1)) ms")
    println("  conditioning:         $(round(timings["conditioning"]*1000, digits=1)) ms")
    println("  posterior stats:      $(round(timings["posterior_stats"]*1000, digits=1)) ms")
    println("-" ^ 50)
    println("  TOTAL:                $(round(timings["total"]*1000, digits=1)) ms")
    println("=" ^ 50)
end

function main()
    args = parse_commandline()
    verbose = !args["quiet"]
    benchmark = args["benchmark"]
    n_runs = args["benchmark-runs"]

    # Load problem
    prob = load_problem(args["problem"])
    verbose && println(prob)

    # Determine data path
    data_path = args["data"]
    if isempty(data_path)
        problem_name = splitext(basename(args["problem"]))[1]
        data_path = joinpath(dirname(args["problem"]), "..", "data", "$(problem_name).npz")
    end

    verbose && println("\nLoading data from: $data_path")
    data = npzread(data_path)

    # Solve (use nothing for lengthscales if default -1.0)
    ls_c = args["lengthscale-c"] < 0 ? nothing : args["lengthscale-c"]
    ls_s = args["lengthscale-s"] < 0 ? nothing : args["lengthscale-s"]

    solver_kwargs = (
        ρ = args["rho"],
        lengthscale_c = ls_c,
        lengthscale_s = ls_s,
        source_amplitude = args["source-amplitude"],
        smoothness = args["smoothness"],
    )

    if benchmark
        # Benchmark mode: warmup + multiple timed runs
        println("\n" * "=" ^ 50)
        println("BENCHMARK MODE")
        println("=" ^ 50)

        println("\nWarmup run...")
        solve_source_identification(prob, data; solver_kwargs..., verbose=false, benchmark=true)
        println("Warmup complete.")

        println("\nRunning $n_runs timed iterations...")
        all_timings = Vector{Dict{String, Float64}}()
        for i in 1:n_runs
            result = solve_source_identification(prob, data; solver_kwargs..., verbose=false, benchmark=true)
            push!(all_timings, result.info.timings)
            println("  Run $i: $(round(result.info.timings["total"]*1000, digits=1)) ms")
        end

        # Compute min timings (best case)
        best_timings = Dict{String, Float64}()
        for key in keys(all_timings[1])
            best_timings[key] = minimum(t[key] for t in all_timings)
        end

        print_benchmark_results(best_timings)

        # Final run with verbose output for results
        println("\nFinal run (with output):")
        result = solve_source_identification(prob, data; solver_kwargs..., verbose=true, benchmark=true)
    else
        # Normal mode
        result = solve_source_identification(prob, data; solver_kwargs..., verbose=verbose, benchmark=false)
    end

    # Prepare output
    problem_name = splitext(basename(args["problem"]))[1]
    output_path = args["output"]
    if isempty(output_path)
        output_path = joinpath(dirname(args["problem"]), "..", "results", "$(problem_name)_gpfvm.npz")
    end

    mkpath(dirname(output_path))

    output = Dict{String, Any}(
        # Grid
        "xs" => result.xs,
        "ys" => result.ys,
        "cell_mx" => result.cell_mx,
        "cell_my" => result.cell_my,

        # Concentration predictions
        "c_mean" => result.c_mean,
        "c_std" => result.c_std,

        # Source predictions
        "s_mean" => result.s_mean,
        "s_std" => result.s_std,
        "s_int_mean" => result.s_int_mean,
        "s_int_std" => result.s_int_std,

        # Solver info
        "rho" => result.info.ρ,
        "lengthscale_c" => result.info.lengthscale_c,
        "lengthscale_s" => result.info.lengthscale_s,
        "n_total" => result.info.n_total,
        "n_obs" => result.info.n_obs,
        "fill_c_pct" => result.info.fill_c,
        "fill_s_pct" => result.info.fill_s,
        "map_x" => result.info.map_x,
        "map_y" => result.info.map_y,
        "location_error" => result.info.location_error,
    )

    npzwrite(output_path, output)
    verbose && println("\nSaved results to: $output_path")

    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
