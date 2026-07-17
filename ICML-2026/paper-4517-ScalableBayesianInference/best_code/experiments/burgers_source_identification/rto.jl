"""
Randomize-Then-Optimize (RTO) uncertainty estimation for Burgers source ID.

Perturbs observations K times, re-solves the MAP each time, and takes the
sample variance of the source estimates as a non-Gaussian uncertainty measure.

For linear forward models, RTO recovers the exact posterior variance.
For nonlinear models, it captures global sensitivity (not just local curvature),
which can reveal posterior asymmetry missed by the Laplace approximation.

Usage:
    julia --project=../.. rto.jl --n-rto 50
    julia --project=../.. rto.jl --n-rto 20 --seed 30  # quick test
"""

using LinearAlgebra, SparseArrays
using Random, Statistics, Printf
using CairoMakie
using TuePlots
using ProgressMeter
using Kronecker: kronecker
using NPZ

push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using GPFiniteVolume
using FunctionalGPs, GaussianMarkovRandomFields

import GaussianMarkovRandomFields: mean, std, precision_map, gaussian_approximation
import FunctionalGPs: ⊗
using GaussianMarkovRandomFields: NonlinearLeastSquaresModel
using SparseConnectivityTracer, SparseMatrixColorings

include("problem.jl")
include("functionals.jl")
include("constraints.jl")
include("solver.jl")
include("plot.jl")

function run_rto(;
        N::Int=16, n_timesteps::Int=10, Δt::Float64=0.05, ν::Float64=0.01,
        n_obs::Int=50, obs_every::Int=2, noise_std::Float64=0.05,
        source_x::Float64=0.4, source_y::Float64=0.5,
        source_amp::Float64=3.0, source_width::Float64=0.1,
        ρ::Float64=2.0, lengthscale_u::Float64=0.15, lengthscale_s::Float64=0.15,
        source_amplitude::Float64=3.0, output_scale::Float64=1.0,
        max_iter::Int=50, log_source::Bool=true,
        n_rto::Int=50, seed::Int=42,
        output_dir::String=joinpath(@__DIR__, "results", "rto"))

    println("=" ^ 60)
    println("Randomize-Then-Optimize (RTO) Uncertainty Estimation")
    println("=" ^ 60)
    @printf("N=%d, n_t=%d, n_obs=%d, n_rto=%d\n", N, n_timesteps, n_obs, n_rto)

    # Setup problem
    if source_x < 0  # use random_problem
        prob_rng = MersenneTwister(seed)
        prob = random_problem(prob_rng; ν=ν, T_end=n_timesteps * Δt, Δt=Δt)
        println("Random sources:")
        for (i, s) in enumerate(prob.sources)
            @printf("  [%d] (%.2f, %.2f), amp=%.2f, width=%.2f\n", i, s.x, s.y, s.strength, s.width)
        end
    else
        source = GaussianSource(source_x, source_y, source_amp, source_width)
        prob = BurgersSourceProblem(ν, (0.0, 1.0, 0.0, 1.0), [source], n_timesteps * Δt, Δt)
    end

    xs = collect(range(0, 1, length=N)); ys = copy(xs)
    u_truth, s_true = forward_solve(prob, xs, ys)

    rng = MersenneTwister(seed)
    obs = generate_observations(rng, u_truth, xs, ys;
        n_obs=n_obs, obs_every=obs_every, noise_std=noise_std)
    obs_data = merge(obs, (; Nx=N, Ny=N))

    println("\nSource: ($source_x, $source_y), amp=$source_amp, width=$source_width")
    @printf("Observations: %d total\n", length(obs.obs_values))

    # Run baseline (unperturbed) to get Laplace estimate
    println("\n--- Baseline (Laplace) solve ---")
    result_base = solve_burgers_source_id(prob, obs_data;
        ρ=ρ, lengthscale_u=lengthscale_u, lengthscale_s=lengthscale_s,
        output_scale=output_scale, source_amplitude=source_amplitude,
        max_iter=max_iter, log_source=log_source, verbose=true)

    _softplus(x) = log(1 + exp(x))
    Nx, Ny = N, N

    # Build the joint prior GMRF (same construction as solver, before BCs/obs)
    # to sample full-state prior perturbations for proper RTO.
    println("\nBuilding joint prior for RTO sampling...")
    approx_u_rto = build_u_precision(xs, ys; smoothness=2, lengthscale=lengthscale_u, ρ=ρ)
    approx_s_rto = build_s_precision(xs, ys; smoothness=2, lengthscale=lengthscale_s, log_source=log_source)
    n_u_spatial = approx_u_rto.info.n
    n_s = approx_s_rto.info.n

    Q_u_spatial = sparse(approx_u_rto.Q) / output_scale
    Q_state0 = [Q_u_spatial spzeros(size(Q_u_spatial)...);
                spzeros(size(Q_u_spatial)...) Q_u_spatial]
    x0_prior = GMRF(zeros(size(Q_state0, 1)), Q_state0)

    sde = IWPSDE(1.0)
    A_t, Σ_noise_t = discretize_vanloan(sde, prob.Δt)
    Q_noise_t = inv(Σ_noise_t); Q_noise_t = 0.5*(Q_noise_t + Q_noise_t')
    A_kr = kronecker(A_t, sparse(I, n_u_spatial, n_u_spatial))
    Q_noise_kr = kronecker(Q_noise_t, Q_u_spatial)

    n_t_total = round(Int, prob.T_end / prob.Δt) + 1
    ssm = ConstantLinearGaussianSSM(x0_prior, A_kr, Q_noise_kr)
    x_u_prior = GPFiniteVolume.joint_gmrf(ssm, n_t_total)

    Q_s_scaled = sparse(approx_s_rto.Q) / (source_amplitude^2 * output_scale)
    Q_joint_prior = blockdiag(sparse(GaussianMarkovRandomFields.precision_map(x_u_prior)), Q_s_scaled)
    n_joint = size(Q_joint_prior, 1)
    joint_prior_gmrf = GMRF(zeros(n_joint), Symmetric(Q_joint_prior))
    println("  Joint prior DOF: $n_joint")

    # Collect RTO source MAPs + local Laplace stds
    println("\n--- RTO: $n_rto perturbed MAP solves (obs + full joint prior perturbation) ---")
    s_maps = zeros(Nx * Ny, n_rto)
    s_laplace_stds = zeros(Nx * Ny, n_rto)
    rto_rng = MersenneTwister(seed + 1_000_000)

    progress = Progress(n_rto; desc="RTO solves: ", showspeed=true)
    for k in 1:n_rto
        # Perturb observations
        perturbed_values = obs.obs_values .+ noise_std .* randn(rto_rng, length(obs.obs_values))
        obs_k = merge(obs, (; obs_values=perturbed_values))
        obs_data_k = merge(obs_k, (; Nx=N, Ny=N))

        # Perturb full joint prior mean: η_k ~ N(0, Q_joint_prior^{-1})
        η_k = rand(joint_prior_gmrf)

        # Solve MAP with perturbed observations AND perturbed joint prior
        result_k = solve_burgers_source_id(prob, obs_data_k;
            ρ=ρ, lengthscale_u=lengthscale_u, lengthscale_s=lengthscale_s,
            output_scale=output_scale, source_amplitude=source_amplitude,
            max_iter=max_iter, log_source=log_source,
            joint_prior_perturbation=η_k, verbose=false)

        s_maps[:, k] = vec(result_k.s_mean)
        s_laplace_stds[:, k] = vec(result_k.s_std)
        next!(progress; showvalues=[
            (:iteration, "$k/$n_rto"),
            (:source_rmse, round(sqrt(Statistics.mean((s_true .- result_k.s_mean).^2)), digits=3)),
        ])
    end

    # Gaussian Mixture Laplace via law of total variance:
    # Var[s] = E[local variance] + Var[local means]
    mean_local_var = Statistics.mean(s_laplace_stds.^2, dims=2)[:]  # average Laplace variance
    var_of_means = Statistics.var(s_maps, dims=2)[:]                 # spread of MAPs
    gml_std = reshape(sqrt.(mean_local_var .+ var_of_means), Nx, Ny)

    # Plain RTO (spread of MAPs only, no local curvature)
    s_rto_mean = reshape(Statistics.mean(s_maps, dims=2)[:], Nx, Ny)
    s_rto_std = reshape(sqrt.(var_of_means), Nx, Ny)

    println("\n\n--- Results ---")
    @printf("Laplace std range:     [%.4f, %.4f]\n", extrema(result_base.s_std)...)
    @printf("RTO std range:         [%.4f, %.4f]\n", extrema(s_rto_std)...)
    @printf("GM-Laplace std range:  [%.4f, %.4f]\n", extrema(gml_std)...)
    @printf("\nCorrelation with |Error|:\n")
    s_err = abs.(s_true .- result_base.s_mean)
    @printf("  Laplace:    %.3f\n", cor(vec(result_base.s_std), vec(s_err)))
    @printf("  RTO:        %.3f\n", cor(vec(s_rto_std), vec(s_err)))
    @printf("  GM-Laplace: %.3f\n", cor(vec(gml_std), vec(s_err)))

    # Plot comparison: Laplace std | RTO std | GM-Laplace std | |Error|
    mkpath(output_dir)

    # Cache the std maps so we can re-plot without re-running RTO
    npzwrite(joinpath(output_dir, "rto_data.npz"), Dict(
        "xs" => collect(xs), "ys" => collect(ys),
        "laplace_std" => Array(result_base.s_std),
        "rto_std" => Array(s_rto_std),
        "gml_std" => Array(gml_std),
        "s_err" => Array(s_err),
        "n_rto" => [n_rto],
    ))

    set_theme!(Theme(
        TuePlots.SETTINGS[:ICML];
        font=true, fontsize=true, figsize=true,
        single_column=false, nrows=1, ncols=4,
        subplot_height_to_width_ratio=1.0,
    ))
    fig = Figure()

    # Robust shared color scale (99th percentile, to limit MC outliers).
    std_all = vcat(vec(result_base.s_std), vec(gml_std), vec(s_rto_std))
    std_max = Statistics.quantile(std_all, 0.99)

    ax1 = Axis(fig[1,1]; aspect=DataAspect(), title="Laplace std", xlabel=L"x", ylabel=L"y")
    heatmap!(ax1, xs, ys, result_base.s_std; colormap=:inferno, colorrange=(0, std_max))

    ax2 = Axis(fig[1,2]; aspect=DataAspect(), title="RTO std (K=$n_rto)", xlabel=L"x", yticklabelsvisible=false)
    heatmap!(ax2, xs, ys, s_rto_std; colormap=:inferno, colorrange=(0, std_max))

    ax3 = Axis(fig[1,3]; aspect=DataAspect(), title="GM-Laplace std (K=$n_rto)", xlabel=L"x", yticklabelsvisible=false)
    hm3 = heatmap!(ax3, xs, ys, gml_std; colormap=:inferno, colorrange=(0, std_max))
    Colorbar(fig[1,4], hm3; width=8)

    ax4 = Axis(fig[1,5]; aspect=DataAspect(), title="|Error|", xlabel=L"x", yticklabelsvisible=false)
    hm4 = heatmap!(ax4, xs, ys, s_err; colormap=:inferno)
    Colorbar(fig[1,6], hm4; width=8)

    filename = joinpath(output_dir, "rto_comparison.pdf")
    save(filename, fig, pt_per_unit=1)
    set_theme!()
    println("\nSaved: $filename")

    return (; s_rto_mean, s_rto_std, gml_std, result_base, s_true, xs, ys)
end

if abspath(PROGRAM_FILE) == @__FILE__
    local kw = Dict{Symbol,Any}()
    local idx = 1
    while idx <= length(ARGS)
        if ARGS[idx] == "--n-rto"
            kw[:n_rto] = parse(Int, ARGS[idx+1]); idx += 2
        elseif ARGS[idx] == "--seed"
            kw[:seed] = parse(Int, ARGS[idx+1]); idx += 2
        elseif ARGS[idx] == "--random-sources"
            kw[:source_x] = -1.0; idx += 1  # trigger random_problem
        else
            idx += 1
        end
    end
    run_rto(; kw...)
end
