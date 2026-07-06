#!/usr/bin/env julia --project=.
# Fast single-rate evaluation: runs PoissonMH + PoissonMALA for one (rate, round)
# and directly computes ESS/s, outputting machine-parseable JSON metrics.
using Random
using ArgParse
using JLD2, FileIO
using Printf
using Statistics
using LinearAlgebra
using Distributions
using MCMCDiagnosticTools

include("../src/gaussian_20d/AliasSampler.jl")
include("../src/gaussian_20d/common.jl")
include("../src/gaussian_20d/methods.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--target_rate"
            help = "Target acceptance rate"
            arg_type = Float64
            required = true
        "--round"
            help = "Experiment round number"
            arg_type = Int64
            required = true
        "--outdir"
            help = "Output directory for .jld2 files"
            arg_type = String
            default = "results/gaussian_20d"
        "--seed"
            help = "Random seed (default: based on round)"
            arg_type = Int64
            default = -1
        "--poismh_steps"
            help = "Number of PoissonMH steps (default: 50000)"
            arg_type = Int64
            default = 50000
        "--poismala_steps"
            help = "Number of PoissonMALA steps (default: 20000)"
            arg_type = Int64
            default = 20000
    end
    return parse_args(s)
end

function ess_per_sec(samples::Array{Float64,2}, time_vec::Vector{Float64})
    nsamples = size(samples, 2)
    d = size(samples, 1)
    total_time = time_vec[end] - time_vec[1]
    if total_time <= 0
        error("non-positive effective runtime")
    end
    out = zeros(d)
    for j in 1:d
        out[j] = ess(samples[j, :]) / total_time
    end
    return out
end

function main()
    args = parse_commandline()
    target_rate = args["target_rate"]
    round = args["round"]
    outdir = args["outdir"]
    seed = args["seed"]
    poismh_steps = args["poismh_steps"]
    poismala_steps = args["poismala_steps"]

    if seed > 0
        Random.seed!(seed)
    else
        Random.seed!(round * 1000 + Int(floor(target_rate * 1000)))
    end

    println("=== FAST_EVAL target_rate=$target_rate round=$round poismh_steps=$poismh_steps poismala_steps=$poismala_steps ===")

    theta_init = randn(20)
    stepsize_list = get_stepsize_list(target_rate)

    dim = 20
    data_size = 100000
    beta = 1e-5
    lam_const = 0.0005
    y, cov_y = generate_data(dim, data_size)
    prec_y = inv(cov_y)

    theta_true, mean_true, cov_true = compute_theta_true(y, cov_y, beta; save_theta_true=false)

    results = Dict{String, Any}()

    # PoissonMH
    params_poismh = Params(y, prec_y, dim, beta, lam_const, stepsize_list[3], poismh_steps)
    println("Running PoissonMH ($poismh_steps steps)...")
    poismh_samples, accept_poismh, time_poismh = poismh(params_poismh, theta_init)
    ess_poismh = ess_per_sec(poismh_samples, time_poismh)
    results["PoissonMH_ESS_s_Min"] = Base.round(minimum(ess_poismh), digits=4)
    results["PoissonMH_ESS_s_Median"] = Base.round(median(ess_poismh), digits=4)
    results["PoissonMH_ESS_s_Max"] = Base.round(maximum(ess_poismh), digits=4)
    results["PoissonMH_accept_rate"] = Base.round(accept_poismh, digits=4)
    println("  PoissonMH: Min=$(results["PoissonMH_ESS_s_Min"]) Median=$(results["PoissonMH_ESS_s_Median"]) Max=$(results["PoissonMH_ESS_s_Max"]) accept=$(results["PoissonMH_accept_rate"])")

    # PoissonMALA
    params_pois_mala = Params(y, prec_y, dim, beta, lam_const, stepsize_list[5], poismala_steps)
    println("Running PoissonMALA ($poismala_steps steps)...")
    pois_mala_samples, accept_pois_mala, time_pois_mala = pois_mala(params_pois_mala, theta_init)
    ess_pois_mala = ess_per_sec(pois_mala_samples, time_pois_mala)
    results["PoissonMALA_ESS_s_Min"] = Base.round(minimum(ess_pois_mala), digits=4)
    results["PoissonMALA_ESS_s_Median"] = Base.round(median(ess_pois_mala), digits=4)
    results["PoissonMALA_ESS_s_Max"] = Base.round(maximum(ess_pois_mala), digits=4)
    results["PoissonMALA_accept_rate"] = Base.round(accept_pois_mala, digits=4)
    println("  PoissonMALA: Min=$(results["PoissonMALA_ESS_s_Min"]) Median=$(results["PoissonMALA_ESS_s_Median"]) Max=$(results["PoissonMALA_ESS_s_Max"]) accept=$(results["PoissonMALA_accept_rate"])")

    # Save result file
    mkpath(outdir)
    save_dict = Dict(
        "theta_init" => theta_init,
        "mean_true" => mean_true,
        "cov_true" => cov_true,
        "theta_true" => theta_true,
        "poismh_samples" => poismh_samples,
        "pois_mala_samples" => pois_mala_samples,
        "time_poismh" => time_poismh,
        "time_pois_mala" => time_pois_mala,
        "accept_poismh" => accept_poismh,
        "accept_pois_mala" => accept_pois_mala,
        "data_size" => data_size,
        "beta" => beta,
        "lam_const" => lam_const,
    )
    outfile = build_output_path(outdir, data_size, lam_const, target_rate, beta, round)
    @save outfile save_dict args
    println("Saved to: $outfile")

    # Output machine-parseable metrics
    results["target_rate"] = target_rate
    results["round"] = round
    results["outfile"] = outfile
    println("=== METRICS_START ===")
    for (k, v) in results
        println("$k=$v")
    end
    println("=== METRICS_END ===")
end

main()
