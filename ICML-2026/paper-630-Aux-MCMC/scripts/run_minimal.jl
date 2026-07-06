#!/usr/bin/env julia --project=.
# Minimal runner: PoissonMALA + PoissonMH only for the rubric
using Random
using ArgParse
using JLD2, FileIO
using Printf
using Statistics
using LinearAlgebra
using Distributions

include("../src/gaussian_20d/AliasSampler.jl")
include("../src/gaussian_20d/common.jl")
include("../src/gaussian_20d/methods.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--target_rate"
            help = "target accept rate"
            arg_type = Float64
            required = true
        "--round"
            help = "experiment round"
            arg_type = Int64
            required = true
        "--outdir"
            help = "output directory"
            arg_type = String
            default = "results/gaussian_20d"
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    target_rate = args["target_rate"]
    round = args["round"]
    outdir = args["outdir"]

    println("target rate: ", target_rate)
    println("round: ", round)

    theta_init = randn(20)
    stepsize_list = get_stepsize_list(target_rate)

    dim = 20
    data_size = 100000
    beta = 1e-5
    lam_const = 0.0005
    y, cov_y = generate_data(dim, data_size)
    prec_y = inv(cov_y)

    theta_true, mean_true, cov_true = compute_theta_true(y, cov_y, beta; save_theta_true=false)

    # PoissonMH: 50000 steps
    params_poismh = Params(y, prec_y, dim, beta, lam_const, stepsize_list[3], 50000)
    println("Running PoissonMH...")
    poismh_samples, accept_poismh, time_poismh = poismh(params_poismh, theta_init)

    # PoissonMALA: 20000 steps
    params_pois_mala = Params(y, prec_y, dim, beta, lam_const, stepsize_list[5], 20000)
    println("Running PoissonMALA...")
    pois_mala_samples, accept_pois_mala, time_pois_mala = pois_mala(params_pois_mala, theta_init)

    result = Dict(
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

    mkpath(outdir)
    outfile = build_output_path(outdir, data_size, lam_const, target_rate, beta, round)
    @save outfile result args
    println("Saved to: ", outfile)
end

main()
