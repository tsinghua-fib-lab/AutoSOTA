using Random
using ArgParse
using JLD2, FileIO
using Printf
using Statistics

include("../src/robust_reg/common.jl")
include("../src/robust_reg/methods.jl")

function mean_mse(samples::Array{Float64,2}, theta_true::Array{Float64,1})
    sample_mean = vec(mean(samples, dims=2))
    return mean((sample_mean .- theta_true).^2)
end

function print_metrics(name::String, samples::Array{Float64,2}, accept_count, nsteps::Int, time_step, theta_true::Array{Float64,1}; print_accept::Bool=true)
    mean_err = mean_mse(samples, theta_true)

    println("\n", "="^20, " ", name, " ", "="^20)

    if print_accept
        println(rpad("accept rate :", 14), round(accept_count / nsteps, digits=4))
    else
        println(rpad("accept rate :", 14), "N/A")
    end

    println(rpad("time :", 14), round(time_step[end], digits=4))
    println(rpad("mean mse :", 14), @sprintf("%.6e", mean_err))
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--experiment"
            help = "experiment name"
            arg_type = String
            required = true

        "--target_rate"
            help = "target accept rate"
            arg_type = Float64
            required = true

        "--round"
            help = "experiment round"
            arg_type = Int64
            required = true

        "--outdir"
            help = "output directory; default is results/<experiment>"
            arg_type = String
            default = ""

        "--mh_steps"
            arg_type = Int64
            default = 4000

        "--mala_steps"
            arg_type = Int64
            default = 4000

        "--poismh_steps"
            arg_type = Int64
            default = 40000

        "--pois_barker_steps"
            arg_type = Int64
            default = 20000

        "--pois_mala_steps"
            arg_type = Int64
            default = 20000

        "--hmc_steps"
            arg_type = Int64
            default = 2000

        "--barker_steps"
            arg_type = Int64
            default = 4000

        "--sgld_steps"
            arg_type = Int64
            default = 100000
    end

    return parse_args(s)
end

function build_steps_list(args::Dict, include_sgld::Bool)
    steps = Int[
        args["mh_steps"],
        args["mala_steps"],
        args["poismh_steps"],
        args["pois_barker_steps"],
        args["pois_mala_steps"],
        args["hmc_steps"],
        args["barker_steps"],
    ]
    if include_sgld
        push!(steps, args["sgld_steps"])
    end
    return steps
end

function main()
    args = parse_commandline()

    experiment = args["experiment"]
    target_rate = args["target_rate"]
    round = args["round"]

    config = get_experiment_config(experiment, target_rate)
    outdir = isempty(args["outdir"]) ? default_outdir(experiment) : args["outdir"]

    println("experiment: ", experiment)
    println("target rate: ", target_rate)
    println("round: ", round)

    theta_init = randn(config.theta_init_dim)
    steps = build_steps_list(args, config.include_sgld)

    result = run_sampler(theta_init, config, steps)

    theta_true = result["theta_true"]

    print_metrics("MH", result["mh_samples"], result["accept_mh"], steps[1], result["time_mh"], theta_true)
    print_metrics("MALA", result["mala_samples"], result["accept_mala"], steps[2], result["time_mala"], theta_true)
    print_metrics("Pois", result["poismh_samples"], result["accept_pois"], steps[3], result["time_pois"], theta_true)
    print_metrics("Pois Barker", result["pois_barker_samples"], result["accept_pois_barker"], steps[4], result["time_pois_barker"], theta_true)
    print_metrics("Pois MALA", result["pois_mala_samples"], result["accept_pois_mala"], steps[5], result["time_pois_mala"], theta_true)
    print_metrics("HMC (lf=5)", result["hmc_samples"], result["accept_hmc"], steps[6], result["time_hmc"], theta_true)
    print_metrics("Barker", result["barker_samples"], result["accept_barker"], steps[7], result["time_barker"], theta_true)
    if config.include_sgld
        print_metrics("SGLD", result["sgld_samples"], result["accept_sgld"], steps[8], result["time_sgld"], theta_true; print_accept=false)
    end

    mkpath(outdir)
    outfile = build_output_path(
        outdir,
        experiment,
        result["data_size"],
        result["c"],
        result["df"],
        result["lam_const"],
        target_rate,
        result["beta"],
        round
    )

    @save outfile result args
    println("\nSaved to: ", outfile)
end

main()
