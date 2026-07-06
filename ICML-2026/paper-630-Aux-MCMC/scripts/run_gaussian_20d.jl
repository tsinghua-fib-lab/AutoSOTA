using Random
using ArgParse
using JLD2, FileIO
using Printf

include("../src/gaussian_20d/common.jl")
include("../src/gaussian_20d/methods.jl")

function mean_mse(samples::Array{Float64,2}, mean_true)
    sample_mean = vec(mean(samples, dims=2))
    true_mean = vec(mean_true)
    return mean((sample_mean .- true_mean).^2)
end

function cov_diag_mse(samples::Array{Float64,2}, cov_true)
    # samples: dim × nsamples
    sample_cov_diag = diag(cov(permutedims(samples)))
    true_cov_diag = vec(cov_true)
    return mean((sample_cov_diag .- true_cov_diag).^2)
end

function print_metrics(name::String, samples::Array{Float64,2}, accept_count, nsteps::Int, time_step, mean_true, cov_true; print_accept::Bool=true)

    mean_err = mean_mse(samples, mean_true)
    cov_err = cov_diag_mse(samples, cov_true)

    println("\n", "="^20, " ", name, " ", "="^20)

    if print_accept
        println(rpad("accept rate :", 14), round(accept_count / nsteps, digits=4))
    else
        println(rpad("accept rate :", 14), "N/A")
    end

    println(rpad("time :", 14), round(time_step[end], digits=4))
    println(rpad("mean mse :", 14), @sprintf("%.6e", mean_err))
    println(rpad("cov mse :", 14), @sprintf("%.6e", cov_err))
end

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

        "--save_theta_true"
            help = "whether to save full theta_true"
            action = :store_true

        "--mh_steps"
            arg_type = Int64
            default = 5000

        "--mala_steps"
            arg_type = Int64
            default = 2500

        "--poismh_steps"
            arg_type = Int64
            default = 50000

        "--pois_barker_steps"
            arg_type = Int64
            default = 20000

        "--pois_mala_steps"
            arg_type = Int64
            default = 20000

        "--barker_steps"
            arg_type = Int64
            default = 5000

        "--sgld_steps"
            arg_type = Int64
            default = 2500000
    end

    return parse_args(s)
end

function build_steps_list(args::Dict)
    return [
        args["mh_steps"],
        args["mala_steps"],
        args["poismh_steps"],
        args["pois_barker_steps"],
        args["pois_mala_steps"],
        args["barker_steps"],
        args["sgld_steps"],
    ]
end

function main()
    args = parse_commandline()

    target_rate = args["target_rate"]
    round = args["round"]
    outdir = args["outdir"]
    save_theta_true = args["save_theta_true"]

    println("target rate: ", target_rate)
    println("round: ", round)

    theta_init = randn(20)
    stepsize_list = get_stepsize_list(target_rate)
    steps = build_steps_list(args)

    result = run_sampler(theta_init, stepsize_list, steps; save_theta_true=save_theta_true)

    print_metrics("MH",
        result["mh_samples"], result["accept_mh"], steps[1], result["time_mh"],
        result["mean_true"], result["cov_true"])

    print_metrics("MALA",
        result["mala_samples"], result["accept_mala"], steps[2], result["time_mala"],
        result["mean_true"], result["cov_true"])

    print_metrics("Pois",
        result["poismh_samples"], result["accept_poismh"], steps[3], result["time_poismh"],
        result["mean_true"], result["cov_true"])

    print_metrics("Pois Barker",
        result["pois_barker_samples"], result["accept_pois_barker"], steps[4], result["time_pois_barker"],
        result["mean_true"], result["cov_true"])

    print_metrics("Pois MALA",
        result["pois_mala_samples"], result["accept_pois_mala"], steps[5], result["time_pois_mala"],
        result["mean_true"], result["cov_true"])

    print_metrics("Barker",
        result["barker_samples"], result["accept_barker"], steps[6], result["time_barker"],
        result["mean_true"], result["cov_true"])

    print_metrics("SGLD",
        result["sgld_samples"], result["accept_sgld"], steps[7], result["time_sgld"],
        result["mean_true"], result["cov_true"]; print_accept=false)

    mkpath(outdir)
    outfile = build_output_path(outdir, result["data_size"], result["lam_const"], target_rate, result["beta"], round)

    if save_theta_true
        @save outfile result args
    else
        result_nosave = copy(result)
        delete!(result_nosave, "theta_true")
        @save outfile result_nosave args
    end

    println("Saved to: ", outfile)
end

main()
