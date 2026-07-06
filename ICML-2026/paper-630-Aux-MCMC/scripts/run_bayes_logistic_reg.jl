using ArgParse
using JLD2, FileIO

include("../src/bayes_logistic_reg/common.jl")
include("../src/bayes_logistic_reg/AliasSampler.jl")
include("../src/bayes_logistic_reg/methods.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--method"
            help = "method: mh | mala | hmc | tuna_mh | tuna_sgld | barker"
            arg_type = String
            default = "mh"

        "--task"
            help = "dataset task: mnist35 | mnist79"
            arg_type = String
            default = "mnist35"

        "--run_all"
            help = "run all methods on the selected task"
            action = :store_true

        "--seed"
            help = "random seed"
            arg_type = Int
            default = 2024

        "--pca_dim"
            help = "PCA dimension"
            arg_type = Int
            default = 50

        "--stepsize"
            help = "generic stepsize for single-method run"
            arg_type = Float64
            default = 4e-3

        "--nsamples"
            help = "generic nsamples for single-method run"
            arg_type = Int
            default = 10000

        "--burnin"
            help = "number of burnin iterations"
            arg_type = Int
            default = 0

        "--T"
            help = "temperature"
            arg_type = Float64
            default = 1.0

        "--lam"
            help = "lambda for tuna_mh / tuna_sgld"
            arg_type = Float64
            default = 1e-5

        "--grad_size"
            help = "mini-batch size for tuna_sgld"
            arg_type = Int
            default = 20

        "--leapfrog"
            help = "number of leapfrog steps for hmc"
            arg_type = Int
            default = 5

        "--outdir"
            help = "output directory"
            arg_type = String
            default = "results/bayes_logistic_reg"

        # per-method settings used only in --run_all mode
        "--mh_stepsize"
            arg_type = Float64
            default = 4e-3
        "--mh_nsamples"
            arg_type = Int
            default = 10000

        "--mala_stepsize"
            arg_type = Float64
            default = 4e-3
        "--mala_nsamples"
            arg_type = Int
            default = 10000

        "--hmc_stepsize"
            arg_type = Float64
            default = 4e-3
        "--hmc_nsamples"
            arg_type = Int
            default = 10000

        "--tuna_mh_stepsize"
            arg_type = Float64
            default = 4e-3
        "--tuna_mh_nsamples"
            arg_type = Int
            default = 100000

        "--tuna_sgld_stepsize"
            arg_type = Float64
            default = 4e-3
        "--tuna_sgld_nsamples"
            arg_type = Int
            default = 100000

        "--barker_stepsize"
            arg_type = Float64
            default = 4e-3
        "--barker_nsamples"
            arg_type = Int
            default = 10000
    end

    return parse_args(s)
end

function build_method_args(args::Dict, method::String)
    local_args = copy(args)
    local_args["method"] = method
    local_args["stepsize"] = args["$(method)_stepsize"]
    local_args["nsamples"] = args["$(method)_nsamples"]
    return local_args
end

function run_one(args::Dict)
    set_seed!(args["seed"])
    train_x, train_y, test_x, test_y = generate_data(args)
    result = dispatch_method(args, train_x, train_y, test_x, test_y)
    outfile = save_result(args, result)

    println("Saved to: $outfile")
    if haskey(result, "acc")
        println("final accuracy: $(result["acc"][end])")
    end
    if haskey(result, "acc_time")
        println("runtime: $(result["acc_time"][end])")
    end
    if haskey(result, "avg_accept_prob")
        println("avg acceptance prob: $(result["avg_accept_prob"])")
    end
    if haskey(result, "total_bs")
        println("avg batch size: $(result["total_bs"] / (args["nsamples"] + args["burnin"]))")
    end
end

function run_all_methods(args::Dict)
    methods = ["mh", "mala", "hmc", "tuna_mh", "tuna_sgld", "barker"]

    for method in methods
        println("\n==============================")
        println("Running method: $method | task: $(args["task"])")
        println("==============================")
        local_args = build_method_args(args, method)
        run_one(local_args)
    end
end

function main()
    args = parse_commandline()

    if args["run_all"]
        run_all_methods(args)
    else
        run_one(args)
    end
end

main()