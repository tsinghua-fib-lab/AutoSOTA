using Random
using Statistics
using LinearAlgebra
using MLDatasets
using MultivariateStats
using JLD2, FileIO

function set_seed!(seed::Int)
    Random.seed!(seed)
end

function task_digits(task::String)
    if task == "mnist35"
        return 3, 5
    elseif task == "mnist79"
        return 7, 9
    else
        error("Unknown task: $task. Supported tasks: mnist35, mnist79")
    end
end

function generate_data(args::Dict)
    digit_neg, digit_pos = task_digits(args["task"])

    train_x, train_y = MNIST(:train)[:]
    train_x = reshape(train_x, 784, :)

    idx1 = findall(x -> x == digit_neg, train_y)
    idx2 = findall(x -> x == digit_pos, train_y)
    idx = sort(vcat(idx1, idx2))

    train_y[idx1] .= 0
    train_y[idx2] .= 1
    train_y = train_y[idx]
    train_x = train_x[:, idx]

    test_x, test_y = MNIST(:test)[:]
    test_x = reshape(test_x, 784, :)

    idx1 = findall(x -> x == digit_neg, test_y)
    idx2 = findall(x -> x == digit_pos, test_y)
    idx = sort(vcat(idx1, idx2))

    test_y[idx1] .= 0
    test_y[idx2] .= 1
    test_y = test_y[idx]
    test_x = test_x[:, idx]

    train_x = convert(Array{Float64}, train_x)
    test_x = convert(Array{Float64}, test_x)

    M = fit(PCA, train_x; maxoutdim=args["pca_dim"])
    train_x = transform(M, train_x)
    test_x = transform(M, test_x)

    return train_x, train_y, test_x, test_y
end

function kaiming_unif_init(pca_dim::Int)
    a = sqrt(5.0)
    fan = pca_dim
    gain = sqrt(2.0 / (1 + a^2))
    std = gain / sqrt(fan)
    bound = sqrt(3.0) * std
    return 2 .* bound .* rand(pca_dim) .- bound
end

sigmoid(z::Real) = one(z) / (one(z) + exp(-z))

function logH(predict::Float64, y::Int)
    return y * log(predict) + (1 - y) * log(1 - predict)
end

function test_accuracy(samples::Array{Float64,2}, test_x::Array{Float64,2}, test_y::AbstractVector{<:Integer})
    avg_sample = vec(mean(samples, dims=2))
    n = size(test_x, 2)
    acc = 0.0

    for i in 1:n
        predict = dot(avg_sample, test_x[:, i])
        if predict > 0
            acc += (test_y[i] == 1)
        else
            acc += (test_y[i] == 0)
        end
    end

    return acc / n
end

function build_output_path(args::Dict)
    parts = [
        args["task"],
        args["method"],
        "pca$(args["pca_dim"])",
        "step$(args["stepsize"])",
        "n$(args["nsamples"])",
        "burn$(args["burnin"])",
        "seed$(args["seed"])",
    ]

    if args["method"] in ["tuna_mh", "tuna_sgld"]
        push!(parts, "lam$(args["lam"])")
    end
    if args["method"] == "tuna_sgld"
        push!(parts, "grad$(args["grad_size"])")
    end
    if args["method"] == "hmc"
        push!(parts, "lf$(args["leapfrog"])")
    end

    filename = join(parts, "_") * ".jld2"
    return joinpath(args["outdir"], filename)
end

function save_result(args::Dict, result::Dict)
    mkpath(args["outdir"])
    outfile = build_output_path(args)
    @save outfile result args
    return outfile
end

function finalize_trajectory(samples::Array{Float64,2},
                             test_x::Array{Float64,2},
                             test_y,
                             interval::Int,
                             total_runtime::Float64,
                             data_value)
    nsamples = size(samples, 2)
    K = Int(floor(nsamples / interval))
    acc = zeros(K)
    acc_time = zeros(K)
    datause = zeros(K)

    for k in 1:K
        upto = k * interval
        acc[k] = test_accuracy(samples[:, 1:upto], test_x, test_y)
        acc_time[k] = total_runtime
        datause[k] = data_value isa Function ? data_value(upto) : data_value
    end

    return acc, acc_time, datause
end