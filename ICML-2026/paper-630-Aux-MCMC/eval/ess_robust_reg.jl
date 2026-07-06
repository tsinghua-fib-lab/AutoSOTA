cd(@__DIR__)

using Statistics
using ArgParse
using JLD2, FileIO
using MCMCDiagnosticTools

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--experiment"
            help = "experiment name"
            arg_type = String
            required = true

        "--base_dir"
            help = "results directory; default is ../results/<experiment>"
            arg_type = String
            default = ""
    end
    return parse_args(s)
end

function detect_rate_and_rounds(base_dir::String, prefix_start::String)
    rates = Dict{Float64, Vector{Int}}()

    if !isdir(base_dir)
        error("Directory not found: $base_dir")
    end

    for f in readdir(base_dir)
        if startswith(f, prefix_start) && endswith(f, ".jld2")
            m = match(r"target_rate([0-9.]+).*round(\d+)\.jld2$", f)
            if m !== nothing
                rate = parse(Float64, m.captures[1])
                rd = parse(Int, m.captures[2])
                if !haskey(rates, rate)
                    rates[rate] = Int[]
                end
                push!(rates[rate], rd)
            end
        end
    end

    for k in keys(rates)
        rates[k] = sort(unique(rates[k]))
    end

    return rates
end

function ess_per_sec(samples, time_vec; burn_frac=0.0)
    nsamples = size(samples, 2)
    d = size(samples, 1)

    burn = Int(floor(nsamples * burn_frac))
    start_idx = burn + 1

    if start_idx >= nsamples
        error("burn_frac too large: no samples left")
    end

    total_time = time_vec[end] - (burn > 0 ? time_vec[burn] : 0.0)
    if total_time <= 0
        error("non-positive effective runtime")
    end

    out = zeros(d)
    for j in 1:d
        out[j] = ess(samples[j, start_idx:end]) / total_time
    end
    return out
end

function main()
    args = parse_commandline()
    experiment = args["experiment"]
    base_dir = isempty(args["base_dir"]) ? joinpath("..", "results", experiment) : args["base_dir"]
    prefix_start = experiment * "-"

    rates_dict = detect_rate_and_rounds(base_dir, prefix_start)
    if isempty(rates_dict)
        error("No matching result files found in $base_dir")
    end

    println("Detected target rates and rounds:")
    for rate in sort(collect(keys(rates_dict)))
        println("  ", rate, " => ", rates_dict[rate])
    end

    base_samplers = [
        ("MH",            "mh_samples",          "time_mh"),
        ("MALA",          "mala_samples",        "time_mala"),
        ("PoisMH",        "poismh_samples",      "time_pois"),
        ("PoisMH-Barker", "pois_barker_samples", "time_pois_barker"),
        ("PoisMALA",      "pois_mala_samples",   "time_pois_mala"),
        ("HMC",           "hmc_samples",         "time_hmc"),
        ("Barker",        "barker_samples",      "time_barker"),
        ("SGLD",          "sgld_samples",        "time_sgld"),
    ]

    results = Dict{Tuple{Float64,String}, Vector{Vector{Float64}}}()
    for rate in keys(rates_dict)
        for (label, _, _) in base_samplers
            results[(rate, label)] = Vector{Vector{Float64}}()
        end
    end

    found = Dict{Float64, Vector{Int}}()
    failed = Dict{Float64, Vector{Int}}()
    for rate in keys(rates_dict)
        found[rate] = Int[]
        failed[rate] = Int[]
    end

    for rate in sort(collect(keys(rates_dict)))
        for i in rates_dict[rate]
            path = joinpath(base_dir, "$(experiment)-" * "data_size" * string(0))
            # Use exact filename search to preserve original naming pattern.
            candidates = filter(f -> occursin("target_rate$(rate)", f) && occursin("round$(i).jld2", f) && startswith(f, prefix_start), readdir(base_dir))
            if isempty(candidates)
                println("missing file: target_rate=$rate round=$i")
                continue
            end
            path = joinpath(base_dir, candidates[1])

            println("loading target_rate=$rate round=$i")
            try
                data = load(path)
                if !haskey(data, "result")
                    error("missing key: result")
                end
                result = data["result"]

                push!(found[rate], i)

                samplers = copy(base_samplers)
                if !haskey(result, "sgld_samples")
                    samplers = filter(x -> x[1] != "SGLD", samplers)
                end

                for (label, sample_key, time_key) in samplers
                    if haskey(result, sample_key) && haskey(result, time_key)
                        samples = result[sample_key]
                        time_vec = result[time_key]
                        ess_vec = ess_per_sec(samples, time_vec; burn_frac=0.0)
                        push!(results[(rate, label)], ess_vec)
                    else
                        println("  warning: $(label) missing in target_rate=$rate round=$i")
                    end
                end
            catch err
                println("failed to load/process target_rate=$rate round=$i")
                println(err)
                push!(failed[rate], i)
                continue
            end
        end
    end

    println()
    println("========== Processed Rounds ==========")
    for rate in sort(collect(keys(rates_dict)))
        println("target_rate = ", rate)
        println("  successful: ", found[rate])
        println("  failed    : ", failed[rate])
    end

    println()
    println("========== ESS/s summary ==========")

    for rate in sort(collect(keys(rates_dict)))
        println()
        println("---- target_rate = $(rate) ----")

        if length(found[rate]) == 1
            println("Only one round detected; summary is for that single round.")
        end

        for (label, _, _) in base_samplers
            vecs = results[(rate, label)]
            if isempty(vecs)
                continue
            end
            mat = hcat(vecs...)
            ess_avg = vec(mean(mat, dims=2))
            println("ESS/s: $label")
            println(
                "Min: ", round(minimum(ess_avg), digits=3),
                " Median: ", round(median(ess_avg), digits=3),
                " Max: ", round(maximum(ess_avg), digits=3)
            )
            println()
        end
    end
end

main()
