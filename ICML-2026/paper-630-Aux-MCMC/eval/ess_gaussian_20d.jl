cd(@__DIR__)

using Statistics
using JLD2, FileIO
using MCMCDiagnosticTools

# --------------------------
# settings
# --------------------------
base_dir = "../results/gaussian_20d"

function make_filename(round::Int, target_rate)
    return joinpath(
        base_dir,
        "20d-all-data_size100000-lam_const0.0005-target_rate$(target_rate)-beta1.0e-5-round$(round).jld2"
    )
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

rates_dict = detect_rate_and_rounds(
    base_dir,
    "20d-all-data_size100000-lam_const0.0005-"
)

if isempty(rates_dict)
    error("No matching result files found in $base_dir")
end

println("Detected target rates and rounds:")
for rate in sort(collect(keys(rates_dict)))
    println("  ", rate, " => ", rates_dict[rate])
end

# --------------------------
# helper: safe ESS/s
# --------------------------
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

# --------------------------
# samplers to evaluate
# --------------------------
samplers = [
    ("MH",            "mh_samples",          "time_mh"),
    ("MALA",          "mala_samples",        "time_mala"),
    ("PoisMH",        "poismh_samples",      "time_poismh"),
    ("PoisMH-Barker", "pois_barker_samples", "time_pois_barker"),
    ("PoisMALA",      "pois_mala_samples",   "time_pois_mala"),
    ("Barker",        "barker_samples",      "time_barker"),
    ("SGLD",          "sgld_samples",        "time_sgld"),
]

results = Dict{Tuple{Float64,String}, Vector{Vector{Float64}}}()
for rate in keys(rates_dict)
    for (label, _, _) in samplers
        results[(rate, label)] = Vector{Vector{Float64}}()
    end
end

found = Dict{Float64, Vector{Int}}()
failed = Dict{Float64, Vector{Int}}()
for rate in keys(rates_dict)
    found[rate] = Int[]
    failed[rate] = Int[]
end

# --------------------------
# main loop
# --------------------------
for rate in sort(collect(keys(rates_dict)))
    for i in rates_dict[rate]
        path = make_filename(i, rate)

        if !isfile(path)
            println("missing file: target_rate=$rate round=$i")
            continue
        end

        println("loading target_rate=$rate round=$i")

        try
            data = load(path)

            if !haskey(data, "result")
                error("missing key: result")
            end
            result = data["result"]

            push!(found[rate], i)

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

# --------------------------
# summarize
# --------------------------
println()
println("========== ESS/s summary ==========")

for rate in sort(collect(keys(rates_dict)))
    println()
    println("---- target_rate = $(rate) ----")

    if length(found[rate]) == 1
        println("Only one round detected; summary is for that single round.")
    end

    for (label, _, _) in samplers
        vecs = results[(rate, label)]

        if isempty(vecs)
            println("ESS/s: $label")
            println("No valid rounds found.")
            println()
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