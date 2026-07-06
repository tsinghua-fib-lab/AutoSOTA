cd(@__DIR__)

using JLD2, FileIO
using Statistics
using StatsBase
using Random
using Printf

# ============================================================
# Settings
# ============================================================

const BASE_DIR = "../results/gaussian_20d"

const ALGOS = ["MH", "MALA", "PoisMH", "PoisMH-Barker", "PoisMALA", "Barker", "SGLD"]

const BURNINS = Dict(
    "MH" => 1000,
    "MALA" => 500,
    "PoisMH" => 10000,
    "PoisMH-Barker" => 4000,
    "PoisMALA" => 4000,
    "Barker" => 1000,
    "SGLD" => 500000,
)

const THINS = Dict(
    "MH" => 1,
    "MALA" => 1,
    "PoisMH" => 1,
    "PoisMH-Barker" => 1,
    "PoisMALA" => 1,
    "Barker" => 1,
    "SGLD" => 1,
)

# ============================================================
# Filename / round detection
# ============================================================

function result_filename(target_rate::Float64, round::Int)
    return joinpath(
        BASE_DIR,
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
    BASE_DIR,
    "20d-all-data_size100000-lam_const0.0005-"
)

if isempty(rates_dict)
    error("No matching result files found in $BASE_DIR")
end

println("Detected target rates and rounds:")
for rate in sort(collect(keys(rates_dict)))
    println("  ", rate, " => ", rates_dict[rate])
end

# ============================================================
# helpers
# ============================================================

function post_burnin_thin(samples; burnin=0, thin=1)
    T = size(samples, 2)
    idx = collect((burnin + 1):thin:T)
    return samples[:, idx]
end

function equalize_sample_sizes(A, B; seed=1234)
    Random.seed!(seed)
    nA = size(A, 2)
    nB = size(B, 2)

    if nA > nB
        idx = sample(1:nA, nB; replace=false)
        return A[:, idx], B
    elseif nB > nA
        idx = sample(1:nB, nA; replace=false)
        return A, B[:, idx]
    else
        return A, B
    end
end

# ============================================================
# KS
# ============================================================

function ks2_statistic(x, y)
    xs = sort(x)
    ys = sort(y)

    n, m = length(xs), length(ys)
    i, j = 1, 1
    Fn, Gm, D = 0.0, 0.0, 0.0

    while i <= n && j <= m
        if xs[i] < ys[j]
            Fn = i / n
            D = max(D, abs(Fn - Gm))
            i += 1
        elseif ys[j] < xs[i]
            Gm = j / m
            D = max(D, abs(Fn - Gm))
            j += 1
        else
            v = xs[i]
            while i <= n && xs[i] == v
                i += 1
            end
            while j <= m && ys[j] == v
                j += 1
            end
            Fn = (i - 1) / n
            Gm = (j - 1) / m
            D = max(D, abs(Fn - Gm))
        end
    end

    return D
end

function marginal_ks(exact, chain)
    d = size(exact, 1)
    D = zeros(d)
    for j in 1:d
        D[j] = ks2_statistic(view(exact, j, :), view(chain, j, :))
    end
    return (
        min_D = minimum(D),
        median_D = median(D),
        max_D = maximum(D),
        mean_D = mean(D),
    )
end

# ============================================================
# load file
# ============================================================

function load_one_file(filepath)
    obj = load(filepath)

    if !haskey(obj, "result")
        error("missing key: result")
    end
    result = obj["result"]

    exact = result["theta_true"]

    chains = Dict(
        "MH"            => result["mh_samples"],
        "MALA"          => result["mala_samples"],
        "PoisMH"        => result["poismh_samples"],
        "PoisMH-Barker" => result["pois_barker_samples"],
        "PoisMALA"      => result["pois_mala_samples"],
        "Barker"        => result["barker_samples"],
        "SGLD"          => result["sgld_samples"],
    )

    return exact, chains
end

# ============================================================
# run one file
# ============================================================

function run_one(filepath; seed=1234)
    exact, chains = load_one_file(filepath)
    results = Dict()

    for algo in ALGOS
        chain = chains[algo]

        chain_use = post_burnin_thin(
            chain;
            burnin=BURNINS[algo],
            thin=THINS[algo]
        )

        exact_use, chain_use = equalize_sample_sizes(exact, chain_use; seed=seed)

        results[algo] = marginal_ks(exact_use, chain_use)
    end

    return results
end

# ============================================================
# batch
# ============================================================

function run_all()
    all = Dict()

    for rate in sort(collect(keys(rates_dict)))
        rounds = rates_dict[rate]
        println("Processing target_rate = ", rate, ", rounds = ", rounds)

        for rd in rounds
            file = result_filename(rate, rd)
            if isfile(file)
                println("Running target_rate=$(rate), round $(rd)")
                all[(rate, rd)] = run_one(file; seed=1000 + rd)
            end
        end
    end

    return all
end

# ============================================================
# markdown summary
# ============================================================

function summarize(all)
    summary = Dict()

    for algo in ALGOS
        summary[algo] = Dict()
        for rate in sort(collect(keys(rates_dict)))
            mins = Float64[]
            meds = Float64[]
            maxs = Float64[]

            rounds = sort([rd for (tr, rd) in keys(all) if tr == rate])

            for rd in rounds
                key = (rate, rd)
                if haskey(all, key)
                    res = all[key][algo]
                    push!(mins, res.min_D)
                    push!(meds, res.median_D)
                    push!(maxs, res.max_D)
                end
            end

            if !isempty(mins)
                summary[algo][rate] = (mean(mins), mean(meds), mean(maxs))
            end
        end
    end

    return summary
end

function print_table(summary)
    rates = sort(collect(keys(rates_dict)))

    header = "| Algorithm "
    sep = "|----------"
    for rate in rates
        header *= "| $(rate) "
        sep *= "|------"
    end
    println(header * "|")
    println(sep * "|")

    for algo in ALGOS
        row = "| $algo "
        for rate in rates
            if haskey(summary[algo], rate)
                a, b, c = summary[algo][rate]
                row *= "| (" *
                       @sprintf("%.3f", a) * ", " *
                       @sprintf("%.3f", b) * ", " *
                       @sprintf("%.3f", c) * ") "
            else
                row *= "| NA "
            end
        end
        println(row * "|")
    end
end

# ============================================================
# run
# ============================================================

all = run_all()
summary = summarize(all)

println("\n=== KS summary ===")
print_table(summary)
