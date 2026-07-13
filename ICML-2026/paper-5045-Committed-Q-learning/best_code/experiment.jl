#!/usr/bin/env julia
using Random

include("qcorridor.jl")

function check_optimal(q_slice)
    return (q_slice[1, 2] > q_slice[1, 1]) && (q_slice[2, 2] > q_slice[2, 1])
end

function parse_args()
    kwargs = Dict{Symbol, Any}()
    for arg in ARGS
        parts = split(arg, "=")
        if length(parts) == 2
            key = Symbol(strip(parts[1]))
            val = strip(parts[2])
            kwargs[key] = tryparse(Int, val) !== nothing ? parse(Int, val) :
                          tryparse(Float64, val) !== nothing ? parse(Float64, val) : val
        end
    end
    return kwargs
end

function run_experiment(; k=5, T=1000, n_runs=1000, alpha0=0.1, alphaT=0.01, eps0=0.1, epsT=0.01, q_init=0.0, bonus_beta=0.0, ucb_c=0.0)
    optimal_com = zeros(Int, n_runs)
    optimal_nocom = zeros(Int, n_runs)

    for seed in 1:n_runs
        _, qs_com = qcorridor(T, k, true, seed, alpha0, alphaT, eps0, epsT, 1, q_init, bonus_beta, ucb_c)
        q_com = qs_com[1, :, :]
        optimal_com[seed] = check_optimal(q_com) ? 1 : 0

        _, qs_nocom = qcorridor(T, k, false, seed, alpha0, alphaT, eps0, epsT, 1, q_init, bonus_beta, ucb_c)
        q_nocom = qs_nocom[1, :, :]
        optimal_nocom[seed] = check_optimal(q_nocom) ? 1 : 0
    end

    frac_com = sum(optimal_com) / n_runs
    frac_nocom = sum(optimal_nocom) / n_runs

    println("=== Results: Corridor k=$k, T=$T, runs=$n_runs ===")
    println("Committed Q-learning:  $(sum(optimal_com))/$n_runs = $frac_com")
    println("Regular Q-learning:    $(sum(optimal_nocom))/$n_runs = $frac_nocom")

    # Bootstrap 95% CI
    rng = Xoshiro(42)
    n_bootstrap = 10000
    boot_com = zeros(n_bootstrap)
    boot_nocom = zeros(n_bootstrap)
    for b in 1:n_bootstrap
        idx = rand(rng, 1:n_runs, n_runs)
        boot_com[b] = sum(optimal_com[idx]) / n_runs
        boot_nocom[b] = sum(optimal_nocom[idx]) / n_runs
    end
    sort!(boot_com)
    sort!(boot_nocom)

    println("Committed 95% bootstrap CI: [$(round(boot_com[250], digits=6)), $(round(boot_com[9750], digits=6))]")
    println("Regular 95% bootstrap CI:   [$(round(boot_nocom[250], digits=6)), $(round(boot_nocom[9750], digits=6))]")

    return frac_com, frac_nocom
end

args = parse_args()
run_experiment(; k=get(args, :k, 5), T=get(args, :T, 1000),
    n_runs=get(args, :n_runs, 1000), alpha0=get(args, :alpha0, 0.1),
    alphaT=get(args, :alphaT, 0.01), eps0=get(args, :eps0, 0.1),
    epsT=get(args, :epsT, 0.01), q_init=get(args, :q_init, 0.0),
    bonus_beta=get(args, :bonus_beta, 0.0), ucb_c=get(args, :ucb_c, 0.0))
