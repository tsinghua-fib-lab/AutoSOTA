# Reproduction script for "Solving Imperfect-Recall Games via Sum-of-Squares Optimization"
# ICML 2026 Paper #2221
# 
# Replaces Mosek with Clarabel (free solver) 
# Experiment: 2 infosets, 3 actions each, degree=4, d_sel=4, 50 instances

using Random
using DynamicPolynomials
using SumOfSquares
using JuMP
using Clarabel
using LinearAlgebra
using Statistics

println("="^70)
println("Reproduction: Solving Imperfect-Recall Games via Sum-of-Squares Optimization")
println("Setup: 2 infosets, 3 actions/infoset, degree=4, d_sel=4, 50 instances")
println("="^70)

# ============================================================
# Polynomial generation (identical to notebook)
# ============================================================
function random_sos_poly(varlists::AbstractVector{<:AbstractVector}; degree::Int=8, nterms::Int=16, seed::Integer=33)
    Random.seed!(seed)
    allvars = reduce(vcat, varlists)
    n = length(allvars)
    poly = zero(allvars[1])
    exps_set = Set{Tuple{Vararg{Int}}}()
    while length(exps_set) < nterms
        d = rand(1:degree)
        exps = zeros(Int, n)
        for _ in 1:d
            idx = rand(1:n)
            exps[idx] += 1
        end
        if 1 <= sum(exps) <= degree
            push!(exps_set, tuple(exps...))
        end
    end
    for tup in exps_set
        coeff = round(rand()*2.0 - 1.0, digits=2)
        term = coeff * prod(allvars[i]^tup[i] for i in 1:n if tup[i] > 0)
        poly += term
    end
    poly += round(rand()*2.0 - 1.0, digits=2)
    return poly
end

# ============================================================
# Reduced-form constraints (identical to notebook)
# ============================================================
function build_reduced_constraints(varlists_reduced::AbstractVector)
    allvars = reduce(vcat, varlists_reduced)
    zp = zero(allvars[1])
    op = one(allvars[1])
    gs = typeof(zp)[]
    for vars in varlists_reduced
        for v in vars
            push!(gs, v + zp)
        end
        push!(gs, op - sum(vars))
        last_var_expr = op - sum(vars)
        push!(gs, op - sum(vars.^2) - last_var_expr^2)
    end
    return basic_semialgebraic_set(FullSpace(), gs)
end

# ============================================================
# SOS solver with Clarabel (replaces Mosek)
# ============================================================
function solve_polynomial_sos_reduced(p, varlists; d_sel, tol)
    t0 = time()
    
    p_sub = p
    reduced_varlists = [group[1:end-1] for group in varlists]
    for group in varlists
        p_sub = subs(p_sub, group[end] => 1 - sum(group[1:end-1]))
    end
    reduced_varlists = [group[1:end-1] for group in varlists]
    Sg_reduced = build_reduced_constraints(reduced_varlists)
    
    t_reduce = time() - t0
    
    # Use Clarabel instead of Mosek
    model = SOSModel(Clarabel.Optimizer)
    set_optimizer_attribute(model, "verbose", false)
    @variable(model, t)
    @objective(model, Min, t)
    @constraint(model, c, p_sub <= t, domain = Sg_reduced, maxdegree = d_sel)
    
    t_solver_start = time()
    optimize!(model)
    t_solver = time() - t_solver_start
    status = termination_status(model)
    
    tval = value(t)
    
    nu = moment_matrix(model[:c])
    nu_meas = atomic_measure(nu, tol)
    
    t_total = time() - t0
    return nu_meas, tval, status, t_reduce, t_solver, t_total
end

# ============================================================
# SOS test harness (identical to notebook)
# ============================================================
function test_random_polynomials_reduced(n_tests, varlists; degree, nterms, d_sel, tol, seed)
    sos_success = 0
    sos_fail = 0
    failing_tests = []
    records = []
    all_reduce = Float64[]
    all_solver = Float64[]
    all_total = Float64[]
    
    for i in 1:n_tests
        s = seed + i - 1
        p = random_sos_poly(varlists; degree=degree, nterms=nterms, seed=s)
        println("-"^60)
        println("Test $i — seed=$s [REDUCED]")
        nu, tval, status, t_reduce, t_solver, t_total = solve_polynomial_sos_reduced(p, varlists; d_sel=d_sel, tol=tol)
        push!(all_reduce, t_reduce)
        push!(all_solver, t_solver)
        push!(all_total, t_total)
        has_atoms = (nu !== nothing)
        if has_atoms
            println("  SOS extraction succeeded; val=$(round(tval,digits=6)); solver=$(round(t_solver,digits=4))s; total=$(round(t_total,digits=4))s")
            sos_success += 1
        else
            println("  SOS extraction failed or no atoms; bound val=$(round(tval,digits=6)); solver=$(round(t_solver,digits=4))s; total=$(round(t_total,digits=4))s")
            sos_fail += 1
            push!(failing_tests, (i=i, seed=s, tval=tval, status=status))
        end
        push!(records, (i, s, tval, has_atoms, t_reduce, t_solver, t_total, status))
    end
    
    println()
    println("="^60)
    println("SUMMARY [REDUCED]: SOS found atoms: $sos_success / $n_tests ; failed: $sos_fail / $n_tests")
    println("Reduce time: total=$(round(sum(all_reduce), digits=2))s, avg=$(round(mean(all_reduce), digits=4))s")
    println("Solver time: total=$(round(sum(all_solver), digits=2))s, avg=$(round(mean(all_solver), digits=4))s")
    println("Total time:  total=$(round(sum(all_total), digits=2))s, avg=$(round(mean(all_total), digits=4))s")
    if !isempty(failing_tests)
        println("Failing tests:")
        for fail in failing_tests
            println("  Test $(fail.i) (seed $(fail.seed)): bound = $(round(fail.tval, digits=6)), status = $(fail.status)")
        end
    end
    
    return records, all_solver, all_total, sos_success, sos_fail
end

# ============================================================
# PGD Utilities (identical to notebook)
# ============================================================
function proj_simplex(v::AbstractVector{<:Real})
    v = collect(float.(v))
    n = length(v)
    u = sort(v, rev=true)
    cssv = cumsum(u)
    rho = findlast(i -> u[i] + (1 - cssv[i]) / i > 0, 1:n)
    if rho === nothing
        return fill(1.0/n, n)
    end
    theta = (cssv[rho] - 1) / rho
    return max.(v .- theta, 0.0)
end

function project_onto_feasible(vars::AbstractVector, varlists)
    out = similar(vars, Float64)
    idx = 1
    for group in varlists
        n = length(group)
        out[idx:idx+n-1] .= proj_simplex(vars[idx:idx+n-1])
        idx += n
    end
    return out
end

function poly_value(p, allvars, vals)
    return convert(Float64, subs(p, allvars => vals))
end

function poly_grad_polys(p, allvars)
    return [differentiate(p, v) for v in allvars]
end

function poly_grad_value(grad_polys, allvars, vals)
    return [convert(Float64, subs(g, allvars => vals)) for g in grad_polys]
end

function pgd_maximize_polynomial(p, varlists;
    learning_rate=0.02, max_iterations=5000, tolerance=1e-8, seed=1)
    allvars = reduce(vcat, varlists)
    grad_polys = poly_grad_polys(p, allvars)
    
    Random.seed!(seed)
    parts = Vector{Float64}()
    for group in varlists
        v = rand(length(group))
        v ./= sum(v)
        append!(parts, v)
    end
    vars = project_onto_feasible(parts, varlists)
    
    start_time = time()
    for iter in 1:max_iterations
        g = poly_grad_value(grad_polys, allvars, vars)
        vars_new = vars + learning_rate .* g  # ascent on p
        vars_new = project_onto_feasible(vars_new, varlists)
        if norm(vars_new - vars) < tolerance
            vars = vars_new
            break
        end
        vars = vars_new
    end
    elapsed = time() - start_time
    final_val = poly_value(p, allvars, vars)
    return vars, final_val, elapsed
end

# ============================================================
# Main experiment
# ============================================================
println("\n=== Setting up variables ===")
@polyvar x[1:3] y[1:3]
varlists = [x, y]  # 2 infosets, 3 actions each
n_tests = 50
seed = 33

println("Variables defined: x[1:3], y[1:3]")
println("n_instances = $n_tests, seed = $seed")
println("degree = 4, nterms = 16, d_sel = 4, tol = 0.1")

# Run SOS experiment
println("\n=== Running SOS Experiment ===")
println("(This may take a while for 50 instances...)")
sos_records, sos_solver_times, sos_total_times, sos_success, sos_fail = 
    test_random_polynomials_reduced(n_tests, varlists; degree=4, nterms=16, d_sel=4, tol=1e-1, seed=seed)

sos_avg_solver_time = mean(sos_solver_times)
sos_avg_total_time = mean(sos_total_times)
sos_success_rate = sos_success / n_tests * 100.0

println("\n=== Running PGD Baseline ===")
println("(100 PGD restarts per instance...)")

# PGD test
pgd_n_runs = 100
pgd_all_times = Float64[]
pgd_all_rates = Float64[]

for i in 1:n_tests
    s = seed + i - 1
    p = random_sos_poly(varlists; degree=4, nterms=16, seed=s)
    
    # Get SOS optimum value as ground truth
    p_sub = p
    for group in varlists
        p_sub = subs(p_sub, group[end] => 1 - sum(group[1:end-1]))
    end
    reduced_varlists_pgd = [group[1:end-1] for group in varlists]
    Sg_reduced_pgd = build_reduced_constraints(reduced_varlists_pgd)
    model_gt = SOSModel(Clarabel.Optimizer)
    set_optimizer_attribute(model_gt, "verbose", false)
    @variable(model_gt, t_gt)
    @objective(model_gt, Min, t_gt)
    @constraint(model_gt, c_gt, p_sub <= t_gt, domain = Sg_reduced_pgd, maxdegree = 4)
    optimize!(model_gt)
    sos_opt_val = value(t_gt)
    
    # PGD: negate polynomial for maximization
    p_neg = -p
    results = []
    for r in 1:pgd_n_runs
        vars, val, elapsed = pgd_maximize_polynomial(p_neg, varlists; 
            learning_rate=0.02, max_iterations=5000, tolerance=1e-8, seed=s*1000 + r)
        push!(results, (val=val, vars=vars, elapsed=elapsed))
    end
    
    max_val = maximum(r.val for r in results)
    total_time = sum(r.elapsed for r in results)
    avg_time = total_time / pgd_n_runs
    # Check how many PGD runs reach within tolerance of SOS optimum
    optimum_count = count(r -> abs(r.val - (-sos_opt_val)) <= 0.01, results)
    rate = optimum_count / pgd_n_runs
    
    push!(pgd_all_times, avg_time)
    push!(pgd_all_rates, rate)
    
    if i <= 5 || i % 10 == 0
        println("PGD test $i: optimum rate=$(round(rate*100, digits=2))%, avg time/run=$(round(avg_time, digits=4))s")
    end
end

pgd_avg_time = mean(pgd_all_times)
pgd_avg_rate = mean(pgd_all_rates) * 100.0
println("\nPGD Overall: avg time/run=$(round(pgd_avg_time, digits=4))s, avg optimum rate=$(round(pgd_avg_rate, digits=2))%")

# ============================================================
# FINAL METRICS
# ============================================================
println("\n" * "="^70)
println("FINAL REPRODUCTION METRICS")
println("="^70)
println("Experiment: ℓ=2 infosets, m=3 actions, D_u=4, d_sel=4, n=50 instances")
println()
println("SOS Time (s):")
println("  Paper:  0.02")
println("  Ours:   $(round(sos_avg_total_time, digits=4))")
println()
println("SOS Success Rate (%):")
println("  Paper:  100.0")
println("  Ours:   $(round(sos_success_rate, digits=2))")
println()
println("PGD Time (s):")
println("  Paper:  0.05")
println("  Ours:   $(round(pgd_avg_time, digits=4))")
println()
println("PGD Success Rate (%):")
println("  Paper:  80.72")
println("  Ours:   $(round(pgd_avg_rate, digits=2))")
println()
println("="^70)
println("Reproduction complete.")

# Write results to file
open("/repo/reproduction_results.txt", "w") do f
    write(f, "SOS_avg_total_time_s: $(round(sos_avg_total_time, digits=4))\n")
    write(f, "SOS_avg_solver_time_s: $(round(sos_avg_solver_time, digits=4))\n")
    write(f, "SOS_success_rate_pct: $(round(sos_success_rate, digits=2))\n")
    write(f, "PGD_avg_time_s: $(round(pgd_avg_time, digits=4))\n")
    write(f, "PGD_success_rate_pct: $(round(pgd_avg_rate, digits=2))\n")
    write(f, "SOS_success_count: $sos_success/$n_tests\n")
    for (i, rec) in enumerate(sos_records)
        write(f, "instance_$(i)_seed_$(rec[2])_has_atoms_$(rec[4])_solver_time_$(round(rec[6], digits=4))_total_time_$(round(rec[7], digits=4))\n")
    end
end
println("\nResults written to /repo/reproduction_results.txt")
