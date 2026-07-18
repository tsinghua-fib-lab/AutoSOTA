#!/usr/bin/env julia
# Iteration 9: ALGO-1 variation (1e-5 tolerances) + CODE-1+CODE-2

using Random, DynamicPolynomials, SumOfSquares, JuMP, Clarabel, LinearAlgebra, Statistics

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

function build_reduced_constraints(varlists_reduced::AbstractVector)
    allvars = reduce(vcat, varlists_reduced)
    zp = zero(allvars[1]); op = one(allvars[1])
    gs = typeof(zp)[]
    for vars in varlists_reduced
        for v in vars; push!(gs, v + zp); end
        push!(gs, op - sum(vars))
        last_var_expr = op - sum(vars)
        push!(gs, op - sum(vars.^2) - last_var_expr^2)
    end
    return basic_semialgebraic_set(FullSpace(), gs)
end

function solve_polynomial_sos_reduced(p, varlists; d_sel, tol, tol_gap_rel=1e-4, tol_gap_abs=1e-4, tol_feas=1e-4)
    t0 = time()
    reduced_varlists = [group[1:end-1] for group in varlists]
    p_sub = p
    for group in varlists
        p_sub = subs(p_sub, group[end] => 1 - sum(group[1:end-1]))
    end
    Sg_reduced = build_reduced_constraints(reduced_varlists)
    t_reduce = time() - t0
    model = SOSModel(Clarabel.Optimizer)
    set_optimizer_attribute(model, "verbose", false)
    set_optimizer_attribute(model, "tol_gap_rel", tol_gap_rel)
    set_optimizer_attribute(model, "tol_gap_abs", tol_gap_abs)
    set_optimizer_attribute(model, "tol_feas", tol_feas)
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

function main()
    println("="^60)
    println("Paper #2221 SOTA Iteration 9 - ALGO-1 (1e-4)")
    println("ALGO-1 variation: Clarabel tolerances 1e-8 → 1e-4")
    println("Includes CODE-1 + CODE-2")
    println("Setting: l=2, m=3, Du=4, d_sel=4, n=50")
    println("Solver: Clarabel (tol_gap_rel=1e-4, tol_gap_abs=1e-4, tol_feas=1e-4)")
    println("="^60)
    
    @polyvar x[1:3] y[1:3]
    varlists = [x, y]
    n_tests, seed = 50, 33
    
    n_success = 0
    solver_times = Float64[]
    total_times = Float64[]
    
    for i in 1:n_tests
        s = seed + i - 1
        p = random_sos_poly(varlists; degree=4, nterms=16, seed=s)
        nu, tval, status, t_reduce, t_solver, t_total = 
            solve_polynomial_sos_reduced(p, varlists; d_sel=4, tol=1e-1)
        push!(solver_times, t_solver)
        push!(total_times, t_total)
        if nu !== nothing; n_success += 1; end
        println("  [$i/$n_tests] val=$(round(tval,digits=4)) solver=$(round(t_solver,digits=4))s total=$(round(t_total,digits=4))s atoms=$(nu !== nothing)")
    end
    
    avg_solver = mean(solver_times)
    avg_total = mean(total_times)
    avg_solver_nojit = mean(solver_times[2:end])
    avg_total_nojit = mean(total_times[2:end])
    success_rate = n_success / n_tests * 100
    
    println()
    println("="^60)
    println("RESULTS")
    println("="^60)
    println("SOS_avg_solver_time_s (all):  $(round(avg_solver, digits=4))")
    println("SOS_avg_total_time_s (all):   $(round(avg_total, digits=4))")
    println("SOS_avg_solver_time_s (noJIT): $(round(avg_solver_nojit, digits=4))")
    println("SOS_avg_total_time_s (noJIT):  $(round(avg_total_nojit, digits=4))")
    println("SOS_success_rate_pct:          $(round(success_rate, digits=1))")
    println("SOS_success_count:             $n_success/$n_tests")
    println("="^60)
    
    open("/repo/eval_results.json", "w") do f
        write(f, "{\n")
        write(f, "  \"SOS_avg_solver_time_s\": $(round(avg_solver, digits=4)),\n")
        write(f, "  \"SOS_avg_total_time_s\": $(round(avg_total, digits=4)),\n")
        write(f, "  \"SOS_avg_solver_time_nojit_s\": $(round(avg_solver_nojit, digits=4)),\n")
        write(f, "  \"SOS_avg_total_time_nojit_s\": $(round(avg_total_nojit, digits=4)),\n")
        write(f, "  \"SOS_success_rate_pct\": $(round(success_rate, digits=1)),\n")
        write(f, "  \"SOS_success_count\": \"$n_success/$n_tests\"\n")
        write(f, "}\n")
    end
    println("Results also written to /repo/eval_results.json")
end

main()
