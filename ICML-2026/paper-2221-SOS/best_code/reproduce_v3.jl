# Reproduction script v3 - Proper scoping
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

function solve_polynomial_sos_reduced(p, varlists; d_sel, tol)
    t0 = time()
    p_sub = p
    for group in varlists
        p_sub = subs(p_sub, group[end] => 1 - sum(group[1:end-1]))
    end
    reduced_varlists = [group[1:end-1] for group in varlists]
    Sg_reduced = build_reduced_constraints(reduced_varlists)
    t_reduce = time() - t0
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

function proj_simplex(v::AbstractVector{<:Real})
    v = collect(float.(v)); n = length(v)
    u = sort(v, rev=true); cssv = cumsum(u)
    rho = findlast(i -> u[i] + (1 - cssv[i]) / i > 0, 1:n)
    isnothing(rho) && return fill(1.0/n, n)
    theta = (cssv[rho] - 1) / rho
    return max.(v .- theta, 0.0)
end

function project_onto_feasible(vars, varlists)
    out = similar(vars, Float64); idx = 1
    for group in varlists
        n = length(group)
        out[idx:idx+n-1] .= proj_simplex(vars[idx:idx+n-1])
        idx += n
    end
    return out
end

function pgd_maximize_polynomial(p, varlists; learning_rate=0.02, max_iterations=5000, tolerance=1e-8, seed=1)
    allvars = reduce(vcat, varlists)
    grad_polys = [differentiate(p, v) for v in allvars]
    Random.seed!(seed)
    parts = Float64[]
    for group in varlists
        v = rand(length(group)); v ./= sum(v)
        append!(parts, v)
    end
    vars = project_onto_feasible(parts, varlists)
    for iter in 1:max_iterations
        g = [convert(Float64, subs(gp, allvars => vars)) for gp in grad_polys]
        vars_new = vars + learning_rate .* g
        vars_new = project_onto_feasible(vars_new, varlists)
        if norm(vars_new - vars) < tolerance
            vars = vars_new; break
        end
        vars = vars_new
    end
    final_val = convert(Float64, subs(p, allvars => vars))
    return vars, final_val
end

function main()
    println("="^70)
    println("Reproduction: Solving Imperfect-Recall Games via Sum-of-Squares Optimization")
    println("Setup: 2 infosets, 3 actions/infoset, degree=4, d_sel=4, 50 instances")
    println("Solver: Clarabel (free, replacing Mosek)")
    println("="^70)
    
    @polyvar x[1:3] y[1:3]
    varlists = [x, y]
    n_tests, seed = 50, 33
    
    # SOS Experiment
    println("\n=== Running SOS Experiment (50 instances) ===")
    sos_success = 0
    sos_solver_times = Float64[]
    sos_total_times = Float64[]
    sos_opt_vals = Float64[]
    
    for i in 1:n_tests
        s = seed + i - 1
        p = random_sos_poly(varlists; degree=4, nterms=16, seed=s)
        nu, tval, status, t_reduce, t_solver, t_total = 
            solve_polynomial_sos_reduced(p, varlists; d_sel=4, tol=1e-1)
        push!(sos_solver_times, t_solver)
        push!(sos_total_times, t_total)
        push!(sos_opt_vals, tval)
        if nu !== nothing; sos_success += 1; end
        if i <= 3 || i % 10 == 0
            println("SOS $i/$n_tests: val=$(round(tval,digits=4)), solver=$(round(t_solver,digits=4))s, total=$(round(t_total,digits=4))s, atoms=$(nu !== nothing)")
        end
    end
    
    sos_avg_total = mean(sos_total_times)
    sos_avg_solver = mean(sos_solver_times)
    sos_success_rate = sos_success / n_tests * 100
    println("\nSOS Summary: atoms found in $sos_success/$n_tests ($(round(sos_success_rate,digits=1))%)")
    println("SOS avg solver time: $(round(sos_avg_solver, digits=4))s")
    println("SOS avg total time: $(round(sos_avg_total, digits=4))s")
    
    # PGD Baseline
    println("\n=== Running PGD Baseline (100 restarts/instance) ===")
    pgd_all_rates = Float64[]
    
    for i in 1:n_tests
        s = seed + i - 1
        p = random_sos_poly(varlists; degree=4, nterms=16, seed=s)
        sos_opt = sos_opt_vals[i]
        
        results = Float64[]
        for r in 1:100
            vars, val = pgd_maximize_polynomial(p, varlists; 
                learning_rate=0.02, max_iterations=5000, tolerance=1e-8, seed=s*1000 + r)
            push!(results, val)
        end
        
        max_val = maximum(results)
        # Notebook definition: within 0.001 of the best PGD run
        optimum_count = count(v -> v >= max_val - 0.001, results)
        rate = optimum_count / 100
        push!(pgd_all_rates, rate)
        
        if i <= 3 || i % 10 == 0
            println("PGD $i/$n_tests: best_PGD=$(round(max_val,digits=4)), SOS_ub=$(round(sos_opt,digits=4)), rate=$(round(rate*100,digits=1))%")
        end
    end
    
    pgd_avg_rate = mean(pgd_all_rates) * 100
    
    # Final report
    println("\n" * "="^70)
    println("FINAL REPRODUCTION METRICS")
    println("="^70)
    println("Experiment: l=2 infosets, m=3 actions, Du=4, d_sel=4, n=$n_tests instances")
    println("Solver: Clarabel (replaces Mosek)")
    println()
    println("SOS Time (s)        - Paper: 0.02   | Ours: $(round(sos_avg_total, digits=4)) (avg total)")
    println("                                 | Ours: $(round(sos_avg_solver, digits=4)) (avg solver)")
    println("SOS Success Rate    - Paper: 100.0% | Ours: $(round(sos_success_rate, digits=1))%")
    println("PGD Success Rate    - Paper: 80.72% | Ours: $(round(pgd_avg_rate, digits=1))%")
    println()
    println("="^70)
    
    # Write results
    open("/repo/reproduction_results.txt", "w") do f
        write(f, "SOS_avg_total_time_s=$(round(sos_avg_total, digits=4))\n")
        write(f, "SOS_avg_solver_time_s=$(round(sos_avg_solver, digits=4))\n")
        write(f, "SOS_success_rate_pct=$(round(sos_success_rate, digits=1))\n")
        write(f, "SOS_success_count=$sos_success/$n_tests\n")
        write(f, "PGD_avg_rate_pct=$(round(pgd_avg_rate, digits=1))\n")
        write(f, "n_instances=$n_tests\n")
        for i in 1:n_tests
            write(f, "instance_$i: sos_val=$(round(sos_opt_vals[i],digits=4))_solver_time=$(round(sos_solver_times[i],digits=4))_total_time=$(round(sos_total_times[i],digits=4))\n")
        end
    end
    println("Results written to /repo/reproduction_results.txt")
    return sos_avg_total, sos_avg_solver, sos_success_rate, pgd_avg_rate
end

main()
