using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DisjunctiveNet

println("=== Large-scale formulation comparison ===")

dm = DisjunctiveModel(4)

set_bounds!(
    dm,
    lower = zeros(4),
    upper = ones(4),
)

# Global constraints.
add_linear_constraint!(dm, [1.0, 1.0, 1.0, 1.0], :>=, 1.2)
add_linear_constraint!(dm, [1.0, 1.0, 1.0, 1.0], :<=, 2.8)

# Hard-coded disjunct counts across 10 rules.
# This intentionally creates a very large full-DNF expansion.
disjunct_counts = [2, 3, 2, 4, 2, 3, 2, 5, 2, 3]

function make_disjunct(rule_index::Int, disj_index::Int, nvars::Int)
    v = ((rule_index - 1) % nvars) + 1

    if disj_index == 1
        return [LinearConstraint([j == v ? 1.0 : 0.0 for j in 1:nvars], :<=, 0.20)]
    elseif disj_index == 2
        return [LinearConstraint([j == v ? 1.0 : 0.0 for j in 1:nvars], :>=, 0.70)]
    elseif disj_index == 3
        return [
            LinearConstraint([j == v ? 1.0 : 0.0 for j in 1:nvars], :>=, 0.35),
            LinearConstraint([j == v ? 1.0 : 0.0 for j in 1:nvars], :<=, 0.55),
        ]
    elseif disj_index == 4
        w = (v % nvars) + 1
        return [LinearConstraint([j == v || j == w ? 1.0 : 0.0 for j in 1:nvars], :<=, 0.95)]
    elseif disj_index == 5
        w = (v % nvars) + 1
        return [LinearConstraint([j == v ? 1.0 : (j == w ? -1.0 : 0.0) for j in 1:nvars], :>=, 0.10)]
    else
        error("Unexpected disjunct index")
    end
end

for (r, nd) in enumerate(disjunct_counts)
    disjuncts = [make_disjunct(r, d, 4) for d in 1:nd]
    add_disjunction!(dm, disjuncts...; name = Symbol("rule_", r))
end

rule_order = [Symbol("rule_", r) for r in 1:length(disjunct_counts)]

full_dnf_scenarios = product_disjunct_count(disjunct_counts)

println()
println("=== Formulation summaries ===")
println(formulation_summary(dm; formulation = :cnf))
println(formulation_summary(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 2,
    rule_ordering = rule_order,
))
println(formulation_summary(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 4,
    rule_ordering = rule_order,
))

yhat = [0.45, 0.25, 0.75, 0.40]

println()
println("Raw yhat = ", yhat)

println()
println("=== Projection timing and model sizes ===")
benchmark_projection(dm, yhat; formulation = :cnf, label = "CNF")

benchmark_projection(
    dm,
    yhat;
    formulation = :partial_dnf,
    num_dnf_rules = 2,
    rule_ordering = rule_order,
    label = "partial-DNF k=2",
)

benchmark_projection(
    dm,
    yhat;
    formulation = :partial_dnf,
    num_dnf_rules = 4,
    rule_ordering = rule_order,
    label = "partial-DNF k=4",
)

# Full DNF can be enormous. Keep it opt-in for large scenario counts.
max_full_dnf_scenarios = 10_000
run_full_dnf =
    full_dnf_scenarios <= max_full_dnf_scenarios ||
    get(ENV, "DDL_RUN_FULL_DNF", "false") == "true"

if !run_full_dnf
    println(
        rpad("full DNF", 18),
        " skipped because scenario_count=",
        full_dnf_scenarios,
        " > ",
        max_full_dnf_scenarios,
        ". Set ENV[\"DDL_RUN_FULL_DNF\"]=\"true\" to force it.",
    )
else
    benchmark_projection(dm, yhat; formulation = :dnf, label = "full DNF")
end

println()
println("Note: first Julia run may include compilation overhead. Run the script twice for steadier timings.")