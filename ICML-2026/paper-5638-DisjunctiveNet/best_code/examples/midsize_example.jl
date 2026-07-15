using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DisjunctiveNet
using Gurobi
import MathOptInterface as MOI

const GRB_ENV = Gurobi.Env()
gurobi_solver = () -> begin
    opt = Gurobi.Optimizer(GRB_ENV)
    #MOI.set(opt, MOI.RawOptimizerAttribute("OutputFlag"), 0)
    #MOI.set(opt, MOI.RawOptimizerAttribute("TimeLimit"), 60.0)
    return opt
end

println("=== Midsize formulation comparison ===")

dm = DisjunctiveModel(3)

set_bounds!(
    dm,
    lower = [0.0, 0.0, 0.0],
    upper = [1.0, 1.0, 1.0],
)

# Global constraints.
add_linear_constraint!(dm, [1.0, 1.0, 1.0], :>=, 1.0)
add_linear_constraint!(dm, [1.0, 1.0, 1.0], :<=, 2.0)

# Rule 1: 2 disjuncts
add_disjunction!(
    dm,
    [LinearConstraint([1.0, 0.0, 0.0], :<=, 0.25)],
    [LinearConstraint([1.0, 0.0, 0.0], :>=, 0.70)];
    name = :x_low_or_high,
)

# Rule 2: 2 disjuncts
add_disjunction!(
    dm,
    [LinearConstraint([0.0, 1.0, 0.0], :<=, 0.30)],
    [LinearConstraint([0.0, 1.0, 0.0], :>=, 0.65)];
    name = :y_low_or_high,
)

# Rule 3: 3 disjuncts
add_disjunction!(
    dm,
    [LinearConstraint([0.0, 0.0, 1.0], :<=, 0.20)],
    [
        LinearConstraint([0.0, 0.0, 1.0], :>=, 0.40),
        LinearConstraint([0.0, 0.0, 1.0], :<=, 0.60),
    ],
    [LinearConstraint([0.0, 0.0, 1.0], :>=, 0.80)];
    name = :z_low_mid_or_high,
)

# Rule 4: 2 disjuncts
add_disjunction!(
    dm,
    [LinearConstraint([1.0, 1.0, 0.0], :<=, 0.80)],
    [LinearConstraint([1.0, 1.0, 0.0], :>=, 1.10)];
    name = :xy_sum_split,
)

# Rule 5: 3 disjuncts
add_disjunction!(
    dm,
    [LinearConstraint([0.0, 1.0, -1.0], :<=, -0.10)],
    [LinearConstraint([0.0, 1.0, -1.0], :>=, 0.20)],
    [LinearConstraint([1.0, 0.0, -1.0], :>=, 0.10)];
    name = :relative_splits,
)

rule_order = [
    :x_low_or_high,
    :y_low_or_high,
    :z_low_mid_or_high,
    :xy_sum_split,
    :relative_splits,
]

println()
println("=== Formulation summaries before solving ===")
println(formulation_summary(dm; formulation = :dnf))
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
    num_dnf_rules = 3,
    rule_ordering = rule_order,
))
println(formulation_summary(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 4,
    rule_ordering = rule_order,
))


yhat = [0.50, 0.20, 0.35]
reg_kwargs = (
    y_regularization = 0.0,
    ycopy_regularization = 1e-8,
    gamma_regularization = 1e-8,
    anchor_regularization = 1e-3,
)

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
    solver = gurobi_solver,
    reg_kwargs...,
)
benchmark_projection(
    dm,
    yhat;
    formulation = :partial_dnf,
    num_dnf_rules = 3,
    rule_ordering = rule_order,
    label = "partial-DNF k=3",
    solver = gurobi_solver,
    reg_kwargs...,
)
benchmark_projection(
    dm,
    yhat;
    formulation = :partial_dnf,
    num_dnf_rules = 4,
    rule_ordering = rule_order,
    label = "partial-DNF k=4",
    solver = gurobi_solver,
    reg_kwargs...,
)

benchmark_projection(
    dm, 
    yhat; 
    formulation = :dnf, 
    label = "full DNF",
    solver = gurobi_solver,
    reg_kwargs...,
)

println()
println("Note: first Julia run may include compilation overhead. Run the script twice for steadier timings.")