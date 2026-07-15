using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DisjunctiveNet
import MathOptInterface as MOI

println("=== Basic disjunctive projection example ===")

dm = DisjunctiveModel(2)

set_bounds!(
    dm,
    lower = [0.0, 0.0],
    upper = [1.0, 1.0],
)

# Global constraint:
# y1 + y2 >= 0.8
add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

# Rule 1:
# y1 <= 0.25 OR y1 >= 0.75
add_disjunction!(
    dm,
    [LinearConstraint([1.0, 0.0], :<=, 0.25)],
    [LinearConstraint([1.0, 0.0], :>=, 0.75)];
    name = :x_split,
)

# Rule 2:
# y2 <= 0.25 OR y2 >= 0.75
add_disjunction!(
    dm,
    [LinearConstraint([0.0, 1.0], :<=, 0.25)],
    [LinearConstraint([0.0, 1.0], :>=, 0.75)];
    name = :y_split,
)

println()
println("=== User-facing model ===")
print_model(dm)

println()
println("=== Full DNF lifted model ===")
print_projection_model(dm; formulation = :dnf)

println()
println("=== CNF lifted model ===")
print_projection_model(dm; formulation = :cnf)

println()
println("=== Partial-DNF lifted model: first rule DNF, rest CNF ===")
print_projection_model(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)

yhat = [0.5, 0.1]

println()
println("Raw prediction yhat = ", yhat)

result_dnf = project(dm, yhat; formulation = :dnf)
result_cnf = project(dm, yhat; formulation = :cnf)
result_partial = project(
    dm,
    yhat;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)

println()
println("=== Projection results ===")
println("DNF status      = ", result_dnf.status)
println("DNF y           = ", result_dnf.y)

println("CNF status      = ", result_cnf.status)
println("CNF y           = ", result_cnf.y)

println("Partial status  = ", result_partial.status)
println("Partial y       = ", result_partial.y)

println()
println("=== Layer API ===")

dnf_layer = DisjunctiveProjectionLayer(dm; formulation = :dnf)
cnf_layer = DisjunctiveProjectionLayer(dm; formulation = :cnf)
partial_layer = DisjunctiveProjectionLayer(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)

println("dnf_layer(yhat)     = ", dnf_layer(yhat))
println("cnf_layer(yhat)     = ", cnf_layer(yhat))
println("partial_layer(yhat) = ", partial_layer(yhat))