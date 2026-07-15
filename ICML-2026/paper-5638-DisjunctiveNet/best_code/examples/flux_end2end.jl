using Flux
using Zygote
using DisjunctiveNet

# -----------------------------
# 1. Build a neural network
# -----------------------------

backbone = Chain(
    Dense(3 => 8, relu),
    Dense(8 => 2),
)

x = Float32[0.2, 0.7, 0.4]
target = Float32[0.8, 0.2]

# -----------------------------
# 2. Build the disjunctive rule model
# -----------------------------

dm = DisjunctiveModel(2)

set_bounds!(
    dm,
    lower = [0.0, 0.0],
    upper = [1.0, 1.0],
)

# Global constraint:
# y1 + y2 >= 0.8
add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

# Rule 1 has three disjuncts:
# y1 <= 0.2 OR 0.4 <= y1 <= 0.6 OR y1 >= 0.8
add_disjunction!(
    dm,
    [LinearConstraint([1.0, 0.0], :<=, 0.2)],
    [
        LinearConstraint([1.0, 0.0], :>=, 0.4),
        LinearConstraint([1.0, 0.0], :<=, 0.6),
    ],
    [LinearConstraint([1.0, 0.0], :>=, 0.8)];
    name = :x_rule,
)

# Rule 2 has three disjuncts:
# y2 <= 0.2 OR 0.35 <= y2 <= 0.55 OR y2 >= 0.7
add_disjunction!(
    dm,
    [LinearConstraint([0.0, 1.0], :<=, 0.2)],
    [
        LinearConstraint([0.0, 1.0], :>=, 0.35),
        LinearConstraint([0.0, 1.0], :<=, 0.55),
    ],
    [LinearConstraint([0.0, 1.0], :>=, 0.7)];
    name = :y_rule,
)

# -----------------------------
# 3. Inspect the model and formulations
# -----------------------------

println()
println("=== User-facing disjunctive model ===")
print_model(dm)

println()
println("=== DNF lifted formulation ===")
print_projection_model(dm; formulation = :dnf)

println()
println("=== CNF lifted formulation ===")
print_projection_model(dm; formulation = :cnf)

println()
println("=== Partial-DNF lifted formulation ===")
print_projection_model(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_rule, :y_rule],
)

# -----------------------------
# 4. Compare DNF, CNF, and partial-DNF on the same NN output
# -----------------------------

yhat = backbone(x)

println()
println("Raw NN output yhat = ", yhat)

dnf_layer = DisjunctiveProjectionLayer(
    dm;
    formulation = :dnf,
    y_regularization = 1e-4,
    ycopy_regularization = 1e-4,
    gamma_regularization = 1e-4,
    anchor_regularization = 1e-4,
)

cnf_layer = DisjunctiveProjectionLayer(
    dm;
    formulation = :cnf,
    y_regularization = 1e-4,
    ycopy_regularization = 1e-4,
    gamma_regularization = 1e-4,
    anchor_regularization = 1e-4,
)

partial_layer = DisjunctiveProjectionLayer(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_rule, :y_rule],
    y_regularization = 1e-4,
    ycopy_regularization = 1e-4,
    gamma_regularization = 1e-4,
    anchor_regularization = 1e-4,
)

y_dnf = dnf_layer(yhat)
y_cnf = cnf_layer(yhat)
y_partial = partial_layer(yhat)

println()
println("=== Projected predictions for the same sample ===")
println("DNF projection         = ", y_dnf, "  sum = ", sum(y_dnf))
println("CNF projection         = ", y_cnf, "  sum = ", sum(y_cnf))
println("Partial-DNF projection = ", y_partial, "  sum = ", sum(y_partial))

println()
println("The three projections can differ because :dnf, :cnf, and :partial_dnf")
println("construct different convexified relaxations of the same logical rule set.")

# -----------------------------
# 5. Build a trainable constrained Flux model
# -----------------------------

# Here we train the partial-DNF constrained model.
# The projection layer is differentiable, but it has no trainable parameters.
# Flux only trains the neural network backbone.
model = constrained_model(
    backbone,
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_rule, :y_rule],
    y_regularization = 1e-4,
    ycopy_regularization = 1e-4,
    gamma_regularization = 1e-4,
    anchor_regularization = 1e-4,
)

# -----------------------------
# 6. Forward pass
# -----------------------------

y = model(x)

println()
println("=== Forward pass through constrained model ===")
println("Projected prediction = ", y)
println("Feasibility check: y1 + y2 = ", sum(y))

# -----------------------------
# 7. One training step
# -----------------------------

loss(m, x, target) = sum(abs2, m(x) .- target)

opt = Flux.setup(Adam(1e-3), model)

l, grads = Flux.withgradient(model) do m
    loss(m, x, target)
end

Flux.update!(opt, model, grads[1])

println()
println("Training loss before update = ", l)

# -----------------------------
# 8. Inference after one update
# -----------------------------

y_after = model(x)

println()
println("=== Inference after one update ===")
println("Projected prediction after one update = ", y_after)
println("Feasibility check after update: y1 + y2 = ", sum(y_after))