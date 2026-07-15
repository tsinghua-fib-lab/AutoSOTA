import MathOptInterface as MOI


"""
    DisjunctiveProjectionLayer(model; mode = :dnf_qp, gradient = :diffopt, solver = nothing, kwargs...)

Create a differentiable projection layer from a `DisjunctiveModel`.
"""
function DisjunctiveProjectionLayer(
    model::DisjunctiveModel;
    mode::Symbol = :dnf_qp,
    formulation::Symbol = :dnf,
    gradient::Symbol = mode == :milp ? :straight_through : :diffopt,
    solver = nothing,
    tol::Real = 1e-6,
    fallback::Symbol = :identity,
    num_dnf_rules::Int = -1,
    rule_ordering = nothing,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 0.0,
    gamma_regularization::Real = 0.0,
    anchor_regularization::Real = 1e-3,
)
    config = ProjectionConfig(
        mode = mode,
        formulation = formulation,
        gradient = gradient,
        solver = solver,
        tol = Float64(tol),
        fallback = fallback,
        num_dnf_rules = num_dnf_rules,
        rule_ordering = rule_ordering,
        y_regularization = Float64(y_regularization),
        ycopy_regularization = Float64(ycopy_regularization),
        gamma_regularization = Float64(gamma_regularization),
        anchor_regularization = Float64(anchor_regularization),
    )
    return DisjunctiveProjectionLayer(model, config)
end


"""
    projection_mode(layer::DisjunctiveProjectionLayer)

Return the projection mode used by the layer.
"""
projection_mode(layer::DisjunctiveProjectionLayer) = layer.config.mode
projection_formulation(layer::DisjunctiveProjectionLayer) = layer.config.formulation

"""
    (layer::DisjunctiveProjectionLayer)(yhat)

Placeholder call overload.

The actual projection implementation will be added later.
"""
function (layer::DisjunctiveProjectionLayer)(yhat)
    result = project(
        layer.model,
        yhat;
        formulation = layer.config.formulation,
        solver = layer.config.solver,
        y_regularization = layer.config.y_regularization,
        ycopy_regularization = layer.config.ycopy_regularization,
        gamma_regularization = layer.config.gamma_regularization,
        anchor_regularization = layer.config.anchor_regularization,
        num_dnf_rules = layer.config.num_dnf_rules,
        rule_ordering = layer.config.rule_ordering,
    )

    if result.status == MOI.OPTIMAL
        return result.y
    end

    if layer.config.fallback == :identity
        return Float64.(collect(yhat))
    else
        error("Projection failed with status $(result.status).")
    end
end