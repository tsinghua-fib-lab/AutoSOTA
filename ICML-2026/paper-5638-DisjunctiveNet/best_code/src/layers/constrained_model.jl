using Functors

"""
    ConstrainedFluxModel(backbone, layer)

A Flux-compatible model that applies a neural network backbone followed by a
disjunctive projection layer.

Calling the model evaluates

    yhat = backbone(x)
    y    = layer(yhat)

The projection layer is differentiable through ChainRules/DiffOpt.
"""
struct ConstrainedFluxModel{B,L}
    backbone::B
    layer::L
end

function (model::ConstrainedFluxModel)(x)
    yhat = model.backbone(x)
    return model.layer(yhat)
end

Flux.trainable(model::ConstrainedFluxModel) = (backbone = model.backbone,)

Functors.children(model::ConstrainedFluxModel) = (backbone = model.backbone,)
Functors.functor(::Type{<:ConstrainedFluxModel}, model) = ((backbone = model.backbone,), children -> ConstrainedFluxModel(children.backbone, model.layer))

"""
    constrained_model(backbone, disjunctive_model)

Wrap a Flux backbone with a `DisjunctiveProjectionLayer`.
"""
function constrained_model(
    backbone,
    dm::DisjunctiveModel;
    formulation::Symbol = :dnf,
    mode::Symbol = :dnf_qp,
    gradient::Symbol = mode == :milp ? :straight_through : :diffopt,
    solver = nothing,
    tol::Real = 1e-6,
    fallback::Symbol = :identity,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 0.0,
    gamma_regularization::Real = 0.0,
    anchor_regularization::Real = 1e-3,
    num_dnf_rules::Int = -1,
    rule_ordering = nothing,
)
    layer = DisjunctiveProjectionLayer(
        dm;
        mode = mode,
        formulation = formulation,
        gradient = gradient,
        solver = solver,
        tol = tol,
        fallback = fallback,
        y_regularization = y_regularization,
        ycopy_regularization = ycopy_regularization,
        gamma_regularization = gamma_regularization,
        anchor_regularization = anchor_regularization,
        num_dnf_rules = num_dnf_rules,
        rule_ordering = rule_ordering,
    )

    return ConstrainedFluxModel(backbone, layer)
end


"""
    constrained_model(backbone, example_input, build_rules!; kwargs...)

Infer the backbone output dimension from `example_input`, create a
`DisjunctiveModel`, call `build_rules!(dm)`, and return a constrained Flux model.

Example:

    model = constrained_model(backbone, x0) do dm
        set_bounds!(dm, lower = zeros(2), upper = ones(2))
        add_disjunction!(...)
    end
"""
function constrained_model(
    backbone,
    example_input,
    build_rules!::Function;
    kwargs...,
)
    yhat = backbone(example_input)
    n_outputs = length(yhat)

    dm = DisjunctiveModel(n_outputs)
    build_rules!(dm)

    return constrained_model(backbone, dm; kwargs...)
end

"""
    constrained_model(build_rules!, backbone, example_input; kwargs...)

Do-block compatible constructor.

This supports:

    model = constrained_model(backbone, x0; kwargs...) do dm
        ...
    end
"""
function constrained_model(
    build_rules!::Function,
    backbone,
    example_input;
    kwargs...,
)
    return constrained_model(backbone, example_input, build_rules!; kwargs...)
end