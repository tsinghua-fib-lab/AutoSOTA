using JuMP
using HiGHS
using DiffOpt
import MathOptInterface as MOI

"""
    build_projection_model(hull, yhat; solver = nothing)

Build the convex-hull projection model.

This function constructs the JuMP model but does not solve it.
"""
function build_projection_model(
    hull::ConvexHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
    y_regularization::Real = 1e-8,
    ycopy_regularization::Real = 1e-8,
    gamma_regularization::Real = 1e-8,
    anchor_regularization::Real = 1e-5,
)
    length(yhat) == hull.n_outputs ||
        throw(DimensionMismatch("Expected yhat of length $(hull.n_outputs), got $(length(yhat))."))

    optimizer = solver === nothing ? HiGHS.Optimizer : solver
    model = Model(() -> DiffOpt.diff_optimizer(optimizer))
    set_silent(model)

    n = hull.n_outputs
    S = num_scenarios(hull)

    @variable(model, y[1:n])
    @variable(model, gamma[1:S] >= 0.0)
    @variable(model, y_copy[1:S, 1:n])
    @variable(model, yhat_param[1:n] in MOI.Parameter.(Float64.(yhat)))
    
    @constraint(model, sum(gamma[s] for s in 1:S) == 1.0)

    @constraint(model, [j in 1:n], y[j] == sum(y_copy[s, j] for s in 1:S))

    for s in 1:S
        for j in 1:n
            if isfinite(hull.lb[j])
                @constraint(model, y_copy[s, j] >= hull.lb[j] * gamma[s])
            end

            if isfinite(hull.ub[j])
                @constraint(model, y_copy[s, j] <= hull.ub[j] * gamma[s])
            end
        end

        for constraint in hull.global_constraints
            _add_perspective_constraint!(model, constraint, y_copy, gamma, s)
        end

        for constraint in hull.scenarios[s].local_constraints
            _add_perspective_constraint!(model, constraint, y_copy, gamma, s)
        end
    end
    
    anchors = _default_scenario_anchors(hull)
    model[:anchors] = anchors

    base_objective =
        sum((y[j] - yhat_param[j])^2 for j in 1:n)

    y_reg_objective =
        Float64(y_regularization) *
        sum(y[j]^2 for j in 1:n)

    ycopy_reg_objective =
        Float64(ycopy_regularization) *
        sum(y_copy[s, j]^2 for s in 1:S, j in 1:n)

    gamma_reg_objective =
        Float64(gamma_regularization) *
        sum(gamma[s]^2 for s in 1:S)

    anchor_objective =
        Float64(anchor_regularization) *
        sum(
            (y_copy[s, j] - gamma[s] * anchors[s, j])^2
            for s in 1:S, j in 1:n
        )

    @objective(
        model,
        Min,
        base_objective +
        y_reg_objective +
        ycopy_reg_objective +
        gamma_reg_objective +
        anchor_objective
    )

    model[:y] = y
    model[:gamma] = gamma
    model[:y_copy] = y_copy
    model[:hull] = hull
    model[:yhat_param] = yhat_param

    return model
end

function build_projection_model(
    hull::CNFConvexHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 0.0,
    gamma_regularization::Real = 0.0,
    anchor_regularization::Real = 1e-3,
)
    length(yhat) == hull.n_outputs ||
        throw(DimensionMismatch("Expected yhat of length $(hull.n_outputs), got $(length(yhat))."))

    optimizer = solver === nothing ? HiGHS.Optimizer : solver
    model = Model(() -> DiffOpt.diff_optimizer(optimizer))
    set_silent(model)

    n = hull.n_outputs
    R = length(hull.blocks)

    @variable(model, y[1:n])
    @variable(model, yhat_param[1:n] in MOI.Parameter.(Float64.(yhat)))

    # Base bounds on y.
    for j in 1:n
        if isfinite(hull.lb[j])
            @constraint(model, y[j] >= hull.lb[j])
        end
        if isfinite(hull.ub[j])
            @constraint(model, y[j] <= hull.ub[j])
        end
    end

    # Global constraints on y.
    for constraint in hull.global_constraints
        _add_constraint_on_y!(model, constraint, y)
    end

    # If there are no disjunctions, this is just ordinary convex projection.
    gamma_refs = Any[]
    ycopy_refs = Any[]

    for block in hull.blocks
        r = block.disjunction_index
        D = length(block.disjuncts)

        gamma = @variable(model, [1:D], lower_bound = 0.0)
        y_copy = @variable(model, [1:D, 1:n])

        @constraint(model, sum(gamma[d] for d in 1:D) == 1.0)
        @constraint(model, [j in 1:n], y[j] == sum(y_copy[d, j] for d in 1:D))

        for d in 1:D
            for j in 1:n
                if isfinite(hull.lb[j])
                    @constraint(model, y_copy[d, j] >= hull.lb[j] * gamma[d])
                end
                if isfinite(hull.ub[j])
                    @constraint(model, y_copy[d, j] <= hull.ub[j] * gamma[d])
                end
            end

            # Copy global constraints into each disjunct block.
            # This gives the tighter hull conv((G ∩ D1) ∪ ... ∪ (G ∩ Dm)).
            for constraint in hull.global_constraints
                _add_perspective_constraint_cnf!(model, constraint, y_copy, gamma, d)
            end

            for constraint in block.disjuncts[d]
                _add_perspective_constraint_cnf!(model, constraint, y_copy, gamma, d)
            end
        end

        push!(gamma_refs, gamma)
        push!(ycopy_refs, y_copy)
    end

    base_objective =
        sum((y[j] - yhat_param[j])^2 for j in 1:n)

    y_reg_objective =
        Float64(y_regularization) *
        sum(y[j]^2 for j in 1:n)

    ycopy_reg_objective = zero(QuadExpr)
    for y_copy_block in ycopy_refs
        for d in axes(y_copy_block, 1)
            for j in 1:n
                ycopy_reg_objective += y_copy_block[d, j]^2
            end
        end
    end
    ycopy_reg_objective *= Float64(ycopy_regularization)

    gamma_reg_objective = zero(QuadExpr)
    for gamma_block in gamma_refs
        for d in eachindex(gamma_block)
            gamma_reg_objective += gamma_block[d]^2
        end
    end
    gamma_reg_objective *= Float64(gamma_regularization)


    @objective(
        model,
        Min,
        base_objective + y_reg_objective + ycopy_reg_objective + gamma_reg_objective
    )

    model[:y] = y
    model[:yhat_param] = yhat_param
    model[:gamma_blocks] = gamma_refs
    model[:ycopy_blocks] = ycopy_refs
    model[:hull] = hull

    return model
end


function build_projection_model(
    hull::PartialDNFHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 0.0,
    gamma_regularization::Real = 0.0,
    anchor_regularization::Real = 1e-3,
)
    length(yhat) == hull.n_outputs ||
        throw(DimensionMismatch("Expected yhat of length $(hull.n_outputs), got $(length(yhat))."))

    optimizer = solver === nothing ? HiGHS.Optimizer : solver
    model = Model(() -> DiffOpt.diff_optimizer(optimizer))
    set_silent(model)

    n = hull.n_outputs

    @variable(model, y[1:n])
    @variable(model, yhat_param[1:n] in MOI.Parameter.(Float64.(yhat)))

    for j in 1:n
        if isfinite(hull.lb[j])
            @constraint(model, y[j] >= hull.lb[j])
        end
        if isfinite(hull.ub[j])
            @constraint(model, y[j] <= hull.ub[j])
        end
    end

    for constraint in hull.global_constraints
        _add_constraint_on_y!(model, constraint, y)
    end

    gamma_blocks = Any[]
    ycopy_blocks = Any[]

    # -------------------------
    # DNF block
    # -------------------------
    if !isempty(hull.dnf_scenarios)
        S = length(hull.dnf_scenarios)

        gamma_dnf = @variable(model, [1:S], lower_bound = 0.0)
        y_dnf_copy = @variable(model, [1:S, 1:n])

        @constraint(model, sum(gamma_dnf[s] for s in 1:S) == 1.0)
        @constraint(model, [j in 1:n], y[j] == sum(y_dnf_copy[s, j] for s in 1:S))

        for s in 1:S
            for j in 1:n
                if isfinite(hull.lb[j])
                    @constraint(model, y_dnf_copy[s, j] >= hull.lb[j] * gamma_dnf[s])
                end
                if isfinite(hull.ub[j])
                    @constraint(model, y_dnf_copy[s, j] <= hull.ub[j] * gamma_dnf[s])
                end
            end

            for constraint in hull.global_constraints
                _add_perspective_constraint!(model, constraint, y_dnf_copy, gamma_dnf, s)
            end

            for constraint in hull.dnf_scenarios[s].local_constraints
                _add_perspective_constraint!(model, constraint, y_dnf_copy, gamma_dnf, s)
            end
        end

        push!(gamma_blocks, gamma_dnf)
        push!(ycopy_blocks, y_dnf_copy)
    end

    # -------------------------
    # Remaining CNF blocks
    # -------------------------
    for block in hull.cnf_blocks
        D = length(block.disjuncts)

        gamma = @variable(model, [1:D], lower_bound = 0.0)
        y_copy = @variable(model, [1:D, 1:n])

        @constraint(model, sum(gamma[d] for d in 1:D) == 1.0)
        @constraint(model, [j in 1:n], y[j] == sum(y_copy[d, j] for d in 1:D))

        for d in 1:D
            for j in 1:n
                if isfinite(hull.lb[j])
                    @constraint(model, y_copy[d, j] >= hull.lb[j] * gamma[d])
                end
                if isfinite(hull.ub[j])
                    @constraint(model, y_copy[d, j] <= hull.ub[j] * gamma[d])
                end
            end

            for constraint in hull.global_constraints
                _add_perspective_constraint_cnf!(model, constraint, y_copy, gamma, d)
            end

            for constraint in block.disjuncts[d]
                _add_perspective_constraint_cnf!(model, constraint, y_copy, gamma, d)
            end
        end

        push!(gamma_blocks, gamma)
        push!(ycopy_blocks, y_copy)
    end

    base_objective =
        sum((y[j] - yhat_param[j])^2 for j in 1:n)

    y_reg_objective =
        Float64(y_regularization) *
        sum(y[j]^2 for j in 1:n)

    ycopy_reg_objective = zero(QuadExpr)
    for y_copy_block in ycopy_blocks
        for d in axes(y_copy_block, 1)
            for j in 1:n
                ycopy_reg_objective += y_copy_block[d, j]^2
            end
        end
    end
    ycopy_reg_objective *= Float64(ycopy_regularization)

    gamma_reg_objective = zero(QuadExpr)
    for gamma_block in gamma_blocks
        for d in eachindex(gamma_block)
            gamma_reg_objective += gamma_block[d]^2
        end
    end
    gamma_reg_objective *= Float64(gamma_regularization)

    @objective(
        model,
        Min,
        base_objective + y_reg_objective + ycopy_reg_objective + gamma_reg_objective
    )

    model[:y] = y
    model[:yhat_param] = yhat_param
    model[:gamma_blocks] = gamma_blocks
    model[:ycopy_blocks] = ycopy_blocks
    model[:hull] = hull

    return model
end

function _add_perspective_constraint!(
    model::JuMP.Model,
    constraint::LinearConstraint,
    y_copy,
    gamma,
    s::Int,
)
    lhs = sum(constraint.a[j] * y_copy[s, j] for j in eachindex(constraint.a))
    rhs = constraint.b * gamma[s]

    if constraint.sense == :<=
        @constraint(model, lhs <= rhs)
    elseif constraint.sense == :>=
        @constraint(model, lhs >= rhs)
    elseif constraint.sense == :(==)
        @constraint(model, lhs == rhs)
    else
        throw(ArgumentError("Unsupported constraint sense $(constraint.sense)."))
    end

    return nothing
end

function project(
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real};
    formulation::Symbol = :dnf,
    solver = nothing,
    num_dnf_rules::Int = -1,
    rule_ordering = nothing,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 0.0,
    gamma_regularization::Real = 0.0,
    anchor_regularization::Real = 1e-3,
)
    hull =
        if formulation == :dnf
            convex_hull_form(model; prune_infeasible = false)
        elseif formulation == :cnf
            cnf_hull_form(model)
        elseif formulation == :partial_dnf
            partial_dnf_hull_form(
                model;
                ordering = rule_ordering,
                num_dnf_rules = num_dnf_rules,
            )
        else
            throw(ArgumentError("Unknown formulation $(formulation). Expected :dnf, :cnf, or :partial_dnf."))
        end

    return project(
        hull,
        yhat;
        solver = solver,
        y_regularization = y_regularization,
        ycopy_regularization = ycopy_regularization,
        gamma_regularization = gamma_regularization,
        anchor_regularization = anchor_regularization,
    )
end


"""
    project(hull, yhat; solver = nothing)

Solve the convex-hull projection problem.
"""
function project(
    hull::ConvexHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
    y_regularization::Real = 1e-8,
    ycopy_regularization::Real = 1e-8,
    gamma_regularization::Real = 1e-8,
    anchor_regularization::Real = 1e-5,
)
    model = build_projection_model(
        hull,
        yhat;
        solver = solver,
        y_regularization = y_regularization,
        ycopy_regularization = ycopy_regularization,
        gamma_regularization = gamma_regularization,
        anchor_regularization = anchor_regularization,
    )

    optimize!(model)

    status = termination_status(model)

    if status != MOI.OPTIMAL
        return ProjectionResult(
            Float64.(collect(yhat)),
            Float64[],
            status,
            model,
        )
    end

    y = value.(model[:y])
    gamma = value.(model[:gamma])

    return ProjectionResult(
        Float64.(y),
        Float64.(gamma),
        status,
        model,
    )
end

function project(
    hull::PartialDNFHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 0.0,
    gamma_regularization::Real = 0.0,
    anchor_regularization::Real = 1e-3,
)
    model = build_projection_model(
        hull,
        yhat;
        solver = solver,
        y_regularization = y_regularization,
        ycopy_regularization = ycopy_regularization,
        gamma_regularization = gamma_regularization,
        anchor_regularization = anchor_regularization,
    )

    optimize!(model)
    status = termination_status(model)

    if status != MOI.OPTIMAL
        return ProjectionResult(
            Float64.(collect(yhat)),
            Float64[],
            status,
            model,
        )
    end

    return ProjectionResult(
        Float64.(value.(model[:y])),
        Float64[],
        status,
        model,
    )
end

function project(
    hull::CNFConvexHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 0.0,
    gamma_regularization::Real = 0.0,
    anchor_regularization::Real = 1e-3,
)
    model = build_projection_model(
        hull,
        yhat;
        solver = solver,
        y_regularization = y_regularization,
        ycopy_regularization = ycopy_regularization,
        gamma_regularization = gamma_regularization,
        anchor_regularization = anchor_regularization,
    )

    optimize!(model)
    status = termination_status(model)

    if status != MOI.OPTIMAL
        return ProjectionResult(
            Float64.(collect(yhat)),
            Float64[],
            status,
            model,
        )
    end

    return ProjectionResult(
        Float64.(value.(model[:y])),
        Float64[],
        status,
        model,
    )
end

function _default_scenario_anchors(hull::ConvexHullForm)
    n = hull.n_outputs
    S = num_scenarios(hull)

    midpoint = zeros(Float64, n)

    for j in 1:n
        if isfinite(hull.lb[j]) && isfinite(hull.ub[j])
            midpoint[j] = 0.5 * (hull.lb[j] + hull.ub[j])
        elseif isfinite(hull.lb[j])
            midpoint[j] = hull.lb[j] + 1.0
        elseif isfinite(hull.ub[j])
            midpoint[j] = hull.ub[j] - 1.0
        else
            midpoint[j] = 0.0
        end
    end

    anchors = zeros(Float64, S, n)

    for s in 1:S
        anchors[s, :] .= midpoint
    end

    return anchors
end

function _add_constraint_on_y!(
    model::JuMP.Model,
    constraint::LinearConstraint,
    y,
)
    lhs = sum(constraint.a[j] * y[j] for j in eachindex(constraint.a))
    rhs = constraint.b

    if constraint.sense == :<=
        @constraint(model, lhs <= rhs)
    elseif constraint.sense == :>=
        @constraint(model, lhs >= rhs)
    elseif constraint.sense == :(==)
        @constraint(model, lhs == rhs)
    else
        throw(ArgumentError("Unsupported constraint sense $(constraint.sense)."))
    end

    return nothing
end


function _add_perspective_constraint_cnf!(
    model::JuMP.Model,
    constraint::LinearConstraint,
    y_copy,
    gamma,
    d::Int,
)
    lhs = sum(constraint.a[j] * y_copy[d, j] for j in eachindex(constraint.a))
    rhs = constraint.b * gamma[d]

    if constraint.sense == :<=
        @constraint(model, lhs <= rhs)
    elseif constraint.sense == :>=
        @constraint(model, lhs >= rhs)
    elseif constraint.sense == :(==)
        @constraint(model, lhs == rhs)
    else
        throw(ArgumentError("Unsupported constraint sense $(constraint.sense)."))
    end

    return nothing
end

