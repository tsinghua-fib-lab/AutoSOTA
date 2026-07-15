using JuMP
using HiGHS
import MathOptInterface as MOI

"""
    convex_hull_form(model::DisjunctiveModel; prune_infeasible = true, interior_tol = 1e-7)

Build the convex-hull scenario expansion from a user-facing model.
"""
function convex_hull_form(
    model::DisjunctiveModel;
    prune_infeasible::Bool = false,
    interior_tol::Real = 1e-7,
)
    return convex_hull_form(
        standard_form(model);
        prune_infeasible = prune_infeasible,
        interior_tol = interior_tol,
    )
end


"""
    convex_hull_form(model::StandardDisjunctiveModel; prune_infeasible = true, interior_tol = 1e-7)

Build the scenario expansion needed by the convex-hull projection formulation.

Each scenario corresponds to one choice of disjunct from every disjunction.

Global constraints are stored separately because the JuMP/DiffOpt backend should
copy them into every scenario using the perspective form

    a' * y_copy[s, :] sense b * gamma[s]
"""
function convex_hull_form(
    model::StandardDisjunctiveModel;
    prune_infeasible::Bool = false,
    interior_tol::Real = 1e-7,
)
    choices = _scenario_choices(model.disjunctions)

    scenarios = ConvexHullScenario[]

    for choice in choices
        local_constraints = _local_constraints_for_choice(model.disjunctions, choice)
        scenario = ConvexHullScenario(choice, local_constraints)

        keep_scenario =
            isempty(model.disjunctions) ||
            !prune_infeasible ||
            scenario_has_interior_point(model, scenario; interior_tol = interior_tol)

        if keep_scenario
            push!(scenarios, scenario)
        end
    end

    if isempty(scenarios)
        throw(ArgumentError("All convex-hull scenarios were pruned as infeasible or non-interior."))
    end

    return ConvexHullForm(
        model.n_outputs,
        model.lb,
        model.ub,
        model.global_constraints,
        scenarios,
    )
end




"""
    num_scenarios(hull::ConvexHullForm)

Return the number of scenarios in the convex-hull expansion.
"""
num_scenarios(hull::ConvexHullForm) = length(hull.scenarios)


"""
    scenario_has_interior_point(model, scenario; interior_tol = 1e-7)

Return `true` if the scenario admits a point with positive margin from all
inequality constraints and finite bounds.

Equalities are enforced exactly. Inequalities and finite bounds are enforced
with a margin variable `τ`, and the LP maximizes `τ`.
"""
function scenario_has_interior_point(
    model::StandardDisjunctiveModel,
    scenario::ConvexHullScenario;
    interior_tol::Real = 1e-7,
)
    lp = Model(HiGHS.Optimizer)
    set_silent(lp)

    n = model.n_outputs

    @variable(lp, z[1:n])
    @variable(lp, τ >= 0.0)

    for j in 1:n
        if isfinite(model.lb[j])
            @constraint(lp, z[j] >= model.lb[j] + τ)
        end

        if isfinite(model.ub[j])
            @constraint(lp, z[j] <= model.ub[j] - τ)
        end
    end

    for constraint in model.global_constraints
        _add_interior_constraint!(lp, constraint, z, τ)
    end

    for constraint in scenario.local_constraints
        _add_interior_constraint!(lp, constraint, z, τ)
    end

    @objective(lp, Max, τ)

    optimize!(lp)

    status = termination_status(lp)

    if status != MOI.OPTIMAL
        return false
    end

    return value(τ) >= Float64(interior_tol)
end


function _add_interior_constraint!(
    model::JuMP.Model,
    constraint::LinearConstraint,
    z,
    τ,
)
    lhs = sum(constraint.a[j] * z[j] for j in eachindex(constraint.a))

    if constraint.sense == :<=
        @constraint(model, lhs <= constraint.b - τ)
    elseif constraint.sense == :>=
        @constraint(model, lhs >= constraint.b + τ)
    elseif constraint.sense == :(==)
        @constraint(model, lhs == constraint.b)
    else
        throw(ArgumentError("Unsupported constraint sense $(constraint.sense)."))
    end

    return nothing
end


function _scenario_choices(disjunctions::Vector{Disjunction})
    if isempty(disjunctions)
        return [Int[]]
    end

    choices = Vector{Vector{Int}}([Int[]])

    for disjunction in disjunctions
        new_choices = Vector{Vector{Int}}()

        for existing_choice in choices
            for disjunct_index in 1:length(disjunction.disjuncts)
                push!(new_choices, vcat(existing_choice, disjunct_index))
            end
        end

        choices = new_choices
    end

    return choices
end


function _local_constraints_for_choice(
    disjunctions::Vector{Disjunction},
    choice::Vector{Int},
)
    length(choice) == length(disjunctions) ||
        throw(DimensionMismatch("Choice length must match number of disjunctions."))

    local_constraints = LinearConstraint[]

    for (j, selected_disjunct_index) in enumerate(choice)
        disjunction = disjunctions[j]

        1 <= selected_disjunct_index <= length(disjunction.disjuncts) ||
            throw(ArgumentError("Invalid disjunct choice $(selected_disjunct_index) for disjunction $(j)."))

        selected_disjunct = disjunction.disjuncts[selected_disjunct_index]

        append!(local_constraints, selected_disjunct)
    end

    return local_constraints
end