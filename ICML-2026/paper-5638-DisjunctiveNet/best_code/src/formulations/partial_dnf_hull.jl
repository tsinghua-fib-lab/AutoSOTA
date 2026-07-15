"""
    partial_dnf_hull_form(model; ordering = nothing, num_dnf_rules = -1)

Build a partial-DNF representation.

- If `ordering` is provided, it should be a vector of disjunction names.
- `num_dnf_rules` controls how many rules are converted to DNF.
- `num_dnf_rules = -1` means all rules in the selected order are converted to DNF.
"""
function partial_dnf_hull_form(
    model::DisjunctiveModel;
    ordering = nothing,
    num_dnf_rules::Int = -1,
)
    return partial_dnf_hull_form(
        standard_form(model);
        ordering = ordering,
        num_dnf_rules = num_dnf_rules,
    )
end


function partial_dnf_hull_form(
    model::StandardDisjunctiveModel;
    ordering = nothing,
    num_dnf_rules::Int = -1,
)
    n_rules = length(model.disjunctions)

    order_indices = _resolve_rule_order(model, ordering)

    k =
        if num_dnf_rules == -1
            n_rules
        else
            clamp(num_dnf_rules, 0, n_rules)
        end

    dnf_indices = order_indices[1:k]
    cnf_indices = order_indices[(k + 1):end]

    dnf_scenarios =
        isempty(dnf_indices) ?
        ConvexHullScenario[] :
        _partial_dnf_scenarios(model.disjunctions, dnf_indices)

    cnf_blocks = CNFConvexHullBlock[]

    for r in cnf_indices
        disjunction = model.disjunctions[r]
        push!(
            cnf_blocks,
            CNFConvexHullBlock(
                r,
                deepcopy(disjunction.disjuncts),
            ),
        )
    end

    return PartialDNFHullForm(
        model.n_outputs,
        model.lb,
        model.ub,
        model.global_constraints,
        dnf_indices,
        dnf_scenarios,
        cnf_blocks,
    )
end


function _resolve_rule_order(model::StandardDisjunctiveModel, ordering)
    n_rules = length(model.disjunctions)

    if ordering === nothing
        return collect(1:n_rules)
    end

    name_to_index = Dict{Symbol, Int}()

    for (i, disjunction) in enumerate(model.disjunctions)
        disjunction.name === nothing &&
            throw(ArgumentError("Ordering by name requires every ordered disjunction to have a name."))

        name_to_index[disjunction.name] = i
    end

    indices = Int[]

    for name in ordering
        sym = Symbol(name)
        haskey(name_to_index, sym) ||
            throw(ArgumentError("Unknown disjunction name $(sym)."))

        push!(indices, name_to_index[sym])
    end

    length(unique(indices)) == length(indices) ||
        throw(ArgumentError("Ordering contains duplicate disjunction names."))

    # Append any unnamed/unmentioned rules after the user-provided order.
    for i in 1:n_rules
        if !(i in indices)
            push!(indices, i)
        end
    end

    return indices
end


function _partial_dnf_scenarios(
    disjunctions::Vector{Disjunction},
    dnf_indices::Vector{Int},
)
    selected_disjunctions = disjunctions[dnf_indices]
    choices = _scenario_choices(selected_disjunctions)

    scenarios = ConvexHullScenario[]

    for choice in choices
        local_constraints = LinearConstraint[]

        for (t, selected_disjunct_index) in enumerate(choice)
            rule_index = dnf_indices[t]
            disjunction = disjunctions[rule_index]
            append!(local_constraints, disjunction.disjuncts[selected_disjunct_index])
        end

        push!(scenarios, ConvexHullScenario(choice, local_constraints))
    end

    return scenarios
end