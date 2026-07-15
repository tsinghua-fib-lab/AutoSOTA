function _sense_string(sense::Symbol)
    if sense == :<=
        return "<="
    elseif sense == :>=
        return ">="
    elseif sense == :(==)
        return "=="
    else
        return string(sense)
    end
end


function _linear_expr_string(a::AbstractVector)
    terms = String[]

    for (j, coeff) in enumerate(a)
        if abs(coeff) <= 1e-12
            continue
        end

        if coeff == 1
            push!(terms, "y[$j]")
        elseif coeff == -1
            push!(terms, "-y[$j]")
        else
            push!(terms, "$(coeff)*y[$j]")
        end
    end

    isempty(terms) && return "0"

    expr = join(terms, " + ")
    return replace(expr, "+ -" => "- ")
end


function _constraint_string(c::LinearConstraint)
    return "$(_linear_expr_string(c.a)) $(_sense_string(c.sense)) $(c.b)"
end

function _scaled_constraint_string(c::LinearConstraint, gamma_name::AbstractString)
    lhs = _linear_expr_string(c.a)
    sense = _sense_string(c.sense)
    rhs = "$(c.b) * $(gamma_name)"
    return "$(lhs) $(sense) $(rhs)"
end


"""
    print_model(model::DisjunctiveModel; io = stdout)

Print the user-facing disjunctive model.
"""
function print_model(model::DisjunctiveModel; io::IO = stdout)
    println(io, "DisjunctiveModel")
    println(io, "  n_outputs: $(model.n_outputs)")

    println(io, "  bounds:")
    for j in 1:model.n_outputs
        println(io, "    $(model.lb[j]) <= y[$j] <= $(model.ub[j])")
    end

    println(io, "  global constraints:")
    if isempty(model.constraints)
        println(io, "    none")
    else
        for (i, c) in enumerate(model.constraints)
            println(io, "    g[$i]: $(_constraint_string(c))")
        end
    end

    println(io, "  disjunctions:")
    if isempty(model.disjunctions)
        println(io, "    none")
    else
        for (r, disj) in enumerate(model.disjunctions)
            name_str = disj.name === nothing ? "" : " name=$(disj.name)"
            println(io, "    disjunction[$r]$name_str:")
            for (d, disjunct) in enumerate(disj.disjuncts)
                println(io, "      disjunct[$d]:")
                for (q, c) in enumerate(disjunct)
                    println(io, "        c[$q]: $(_constraint_string(c))")
                end
            end
        end
    end

    return nothing
end


"""
    print_hull(hull::ConvexHullForm; io = stdout)

Print the DNF convex-hull scenario expansion.
"""
function print_hull(hull::ConvexHullForm; io::IO = stdout)
    println(io, "DNF ConvexHullForm")
    println(io, "  n_outputs: $(hull.n_outputs)")
    println(io, "  scenarios: $(length(hull.scenarios))")

    println(io, "  variables:")
    println(io, "    y[1:$(hull.n_outputs)]")
    println(io, "    gamma[1:$(length(hull.scenarios))]")
    println(io, "    y_copy[1:$(length(hull.scenarios)), 1:$(hull.n_outputs)]")

    println(io, "  linking constraints:")
    println(io, "    sum_s gamma[s] == 1")
    println(io, "    y[j] == sum_s y_copy[s,j] for each j")

    println(io, "  global constraints copied into every scenario:")
    if isempty(hull.global_constraints)
        println(io, "    none")
    else
        for (i, c) in enumerate(hull.global_constraints)
            println(io, "    g[$i,s]: $(_scaled_constraint_string(c, "gamma[s]"))")
        end
    end

    println(io, "  scenarios:")
    for (s, scenario) in enumerate(hull.scenarios)
        println(io, "    scenario[$s], choices = $(scenario.choices):")
        if isempty(scenario.local_constraints)
            println(io, "      local constraints: none")
        else
            for (q, c) in enumerate(scenario.local_constraints)
                println(io, "      c[$q,s]: $(_scaled_constraint_string(c, "gamma[$s]"))")
            end
        end
    end

    return nothing
end


"""
    print_hull(hull::CNFConvexHullForm; io = stdout)

Print the CNF convex-hull block representation.
"""
function print_hull(hull::CNFConvexHullForm; io::IO = stdout)
    println(io, "CNF ConvexHullForm")
    println(io, "  n_outputs: $(hull.n_outputs)")
    println(io, "  blocks: $(length(hull.blocks))")

    println(io, "  variables:")
    println(io, "    y[1:$(hull.n_outputs)]")
    println(io, "    for each block r:")
    println(io, "      gamma_r[1:num_disjuncts(r)]")
    println(io, "      y_copy_r[1:num_disjuncts(r), 1:$(hull.n_outputs)]")



    println(io, "  global constraints on y:")
    if isempty(hull.global_constraints)
        println(io, "    none")
    else
        for (i, c) in enumerate(hull.global_constraints)
            println(io, "    g[$i]: $(_constraint_string(c))")
        end
    end

    println(io, "  blocks:")
    for block in hull.blocks
        println(io, "    block[$(block.disjunction_index)]:")
        println(io, "      sum_d gamma[d] == 1")
        println(io, "      y[j] == sum_d y_copy[d,j] for each j")

        for (d, disjunct) in enumerate(block.disjuncts)
            println(io, "      disjunct[$d]:")
            println(io, "        bounds and global constraints copied with gamma[$d]")
            for (q, c) in enumerate(disjunct)
                println(io, "        c[$q]: $(_scaled_constraint_string(c, "gamma[$d]"))")
            end

            println(io, "  global constraints on y:")
            if isempty(hull.global_constraints)
                println(io, "    none")
            else
                for (i, c) in enumerate(hull.global_constraints)
                    println(io, "        g[$i,d]: $(_scaled_constraint_string(c, "gamma[d]"))")
                end
            end
        end
    end

    return nothing
end


"""
    print_hull(hull::PartialDNFHullForm; io = stdout)

Print the partial-DNF lifted representation.

The selected DNF rules are expanded jointly into one scenario block.
The remaining rules are kept as separate CNF convex-hull blocks.
"""
function print_hull(hull::PartialDNFHullForm; io::IO = stdout)
    println(io, "Partial-DNF HullForm")
    println(io, "  n_outputs: $(hull.n_outputs)")
    println(io, "  DNF-expanded rules: $(hull.dnf_indices)")
    println(io, "  DNF scenarios: $(length(hull.dnf_scenarios))")
    println(io, "  CNF blocks: $(length(hull.cnf_blocks))")

    println(io, "  variables:")
    println(io, "    y[1:$(hull.n_outputs)]")

    if !isempty(hull.dnf_scenarios)
        println(io, "    DNF block:")
        println(io, "      gamma_dnf[1:$(length(hull.dnf_scenarios))]")
        println(io, "      y_dnf_copy[1:$(length(hull.dnf_scenarios)), 1:$(hull.n_outputs)]")
    else
        println(io, "    DNF block: none")
    end

    if !isempty(hull.cnf_blocks)
        println(io, "    CNF blocks:")
        println(io, "      for each remaining rule r:")
        println(io, "        gamma_r[1:num_disjuncts(r)]")
        println(io, "        y_copy_r[1:num_disjuncts(r), 1:$(hull.n_outputs)]")
    else
        println(io, "    CNF blocks: none")
    end

    println(io, "  global constraints on y:")
    if isempty(hull.global_constraints)
        println(io, "    none")
    else
        for (i, c) in enumerate(hull.global_constraints)
            println(io, "    g[$i]: $(_constraint_string(c))")
        end
    end

    println(io, "  DNF block:")
    if isempty(hull.dnf_scenarios)
        println(io, "    none")
    else
        println(io, "    linking constraints:")
        println(io, "      sum_s gamma_dnf[s] == 1")
        println(io, "      y[j] == sum_s y_dnf_copy[s,j] for each j")
        println(io, "    global constraints copied into every DNF scenario:")
        if isempty(hull.global_constraints)
            println(io, "      none")
        else
            for (i, c) in enumerate(hull.global_constraints)
                println(io, "      g[$i,s]: $(_scaled_constraint_string(c, "gamma_dnf[s]"))")
            end
        end

        println(io, "    scenarios:")
        for (s, scenario) in enumerate(hull.dnf_scenarios)
            println(io, "      scenario[$s], choices = $(scenario.choices):")
            if isempty(scenario.local_constraints)
                println(io, "        local constraints: none")
            else
                for (q, c) in enumerate(scenario.local_constraints)
                    println(io, "        c[$q,s]: $(_scaled_constraint_string(c, "gamma_dnf[$s]"))")
                end
            end
        end
    end

    println(io, "  CNF blocks:")
    if isempty(hull.cnf_blocks)
        println(io, "    none")
    else
        for block in hull.cnf_blocks
            println(io, "    block[$(block.disjunction_index)]:")
            println(io, "      sum_d gamma[d] == 1")
            println(io, "      y[j] == sum_d y_copy[d,j] for each j")
            println(io, "      global constraints copied into every disjunct copy:")

            if isempty(hull.global_constraints)
                println(io, "        none")
            else
                for (i, c) in enumerate(hull.global_constraints)
                    println(io, "        g[$i,d]: $(_scaled_constraint_string(c, "gamma[d]"))")
                end
            end

            for (d, disjunct) in enumerate(block.disjuncts)
                println(io, "      disjunct[$d]:")
                if isempty(disjunct)
                    println(io, "        local constraints: none")
                else
                    for (q, c) in enumerate(disjunct)
                        println(io, "        c[$q]: $(_scaled_constraint_string(c, "gamma[$d]"))")
                    end
                end
            end
        end
    end

    return nothing
end


"""
    print_projection_model(model; formulation = :dnf, io = stdout, kwargs...)

Print the lifted projection model that will be built for DNF, CNF, or partial-DNF.
"""
function print_projection_model(
    model::DisjunctiveModel;
    formulation::Symbol = :dnf,
    io::IO = stdout,
    prune_infeasible::Bool = false,
    num_dnf_rules::Int = -1,
    rule_ordering = nothing,
)
    print_model(model; io = io)

    println(io)
    println(io, "Lifted projection formulation:")

    if formulation == :dnf
        hull = convex_hull_form(model; prune_infeasible = prune_infeasible)
        print_hull(hull; io = io)
    elseif formulation == :cnf
        hull = cnf_hull_form(model)
        print_hull(hull; io = io)
    elseif formulation == :partial_dnf
        hull = partial_dnf_hull_form(
            model;
            ordering = rule_ordering,
            num_dnf_rules = num_dnf_rules,
        )
        print_hull(hull; io = io)
    else
        throw(ArgumentError("Unknown formulation $(formulation). Expected :dnf, :cnf, or :partial_dnf."))
    end

    return nothing
end

"""
    count_constraints(model::JuMP.Model)

Return the total number of constraints in a JuMP model.
"""
function count_constraints(model::JuMP.Model)
    total = 0

    for (F, S) in JuMP.list_of_constraint_types(model)
        total += JuMP.num_constraints(model, F, S)
    end

    return total
end


"""
    model_size(model::JuMP.Model)

Return a named tuple with the number of variables and constraints in a JuMP model.
"""
function model_size(model::JuMP.Model)
    return (
        variables = JuMP.num_variables(model),
        constraints = count_constraints(model),
    )
end


"""
    product_disjunct_count(counts)

Return the full-DNF scenario count implied by a vector of disjunct counts.
"""
product_disjunct_count(counts::AbstractVector{<:Integer}) = prod(counts)


"""
    formulation_summary(model; formulation = :dnf, num_dnf_rules = -1, rule_ordering = nothing)

Return a lightweight summary of the lifted formulation size before building the JuMP model.
"""
function formulation_summary(
    model::DisjunctiveModel;
    formulation::Symbol = :dnf,
    num_dnf_rules::Int = -1,
    rule_ordering = nothing,
    prune_infeasible::Bool = false,
)
    if formulation == :dnf
        hull = convex_hull_form(model; prune_infeasible = prune_infeasible)

        return (
            formulation = :dnf,
            scenarios = num_scenarios(hull),
            cnf_blocks = 0,
            dnf_rules = length(hull.scenarios) == 0 ? 0 : length(hull.scenarios[1].choices),
        )

    elseif formulation == :cnf
        hull = cnf_hull_form(model)

        return (
            formulation = :cnf,
            scenarios = 0,
            cnf_blocks = num_blocks(hull),
            dnf_rules = 0,
        )

    elseif formulation == :partial_dnf
        hull = partial_dnf_hull_form(
            model;
            ordering = rule_ordering,
            num_dnf_rules = num_dnf_rules,
        )

        return (
            formulation = :partial_dnf,
            scenarios = length(hull.dnf_scenarios),
            cnf_blocks = length(hull.cnf_blocks),
            dnf_rules = length(hull.dnf_indices),
        )

    else
        throw(ArgumentError("Unknown formulation $(formulation). Expected :dnf, :cnf, or :partial_dnf."))
    end
end


"""
    benchmark_projection(model, yhat; formulation, label, kwargs...)

Build and solve a projection model, printing formulation size, build time,
solve time, variable count, and constraint count.

This is intended for examples and debugging.
"""
function benchmark_projection(
    dm::DisjunctiveModel,
    yhat;
    formulation::Symbol,
    label::AbstractString,
    num_dnf_rules::Int = -1,
    rule_ordering = nothing,
    solver = nothing,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 1e-8,
    gamma_regularization::Real = 1e-8,
    anchor_regularization::Real = 1e-3,
    run::Bool = true,
    io::IO = stdout,
)
    if !run
        println(io, rpad(label, 20), " skipped")
        return nothing
    end

    hull =
        if formulation == :dnf
            convex_hull_form(dm; prune_infeasible = false)
        elseif formulation == :cnf
            cnf_hull_form(dm)
        elseif formulation == :partial_dnf
            partial_dnf_hull_form(
                dm;
                ordering = rule_ordering,
                num_dnf_rules = num_dnf_rules,
            )
        else
            throw(ArgumentError("Unknown formulation $(formulation)."))
        end

    summary =
        if formulation == :dnf
            "scenarios=$(num_scenarios(hull))"
        elseif formulation == :cnf
            "blocks=$(num_blocks(hull))"
        else
            "dnf_scenarios=$(length(hull.dnf_scenarios)), cnf_blocks=$(length(hull.cnf_blocks))"
        end

    println(io, rpad(label, 20), " building model... ", summary)

    model_ref = Ref{JuMP.Model}()

    build_time = @elapsed begin
        model_ref[] = build_projection_model(
            hull,
            yhat;
            solver = solver,
            y_regularization = y_regularization,
            ycopy_regularization = ycopy_regularization,
            gamma_regularization = gamma_regularization,
            anchor_regularization = anchor_regularization,
        )
    end

    jump_model = model_ref[]
    sz = model_size(jump_model)

    println(
        io,
        rpad(label, 20),
        " built in ", round(build_time; digits = 4), "s",
        " vars=", sz.variables,
        " cons=", sz.constraints,
        ". solving...",
    )

    solve_time = @elapsed begin
        optimize!(jump_model)
    end

    status = termination_status(jump_model)

    y =
        status == MOI.OPTIMAL ?
        Float64.(value.(jump_model[:y])) :
        Float64.(collect(yhat))

    println(
        io,
        rpad(label, 20),
        " status=", status,
        " build=", round(build_time; digits = 4), "s",
        " solve=", round(solve_time; digits = 4), "s",
        " total=", round(build_time + solve_time; digits = 4), "s",
        " vars=", sz.variables,
        " cons=", sz.constraints,
        " y=", round.(y; digits = 5),
    )

    return ProjectionResult(
        y,
        haskey(jump_model.obj_dict, :gamma) ? Float64.(value.(jump_model[:gamma])) : Float64[],
        status,
        jump_model,
    )
end