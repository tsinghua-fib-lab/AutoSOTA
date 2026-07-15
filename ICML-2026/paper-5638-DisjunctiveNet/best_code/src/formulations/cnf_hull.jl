"""
    cnf_hull_form(model::DisjunctiveModel)

Build a CNF-style convex-hull representation.
"""
cnf_hull_form(model::DisjunctiveModel) = cnf_hull_form(standard_form(model))


"""
    cnf_hull_form(model::StandardDisjunctiveModel)

Build one convex-hull block per disjunction.

This avoids full DNF scenario enumeration.
"""
function cnf_hull_form(model::StandardDisjunctiveModel)
    blocks = CNFConvexHullBlock[]

    for (r, disjunction) in enumerate(model.disjunctions)
        push!(
            blocks,
            CNFConvexHullBlock(
                r,
                deepcopy(disjunction.disjuncts),
            ),
        )
    end

    return CNFConvexHullForm(
        model.n_outputs,
        model.lb,
        model.ub,
        model.global_constraints,
        blocks,
    )
end


num_blocks(hull::CNFConvexHullForm) = length(hull.blocks)