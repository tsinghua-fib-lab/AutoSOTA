"""
    standard_form(model::DisjunctiveModel)

Convert a user-facing `DisjunctiveModel` into the package's internal standard
form.

This step preserves constraint senses. Constraints are not converted to a
single inequality direction.
"""
function standard_form(model::DisjunctiveModel)
    return StandardDisjunctiveModel(
        model.n_outputs,
        model.lb,
        model.ub,
        model.constraints,
        model.disjunctions,
    )
end