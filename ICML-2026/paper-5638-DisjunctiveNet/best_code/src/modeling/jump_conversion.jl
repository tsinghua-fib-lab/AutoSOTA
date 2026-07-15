using JuMP
import MathOptInterface as MOI

function _term_coefficient_variable(term)
    if term isa Tuple
        return term[1], term[2]
    else
        return term.coefficient, term.variable
    end
end

"""
    linear_constraint_from_jump(model, constraint)

Convert a JuMP scalar affine constraint into a `LinearConstraint`.
"""
function linear_constraint_from_jump(dm::DisjunctiveModel, constraint::JuMP.ScalarConstraint)
    func = constraint.func
    set = constraint.set

    func isa JuMP.GenericAffExpr || throw(
        ArgumentError("Only scalar affine constraints are supported.")
    )

    a = zeros(Float64, dm.n_outputs)

    for term in JuMP.linear_terms(func)
        coeff, var = _term_coefficient_variable(term)
        idx = _output_variable_index(dm, var)
        a[idx] += Float64(coeff)
    end

    constant = Float64(JuMP.constant(func))

    if set isa MOI.LessThan
        # affine <= upper
        sense = :<=
        b = Float64(set.upper) - constant
    elseif set isa MOI.GreaterThan
        # affine >= lower
        sense = :>=
        b = Float64(set.lower) - constant
    elseif set isa MOI.EqualTo
        # affine == value
        sense = :(==)
        b = Float64(set.value) - constant
    else
        throw(ArgumentError("Unsupported JuMP constraint set $(typeof(set))."))
    end

    return LinearConstraint(a, sense, b)
end


function _output_variable_index(dm::DisjunctiveModel, var::JuMP.VariableRef)
    for (j, yj) in enumerate(dm.y)
        if var == yj
            return j
        end
    end

    throw(ArgumentError("Constraint contains variable $(var) that does not belong to this DisjunctiveModel."))
end