"""
    output_dimension(model::DisjunctiveModel)

Return the output dimension of the projection variable.
"""
output_dimension(model::DisjunctiveModel) = model.n_outputs


"""
    lower_bounds(model::DisjunctiveModel)

Return the lower bounds of the projection variable.
"""
lower_bounds(model::DisjunctiveModel) = copy(model.lb)


"""
    upper_bounds(model::DisjunctiveModel)

Return the upper bounds of the projection variable.
"""
upper_bounds(model::DisjunctiveModel) = copy(model.ub)


function _check_dimension(model::DisjunctiveModel, a::AbstractVector)
    length(a) == model.n_outputs ||
        throw(DimensionMismatch("Expected vector of length $(model.n_outputs), got length $(length(a))."))
    return nothing
end


function _check_constraint_dimension(model::DisjunctiveModel, constraint::LinearConstraint)
    _check_dimension(model, constraint.a)
    return nothing
end


"""
    set_bounds!(model; lower, upper)

Set componentwise lower and upper bounds for the projection variable.
"""
function set_bounds!(
    model::DisjunctiveModel;
    lower::AbstractVector{<:Real} = fill(-Inf, model.n_outputs),
    upper::AbstractVector{<:Real} = fill(Inf, model.n_outputs),
)
    length(lower) == model.n_outputs ||
        throw(DimensionMismatch("Lower bound length must be $(model.n_outputs)."))
    length(upper) == model.n_outputs ||
        throw(DimensionMismatch("Upper bound length must be $(model.n_outputs)."))

    lb = Float64.(collect(lower))
    ub = Float64.(collect(upper))

    all(lb .<= ub) ||
        throw(ArgumentError("Each lower bound must be less than or equal to the corresponding upper bound."))

    model.lb .= lb
    model.ub .= ub

    return model
end


"""
    add_linear_constraint!(model, a, sense, b)

Add a scalar affine constraint `a' * y sense b`.
"""
function add_linear_constraint!(
    model::DisjunctiveModel,
    a::AbstractVector{<:Real},
    sense::Symbol,
    b::Real,
)
    _check_dimension(model, a)
    constraint = LinearConstraint(a, sense, b)
    push!(model.constraints, constraint)
    return constraint
end


function add_linear_constraint!(
    model::DisjunctiveModel,
    constraint::LinearConstraint,
)
    _check_constraint_dimension(model, constraint)
    push!(model.constraints, constraint)
    return constraint
end

"""
    add_linear_constraint!(model, constraint)

Add a JuMP-style scalar affine constraint to a `DisjunctiveModel`.

Example:

    y = output_variables(dm)
    add_linear_constraint!(dm, y[1] + y[2] <= 1.0)
"""
function add_linear_constraint!(
    dm::DisjunctiveModel,
    constraint::JuMP.ScalarConstraint,
)
    lc = linear_constraint_from_jump(dm, constraint)
    push!(dm.constraints, lc)
    return lc
end


"""
    add_disjunction!(model, disjuncts...; name = nothing)

Add a disjunction to the model.

Each argument should be a vector of `LinearConstraint`s.
If `name` is provided, it must be unique among named disjunctions.
"""
function add_disjunction!(
    model::DisjunctiveModel,
    disjuncts::Vector{LinearConstraint}...;
    name::Union{Nothing, Symbol} = nothing,
)
    isempty(disjuncts) && throw(ArgumentError("At least one disjunct is required."))

    if name !== nothing
        for existing in model.disjunctions
            existing.name == name &&
                throw(ArgumentError("A disjunction named $(name) already exists."))
        end
    end

    copied_disjuncts = Vector{Vector{LinearConstraint}}()

    for disjunct in disjuncts
        isempty(disjunct) && throw(ArgumentError("Each disjunct must contain at least one constraint."))
        for constraint in disjunct
            _check_constraint_dimension(model, constraint)
        end
        push!(copied_disjuncts, collect(disjunct))
    end

    disjunction = Disjunction(copied_disjuncts; name = name)
    push!(model.disjunctions, disjunction)
    return disjunction
end

function _convert_jump_disjunct(dm::DisjunctiveModel, disjunct::Vector{<:JuMP.ScalarConstraint},)
    return [linear_constraint_from_jump(dm, c) for c in disjunct]
end


function add_disjunction!(
    dm::DisjunctiveModel,
    disjuncts::Vector{<:JuMP.ScalarConstraint}...;
    name::Union{Nothing, Symbol} = nothing,
)
    converted = [_convert_jump_disjunct(dm, disjunct) for disjunct in disjuncts]
    return add_disjunction!(dm, converted...; name = name)
end

"""
    linear_constraints(model)

Return ordinary linear constraints.
"""
linear_constraints(model::DisjunctiveModel) = copy(model.constraints)


"""
    disjunctions(model)

Return disjunctive constraints.
"""
disjunctions(model::DisjunctiveModel) = copy(model.disjunctions)

"""
    output_variables(model)

Return the JuMP variables used for JuMP-style constraint construction.
"""
output_variables(dm::DisjunctiveModel) = dm.y