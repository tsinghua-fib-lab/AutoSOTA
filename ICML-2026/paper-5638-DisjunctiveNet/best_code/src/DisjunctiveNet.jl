module DisjunctiveNet

include("core/types.jl")
include("core/utils.jl")

include("modeling/modeling.jl")
include("modeling/jump_conversion.jl")
include("modeling/standard_form.jl")

include("formulations/convex_hull.jl")
include("formulations/cnf_hull.jl")
include("formulations/partial_dnf_hull.jl")

include("backend/projection_backend.jl")
include("backend/differentiation.jl")

include("layers/projection_layer.jl")
include("layers/flux.jl")
include("layers/constrained_model.jl")

include("display/display.jl")


export DisjunctiveModel
export DisjunctiveProjectionLayer
export ProjectionConfig

export LinearConstraint
export Disjunction
export StandardDisjunctiveModel
export ConvexHullForm
export ConvexHullScenario
export PartialDNFHullForm
export partial_dnf_hull_form

export CNFConvexHullForm
export CNFConvexHullBlock
export cnf_hull_form
export num_blocks

export output_dimension
export lower_bounds
export upper_bounds
export set_bounds!
export add_linear_constraint!
export add_disjunction!
export linear_constraints
export disjunctions
export output_variables
export linear_constraint_from_jump

export standard_form
export convex_hull_form
export num_scenarios
export scenario_has_interior_point

export projection_mode

export ProjectionResult
export project
export build_projection_model
export projection_formulation
export ConstrainedFluxModel
export constrained_model

export print_model
export print_hull
export print_projection_model

export count_constraints
export model_size
export product_disjunct_count
export formulation_summary
export benchmark_projection

end