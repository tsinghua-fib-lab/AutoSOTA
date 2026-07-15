using Test
using DisjunctiveNet
using JuMP
using Flux
using Zygote
import MathOptInterface as MOI

@testset "DisjunctiveNet.jl" begin
    include("test_modeling.jl")
    include("test_standard_form.jl")
    include("test_hulls.jl")
    include("test_projection_backend.jl")
    include("test_differentiation.jl")
    include("test_flux.jl")
    include("test_display.jl")
    include("test_stress.jl")
    include("test_jump_conversion.jl")
end