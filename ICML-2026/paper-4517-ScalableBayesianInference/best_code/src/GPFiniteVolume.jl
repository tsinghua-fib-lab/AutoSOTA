module GPFiniteVolume

using PrecompileTools: @setup_workload, @compile_workload

# Core utilities
include("utils.jl")

# Named slices for spacetime indexing (needed by sparse_precision.jl)
include("named_slices.jl")

# Functional type classification
include("functional_types.jl")

# Sparse precision construction
include("sparse_precision.jl")

# Efficient conditioning (skip intermediate GMRF construction)
include("condition_precision.jl")

# Time evolution (SDEs and SSMs)
include("linear_sdes.jl")
include("block_tridiagonal_prec.jl")
include("linear_ssm.jl")

# EKF-style sequential solver
include("ekf_solver.jl")

# ------------------------------------------------------------------------------
# Precompilation Workloads
# ------------------------------------------------------------------------------

@setup_workload begin
    using FunctionalGPs
    import FunctionalGPs: ⊗

    @compile_workload begin
        # 1D problem (non-Kronecker blocks, tests fallback path)
        xs_1d = range(0.0, 1.0, length=15)
        intervals_1d = intervals_from_endpoints(collect(xs_1d))
        k_1d = HalfIntegerMaternKernel(2, [0.2])

        sparse_precision([
            :f => EvaluationFunctional(xs_1d),
            :f_int => VectorizedLebesgueIntegral(intervals_1d),
        ], k_1d; ρ=2.0)

        # 2D tensor product problem (Kronecker blocks, tests FastBlockMatrix)
        xs_2d = range(0.0, 1.0, length=8)
        ys_2d = range(0.0, 1.0, length=8)
        xi_2d = intervals_from_endpoints(collect(xs_2d))
        yi_2d = intervals_from_endpoints(collect(ys_2d))
        grid_2d = FactorizedGrid(xs_2d, ys_2d)
        k_2d = HalfIntegerMaternKernel(2, [0.2]) ⊗ HalfIntegerMaternKernel(2, [0.2])

        sparse_precision([
            :c => EvaluationFunctional(grid_2d),
            :c_vert => EvaluationFunctional(xs_2d) ⊗ VectorizedLebesgueIntegral(yi_2d),
        ], k_2d; ρ=2.0)
    end
end

end
