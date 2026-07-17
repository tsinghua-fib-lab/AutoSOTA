using GPFiniteVolume
using FunctionalGPs
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using Test

import GaussianMarkovRandomFields: mean, std

@testset "GPFiniteVolume.jl" begin
    @testset "1D Poisson via GP-FVM" begin
        # Solve  -u''(x) = π² sin(πx)  on [0,1] with u(0) = u(1) = 0.
        # Exact:  u(x) = sin(πx).
        #
        # The defining feature of GP-FVM is that the joint state vector contains
        # cell averages as first-class variables alongside face values and face
        # derivatives:
        #     [ u at face points ;  u' at face points ;  ∫_cell u(x) dx ]
        # The joint Gaussian prior over these functionals couples them
        # automatically — there is no need to recover cell averages by
        # post-hoc quadrature of face values. The FVM cell-balance constraint
        # itself is linear in the face derivatives:
        #     ∫_cell -u''(x) dx  =  ∫_cell f(x) dx
        #  ⇔  u'(x_i) − u'(x_{i+1})  =  ∫_{x_i}^{x_{i+1}} π² sin(πx) dx
        #                            =  π [cos(π x_i) − cos(π x_{i+1})].
        # We condition the joint prior on this constraint plus the Dirichlet
        # boundary conditions and then verify that BOTH the face values and the
        # cell averages of the posterior agree with the analytical solution.

        N = 21
        endpoints = collect(range(0.0, 1.0, length=N))
        intervals = intervals_from_endpoints(endpoints)
        k = HalfIntegerMaternKernel(2, [0.2])

        x0, layout, approx = sparse_gmrf([
            :f      => EvaluationFunctional(endpoints),
            :f_dx   => EvaluationFunctional(endpoints) ∘ PartialDerivative((1,)),
            :f_int  => VectorizedLebesgueIntegral(intervals),
        ], k; ρ=2.0, ordering=:integrals_coarsest)

        @test x0 isa GMRF
        @test approx.info.n == 3N - 1
        @test approx.info.fill_pct < 35  # sparse approximation stays sparse (small N → higher relative fill)

        f_idx   = indices(layout, :f)
        dx_idx  = indices(layout, :f_dx)
        int_idx = indices(layout, :f_int)
        n_cells = N - 1

        # PDE cell-balance (rows 1..n_cells) + Dirichlet BCs (rows n_cells+1, +2)
        A = spzeros(n_cells + 2, approx.info.n)
        y = zeros(n_cells + 2)
        for i in 1:n_cells
            A[i, dx_idx[i]]     =  1.0
            A[i, dx_idx[i + 1]] = -1.0
            y[i] = π * (cos(π * endpoints[i]) - cos(π * endpoints[i + 1]))
        end
        A[n_cells + 1, f_idx[1]] = 1.0    # u(0) = 0
        A[n_cells + 2, f_idx[N]] = 1.0    # u(1) = 0

        σ² = 1e-8
        x_post = condition_precision(approx.Q; A=A, Q_ϵ=(1/σ²) * I, y=y)
        μ = mean(x_post)
        σ_post = std(x_post)

        # --- Face values: posterior u at the face points matches truth ---
        u_face_truth = sinpi.(endpoints)
        rmse_face = sqrt(sum(abs2, μ[f_idx] .- u_face_truth) / N)
        @test rmse_face < 1e-2
        @test abs(μ[f_idx[1]]) < 1e-3
        @test abs(μ[f_idx[N]]) < 1e-3

        # --- Cell averages: this is what GP-FVM gives you on top of collocation ---
        # The :f_int block stores ∫_cell u(x) dx directly; divide by Δx for averages.
        Δx = [intervals[i][2] - intervals[i][1] for i in 1:n_cells]
        u_cell_post  = μ[int_idx] ./ Δx
        # Analytical cell average of sin(πx) on [a, b] is (cos(πa) - cos(πb)) / (π·Δx).
        u_cell_truth = [
            (cos(π * intervals[i][1]) - cos(π * intervals[i][2])) / (π * Δx[i])
            for i in 1:n_cells
        ]
        rmse_cell = sqrt(sum(abs2, u_cell_post .- u_cell_truth) / n_cells)
        @test rmse_cell < 1e-2

        # Posterior std on the cell-integral block is small (the joint constraints
        # pin down the cell averages, not just the face values).
        @test maximum(σ_post[int_idx]) < 0.1
    end
end
