@testset "Differentiation" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Simple global constraint: y1 + y2 >= 0.8
        add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

        d1 = [
            LinearConstraint([1.0, 0.0], :<=, 0.25),
        ]

        d2 = [
            LinearConstraint([1.0, 0.0], :>=, 0.75),
        ]

        add_disjunction!(dm, d1, d2)

        layer = DisjunctiveProjectionLayer(dm)

        yhat = [0.5, 0.1]

        grad = Zygote.gradient(y -> sum(layer(y)), yhat)[1]

        @test grad !== nothing
        @test length(grad) == 2
        @test all(isfinite, grad)
    end