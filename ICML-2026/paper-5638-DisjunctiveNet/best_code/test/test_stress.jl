@testset "Differentiation stress: symmetric 4-scenario hull" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Global equality: keeps the solution on a line.
        add_linear_constraint!(dm, [1.0, 1.0], :(==), 1.0)

        # Disjunction 1:
        # y1 <= 0.25 OR y1 >= 0.75
        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0], :<=, 0.25)],
            [LinearConstraint([1.0, 0.0], :>=, 0.75)],
        )

        # Disjunction 2:
        # y2 <= 0.25 OR y2 >= 0.75
        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 1.0], :<=, 0.25)],
            [LinearConstraint([0.0, 1.0], :>=, 0.75)],
        )

        layer = DisjunctiveProjectionLayer(
            dm;
            y_regularization = 1e-4,
            ycopy_regularization = 1e-4,
            gamma_regularization = 1e-4,
            anchor_regularization = 1e-4,
        )

        # Exactly central. This is intentionally degenerate.
        yhat = [0.5, 0.5]

        y = layer(yhat)

        @test length(y) == 2
        @test all(isfinite, y)
        @test isapprox(sum(y), 1.0; atol = 1e-5)

        grad = Zygote.gradient(z -> sum(layer(z)), yhat)[1]

        @test grad !== nothing
        @test length(grad) == 2
        @test all(isfinite, grad)
    end

    @testset "Differentiation stress: redundant constraints" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Same equality expressed redundantly.
        add_linear_constraint!(dm, [1.0, 1.0], :(==), 1.0)
        add_linear_constraint!(dm, [2.0, 2.0], :(==), 2.0)

        # Redundant inequalities implied by the equality.
        add_linear_constraint!(dm, [1.0, 1.0], :>=, 1.0)
        add_linear_constraint!(dm, [1.0, 1.0], :<=, 1.0)

        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0], :<=, 0.25)],
            [LinearConstraint([1.0, 0.0], :>=, 0.75)],
        )

        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 1.0], :<=, 0.25)],
            [LinearConstraint([0.0, 1.0], :>=, 0.75)],
        )

        layer = DisjunctiveProjectionLayer(
            dm;
            anchor_regularization = 1e-3,
        )

        yhat = [0.5, 0.5]

        y = layer(yhat)

        @test length(y) == 2
        @test all(isfinite, y)
        @test isapprox(sum(y), 1.0; atol = 1e-5)

        grad = Zygote.gradient(z -> sum(layer(z)), yhat)[1]

        @test grad !== nothing
        @test length(grad) == 2
        @test all(isfinite, grad)
    end

    @testset "Differentiation stress: 3D 8-scenario hull" begin
        dm = DisjunctiveModel(3)

        set_bounds!(
            dm,
            lower = [0.0, 0.0, 0.0],
            upper = [1.0, 1.0, 1.0],
        )

        # Global simplex equality.
        add_linear_constraint!(dm, [1.0, 1.0, 1.0], :(==), 1.0)

        # Symmetric split on each variable.
        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0, 0.0], :<=, 0.2)],
            [LinearConstraint([1.0, 0.0, 0.0], :>=, 0.6)],
        )

        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 1.0, 0.0], :<=, 0.2)],
            [LinearConstraint([0.0, 1.0, 0.0], :>=, 0.6)],
        )

        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 0.0, 1.0], :<=, 0.2)],
            [LinearConstraint([0.0, 0.0, 1.0], :>=, 0.6)],
        )

        layer = DisjunctiveProjectionLayer(
            dm;
            anchor_regularization = 1e-3,
        )

        yhat = [1/3, 1/3, 1/3]

        y = layer(yhat)

        @test length(y) == 3
        @test all(isfinite, y)
        @test isapprox(sum(y), 1.0; atol = 1e-5)

        grad = Zygote.gradient(z -> sum(layer(z)), yhat)[1]

        @test grad !== nothing
        @test length(grad) == 3
        @test all(isfinite, grad)
    end
