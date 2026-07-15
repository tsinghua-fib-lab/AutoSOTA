@testset "Projection backend" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Global constraint: y1 + y2 >= 0.8
        add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

        # Disjunction:
        # either y1 <= 0.25
        # or     y1 >= 0.75
        d1 = [
            LinearConstraint([1.0, 0.0], :<=, 0.25),
        ]

        d2 = [
            LinearConstraint([1.0, 0.0], :>=, 0.75),
        ]

        add_disjunction!(dm, d1, d2)

        yhat = [0.5, 0.1]

        result = project(dm, yhat)

        @test result.status == MOI.OPTIMAL
        @test length(result.y) == 2
        @test length(result.gamma) == 2

        # Bounds
        @test result.y[1] >= -1e-6
        @test result.y[1] <= 1.0 + 1e-6
        @test result.y[2] >= -1e-6
        @test result.y[2] <= 1.0 + 1e-6

        # Global constraint
        @test result.y[1] + result.y[2] >= 0.8 - 1e-6

        # Convex hull of y1 <= 0.25 OR y1 >= 0.75 over [0,1] is actually [0,1],
        # so y1 may remain near 0.5. This is expected for convex-hull relaxation.
        @test isapprox(sum(result.gamma), 1.0; atol = 1e-6)

        layer = DisjunctiveProjectionLayer(dm)
        yproj = layer(yhat)

        @test length(yproj) == 2
        @test yproj[1] + yproj[2] >= 0.8 - 1e-6
    end

@testset "DisjunctiveProjectionLayer" begin
        dm = DisjunctiveModel(3)

        layer = DisjunctiveProjectionLayer(
            dm;
            y_regularization = 0.0,
            ycopy_regularization = 0.0,
            gamma_regularization = 0.0,
            anchor_regularization = 0.0
        )
        @test projection_mode(layer) == :dnf_qp

        milp_layer = DisjunctiveProjectionLayer(dm; mode = :milp)
        @test projection_mode(milp_layer) == :milp
        @test milp_layer.config.gradient == :straight_through

        yhat = [1.0, 2.0, 3.0]
        @test isapprox(layer(yhat), yhat; atol = 1e-3)
    end


@testset "CNF projection backend" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

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

        yhat = [0.5, 0.5]

        result = project(dm, yhat; formulation = :cnf)

        @test result.status == MOI.OPTIMAL
        @test length(result.y) == 2
        @test all(isfinite, result.y)
        @test result.y[1] >= -1e-6
        @test result.y[1] <= 1.0 + 1e-6
        @test result.y[2] >= -1e-6
        @test result.y[2] <= 1.0 + 1e-6

        layer = DisjunctiveProjectionLayer(dm; formulation = :cnf)
        y = layer(yhat)

        @test length(y) == 2
        @test all(isfinite, y)
    end

@testset "Partial DNF formulation" begin
        dm = DisjunctiveModel(2)

        set_bounds!(dm, lower = [0.0, 0.0], upper = [1.0, 1.0])

        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0], :<=, 0.25)],
            [LinearConstraint([1.0, 0.0], :>=, 0.75)];
            name = :rule_x,
        )

        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 1.0], :<=, 0.25)],
            [LinearConstraint([0.0, 1.0], :>=, 0.75)];
            name = :rule_y,
        )

        hull0 = partial_dnf_hull_form(dm; num_dnf_rules = 0)
        @test length(hull0.dnf_scenarios) == 0
        @test length(hull0.cnf_blocks) == 2

        hull1 = partial_dnf_hull_form(dm; num_dnf_rules = 1)
        @test length(hull1.dnf_scenarios) == 2
        @test length(hull1.cnf_blocks) == 1

        hull2 = partial_dnf_hull_form(dm; num_dnf_rules = -1)
        @test length(hull2.dnf_scenarios) == 4
        @test length(hull2.cnf_blocks) == 0

        hull_named = partial_dnf_hull_form(
            dm;
            ordering = [:rule_y, :rule_x],
            num_dnf_rules = 1,
        )

        @test hull_named.dnf_indices == [2]
        @test length(hull_named.dnf_scenarios) == 2
        @test length(hull_named.cnf_blocks) == 1

        yhat = [0.5, 0.5]

        layer = DisjunctiveProjectionLayer(
            dm;
            formulation = :partial_dnf,
            num_dnf_rules = 1,
            rule_ordering = [:rule_x, :rule_y],
        )

        y = layer(yhat)

        @test length(y) == 2
        @test all(isfinite, y)
    end