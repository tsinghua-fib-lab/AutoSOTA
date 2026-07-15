@testset "Standard form and convex hull" begin
        dm = DisjunctiveModel(3)

        set_bounds!(
            dm,
            lower = [0.0, 0.0, 0.0],
            upper = [1.0, 1.0, 1.0],
        )

        add_linear_constraint!(dm, [0.0, 0.0, 1.0], :>=, 0.2)
        add_linear_constraint!(dm, [1.0, 1.0, 0.0], :<=, 1.2)
        add_linear_constraint!(dm, [1.0, -1.0, 0.0], :(==), 0.0)

        d1 = [
            LinearConstraint([1.0, 0.0, 0.0], :<=, 0.4),
        ]

        d2 = [
            LinearConstraint([1.0, 0.0, 0.0], :>=, 0.6),
        ]

        add_disjunction!(dm, d1, d2)

        sm = standard_form(dm)

        @test sm.n_outputs == 3
        @test sm.lb == [0.0, 0.0, 0.0]
        @test sm.ub == [1.0, 1.0, 1.0]

        @test length(sm.global_constraints) == 3
        @test sm.global_constraints[1].sense == :>=
        @test sm.global_constraints[2].sense == :<=
        @test sm.global_constraints[3].sense == :(==)

        @test length(sm.disjunctions) == 1
        @test length(sm.disjunctions[1].disjuncts) == 2

        hull = convex_hull_form(sm; prune_infeasible=false)

        @test hull.n_outputs == 3
        @test hull.global_constraints == sm.global_constraints
        @test num_scenarios(hull) == 2

        @test hull.scenarios[1].choices == [1]
        @test length(hull.scenarios[1].local_constraints) == 1
        @test hull.scenarios[1].local_constraints[1].sense == :<=
        @test hull.scenarios[1].local_constraints[1].b == 0.4

        @test hull.scenarios[2].choices == [2]
        @test length(hull.scenarios[2].local_constraints) == 1
        @test hull.scenarios[2].local_constraints[1].sense == :>=
        @test hull.scenarios[2].local_constraints[1].b == 0.6
    end

@testset "Scenario pruning" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Global equality.
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

        sm = standard_form(dm)

        hull_unpruned = convex_hull_form(
            sm;
            prune_infeasible = false,
        )

        @test num_scenarios(hull_unpruned) == 4

        hull_pruned = convex_hull_form(
            sm;
            prune_infeasible = true,
            interior_tol = 1e-7,
        )

        # Two scenarios are infeasible:
        # y1 <= 0.25, y2 <= 0.25 conflicts with y1 + y2 == 1.
        # y1 >= 0.75, y2 >= 0.75 conflicts with y1 + y2 == 1.
        @test num_scenarios(hull_pruned) == 2

        choices = sort(hull_pruned.scenarios .|> s -> s.choices)

        @test choices == [[1, 2], [2, 1]]
    end