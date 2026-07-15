@testset "CNF and DNF formulation selection" begin
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

        dnf = convex_hull_form(dm; prune_infeasible = false)
        cnf = cnf_hull_form(dm)

        @test num_scenarios(dnf) == 4
        @test num_blocks(cnf) == 2
        @test length(cnf.blocks[1].disjuncts) == 2
        @test length(cnf.blocks[2].disjuncts) == 2

        yhat = [0.5, 0.5]

        dnf_layer = DisjunctiveProjectionLayer(dm; formulation = :dnf)
        cnf_layer = DisjunctiveProjectionLayer(dm; formulation = :cnf)

        y_dnf = dnf_layer(yhat)
        y_cnf = cnf_layer(yhat)

        @test length(y_dnf) == 2
        @test length(y_cnf) == 2
        @test all(isfinite, y_dnf)
        @test all(isfinite, y_cnf)

        @test projection_formulation(dnf_layer) == :dnf
        @test projection_formulation(cnf_layer) == :cnf
    end

@testset "Partial DNF formulation" begin
    # existing partial DNF test
end