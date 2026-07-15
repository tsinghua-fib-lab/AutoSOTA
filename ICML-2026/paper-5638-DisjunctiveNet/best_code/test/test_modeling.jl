@testset "DisjunctiveModel" begin
    dm = DisjunctiveModel(3)

    @test output_dimension(dm) == 3
    @test dm.metadata isa Dict{Symbol, Any}

    @test_throws ArgumentError DisjunctiveModel(0)
    @test_throws ArgumentError DisjunctiveModel(-1)
end

@testset "ProjectionConfig" begin
    cfg = ProjectionConfig()
    @test cfg.mode == :dnf_qp
    @test cfg.gradient == :diffopt
    @test cfg.fallback == :identity

    @test ProjectionConfig().formulation == :dnf
    @test ProjectionConfig(formulation = :cnf).formulation == :cnf
    @test ProjectionConfig(formulation = :partial_dnf).formulation == :partial_dnf

    @test ProjectionConfig(num_dnf_rules = -1).num_dnf_rules == -1
    @test ProjectionConfig(num_dnf_rules = 1).num_dnf_rules == 1

    @test_throws ArgumentError ProjectionConfig(formulation = :bad_formulation)
    @test_throws ArgumentError ProjectionConfig(mode = :bad_mode)
    @test_throws ArgumentError ProjectionConfig(gradient = :bad_gradient)
    @test_throws ArgumentError ProjectionConfig(fallback = :bad_fallback)
    @test_throws ArgumentError ProjectionConfig(mode = :milp, gradient = :diffopt)
    @test_throws ArgumentError ProjectionConfig(num_dnf_rules = -2)
end

@testset "Modeling interface" begin
    dm = DisjunctiveModel(3)

    set_bounds!(dm, lower = [0.0, 0.0, 0.0], upper = [1.0, 2.0, 3.0])
    @test lower_bounds(dm) == [0.0, 0.0, 0.0]
    @test upper_bounds(dm) == [1.0, 2.0, 3.0]

    c1 = add_linear_constraint!(dm, [0.0, 0.0, 1.0], :>=, 0.2)
    @test c1 isa LinearConstraint
    @test length(linear_constraints(dm)) == 1

    d1 = [
        LinearConstraint([1.0, 0.0, 0.0], :<=, 0.4),
        LinearConstraint([0.0, 1.0, 0.0], :<=, 0.7),
    ]

    d2 = [
        LinearConstraint([1.0, 0.0, 0.0], :>=, 0.6),
        LinearConstraint([0.0, 1.0, 0.0], :>=, 0.3),
    ]

    disj = add_disjunction!(dm, d1, d2)
    @test disj isa Disjunction
    @test length(disjunctions(dm)) == 1
    @test length(disjunctions(dm)[1].disjuncts) == 2

    @test_throws DimensionMismatch add_linear_constraint!(dm, [1.0, 2.0], :<=, 1.0)
    @test_throws DimensionMismatch set_bounds!(dm, lower = [0.0], upper = [1.0])
    @test_throws ArgumentError set_bounds!(dm, lower = [1.0, 0.0, 0.0], upper = [0.0, 1.0, 1.0])
    @test_throws ArgumentError LinearConstraint([1.0, 0.0, 0.0], :bad_sense, 1.0)
end