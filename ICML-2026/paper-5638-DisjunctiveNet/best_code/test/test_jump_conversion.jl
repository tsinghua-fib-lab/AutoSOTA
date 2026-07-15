using JuMP

@testset "JuMP-style constraint conversion" begin
    dm = DisjunctiveModel(3)
    y = output_variables(dm)

    c1 = add_linear_constraint!(
        dm,
        @build_constraint(y[1] + 2.0 * y[2] <= 3.0),
    )

    @test c1.a == [1.0, 2.0, 0.0]
    @test c1.sense == :<=
    @test c1.b == 3.0

    c2 = add_linear_constraint!(
        dm,
        @build_constraint(y[1] - y[3] >= 0.5),
    )

    @test c2.a == [1.0, 0.0, -1.0]
    @test c2.sense == :>=
    @test c2.b == 0.5

    c3 = add_linear_constraint!(
        dm,
        @build_constraint(y[1] + y[2] + y[3] == 1.0),
    )


    @test c3.a == [1.0, 1.0, 1.0]
    @test c3.sense == :(==)
    @test c3.b == 1.0

    c4 = add_linear_constraint!(
        dm,
        @build_constraint(sum(y[i] for i in 1:3) <= 2.0),
    )

    @test c4.a == [1.0, 1.0, 1.0]
    @test c4.sense == :<=
    @test c4.b == 2.0

    disj = add_disjunction!(
        dm,
        [@build_constraint(y[1] <= 0.25)],
        [@build_constraint(y[1] >= 0.75)];
        name = :x_split,
    )

    @test disj.name == :x_split
    @test length(disj.disjuncts) == 2
    @test disj.disjuncts[1][1].a == [1.0, 0.0, 0.0]
    @test disj.disjuncts[1][1].sense == :<=
    @test disj.disjuncts[1][1].b == 0.25
    @test disj.disjuncts[2][1].sense == :>=
end

@testset "JuMP-style constraints with constants" begin
    dm = DisjunctiveModel(2)
    y = output_variables(dm)

    c = add_linear_constraint!(
        dm,
        @build_constraint(y[1] + y[2] + 2.0 <= 5.0),
    )
    @test c.a == [1.0, 1.0]
    @test c.sense == :<=
    @test c.b == 3.0
end