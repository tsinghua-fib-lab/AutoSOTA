@testset "Display utilities" begin
        dm = DisjunctiveModel(2)
        set_bounds!(dm, lower = [0.0, 0.0], upper = [1.0, 1.0])

        add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0], :<=, 0.25)],
            [LinearConstraint([1.0, 0.0], :>=, 0.75)],
        )

        buf = IOBuffer()
        print_model(dm; io = buf)
        str = String(take!(buf))

        @test occursin("DisjunctiveModel", str)
        @test occursin("global constraints", str)
        @test occursin("disjunction[1]", str)

        buf = IOBuffer()
        print_projection_model(dm; formulation = :dnf, io = buf)
        str = String(take!(buf))

        @test occursin("DNF ConvexHullForm", str)

        buf = IOBuffer()
        print_projection_model(dm; formulation = :cnf, io = buf)
        str = String(take!(buf))

        @test occursin("CNF ConvexHullForm", str)
    end

@testset "Display utilities: partial DNF" begin
    dm = DisjunctiveModel(2)
    set_bounds!(dm, lower = [0.0, 0.0], upper = [1.0, 1.0])

    add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

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

    buf = IOBuffer()
    print_model(dm; io = buf)
    str = String(take!(buf))

    @test occursin("name=rule_x", str)
    @test occursin("name=rule_y", str)

    buf = IOBuffer()
    print_projection_model(
        dm;
        formulation = :partial_dnf,
        num_dnf_rules = 1,
        rule_ordering = [:rule_y, :rule_x],
        io = buf,
    )
    str = String(take!(buf))

    @test occursin("Partial-DNF HullForm", str)
    @test occursin("DNF-expanded rules: [2]", str)
    @test occursin("DNF scenarios: 2", str)
    @test occursin("CNF blocks: 1", str)
    @test occursin("scenario[1]", str)
    @test occursin("block[1]", str)
end

@testset "Display and benchmark utilities" begin
    dm = DisjunctiveModel(2)
    set_bounds!(dm, lower = [0.0, 0.0], upper = [1.0, 1.0])

    add_disjunction!(
        dm,
        [LinearConstraint([1.0, 0.0], :<=, 0.25)],
        [LinearConstraint([1.0, 0.0], :>=, 0.75)];
        name = :x_split,
    )

    add_disjunction!(
        dm,
        [LinearConstraint([0.0, 1.0], :<=, 0.25)],
        [LinearConstraint([0.0, 1.0], :>=, 0.75)];
        name = :y_split,
    )

    @test product_disjunct_count([2, 2, 3]) == 12

    dnf_summary = formulation_summary(dm; formulation = :dnf)
    @test dnf_summary.formulation == :dnf
    @test dnf_summary.scenarios == 4

    cnf_summary = formulation_summary(dm; formulation = :cnf)
    @test cnf_summary.formulation == :cnf
    @test cnf_summary.cnf_blocks == 2

    partial_summary = formulation_summary(
        dm;
        formulation = :partial_dnf,
        num_dnf_rules = 1,
        rule_ordering = [:x_split, :y_split],
    )
    @test partial_summary.formulation == :partial_dnf
    @test partial_summary.scenarios == 2
    @test partial_summary.cnf_blocks == 1

    result = project(dm, [0.5, 0.5]; formulation = :cnf)
    sz = model_size(result.model)

    @test sz.variables > 0
    @test sz.constraints > 0

    buf = IOBuffer()
    bench_result = benchmark_projection(
        dm,
        [0.5, 0.5];
        formulation = :cnf,
        label = "CNF",
        io = buf,
    )

    output = String(take!(buf))

    @test bench_result.status == MOI.OPTIMAL
    @test occursin("CNF", output)
    @test occursin("vars=", output)
    @test occursin("cons=", output)
end