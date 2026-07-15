@testset "Flux integration" begin
    dm = DisjunctiveModel(2)

    set_bounds!(
        dm,
        lower = [0.0, 0.0],
        upper = [1.0, 1.0],
    )

    # Global constraint: y1 + y2 >= 0.8
    add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

    d1 = [
        LinearConstraint([1.0, 0.0], :<=, 0.25),
    ]

    d2 = [
        LinearConstraint([1.0, 0.0], :>=, 0.75),
    ]

    add_disjunction!(dm, d1, d2)

    layer = DisjunctiveProjectionLayer(
        dm;
        y_regularization = 1e-4,
        ycopy_regularization = 1e-4,
        gamma_regularization = 1e-4,
        anchor_regularization = 1e-4,
    )

    model = Chain(
        Dense(2 => 8, relu),
        Dense(8 => 2),
        layer,
    )

    x = Float32[0.3, 0.4]

    y = model(x)

    @test length(y) == 2
    @test y[1] >= -1e-6
    @test y[1] <= 1.0 + 1e-6
    @test y[2] >= -1e-6
    @test y[2] <= 1.0 + 1e-6
    @test y[1] + y[2] >= 0.8 - 1e-6

    ps = Flux.trainables(model)
    @test !isempty(ps)

    loss(m, x) = sum(m(x))
    grads = Flux.gradient(m -> loss(m, x), model)
    @test grads !== nothing
end

@testset "ConstrainedFluxModel wrapper" begin
    backbone = Chain(
        Dense(2 => 8, relu),
        Dense(8 => 2),
    )

    x0 = Float32[0.3, 0.4]

    model = constrained_model(
        backbone,
        x0;
        formulation = :cnf,
        y_regularization = 1e-4,
        ycopy_regularization = 1e-4,
        gamma_regularization = 1e-4,
        anchor_regularization = 1e-4,
    ) do dm
        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

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
    end

    y = model(x0)

    @test length(y) == 2
    @test all(isfinite, y)
    @test y[1] + y[2] >= 0.8 - 1e-6

    ps = Flux.trainables(model)
    @test !isempty(ps)

    loss(m, x) = sum(m(x))
    grads = Flux.gradient(m -> loss(m, x0), model)

    @test grads !== nothing
end

@testset "ConstrainedFluxModel training setup" begin
    backbone = Chain(
        Dense(3 => 8, relu),
        Dense(8 => 2),
    )

    x = Float32[0.2, 0.7, 0.4]
    target = Float32[0.8, 0.2]

    model = constrained_model(
        backbone,
        x;
        formulation = :partial_dnf,
        num_dnf_rules = 1,
        rule_ordering = [:x_rule, :y_rule],
        y_regularization = 1e-4,
        ycopy_regularization = 1e-4,
        gamma_regularization = 1e-4,
        anchor_regularization = 1e-4,
    ) do dm
        set_bounds!(dm, lower = [0.0, 0.0], upper = [1.0, 1.0])
        add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

        add_disjunction!(
            dm,
            [LinearConstraint([1.0, 0.0], :<=, 0.2)],
            [
                LinearConstraint([1.0, 0.0], :>=, 0.4),
                LinearConstraint([1.0, 0.0], :<=, 0.6),
            ],
            [LinearConstraint([1.0, 0.0], :>=, 0.8)];
            name = :x_rule,
        )

        add_disjunction!(
            dm,
            [LinearConstraint([0.0, 1.0], :<=, 0.2)],
            [
                LinearConstraint([0.0, 1.0], :>=, 0.35),
                LinearConstraint([0.0, 1.0], :<=, 0.55),
            ],
            [LinearConstraint([0.0, 1.0], :>=, 0.7)];
            name = :y_rule,
        )
    end

    y = model(x)
    @test length(y) == 2
    @test sum(y) >= 0.8 - 1e-6

    opt = Flux.setup(Adam(1e-3), model)

    loss(m, x, target) = sum(abs2, m(x) .- target)

    l, grads = Flux.withgradient(model) do m
        loss(m, x, target)
    end

    Flux.update!(opt, model, grads[1])

    @test isfinite(l)

    y_after = model(x)
    @test length(y_after) == 2
    @test sum(y_after) >= 0.8 - 1e-6
end