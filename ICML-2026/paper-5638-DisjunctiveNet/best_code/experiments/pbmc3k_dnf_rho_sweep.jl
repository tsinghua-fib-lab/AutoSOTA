#!/usr/bin/env julia --project=.
# Quick 1-run rho sweep to find optimal constraint strength
# Uses multi-gene rules, HIDDEN_DIM=32

using Flux
using Statistics
using Random
using LinearAlgebra
using DisjunctiveNet
using NPZ
import MathOptInterface as MOI
using Printf

const N_FEATURES = 1838
const N_CLASSES = 8
const HIDDEN_DIM = 32
const TRAIN_FRACTION = 0.005
const TEST_FRACTION = 0.1
const SPLIT_SEED = 42
const BASE_EPOCHS = 500
const LEARNING_RATE = 3e-3

const Y_REG = 1e-4
const YCOPY_REG = 1e-4
const GAMMA_REG = 1e-4
const ANCHOR_REG = 1e-3

const RHO_VALUES = Float64[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

function load_pbmc3k_data()
    data_dir = "/datasets/pbmc3k"
    X = Float32.(NPZ.npzread(joinpath(data_dir, "X.npy")))
    y = Int.(NPZ.npzread(joinpath(data_dir, "y.npy")))
    gene_names = String[]
    open(joinpath(data_dir, "gene_names.txt"), "r") do f
        for line in eachline(f)
            push!(gene_names, strip(line))
        end
    end
    cell_type_names = String[]
    open(joinpath(data_dir, "cell_type_names.txt"), "r") do f
        for line in eachline(f)
            push!(cell_type_names, strip(line))
        end
    end
    X = Float32.(permutedims(X, (2, 1)))  # (cells, features) -> (features, cells)
    Y = Flux.onehotbatch(y .+ 1, 1:N_CLASSES)
    return X, y, Y, gene_names, cell_type_names
end

function split_data(X, Y, y; train_frac=TRAIN_FRACTION, test_frac=TEST_FRACTION, seed=SPLIT_SEED)
    rng = MersenneTwister(seed)
    n_cells = size(X, 2)
    idx = randperm(rng, n_cells)
    n_test = Int(round(test_frac * n_cells))
    n_train = Int(round(train_frac * n_cells))
    test_idx = idx[1:n_test]
    train_idx = idx[(n_test+1):(n_test+n_train)]
    X_train = Float32.(X[:, train_idx])
    X_test = Float32.(X[:, test_idx])
    Y_train = Float32.(Y[:, train_idx])
    Y_test = Float32.(Y[:, test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]
    return X_train, Y_train, y_train, X_test, Y_test, y_test, train_idx, test_idx
end

function build_backbone(rng)
    W1 = Flux.glorot_uniform(rng, HIDDEN_DIM, N_FEATURES)
    b1 = zeros(Float32, HIDDEN_DIM)
    W2 = Flux.glorot_uniform(rng, N_CLASSES, HIDDEN_DIM)
    b2 = zeros(Float32, N_CLASSES)
    backbone = Chain(Dense(W1, b1, relu), Dense(W2, b2), softmax)
    return backbone
end

function get_marker_gene_rules(gene_names, cell_type_names)
    gene_to_idx = Dict{String, Int}()
    for (i, g) in enumerate(gene_names)
        gene_to_idx[String(g)] = i
    end
    cell_type_to_idx = Dict{String, Int}()
    for (i, ct) in enumerate(cell_type_names)
        cell_type_to_idx[String(ct)] = i
    end
    marker_rules = [
        (["MS4A1"], "B", Float32[0.5]),
        (["IL7R", "CD3D"], "CD4_T", Float32[0.5, 0.5]),
        (["CD8A", "CD3D"], "CD8_T", Float32[0.5, 0.5]),
        (["NKG7", "GNLY"], "NK", Float32[1.0, 0.5]),
        (["CD14"], "CD14_Mono", Float32[0.5]),
        (["FCGR3A", "LYZ"], "FCGR3A_Mono", Float32[1.0, 0.5]),
        (["CST3"], "DC", Float32[0.5]),
        (["PPBP"], "Megakaryocyte", Float32[1.0]),
    ]
    rules = Tuple{Vector{String}, String, Vector{Float32}, Vector{Int}, Int}[]
    for (genes, ct, thresholds) in marker_rules
        gene_indices = Int[get(gene_to_idx, g, 0) for g in genes]
        ct_idx = get(cell_type_to_idx, ct, 0)
        if all(gi > 0 for gi in gene_indices) && ct_idx > 0
            push!(rules, (genes, ct, thresholds, gene_indices, ct_idx))
        end
    end
    return rules
end

function get_active_rules(sample_x, rules)
    active = Int[]
    for (r_idx, (genes, ct, thresholds, gene_indices, ct_idx)) in enumerate(rules)
        if all(sample_x[gi] > th for (gi, th) in zip(gene_indices, thresholds))
            push!(active, r_idx)
        end
    end
    return active
end

function project_sample(y_hat_i, active_rules, rules, rho)
    if isempty(active_rules)
        return Float32.(y_hat_i)
    end
    dm = DisjunctiveModel(N_CLASSES)
    set_bounds!(dm, lower=zeros(N_CLASSES), upper=ones(N_CLASSES))
    add_linear_constraint!(dm, ones(N_CLASSES), :(==), 1.0)
    for r_idx in active_rules
        genes, ct, thresholds, gene_indices, ct_idx = rules[r_idx]
        coeffs = zeros(N_CLASSES)
        coeffs[ct_idx] = 1.0
        add_linear_constraint!(dm, coeffs, :>=, rho)
    end
    result = try
        project(dm, Float64.(y_hat_i);
            formulation=:dnf,
            y_regularization=Y_REG,
            ycopy_regularization=YCOPY_REG,
            gamma_regularization=GAMMA_REG,
            anchor_regularization=ANCHOR_REG,
        )
    catch e
        nothing
    end
    if result !== nothing && result.status == MOI.OPTIMAL
        return Float32.(result.y)
    else
        return Float32.(y_hat_i)
    end
end

function project_batch(Y_hat, X_batch, rules, rho)
    Y_proj = similar(Y_hat)
    n = size(Y_hat, 2)
    for i in 1:n
        active = get_active_rules(X_batch[:, i], rules)
        Y_proj[:, i] = project_sample(Y_hat[:, i], active, rules, rho)
    end
    return Y_proj
end

function train_base_model!(backbone, X_train, Y_train, n_epochs; lr=LEARNING_RATE)
    opt_state = Flux.setup(AdamW(lr), backbone)
    for epoch in 1:n_epochs
        total_loss = 0.0f0
        n = size(X_train, 2)
        for i in 1:n
            x_i = X_train[:, i:i]
            y_i = Y_train[:, i:i]
            loss_val, grads = Flux.withgradient(backbone) do model
                Flux.crossentropy(model(x_i), y_i)
            end
            Flux.update!(opt_state, backbone, grads[1])
            total_loss += loss_val
        end
        if epoch % 500 == 0 || epoch == 1 || epoch == n_epochs
            @printf("  Epoch %d/%d: loss = %.6f\n", epoch, n_epochs, total_loss / n)
        end
    end
end

function compute_metrics(Y_pred, y_true)
    y_pred_idx = mapslices(argmax, Y_pred, dims=1)[:]
    y_true_1 = y_true .+ 1
    nc = size(Y_pred, 1)
    acc = mean(y_pred_idx .== y_true_1)
    macro_prec = 0.0
    macro_rec = 0.0
    for c in 1:nc
        tp = sum((y_pred_idx .== c) .& (y_true_1 .== c))
        fp = sum((y_pred_idx .== c) .& (y_true_1 .!= c))
        fn = sum((y_pred_idx .!= c) .& (y_true_1 .== c))
        macro_prec += tp / max(tp + fp, 1)
        macro_rec += tp / max(tp + fn, 1)
    end
    macro_prec /= nc
    macro_rec /= nc
    macro_f1 = 2 * macro_prec * macro_rec / max(macro_prec + macro_rec, 1e-10)
    return (accuracy=acc, macro_f1=macro_f1, macro_prec=macro_prec, macro_rec=macro_rec)
end

function compute_csat(Y_pred, X, rules, rho)
    n = size(X, 2)
    total = 0
    sat = 0
    for i in 1:n
        active = get_active_rules(X[:, i], rules)
        if !isempty(active)
            all_ok = true
            for r_idx in active
                genes, ct, thresholds, gene_indices, ct_idx = rules[r_idx]
                if Y_pred[ct_idx, i] < rho
                    all_ok = false
                    break
                end
            end
            dm = DisjunctiveModel(N_CLASSES)
            set_bounds!(dm, lower=zeros(N_CLASSES), upper=ones(N_CLASSES))
            add_linear_constraint!(dm, ones(N_CLASSES), :(==), 1.0)
            has_constraint = false
            for r_idx in active
                genes, ct, thresholds, gene_indices, ct_idx = rules[r_idx]
                coeffs = zeros(N_CLASSES)
                coeffs[ct_idx] = 1.0
                add_linear_constraint!(dm, coeffs, :>=, rho)
                has_constraint = true
            end
            if has_constraint
                result = try
                    project(dm, Float64.(Y_pred[:, i]);
                        formulation=:dnf,
                        y_regularization=Y_REG,
                        ycopy_regularization=YCOPY_REG,
                        gamma_regularization=GAMMA_REG,
                        anchor_regularization=ANCHOR_REG,
                    )
                catch e
                    nothing
                end
                if result !== nothing && result.status == MOI.OPTIMAL
                    total += 1
                    if all_ok
                        sat += 1
                    end
                end
            end
        end
    end
    return total > 0 ? sat / total : 0.0
end

function main()
    println("="^60)
    println("RHO SWEEP (1 run, seed=1, multi-gene rules, HIDDEN_DIM=32)")
    println("="^60)

    X, y, Y, gene_names, cell_type_names = load_pbmc3k_data()
    rules = get_marker_gene_rules(gene_names, cell_type_names)
    X_train, Y_train, y_train, X_test, Y_test, y_test, _, _ = split_data(X, Y, y; seed=SPLIT_SEED)
    n_train = size(X_train, 2)
    n_test = size(X_test, 2)
    println("Train: $n_train, Test: $n_test")

    rng = MersenneTwister(1)
    backbone = build_backbone(rng)
    println("Training base model ($BASE_EPOCHS epochs)...")
    train_base_model!(backbone, X_train, Y_train, BASE_EPOCHS)

    Y_pred_raw = backbone(X_test)
    metrics_raw = compute_metrics(Y_pred_raw, y_test)
    println("\nBase model (no projection):")
    @printf("  Accuracy=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f\n",
            metrics_raw.accuracy, metrics_raw.macro_f1, metrics_raw.macro_prec, metrics_raw.macro_rec)

    results = []
    for rho in RHO_VALUES
        Y_pred_dnf = project_batch(Y_pred_raw, X_test, rules, rho)
        m = compute_metrics(Y_pred_dnf, y_test)
        csat = compute_csat(Y_pred_dnf, X_test, rules, rho)
        push!(results, (rho=rho, metrics=m, csat=csat))
    end

    println("\n" * "="^60)
    println("RHO SWEEP RESULTS")
    println("="^60)
    println(rpad("RHO", 8), rpad("Accuracy", 12), rpad("Macro_F1", 12),
            rpad("Macro_Prec", 12), rpad("Macro_Rec", 12), "CSAT")
    println("-"^68)

    best_acc_rho = 0.0
    best_acc = 0.0
    best_f1_rho = 0.0
    best_f1 = 0.0

    for (rho, m, csat) in results
        @printf("%-8.2f%-12.4f%-12.4f%-12.4f%-12.4f%.4f\n",
                rho, m.accuracy, m.macro_f1, m.macro_prec, m.macro_rec, csat)
        if m.accuracy > best_acc
            best_acc = m.accuracy
            best_acc_rho = rho
        end
        if m.macro_f1 > best_f1
            best_f1 = m.macro_f1
            best_f1_rho = rho
        end
    end

    println("\nBest Accuracy: $(round(best_acc, digits=4)) at rho=$(best_acc_rho)")
    println("Best Macro F1:  $(round(best_f1, digits=4)) at rho=$(best_f1_rho)")
end

main()
