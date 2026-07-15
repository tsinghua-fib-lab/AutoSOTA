#!/usr/bin/env julia --project=.
# PBMC3k scRNA-seq classification with DNF projection layer
# Based on DisjunctiveNet paper (Section 4.2, Appendix E.5)
#
# Settings:
#   - Dataset: PBMC3k, 1838 features, 8 classes
#   - MLP: 1838 -> 32 -> 8, Glorot uniform init (HIDDEN_DIM=32)
#   - AdamW, lr=3e-3, batch_size=1
#   - Base training: 500 epochs (no projection)
#   - DNF projection applied at inference time
#   - Multi-gene rules with AND logic for CD4_T, CD8_T, NK, FCGR3A_Mono
#   - Test fraction: 0.1, split seed: 42
#   - Training size: 12 (0.005 fraction)
#   - 3 random seeds

using Flux
using Statistics
using Random
using LinearAlgebra
using DisjunctiveNet
using NPZ
import MathOptInterface as MOI
using Printf

# ===== Configuration =====
const N_FEATURES = 1838
const N_CLASSES = 8
const HIDDEN_DIM = 32
const N_RUNS = 3
const TRAIN_FRACTION = 0.005
const TEST_FRACTION = 0.1
const SPLIT_SEED = 42
const BASE_EPOCHS = 500
const LEARNING_RATE = 3e-3

const Y_REG = 1e-4
const YCOPY_REG = 1e-4
const GAMMA_REG = 1e-4
const ANCHOR_REG = 1e-3
const RHO = 0.5

# ===== Load data =====
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

    X = Float32.(X')  # (cells, features) -> (features, cells)
    Y = Flux.onehotbatch(y .+ 1, 1:N_CLASSES)
    return X, y, Y, gene_names, cell_type_names
end

# ===== Data splitting =====
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

# ===== Build backbone =====
function build_backbone(rng)
    W1 = Flux.glorot_uniform(rng, HIDDEN_DIM, N_FEATURES)
    b1 = zeros(Float32, HIDDEN_DIM)
    W2 = Flux.glorot_uniform(rng, N_CLASSES, HIDDEN_DIM)
    b2 = zeros(Float32, N_CLASSES)

    backbone = Chain(
        Dense(W1, b1, relu),
        Dense(W2, b2),
        softmax,
    )
    return backbone
end

# ===== Marker gene rules =====
# Rule format: (genes::Vector{String}, ct::String, thresholds::Vector{Float32}, gene_indices::Vector{Int}, ct_idx::Int)
# Multi-gene rules use AND logic (all genes must exceed their thresholds)
# Single-gene rules use single-element vectors
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
        missing_genes = [genes[i] for i in eachindex(genes) if gene_indices[i] == 0]
        if all(gi > 0 for gi in gene_indices) && ct_idx > 0
            push!(rules, (genes, ct, thresholds, gene_indices, ct_idx))
        else
            if !isempty(missing_genes)
                println("WARNING: Could not find gene(s) $(join(missing_genes, ", ")) for cell_type=$ct")
            elseif ct_idx == 0
                println("WARNING: Could not find cell_type=$ct")
            end
        end
    end
    return rules
end

# ===== Active rules (AND logic for multi-gene conjunctions) =====
function get_active_rules(sample_x, rules)
    active = Int[]
    for (r_idx, (genes, ct, thresholds, gene_indices, ct_idx)) in enumerate(rules)
        # All genes in the conjunction must exceed their thresholds (AND logic)
        if all(sample_x[gi] > th for (gi, th) in zip(gene_indices, thresholds))
            push!(active, r_idx)
        end
    end
    return active
end

# ===== Build + Project (non-gradient, for inference only) =====
function project_sample(y_hat_i, active_rules, rules)
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
        add_linear_constraint!(dm, coeffs, :>=, RHO)
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
        return Float32.(y_hat_i)  # bypass on infeasibility
    end
end

# ===== Batch projection =====
function project_batch(Y_hat, X_batch, rules)
    Y_proj = similar(Y_hat)
    n = size(Y_hat, 2)
    for i in 1:n
        active = get_active_rules(X_batch[:, i], rules)
        Y_proj[:, i] = project_sample(Y_hat[:, i], active, rules)
    end
    return Y_proj
end

# ===== Train base model =====
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
        if epoch % 100 == 0 || epoch == 1 || epoch == n_epochs
            @printf("  Epoch %d/%d: loss = %.6f\n", epoch, n_epochs, total_loss / n)
        end
    end
end

# ===== Metrics =====
function compute_metrics(Y_pred, y_true)
    y_pred_idx = mapslices(argmax, Y_pred, dims=1)[:]
    y_true_1 = y_true .+ 1  # 1-based
    N = length(y_true)
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

# ===== CSAT =====
function compute_csat(Y_pred, X, rules)
    n = size(X, 2)
    total = 0
    sat = 0
    for i in 1:n
        active = get_active_rules(X[:, i], rules)
        if !isempty(active)
            all_ok = true
            for r_idx in active
                genes, ct, thresholds, gene_indices, ct_idx = rules[r_idx]
                if Y_pred[ct_idx, i] < RHO
                    all_ok = false
                    break
                end
            end
            # Check feasibility
            dm = DisjunctiveModel(N_CLASSES)
            set_bounds!(dm, lower=zeros(N_CLASSES), upper=ones(N_CLASSES))
            add_linear_constraint!(dm, ones(N_CLASSES), :(==), 1.0)
            has_constraint = false
            for r_idx in active
                genes, ct, thresholds, gene_indices, ct_idx = rules[r_idx]
                coeffs = zeros(N_CLASSES)
                coeffs[ct_idx] = 1.0
                add_linear_constraint!(dm, coeffs, :>=, RHO)
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

# ===== Active rule statistics =====
function rule_stats(X, rules, label)
    n = size(X, 2)
    counts = zeros(Int, length(rules))
    active_any = 0
    for i in 1:n
        active = get_active_rules(X[:, i], rules)
        if !isempty(active)
            active_any += 1
        end
        for r_idx in active
            counts[r_idx] += 1
        end
    end
    println("  $label: $active_any/$n samples have >=1 active rule")
    for (r_idx, (genes, ct, thresholds, _, _)) in enumerate(rules)
        pct = 100 * counts[r_idx] / max(n, 1)
        gene_str = join(["$g > $t" for (g, t) in zip(genes, thresholds)], " AND ")
        println("    Rule $r_idx ($gene_str -> $ct): $(counts[r_idx]) samples ($(round(pct, digits=1))%)")
    end
end

# ===== Main =====
function main()
    println("="^60)
    println("PBMC3k DNF Reproduction Experiment")
    println("="^60)

    X, y, Y, gene_names, cell_type_names = load_pbmc3k_data()
    println("Data: $(size(X, 2)) cells, $(size(X, 1)) features, $(length(unique(y))) classes")
    println("Cell types: $(cell_type_names)")

    rules = get_marker_gene_rules(gene_names, cell_type_names)
    println("Marker gene rules: $(length(rules))")
    for (genes, ct, thresholds, _, _) in rules
        gene_str = join(["$g > $t" for (g, t) in zip(genes, thresholds)], " AND ")
        println("  $gene_str => y[$ct] >= $RHO")
    end

    all_base = []
    all_dnf = []

    for run_seed in 1:N_RUNS
        println("\n" * "-"^40)
        println("Run $run_seed/$N_RUNS (training seed=$run_seed)")
        println("-"^40)

        rng = MersenneTwister(run_seed)
        Random.seed!(run_seed)

        X_train, Y_train, y_train, X_test, Y_test, y_test, _, _ =
            split_data(X, Y, y; seed=SPLIT_SEED)

        n_train = size(X_train, 2)
        n_test = size(X_test, 2)
        println("Train: $n_train, Test: $n_test")

        rule_stats(X_train, rules, "Train")
        rule_stats(X_test, rules, "Test")

        backbone = build_backbone(rng)
        println("Backbone: MLP($N_FEATURES -> $HIDDEN_DIM -> $N_CLASSES) + softmax")

        println("\nTraining base model ($BASE_EPOCHS epochs)...")
        train_base_model!(backbone, X_train, Y_train, BASE_EPOCHS)

        println("\nEvaluating on test set...")
        Y_pred_raw = backbone(X_test)

        # Apply DNF projection at inference time
        Y_pred_dnf = project_batch(Y_pred_raw, X_test, rules)

        # Metrics
        metrics_raw = compute_metrics(Y_pred_raw, y_test)
        metrics_dnf = compute_metrics(Y_pred_dnf, y_test)
        csat_raw = compute_csat(Y_pred_raw, X_test, rules)
        csat_dnf = compute_csat(Y_pred_dnf, X_test, rules)

        println("\n  Base model:")
        println("    Accuracy:   $(round(metrics_raw.accuracy, digits=4))")
        println("    Macro F1:   $(round(metrics_raw.macro_f1, digits=4))")
        println("    Macro Prec: $(round(metrics_raw.macro_prec, digits=4))")
        println("    Macro Rec:  $(round(metrics_raw.macro_rec, digits=4))")
        println("    CSAT:       $(round(csat_raw, digits=4))")
        println("  DNF model (inference projection):")
        println("    Accuracy:   $(round(metrics_dnf.accuracy, digits=4))")
        println("    Macro F1:   $(round(metrics_dnf.macro_f1, digits=4))")
        println("    Macro Prec: $(round(metrics_dnf.macro_prec, digits=4))")
        println("    Macro Rec:  $(round(metrics_dnf.macro_rec, digits=4))")
        println("    CSAT:       $(round(csat_dnf, digits=4))")

        push!(all_base, (metrics_raw, csat_raw))
        push!(all_dnf, (metrics_dnf, csat_dnf))
    end

    # Aggregate
    println("\n" * "="^60)
    println("AGGREGATE RESULTS (n=$N_RUNS runs)")
    println("="^60)
    for (label, results) in [("Base", all_base), ("DNF", all_dnf)]
        for metric_name in [:accuracy, :macro_f1, :macro_prec, :macro_rec]
            vals = [getfield(r[1], metric_name) for r in results]
            μ = mean(vals)
            σ = std(vals)
            println("$label $(String(metric_name)): $(round(μ, digits=4)) ± $(round(σ, digits=4))")
        end
        csat_vals = [r[2] for r in results]
        println("$label CSAT: $(round(mean(csat_vals), digits=4)) ± $(round(std(csat_vals), digits=4))")
    end

    println("\nPaper reference (DNF, n=12):")
    println("  Acc=0.342±0.197  F1=0.326±0.096  Prec=0.450±0.091  Rec=0.462±0.100  CSAT=0.908±0.083")

    return all_base, all_dnf
end

main()
