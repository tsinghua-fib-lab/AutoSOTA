#include <algorithm>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>

using std::cout;

struct RIDResult {
    std::vector<double> mean_sub_mr;
    std::vector<std::vector<double>> cdf_x;
    std::vector<std::vector<double>> cdf_p;

    // optionally (when return_joint_samples=true return one vector per tree per bootstrap of feature importances
    std::vector<std::vector<double>> feature_importance_weight_samples;
};

static inline uint64_t popcnt64_u(uint64_t x) {
#if defined(_MSC_VER)
    return (uint64_t)__popcnt64(x);
#else
    return (uint64_t)__builtin_popcountll(x);
#endif
}

// y_bits[c] has bit i = 1 iff y[i] == c
static inline std::vector<Packed> build_yc_packed(
    const std::vector<int>& y,
    int n_classes,
    int n_words,
    uint64_t tail_mask
) {
    std::vector<Packed> y_bits;
    y_bits.reserve((size_t)n_classes);
    for (int c = 0; c < n_classes; ++c) {
        y_bits.emplace_back((size_t)n_words);
    }

    for (int i = 0; i < (int)y.size(); ++i) {
        const int c = y[i];
        y_bits[(size_t)c].w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
    }

    if (n_words > 0) {
        for (int c = 0; c < n_classes; ++c) {
            y_bits[(size_t)c].w[(size_t)(n_words - 1)] &= tail_mask;
        }
    }

    return y_bits;
}

static inline int count_correct_packed_multi(
    const PackedPredMulti& pred,
    const std::vector<Packed>& y_bits,
    int n_words,
    uint64_t tail_mask
) {
    const int C = (int)y_bits.size();
    uint64_t correct = 0;

    for (int c = 0; c < C; ++c) {
        const auto& pw = pred.by_class[(size_t)c].w;
        const auto& yw = y_bits[(size_t)c].w;

        for (int w = 0; w < n_words; ++w) {
            uint64_t bits = pw[(size_t)w] & yw[(size_t)w];
            if (w == n_words - 1) bits &= tail_mask;
            correct += popcnt64_u(bits);
        }
    }

    return (int)correct;
}

static inline void bootstrap_indices(
    int n,
    std::mt19937_64& rng,
    std::vector<int>& idx
) {
    std::uniform_int_distribution<int> unif(0, n - 1);
    idx.resize(n);

    for (int i = 0; i < n; ++i) {
        idx[i] = unif(rng);
    }
}

static inline void make_bootstrap_dataset(
    const std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& y,
    const std::vector<int>& idx,
    std::vector<std::vector<uint8_t>>& Xb,
    std::vector<int>& yb
) {
    const int n = (int)idx.size();
    const int d = (int)X[0].size();

    Xb.assign(n, std::vector<uint8_t>(d));
    yb.assign(n, 0);

    for (int i = 0; i < n; ++i) {
        const int s = idx[i];
        Xb[i] = X[s];
        yb[i] = y[s];
    }
}

static inline void make_bootstrap_vector_int(
    const std::vector<int>& v,
    const std::vector<int>& idx,
    std::vector<int>& vb
) {
    const int n = (int)idx.size();
    vb.assign(n, 0);

    for (int i = 0; i < n; ++i) {
        vb[i] = v[(size_t)idx[(size_t)i]];
    }
}

static inline void rowmajor_to_colmajor_bool(
    const std::vector<std::vector<uint8_t>>& X_row,
    std::vector<std::vector<bool>>& X_col
) {
    const int n = (int)X_row.size();
    const int d = (int)X_row[0].size();

    X_col.assign(d, std::vector<bool>(n, false));

    for (int i = 0; i < n; ++i) {
        const auto& row = X_row[i];

        for (int j = 0; j < d; ++j) {
            X_col[j][i] = (row[j] != 0);
        }
    }
}

static inline void make_permutation(
    int n,
    std::mt19937_64& rng,
    std::vector<int>& perm
) {
    perm.resize(n);

    for (int i = 0; i < n; ++i) {
        perm[i] = i;
    }

    std::shuffle(perm.begin(), perm.end(), rng);
}

// scramble a single original feature represented by one or more binary columns
static inline void scramble_block_inplace(
    std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& cols,
    const std::vector<int>& perm,
    std::vector<std::vector<uint8_t>>& saved_cols
) {
    const int n = (int)X.size();

    saved_cols.assign(cols.size(), std::vector<uint8_t>(n));

    // save originals
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        const int col = cols[ci];

        for (int i = 0; i < n; ++i) {
            saved_cols[ci][i] = X[i][col];
        }
    }

    // apply same permutation to every column in the block
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        const int col = cols[ci];

        for (int i = 0; i < n; ++i) {
            X[i][col] = saved_cols[ci][perm[i]];
        }
    }
}

static inline void restore_block_inplace(
    std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& cols,
    const std::vector<std::vector<uint8_t>>& saved_cols
) {
    const int n = (int)X.size();

    for (size_t ci = 0; ci < cols.size(); ++ci) {
        const int col = cols[ci];

        for (int i = 0; i < n; ++i) {
            X[i][col] = saved_cols[ci][i];
        }
    }
}

static inline int count_correct(
    const std::vector<uint8_t>& preds,
    const std::vector<int>& y
) {
    const int n = (int)y.size();
    int c = 0;

    for (int i = 0; i < n; ++i) {
        c += (preds[i] == (uint8_t)y[i]);
    }

    return c;
}

static inline int rid_eval_objective_from_mis_def(
    int misclassifications,
    int deferrals,
    bool use_deferral,
    double eta_defer
) {
    if (!use_deferral) {
        return misclassifications;
    }

    return misclassifications + (int)llround(eta_defer * (double)deferrals);
}

RIDResult compute_rid_subtractive_mr_bootstrap(
    const std::vector<std::vector<uint8_t>>& X_row_major,
    const std::vector<int>& y,
    int n_bootstraps,
    double lambda,
    int depth_budget,
    double rashomon_mult,
    int lookahead_k,
    uint64_t seed,
    bool memory_efficient,
    const std::vector<std::vector<int>>& binning_map_vars = {},
    bool use_deferral = false,
    double eta_defer = 0.0,
    const std::vector<int>& bb_pred = {},
    bool return_joint_samples = false
) {
    (void)memory_efficient;

    const int n_full = (int)X_row_major.size();

    if (n_full == 0) {
        throw std::runtime_error("compute_rid_subtractive_mr_bootstrap: X is empty.");
    }

    const int d = (int)X_row_major[0].size();

    if ((int)y.size() != n_full) {
        throw std::runtime_error(
            "compute_rid_subtractive_mr_bootstrap: y has different number of rows than X."
        );
    }

    if (use_deferral) {
        if ((int)bb_pred.size() != n_full) {
            throw std::runtime_error(
                "compute_rid_subtractive_mr_bootstrap: use_deferral=true requires "
                "bb_pred with the same number of rows as X/y."
            );
        }

        if (!std::isfinite(eta_defer) || eta_defer < 0.0) {
            throw std::runtime_error(
                "compute_rid_subtractive_mr_bootstrap: eta_defer must be finite and nonnegative."
            );
        }
    }

    // build var -> binary-column mapping.
    // if no binning map is provided, assume each binary column is its own variable.
    std::vector<std::vector<int>> var_cols;

    if (!binning_map_vars.empty()) {
        var_cols = binning_map_vars;
    } else {
        var_cols.resize((size_t)d);

        for (int j = 0; j < d; ++j) {
            var_cols[(size_t)j] = std::vector<int>{j};
        }
    }

    const int V = (int)var_cols.size();

    std::mt19937_64 rng(seed);

    RIDResult out;
    out.mean_sub_mr.assign(V, 0.0);
    out.cdf_x.assign(V, {});
    out.cdf_p.assign(V, {});

    // old non-deferral interpretation:
    // delta = scr_mis - orig_mis = correct_orig - correct_scr.
    //
    // deferral interpretation:
    // delta = scr_eval_objective - orig_eval_objective,
    // where eval objective ignores leaf count because it is constant under permutation:
    // eval_objective = misclassifications + round(eta_defer * num_deferrals).
    std::vector<std::unordered_map<int, double>> mass_by_delta(V);

    for (int b = 0; b < n_bootstraps; ++b) {
        std::vector<int> idx;
        bootstrap_indices(n_full, rng, idx);

        std::vector<std::vector<uint8_t>> Xb;
        std::vector<int> yb;
        make_bootstrap_dataset(X_row_major, y, idx, Xb, yb);

        std::vector<int> bb_pred_b;
        if (use_deferral) {
            make_bootstrap_vector_int(bb_pred, idx, bb_pred_b);
        }

        const int n = (int)Xb.size();

        // row-major -> col-major bool for training.
        std::vector<std::vector<bool>> Xcol;
        rowmajor_to_colmajor_bool(Xb, Xcol);

        PRAXIS model;
        model.fit(
            Xcol,
            yb,
            lambda,
            depth_budget,
            rashomon_mult,
            lookahead_k,
            -1,                 // root_budget
            true,               // use_multipass_flag
            false,              // rule_list_mode_flag
            0,                  // proxy_style_in
            false,              // majority_leaf_only_flag
            false,              // cache_cheap_subproblems_flag
            true,               // proxy_caching_flag
            0,                  // num_proxy_features_in
            true,               // rashomon_mode
            false,              // stronger_rollout_flag
            use_deferral,       // use_deferral_flag
            eta_defer,          // eta_defer_in
            bb_pred_b           // bb_pred
        );

        const uint64_t T64 = model.result ? model.result->count_trees() : 0ULL;
        const int T = (int)T64;

        if (T == 0) {
            continue;
        }

        cout << "Finished RID bootstrap: "
             << (b + 1)
             << " / "
             << n_bootstraps
             << " with "
             << T
             << " trees\n";

        // pre-sample permutations for each original variable.
        // one scramble per variable per bootstrap.
        std::vector<std::vector<int>> perms((size_t)V);

        for (int v = 0; v < V; ++v) {
            make_permutation(n, rng, perms[(size_t)v]);
        }

        // reuse buffer for column/block scrambling.
        std::vector<std::vector<uint8_t>> saved_cols;

        const int budget_override = (int)llround(
            (1.0 + rashomon_mult) * (double)model.result->min_objective
        );

        std::vector<int> orig_mis;
        if (use_deferral) {
            orig_mis = model.get_all_misclassifications_packed_trie(
                Xb,
                yb,
                budget_override,
                bb_pred_b
            );
        } else {
            orig_mis = model.get_all_misclassifications_packed_trie(
                Xb,
                yb,
                budget_override
            );
        }

        std::vector<int> orig_def;
        if (use_deferral) {
            orig_def = model.get_all_deferrals_packed_trie(
                Xb,
                budget_override
            );

            if (orig_def.size() != orig_mis.size()) {
                throw std::runtime_error(
                    "RID deferral: orig_def and orig_mis have different lengths."
                );
            }
        }

        const uint64_t Tvec = (uint64_t)orig_mis.size();

        if (Tvec == 0) {
            continue;
        }

        // weight per tree per bootstrap.
        const double wt_tree = 1.0 / ((double)n_bootstraps * (double)Tvec);

        // if requested, allocate one dense feature-importance row per tree in this bootstrap.
        // the final column stores the tree weight
        std::size_t sample_offset = 0;
        if (return_joint_samples) {
            sample_offset = out.feature_importance_weight_samples.size();

            out.feature_importance_weight_samples.resize(
                sample_offset + (std::size_t)Tvec,
                std::vector<double>((std::size_t)V + 1, 0.0)
            );

            for (uint64_t t = 0; t < Tvec; ++t) {
                out.feature_importance_weight_samples[
                    sample_offset + (std::size_t)t
                ][(std::size_t)V] = wt_tree;
            }
        }

        // possible future optimization:
        // convert the bootstrap to column-major once, scramble columns in column-major,
        // and add a packed-trie extraction method that accepts column-major eval data.
        for (int v = 0; v < V; ++v) {
            const std::vector<int>& cols = var_cols[(size_t)v];

            scramble_block_inplace(
                Xb,
                cols,
                perms[(size_t)v],
                saved_cols
            );

            std::vector<int> scr_mis;
            if (use_deferral) {
                scr_mis = model.get_all_misclassifications_packed_trie(
                    Xb,
                    yb,
                    budget_override,
                    bb_pred_b
                );
            } else {
                scr_mis = model.get_all_misclassifications_packed_trie(
                    Xb,
                    yb,
                    budget_override
                );
            }

            std::vector<int> scr_def;
            if (use_deferral) {
                scr_def = model.get_all_deferrals_packed_trie(
                    Xb,
                    budget_override
                );

                if (scr_def.size() != scr_mis.size()) {
                    throw std::runtime_error(
                        "RID deferral: scr_def and scr_mis have different lengths."
                    );
                }
            }

            const uint64_t Tuse = Tvec;

            if ((uint64_t)scr_mis.size() != Tuse) {
                throw std::runtime_error(
                    "RID: scrambled and original misclassification vectors have different lengths."
                );
            }

            for (uint64_t t = 0; t < Tuse; ++t) {
                const int orig_def_t = use_deferral ? orig_def[(size_t)t] : 0;
                const int scr_def_t  = use_deferral ? scr_def[(size_t)t]  : 0;

                const int orig_obj = rid_eval_objective_from_mis_def(
                    orig_mis[(size_t)t],
                    orig_def_t,
                    use_deferral,
                    eta_defer
                );

                const int scr_obj = rid_eval_objective_from_mis_def(
                    scr_mis[(size_t)t],
                    scr_def_t,
                    use_deferral,
                    eta_defer
                );

                // non-deferral case:
                // delta_obj = scr_mis - orig_mis
                // = correct_orig - correct_scr.
                //
                // deferral case:
                // delta_obj = scrambled eval objective - original eval objective.
                // larger positive means the feature matters more.
                const int delta_obj = scr_obj - orig_obj;
                const double importance = (double)delta_obj / (double)n;

                out.mean_sub_mr[v] += wt_tree * importance;
                mass_by_delta[(size_t)v][delta_obj] += wt_tree;

                if (return_joint_samples) {
                    out.feature_importance_weight_samples[
                        sample_offset + (std::size_t)t
                    ][(std::size_t)v] = importance;
                }
            }

            restore_block_inplace(Xb, cols, saved_cols);
        }
    }

    // build weighted CDF for each feature from the mass map.
    for (int v = 0; v < V; ++v) {
        std::vector<std::pair<int, double>> items;
        items.reserve(mass_by_delta[(size_t)v].size());

        for (const auto& kv : mass_by_delta[(size_t)v]) {
            items.push_back(kv);
        }

        std::sort(
            items.begin(),
            items.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            }
        );

        out.cdf_x[(size_t)v].reserve(items.size());
        out.cdf_p[(size_t)v].reserve(items.size());

        double cum = 0.0;

        for (const auto& kv : items) {
            const int delta = kv.first;
            const double w = kv.second;

            cum += w;

            out.cdf_x[(size_t)v].push_back((double)delta / (double)n_full);
            out.cdf_p[(size_t)v].push_back(cum);
        }
    }

    return out;
}