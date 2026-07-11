#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cpp/praxis.cpp"
#include "cpp/rid.cpp"

namespace py = pybind11;

namespace {

static inline std::vector<std::vector<uint8_t>>
numpy_uint8_to_row_major(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X
) {
    py::buffer_info xinfo = X.request();

    if (xinfo.ndim != 2) {
        throw std::runtime_error("X must be 2D");
    }

    const int n_samples  = static_cast<int>(xinfo.shape[0]);
    const int n_features = static_cast<int>(xinfo.shape[1]);
    auto *x_ptr = static_cast<uint8_t*>(xinfo.ptr);

    std::vector<std::vector<uint8_t>> X_row_major(
        n_samples,
        std::vector<uint8_t>(n_features)
    );

    for (int i = 0; i < n_samples; ++i) {
        std::memcpy(
            X_row_major[(size_t)i].data(),
            x_ptr + (std::size_t)i * (std::size_t)n_features,
            (std::size_t)n_features * sizeof(uint8_t)
        );
    }

    return X_row_major;
}

static inline std::vector<std::vector<bool>>
numpy_uint8_to_col_major_bool(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X
) {
    py::buffer_info xinfo = X.request();

    if (xinfo.ndim != 2) {
        throw std::runtime_error("X must be 2D (n_samples x n_features)");
    }

    const int n_samples  = static_cast<int>(xinfo.shape[0]);
    const int n_features = static_cast<int>(xinfo.shape[1]);
    auto *x_ptr = static_cast<uint8_t*>(xinfo.ptr);

    std::vector<std::vector<bool>> X_col_major(
        n_features,
        std::vector<bool>(n_samples)
    );

    for (int f = 0; f < n_features; ++f) {
        for (int i = 0; i < n_samples; ++i) {
            const uint8_t v = x_ptr[(std::size_t)i * (std::size_t)n_features + (std::size_t)f];
            X_col_major[(size_t)f][(size_t)i] = (v != 0);
        }
    }

    return X_col_major;
}

static inline std::vector<int>
numpy_int_to_vector(
    py::array_t<int, py::array::c_style | py::array::forcecast> arr,
    const char *name
) {
    py::buffer_info info = arr.request();

    if (info.ndim != 1) {
        throw std::runtime_error(std::string(name) + " must be 1D");
    }

    auto *ptr = static_cast<int*>(info.ptr);
    const int n = static_cast<int>(info.shape[0]);

    return std::vector<int>(ptr, ptr + n);
}

static inline std::vector<int>
optional_numpy_int_to_vector(
    py::object obj,
    int expected_n,
    const char *name
) {
    std::vector<int> out;

    if (obj.is_none()) {
        return out;
    }

    py::array_t<int, py::array::c_style | py::array::forcecast> arr =
        obj.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();

    py::buffer_info info = arr.request();

    if (info.ndim != 1) {
        throw std::runtime_error(std::string(name) + " must be 1D");
    }

    if ((int)info.shape[0] != expected_n) {
        throw std::runtime_error(std::string(name) + " must match X rows");
    }

    auto *ptr = static_cast<int*>(info.ptr);
    out.assign(ptr, ptr + expected_n);

    return out;
}

static inline int numpy_n_rows(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X
) {
    py::buffer_info xinfo = X.request();

    if (xinfo.ndim != 2) {
        throw std::runtime_error("X must be 2D");
    }

    return static_cast<int>(xinfo.shape[0]);
}

static inline int numpy_n_features(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X
) {
    py::buffer_info xinfo = X.request();

    if (xinfo.ndim != 2) {
        throw std::runtime_error("X must be 2D");
    }

    return static_cast<int>(xinfo.shape[1]);
}

static inline py::array_t<uint8_t>
vector_uint8_to_numpy(const std::vector<uint8_t>& v) {
    py::array_t<uint8_t> out((py::ssize_t)v.size());

    auto info = out.request();
    auto *ptr = static_cast<uint8_t*>(info.ptr);

    if (!v.empty()) {
        std::memcpy(
            ptr,
            v.data(),
            v.size() * sizeof(uint8_t)
        );
    }

    return out;
}

static inline py::array_t<int>
vector_int_to_numpy(const std::vector<int>& v) {
    py::array_t<int> out((py::ssize_t)v.size());

    auto info = out.request();
    auto *ptr = static_cast<int*>(info.ptr);

    if (!v.empty()) {
        std::memcpy(
            ptr,
            v.data(),
            v.size() * sizeof(int)
        );
    }

    return out;
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "PRAXIS C++ core bindings";

    py::class_<ExportLeafNode>(m, "ExportLeafNode")
        .def_readonly("id", &ExportLeafNode::id)
        .def_readonly("parent_trie_id", &ExportLeafNode::parent_trie_id)
        .def_readonly("prediction", &ExportLeafNode::prediction)
        .def_readonly("loss", &ExportLeafNode::loss)
        .def_readonly("subproblem_size", &ExportLeafNode::subproblem_size);

    py::class_<ExportSplitNode>(m, "ExportSplitNode")
        .def_readonly("id", &ExportSplitNode::id)
        .def_readonly("parent_trie_id", &ExportSplitNode::parent_trie_id)
        .def_readonly("feature", &ExportSplitNode::feature)
        .def_readonly("left_trie_id", &ExportSplitNode::left_trie_id)
        .def_readonly("right_trie_id", &ExportSplitNode::right_trie_id)
        .def_readonly("min_objective", &ExportSplitNode::min_objective);

    py::class_<ExportTreeTrieNode>(m, "ExportTreeTrieNode")
        .def_readonly("id", &ExportTreeTrieNode::id)
        .def_readonly("budget", &ExportTreeTrieNode::budget)
        .def_readonly("min_objective", &ExportTreeTrieNode::min_objective)
        .def_readonly("subproblem_size", &ExportTreeTrieNode::subproblem_size)
        .def_readonly("leaf_ids", &ExportTreeTrieNode::leaf_ids)
        .def_readonly("split_ids", &ExportTreeTrieNode::split_ids);

    py::class_<ExportANDORGraph>(m, "ExportANDORGraph")
        .def_readonly("root_trie_id", &ExportANDORGraph::root_trie_id)
        .def_readonly("trie_nodes", &ExportANDORGraph::trie_nodes)
        .def_readonly("split_nodes", &ExportANDORGraph::split_nodes)
        .def_readonly("leaf_nodes", &ExportANDORGraph::leaf_nodes);

    py::class_<TrieNodeCountStats>(m, "TrieNodeCountStats")
        .def_readonly(
            "total_trie_nodes",
            &TrieNodeCountStats::total_trie_nodes
        )
        .def_readonly(
            "distinct_subproblem_depth",
            &TrieNodeCountStats::distinct_subproblem_depth
        )
        .def_readonly(
            "distinct_subproblem_depth_budget",
            &TrieNodeCountStats::distinct_subproblem_depth_budget
        );

    py::class_<PRAXIS>(m, "PRAXIS")
        .def(py::init<>())

        .def(
            "fit",
            [](PRAXIS &self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               py::array_t<int,     py::array::c_style | py::array::forcecast> y,
               double lambda_reg,
               int depth_budget,
               double rashomon_mult,
               double multiplicative_slack,
               std::string key_mode_str,
               bool trie_cache_enabled,
               int lookahead_k,
               int root_budget,
               bool use_multipass,
               bool rule_list_mode,
               int oracle_style,
               bool majority_leaf_only,
               bool cache_cheap_subproblems,
               int greedy_split_mode,
               bool proxy_caching,
               int num_proxy_features,
               bool rashomon_mode,
               bool stronger_rollout,
               bool use_deferral,
               double eta_defer,
               py::object bb_pred_obj
            ) {
                py::buffer_info xinfo = X.request();
                py::buffer_info yinfo = y.request();

                if (xinfo.ndim != 2) {
                    throw std::runtime_error("X must be 2D (n_samples x n_features)");
                }

                if (yinfo.ndim != 1) {
                    throw std::runtime_error("y must be 1D");
                }

                const int n_samples = static_cast<int>(xinfo.shape[0]);

                if ((int)yinfo.shape[0] != n_samples) {
                    throw std::runtime_error("y must match X rows");
                }

                std::vector<std::vector<bool>> X_col_major =
                    numpy_uint8_to_col_major_bool(X);

                std::vector<int> y_vec = numpy_int_to_vector(y, "y");

                std::vector<int> bb_pred_vec =
                    optional_numpy_int_to_vector(
                        bb_pred_obj,
                        n_samples,
                        "bb_pred"
                    );

                if (use_deferral && bb_pred_vec.empty()) {
                    throw std::runtime_error("use_deferral=true requires bb_pred");
                }

                PRAXIS::KeyMode km;

                if (key_mode_str == "exact" || key_mode_str == "bitvector") {
                    km = PRAXIS::KeyMode::EXACT;
                } else if (
                    key_mode_str == "literal" ||
                    key_mode_str == "lits" ||
                    key_mode_str == "lits_exact" ||
                    key_mode_str == "itemset"
                ) {
                    km = PRAXIS::KeyMode::LITS_EXACT;
                } else {
                    km = PRAXIS::KeyMode::HASH64;
                }

                self.set_key_mode(km);
                self.set_trie_cache_enabled(trie_cache_enabled);
                self.set_multiplicative_slack(multiplicative_slack);
                self.set_use_multipass(use_multipass);
                self.set_rule_list_mode(rule_list_mode);
                self.set_cache_cheap_subproblems(cache_cheap_subproblems);
                self.set_greedy_split_mode(greedy_split_mode);
                self.set_majority_leaf_only(majority_leaf_only);
                self.set_proxy_caching_enabled(proxy_caching);
                self.set_stronger_rollout(stronger_rollout);

                self.fit(
                    X_col_major,
                    y_vec,
                    lambda_reg,
                    depth_budget,
                    rashomon_mult,
                    lookahead_k,
                    root_budget,
                    use_multipass,
                    rule_list_mode,
                    oracle_style,
                    majority_leaf_only,
                    cache_cheap_subproblems,
                    proxy_caching,
                    num_proxy_features,
                    rashomon_mode,
                    stronger_rollout,
                    use_deferral,
                    eta_defer,
                    bb_pred_vec
                );
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("lambda_reg") = 0.01,
            py::arg("depth_budget") = 5,
            py::arg("rashomon_mult") = 0.01,
            py::arg("multiplicative_slack") = 0.0,
            py::arg("key_mode") = "hash",
            py::arg("trie_cache_enabled") = false,
            py::arg("lookahead_k") = 1,
            py::arg("root_budget") = -1,
            py::arg("use_multipass") = true,
            py::arg("rule_list_mode") = false,
            py::arg("oracle_style") = 0,
            py::arg("majority_leaf_only") = false,
            py::arg("cache_cheap_subproblems") = false,
            py::arg("greedy_split_mode") = 1,
            py::arg("proxy_caching") = true,
            py::arg("num_proxy_features") = 0,
            py::arg("rashomon_mode") = true,
            py::arg("stronger_rollout") = false,
            py::arg("use_deferral") = false,
            py::arg("eta_defer") = 0.0,
            py::arg("bb_pred") = py::none()
        )

        .def(
            "count_trees",
            [](PRAXIS &self) {
                return self.result ? self.result->count_trees() : 0ULL;
            }
        )

        .def(
            "get_min_objective",
            [](PRAXIS &self) {
                return self.result
                    ? self.result->min_objective
                    : std::numeric_limits<int>::max();
            }
        )

        .def(
            "get_root_histogram",
            [](PRAXIS &self) {
                if (!self.result) {
                    return std::vector<std::pair<int, std::uint64_t>>{};
                }

                self.result->ensure_hist_built();

                const auto &hist = self.result->hist;

                std::vector<std::pair<int, std::uint64_t>> out;
                out.reserve(hist.size());

                for (const auto &e : hist) {
                    out.emplace_back(e.obj, e.cnt);
                }

                return out;
            }
        )

        .def(
            "export_andor_graph",
            [](const PRAXIS &self) {
                return self.export_andor_graph();
            },
            "Export the compact AND/OR graph structure of the Rashomon trie."
        )

        .def(
            "count_reconstructed_trie_node_stats",
            [](const PRAXIS &self) {
                return self.count_reconstructed_trie_node_stats();
            },
            "Count reachable TreeTrieNode OR nodes, distinct reconstructed "
            "(subproblem, depth) pairs, and distinct reconstructed "
            "(subproblem, depth, budget) triples."
        )

        .def(
            "get_predictions",
            [](const PRAXIS &self,
               std::uint64_t tree_index,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               py::object bb_pred_obj,
               int defer_placeholder
            ) {
                const int n_samples = numpy_n_rows(X);

                std::vector<std::vector<uint8_t>> X_row_major =
                    numpy_uint8_to_row_major(X);

                std::vector<int> bb_pred_vec =
                    optional_numpy_int_to_vector(
                        bb_pred_obj,
                        n_samples,
                        "bb_pred"
                    );

                auto preds = self.get_predictions(
                    tree_index,
                    X_row_major,
                    bb_pred_vec,
                    defer_placeholder
                );

                return vector_uint8_to_numpy(preds);
            },
            py::arg("tree_index"),
            py::arg("X"),
            py::arg("bb_pred") = py::none(),
            py::arg("defer_placeholder") = 99
        )

        .def(
            "get_all_predictions",
            [](const PRAXIS &self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               bool stack,
               py::object bb_pred_obj,
               int defer_placeholder
            ) {
                const int n_samples = numpy_n_rows(X);

                std::vector<std::vector<uint8_t>> X_row_major =
                    numpy_uint8_to_row_major(X);

                std::vector<int> bb_pred_vec =
                    optional_numpy_int_to_vector(
                        bb_pred_obj,
                        n_samples,
                        "bb_pred"
                    );

                auto all_preds = self.get_all_predictions(
                    X_row_major,
                    bb_pred_vec,
                    defer_placeholder
                );

                const std::uint64_t total = all_preds.size();

                if (!stack) {
                    py::list lst;

                    for (std::uint64_t t = 0; t < total; ++t) {
                        lst.append(vector_uint8_to_numpy(all_preds[(size_t)t]));
                    }

                    return py::object(lst);
                }

                py::array_t<uint8_t> out(
                    {static_cast<py::ssize_t>(total),
                     static_cast<py::ssize_t>(n_samples)}
                );

                auto out_info = out.request();
                auto *out_ptr = static_cast<uint8_t*>(out_info.ptr);

                for (std::uint64_t t = 0; t < total; ++t) {
                    std::memcpy(
                        out_ptr + (std::size_t)t * (std::size_t)n_samples,
                        all_preds[(size_t)t].data(),
                        static_cast<std::size_t>(n_samples) * sizeof(uint8_t)
                    );
                }

                return py::object(out);
            },
            py::arg("X"),
            py::arg("stack") = false,
            py::arg("bb_pred") = py::none(),
            py::arg("defer_placeholder") = 99
        )

        .def(
            "get_tree_objective",
            [](const PRAXIS &self, std::uint64_t tree_index) {
                auto obj_pair = self.get_ith_tree_objective(tree_index);

                // obj_pair.first  = unnormalized objective (int)
                // obj_pair.second = normalized objective (double)
                return py::make_tuple(obj_pair.first, obj_pair.second);
            },
            py::arg("tree_index")
        )

        .def(
            "get_tree_paths",
            [](const PRAXIS &self, std::uint64_t tree_index) {
                auto result = self.get_tree_paths(tree_index);

                const auto &paths = result.first;
                const auto &preds = result.second;

                py::list py_paths;

                for (const auto &p : paths) {
                    py::list py_path;

                    for (int v : p) {
                        py_path.append(v);
                    }

                    py_paths.append(py_path);
                }

                py::array_t<int> py_preds((py::ssize_t)preds.size());
                auto info = py_preds.request();
                auto *ptr = static_cast<int*>(info.ptr);

                for (std::size_t i = 0; i < preds.size(); ++i) {
                    ptr[i] = preds[i];
                }

                return py::make_tuple(py_paths, py_preds);
            },
            py::arg("tree_index")
        )

        .def(
            "root_lickety_objective_lookahead1",
            [](PRAXIS &self, int depth_budget) {
                return self.root_lickety_objective_lookahead1(depth_budget);
            },
            py::arg("depth_budget")
        )

        .def(
            "get_all_misclassifications_packed_trie",
            [](const PRAXIS &self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               py::array_t<int,     py::array::c_style | py::array::forcecast> y,
               int budget_override,
               py::object bb_pred_obj
            ) {
                const int n_samples = numpy_n_rows(X);

                py::buffer_info yinfo = y.request();
                if (yinfo.ndim != 1) {
                    throw std::runtime_error("y must be 1D");
                }
                if ((int)yinfo.shape[0] != n_samples) {
                    throw std::runtime_error("y must match X rows");
                }

                std::vector<std::vector<uint8_t>> X_row_major =
                    numpy_uint8_to_row_major(X);

                std::vector<int> y_vec = numpy_int_to_vector(y, "y");

                std::vector<int> bb_pred_vec =
                    optional_numpy_int_to_vector(
                        bb_pred_obj,
                        n_samples,
                        "bb_pred"
                    );

                std::vector<int> out = self.get_all_misclassifications_packed_trie(
                    X_row_major,
                    y_vec,
                    budget_override,
                    bb_pred_vec
                );

                return vector_int_to_numpy(out);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("budget_override"),
            py::arg("bb_pred") = py::none()
        )

        .def(
            "get_all_deferrals_packed_trie",
            [](const PRAXIS &self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               int budget_override
            ) {
                std::vector<std::vector<uint8_t>> X_row_major =
                    numpy_uint8_to_row_major(X);

                std::vector<int> out = self.get_all_deferrals_packed_trie(
                    X_row_major,
                    budget_override
                );

                return vector_int_to_numpy(out);
            },
            py::arg("X"),
            py::arg("budget_override")
        );

    m.def(
        "rid_subtractive_model_reliance",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
           py::array_t<int,     py::array::c_style | py::array::forcecast> y,
           int n_boot,
           double lambda_reg,
           int depth_budget,
           double rashomon_mult,
           int lookahead_k,
           std::uint64_t seed,
           bool memory_efficient,
           py::object binning_map_obj,
           bool use_deferral,
           double eta_defer,
           py::object bb_pred_obj,
           bool return_joint_samples
        ) {
            py::buffer_info xinfo = X.request();
            py::buffer_info yinfo = y.request();

            if (xinfo.ndim != 2) {
                throw std::runtime_error("X must be 2D");
            }

            if (yinfo.ndim != 1) {
                throw std::runtime_error("y must be 1D");
            }

            const int n_samples  = (int)xinfo.shape[0];
            const int n_features = (int)xinfo.shape[1];

            if ((int)yinfo.shape[0] != n_samples) {
                throw std::runtime_error("y must match X rows");
            }

            std::vector<std::vector<uint8_t>> X_row_major =
                numpy_uint8_to_row_major(X);

            std::vector<int> y_vec = numpy_int_to_vector(y, "y");

            std::vector<int> bb_pred_vec =
                optional_numpy_int_to_vector(
                    bb_pred_obj,
                    n_samples,
                    "bb_pred"
                );

            if (use_deferral && bb_pred_vec.empty()) {
                throw std::runtime_error("use_deferral=true requires bb_pred");
            }

            // binning map.
            // if binning_map is None, each binary column is its own feature group.
            std::vector<std::vector<int>> groups;

            if (binning_map_obj.is_none()) {
                groups.resize((size_t)n_features);

                for (int j = 0; j < n_features; ++j) {
                    groups[(size_t)j] = std::vector<int>{j};
                }
            } else {
                py::dict bm = binning_map_obj.cast<py::dict>();

                std::vector<int> keys;
                for (auto item : bm) {
                    keys.push_back(py::cast<int>(item.first));
                }

                std::sort(keys.begin(), keys.end());

                groups.reserve(keys.size());

                for (int k : keys) {
                    py::list lst = bm[py::int_(k)].cast<py::list>();

                    std::vector<int> cols;
                    cols.reserve((size_t)py::len(lst));

                    for (auto h : lst) {
                        const int col = py::cast<int>(h);

                        if (col < 0 || col >= n_features) {
                            throw std::runtime_error(
                                "binning_map contains column index outside [0, n_features)"
                            );
                        }

                        cols.push_back(col);
                    }

                    groups.push_back(std::move(cols));
                }
            }

            RIDResult r = compute_rid_subtractive_mr_bootstrap(
                X_row_major,
                y_vec,
                n_boot,
                lambda_reg,
                depth_budget,
                rashomon_mult,
                lookahead_k,
                seed,
                memory_efficient,
                groups,
                use_deferral,
                eta_defer,
                bb_pred_vec,
                return_joint_samples
            );

            py::dict out;
            out["mean_sub_mr"] = r.mean_sub_mr;
            out["cdf_x"] = r.cdf_x;
            out["cdf_p"] = r.cdf_p;
            if (return_joint_samples) {
                out["feature_importance_weight_samples"] =
                    r.feature_importance_weight_samples;
            }

            return out;
        },
        py::arg("X"),
        py::arg("y"),
        py::arg("n_boot") = 10,
        py::arg("lambda_reg") = 0.01,
        py::arg("depth_budget") = 5,
        py::arg("rashomon_mult") = 0.05,
        py::arg("lookahead_k") = 1,
        py::arg("seed") = 0,
        py::arg("memory_efficient") = false,
        py::arg("binning_map") = py::none(),
        py::arg("use_deferral") = false,
        py::arg("eta_defer") = 0.0,
        py::arg("bb_pred") = py::none(),
        py::arg("return_joint_samples") = false
    );
}