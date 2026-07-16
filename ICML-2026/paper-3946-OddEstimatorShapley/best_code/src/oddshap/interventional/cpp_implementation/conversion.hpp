#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace conversion {

struct LeafMatrix {
	std::vector<std::vector<int>> feature_paths;
	std::vector<double> leaf_values;
};

struct TreeArrays {
	const int* children_left;
	const int* children_right;
	const int* features;
	const double* values;
	std::size_t node_count;
};

LeafMatrix tree_to_leaf_matrix(
	const int* children_left,
	const int* children_right,
	const int* features,
	const double* values,
	std::size_t node_count,
	std::size_t feature_count_hint = 0
);

LeafMatrix forest_to_leaf_matrix(
	const std::vector<TreeArrays>& trees,
	std::size_t feature_count_hint = 0
);

}  // namespace conversion