#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

namespace interventional_cpp {

using FeatureIndex = std::uint32_t;
using InteractionKey = std::vector<FeatureIndex>;
using IntervalTuple = std::tuple<
    std::vector<FeatureIndex>,  // A
    std::vector<FeatureIndex>,  // B
    std::vector<FeatureIndex>,   // NB
    double                    // interval value
>;

struct VectorHash {
    std::size_t operator()(const InteractionKey& key) const noexcept;
};

struct VectorEqual {
    bool operator()(const InteractionKey& lhs, const InteractionKey& rhs) const noexcept;
};

using InteractionMap = std::unordered_map<InteractionKey, double, VectorHash, VectorEqual>;

void update_interaction_values(
    InteractionMap& interaction_map,
    double const_coalition,
    const std::vector<FeatureIndex>& A,
    const std::vector<FeatureIndex>& B,
    const std::vector<FeatureIndex>& NB,
    int max_order,
    const std::string& weight_func
);

}  // namespace interventional_cpp
