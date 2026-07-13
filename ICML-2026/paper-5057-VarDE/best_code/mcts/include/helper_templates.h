#pragma once

#include "mcts_manager.h"
#include "mcts_types.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace mcts::helper {
    /**
     * Adds an object into a hash value. Can be used to create a hash for composite objects, by combinding the hashes 
     * of the composed objects.
     * 
     * Args:
     *      cur_hash: A value representing an existing hash value
     *      v: An object that we wish to combine into the hash value
     * 
     * Returns:
     *      A hash value based on the existing 'cur_hash' and additional value 'v'.
     */
    template <typename T>
    std::size_t hash_combine(const std::size_t cur_hash, const T& v);

    /**
     * Helper for selecting the maximum key from a map of values, breaking ties randomly.
     * 
     * Args:
     *      map: The map of values to select the maximum from
     *      rand_manager: 
     *          An instance of RandManager, so we can use our 'get_random_int' wrapper around random number generation 
     * 
     * Returns:
     *      The key corresponding to the maximum value, breaking ties randomly
     */
    template <typename T, typename NumericT>
    T get_max_key_break_ties_randomly(std::unordered_map<T,NumericT>& map, RandManager& rand_manager);

    /**
     * Helper function to sample from a discrete distribution
     * 
     * Args:
     *      distribution: A mapping from the support (discrete categories) to their weights
     *      rand_manager: An instance of RandManager, so we can use our wrappers around random number generation
     *      normalised: true iff the weights sum to 1.0
     * 
     * Returns:
     *      An item sampled from the distribution
     */
    template <typename T>
    T sample_from_distribution(
        std::unordered_map<T,double>& distribution, RandManager& rand_manager, bool normalised=true);

    /**
     * Helper function for printing vector types to strings. Assumes that the type T can be fed into an ostream.
     * 
     * Args:
     *      vec: The vector to pretty print
     * 
     * Returns:
     *      A string representing a pretty printed version of the vector.
     */
    template <typename T>
    std::string vector_pretty_print_string(const std::vector<T>& vec);

    /**
     * Helper function for printing unordered_map types to strings. Assumes that the types K,V can be fed into an 
     * ostream.
     * 
     * Args:
     *      mp: The map to pretty print
     *      delimiter: A delimiter to use between the keys and values
     * 
     * Returns:
     *      A string representing a pretty printed version of the vector.
     */
    template <typename K, typename V>
    std::string unordered_map_pretty_print_string(const std::unordered_map<K,V>& mp, std::string delimiter=":");
}

#include "helper_templates.cc"