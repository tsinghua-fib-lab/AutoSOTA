#include "helper_templates.h"

#include <cstddef>
#include <float.h>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <iostream>


namespace mcts::helper {
    using namespace std;

    // Static epsilon for 
    static double EPS = 1e-12;

    /**
     * Adapted from boost: https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
     */
    template <typename T>
    size_t hash_combine(const size_t cur_hash, const T& v) {
        hash<T> hash_fn;
        return cur_hash ^ (hash_fn(v) + 0x9e3779b9 + (cur_hash << 6) + (cur_hash >> 2));
    }

    /**
     * Select max value with randomly breaking ties
     */
    template <typename T, typename NumericT>
    T get_max_key_break_ties_randomly(unordered_map<T,NumericT>& map, RandManager& rand_manager) {
        NumericT best_val = numeric_limits<NumericT>::lowest();
        vector<T> best_keys;
        for (pair<const T,NumericT>& pr : map) {
            NumericT val = pr.second;
            if (val < best_val) continue;
            if (val > best_val) {
                best_keys = vector<T>();
                best_val = val;
            }
            const T key = pr.first;
            best_keys.push_back(key);
        }

        if (best_keys.size() == 1) return best_keys[0];
        int indx = rand_manager.get_rand_int(0,best_keys.size());
        return best_keys[indx];
    }

    /**
     * Sampling from distribution
     * 
     * Implementation includes useful (probabilistic) error checking on distributions (which should only occur when 
     * normalised=true):
     * - Firstly, if probability masses reach a sum of 1.0 or greater too quickly then we throw an exception. Consider 
     *      if a distribution {s1: 1.0, s2: 0.5, s3: 0.5} is passed in. This exception will often catch these cases, 
     *      but can also be extremely unlikely to be thrown (when we would want it to be) in edge cases: consider the
     *      distribution {s1: 0.99999, s2: 0.00001, s3: 0.5}. 
     * - Secondly, if we get to the end of the routine without returning a sampled object, then the probability masses 
     *      must have summed to less than 1.0. This exception thrown with probability (1.0-sum_weights).
     */
    template <typename T>
    T sample_from_distribution(unordered_map<T,double>& distribution, RandManager& rand_manager, bool normalised) {
        double sum_weights = 1.0;
        if (!normalised) {
            sum_weights = 0.0;
            for (auto pr : distribution) {
                sum_weights += pr.second;
            }
        }
        
        int i = 0;
        int distr_size = distribution.size();
        double rand_val = rand_manager.get_rand_uniform();
        double running_prob_mass = 0.0;

        for (pair<T,double> pr : distribution) {
            // update mass considered
            running_prob_mass += pr.second / sum_weights;

            // error checking
            i++;
            bool too_much_mass = running_prob_mass > sum_weights + EPS;
            bool complete_mass_too_early = running_prob_mass >= sum_weights && i < distr_size;
            if (too_much_mass || complete_mass_too_early) {
                stringstream error_msg_ss;
                error_msg_ss 
                    << "Probability masses sum to greater than 1.0, have you forgotten to set normalised=false? "
                    << "Distribution was: "<< unordered_map_pretty_print_string(distribution);
                throw runtime_error(error_msg_ss.str());
            }
            
            // return if its time
            if (rand_val < running_prob_mass) {
                return pr.first;
            }
        }

        stringstream error_msg_ss;
        error_msg_ss << "Probability masses sum to less than 1.0, have you forgotten to set normalised=false? "
            << "Distribution was: "
            << unordered_map_pretty_print_string(distribution);
        double sum = 0.0;
        for (pair<T,double> pr : distribution) { 
            sum += pr.second;
        }
        cout << sum << endl;
        throw runtime_error(error_msg_ss.str());
    }


    /**
     * Printing vectors
     */
    template <typename T>
    string vector_pretty_print_string(const vector<T>& vec) {
        stringstream ss;
        ss << "[";
        bool first_iter = true;
        for (const T& val : vec) {
            if (!first_iter) {
                ss << ",";
            } else {
                first_iter = false;
            }
            ss << val;
        }
        ss << "]";
        return ss.str();
    }

    /**
     * Printing maps
     */
    template <typename K, typename V>
    string unordered_map_pretty_print_string(const unordered_map<K,V>& mp, string delimiter) {
        stringstream ss;
        ss << "{";
        bool first_iter = true;
        for (const pair<const K, V>& pr : mp) {
            if (!first_iter) {
                ss << ",";
            } else {
                first_iter = false;
            }
            ss << pr.first << delimiter << pr.second ;
        }
        ss << "}";
        return ss.str();
    }
}