#pragma once

#include "distributions/distribution.h"

#include "mcts_manager.h"

#include <unordered_map>
#include <vector>


namespace mcts {
    /**
     * A catagorical distribution
     * 
     * Implements a distribution over objects of type T, with given probability (weights). By default we use the 
     * O(n) sampling from helper::sample_from_distribution, but optionally construct an alias table for O(1) sampling, 
     * which takes O(n) time to construct.
     * 
     * We povide a function that allows the weights to be updated, but will take O(n) time if the alias table needs to 
     * be reconstructed. To allow this class to be used in amortised O(1) time we only update the alias table every 
     * 'reconstruct_alias_table_freq' updates, so if the 'reconstruct_alias_table_freq' is set to O(n), then we only 
     * reconstruct the table every O(n) updates, which means the distribution will typically be a bit outdated, but 
     * allows amortised O(1) use of the update function.
     * 
     * If the alias table is 'at' and is of size n (i.e. there are n items in the categorical distribution) then we 
     * sample i discretely from [0,n) uniformly and sample x from [0,1). The value returned is then 
     * at[i].first if x < at[i].threshold and otherwise we return at[i].second.
     * 
     * The alias method is originally published here:
     * https://digital-library.theiet.org/content/journals/10.1049/el_19740097
     * 
     * The wikipedia page has a good description of the method:
     * https://en.wikipedia.org/wiki/Alias_method
     * 
     * Member variables:
     *      distr: 
     *          A dictionary mapping from objects of type T to doubles, defining the categorical distribution
     *      use_alias_method: 
     *          A boolean stating if the alias method should be used
     *      reconstruct_alias_table_freq: 
     *          The integer frequency of how often to reconstruct the alias table (with respect to the number of times 
     *          the update function is called). Default value is 1, to update the alias table every time 'update' is 
     *          called.
     *      num_updates: 
     *          An integer keeping track of how many times probabilities have been updated 
     *      alias_table: 
     *          The alias table to use for O(1) sampling. It is a list of AliasTableEntries
    */
    template <typename T>
    class CategoricalDistribution : public Distribution<T> {

        /**
         * An internal data structure used in the alias table
         * If this entry is selected from the alias table, then we should return 'first' with probabiltiy 'threshold' 
         * and we should return 'second' with probability '1.0-threshold'.
        */
        struct AliasTableEntry {
            double threshold;
            T first;
            T second;
            AliasTableEntry(
                double threshold, T first) : threshold(threshold), first(first), second(first) {};
            AliasTableEntry(
                double threshold, T first, T second) : threshold(threshold), first(first), second(second) {};
        };

        /**
         * Member variables
        */
        protected:
            std::shared_ptr<std::unordered_map<T,double>> distr;
            bool use_alias_method;
            int reconstruct_alias_table_freq;
            int num_updates;
            std::vector<AliasTableEntry> alias_table;
        
        public:
            /**
             * Constructor
            */
            CategoricalDistribution(
                std::shared_ptr<std::unordered_map<T,double>> distr, 
                bool use_alias_method=false, 
                int reconstruct_alias_table_freq=1);

            /**
             * Samples a random T from the categorical distribution 'distr'
            */
            virtual T sample(RandManager& rand_manager);

            /**
             * Updates the probability weight of 'key' in 'distr' to have density 'weight'. 
             * I.e. sets distr[key] = weight.
            */
            void update(T key, double weight);

            /**
             * Updates the distribution with a completely new 'new_distr' mapping. If using the alias method, this will 
             * always construct a new alias table
            */
            void update(std::shared_ptr<std::unordered_map<T,double>> new_distr);
        
        private:
            /**
             * Initially constructs the 'alias_table' for the initial 'distr'.
             * This will just initially construct the alias table and then call 'reconstruct_alias_table(true)'
            */
            void construct_alias_table();

            /**
             * Reconstructs 'alias_table' for the current 'distr'.
             * This will reset the alias table at the start, unless 'just_constructed == true'
            */
            void reconstruct_alias_table(bool just_constructed=false);
        
    };
}

#include "distributions/categorical_distribution.cc"