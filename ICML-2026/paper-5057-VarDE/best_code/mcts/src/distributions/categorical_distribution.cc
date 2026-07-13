#include "distributions/categorical_distribution.h"

#include "helper_templates.h"
#include "mcts_manager.h"

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <iostream>


namespace mcts {
    using namespace std;

    /**
     * Constructor 
    */
    template <typename T>
    CategoricalDistribution<T>::CategoricalDistribution(
        shared_ptr<unordered_map<T,double>> distr,
        bool use_alias_method,
        int reconstruct_alias_table_freq) :
            distr(distr),
            use_alias_method(use_alias_method),
            reconstruct_alias_table_freq(reconstruct_alias_table_freq),
            num_updates(0),
            alias_table()
    {
        if (use_alias_method) {
            construct_alias_table();
        }
    }

    template <typename T>
    T CategoricalDistribution<T>::sample(RandManager& rand_manager) {
        if (!use_alias_method) {
            return helper::sample_from_distribution(*distr, rand_manager, false);
        }

        int rng = rand_manager.get_rand_int(0, distr->size());
        AliasTableEntry& alias_entry = alias_table[rng];
        if (1.0 > alias_entry.threshold && alias_entry.threshold > rand_manager.get_rand_uniform()) {
            return alias_entry.first;
        }
        return alias_entry.second;
    };

    template <typename T>
    void CategoricalDistribution<T>::update(T key, double weight) {
        distr->insert_or_assign(key, weight);
        num_updates++;
        if (use_alias_method && num_updates == reconstruct_alias_table_freq) {
            reconstruct_alias_table();
            num_updates = 0;
        }
    }

    /**
     * If using alias table, then should reconstruct it
     * If size(distr) == size(new_distr), then call reconstruct_alias_table, which assumes size(distr) hasn't changed
     * If distirubtion has changes in size (num outcomes), then clear alias table and construct from scratch
    */
    template <typename T>
    void CategoricalDistribution<T>::update(shared_ptr<unordered_map<T,double>> new_distr) {
        distr = new_distr;
        num_updates = 0;
        if (use_alias_method) {
            if (new_distr->size() == distr->size()) {
                reconstruct_alias_table(false);
            } else {
                alias_table.clear();
                construct_alias_table();
            }
        }
    }
    
    /**
     * Initial construction of the alias table. The only difference to this is that we initially fill the alias table 
     * with entries, whereas reconstruct will reset all of the entries (to prevent frivilously recreating the whole 
     * array every time). I'm anticipating reconstruct_alias_table to be called a lot.
    */
    template <typename T>
    void CategoricalDistribution<T>::construct_alias_table() {
        double sum_weights = 0.0;
        for (pair<const T,double>& kv_pair : *distr) {
            sum_weights += kv_pair.second;
        }

        int n = distr->size();
        for (pair<const T,double> kv_pair : *distr) {
            double u = kv_pair.second * n / sum_weights;
            alias_table.push_back(AliasTableEntry(u, kv_pair.first));
        }

        reconstruct_alias_table(true);
    }

    /**
     * Firstly resets all of the entries in the table. Let n be the number of items in the distribution. 
     * For each t,p in the distribtuion, set an entry e to have the values:
     * e.threshold = p*n
     * e.first = t,
     * e.second = t
     * 
     * We then partition the entries into 'small_indices' which correspond to the entries where 'threshold < 1.0'. And 
     * similarly 'large_indices' for where 'threshold > 1.0'. 
     * 
     * The main loop of construction is as follows:
     *      1. Take any entry 'el' from the large indices, and any entry 'es' from the small indices. 
     *      2. We allocate the spare probability density in es to el by setting es.second = el.first
     *          (Note that el.threshold > 1.0 > 1.0 - es.threshold, so el always has enough density to fill es up)
     *      3. We update the density remaining for el by setting el.threshold = el.threshold - (1.0 - es.threshold)
     *      4. If 'el.threshold < 1.0' then add el to the small indices, and if 'el.threshold > 1.0' add it to large
     *      5. Repeat this until one of the large or small sets is empty.
     * 
     * When we only have one of the small or large sets left, then we set all of their remaining thresholds to 1.0.
     * According to the wikipedia page (https://en.wikipedia.org/wiki/Alias_method) there is some numerical rounding 
     * issues with this method for floating point computations. But we avoid this by the setting these remaining values 
     * to 1.0. We shouldn't care if the densities sampled here are 1e-16 off, I just means that less frequently than 
     * once in a billion we sample the wrong thing. And theoretically fixing it requires sorting the entries and a 
     * O(nlogn) cost, which isn't worth it.
    */
    template <typename T>
    void CategoricalDistribution<T>::reconstruct_alias_table(bool just_constructed) {
        double sum_weights = 0.0;
        for (pair<const T,double>& kv_pair : *distr) {
            sum_weights += kv_pair.second;
        }

        if (!just_constructed) {
            int i = 0;
            int n = distr->size();
            for (pair<const T,double> kv_pair : *distr) {
                double u = kv_pair.second * n / sum_weights;
                alias_table[i].threshold = u;
                alias_table[i].first = kv_pair.first;
                alias_table[i].second = kv_pair.first;
                i++;
            }
        }

        vector<int> large_indices;
        vector<int> small_indices;
        for (size_t i = 0; i < alias_table.size(); i++) {
            if (alias_table[i].threshold > 1.0) {
                large_indices.push_back(i);
            } else if (alias_table[i].threshold < 1.0) {
                small_indices.push_back(i);
            }
        }

        while (large_indices.size() > 0 && small_indices.size() > 0) {
            int l_index = large_indices.back();
            int s_index = small_indices.back();
            large_indices.pop_back();
            small_indices.pop_back();
            
            alias_table[s_index].second = alias_table[l_index].first;
            alias_table[l_index].threshold -= 1.0 - alias_table[s_index].threshold;

            if (alias_table[l_index].threshold > 1.0) {
                large_indices.push_back(l_index);
            } else if (alias_table[l_index].threshold < 1.0) {
                small_indices.push_back(l_index);
            }
        }

        for (int l_index : large_indices) {
            alias_table[l_index].threshold = 1.0;
        }

        for (int s_index : small_indices) {
            alias_table[s_index].threshold = 1.0;
        }
    };
}