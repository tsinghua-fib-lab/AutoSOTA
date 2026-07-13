#pragma once

#include "algorithms/common/decaying_temp.h"
#include "mcts_manager.h"

namespace mcts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct MentsManagerArgs : public MctsManagerArgs {
        static constexpr double temp_default=1.0;
        static constexpr double prior_policy_search_weight_default=0.0;
        static constexpr double epsilon_default=0.5;
        static constexpr double root_node_epsilon_default=-1.0;
        static constexpr double max_explore_prob_default=1.0;

        static constexpr TempDecayFnPtr temp_decay_fn_default=nullptr;
        static constexpr double temp_decay_min_temp_default=1.0e-6;
        static constexpr double temp_decay_visits_scale_default=1.0;
        static constexpr double temp_decay_root_node_visits_scale_default=-1.0;

        static constexpr double default_q_value_default=0.0;
        static const bool shift_pseudo_q_values_default=false;
        static constexpr double psuedo_q_value_offset_default=0.0;

        static const int recommend_visit_threshold_default=0;
        static const bool recommend_most_visited_default=false;

        double temp;
        double prior_policy_search_weight;
        double epsilon;
        double root_node_epsilon;
        double max_explore_prob;

        TempDecayFnPtr temp_decay_fn;
        double temp_decay_min_temp;
        double temp_decay_visits_scale;
        double temp_decay_root_node_visits_scale;

        double default_q_value;
        bool shift_pseudo_q_values;
        double psuedo_q_value_offset;
        
        int recommend_visit_threshold;
        bool recommend_most_visited;

        MentsManagerArgs(std::shared_ptr<MctsEnv> mcts_env) :
            MctsManagerArgs(mcts_env),
            temp(temp_default),
            prior_policy_search_weight(prior_policy_search_weight_default),
            epsilon(epsilon_default),
            root_node_epsilon(root_node_epsilon_default),
            max_explore_prob(max_explore_prob_default),
            temp_decay_fn(temp_decay_fn_default),
            temp_decay_min_temp(temp_decay_min_temp_default),
            temp_decay_visits_scale(temp_decay_visits_scale_default),
            temp_decay_root_node_visits_scale(temp_decay_root_node_visits_scale_default),
            default_q_value(default_q_value_default),
            shift_pseudo_q_values(shift_pseudo_q_values_default),
            psuedo_q_value_offset(psuedo_q_value_offset_default),
            recommend_visit_threshold(recommend_visit_threshold_default),
            recommend_most_visited(recommend_most_visited_default) {}

        virtual ~MentsManagerArgs() = default;
    };

    /**
     * A specific instance of MctsManager for MENTS algorithms.
     * 
     * If p1 is the energy based policy from ments, p2 is the uniform policy and pp is the prior policy, with weightings
     * w2, wp for p2 and pp respectively (weights computed using 'epsilon' and 'prior_policy_search_weight'). Then the 
     * search policy will have probabilities according to:
     *      (1-w2)(1-wp)*p1 + (1-w2)wp*pp + w2*p2
     *      
     * 
     * Member variables (search):
     *      temp:
     *          The temperature to use in energy based policy (and soft (max entropy) backups) 
     *      prior_policy_search_weight:
     *          A weighting to use for the prior policy
     *      epsilon:
     *          The epsilon exploration parameters from MENTS. I.e. MENTS will uniformly randomly sample an action with 
     *          probability 'epsilon_exploration / log(num_visits+1)' (assuming its a valid probability!).
     *      root_node_epsilon:
     *          An alternative value to use for 'epsilon' at the root search node. The default value of -1.0 indicates 
     *          that we should use the value of 'epsilon' at the root node too.
     *      max_explore_prob:
     *          In MENTS action selection an action is uniformly randomly sampled with probability 
     *          'epsilon_exploration / log(num_visits+1)'. This value provides a maximum probability of uniformly 
     *          exploring. This value must be in the range [0,1].
     * 
     * Member variables (temperature decay):
     *      temp_decay_fn: 
     *          The temperature decay function to use. Default = nullptr, meaning no temperature decay
     *      temp_decay_min_temp:
     *          The minimum temperature that we allow the temperature to be decayed to
     *      temp_decay_visits_scale:
     *          A weight to scale 'num_visits' by in the input to the decay function. Essentially scaling the x-axis of 
     *          the decay function.
     *      temp_decay_root_node_visits_scale:
     *          An alternative value to use for 'visits_scale' at the root search node. The default value of -1.0 
     *          indicates we should use the value of 'visits_scale' at the root node too.
     * 
     * Member variables (values / backups):
     *      default_q_value:
     *          A default value to use for q-values, typically 0 in a reward maximisation setting. An example of this 
     *          being useful is in cost minimisation settings, where values are non-positive, in which initialising 
     *          q-values to 0.0 leads to running a BFS, which is not the intended operation.
     *      shift_pseudo_q_values:
     *          If true, shifts the 'psuedo q-value's computed from a policy prior, similarly to 'psuedo_q_value_offset' 
     *          such that the initial mean value q-value is zero. Note that 'pseudo_q_value_offset' is applied *after* 
     *          this shift is applied (so the mean value will be at 'psuedo_q_value_offset'). 
     *      psuedo_q_value_offset:
     *          For a given state action pair (s,a), when we have policy_prior pi(a|s), the 'psuedo q-value' is set to 
     *          'log(pi(a|s)) + psuedo_q_value_offset'. This doesn't change the distribution when there are zero children 
     *          at a decision node, but when a decision node has children, it will change the relative weight of 
     *          actions that do and dont have a child node created. This parameter is meaningless when 
     *          'shift_psuedo_q_values' == false.
     * 
     * Member variables (recommendations):
     *      recommend_visit_threshold:
     *          A minimum number of visits required to recommend an action. We can use this to force recommendations 
     *          to have a minimum number of samples before its a candidate for recommendation.
     *      recommend_most_visited:
     *          If we should recommend the most visited child node instead of the largest value.
     *          
     */
    class MentsManager : public MctsManager {
        public:
            double temp;
            double prior_policy_search_weight;
            double epsilon;
            double root_node_epsilon;
            double max_explore_prob;

            TempDecayFnPtr temp_decay_fn;
            double temp_decay_min_temp;
            double temp_decay_visits_scale;
            double temp_decay_root_node_visits_scale;

            double default_q_value;
            bool shift_pseudo_q_values;
            double psuedo_q_value_offset;

            int recommend_visit_threshold;
            bool recommend_most_visited;

            MentsManager(const MentsManagerArgs& args) :
                MctsManager(args),
                temp(args.temp),
                prior_policy_search_weight(args.prior_policy_search_weight),
                epsilon(args.epsilon),
                root_node_epsilon(args.root_node_epsilon),
                max_explore_prob(args.max_explore_prob),
                temp_decay_fn(args.temp_decay_fn),
                temp_decay_min_temp(args.temp_decay_min_temp),
                temp_decay_visits_scale(args.temp_decay_visits_scale),
                temp_decay_root_node_visits_scale(args.temp_decay_root_node_visits_scale),
                default_q_value(args.default_q_value),
                shift_pseudo_q_values(args.shift_pseudo_q_values),
                psuedo_q_value_offset(args.psuedo_q_value_offset),
                recommend_visit_threshold(args.recommend_visit_threshold),
                recommend_most_visited(args.recommend_most_visited) {};
    };
}