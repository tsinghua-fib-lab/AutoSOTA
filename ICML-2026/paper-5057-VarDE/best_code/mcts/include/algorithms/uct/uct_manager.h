#pragma once

#include "mcts_manager.h"

namespace mcts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct UctManagerArgs : public MctsManagerArgs {
        static constexpr double USE_AUTO_BIAS = -1.0;
        static constexpr double AUTO_BIAS_MIN_BIAS = 0.001;

        static constexpr double bias_default=USE_AUTO_BIAS;
        static const int heuristic_psuedo_trials_default=0;
        static const bool recommend_most_visited_default=true;
        static constexpr double epsilon_exploration_default=0.0;

        double bias;
        int heuristic_psuedo_trials;
        bool recommend_most_visited;
        double epsilon_exploration;

        UctManagerArgs(std::shared_ptr<MctsEnv> mcts_env) :
            MctsManagerArgs(mcts_env),
            bias(bias_default),
            heuristic_psuedo_trials(heuristic_psuedo_trials_default),
            recommend_most_visited(recommend_most_visited_default),
            epsilon_exploration(epsilon_exploration_default) {}

        virtual ~UctManagerArgs() = default;
    };

    /**
     * A specific instance of MctsManager for UCT algorithms.
     * 
     * Member variables:
     *      bias:
     *          The bias to use in the ucb values at decision nodes. If set to 'USE_AUTO_BIAS' then an adaptive bias is
     *          used as outlined by the PROST planner (https://www.aaai.org/ocs/index.php/ICAPS/ICAPS12/paper/viewFile/4715/4721).
     *      heuristic_psuedo_trials:
     *          The number of 'psuedo trials' to weight the value of the heuristic functino by. Should be used to 
     *          initialise the 'num_visits' of UCT nodes. A value of zero indicates that the heuristic function should 
     *          be ignored entirely (bool use_heuristic_fn == (heuristic_psuedo_trials == 0)).
     *      recommend_most_visited:
     *          If true then on recommendations return the action corresponding to the child that has been visited the 
     *          most. When false, recommend the child with the best empirical average.
     *      epsilon_exploration:
     *          Defines the proportion of time to be spent exploring uniformly randomly (i.e. select a random action 
     *          rather than using the UCB formula). Default set to zero and to purely use the primary action selection. 
     *          This value should be in the range [0,1].
     */
    class UctManager : public MctsManager {
        public:
            static constexpr double USE_AUTO_BIAS = UctManagerArgs::USE_AUTO_BIAS;
            static constexpr double AUTO_BIAS_MIN_BIAS = UctManagerArgs::AUTO_BIAS_MIN_BIAS;

            double bias;
            int heuristic_psuedo_trials;
            bool recommend_most_visited;
            double epsilon_exploration;

            UctManager(const UctManagerArgs& args) :
                MctsManager(args),
                bias(args.bias),
                heuristic_psuedo_trials(args.heuristic_psuedo_trials),
                recommend_most_visited(args.recommend_most_visited),
                epsilon_exploration(args.epsilon_exploration) {};
    };
}