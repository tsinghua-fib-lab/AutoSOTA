#pragma once

#include "algorithms/ments/ments_manager.h"

namespace mcts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct DentsManagerArgs : public MentsManagerArgs {
        static constexpr TempDecayFnPtr value_temp_decay_fn_default=decayed_temp_inv_sqrt;
        static constexpr double value_temp_init_default=1.0;
        static constexpr double value_temp_decay_min_temp_default=1.0e-6;
        static constexpr double value_temp_decay_visits_scale_default=1.0;
        static constexpr double value_temp_decay_root_node_visits_scale_default=-1.0;
        static const bool use_dp_value_default=true;

        TempDecayFnPtr value_temp_decay_fn;
        double value_temp_init;
        double value_temp_decay_min_temp;
        double value_temp_decay_visits_scale;
        double value_temp_decay_root_node_visits_scale;

        bool use_dp_value;

        DentsManagerArgs(std::shared_ptr<MctsEnv> mcts_env) :
            MentsManagerArgs(mcts_env),
            value_temp_decay_fn(value_temp_decay_fn_default),
            value_temp_init(value_temp_init_default),
            value_temp_decay_min_temp(value_temp_decay_min_temp_default),
            value_temp_decay_visits_scale(value_temp_decay_visits_scale_default),
            value_temp_decay_root_node_visits_scale(value_temp_decay_root_node_visits_scale_default),
            use_dp_value(use_dp_value_default) {}

        virtual ~DentsManagerArgs() = default;
    };

    /**
     * A specific instance of MctsManager for DENTS algorithms.
     * 
     * The 'value_temp' refers to the temperature coefficient of entropy when computing (soft) values
     * 
     * Member variables (value temp decay):
     *      value_temp_decay_fn:
     *          The decay function to use for the value temp. A value of nullptr can be used to indicate that this 
     *          decay shouldn't be used (in which case this algorithm will run similarly to MENTS, but with seperately 
     *          computed value and entropy, rather than the single soft value)
     *      value_temp_init:
     *          The initial value for the value temp
     *      value_temp_decay_min_temp:
     *          The minimum temperature to use for the value temp
     *      value_temp_decay_visits_scale:
     *          The scaling to use with the valuje_temp decay function. This is an x-axis scaling for the number of 
     *          visits input into the temp decay function
     *      value_temp_decay_root_node_visits_scale:
     *          Same as the visits scale, but for the root node. Default value of -1.0 indicates to use the same value 
     *          as 'value_temp_decay_visits_scale'
     * 
     * Member variables (values):
     *      use_dp_value:
     *          If true (default) then dynamic programming backups will be used for (*not* soft) value estimates. If 
     *          false then the average return is used. Additionally, when MentsManager::recommend_most_visited is false,
     *          this option decides if we use dp values or average returns to recommend actions.
     */
    class DentsManager : public MentsManager {
        public:
            TempDecayFnPtr value_temp_decay_fn;
            double value_temp_init;
            double value_temp_decay_min_temp;
            double value_temp_decay_visits_scale;
            double value_temp_decay_root_node_visits_scale;

            bool use_dp_value;

            DentsManager(const DentsManagerArgs& args) :
                MentsManager(args),
                value_temp_decay_fn(args.value_temp_decay_fn),
                value_temp_init(args.value_temp_init),
                value_temp_decay_min_temp(args.value_temp_decay_min_temp),
                value_temp_decay_visits_scale(args.value_temp_decay_visits_scale),
                value_temp_decay_root_node_visits_scale(args.value_temp_decay_root_node_visits_scale),
                use_dp_value(args.use_dp_value) {};
    };
}