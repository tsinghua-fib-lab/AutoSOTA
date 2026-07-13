#pragma once

#include "mcts_manager.h"

namespace mcts {
    struct RvipManagerArgs : public MctsManagerArgs {
        static constexpr double temp_default = 1.0;
        static constexpr double variance_floor_default = 0.0;
        static constexpr bool use_dp_value_default = true;
        
        double temp;
        double variance_floor;
        bool use_dp_value;
        
        RvipManagerArgs(std::shared_ptr<MctsEnv> mcts_env) :
            MctsManagerArgs(mcts_env),
            temp(temp_default),
            variance_floor(variance_floor_default),
            use_dp_value(use_dp_value_default) {}

        virtual ~RvipManagerArgs() = default;
    };

    class RvipManager : public MctsManager {
        public:
            double temp;
            double variance_floor;
            bool use_dp_value;

            RvipManager(RvipManagerArgs args);
            virtual ~RvipManager() = default;

            virtual std::shared_ptr<MctsDNode> create_decision_node_helper(
                std::shared_ptr<const State> state, 
                int decision_depth, 
                int decision_timestep, 
                std::shared_ptr<const MctsCNode> parent);

            virtual std::shared_ptr<MctsCNode> create_chance_node_helper(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MctsDNode> parent);
    };
}
