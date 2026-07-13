#pragma once

#include "algorithms/varde/varde_decision_node.h"
#include "algorithms/varde/varde_manager.h"
#include "algorithms/common/dp_chance_node.h"
#include "mcts_types.h"
#include "mcts_chance_node.h"
#include "mcts_decision_node.h"
#include "mcts_env.h"
#include "mcts_env_context.h"
#include "mcts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace mcts {
    class RvipDNode;
    
    class RvipCNode : public MctsCNode, public DPCNode {
        friend RvipDNode;

        protected:
            int num_backups;
            double avg_return;
            double m2_return;
            double local_reward;
            std::shared_ptr<StateDistr> next_state_distr;

            std::shared_ptr<const State> sample_observation_random();
            std::shared_ptr<RvipDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state = nullptr) const;

        public:
            RvipCNode(
                std::shared_ptr<RvipManager> mcts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const RvipDNode> parent);

            virtual ~RvipCNode() = default;

            bool has_child_node(std::shared_ptr<const State> observation) const;
            std::shared_ptr<RvipDNode> get_child_node(std::shared_ptr<const State> observation) const;
            std::shared_ptr<RvipDNode> create_child_node(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state = nullptr);

            virtual void visit(MctsEnvContext& ctx);
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                MctsEnvContext& ctx);
            virtual std::shared_ptr<const State> sample_observation(MctsEnvContext& ctx);
            virtual std::string get_pretty_print_val() const;
            double get_variance() const;

            virtual void visit_itfc(MctsEnvContext& ctx) override;
            virtual std::shared_ptr<const Observation> sample_observation_itfc(MctsEnvContext& ctx) override;
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                MctsEnvContext& ctx) override;
            virtual std::shared_ptr<MctsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, 
                std::shared_ptr<const State> next_state = nullptr) const override;
    };
}
