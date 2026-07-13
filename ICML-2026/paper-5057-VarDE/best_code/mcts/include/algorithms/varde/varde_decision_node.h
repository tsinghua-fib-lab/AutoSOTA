#pragma once

#include "algorithms/varde/varde_chance_node.h"
#include "algorithms/varde/varde_manager.h"
#include "algorithms/common/dp_decision_node.h"
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
    class RvipCNode;

    class RvipDNode : public MctsDNode, public DPDNode {
        friend RvipCNode;

        protected:
            int num_backups;
            double avg_return;
            std::shared_ptr<ActionVector> actions;

            void ensure_actions_initialized();

            double compute_influence_weight(std::shared_ptr<const Action> action) const;
            double compute_variance_reduction_score(std::shared_ptr<const Action> action) const;
            std::shared_ptr<const Action> select_action_varde(MctsEnvContext& ctx);
            std::shared_ptr<RvipCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

        public:
            RvipDNode(
                std::shared_ptr<RvipManager> mcts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const RvipCNode> parent = nullptr);

            virtual ~RvipDNode() = default;

            bool has_child_node(std::shared_ptr<const Action> action) const;
            std::shared_ptr<RvipCNode> get_child_node(std::shared_ptr<const Action> action) const;
            std::shared_ptr<RvipCNode> create_child_node(std::shared_ptr<const Action> action);

            virtual void visit(MctsEnvContext& ctx);
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                MctsEnvContext& ctx);
            virtual std::shared_ptr<const Action> select_action(MctsEnvContext& ctx);
            virtual std::shared_ptr<const Action> recommend_action(MctsEnvContext& ctx) const;
            virtual std::string get_pretty_print_val() const;

            virtual void visit_itfc(MctsEnvContext& ctx) override;
            virtual std::shared_ptr<const Action> select_action_itfc(MctsEnvContext& ctx) override;
            virtual std::shared_ptr<const Action> recommend_action_itfc(MctsEnvContext& ctx) const override;
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                MctsEnvContext& ctx) override;
            virtual std::shared_ptr<MctsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const override;
    };
}
