#pragma once

#include "algorithms/common/dp_decision_node.h"

#include "mcts_types.h"
#include "mcts_chance_node.h"
#include "mcts_decision_node.h"


namespace mcts {
    // forward declare corresponding DPDNode class
    class DPDNode;

    // Typedef for children map
    typedef std::unordered_map<std::shared_ptr<const Observation>, std::shared_ptr<DPDNode>> DPDNodeChildMap;

    /**
     * An implementation of dynamic programming backups for nodes to use.
     * 
     * Member variables:
     *      num_backups: 
     *          The number of backups this node has performed (== "number of visits" with respect to dp backup)
     *      dp_value: 
     *          The dynamic programming value at this node
     */
    class DPCNode {
        // Alloow DPDNode access to private members
        friend DPDNode;

        protected:
            int num_backups;
            double dp_value;

            /**
             * Constructor 
             */
            DPCNode() : num_backups(0), dp_value(0.0){};

            /**
             * Destructor
            */
           virtual ~DPCNode() = default;

            /**
             * Performs a dynamic programming backup.
             * 
             * I.e. Q(s,a) = R(s,a) + E_{s'}[V(s')]
             * 
             * Assumes all children are locked.
             * 
             * Args:
             *      children: The children map for this node
             *      local_reward: A value for the reward at this node (i.e. R(s,a))
             *      is_opponent: True if this node is acting as an opponent in a two player game
             */
            void backup_dp_impl(DPDNodeChildMap& children, double local_reward, bool is_opponent);


            /**
             * Helper to convert children maps into children maps for DP Nodes.
             * 
             * Templated with the top-level class of the MctsDNode, so that the children can be cast from
             * MctsDNode -> T -> DPDNode
             * 
             * Args:
             *      children: A children map for a MctsCNode, mapping observations to DNodes.
             * 
             * Returns:
             *      A map from observations to DPDNodes, to be used in backup.
             */
            template <typename T>
            std::shared_ptr<DPDNodeChildMap> convert_child_map(DNodeChildMap& children) const {
                std::shared_ptr<DPDNodeChildMap> dp_children = std::make_shared<DPDNodeChildMap>();
                for (std::pair<std::shared_ptr<const Observation>,std::shared_ptr<MctsDNode>> pr : children) {
                    std::shared_ptr<T> superclass_ptr = std::static_pointer_cast<T>(pr.second);
                    std::shared_ptr<DPDNode> dp_node_ptr = std::static_pointer_cast<DPDNode>(superclass_ptr);
                    dp_children->insert_or_assign(pr.first, dp_node_ptr);
                }
                return dp_children;
            }

        public:
            /**
             * Interface for calling the backup function for MctsCNode classes subclassing this DPCNode.
             * 
             * Casts the child map so that the DPCNode can use it, and assures that all children are 
             * locked around the backup_impl call.
             * 
             * Templated with the top-level class of the MctsDNode, so that the children can be cast from
             * MctsDNode -> T -> DPDNode.
             * 
             * Args:
             *      children: The children map for a MctsDNode (that are ultimately of type T)
             *      local_reward: A value for the reward at this node (i.e. R(s,a))
             *      is_opponent: True if this node is acting as an opponent in a two player game.
             */
            template <typename T>
            void backup_dp(DNodeChildMap& children, double local_reward, bool is_opponent=false) {
                std::shared_ptr<DPDNodeChildMap> dp_children = convert_child_map<T>(children);
                for (auto pr : children) pr.second->lock();
                backup_dp_impl(*dp_children, local_reward, is_opponent);
                for (auto pr : children) pr.second->unlock();
            }
    };
}