#pragma once

#include "algorithms/common/ent_decision_node.h"

#include "mcts_types.h"
#include "mcts_chance_node.h"
#include "mcts_decision_node.h"


namespace mcts {
    // forward declare corresponding EntDNode class
    class EntDNode;

    // Typedef for children map
    typedef std::unordered_map<std::shared_ptr<const Observation>, std::shared_ptr<EntDNode>> EntDNodeChildMap;

    /**
     * An implementation of dynamic programming backups for nodes to use.
     * 
     * Member variables:
     *      num_backups: 
     *          The number of backups this node has performed (== "number of visits" with respect to dp backup)
     *      subtree_entropy:
     *          The entropy of the policy over the subtree, rooted at this node
     */
    class EntCNode {
        // Alloow EntDNode access to private members
        friend EntDNode;

        protected:
            int num_backups;
            double subtree_entropy;

            /**
             * Constructor 
             */
            EntCNode() : num_backups(0), subtree_entropy(0.0) {};

            /**
             * Destructor
            */
           virtual ~EntCNode() = default;

            /**
             * Computes the subtree entropy as a backup
             * 
             * Assumes all children are locked.
             * 
             * Args:
             *      children: The children map for this node
             *      is_opponent: True if this node is acting as an opponent in a two player game
             */
            void backup_ent_impl(EntDNodeChildMap& children);


            /**
             * Helper to convert children maps into children maps for DP Nodes.
             * 
             * Templated with the top-level class of the MctsDNode, so that the children can be cast from
             * MctsDNode -> T -> EntDNode
             * 
             * Args:
             *      children: A children map for a MctsCNode, mapping observations to DNodes.
             * 
             * Returns:
             *      A map from observations to EntDNodes, to be used in backup.
             */
            template <typename T>
            std::shared_ptr<EntDNodeChildMap> convert_child_map(DNodeChildMap& children) const {
                std::shared_ptr<EntDNodeChildMap> ent_children = std::make_shared<EntDNodeChildMap>();
                for (std::pair<std::shared_ptr<const Observation>,std::shared_ptr<MctsDNode>> pr : children) {
                    std::shared_ptr<T> superclass_ptr = std::static_pointer_cast<T>(pr.second);
                    std::shared_ptr<EntDNode> ent_node_ptr = std::static_pointer_cast<EntDNode>(superclass_ptr);
                    ent_children->insert_or_assign(pr.first, ent_node_ptr);
                }
                return ent_children;
            }

        public:
            /**
             * Interface for calling the backup function for MctsCNode classes subclassing this EntCNode.
             * 
             * Casts the child map so that the EntCNode can use it, and assures that all children are 
             * locked around the backup_impl call.
             * 
             * Templated with the top-level class of the MctsDNode, so that the children can be cast from
             * MctsDNode -> T -> EntDNode.
             * 
             * Args:
             *      children: The children map for a MctsDNode (that are ultimately of type T)
             *      local_reward: A value for the reward at this node (i.e. R(s,a))
             *      is_opponent: True if this node is acting as an opponent in a two player game.
             */
            template <typename T>
            void backup_ent(DNodeChildMap& children) {
                std::shared_ptr<EntDNodeChildMap> ent_children = convert_child_map<T>(children);
                for (auto pr : children) pr.second->lock();
                backup_ent_impl(*ent_children);
                for (auto pr : children) pr.second->unlock();
            }
    };
}