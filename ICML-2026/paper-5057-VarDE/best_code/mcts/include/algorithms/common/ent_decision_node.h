#pragma once

#include "algorithms/common/ent_chance_node.h"

#include "mcts_types.h"
#include "mcts_chance_node.h"
#include "mcts_decision_node.h"


namespace mcts {
    // forward declare corresponding EntCNode class
    class EntCNode;

    // Typedef for children map
    typedef std::unordered_map<std::shared_ptr<const Action>, std::shared_ptr<EntCNode>> EntCNodeChildMap;

    /**
     * An implementation of entropy backups for nodes to use.
     * 
     * Member variables:
     *      num_backups: 
     *          The number of backups this node has performed (== "number of visits" with respect to dp backup)
     *      local_entropy: 
     *          The dynamic programming value at this node
     *      subtree_entropy:
     *          The entropy of the policy over the subtree, rooted at this node
     */
    class EntDNode {
        // Alloow EntCNode access to private members
        friend EntCNode;

        protected:
            int num_backups;
            double local_entropy;
            double subtree_entropy;

            /**
             * Constructor 
             */
            EntDNode() : 
                num_backups(0), local_entropy(0.0), subtree_entropy(0.0) {};

            /**
             * Destructor
             */
           virtual ~EntDNode() = default;

            /**
             * Visit function. 
             * 
             * Necessry so leaf nodes values of 'num_backups' are reliable
             * 
             * Args:
             *      is_leaf: If this node is a leaf node or not
             */
            void visit_ent(bool is_leaf);

            /**
             * Computes the local and subtree entropy as a backup
             * 
             * Assumes all children are locked.
             * 
             * Args:
             *      children: The children map for this node
             *      policy: The current policy this node would use for action selection
             *      is_opponent: True if this node is acting as an opponent in a two player game.
             */
            void backup_ent_impl(EntCNodeChildMap& children, ActionDistr& policy, bool is_opponent);

            /**
             * Helper to convert children maps into children maps for Ent Nodes.
             * 
             * Templated with the top-level class of the MctsCNode, so that the children can be cast from
             * MctsCNode -> T -> EntCNode
             * 
             * Args:
             *      children: A children map for a MctsDNode, mapping actions to CNodes.
             * 
             * Returns:
             *      A map from actions to EntCNodes, to be used in recommend_action or backup.
             */
            template <typename T>
            std::shared_ptr<EntCNodeChildMap> convert_child_map(const CNodeChildMap& children) const {
                std::shared_ptr<EntCNodeChildMap> ent_children = std::make_shared<EntCNodeChildMap>();
                for (std::pair<std::shared_ptr<const Action>,std::shared_ptr<MctsCNode>> pr : children) {
                    std::shared_ptr<T> superclass_ptr = std::static_pointer_cast<T>(pr.second);
                    std::shared_ptr<EntCNode> ent_node_ptr = std::static_pointer_cast<EntCNode>(superclass_ptr);
                    ent_children->insert_or_assign(pr.first, ent_node_ptr);
                }
                return ent_children;
            }

        public:
            
            /**
             * Interface for calling the backup function for MctsDNode classes subclassing this EntDNode.
             * 
             * Casts the child map so that the EntDNode can use it, and assures that all children are 
             * locked around the backup_impl call.
             * 
             * Templated with the top-level class of the MctsCNode, so that the children can be cast from
             * MctsCNode -> T -> EntCNode.
             * 
             * Args:
             *      children: The children map for a MctsDNode (that are ultimately of type T)
             *      policy: The current policy this node would use for action selection
             *      is_opponent: True if this node is acting as an opponent in a two player game.
             */
            template <typename T>
            void backup_ent(const CNodeChildMap& children, ActionDistr& policy, bool is_opponent=false) {
                std::shared_ptr<EntCNodeChildMap> ent_children = convert_child_map<T>(children);
                for (auto pr : children) pr.second->lock();
                backup_ent_impl(*ent_children, policy, is_opponent);
                for (auto pr : children) pr.second->unlock();
            }
    };
}