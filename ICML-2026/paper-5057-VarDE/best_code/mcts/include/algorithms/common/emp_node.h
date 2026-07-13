#pragma once

#include "mcts_types.h"
#include "mcts_chance_node.h"
#include "mcts_decision_node.h"


namespace mcts {
    // forward declare EmpNode class
    class EmpNode;

    // Typedef for children map
    typedef std::unordered_map<std::shared_ptr<const Action>, std::shared_ptr<EmpNode>> EmpNodeChildMap;

    /**
     * An implementation of empircal average return backups for nodes to use. Note that the behaviour at decision and 
     * chance nodes is the same, so we can actually just have one implementation of this. The chance node's using 
     * this can just ignore the action recommendation functions, as they are nonsensical for it anyway.
     * 
     * Member variables:
     *      num_backups: 
     *          The number of backups this node has performed (== "number of visits" with respect to dp backup)
     *      avg_return: 
     *          The average return from this node
     */
    class EmpNode {
        protected:
            int num_backups;
            double avg_return;

            /**
             * Constructor 
             */
            EmpNode(int psuedo_trials=0, double heuristic_value=0.0) : 
                num_backups(psuedo_trials), avg_return(heuristic_value) {};

            /**
             * Destructor
             */
           virtual ~EmpNode() = default;

            /**
             * Visit function. 
             * 
             * Necessry so leaf nodes values of 'num_backups' are reliable
             * 
             * Args:
             *      is_leaf: If this node is a leaf node or not
             */
            void visit_emp(bool is_leaf);

            /**
             * Returns an action recommendation from this node.
             * 
             * Assumes all children are locked. 
             * 
             * And assumes that children is not empty.
             * 
             * Args:
             *      children: The children map for this node
             *      rand_manager: Manager for RNG for breaking ties
             *      visit_threshold: 
             *          A threshold value of visits to recommend a child node (if no child has met the threshold then 
             *          we still recommend the highest value ignoring any thresholding).
             *      is_opponent: True if this node is acting as an opponent in a two player game.
             * 
             * Returns:
             *      An action recommendation from this node.
             */
            std::shared_ptr<const Action> recommend_action_best_emp_value_impl(
                EmpNodeChildMap& children, RandManager& rand_manager, int visit_threshold, bool is_opponent) const;

            /**
             * Updates the average return given a return from this node from a trial
             * 
             * Args:
             *      _return: The return from this trial 
             */
            void backup_emp(double _return);

            /**
             * Helper to convert children maps into children maps for Emp Nodes.
             * 
             * Templated with the top-level class of the MctsCNode, so that the children can be cast from
             * MctsCNode -> T -> EmpNode
             * 
             * Args:
             *      children: A children map for a MctsDNode, mapping actions to CNodes.
             * 
             * Returns:
             *      A map from actions to EmpNodes, to be used in recommend_action or backup.
             */
            template <typename T>
            std::shared_ptr<EmpNodeChildMap> convert_child_map(const CNodeChildMap& children) const {
                std::shared_ptr<EmpNodeChildMap> emp_children = std::make_shared<EmpNodeChildMap>();
                for (std::pair<std::shared_ptr<const Action>,std::shared_ptr<MctsCNode>> pr : children) {
                    std::shared_ptr<T> superclass_ptr = std::static_pointer_cast<T>(pr.second);
                    std::shared_ptr<EmpNode> emp_node_ptr = std::static_pointer_cast<EmpNode>(superclass_ptr);
                    emp_children->insert_or_assign(pr.first, emp_node_ptr);
                }
                return emp_children;
            }

        public:
            /**
             * Interface for calling the recommend_action function for MctsDNode classes subclassing this EmpNode.
             * 
             * Casts the child map so that the EmpNode can use it, and assures that all children are 
             * locked around the backup_impl call.
             * 
             * Templated with the top-level class of the MctsCNode, so that the children can be cast from
             * MctsCNode -> T -> EmpNode.
             * 
             * Args:
             *      children: The children map for a MctsDNode (that are ultimately of type T)
             *      rand_manager: Manager for RNG for breaking ties
             *      visit_threshold: 
             *          A threshold value of visits to recommend a child node (if no child has met the threshold then 
             *          we still recommend the highest value ignoring any thresholding).
             *      is_opponent: True if this node is acting as an opponent in a two player game.
             * 
             * Returns:
             *      An action recommendation from this node.
             */
            template <typename T>
            std::shared_ptr<const Action> recommend_action_best_emp_value(
                const CNodeChildMap& children, 
                RandManager& rand_manager, 
                int visit_threshold=0, 
                bool is_opponent=false) const 
            {
                std::shared_ptr<EmpNodeChildMap> emp_children = convert_child_map<T>(children);
                for (auto pr : children) pr.second->lock();
                std::shared_ptr<const Action> action = recommend_action_best_emp_value_impl(
                    *emp_children, rand_manager, visit_threshold, is_opponent);
                for (auto pr : children) pr.second->unlock();
                return action;
            }
    };
}