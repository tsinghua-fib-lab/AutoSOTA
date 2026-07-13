#include "algorithms/common/emp_node.h"

#include "helper_templates.h"

#include <limits>
#include <memory>
#include <unordered_map>

using namespace std;

namespace mcts {
    /**
     * Visit function to update 'num_backups' at child nodes
    */
    void EmpNode::visit_emp(bool is_leaf) {
        if (is_leaf) num_backups++;
    }
    
    /**
     * Builds a map of actions to q-values for actions that do and do not meet the 'recommend_visit_threshold'. 
     * And then recommends the max from the thresholded map, and from the unthresholded map if the thresholded one is 
     * empty.
     */
    shared_ptr<const Action> EmpNode::recommend_action_best_emp_value_impl(
        EmpNodeChildMap& children, RandManager& rand_manager, int visit_threshold, bool is_opponent) const 
    {
        double opp_coeff = is_opponent ? -1.0 : 1.0;
        unordered_map<shared_ptr<const Action>, double> avg_returns_thresholded;
        unordered_map<shared_ptr<const Action>, double> avg_returns;
        for (pair<shared_ptr<const Action>,shared_ptr<EmpNode>> pr : children) {
            shared_ptr<const Action> action = pr.first;
            EmpNode& child = *pr.second;
            if (child.num_backups >= visit_threshold) {
                avg_returns_thresholded[action] = opp_coeff * child.avg_return;
            } else {
                avg_returns[action] = opp_coeff * child.avg_return;
            }
        }

        if (avg_returns_thresholded.size() > 0) {
            return helper::get_max_key_break_ties_randomly(avg_returns_thresholded, rand_manager);
        }
        return helper::get_max_key_break_ties_randomly(avg_returns, rand_manager);
    }
    
    /**
     * Computes running average.
     */
    void EmpNode::backup_emp(double _return) {
        num_backups++;
        avg_return += (_return - avg_return) / (double) num_backups;
    }
}