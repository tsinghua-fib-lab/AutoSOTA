#include "algorithms/common/dp_decision_node.h"

#include "helper_templates.h"

#include <limits>
#include <memory>
#include <unordered_map>

using namespace std;

namespace mcts {
    /**
     * Visit function to update 'num_backups' at child nodes
    */
    void DPDNode::visit_dp(bool is_leaf) {
        if (is_leaf) num_backups++;
    }

    /**
     * Builds a map of actions to q-values for actions that do and do not meet the 'recommend_visit_threshold'. 
     * And then recommends the max from the thresholded map, and from the unthresholded map if the thresholded one is 
     * empty.
     */
    shared_ptr<const Action> DPDNode::recommend_action_best_dp_value_impl(
        DPCNodeChildMap& children, RandManager& rand_manager, int visit_threshold, bool is_opponent) const 
    {
        double opp_coeff = is_opponent ? -1.0 : 1.0;
        unordered_map<shared_ptr<const Action>, double> dp_values_thresholded;
        unordered_map<shared_ptr<const Action>, double> dp_values;
        for (pair<shared_ptr<const Action>,shared_ptr<DPCNode>> pr : children) {
            shared_ptr<const Action> action = pr.first;
            DPCNode& child = *pr.second;
            if (child.num_backups >= visit_threshold) {
                dp_values_thresholded[action] = opp_coeff * child.dp_value;
            } else {
                dp_values[action] = opp_coeff * child.dp_value;
            }
        }

        if (dp_values_thresholded.size() > 0) {
            return helper::get_max_key_break_ties_randomly(dp_values_thresholded, rand_manager);
        }
        return helper::get_max_key_break_ties_randomly(dp_values, rand_manager);
    }
    
    /**
     * Implements a dp backup for decision nodes.
     * 
     * I.e. V(s) = max_a Q(s,a).
     * 
     * Note that when an opponent, actually want to have V(s) = min_a Q(s,a). Multiplying by 'opp_coeff' in the 
     * if statement means it passes with opp_coeff==1, when its max so far, and with opp_coeff==-1, when its min so far.
     * Also note that we want opp_coeff * dp_value == -inf after initialisation, so we set dp_value initially to 
     * opp_coeff * -inf.
     * 
     * Continue when child number of backups == 0. Consider if using costs, then all returns negative, but value of 
     * chance node initialised to 0. In cuncurrent settings, we may erroneously backup a zero. Alternatively, we may 
     * accidentally erase heuristic values that we wanted to use in concurrent settings (which is why this line was 
     * added originally).
     */
    void DPDNode::backup_dp_impl(DPCNodeChildMap& children, bool is_opponent) {
        double opp_coeff = is_opponent ? -1.0 : 1.0;
        dp_value = opp_coeff * -numeric_limits<double>::infinity();

        for (pair<shared_ptr<const Action>,shared_ptr<DPCNode>> pr : children) {
            DPCNode& child = *pr.second;
            if (child.num_backups == 0) continue;
            if (opp_coeff * child.dp_value > opp_coeff * dp_value) {
                dp_value = child.dp_value;
            }
        }

        num_backups++;
    }
}