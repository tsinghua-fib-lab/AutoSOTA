#include "algorithms/common/ent_decision_node.h"

#include "helper_templates.h"

#include <memory>
#include <unordered_map>

using namespace std;

namespace mcts {
    /**
     * Visit function to update 'num_backups' at child nodes
    */
    void EntDNode::visit_ent(bool is_leaf) {
        if (is_leaf) num_backups++;
    }
    
    /**
     * Updates the values of the entropies.
     * 
     * Computes the local entropy of the policy at this node
     * 
     * In the two player case, subtree_entropy = entropy_player - entropy_opponent, assuming we have it computed at 
     * subnodes, then we need to do the following:
     * 1. compute expected value of child entropy given the current local policy
     * 2. add local entropy (or subtract if we are the opponent)
     * 
     * N.B. with some maths, we could show H = H_local + sum(Pr(a) * H(a)), where:
     * H = subtree entropy
     * H_local = local entropy
     * Pr(a) = prob select action a
     * H(a) = subtree entropy of child node corresponding to action a
    */
    void EntDNode::backup_ent_impl(EntCNodeChildMap& children, ActionDistr& policy, bool is_opponent) {
        // Remember to update num backups
        num_backups++;

        // Compute local entropy 
        local_entropy = 0.0;
        for (pair<shared_ptr<const Action>,double> pr : policy) {
            double prob = pr.second;
            if (prob == 0.0) continue;
            local_entropy -= prob * log(prob);
        }

        // Update subtree entropy == expected child subtree entropies + local
        double opp_coeff = is_opponent ? -1.0 : 1.0;
        subtree_entropy = opp_coeff * local_entropy;
        for (pair<shared_ptr<const Action>,shared_ptr<EntCNode>> pr : children) {
            shared_ptr<const Action> action = pr.first;
            EntCNode& child = (EntCNode&) *pr.second;
            subtree_entropy += policy[action] * child.subtree_entropy;
        }
    }
}
