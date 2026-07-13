#include "algorithms/common/dp_chance_node.h"

#include "helper_templates.h"

#include <limits>
#include <memory>
#include <unordered_map>

using namespace std;

namespace mcts {
    /**
     * Implements a dp backup for decision nodes.
     * 
     * I.e. Q(s,a) = R(s,a) + E_{s'}[V(s')]
     * 
     * Implemented as a running average to avoid using two loops. Feels like it should be more efficient, but would be 
     * nice to check at some point if really optimising.
     * 
     * Made 'sum_child_backups' a double to force computations to all be floating point rather than integer (otherwise 
     * get an integer div).
     * 
     * Additional note on concurrency fun. It is possible and valid to currently have a child with zero backups. 
     * Consider if we have another trial that also searches this node, it made a new child, but hasn't backed it up 
     * yet. Hence it's necessary to include the line "if (child.num_backups == 0) continue;" to avoid a division by 
     * zero causing NaNs.
     */
    void DPCNode::backup_dp_impl(DPDNodeChildMap& children, double local_reward, bool is_opponent) {
        dp_value = 0.0;
        double sum_child_backups = 0;
        for (pair<shared_ptr<const Observation>,shared_ptr<DPDNode>> pr : children) {
            DPDNode& child = (DPDNode&) *pr.second;
            if (child.num_backups == 0) continue;
            sum_child_backups += child.num_backups;
            dp_value *= (sum_child_backups - child.num_backups) / sum_child_backups;
            dp_value += child.num_backups * child.dp_value / sum_child_backups; 
        }
        dp_value += local_reward; // +R(s,a)

        num_backups++;
    }
}