#include "algorithms/common/ent_chance_node.h"

#include "helper_templates.h"

#include <memory>
#include <unordered_map>

using namespace std;

namespace mcts {

    /**
     * Entropy = expected value of child entropies (i.e. empirical average)
     * 
     * Adapted from DPDNode DPBackup function
    */
    void EntCNode::backup_ent_impl(EntDNodeChildMap& children) {
        num_backups++;

        subtree_entropy = 0.0;
        double sum_child_backups = 0;
        for (pair<shared_ptr<const Observation>,shared_ptr<EntDNode>> pr : children) {
            EntDNode& child = (EntDNode&) *pr.second;
            int child_backups = child.num_backups;
            if (child_backups == 0) continue;
            sum_child_backups += child_backups;
            subtree_entropy *= (sum_child_backups - child_backups) / sum_child_backups;
            subtree_entropy += child_backups * child.subtree_entropy / sum_child_backups; 
        }
    }
}