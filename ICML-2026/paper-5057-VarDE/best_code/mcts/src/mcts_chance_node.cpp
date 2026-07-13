#include "mcts_decision_node.h"

#include "helper_templates.h"
#include "mcts_manager.h"
#include "mcts_types.h"

#include <cstddef>
#include <functional>
#include <mutex>
#include <tuple>
#include <utility>

using namespace std;
using namespace mcts;


namespace mcts {
    /**
     * Constructor mostly uses initialisation list. 
     */
    MctsCNode::MctsCNode(
        shared_ptr<MctsManager> mcts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MctsDNode> parent) :
            node_lock(),
            mcts_manager(mcts_manager),
            state(state),
            action(action),
            decision_depth(decision_depth),
            decision_timestep(decision_timestep),
            parent(parent),
            num_visits(0)
    {
    }

    /**
     * Aquires the lock for this node.
     */
    void MctsCNode::lock() { 
        node_lock.lock(); 
    }

    /**
     * Releases the lock for this node.
     */
    void MctsCNode::unlock() { 
        node_lock.unlock(); 
    }

    /**
     * Gets a reference to the lock for this node (so can use in a lock_guard for example)
     */
    std::mutex& MctsCNode::get_lock() { 
        return node_lock; 
    }

    /**
     * Helper function to lock all children nodes.
     */
    void MctsCNode::lock_all_children() const {
        for (auto action_child_pair : children) {
            action_child_pair.second->lock();
        }
    }

    /**
     * Helper function to unlock all children nodes.
     */
    void MctsCNode::unlock_all_children() const {
        for (auto action_child_pair : children) {
            action_child_pair.second->unlock();
        }
    }

    /**
     * Default implementation of visit just increments the number of times visited counter.
     */
    void MctsCNode::visit_itfc(MctsEnvContext& ctx) {
        num_visits += 1;
    }

    /**
     * Wrapper around 'create_child_node_helper' that include logic for using a transposition table.
     * 
     * If child already exists then just return it.
     * 
     * If not using a transposition table, we call the helper and put the child in our children map. 
     * If using a transposition table, we first check the transposition table to try get it from there. If it's not in 
     * the table, we make the child and insert it in children and the transposition table.
     * 
     * Additionally, we protect accessing 'dmap[dnode_id]' with the mutex 'mcts_manager->dmap_mutexes[mutex_indx]' 
     * where 'mutex_indx = hash(dnode_id) % mcts_manager->dmap_mutexes.size()', by locking it using a lock_guard.
     */
    shared_ptr<MctsDNode> MctsCNode::create_child_node_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) 
    {
        if (has_child_node_itfc(observation)) return get_child_node_itfc(observation);

        if (!mcts_manager->use_transposition_table) {
            shared_ptr<MctsDNode> child_node = create_child_node_helper_itfc(observation, next_state);
            children[observation] = child_node;
            return child_node;
        }

        DNodeTable& dmap = mcts_manager->dmap;
        DNodeIdTuple dnode_id = make_tuple(decision_timestep, observation);

        int mutex_indx = 0;
        if (mcts_manager->dmap_mutexes.size() > 1) {
            size_t tpl_hash = hash<DNodeIdTuple>()(dnode_id);
            mutex_indx = tpl_hash % mcts_manager->dmap_mutexes.size();
        }

        lock_guard<mutex> lg(mcts_manager->dmap_mutexes[mutex_indx]);

        auto iter = dmap.find(dnode_id);
        if (iter != dmap.end()) {
            shared_ptr<MctsDNode> child_node = shared_ptr<MctsDNode>(dmap[dnode_id]);
            children[observation] = child_node;
            return child_node;
        }

        shared_ptr<MctsDNode> child_node = create_child_node_helper_itfc(observation, next_state);
        children[observation] = child_node;
        dmap[dnode_id] = child_node;
        return child_node;
    }

    /**
     * Just passes information out of the Mcts manager
     */
    bool MctsCNode::is_two_player_game() const {
        return mcts_manager->is_two_player_game;
    }

    /**
     * In 2 player games, opponent is the agent going second. If the decision timestep is odd, then this node is an 
     * opponent node. (And we can check for oddness by checking last bit of decision timestep).
     */
    bool MctsCNode::is_opponent() const {
        if (!is_two_player_game()) return false;
        return (decision_timestep & 1) == 1;
    }

    /**
     * Number of children = length of children map
     */
    int MctsCNode::get_num_children() const {
        return children.size();
    }

    /**
     * Has child if it's in the children map. Find returns an iterator pointing at the element found, or the 'end' 
     * iterator if it is not in the map. So if the returned iterator == children.end() then a child doesn't exist for 
     * that action in the children map.
     */
    bool MctsCNode::has_child_node_itfc(shared_ptr<const Observation> observation) const {
        auto iterator = children.find(observation);
        return iterator != children.end();
    }
    
    /**
     * Just looks up observation in 'children' map.
     */
    shared_ptr<MctsDNode> MctsCNode::get_child_node_itfc(shared_ptr<const Observation> observation) const {
        return children.at(observation);
    }

    /**
     * Returns a pretty printing of the node as a string. This is just a wrapper around the helper function. 
     * 
     * The helper function uses a depth with respect to the tree, rather than decision depth, hence why it is multiplied
     * by two (and plus one). A decision depth of zero would print out the child decision nodes still.
     */
    string MctsCNode::get_pretty_print_string(int depth) const {   
        int num_tabs = 0;
        stringstream ss;
        get_pretty_print_string_helper(ss, 2*depth+1, num_tabs);
        return ss.str();
    }

    /**
     * Recursively pretty prints a tree and the values given by nodes 'get_pretty_print_val' functions.
     * 
     * Should be a one-and-done function that can be reused. It's pretty much all just building a string that lays out 
     * nodes 'get_pretty_print_val' values in a nice format. 
     * 
     * TODO: add nice way of only displaying the X most sampled outcomes
     */
    void MctsCNode::get_pretty_print_string_helper(stringstream& ss, int depth, int num_tabs) const {
        // Print out this nodes info
        ss << "C(vl=" << get_pretty_print_val() << ",#v=" << num_visits << ")[";

        // print out child trees recursively
        for (const pair<const shared_ptr<const Observation>,shared_ptr<MctsDNode>>& key_val_pair : children) {
            const Observation& observation = *(key_val_pair.first);
            MctsDNode& child_node = *(key_val_pair.second);
            ss << "\n";
            for (int i=0; i<num_tabs+1; i++) ss << "|\t";
            ss << "{" << observation << "}->";
            child_node.get_pretty_print_string_helper(ss, depth-1, num_tabs+1);
        }

        // Print out closing bracket
        ss << "\n";
        for (int i=0; i<num_tabs; i++) ss << "|\t";
        ss << "],";
    }
}