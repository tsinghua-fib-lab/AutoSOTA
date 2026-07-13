#include "algorithms/varde/varde_decision_node.h"

#include "helper_templates.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace std;

namespace mcts {
	RvipDNode::RvipDNode(
		shared_ptr<RvipManager> mcts_manager,
		shared_ptr<const State> state,
		int decision_depth,
		int decision_timestep,
		shared_ptr<const RvipCNode> parent) :
			MctsDNode(
				static_pointer_cast<MctsManager>(mcts_manager),
				state,
				decision_depth,
				decision_timestep,
				static_pointer_cast<const MctsCNode>(parent)),
			DPDNode(heuristic_value),
			num_backups(0),
			avg_return(heuristic_value),
			actions(mcts_manager->mcts_env->get_valid_actions_itfc(state))
	{
	}

	void RvipDNode::ensure_actions_initialized() {
		if (actions == nullptr) {
			actions = mcts_manager->mcts_env->get_valid_actions_itfc(state);
		}
	}

	double RvipDNode::compute_influence_weight(shared_ptr<const Action> action) const {
		RvipManager& manager = (RvipManager&) *mcts_manager;
		double temp = manager.temp;

		double max_scaled = -numeric_limits<double>::infinity();
		lock_all_children();
		for (shared_ptr<const Action> act : *actions) {
			double q_est;
			if (has_child_node(act)) {
				q_est = manager.use_dp_value ? get_child_node(act)->dp_value : get_child_node(act)->avg_return;
			} else {
				q_est = manager.use_dp_value ? dp_value : avg_return;
			}
			double scaled = q_est / temp;
			if (scaled > max_scaled) max_scaled = scaled;
		}

		double numerator = 0.0;
		double denom = 0.0;
		for (shared_ptr<const Action> act : *actions) {
			double q_est;
			if (has_child_node(act)) {
				q_est = manager.use_dp_value ? get_child_node(act)->dp_value : get_child_node(act)->avg_return;
			} else {
				q_est = manager.use_dp_value ? dp_value : avg_return;
			}
			double weight = exp(q_est / temp - max_scaled);
			denom += weight;
			if (act == action) {
				numerator = weight;
			}
		}
		unlock_all_children();

		if (denom <= 0.0) return 0.0;
		return numerator / denom;
	}

	double RvipDNode::compute_variance_reduction_score(shared_ptr<const Action> action) const {
		double influence = compute_influence_weight(action);
		double variance = numeric_limits<double>::infinity();
		if (has_child_node(action)) {
			variance = get_child_node(action)->get_variance();
		}
		return influence * variance;
	}

	bool RvipDNode::has_child_node(shared_ptr<const Action> action) const {
		return MctsDNode::has_child_node_itfc(static_pointer_cast<const Action>(action));
	}

	shared_ptr<RvipCNode> RvipDNode::get_child_node(shared_ptr<const Action> action) const {
		shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
		shared_ptr<MctsCNode> new_child = MctsDNode::get_child_node_itfc(act_itfc);
		return static_pointer_cast<RvipCNode>(new_child);
	}

	shared_ptr<RvipCNode> RvipDNode::create_child_node(shared_ptr<const Action> action) {
		shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
		shared_ptr<MctsCNode> new_child = MctsDNode::create_child_node_itfc(act_itfc);
		return static_pointer_cast<RvipCNode>(new_child);
	}

	void RvipDNode::visit(MctsEnvContext& ctx) {
		MctsDNode::visit_itfc(ctx);
		DPDNode::visit_dp(is_leaf());
		if (is_leaf()) {
			num_backups++;
		}
	}

	shared_ptr<const Action> RvipDNode::select_action_varde(MctsEnvContext& ctx) {
		(void)ctx;
		ensure_actions_initialized();

		vector<shared_ptr<const Action>> unexpanded;
		for (shared_ptr<const Action> action : *actions) {
			if (!has_child_node(action)) {
				unexpanded.push_back(action);
			}
		}

		if (!unexpanded.empty()) {
			int idx = mcts_manager->get_rand_int(0, static_cast<int>(unexpanded.size()));
			shared_ptr<const Action> action = unexpanded[idx];
			create_child_node(action);
			return action;
		}

		RvipManager& manager = (RvipManager&) *mcts_manager;
		double temp = manager.temp;

		unordered_map<shared_ptr<const Action>, double> scaled_q_values;
		double max_scaled = -numeric_limits<double>::infinity();

		lock_all_children();
		for (shared_ptr<const Action> action : *actions) {
			shared_ptr<RvipCNode> child = get_child_node(action);
			double q_est = manager.use_dp_value ? child->dp_value : child->avg_return;
			double scaled = q_est / temp;
			scaled_q_values[action] = scaled;
			if (scaled > max_scaled) max_scaled = scaled;
		}

		double sum_weights = 0.0;
		for (auto& pr : scaled_q_values) {
			double weight_raw = exp(pr.second - max_scaled);
			pr.second = weight_raw;
			sum_weights += weight_raw;
		}

		unordered_map<shared_ptr<const Action>, double> scores;
		for (shared_ptr<const Action> action : *actions) {
			double influence = (sum_weights > 0.0) ? scaled_q_values[action] / sum_weights : 0.0;
			double raw_variance = get_child_node(action)->get_variance();
			double effective_variance = std::max(raw_variance, manager.variance_floor);
			
			int n = get_child_node(action)->num_visits;
			double exploration_bonus = 1.0 / (static_cast<double>(n) * static_cast<double>(n + 1));
			
			scores[action] = influence * effective_variance * exploration_bonus;
		}
		unlock_all_children();

		return helper::get_max_key_break_ties_randomly(scores, *mcts_manager);
	}

	shared_ptr<const Action> RvipDNode::select_action(MctsEnvContext& ctx) {
		return select_action_varde(ctx);
	}

	shared_ptr<const Action> RvipDNode::recommend_action(MctsEnvContext& ctx) const {
		(void)ctx;
		double opp_coeff = is_opponent() ? -1.0 : 1.0;
		unordered_map<shared_ptr<const Action>, double> action_values;

		for (shared_ptr<const Action> action : *actions) {
			if (!has_child_node(action)) continue;
			shared_ptr<RvipCNode> child = get_child_node(action);
			double val_est = ((RvipManager&)*mcts_manager).use_dp_value ? child->dp_value : child->avg_return;
			action_values[action] = opp_coeff * val_est;
		}

		if (action_values.empty()) {
			int index = mcts_manager->get_rand_int(0, actions->size());
			return actions->at(index);
		}

		return helper::get_max_key_break_ties_randomly(action_values, *mcts_manager);
	}

	void RvipDNode::backup(
		const vector<double>& trial_rewards_before_node,
		const vector<double>& trial_rewards_after_node,
		const double trial_cumulative_return_after_node,
		const double trial_cumulative_return,
		MctsEnvContext& ctx)
	{
		(void)trial_rewards_before_node;
		(void)trial_rewards_after_node;
		(void)trial_cumulative_return;
		(void)ctx;

		backup_dp<RvipCNode>(children, is_opponent());

		num_backups++;
		double delta = trial_cumulative_return_after_node - avg_return;
		avg_return += delta / static_cast<double>(num_backups);
	}

	shared_ptr<RvipCNode> RvipDNode::create_child_node_helper(shared_ptr<const Action> action) const {
		return make_shared<RvipCNode>(
			static_pointer_cast<RvipManager>(MctsDNode::mcts_manager),
			state,
			action,
			decision_depth,
			decision_timestep,
			static_pointer_cast<const RvipDNode>(shared_from_this()));
	}

	string RvipDNode::get_pretty_print_val() const {
		stringstream ss;
		ss << dp_value << "(mean:" << avg_return << ")";
		return ss.str();
	}
}

namespace mcts {
	void RvipDNode::visit_itfc(MctsEnvContext& ctx) {
		MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
		visit(ctx_itfc);
	}

	shared_ptr<const Action> RvipDNode::select_action_itfc(MctsEnvContext& ctx) {
		MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
		return select_action(ctx_itfc);
	}

	shared_ptr<const Action> RvipDNode::recommend_action_itfc(MctsEnvContext& ctx) const {
		MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
		return recommend_action(ctx_itfc);
	}

	void RvipDNode::backup_itfc(
		const vector<double>& trial_rewards_before_node,
		const vector<double>& trial_rewards_after_node,
		const double trial_cumulative_return_after_node,
		const double trial_cumulative_return,
		MctsEnvContext& ctx)
	{
		MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
		backup(
			trial_rewards_before_node,
			trial_rewards_after_node,
			trial_cumulative_return_after_node,
			trial_cumulative_return,
			ctx_itfc);
	}

	shared_ptr<MctsCNode> RvipDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
		shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
		shared_ptr<RvipCNode> child_node = create_child_node_helper(act_itfc);
		return static_pointer_cast<MctsCNode>(child_node);
	}
}
