#include "algorithms/varde/varde_chance_node.h"

#include "helper_templates.h"

#include <limits>
#include <memory>
#include <sstream>
#include <vector>

using namespace std;

namespace mcts {
	RvipCNode::RvipCNode(
		shared_ptr<RvipManager> mcts_manager,
		shared_ptr<const State> state,
		shared_ptr<const Action> action,
		int decision_depth,
		int decision_timestep,
		shared_ptr<const RvipDNode> parent) :
			MctsCNode(
				static_pointer_cast<MctsManager>(mcts_manager),
				state,
				action,
				decision_depth,
				decision_timestep,
				static_pointer_cast<const MctsDNode>(parent)),
			DPCNode(),
			num_backups(0),
			avg_return(0.0),
			m2_return(0.0),
			local_reward(mcts_manager->mcts_env->get_reward_itfc(state, action)),
			next_state_distr(mcts_manager->mcts_env->get_transition_distribution_itfc(state, action))
	{
	}

	void RvipCNode::visit(MctsEnvContext& ctx) {
		MctsCNode::visit_itfc(ctx);
	}

	shared_ptr<const State> RvipCNode::sample_observation_random() {
		shared_ptr<const State> sampled_state = helper::sample_from_distribution(*next_state_distr, *mcts_manager);
		if (!has_child_node(sampled_state)) {
			create_child_node(sampled_state);
		}
		return sampled_state;
	}

	shared_ptr<const State> RvipCNode::sample_observation(MctsEnvContext& ctx) {
		return sample_observation_random();
	}

	void RvipCNode::backup(
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

		num_backups++;
		double delta = trial_cumulative_return_after_node - avg_return;
		avg_return += delta / static_cast<double>(num_backups);
		double delta2 = trial_cumulative_return_after_node - avg_return;
		m2_return += delta * delta2;

		backup_dp<RvipDNode>(children, local_reward, is_opponent());
	}

	shared_ptr<RvipDNode> RvipCNode::create_child_node_helper(
		shared_ptr<const State> observation, shared_ptr<const State> next_state) const
	{
		shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
		(void)next_state;
		return make_shared<RvipDNode>(
			static_pointer_cast<RvipManager>(mcts_manager),
			mdp_next_state,
			decision_depth + 1,
			decision_timestep + 1,
			static_pointer_cast<const RvipCNode>(shared_from_this()));
	}

	string RvipCNode::get_pretty_print_val() const {
		stringstream ss;
		ss << dp_value << "(mean:" << avg_return << ",var:" << get_variance() << ")";
		return ss.str();
	}

	double RvipCNode::get_variance() const {
		if (num_backups < 2) return numeric_limits<double>::infinity();
		return m2_return / static_cast<double>(num_backups - 1);
	}

	shared_ptr<RvipDNode> RvipCNode::create_child_node(
		shared_ptr<const State> observation, shared_ptr<const State> next_state)
	{
		shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
		shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
		shared_ptr<MctsDNode> new_child = MctsCNode::create_child_node_itfc(obsv_itfc, next_state_itfc);
		return static_pointer_cast<RvipDNode>(new_child);
	}

	bool RvipCNode::has_child_node(shared_ptr<const State> observation) const {
		return MctsCNode::has_child_node_itfc(static_pointer_cast<const Observation>(observation));
	}

	shared_ptr<RvipDNode> RvipCNode::get_child_node(shared_ptr<const State> observation) const {
		shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
		shared_ptr<MctsDNode> new_child = MctsCNode::get_child_node_itfc(obsv_itfc);
		return static_pointer_cast<RvipDNode>(new_child);
	}
}

namespace mcts {
	void RvipCNode::visit_itfc(MctsEnvContext& ctx) {
		MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
		visit(ctx_itfc);
	}

	shared_ptr<const Observation> RvipCNode::sample_observation_itfc(MctsEnvContext& ctx) {
		MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
		shared_ptr<const State> obsv = sample_observation(ctx_itfc);
		return static_pointer_cast<const Observation>(obsv);
	}

	void RvipCNode::backup_itfc(
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

	shared_ptr<MctsDNode> RvipCNode::create_child_node_helper_itfc(
		shared_ptr<const Observation> observation,
		shared_ptr<const State> next_state) const
	{
		shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
		shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
		shared_ptr<RvipDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
		return static_pointer_cast<MctsDNode>(child_node);
	}
}
