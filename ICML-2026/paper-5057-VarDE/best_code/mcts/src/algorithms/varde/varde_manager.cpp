#include "algorithms/varde/varde_manager.h"

#include "algorithms/varde/varde_chance_node.h"
#include "algorithms/varde/varde_decision_node.h"

#include <stdexcept>

using namespace std;

namespace mcts {
	RvipManager::RvipManager(RvipManagerArgs args) : 
		MctsManager(args),
		temp(args.temp),
			variance_floor(args.variance_floor),
			use_dp_value(args.use_dp_value) {}

	shared_ptr<MctsDNode> RvipManager::create_decision_node_helper(
		shared_ptr<const State> /*state*/,
		int /*decision_depth*/,
		int /*decision_timestep*/,
		shared_ptr<const MctsCNode> /*parent*/)
	{
		throw runtime_error("Use explicit VarDE node construction for now");
	}

	shared_ptr<MctsCNode> RvipManager::create_chance_node_helper(
		shared_ptr<const State> /*state*/,
		shared_ptr<const Action> /*action*/,
		int /*decision_depth*/,
		int /*decision_timestep*/,
		shared_ptr<const MctsDNode> /*parent*/)
	{
		throw runtime_error("Use explicit VarDE node construction for now");
	}
}
