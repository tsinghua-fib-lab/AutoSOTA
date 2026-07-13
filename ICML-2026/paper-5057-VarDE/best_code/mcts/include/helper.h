#pragma once

#include "mcts_types.h"

#include <memory>

// forward declar MctsEnv
namespace mcts {
    class MctsEnv;
}

namespace mcts::helper {
    /**
     * A default heuristic function that returns a constant zero
     */
    double zero_heuristic_fn(std::shared_ptr<const State> state, std::shared_ptr<MctsEnv> env=nullptr);
}