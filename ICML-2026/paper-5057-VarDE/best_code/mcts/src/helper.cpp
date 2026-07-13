#include "helper.h"

using namespace std;

namespace mcts::helper {
    /**
     * Implementation of the default zero heuristic function.
     */
    double zero_heuristic_fn(shared_ptr<const State> state, shared_ptr<MctsEnv> env) {
        return 0.0;
    }
}