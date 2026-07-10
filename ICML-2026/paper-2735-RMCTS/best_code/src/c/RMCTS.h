#include <stdint.h>
#include <math.h>

// #include "dict32.h"

void select_chunksizes(int* chunksizes, int budget, float* pi, int n);
void new_policy_common_ucb_Newton(float* pi1, int n, float* Q, float c, float* pi0, int T);

typedef struct {
	int num_lanes;
	int numSims;
	float c_puct;

	float* new_policy;
	float* new_value;	
	
	float* G; //game state
	float* policy;
	float* value;
	float* Q;
	float* N;
	int32_t* parent;
	int32_t* a0;
	int32_t* sims;
	int32_t* sims_remaining;
	int32_t* inference_stack;
	int32_t* inference_stack_size;
	int32_t* new_stack;
	int32_t* new_stack_size;
	int32_t* num_completed;
	int32_t* row_count;
} MCTS_new_t;

// all the array data is allocated in python
// partly because we want to see this data in python
void* MCTS_init(int const num_lanes,  
				int const numSims,
				float const c_puct,
				float* new_policy,
				float* new_value,
				float* G,
				float* policy,
				float* value,
				float* Q,
				float* N,
				int32_t* parent,
				int32_t* a0,
				int32_t* sims,
				int32_t* sims_remaining,
				int32_t* inference_stack,
				int32_t* inference_stack_size,
				int32_t* new_stack,
				int32_t* new_stack_size,
				int32_t* num_completed,
				int32_t* row_count);


void MCTS_free(void* mcts);


// Stage 1: completely flushes the new_stack and populates the inference_stack
// with whatever inferences are required. Terminal states are stored as rows
// in the tree but are not propagated. Leaf nodes (sims==1) are also left
// in place. No propagation occurs during Stage 1.
void MCTS_flush_new_stack(void* const mcts);

// Stage 2: propagate values bottom-up through the completed tree.
// Call this once after the tree is fully built (inference_stack is empty).
void MCTS_propagate_all(void* const mcts);


