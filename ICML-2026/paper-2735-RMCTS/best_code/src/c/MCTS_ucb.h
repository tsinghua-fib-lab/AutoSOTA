#include <math.h>

#include "dict32.h"

#define GAMEPATH_DEFAULT_LOG2_CAPACITY 5

typedef struct {
  float* root;
  float* game;
  int* action;
  int len_path;
  int log2_capacity;
} GamePath_t;

typedef struct {
  int stacksize; // == len(root_states) == len(num_simulations) == len(PQNv)
	int numSims;
	int stage;
	float* G;
	float* P;
	float* V;
	float* N;
	float* Q;
  float c_puct;
  float* root_states;
  dict32_t* PQNv; // dictionaries of P=policy,Q=quality,N=freq,v=value
  GamePath_t* game_paths;
  int* num_simulations; //how many sims left to go in each lane
  int* requests;
  int len_requests;
  int* lane_needs_inference;
  double time[4];
} MCTS_t;

typedef struct {
	int verbosity;
	MCTS_t* t;
} MCTS_threadargs_t;

float sum_of_float_array(float* a, int len_a);
int argmax_of_float_array(float* a, int len_a);
int maximum_of_int_array(int* a, int len_a);

void GamePath_init(GamePath_t* const gpath);
void GamePath_newroot(GamePath_t* const gpath, const float* const root);
void GamePath_reset(float* g, GamePath_t* const gpath);
void GamePath_endnode(float* const g, const GamePath_t* const gpath);
void GamePath_update(GamePath_t* const gpath, int const a);
void GamePath_free(GamePath_t* const gpath);
void GamePath_randompath(GamePath_t* const gpath);
void GamePath_print(const GamePath_t* gpath);
void GamePath_propagate_value(dict32_t* const d, const GamePath_t* const gpath, float const terminal_score);


void* MCTS_init(int const stacksize,
		int const numSims,
		float* root_states,
		float* G,
		float* P,
		float* V,
		float* N,
		float* Q,
		float const c_puct);

void MCTS_free(void* mcts);

// if return value k is positive, then MCTS_update is not finished, and needs k predictions to be made
// the input for the network prediction is placed into X.
// these predictions should then be put into P (policy) and V (value) arrays,
// and then it should be called again.  It keeps track of the state internally, so that it 
// knows the next thing to do.
// if the return value is 0, then it has finished the update.
int MCTS_update(void* const mcts, int const verbosity, int const num_threads);

void defragment(MCTS_t* t);

void* thread_MCTS_update(void* thread_args_ptr);

void get_time(double* time, void* mcts);

// returns 1 if game state g is found in the given lane; if so, populates P,Q,N,v
int MCTS_lookup(float* PQNv, void* const mcts, int const lane, const float* const g);

