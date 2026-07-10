#include <stdlib.h>

// ELEVEN REQUIRED METHODS, GAME STATE IS AN ARRAY OF FLOATS (32bit) (1 DIML), DENOTED 'g' BELOW:
// All input and output arrays will be allocated outside of game.c
// However, one could of course introduce internal arrays to facilitate computations,
// but the primary thought is to use g for all useful data concerning the game state.

const int numActions(void);

const int gameLength(void); // was fullDim()

const int inputLength(void);

// copies root state of the game into memory at address g
void rootState(float* const g);

// playerId returns identity of next player (+1.0 = first, -1.0 = second)
float playerId(const float* const g);

// moving this to an input layer for the network
// writes input for network from game states g to x
// x should be a flattened input for neural network 
void inputNetwork(float* const x, const float* const g);
//void inputNetwork(float* const X, const float* const G, const int num_games);

// says whether game is ended, and also records terminal score
// returns 1 if game is ended, 0 otherwise
int gameEnded(float* const terminal_score, const float* const g);

// returns 1 if action is valid, 0 otherwise
int isValidAction(const float* const g, int const a);

// returns number of valid actions; populates actions with these
int getValidActions(int* const actions, const float* const g);

// records the next state of the game g given action a into ga.
// Returns -1 if action=a is invalid. Returns 1 if ga is terminal, and 0 otherwise.
int nextState(float* const ga, const float* const g, const int a);

void printGame(const float* const g);

