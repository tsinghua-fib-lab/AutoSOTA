#include "game.h"

// this is simply so that we don't need to do this loop in python
void games_to_nnet_inputs(float* X, float* G, int num_games) {
    int i;
    int input_length = inputLength();
    int game_length = gameLength();
    for (i = 0; i < num_games; i++) {
        inputNetwork(X + i*input_length, G + i*game_length);
    }
}
