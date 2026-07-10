#include <stdlib.h>
#include <string.h>
#include "random.h"
#include "game.h"

uint64_t Knuth_seed = 0ull;

void Knuth_init(uint64_t seed) {
  int i;
  Knuth_seed = seed;
  for(i=0;i<20;i++) {
    Knuth_seed *= Knuth_a;
    Knuth_seed += Knuth_c;
  }
}

uint32_t Knuth_lrand(void) {
  Knuth_seed *= Knuth_a;
  Knuth_seed += Knuth_c;
  return (uint32_t) (Knuth_seed >> 32);
}

float Knuth_drand(void) {
  Knuth_seed *= Knuth_a;
  Knuth_seed += Knuth_c;
  return ((float) (Knuth_seed >> 40)) / ((float) (1u << 24));
}

//assumes that p is a probability distribution
// if not, then there are no guarantees!
int random_index(const float* const p, int const len_p) {
  float s = 0.0;
  int i;
  float x;  
  x = Knuth_drand();
  for(i=0;i<len_p;i++) {
    s += p[i];
    if(s >= x) return i;
  }
  return len_p - 1;
}


void random_game_state(float* const g) {
  int len_gamestate = gameLength();
  int n = numActions();
  float* G = (float*) malloc(100*len_gamestate*sizeof(float));
  float* h = (float*) malloc(len_gamestate*sizeof(float));
  int* actions = (int*) malloc(n * sizeof(int));
  int num_actions;
  float terminal_score;
  int i,j;
  int a;

  rootState(h);
  memcpy(G,h,len_gamestate*sizeof(float));
  i = 0;
  while(!gameEnded(&terminal_score,h) && (i+1)<100) {
    num_actions = getValidActions(actions,h);
    if(num_actions == 0) break;
    a = actions[Knuth_lrand() % num_actions];
    nextState(G+(i+1)*len_gamestate,h,a);
    memcpy(h, G + (i+1)*len_gamestate, len_gamestate*sizeof(float));
    i++;
  }
  j = Knuth_lrand() % (i+1);
  memcpy(g, G+j*len_gamestate, len_gamestate*sizeof(float));
  free(G);
  free(h);
  free(actions);
}
