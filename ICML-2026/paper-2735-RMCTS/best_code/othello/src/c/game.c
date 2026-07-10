#include <stdio.h>
#include <assert.h>
#include <string.h>

#include <game.h>

#define MY_ABS(X) (((X) < 0) ? (-X) : (X))

#define N 8
#define B 64
#define PLAYER B /* pieces, then totals for all 4 lines: E, NE, N, NW */
#define ENDED B+1
#define TERMINALSCORE B+2
#define NUMACTIONS B
#define GAMESIZE (B+3)
#define INPUTSIZE B // board, signed wrt player

#define MYMAX(x,y) (((x) >= (y)) ? (x) : (y))

const int numActions(void) {
  /* square board */
  return NUMACTIONS;
}

const int gameLength(void) {
  return GAMESIZE;
}

const int inputLength(void) {
  return INPUTSIZE; 
}

// Othello board layout:
//     0      1      2  ...    7
//    ... 
//    56     57     58  ...   63        

void rootState(float* const g)
{
  memset(g, 0, GAMESIZE*sizeof(float));
  g[PLAYER] = 1.0; // x (black)
  g[27] = -1.0;
  g[28] = 1.0;
  g[35] = 1.0;
  g[36] = -1.0;
}

void inputNetwork(float* const x, const float* const g) 
{
  int j;
  
  memcpy(x, g, B*sizeof(float));
  if(g[PLAYER] < 0.0) {
    /* flip sign */
    for(j=0;j<B;j++) x[j] = -x[j];
  }
  
}

float playerId(const float* const g)
{
  return g[PLAYER];
}

int gameEnded(float* const terminal_score, const float* const g)
{
  *terminal_score = g[TERMINALSCORE];
  return (int) g[ENDED];
}

int isValidAction(const float* const g, int const a) {
  assert((0<=a) && (a < NUMACTIONS));
  float p = g[PLAYER];

  if(g[a] != 0.0) return 0;

  int i,j,ii,jj,aa,l;

  // E, NE, N, NW, W, SW, S, SE
  int dx[] = {1,1,0,-1,-1,-1,0,1};
  int dy[] = {0,1,1,1,0,-1,-1,-1};

  i = a >> 3;
  j = a % 8;

  for(l=0;l<8;l++) {
    ii = i+dx[l];
    jj = j+dy[l];
    if((ii>=0) && (ii < 8) && (jj >= 0) && (jj < 8)) {
      aa = 8*ii+jj;
      if(g[aa] != -p) continue;
      ii += dx[l];
      jj += dy[l];
      while((ii>=0) && (ii < 8) && (jj >= 0) && (jj < 8)) {
        aa = 8*ii + jj;
        if(g[aa] == p) return 1;
        if(g[aa] == 0.0) break;
        ii += dx[l];
        jj += dy[l];
      }
    }
  }
  return 0;
}

int getValidActions(int* const actions, const float* const g)
{
  int a;
  int count = 0;
  for(a=0;a<NUMACTIONS;a++) {
    if(isValidAction(g,a)) {
      actions[count] = a;
      count++;
    }
  }
  return count;
}

//returns 1 if next state is terminal, 0 otherwise; returns -1 if action=a is not valid
int nextState(float* const ga, const float* const g, const int a)
{
  float p = g[PLAYER];

  assert((a >= 0) && (a < NUMACTIONS));

  if(!isValidAction(g,a)) return -1;

  //first copy g into ga
  memcpy(ga, g, GAMESIZE*sizeof(float));

  // set player p stone down on the board at position a
  ga[a] = p;

  int i,j,ii,jj,aa,l;

  // E, NE, N, NW, W, SW, S, SE
  int dx[] = {1,1,0,-1,-1,-1,0,1};
  int dy[] = {0,1,1,1,0,-1,-1,-1};

  int flip;

  i = a >> 3;
  j = a % 8;

  for(l=0;l<8;l++) {
    flip=0;
    ii = i+dx[l];
    jj = j+dy[l];
    if((ii>=0) && (ii < 8) && (jj >= 0) && (jj < 8)) {
      aa = 8*ii+jj;
      if(ga[aa] != -p) continue;
      ii += dx[l];
      jj += dy[l];
      while((ii>=0) && (ii < 8) && (jj >= 0) && (jj < 8)) {
        aa = 8*ii + jj;
        if(ga[aa] == p) {
          flip=1;
          break;
        }
        if(ga[aa] == 0.0) break;
        ii += dx[l];
        jj += dy[l];
      }
    }
    if(flip) {
      ii = i+dx[l];
      jj = j+dy[l];
      aa = 8*ii+jj;
      while(ga[aa] == -p) {
        ga[aa] = p;
        ii += dx[l];
        jj += dy[l];
        aa = 8*ii+jj;
      }
    }
  }
  // tentatively flip player
  ga[PLAYER] = -p;
  int actions[64];
  int num_actions;
  num_actions = getValidActions(actions, ga);
  if(num_actions == 0) {
    // try flipping back to p again
    ga[PLAYER] = p;
    num_actions = getValidActions(actions, ga);
    if(num_actions == 0) {
      // game is over since neither player can move
      ga[ENDED] = 1.0;
      float score = 0.0;
      for(aa=0;aa<64;aa++) score += ga[aa];
      ga[TERMINALSCORE] = score/8.0; // normalize by sqrt(64) (random walk)
      return 1;
    }
  }
  /* game not ended, returning 0 */
  return 0;
}

void printGame(const float* const g) {
  char lc[] = "abcdefgh";
  int i,j;
  float p;
  
  printf("   ");
  for(j=0;j<8;j++) printf(" %c ",lc[j]);
  printf("\n");

  for(i=0;i<8;i++) {
    printf("%3d",i+1);
    for(j=0;j<8;j++) {
      p = g[i*8 + j];
      if(p==1.0) printf(" x ");
      else if(p == -1.0) printf(" o ");
      else printf(" . ");
    }
    printf("%3d\n",i+1);
  }

  printf("   ");
  for(j=0;j<8;j++) printf(" %c ",lc[j]);
  printf("\n");

  if(g[ENDED] == 0.0) {
    if(g[PLAYER] == 1.0) printf("x to move.\n");
    else printf("o to move.\n");
  }
  else {
    printf("Game over, final score = %f\n",8.0*g[TERMINALSCORE]);
  }
}
