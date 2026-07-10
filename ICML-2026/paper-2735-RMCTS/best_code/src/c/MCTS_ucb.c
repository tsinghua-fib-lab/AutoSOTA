#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#include "MCTS_ucb.h"
#include "random.h"
#include "game.h"

#define SOFTPOWER 16.0
#define UCB_EPSILON 0.1

typedef struct {
  pthread_mutex_t mutex;
  int next_job;
  int num_jobs;
} next_job_t;

next_job_t NextJob = {PTHREAD_MUTEX_INITIALIZER,0,0};

float sum_of_float_array(float* a, int len_a) {
  int i;
  float s = 0.0;
  for(i=0;i<len_a;i++) s += a[i];
  return s;
}

int argmax_of_float_array(float* a, int len_a) {
  int i;
  float m;
  int i_max; 

  assert(len_a > 0); 
  m = a[0];
  i_max = 0;
  for(i=1;i<len_a;i++) {
    if(a[i]>m) {
      m = a[i];
      i_max = i;
    }
  }
  return i_max;
}

int softmax_of_float_array(float* a, int len_a) {
  int i, i_max;
  float m, sum_p;
  float* p;
  if(len_a <= 0) return -1;
  p = (float*) malloc(len_a * sizeof(float));
  i_max = argmax_of_float_array(a, len_a);
  m = a[i_max];
  for(i=0;i<len_a;i++) {
    p[i] = exp(SOFTPOWER*(a[i]-m));
  }
  sum_p = sum_of_float_array(p, len_a);
  for(i=0;i<len_a;i++) {
    p[i] /= sum_p;
  }
  i_max = random_index(p, len_a);
  free(p);
  return i_max;
}

int maximum_of_int_array(int* a, int len_a) {
  int i;
  int m;
  if(len_a==0) return 0;
  m = a[0];
  for(i=1;i<len_a;i++) if(a[i] > m) m = a[i];
  return m;
}

//debug
int bound_test(float* x, int len_x, float bound) {
  int i;
  for(i=0;i<len_x;i++) {
    if(abs(x[i]) > bound) return 0;
  }
  return 1;
}

void GamePath_init(GamePath_t* const gpath) {
  int len_gamestate = gameLength();
  int log2_capacity = GAMEPATH_DEFAULT_LOG2_CAPACITY;
  gpath->root = (float*) calloc(len_gamestate, sizeof(float));
  gpath->game = (float*) calloc((1ll<<log2_capacity)*len_gamestate,sizeof(float));
  gpath->action = (int*) calloc((1ll<<log2_capacity),sizeof(int));
  gpath->len_path = 0;
  gpath->log2_capacity = log2_capacity;
}

void GamePath_newroot(GamePath_t* const gpath, const float* const root) {
  int len_gamestate = gameLength();
  memcpy(gpath->root, root, len_gamestate*sizeof(float));
  gpath->len_path=0;
}

void GamePath_reset(float* g, GamePath_t* gpath) {
  int len_gamestate = gameLength();
  gpath->len_path=0;
  memcpy(g, gpath->root, len_gamestate*sizeof(float));
}

void GamePath_endnode(float* const g, const GamePath_t* const gpath) {
  int len_gamestate = gameLength();
  int l = gpath->len_path;
  if(l>0) memcpy(g, gpath->game + (l-1)*len_gamestate, len_gamestate*sizeof(float));
  else memcpy(g, gpath->root, len_gamestate*sizeof(float));
}

void GamePath_update(GamePath_t* const gpath, int const a) {
  int len_gamestate = gameLength();
  int l;
  float* g;
  float* ga = (float*) malloc(len_gamestate*sizeof(float));
  if((gpath->len_path + 1) >> (gpath->log2_capacity)) {
    gpath->log2_capacity++;
    gpath->game = (float*) realloc(gpath->game, (1ll<<gpath->log2_capacity)*len_gamestate*sizeof(float));
    gpath->action = (int*) realloc(gpath->action, (1ll<<gpath->log2_capacity)*sizeof(int));
  }
  l = gpath->len_path;
  if(l>0) g = gpath->game + (l-1)*len_gamestate;
  else g = gpath->root;
  nextState(ga,g,a);
  memcpy(gpath->game + (gpath->len_path)*len_gamestate, ga, len_gamestate*sizeof(float));
  gpath->action[gpath->len_path] = a;
  gpath->len_path++;
  free(ga);
}

void GamePath_free(GamePath_t* const gpath) {
  free(gpath->root);
  free(gpath->game);
  free(gpath->action);
  gpath->root = NULL;
  gpath->game = NULL;
  gpath->action = NULL;
  gpath->len_path=0;
  gpath->log2_capacity=-1;
}

void GamePath_randompath(GamePath_t* const gpath) {
  int len_gamestate = gameLength();
  int n = numActions();

  float* g = (float*) malloc(len_gamestate * sizeof(float)); 
  int* actions = (int*) malloc(n * sizeof(int));
  int num_valid_actions;
  int i,a;
  int ended;
  float terminal_score;
  rootState(g);
  GamePath_newroot(gpath, g);
  ended = gameEnded(&terminal_score,g);
  while(!ended) {
    num_valid_actions = getValidActions(actions, g);
    if(num_valid_actions == 0) break;
    i = Knuth_lrand()%num_valid_actions;
    a = actions[i];
    GamePath_update(gpath,a);
    GamePath_endnode(g, gpath);
    ended = gameEnded(&terminal_score,g);
  }
  free(g);
  free(actions);
}

void GamePath_print(const GamePath_t* const gpath) {
  int len_gamestate = gameLength();
  int i;
  printGame(gpath->root);
  for(i=0;i<gpath->len_path;i++) {
    printf("action %d taken:\n",gpath->action[i]);
    printGame(gpath->game + i*len_gamestate);
    printf("\n");
  }
}

void GamePath_propagate_value(dict32_t* const d, const GamePath_t* const gpath, float const terminal_score) {
  /* this is called by the update method (see below) */
  /* takes newly learned value terminal_score, and propagates it up the game path */
  /* terminal_score should be the score relative to player 1 */
  /* if the end node of the path is not a terminal game state, */
  /* then terminal_score is actually the network's evaluation */
  int len_gamestate = gameLength();
  float* g;
  int j;
  int a;
  float value_g;
  int n = numActions();
  float* localPQNv;
  float* lQ;
  float* lN;
  int found_in_dict;
  if(gpath->len_path == 0) return;

  localPQNv = (float*) malloc((3*n+1)*sizeof(float));

  //printf("propagating value %.2f\n",terminal_score);
  lQ = localPQNv + n;
  lN = localPQNv + 2*n;

  g = gpath->root;
  
  for(j=0;j < gpath->len_path; j++) {
    value_g = playerId(g) * terminal_score;
    found_in_dict = lookup_dict32(localPQNv, d, g);
    assert(found_in_dict);
    a = gpath->action[j];
    lQ[a] = (lN[a]*lQ[a] + value_g) / (lN[a] + 1.0);
    lN[a] += 1.0;
    //debug
    //assert(bound_test(localPQNv, 2*n, 10.0));
    //assert(abs(localPQNv[3*n]) <= 10.0);
    insert_dict32(d, g, localPQNv);
    g = gpath->game + j*len_gamestate;
  }
  free(localPQNv);
}


void* MCTS_init(int const stacksize,
    int const numSims,
    float* root_states,
    float* G,
    float* P,
    float* V,
    float* N,
    float* Q,
    float const c_puct)
{
  MCTS_t* t;
  int i;
  int len_gamestate = gameLength();

  int n = numActions();

  t = (MCTS_t*) calloc(1,sizeof(MCTS_t));
  t->G = G;
  t->P = P;
  t->V = V;
  t->Q = Q;
  t->N = N;
  t->lane_needs_inference = (int*) calloc(stacksize, sizeof(int));
  t->stage = 0;
  t->stacksize = stacksize;
  t->numSims = numSims;
  t->root_states = root_states;
  t->c_puct = c_puct;
  t->PQNv = (dict32_t*) calloc(stacksize, sizeof(dict32_t));
  for(i=0;i<stacksize;i++) init_dict32(t->PQNv + i, len_gamestate, 3*n+1, 15, 10);
  t->game_paths = (GamePath_t*) calloc(stacksize, sizeof(GamePath_t));
  for(i=0;i<stacksize;i++) GamePath_init(t->game_paths + i); 
  t->num_simulations = (int*) calloc(stacksize, sizeof(int));
  t->requests = (int*) calloc(stacksize, sizeof(int));
  t->len_requests = 0;
  memset(t->time, 0, 4*sizeof(double));
  return (void*) t;
}

void MCTS_free(void* const mcts) {
  MCTS_t* t = (MCTS_t*) mcts;
  int i;
  if(!t) return;
  if(t->requests) free(t->requests);
  if(t->num_simulations) free(t->num_simulations);
  if(t->game_paths) {
    for(i=0;i<t->stacksize;i++) GamePath_free(t->game_paths + i);
    free(t->game_paths);
  }
  if(t->PQNv) {
    for(i=0;i<t->stacksize;i++) free_dict32((t->PQNv)+i);
    free(t->PQNv);
  }
  if(t->lane_needs_inference) free(t->lane_needs_inference);
  free(t);
}

/*
we run MCTS_update with multiple threads; these threads 
work on disjoint parts of the data, and when they finish,
various rows of G (the game states that need input) are 
marked as needing a request in t->lane_needs_inference.  
The total number of rows 
is the stacksize (number of lanes).  
defragment puts those requested rows into 
a contiguous array.  The relevant lanes are kept track 
of through the t->requests array.  
t->requests[ii] = i means that row ii of G pertains to lane i.
*/
void defragment(MCTS_t* t) {
  int len_gamestate = gameLength();
  int i, ii, num_lanes;
  num_lanes = 0;
  for(i=0; i<t->stacksize; i++) {
    if(t->lane_needs_inference[i]) {
      t->requests[num_lanes] = i;
      num_lanes++;
    }
  }
  t->len_requests = num_lanes;  
  for(ii=0; ii < (t->len_requests); ii++) {
    i = t->requests[ii];
    if(i > ii) {
      memcpy((t->G) + ii*len_gamestate, (t->G) + i*len_gamestate, len_gamestate*sizeof(float));
    }
  }
}

// If return value k is positive, then MCTS_update is not finished, and is requesting k predictions from the network.
// The input for the network requested prediction is stored into G by MCTS_update. 
// The array G holds the requested game states contiguously. The predictions should be put into P (policy)
// and V (value) arrays, and then MCTS_update should be called again.  It keeps track of
// the state internally, so that it knows how to proceed the next time that it is called.
// This recalling process should continue, multiple times, until MCTS_update returns 0, when it has finished.
//   It is also possible to know if MCTS_update has finished by looking at the value of stage;
// If stage==2 then MCTS_update has finished. The first time this function is called, stage should hold value 0 (stage 0).
//   Once the return value is 0, the appropriate results are stored in the Q and N arrays, lining up with the root_states.
// That is, Q[i*numActions() + j], for j=0...numActions()-1, will hold the Q-action values for the i-th root state,
// once MCTS_update has completed.  Similar for the N array.
int MCTS_update(void* const mcts, int const verbosity, int const num_threads)
{
  int len_gamestate = gameLength();

  struct timeval Time0, Time1;
  long secs, microsecs;
  double time_elapsed;

  gettimeofday(&Time0, 0);

  MCTS_t* t = (MCTS_t*) mcts;
  int n = numActions();
  int i,j;
  // g and localPQNv will need to be freed
  float* g = (float*) malloc(len_gamestate*sizeof(float));
  float* localPQNv = (float*) malloc((3*n+1)*sizeof(float));
  int ended;
  float terminal_score;
  float *gQ;
  float* gN;
  float* root_states = t->root_states;
  float* G = t->G;
  float* N = t->N;
  float* Q = t->Q;
  int numSims = t->numSims;

  
  if(t->stage == 0) {
    // stage 0: setting up root states
    // set up any needed requests for network
    if(verbosity >= 1) printf("initializing root states.\n");
    t->len_requests = 0;
    for(i=0;i<t->stacksize;i++) {
      memcpy(g, root_states + i*len_gamestate, len_gamestate*sizeof(float));
      GamePath_newroot(t->game_paths + i, g);
      ended = gameEnded(&terminal_score, g);
      if(ended) {
	     t->num_simulations[i]=0;
      }
      else {
	     t->num_simulations[i]=numSims;
	     if(!in_dict32(t->PQNv + i, g)) {
	       t->requests[t->len_requests] = i;
	       memcpy(G + (t->len_requests)*len_gamestate, g, len_gamestate*sizeof(float));
	       t->len_requests++;
	     }
      }
    }
    t->stage = 1;

    gettimeofday(&Time1, 0);
    secs = Time1.tv_sec - Time0.tv_sec;
    microsecs = Time1.tv_usec - Time0.tv_usec;
    time_elapsed = ((double) secs) + ((double) microsecs)*1.0e-6;
    t->time[0] = time_elapsed;
    gettimeofday(&Time0, 0);

    if(t->len_requests > 0) {
      free(g);
      free(localPQNv);
      return t->len_requests;
    }
  }

  if(t->stage == 1) {
    if(verbosity >= 1) {
      printf("Top of Stage 1: Number of simulations to go per lane:\n");
      for(i=0;i<t->stacksize;i++) printf(" %d",t->num_simulations[i]);
      printf("\n");
    }

    //zero out t->lane_needs_inference
    memset(t->lane_needs_inference, 0, (t->stacksize)*sizeof(int));

    pthread_mutex_init(&(NextJob.mutex), NULL);
    NextJob.next_job = 0;
    NextJob.num_jobs = t->len_requests;

    //prepare threads and thread arguments
    pthread_t* thread;
    MCTS_threadargs_t* args;
    thread = (pthread_t*) malloc(num_threads * sizeof(pthread_t));
    args = (MCTS_threadargs_t*) malloc(num_threads * sizeof(MCTS_threadargs_t));

    for(j=0;j<num_threads;j++) {
      args[j].verbosity = verbosity;
      args[j].t = t;
    }

    gettimeofday(&Time1, 0);
    secs = Time1.tv_sec - Time0.tv_sec;
    microsecs = Time1.tv_usec - Time0.tv_usec;
    time_elapsed = ((double) secs) + ((double) microsecs)*1.0e-6;
    t->time[1] += time_elapsed;
    gettimeofday(&Time0, 0);

    for(j=0;j<num_threads;j++) {
      pthread_create(thread + j, NULL, thread_MCTS_update, (void*) (args+j));
    }
    for(j=0;j<num_threads;j++) {
      pthread_join(thread[j], NULL);
    }

    free(thread);
    free(args);

    pthread_mutex_destroy(&(NextJob.mutex));

    gettimeofday(&Time1, 0);
    secs = Time1.tv_sec - Time0.tv_sec;
    microsecs = Time1.tv_usec - Time0.tv_usec;
    time_elapsed = ((double) secs) + ((double) microsecs)*1.0e-6;
    t->time[2] += time_elapsed;
    gettimeofday(&Time0, 0);

    //make G array contiguous; resets t->requests and t->len_requests.
    defragment(t);

    gettimeofday(&Time1, 0);
    secs = Time1.tv_sec - Time0.tv_sec;
    microsecs = Time1.tv_usec - Time0.tv_usec;
    time_elapsed = ((double) secs) + ((double) microsecs)*1.0e-6;
    t->time[3] += time_elapsed;
    gettimeofday(&Time0, 0);


    if(verbosity >= 1) {
      printf("Bottom of Stage 1: Number of simulations to go per lane:\n");
      for(i=0;i<t->stacksize;i++) printf(" %d",t->num_simulations[i]);
      printf("\n");
      printf("maximum = %d\n",maximum_of_int_array(t->num_simulations, t->stacksize));
      printf("Bottom of Stage 1: Number of requests = %d\n",t->len_requests);
    }
    
    // check if we are finished with stage 1 (i.e. no more sims to go in any lane)
    if(t->len_requests == 0) {
      // we must load Q,N with learned values (from MCTS sims) at the various root states
      for(i=0;i<t->stacksize;i++) {
      	GamePath_reset(g, t->game_paths + i);
      	ended = gameEnded(&terminal_score, g);
      	if(ended) {
          memset(N + i*n, 0, n*sizeof(float));
      	  //V[i] = playerId(g) * terminal_score;
      	}
      	else {
      	  assert(lookup_dict32(localPQNv, t->PQNv + i, g));
      	  //gP = localPQNv;
      	  gQ = localPQNv + n;
      	  gN = localPQNv + 2*n;
      	  //memcpy(P + i*n, gP, n*sizeof(float));
      	  memcpy(Q + i*n, gQ, n*sizeof(float));
      	  memcpy(N + i*n, gN, n*sizeof(float));
      	  //V[i] = localPQNv[3*n];
      	}
      } // end for(i=0..
      if(verbosity >= 1) printf("Setting stage to 2\n");
      t->stage = 2; // indicates that we've finished all the simulations
      gettimeofday(&Time1, 0);
      secs = Time1.tv_sec - Time0.tv_sec;
      microsecs = Time1.tv_usec - Time0.tv_usec;
      time_elapsed = ((double) secs) + ((double) microsecs)*1.0e-6;
      t->time[1] += time_elapsed;
      gettimeofday(&Time0, 0);
      free(g);
      free(localPQNv);
      return 0;
    } 
    else {
      gettimeofday(&Time1, 0);
      secs = Time1.tv_sec - Time0.tv_sec;
      microsecs = Time1.tv_usec - Time0.tv_usec;
      time_elapsed = ((double) secs) + ((double) microsecs)*1.0e-6;
      t->time[1] += time_elapsed;
      gettimeofday(&Time0, 0);

      free(g);
      free(localPQNv);
      return t->len_requests;
    }
  } // end if(*stage == 1)
  
  return 0; // but we should never reach this point
}


/*typedef struct {
  int verbosity;
  MCTS_t* t;
} MCTS_threadargs_t;*/

void* thread_MCTS_update(void* thread_args_ptr)
{
  int len_gamestate = gameLength();
  MCTS_threadargs_t* args = (MCTS_threadargs_t*) thread_args_ptr;
  MCTS_t* t = args->t;
  int n = numActions();
  int num_jobs;
  int i,ii,j,jj;
  float* g = (float*) malloc(len_gamestate*sizeof(float));
  float* localPQNv = (float*) malloc((3*n+1)*sizeof(float)); 
  int ended;
  float terminal_score;
  float* ucb = (float*) calloc(n, sizeof(float));
  float* ucb_valid = (float*) calloc(n, sizeof(float));
  int* validactions = (int*) calloc(n, sizeof(float));
  int num_validactions;
  float c_puct;
  float* gP;
  float* gQ;
  float* gN;
  float* root_policy = (float*) malloc(n*sizeof(float));
  int a;
  float* G = t->G;
  float* P = t->P;
  float* V = t->V;
  int verbosity = args->verbosity;

  // BEFORE RETURN:
  // free(g); free(localPQNv); free(actions); free(ucb); free(ucb_actions); free(root_policy);

  pthread_mutex_lock(&(NextJob.mutex));
  ii = NextJob.next_job; // these begin at 0 and go up 1 at a time.
  num_jobs = NextJob.num_jobs; // total number of jobs to be completed.
  NextJob.next_job++; // increment job index for next job.
  pthread_mutex_unlock(&(NextJob.mutex));

  while(ii < num_jobs) {
    i = t->requests[ii]; // lane where a network request was fulfilled (hopefully!)
    //set up localPQNv
    memcpy(localPQNv, P+ii*n, n*sizeof(float)); // P + ii*n should contain the requested network policy
    for(jj=n;jj<2*n;jj++) localPQNv[jj] = 0.0; // initialize Q to zero
    // ALTERNATIVE: (better?) initialize Q to parent state value V[ii]
    // for(jj=n;jj<2*n;jj++) localPQNv[jj] = V[ii]; // initialize Q to V[ii] = network value of parent state
    for(jj=2*n;jj<3*n;jj++) localPQNv[jj] = 0.0; // initialize N to all 0.0
    localPQNv[3*n] = V[ii]; // V[ii] should contain the requested value
    // jump to the end of the current game path
    // this game g is the game state where the ii-th request was needed
    // it should not be in the dictionary t->PQNv + i (yet)
    GamePath_endnode(g, t->game_paths + i); 
    //debug
    //assert(bound_test(localPQNv, 2*n, 10.0));
    //assert(abs(localPQNv[3*n]) <= 10.0);
    insert_dict32(t->PQNv + i, g, localPQNv);
    // now we should inform MCTS algorithm by back propogating the learned value V[ii] for g
    // all the way up to the i-th root state along the game path
    // the value at g is made relative to player number 1, so must be multiplied by playerId(g)
    GamePath_propagate_value(t->PQNv + i, t->game_paths + i, playerId(g)*V[ii]);
    // test if g is strictly below the root 
    if(t->game_paths[i].len_path) {
      /* g is not a root state, so we should reduce simulations to go */
      /* we do not count the root state request as one of the simulations, */
      /* because we want the N-array to sum up to the number of simulations */
      assert(t->num_simulations[i] > 0);
      t->num_simulations[i]--;
    }
    // continue in this lane until a network request prevents further progress
    if(t->num_simulations[i] == 0) {
      // no more simulations for this lane, so grab the next job index and continue
      // at the top of the while(ii < num_jobs) loop.
      pthread_mutex_lock(&(NextJob.mutex));
      ii = NextJob.next_job;
      //num_jobs = NextJob.num_jobs;
      NextJob.next_job++;
      pthread_mutex_unlock(&(NextJob.mutex));     
      continue;
    }
    // the simulation starts at the root state
    GamePath_reset(g, t->game_paths + i); // now g is a copy of the i-th root state
    while(1) {
      if(verbosity >= 2) {
        printf("lane %d, with %d sims to go. Current game:\n",i,t->num_simulations[i]);
        printGame(g);
      }
      ended = gameEnded(&terminal_score, g);
      if(ended) {
        if(verbosity >= 2) printf("lane %d, game has terminated.\n",i);
        // valgrind reports uninitialized error here (sometimes/always?) resolved??
        GamePath_propagate_value(t->PQNv + i, t->game_paths + i, terminal_score); 
        GamePath_reset(g, t->game_paths + i);
        assert(t->num_simulations[i] > 0);
        t->num_simulations[i]--;
        if(verbosity >= 2) {
          printf("lane %d, with %d sims to go. Now at root state of lane i:\n",i,t->num_simulations[i]);
          printGame(g);
        }
        if(t->num_simulations[i]==0) break;
      }
      else if(!in_dict32(t->PQNv + i, g)) {
        // need to ask network for policy and value
        t->lane_needs_inference[i] = 1;
        memcpy(G + i*len_gamestate, g, len_gamestate*sizeof(float));
        if(verbosity >= 2) printf("lane %d, request made.\n",i);
        break;
      }
      // now g has been visited, is not terminal, and we have necessary value,policy
      // so we can compute ucb and select the next action to take
      assert(lookup_dict32(localPQNv, t->PQNv + i, g));
      gP = localPQNv;
      gQ = localPQNv + n;
      gN = localPQNv + 2*n;
      /* compute ucb values */
      /* not using this dynamic c_puct formula*/
      /* c_puct = t->pb_c_init + log((sum_of_float_array(gN,n) + t->pb_c_base + 1.0)/t->pb_c_base); */
      c_puct = t->c_puct;
      if(verbosity >= 2) printf("lane %d, c_puct = %.2f\n",i,c_puct);
      for(j=0;j<n;j++) ucb[j] = gQ[j] + c_puct*gP[j]*sqrt(sum_of_float_array(gN,n) + UCB_EPSILON)/(1.0 + gN[j]);      
      // float* ucb_valid = (float*) calloc(n, sizeof(float));
      // int* validactions = (int*) calloc(n, sizeof(float));
      // int num_validactions;      
      num_validactions = getValidActions(validactions, g);
      assert(num_validactions > 0);
      for(j=0;j<num_validactions;j++) {
        a = validactions[j];
        ucb_valid[j] = ucb[a];
      }
      jj = argmax_of_float_array(ucb_valid, num_validactions);
      a = validactions[jj];
      assert((0 <= a) && (a<n)); 
      assert(isValidAction(g,a));
      if(verbosity >= 2) printf("lane %d, choosing action %d\n",i,a);
      GamePath_update(t->game_paths + i, a);
      GamePath_endnode(g, t->game_paths + i);
    } // end while(1)
    pthread_mutex_lock(&(NextJob.mutex));
    ii = NextJob.next_job;
    //num_jobs = NextJob.num_jobs;
    NextJob.next_job++;
    pthread_mutex_unlock(&(NextJob.mutex));
  } // end while(ii < num_jobs)
  free(g); free(localPQNv); free(ucb); free(root_policy); 
  free(validactions); free(ucb_valid);
  return NULL;
}

void get_time(double* time, void* mcts) {
  MCTS_t* t = (MCTS_t*) mcts;
  memcpy(time, t->time, 4*sizeof(double));
}

// returns 1 if game state g is found in the given lane; if so, populates PQNv
// if g is not found (or lane is invalid) it returns 0
int MCTS_lookup(float* PQNv, void* const mcts, int const lane, const float* const g) {
  MCTS_t* t = (MCTS_t*) mcts;
  if((lane < 0) || (lane >= t->stacksize)) return 0;
  return lookup_dict32(PQNv, t->PQNv + lane, g);
}
