#include <stdio.h>
#include <assert.h>
#include <string.h>

#include <game.h>

#define N 5
// N*N dots, (N-1)*(N-1) boxes
// N*(N-1) horizontal lines, (N-1)*N vertical lines
#define NUMACTIONS (2*N*(N-1))
#define GAMELENGTH (NUMACTIONS + (N-1)*(N-1) + 4) // board, player, ended, score, boxes
#define PLAYER (GAMELENGTH - 4)
#define ENDED (GAMELENGTH - 3)
#define SCORE (GAMELENGTH - 2)
#define BOXES (GAMELENGTH - 1)

const int numActions(void) {
    return NUMACTIONS; // horizontal and vertical lines
}

const int gameLength(void) {
    return GAMELENGTH; // board, player, ended, score
}

void rootState(float* const g) {
    memset(g, 0, GAMELENGTH*sizeof(float));
    // last move unset empty
    g[PLAYER] = 1.0;
}

float playerId(const float* const g) {
    return g[PLAYER];
}

const int inputLength(void) {
    return 3*N*N;
}

// horizontal, vertical, boxes
void inputNetwork(float* const x, const float* const g) {
    int i,j;
    memset(x, 0, 3*N*N*sizeof(float)); // clear input
    for(i=0; i<N; i++) {
        for(j=0; j<N-1; j++) {
            x[i*N + j] = g[i*(N-1) + j]; // horizontal
        }
    }
    for(i=0; i<N-1; i++) {
        for(j=0; j<N; j++) {
            x[N*N + i*N + j] = g[N*(N-1) + i*N + j]; // vertical
        }
    }
    float p = g[PLAYER];
    for(i=0; i<N-1; i++) {
        for(j=0; j<N-1; j++) {
            x[2*N*N + i*N + j] = p*g[2*N*(N-1) + i*(N-1) + j]; // boxes
        }
    }
}

int gameEnded(float* const terminal_score, const float* const g) {
    *terminal_score = g[SCORE];
    return (int) g[ENDED];
}

int isValidAction(const float* const g, int const a) {
    assert((0<=a) && (a < NUMACTIONS));
    if(g[ENDED] != 0.0) return 0; // game ended
    if(g[a] != 0.0) return 0; // line already drawn
    return 1;
}

int getValidActions(int* actions, const float* const g) {
    int a, num_actions = 0;
    for(a=0; a<NUMACTIONS; a++) {
        if(g[a] == 0.0) { // line not drawn) {
            actions[num_actions++] = a;
        }
    }
    return num_actions;
}

//returns 1 if next state is terminal, 0 otherwise; returns -1 if action=a is not valid
int nextState(float* const ga, const float* const g, const int a) {
    assert((0<=a) && (a < NUMACTIONS));
    memcpy(ga, g, GAMELENGTH*sizeof(float)); // copy current state
    if(!isValidAction(g, a)) return -1; // invalid action
    float p = g[PLAYER];
    float q; // next player

    ga[a] = 1.0; // set line drawn

    // check if any box is completed
    int i,j;
    int aa, ii, jj;
    int count;

    if(a < N*(N-1)) {
        // horizontal line
        i = a / (N-1);
        j = a % (N-1);
        // check box above
        if(i > 0) {
            count = 0;
            // horizontal line above
            aa = a - (N-1);
            if(ga[aa] != 0.0) count++;
            // vertical line left above
            ii = i-1;
            jj = j;
            aa = N*(N-1) + ii*N + jj;
            if(ga[aa] != 0.0) count++;
            // vertical line right above
            ii = i-1;
            jj = j+1;
            aa = N*(N-1) + ii*N + jj;
            if(ga[aa] != 0.0) count++;
            if(count == 3) {
                // box completed
                ga[BOXES] += 1.0; // increment box count
                ga[2*N*(N-1) + (i-1)*(N-1) + j] = p; // set box to player p
                ga[SCORE] += p; // increment score
            }
        }
        // check box below
        if(i < N-1) {
            count = 0;
            // horizontal line below
            aa = a + (N-1);
            if(ga[aa] != 0.0) count++;
            // vertical line left below
            ii = i;
            jj = j;
            aa = N*(N-1) + ii*N + jj;
            if(ga[aa] != 0.0) count++;
            // vertical line right below
            ii = i;
            jj = j+1;
            aa = N*(N-1) + ii*N + jj;
            if(ga[aa] != 0.0) count++;
            if(count == 3) {
                // box completed
                ga[BOXES] += 1.0; // increment box count
                ga[2*N*(N-1) + i*(N-1) + j] = p; // set box to player p
                ga[SCORE] += p; // increment score
            }
        }
    }
    else {
        // vertical line
        i = (a - N*(N-1)) / N;
        j = (a - N*(N-1)) % N;
        // check box to the left
        if(j > 0) {
            count = 0;
            // vertical line to the left
            aa = a - 1;
            if(ga[aa] != 0.0) count++;
            // horizontal line above left
            ii = i;
            jj = j-1;
            aa = ii*(N-1) + jj;
            if(ga[aa] != 0.0) count++;
            // horizontal line below left
            ii = i+1;
            jj = j-1;
            aa = ii*(N-1) + jj;
            if(ga[aa] != 0.0) count++;
            if(count == 3) {
                // box completed
                ga[BOXES] += 1.0; // increment box count
                ga[2*N*(N-1) + i*(N-1) + (j-1)] = p; // set box to player p
                ga[SCORE] += p; // increment score
            }
        }
        // check box to the right
        if(j < N-1) {
            count = 0;
            // vertical line to the right
            aa = a + 1;
            if(ga[aa] != 0.0) count++;
            // horizontal line above right
            ii = i;
            jj = j;
            aa = ii*(N-1) + jj;
            if(ga[aa] != 0.0) count++;
            // horizontal line below right
            ii = i+1;
            jj = j;
            aa = ii*(N-1) + jj;
            if(ga[aa] != 0.0) count++;
            if(count == 3) {
                // box completed
                ga[BOXES] += 1.0; // increment box count
                ga[2*N*(N-1) + i*(N-1) + j] = p; // set box to player p
                ga[SCORE] += p; // increment score
            }
        }
    }

    if(ga[BOXES] > g[BOXES]) {
        // player p gets another turn
        q = p;
    }
    else {
        // switch players
        q = -p;
    }
    ga[PLAYER] = q; // set next player

    // check if game ended
    if(ga[BOXES] == (N-1)*(N-1)) {
        ga[ENDED] = 1.0; // game ended
        return 1; // game ended
    }

    return 0; // game not ended
}

void printGame(const float* const g) {
    int i,j; // indicates position of dot
    int a;
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            printf("*");
            // horizontal line?
            if(j < N-1) {
                a = i*(N-1) + j;
                printf("%s", (g[a] == 0.0) ? "   " : "---");
            }
        }
        printf("\n");
        if(i == N-1) break;
        for(j=0; j<N; j++) {
            // vertical line?
            a = N*(N-1) + i*N + j;
            printf("%c", (g[a] == 0.0) ? ' ' : '|');
            if(j < N-1) {
                // box?
                a = 2*N*(N-1) + i*(N-1) + j;
                switch ((int) g[a]) {
                    case 0: printf("   "); break; // empty box
                    case 1: printf(" X "); break; // player 1 box
                    case -1: printf(" O "); break; // player 2 box
                    default: assert(0); // invalid box state
                }
            }
        }
        printf("\n");
    }
    if(g[ENDED] != 0.0) {
        printf("\nGame Over! ");
        if(g[SCORE] > 0.0) {
            printf("X wins with score %.1f\n", g[SCORE]);
        }
        else if(g[SCORE] < 0.0) {
            printf("O wins with score %.1f\n", -g[SCORE]);
        }
        else {
            printf("It's a draw!\n");
        }
    }
    else {
        printf("\nCurrent player: %s\n", (g[PLAYER] > 0.0) ? "X" : "O");
    }
}
