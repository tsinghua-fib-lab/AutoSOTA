#include <stdint.h>
#include <stdlib.h>

// Implements dictionary (hash table) with 32-bit hashes.
// it consists of a contiguous stack of key,value,next pairs
// it also has an array called "first"
// first[hash(key)] = position in stack of first item having hash=hash(key)
// if there are none, then first[a] = 2**32 (an invalid address)
// the "next" tells the position in the stack of the next item having the same hash
// if "next" is 2**32, then this is the final item having the given hash.
// advantages are that memory is stored contiguously, and there is
// not often need to call malloc.  however, stack positions are
// essentially random, so access is random access.
// sometimes this isn't bad at all, if the dict is small and loads into cache.
#define MINDELTA 1


typedef struct {
  int len_key; // <game.h> gameLength()
  int len_value; // 3 * (<game.h> numActions()) + 1; {P,Q,N,v}
  uint32_t mask;
  int log2_len_first;
  uint64_t* first;
  int log2_len_stack;
  float* stack_key;
  float* stack_value;
  uint64_t* stack_next;
  uint64_t num_stack;
} dict32_t;

int init_dict32(dict32_t* const d, int const len_key, int const len_value, int const log2_len_first, int const log2_len_stack);
void free_dict32(dict32_t* const d);
int upsize_dict32(dict32_t* d);
int insert_dict32(dict32_t* d, const float* const key, const float* const value);
int in_dict32(const dict32_t* const d, const float* const key);
int lookup_dict32(float* const value, const dict32_t* const d, const float* const key);
void print_dict32(const dict32_t* const d);
int64_t sizeof_dict32(dict32_t* d);

