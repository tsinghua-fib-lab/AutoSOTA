#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "dict32.h"
#include "aes_hash.h"


// employs non-cryptographic variant of AES defined in aes_hash.c;
// uses potentially reduced rounds (cf. AES_ROUNDS from aes_hash.c) and a constant (hard-coded) key.
// enjoys speed from the intrinsic _mm_aesenc_si128 round function.

uint32_t MYHASH(const char* const data, int const keysize) {
  int64_t len_data = (int64_t) keysize;
  uint64_t h2;
  uint32_t h;
  h2 = aes_hash(data, len_data);
  h = ((uint32_t) (h2 >> 32)) ^ ((uint32_t) h2);
  return h;
}

int keys_equal(const float* const key1, const float* const key2, int const len_key) {
  int i;
  for(i=0;i<len_key;i++) {
    if(key1[i] != key2[i]) return 0;
  }
  return 1;
}

int init_dict32(dict32_t* const d, int const len_key, int const len_value, int const log2_len_first, int const log2_len_stack) {
  uint64_t const nopos = 1ull<<32;
  uint64_t i;
  
  d->len_key = len_key;
  d->len_value = len_value;
  if(log2_len_first > 32) return 0;
  d->log2_len_first = log2_len_first;
  d->mask = ((uint32_t) (-1)) >> (32-log2_len_first);
  d->num_stack = 0ll;
  d->first = (uint64_t *) malloc((1ull << log2_len_first) * sizeof(uint64_t));
  if(!d->first) return 0;
  for(i=0ull;i<(1ull << log2_len_first);i++) d->first[i] = nopos;
  d->log2_len_stack = log2_len_stack;
  d->stack_key = (float*) malloc((1ull << log2_len_stack)*len_key*sizeof(float));
  if(!d->stack_key) return 0;
  d->stack_value = (float*) malloc((1ull << log2_len_stack)*len_value*sizeof(float));
  if(!d->stack_value) return 0;
  d->stack_next = (uint64_t*) malloc((1ull << log2_len_stack)*sizeof(uint64_t));
  if(!d->stack_next) return 0;
  return 1;
}

void free_dict32(dict32_t* const d) {
  if(!d) return;
  if(d->first) free(d->first);
  if(d->stack_key) free(d->stack_key);
  if(d->stack_value) free(d->stack_value);
  if(d->stack_next) free(d->stack_next);
  d->first = NULL;
  d->stack_key = NULL;
  d->stack_value = NULL;
  d->stack_next = NULL;
  d->num_stack=0ull;
}

//returns 1 if succcess, returns 0 if failure
//there is always at least one free spot available for a new item
//if by insert, we use up the last spot available,
//then we reallocate space (doubling) in the dictionary after insertion
int insert_dict32(dict32_t* d, const float* const key, const float* const value) {
  uint32_t a;
  uint64_t i;
  uint64_t const nopos = 1ull<<32;
  uint64_t previous_i=nopos;
  int keysize = (d->len_key)*sizeof(float); // size of key in bytes

  if(!d) return 0;
  
  a = MYHASH((char*) key, keysize) & (d->mask);

  // i is position in stack of first item with hash=a
  // if there is no such item, then i is nopos
  i = d->first[a]; 
  if(i == nopos) {
    // this hash a has never been seen before; we must initialize d->first[a]
    // d->first[a] will be the current length of the stack of items
    // this is the index where the new item will be stored
    d->first[a] = d->num_stack;
  }
  else while(i != nopos) {
    if(keys_equal(d->stack_key + i*(d->len_key), key, d->len_key)) break;
    previous_i = i;
    i = d->stack_next[i];
  }

  if(i != nopos) {
    // item is not new; only need to overwrite old value with new value
    memcpy(d->stack_value + i*(d->len_value), value, (d->len_value)*sizeof(float));
  }
  else {
    // new item
    // there should already be space for this new item in d->stack
    if( (d->num_stack) >> (d->log2_len_stack)) {
      //this should not happen -- insertion fails
      if(previous_i == nopos) {
	//reset first at hash address a, back to nopos again
	d->first[a] = nopos;
      }
      return 0;
    }
    if(previous_i < nopos) {
      // used to have d->stack_next[previous_i] == nopos; must update it.
      d->stack_next[previous_i] = d->num_stack;
    }
    //push the new item onto the top of the stack
    memcpy(d->stack_key + (d->num_stack)*(d->len_key), key, keysize);
    memcpy(d->stack_value + (d->num_stack)*(d->len_value), value, (d->len_value)*sizeof(float));
    d->stack_next[d->num_stack] = nopos; //terminal item in this bucket
    d->num_stack++;
    // finally create extra room if stack is full
    if((d->num_stack) >>(d->log2_len_stack)) {
      if(upsize_dict32(d)) return 1;
      else return 0;
    }
  }
  return 1;
}


// returns 0 if not successful with memory allocation, returns 1 is all is ok
int upsize_dict32(dict32_t* d) {
  dict32_t new_d;
  uint64_t i;
  int keysize = (d->len_key)*sizeof(float);

  //we MUST increase the size of stack here, and possibly first as well

  if(d->log2_len_stack + MINDELTA < d->log2_len_first) {
    // we realloc the stacks, but not first
    d->stack_key = (float*) realloc(d->stack_key, (1ull<<((d->log2_len_stack) + 1))*keysize);
    if(!d->stack_key) return 0;
    d->stack_value = (float*) realloc(d->stack_value, (1ull<<((d->log2_len_stack) + 1))*(d->len_value)*sizeof(float));
    if(!d->stack_value) return 0;
    d->stack_next = (uint64_t*) realloc(d->stack_next, (1ull<<((d->log2_len_stack) + 1))*sizeof(uint64_t));
    if(!d->stack_next) return 0;
    d->log2_len_stack++;
    return 1;
  }

  if(d->log2_len_first < 32) {
    // we have room to expand the d->first array
    //double the size of both first array and stack arrays:
    init_dict32(&new_d, d->len_key, d->len_value, d->log2_len_first + 1, d->log2_len_stack + 1);

    //now populate new_d with items from d
    //this must be redone from scratch, because of broadened hash range
    for(i=0ull;i<d->num_stack;i++) {
      if(!insert_dict32(&new_d, d->stack_key + i*(d->len_key), d->stack_value + i*(d->len_value))) {
	     free_dict32(&new_d);
	     return 0;
      }
    }
    
    free_dict32(d);
    d->mask = new_d.mask;
    d->log2_len_first = new_d.log2_len_first;
    d->first = new_d.first;
    d->log2_len_stack = new_d.log2_len_stack;
    d->stack_key = new_d.stack_key;
    d->stack_value = new_d.stack_value;
    d->stack_next = new_d.stack_next;
    d->num_stack = new_d.num_stack;
    
    return 1;
  }
  // there is no room to expand the d->first array any further, so we can only realloc d->stack
  // we realloc the stacks, but not first
  d->stack_key = (float*) realloc(d->stack_key, (1ull<<((d->log2_len_stack) + 1))*keysize);
  if(!d->stack_key) return 0;
  d->stack_value = (float*) realloc(d->stack_value, (1ull<<((d->log2_len_stack) + 1))*(d->len_value)*sizeof(float));
  if(!d->stack_value) return 0;
  d->stack_next = (uint64_t*) realloc(d->stack_next, (1ull<<((d->log2_len_stack) + 1))*sizeof(uint64_t));
  if(!d->stack_next) return 0;
  d->log2_len_stack++;
  return 1;
}

int in_dict32(const dict32_t* const d, const float* const key) {
  uint32_t a;
  uint64_t i;
  uint64_t const nopos = 1ull<<32;
  int keysize = (d->len_key)*sizeof(float);

  a = MYHASH((char*) key, keysize) & (d->mask);
  i = d->first[a]; // i = position in stack of first item with hash=a  
  while( (i<nopos) && !keys_equal(d->stack_key + i*(d->len_key), key, d->len_key) ) i = d->stack_next[i];
  if(i==nopos) return 0;
  return 1;
}

int lookup_dict32(float* const value, const dict32_t* const d, const float* const key) {
  uint32_t a;
  uint64_t i;
  uint64_t const nopos = 1ull<<32;
  int keysize = (d->len_key)*sizeof(float);

  a = MYHASH((char*) key, keysize) & (d->mask);
  i = d->first[a]; // i = position in stack of first item with hash=a  
  while( (i<nopos) && !keys_equal(d->stack_key + i*(d->len_key), key, d->len_key) ) i = d->stack_next[i];
  if(i==nopos) return 0;
  memcpy(value, d->stack_value + i*(d->len_value), (d->len_value)*sizeof(float));
  return 1;
}

void print_dict32(const dict32_t* const d) {
  uint64_t i;
  uint64_t const nopos = 1ull<<32;
  float* key;
  float* value;
  int j;
  int keysize = (d->len_key)*sizeof(float);

  printf("Key Length (num float) = %d\n", d->len_key);
  printf("Value Length (num float) = %d\n", d->len_value);

  printf("First:\n");
  for(i=0ull;i<(1ull<<(d->log2_len_first));i++) {
    if(d->first[i] < nopos) printf(" %lu", d->first[i]);
    else printf(" .");
  }
  printf("\n");

  printf("Stack:\n");
  for(i=0ull;i<d->num_stack;i++) {
    printf("%lu  ",i);
    key = d->stack_key + i*(d->len_key);
    for(j=0;j<d->len_key;j++) {
	   printf(" %.2f",key[j]);
    }
    value = d->stack_value + i*(d->len_value);
    printf(" : ");
    for(j=0;j<d->len_value;j++) {
      printf(" %.0f", value[j]);
    }
    if(d->stack_next[i] < nopos) printf(" ; next = %lu",d->stack_next[i]);
    printf(" ; hash = %u\n", MYHASH((char*) (d->stack_key + i*(d->len_key)), keysize) & (d->mask));
  }
}

int64_t sizeof_dict32(dict32_t* d) {
  int keysize = (d->len_key)*sizeof(float);

  uint64_t s=0ull;
  if(!d) return 0ull;
  s += sizeof(dict32_t);
  s += (1ull<<d->log2_len_first)*sizeof(int64_t);
  s += (1ull<<d->log2_len_stack)*(keysize + (d->len_value)*sizeof(float) + sizeof(uint64_t));
  return s;
}
