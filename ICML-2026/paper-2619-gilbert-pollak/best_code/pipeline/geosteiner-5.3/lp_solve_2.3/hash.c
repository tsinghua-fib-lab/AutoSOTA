#include "lpkit.h" /* only for MALLOC, CALLOC */
#include <string.h>
#include <limits.h>

/* hash functions for open hashing */

hashtable *create_hash_table(int size)
{
  hashtable *ht;

  MALLOC(ht, 1);
  CALLOC(ht->table, size);
  ht->size = size;
  return(ht);
}

void free_hash_table(hashtable *ht)
{
  int i;
  hashelem *hp, *thp;

  for(i = 0; i < ht->size; i++) {
    hp = ht->table[i];
    while(hp != NULL) {
      thp = hp;
      hp = hp->next;
      free(thp->name);
      free(thp);
    }
  }
  free(ht->table); 
  free(ht);
}

/* make a good hash function for any int size */
/* inspired by Aho, Sethi and Ullman, Compilers ..., p436 */
#define HASH_1 sizeof(unsigned int)
#define HASH_2 (sizeof(unsigned int) * 6)
#define HASH_3 (((unsigned int)0xF0) << ((sizeof(unsigned int) - 1) * CHAR_BIT))

static int hashval(const char *string, int size)
{
  unsigned int result = 0, tmp;

  for(; *string; string++) {
    result = (result << HASH_1) + *string;
    if((tmp = result & HASH_3) != 0) {
      /* if any of the most significant bits is on */
      result ^= tmp >> HASH_2; /* xor them in in a less significant part */
      result ^= tmp; /* and reset the most significant bits to 0 */
    }
  }
  return(result % size);
} /* hashval */

hashelem *findhash(const char *name, hashtable *ht)
{
  hashelem *h_tab_p;
  for(h_tab_p = ht->table[hashval(name, ht->size)];
      h_tab_p != NULL;
      h_tab_p = h_tab_p->next)
    if(strcmp(name, h_tab_p->name) == 0) /* got it! */
      break;
  return(h_tab_p);
} /* gethash */

hashelem *puthash(const char *name, hashtable *ht)
{
  hashelem *hp;
  int index;

  if((hp = findhash(name, ht)) == NULL) {
    index = hashval(name, ht->size);
    CALLOC(hp, 1);
    MALLOC(hp->name, strlen(name) + 1);
    strcpy(hp->name, name);
    hp->next = ht->table[index];
    ht->table[index] = hp;
  }
  return(hp);
}

hashtable *copy_hash_table(hashtable *ht)
{
  hashtable *copy;
  hashelem *elem, *new_elem;
  int i;

  copy = create_hash_table(ht->size);

  for(i = 0; i < ht->size; i++) {
    for(elem = ht->table[i]; elem != NULL; elem = elem->next) {
      CALLOC(new_elem, 1);
      /* copy entire struct */
      *new_elem = *elem;
      /* ... but link it into the new list */
      new_elem->next = copy->table[i];
      copy->table[i] = new_elem;
    }
  }

  return(copy);
}
