/***********************************************************************

	$Id: bbsubs.c,v 1.26 2022/11/19 13:45:49 warme Exp $

	File:	bbsubs.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1995, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Low-level subtroutines of the branch-and-cut.

************************************************************************

	Modification Log:

	a-1:	02/28/2001	warme
		: Split off from bb.c.  Changes for 3.1 release.
	b-1:	08/05/2002	benny
		: Changes for library release.
		: Uses parameters.
		: Fixed some memory leaks.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.  Upgrade fatals.
	e-4:	11/18/2022	warme
		: Initialize bbnode owner and delrow_z fields.
		: Use function to free list of components.

************************************************************************/

#include "bbsubs.h"

#include "bb.h"
#include "constrnt.h"
#include "cutset.h"
#include "fatal.h"
#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include "parmblk.h"
#include "sec_comp.h"
#include "steiner.h"
#include <string.h>
#include "ub.h"


/*
 * Global Routines
 */

void			_gst_add_bbnode (struct bbinfo *	bbip,
					 int			var,
					 int			dir,
					 double			z);
void			_gst_append_node_to_tree (struct bbnode *	p,
						  struct bbtree *	tp);
void			_gst_bbheap_insert (struct bbnode *	p,
					    struct bbtree *	tp,
					    int			heap_no);
struct bbtree *		_gst_create_bbtree (int nmasks);
void			_gst_delete_node_from_bbtree (struct bbnode *	p,
						      struct bbtree *	tp);
void			_gst_destroy_bbinfo (struct bbinfo * bbip);


/*
 * Local Routines
 */

static void		bbheap_delete (struct bbnode *, struct bbtree *, int);
static void		bbheap_free (struct bbheap *);
static void		bbheap_init (struct bbheap *, bbheap_func_t *);
static void		destroy_bbnode (struct bbnode *);
static int		node_is_better (struct bbnode *, struct bbnode *);
static int		node_is_worse (struct bbnode *, struct bbnode *);

/*
 * This routine adds a new node to the branch-and-bound tree.
 */

	void
_gst_add_bbnode (

struct bbinfo *		bbip,		/* IN - branch-and-bound info */
int			var,		/* IN - variable to branch on */
int			dir,		/* IN - branch direction */
double			z		/* IN - value to give to node */
)
{
int			i;
int			nedges;
int			nmasks;
struct bbnode *		parent;
struct bbtree *		tp;
struct bbnode *		p;
struct bbnode *		p1;

	if (z >= bbip -> best_z) {
		/* Node is already worse than the cutoff value!	*/
		/* Don't bother adding it to the tree!		*/
		return;
	}

	parent	= bbip -> node;		/* current node becomes parent */
	tp	= bbip -> bbtree;	/* tree to insert in */

	gst_channel_printf (bbip -> params -> print_solve_trace,
		"@NC %4d %4d	x%d = %d	%f\n",
		tp -> snum, parent -> num, var, dir, z);

	nmasks	= tp -> nmasks;
	nedges	= bbip -> cip -> num_edges;

	/* Get a new tree node... */
	p = tp -> free;
	if (p NE NULL) {
		tp -> free = p -> next;
	}
	else {
		p = NEW (struct bbnode);
		p -> x	   = NEWA (nedges, double);
		p -> zlb   = NEWA (2 * nedges, double);
		p -> fixed = NEWA (nmasks, bitmap_t);
		p -> value = NEWA (nmasks, bitmap_t);
		p -> bheur = NEWA (nedges, double);
	}
	p -> owner	= bbip;
	p -> z		= z;
	p -> delrow_z	= z;
	p -> optimal	= FALSE;
	p -> num	= (tp -> snum)++;
	p -> iter	= 0;
	p -> parent	= parent -> num;
	p -> var	= var;
	p -> dir	= dir;
	p -> depth	= 1 + parent -> depth;
	if (dir EQ 0) {
		p -> br1cnt = parent -> br1cnt;
	}
	else {
		p -> br1cnt = parent -> br1cnt + 1;
	}
	p -> cpiter = -1;	/* force re-solve of LP. */

	/* Get most up-to-date fixed variables... */
	for (i = 0; i < nmasks; i++) {
		p -> fixed [i] = bbip -> fixed [i];
		p -> value [i] = bbip -> value [i];
	}
	SETBIT (p -> fixed, var);
	if (dir EQ 0) {
		CLRBIT (p -> value, var);
	}
	else {
		SETBIT (p -> value, var);
	}
	p -> n_uids	= 0;
	p -> bc_uids	= NULL;
	p -> bc_row	= NULL;
	p -> rstat	= NULL;
	p -> cstat	= NULL;

	/* Copy parent's LP solution, etc. */
	memcpy (p -> x, parent -> x, nedges * sizeof (p -> x [0]));
	memcpy (p -> zlb, parent -> zlb, nedges * (2 * sizeof (p -> zlb [0])));
	memcpy (p -> bheur, parent -> bheur, nedges * sizeof (p -> bheur [0]));

	/* Save the current basis (actually the parent's basis)	*/
	/* into this node.					*/
	_gst_save_node_basis (p, bbip);

	/* Insert node into depth-first list... */
	p1 = tp -> first;
	if (p1 NE NULL) {
		p1 -> prev = p;
	}
	p -> next	= p1;
	p -> prev	= NULL;
	tp -> first = p;

	/* Insert node into "best-node" heap... */
	_gst_bbheap_insert (p, tp, BEST_NODE_HEAP);

	/* Insert node into "worst-node" heap... */
	_gst_bbheap_insert (p, tp, WORST_NODE_HEAP);
}

/*
 * Append the given branch-and-bound node to the given tree.  It is added
 * to the END of the node list, and sorted into each of the heaps.
 */

	void
_gst_append_node_to_tree (

struct bbnode *		p,	/* IN - node to append to tree */
struct bbtree *		tp	/* IN - tree to append it to */
)
{
struct bbnode *		p1;

	p1 = tp -> first;
	if (p1 EQ NULL) {
		tp -> first = p;
	}
	else {
		while (p1 -> next NE NULL) {
			p1 = p1 -> next;
		}
		p1 -> next = p;
	}
	p -> next = NULL;
	p -> prev = p1;

	_gst_bbheap_insert (p, tp, BEST_NODE_HEAP);
	_gst_bbheap_insert (p, tp, WORST_NODE_HEAP);
}

/*
 * This routine deletes an arbitrary node from an arbitrary position
 * within the given branch-and-bound tree.
 */

	void
_gst_delete_node_from_bbtree (

struct bbnode *		p,	/* IN - node to delete */
struct bbtree *		tp	/* IN - tree to delete it from */
)
{
struct bbnode * p1;
struct bbnode * p2;


	/* Delete it from LIFO list... */
	p1 = p -> prev;
	p2 = p -> next;
	if (p1 NE NULL) {
		p1 -> next = p2;
	}
	else {
		tp -> first = p2;
	}
	if (p2 NE NULL) {
		p2 -> prev = p1;
	}
	p -> next = NULL;
	p -> prev = NULL;

	/* Delete it from the best-node heap... */
	bbheap_delete (p, tp, BEST_NODE_HEAP);

	/* Delete it from the worst-node heap... */
	bbheap_delete (p, tp, WORST_NODE_HEAP);
}

/*
 * This routine creates an initial, empty branch-and-bound tree.
 */

	struct bbtree *
_gst_create_bbtree (

int		nmasks		/* IN - number of edge masks */
)
{
struct bbtree *		tp;

	tp = NEW (struct bbtree);

	tp -> first		= NULL;
	tp -> free		= NULL;
	tp -> snum		= 0;
	tp -> nmasks		= nmasks;
	tp -> node_policy	= NN_BEST_NODE;

	/* Initialize best and worst order heaps */
	bbheap_init (&(tp -> heap [BEST_NODE_HEAP]), node_is_better);
	bbheap_init (&(tp -> heap [WORST_NODE_HEAP]), node_is_worse);

	return (tp);
}

/*
 * This routine initializes a branch-and-bound node heap.
 */

	static
	void
bbheap_init (

struct bbheap *		hp,	/* IN - the heap to initialize */
bbheap_func_t *		funcp	/* IN - node comparison function */
)
{
	hp -> array		= NEWA (INIT_HSIZE, struct bbnode *);
	hp -> hsize		= INIT_HSIZE;
	hp -> nheap		= 0;
	hp -> is_parent_funcp	= funcp;
}



	static
	void
bbheap_free (

struct bbheap *		hp	/* IN - heap to free up */
)
{
int		i;

	for (i = 0; i < hp -> nheap; i++) {
		destroy_bbnode (hp -> array [i]);
	}

	free ((char *) (hp -> array));
}

/*
 * This routine adds the given branch-and-bound node to the specified
 * heap of the given branch-and-bound tree.
 */

	void
_gst_bbheap_insert (

struct bbnode *		p,	/* IN - node to insert */
struct bbtree *		tp,	/* IN - branch-and-bound tree with heaps */
int			heap_no	/* IN - heap number to insert node into */
)
{
int			i;
int			j;
int			n;
struct bbheap *		hp;
struct bbnode **	tmp;
struct bbnode *		p2;
bbheap_func_t *		is_parent;

	/* Use specified heap... */
	hp = &(tp -> heap [heap_no]);

	/* Verify sufficient heap space to hold new node... */
	n = hp -> nheap;
	if (n >= hp -> hsize) {
		tmp = NEWA (2 * n, struct bbnode *);
		for (i = 0; i < n; i++) {
			tmp [i] = hp -> array [i];
		}
		free ((char *) (hp -> array));
		hp -> array = tmp;
		hp -> hsize = 2 * n;
	}

	is_parent = hp -> is_parent_funcp;

	/* Insert node into heap by placing it at the end of	*/
	/* the array and sifting up...				*/
	for (i = n; i > 0;) {
		j = ((i - 1) >> 1);
		p2 = hp -> array [j];
		if (is_parent (p2, p)) break;
		p2 -> index [heap_no] = i;
		hp -> array [i] = p2;
		i = j;
	}
	p -> index [heap_no] = i;
	hp -> array [i] = p;
	hp -> nheap = n + 1;
}

/*
 * This routine deletes a single node from the specified heap of the
 * given branch-and-bound tree.
 */

	static
	void
bbheap_delete (

struct bbnode *		p,	/* IN - the node to delete from the heap */
struct bbtree *		tp,	/* IN - branch-and-bound tree with heaps */
int			heap_no	/* IN - heap number from which to delete */
)
{
int			i;
int			j;
int			n;
struct bbheap *		hp;
struct bbnode *		p1;
struct bbnode *		p2;
struct bbnode *		p3;
bbheap_func_t *		is_parent;

	/* Use proper heap to delete from... */
	hp = &(tp -> heap [heap_no]);
	n = hp -> nheap;

	FATAL_ERROR_IF ((n <= 0) OR (n > hp -> hsize));

	/* Deleting from a heap requires three steps:		*/
	/*	1. Move the last node into the deleted node's	*/
	/*	   current position.				*/
	/*	2. Sift this replacement node up.		*/
	/*	3. Sift this replacement node down.		*/

	/* Get position of node being deleted... */
	i = p -> index [heap_no];
	FATAL_ERROR_IF (i >= n);
	FATAL_ERROR_IF (hp -> array [i] NE p);

	/* Deleted node is no longer in the array... */
	p -> index [heap_no] = -1;

	/* Heap now has one fewer element in it... */
	--n;
	hp -> nheap = n;
	if (n <= 0) {
		/* Heap is now empty -- done! */
		return;
	}

	/* Move last heap element into vacated spot. */
	p1 = hp -> array [n];
	if (p1 EQ p) {
		/* Deleting last item from heap -- we are done! */
		return;
	}
	FATAL_ERROR_IF (p1 -> index [heap_no] NE n);

	/* We now assume that node "p1" will be in position "i"... */

	is_parent = hp -> is_parent_funcp;

	/* First sift up... */
	while (i > 0) {
		j = ((i - 1) >> 1);
		p2 = hp -> array [j];
		if (is_parent (p2, p1)) break;
		p2 -> index [heap_no] = i;
		hp -> array [i] = p2;
		i = j;
	}

	/* Now sift down... */
	while (i < n) {
		j = (i << 1) + 1;
		if (j >= n) break;
		p2 = hp -> array [j];
		if ((j + 1) < n) {
			p3 = hp -> array [j + 1];
			if (is_parent (p3, p2)) {
				++j;
				p2 = p3;
			}
		}
		if (is_parent (p1, p2)) break;
		p2 -> index [heap_no] = i;
		hp -> array [i] = p2;
		i = j;
	}
	p1 -> index [heap_no] = i;
	hp -> array [i] = p1;

	hp -> nheap = n;
}

/*
 * This routine returns TRUE if-and-only-if node 1 is "better" than
 * node 2 -- in other words, node 1 must be above node 2 in the
 * "best-node" heap.
 */

	static
	int
node_is_better (

struct bbnode *		p1,	/* IN - node 1 */
struct bbnode *		p2	/* IN - node 2 */
)
{
	if (p1 -> z < p2 -> z) return (TRUE);
	if (p1 -> z > p2 -> z) return (FALSE);

	/* Objective values are equal -- use node creation	*/
	/* order to break the tie.  What we want to achieve is	*/
	/* that the "var=0" and "var=1" children of a node	*/
	/* (which will have equal objective values) get taken	*/
	/* from the "best-node" heap in proper order with	*/
	/* respect to the UP_FIRST switch.  (The node that was	*/
	/* created last should be taken from the heap first.)	*/

	if (p1 -> num >= p2 -> num) return (TRUE);
	return (FALSE);
}


/*
 * This routine returns TRUE if-and-only-if node 1 is "worse" than
 * node 2 -- in other words, node 2 must be above node 2 in the
 * "worst-node" heap.
 */

	static
	int
node_is_worse (

struct bbnode *		p1,	/* IN - node 1 */
struct bbnode *		p2	/* IN - node 2 */
)
{
	return (p1 -> z >= p2 -> z);
}

/*
 * This routine destroys the given branch-and-bound info structure,
 * freeing all of the memory it points to.
 */

	void
_gst_destroy_bbinfo (

struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
struct bbtree *		bbtree;
struct bbnode *		p1;
struct bbnode *		p2;

	bbip -> ubip = NULL;

	if (bbip -> statp NE NULL) {
		free ((char *) (bbip -> statp));
	}
	bbip -> value = NULL;
	bbip -> fixed = NULL;
	free ((char *) (bbip -> dj));
	if (bbip -> slack NE NULL) {
		free ((char *) (bbip -> slack));
	}
	bbip -> node = NULL;
	bbip -> _smt  = NULL;

	_gst_free_cutset_separation_formulation (bbip -> csip);

	bbtree = bbip -> bbtree;
	if (bbtree NE NULL) {
		p1 = bbtree -> free;
		for (;;) {
			if (p1 EQ NULL) break;
			p2 = p1 -> next;
			destroy_bbnode (p1);
			p1 = p2;
		}
		bbtree -> first = NULL;
		bbtree -> free = NULL;

		/* Every suspended node is in BOTH heaps.  Free only one of them! */
		bbheap_free (&(bbtree -> heap [BEST_NODE_HEAP]));

		/* Only free the array on this one, not the nodes... */
		free ((char *) (bbtree -> heap [WORST_NODE_HEAP].array));
		free ((char *) bbtree);
	}

	if (bbip -> cpool NE NULL) {
		_gst_free_constraint_pool (bbip -> cpool);
	}

	if (bbip -> lp NE NULL) {
		_gst_destroy_initial_formulation (bbip);
	}

	if (bbip -> lpmem NE NULL) {
		free ((char *) (bbip -> lpmem));
	}

	_gst_stop_using_lp_solver ();

	/* These items all belong to the gst_hypergraph.  Just zap them. */
	bbip -> cip		= NULL;
	bbip -> vert_mask	= NULL;
	bbip -> edge_mask	= NULL;

	/* Free failed components */
	_gst_free_congested_component_list (bbip -> failed_fcomps);

	free ((char *) bbip);
}

/*
 * Destroy the given branch-and-bound node, freeing all of the memory
 * it refers to.
 */

	static
	void
destroy_bbnode (

struct bbnode *		p		/* IN - node to destroy */
)
{
	free ((char *) (p -> fixed));
	free ((char *) (p -> value));

	if (p -> x NE NULL) {
		free ((char *) (p -> x));
	}
	if (p -> zlb NE NULL) {
		free ((char *) (p -> zlb));
	}
	if (p -> bc_uids NE NULL) {
		free ((char *) (p -> bc_uids));
	}
	if (p -> bc_row NE NULL) {
		free ((char *) (p -> bc_row));
	}
	if (p -> rstat NE NULL) {
		free ((char *) (p -> rstat));
	}
	if (p -> cstat NE NULL) {
		free ((char *) (p -> cstat));
	}
	free ((char *) (p -> bheur));
	free ((char *) p);
}
