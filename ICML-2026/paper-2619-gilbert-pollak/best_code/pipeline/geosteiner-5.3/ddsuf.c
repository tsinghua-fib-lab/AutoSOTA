/***********************************************************************

	$Id: ddsuf.c,v 1.10 2022/11/19 13:45:50 warme Exp $

	File:	ddsuf.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1998, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	The "dynamic disjoint set union-find" data structure.

************************************************************************

	Modification Log:

	a-1:	11/03/98	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.

************************************************************************/

#include "ddsuf.h"

#include "fatal.h"
#include "logic.h"
#include "memory.h"
#include <stdlib.h>


/*
 * Global Routines
 */

void	_gst_ddsuf_create (struct ddsuf * dsp, int n);
void	_gst_ddsuf_destroy (struct ddsuf * dsp);
int	_gst_ddsuf_find (struct ddsuf * dsp, int i);
void	_gst_ddsuf_makeset (struct ddsuf * dsp, int i);
void	_gst_ddsuf_restore (struct ddsuf * dsp, int old_state);
void	_gst_ddsuf_unite (struct ddsuf * dsp, int i, int j);


/*
 * Local Macros
 */

#define VERIFY_STACK(dsp,n) \
{	if ((dsp) -> sp + (n) > (dsp) -> stack_size) { \
		grow_stack ((dsp),(n)); \
	} \
}

#define STORE(dsp, loc, newval) \
{	struct ddsuf_restore *	rp; \
	rp = &((dsp) -> stack [(dsp) -> sp++]); \
	rp -> ptr = &(loc); \
	rp -> val = (loc); \
	(loc) = (newval); \
}


/*
 * Local Routines
 */

static void	grow_stack (struct ddsuf *, int);

/*
 * This routine creates a collection of N disjoint sets.  They are left
 * uninitialized so that a sparse collection can be accessed quickly.
 */

	void
_gst_ddsuf_create (

struct ddsuf *	dsp,		/* IN/OUT - sets to create */
int		n		/* IN - number of disjoint sets */
)
{
	FATAL_ERROR_IF (n <= 0);

	dsp -> set_size		= n;
	dsp -> parent		= NEWA (n, int);
	dsp -> rank		= NEWA (n, int);

	n *= 2;
	dsp -> stack		= NEWA (n, struct ddsuf_restore);
	dsp -> sp		= 0;
	dsp -> stack_size	= n;
}


/*
 * Destroy the given collection of disjoint sets.
 */

	void
_gst_ddsuf_destroy (

struct ddsuf *	dsp		/* IN - sets to destroy */
)
{
	free ((char *) (dsp -> stack));
	free ((char *) (dsp -> rank));
	free ((char *) (dsp -> parent));

	dsp -> parent		= NULL;
	dsp -> rank		= NULL;
	dsp -> set_size		= 0;
	dsp -> stack		= NULL;
	dsp -> sp		= 0;
	dsp -> stack_size	= 0;
}

/*
 * This routine makes a single disjoint set for item "i".
 * The "makeset" operation is not reversable.
 */

	void
_gst_ddsuf_makeset (

struct ddsuf *	dsp,		/* IN - collection of sets */
int		i		/* IN - item to make into a disjoint set */
)
{
	if ((i < 0) OR (i >= dsp -> set_size)) {
		/* Item out of bounds. */
		FATAL_ERROR;
	}
	dsp -> parent [i]	= i;
	dsp -> rank [i]		= 0;
}

/*
 * This routine "unites" two sets that were previously disjoint.  I and J
 * must be the "canonical" member of each disjoint set (i.e. they must
 * each be the output of a "find" operation), and must be distinct.
 *
 * We perform the "union by rank" heuristic here.
 */

	void
_gst_ddsuf_unite (

struct ddsuf *	dsp,		/* IN - collection of sets */
int		i,		/* IN - first set to unite */
int		j		/* IN - second set to unite */
)
{
int		ri;
int		rj;

	FATAL_ERROR_IF ((i < 0) OR (i >= dsp -> set_size));
	FATAL_ERROR_IF ((j < 0) OR (j >= dsp -> set_size));
	FATAL_ERROR_IF (i EQ j);

	ri = dsp -> rank [i];
	rj = dsp -> rank [j];

	VERIFY_STACK (dsp, 2);

	if (ri EQ rj) {
		/* Both subtrees have the same maximum depth.  We	*/
		/* arbitrarily choose I to be underneath J.  The rank	*/
		/* of J must then increase.				*/
		STORE (dsp, dsp -> parent [i], j);
		STORE (dsp, dsp -> rank [j], rj + 1);
	}
	else if (ri > rj) {
		/* Tree I is (probably) deeper.  Putting J underneath	*/
		/* will not increase I's rank.				*/
		STORE (dsp, dsp -> parent [j], i);
	}
	else {
		/* Tree J is (probably) deeper... */
		STORE (dsp, dsp -> parent [i], j);
	}
}

/*
 * This routine, given a member I of one of the disjoint sets A, will
 * choose a cannonical member J of set A and return it.  Until set A gets
 * united with some other set, find (I) will always return the same J.
 *
 * This routine performs the "path compression" heuristic.
 */

	int
_gst_ddsuf_find (

struct ddsuf *	dsp,		/* IN - collection of sets */
int		i		/* IN - item to find cannonical item for */
)
{
int		n;
int		j;
int		k;

	/* Yes, I know this routine is very elegent when coded	*/
	/* recursively...  Here's the iterative version.	*/

	j = dsp -> parent [i];
	if (i EQ j) {
		/* A cannonical element has itself as parent. */
		return (i);
	}

	/* We must search up the tree -- and compress when done... */
	n = 0;
	while (TRUE) {
		k = dsp -> parent [j];
		if (j EQ k) break;
		++n;
		j = k;
	}

	VERIFY_STACK (dsp, n);

	/* Now compress the path (make all items in chain point directly */
	/* at the root K) -- we never have to do this search again!	 */
	for (;;) {
		j = dsp -> parent [i];
		if (j EQ k) break;
		STORE (dsp, dsp -> parent [i], k);
		i = j;
	}

	return (k);
}

/*
 * Restore a dynamic DSUF to a previous state.
 */

	void
_gst_ddsuf_restore (

struct ddsuf *	dsp,		/* IN - collection of sets */
int		old_state	/* IN - previous state to restore */
)
{
int			sp;
struct ddsuf_restore *	rp;

	sp = dsp -> sp;
	if ((old_state < 0) OR (sp < old_state)) {
		/* Invalid state to restore. */
		FATAL_ERROR;
	}
	else if (sp > old_state) {
		--sp;
		rp = &(dsp -> stack [sp]);
		for (;;) {
			*(rp -> ptr) = rp -> val;
			if (sp <= old_state) break;
			--sp;
			--rp;
		}
		dsp -> sp = sp;
	}
}

/*
 * Grow the stack to be AT LEAST the given amount larger than it is.
 * We are guaranteed that (dsp -> sp + npush > dsp -> stack_size),
 * so action MUST be taken.
 */

	static
	void
grow_stack (

struct ddsuf *	dsp,		/* IN - collection of sets */
int		npush		/* IN - number of STOREs we intend to push */
)
{
int			i;
int			old_size;
int			new_size;
struct ddsuf_restore *	newp;

	old_size = dsp -> stack_size;
	new_size = 2 * old_size;
	if (new_size < dsp -> sp + npush) {
		new_size = dsp -> sp + npush;
	}

	newp = NEWA (new_size, struct ddsuf_restore);
	for (i = 0; i < dsp -> sp; i++) {
		newp [i] = dsp -> stack [i];
	}
	free ((char *) (dsp -> stack));
	dsp -> stack = newp;
	dsp -> stack_size = new_size;
}
