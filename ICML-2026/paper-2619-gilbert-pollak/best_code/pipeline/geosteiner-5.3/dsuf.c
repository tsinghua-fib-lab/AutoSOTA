/***********************************************************************

	$Id: dsuf.c,v 1.10 2022/11/19 13:45:51 warme Exp $

	File:	dsuf.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1998, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	The "disjoint set union-find" data structure.

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

#include "dsuf.h"

#include "fatal.h"
#include "logic.h"
#include "memory.h"
#include <stdlib.h>


/*
 * Global Routines
 */

void		_gst_dsuf_create (struct dsuf * dsp, int n);
void		_gst_dsuf_destroy (struct dsuf * dsp);
int		_gst_dsuf_find (struct dsuf * dsp, int i);
void		_gst_dsuf_makeset (struct dsuf * dsp, int i);
void		_gst_dsuf_unite (struct dsuf * dsp, int i, int j);


/*
 * Local Routines
 */

	/* none */

/*
 * This routine creates a collection of N disjoint sets.  They are left
 * uninitialized so that a sparse collection can be accessed quickly.
 */

	void
_gst_dsuf_create (

struct dsuf *	dsp,		/* IN/OUT - sets to create */
int		n		/* IN - number of disjoint sets */
)
{
	FATAL_ERROR_IF (n <= 0);

	dsp -> set_size		= n;
	dsp -> parent		= NEWA (n, int);
	dsp -> rank		= NEWA (n, int);
}


/*
 * Destroy the given collection of disjoint sets.
 */

	void
_gst_dsuf_destroy (

struct dsuf *	dsp		/* IN - sets to destroy */
)
{
	free ((char *) (dsp -> rank));
	free ((char *) (dsp -> parent));

	dsp -> set_size	= 0;
	dsp -> parent	= NULL;
	dsp -> rank	= NULL;
}

/*
 * This routine makes a single disjoint set for item "i".
 */

	void
_gst_dsuf_makeset (

struct dsuf *	dsp,		/* IN - collection of sets */
int		i		/* IN - item to make into a disjoint set */
)
{
	FATAL_ERROR_IF ((i < 0) OR (i >= dsp -> set_size));

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
_gst_dsuf_unite (

struct dsuf *	dsp,		/* IN - collection of sets */
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

	if (ri EQ rj) {
		/* Both subtrees have the same maximum depth.  We	*/
		/* arbitrarily choose I to be underneath J.  The rank	*/
		/* of J must then increase.				*/
		dsp -> parent [i] = j;
		dsp -> rank [j]	  = rj + 1;
	}
	else if (ri > rj) {
		/* Tree I is (probably) deeper.  Putting J underneath	*/
		/* will not increase I's rank.				*/
		dsp -> parent [j] = i;
	}
	else {
		/* Tree J is (probably) deeper... */
		dsp -> parent [i] = j;
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
_gst_dsuf_find (

struct dsuf *	dsp,		/* IN - collection of sets */
int		i		/* IN - item to find cannonical item for */
)
{
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
	while (TRUE) {
		k = dsp -> parent [j];
		if (j EQ k) break;
		j = k;
	}

	/* Now compress the path (make all items in chain point directly */
	/* at the root K) -- we never have to do this search again!	 */
	while (i NE k) {
		j = dsp -> parent [i];
		dsp -> parent [i] = k;
		i = j;
	}

	return (k);
}
