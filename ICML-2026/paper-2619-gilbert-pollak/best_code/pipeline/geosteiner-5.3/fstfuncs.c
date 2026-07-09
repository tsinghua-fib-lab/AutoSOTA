/***********************************************************************

	$Id: fstfuncs.c,v 1.11 2022/11/19 13:45:52 warme Exp $

	File:	fstfuncs.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 2002, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	General functions used by the fst generators.

************************************************************************

	Modification Log:

	a-1:	06/17/02	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.
	e-4:	11/18/2022	warme
		: Fix -Wall issue.

************************************************************************/

#include "fstfuncs.h"

#include "fatal.h"
#include "logic.h"
#include "memory.h"
#include "p1read.h"
#include "point.h"
#include "steiner.h"
#include <stdlib.h>
#include <string.h>


/*
 * Global Routines
 */

struct pset *		_gst_create_pset (int nterms, double * terms);
int			_gst_generate_duplicate_terminal_groups (
						struct pset *	pts,
						int *		xorder,
						int ***		grps);
void			_gst_initialize_hypergraph (
						struct gst_hypergraph *	cip);

struct full_set **	_gst_put_trees_in_array (
					struct full_set *	fsp,
					int *			acount);
struct pset *		_gst_remove_duplicates (
					struct pset *	pts,
					int		ndg,
					int **		list,
					int **		fwd_map,
					int **		rev_map);

/*
 * Create a point set from a simple array of (x,y) pairs.
 */

	struct pset *
_gst_create_pset (

int		nterms,
double *	terms
)
{
int		n;
struct point *	p;
struct pset *	pts;

	pts = NEW_PSET (nterms);

	/* Zero out the entire point set. */
	ZERO_PSET (pts, nterms);

	/* Copy the given terminals to the pset array */
	pts -> n = nterms;
	n = 0;
	p = &(pts -> a [0]);
	while(n != nterms) {
		p -> x	= *terms++;
		p -> y	= *terms++;
		++n;
		++p;
	}

	return pts;
}

/*
 * Initialize various fields in a hypergraph.
 */
	void
_gst_initialize_hypergraph (

struct gst_hypergraph *	cip
)
{
int	i;
int	nedges;
int	nmasks;

	nedges = cip -> num_edges;
	nmasks = BMAP_ELTS (nedges);

	cip -> initial_edge_mask = NEWA (nmasks, bitmap_t);
	memset (cip -> initial_edge_mask, 0, nmasks * sizeof (bitmap_t));
	for (i = 0; i < nedges; i++)
		SETBIT (cip -> initial_edge_mask, i);

	cip -> required_edges = NEWA (nmasks, bitmap_t);
	memset (cip -> required_edges, 0, nmasks * sizeof (bitmap_t));

	_gst_init_term_trees (cip);
	cip -> inc_edges = NULL; /* _gst_compute_basic_incompat (cip); */
}

/*
 * This routine finds all pairs of points whose coordinates are exactly
 * identical.  This is a degeneracy that can cause successive algorithms
 * much heartburn, so we take care of them immediately by keeping only
 * the earliest (lowest index) point, and marking the others as duplicates.
 * We generate a partition of such terminals into subsets -- if terminals
 * A and B reside in the same subset then they have identical coordinates.
 * We refer to such a subset as a Duplicate Terminal Group.
 *
 * For each terminal group we retain ONLY THE FIRST member -- the terminal
 * map bits are turned OFF for all other members of a terminal group,
 * effectively eliminating those terminals from the problem.
 */

	int
_gst_generate_duplicate_terminal_groups (

struct pset *		pts,	/* IN - original terminal set */
int *			xorder, /* IN - terminal numbers sorted by X coord */
int ***			grps	/* OUT - duplicate terminal groups */
)
{
int			i;
int			j;
int			n;
int			n_grps;
struct point *		p0;
struct point *		p1;
struct point *		p2;
int *			ip;
int **			real_ptrs;
int *			real_terms;
int **			ptrs;
int *			terms;

	n = pts -> n;

	n_grps = 0;

	ptrs	= NEWA (n + 1, int *);
	terms	= NEWA (n, int);

	ip = &terms [0];
	for (i = 1; i < n; ) {
		p0 = &(pts -> a [xorder [i - 1]]);
		p1 = &(pts -> a [xorder [i]]);

		if ((p0 -> y NE p1 -> y) OR (p0 -> x NE p1 -> x)) {
			/* Not identical. */
			++i;
			continue;
		}

		/* Terminals xorder[i-1] and xorder[i] are identical. */

		for (j = i + 1; j < n; j++) {
			p2 = &(pts -> a [xorder [j]]);
			if (p0 -> y NE p2 -> y) break;
			if (p0 -> x NE p2 -> x) break;
		}
		/* j now points to the first non-equal terminal */

		/* Make a new duplicate terminal group... */
		ptrs [n_grps++] = ip;
		*ip++ = xorder [i - 1];
		while (i < j) {
			*ip++ = xorder [i++];
		}

		/* Skip whole group of coincident points and continue */
	}
	ptrs [n_grps] = ip;

	if (n_grps <= 0) {
		*grps = NULL;
	}
	else {
		/* Transfer to permanent memory of proper size... */
		real_ptrs = NEWA (n_grps + 1, int *);
		real_terms = NEWA (ip - terms, int);
		(void) memcpy ((char *) real_terms,
			       (char *) terms,
			       (ip - terms) * sizeof (int));
		for (i = 0; i <= n_grps; i++) {
			real_ptrs [i] = &real_terms [ptrs [i] - ptrs [0]];
		}
		*grps = real_ptrs;
	}

	free ((char *) terms);
	free ((char *) ptrs);

	return (n_grps);
}

/*
 * This routine removes all but the first terminal in each duplicate
 * terminal group.  We also prepare arrays for mapping forward from
 * old to new terminal numbers, and backward from new terminal numbers
 * to old.
 */

	struct pset *
_gst_remove_duplicates (

struct pset *		pts,		/* IN - original point set */
int			ndg,		/* IN - number of duplicate groups */
int **			list,		/* IN - list of duplicate groups */
int **			fwd_map_ptr,	/* OUT - map to renumber old to new */
int **			rev_map_ptr	/* OUT - map to renumber new to old */
)
{
int		i;
int		j;
int		n;
int		kmasks;
int		numdel;
int		new_n;
int *		ip1;
int *		ip2;
int *		fwd;
int *		rev;
bitmap_t *	deleted;
struct pset *	newpts;

	n = pts -> n;

	kmasks = BMAP_ELTS (n);
	deleted = NEWA (kmasks, bitmap_t);
	for (i = 0; i < kmasks; i++) {
		deleted [i] = 0;
	}

	numdel = 0;
	for (i = 0; i < ndg; i++) {
		ip1 = list [i];
		ip2 = list [i + 1];

		/* Retain the first in this group, exclude all	*/
		/* of the others...				*/
		while (++ip1 < ip2) {
			FATAL_ERROR_IF (BITON (deleted, *ip1));
			++numdel;
			SETBIT (deleted, *ip1);
		}
	}

	new_n = n - numdel;

	fwd = NEWA (n, int);
	rev = NEWA (new_n, int);
	newpts = NEW_PSET (new_n);
	ZERO_PSET (newpts, new_n);
	newpts -> n = new_n;
	j = 0;
	for (i = 0; i < n; i++) {
		if (BITON (deleted, i)) {
			fwd [i] = -1;
		}
		else {
			newpts -> a [j].x	= pts -> a [i].x;
			newpts -> a [j].y	= pts -> a [i].y;
			rev [j] = i;
			fwd [i] = j;
			++j;
		}
	}

	free ((char *) deleted);

	*fwd_map_ptr = fwd;
	*rev_map_ptr = rev;

	return (newpts);
}

/*
 * This routine allocates an array of pointers that lets us access a particular
 * tree number directly.
 */

	struct full_set **
_gst_put_trees_in_array (

struct full_set *	fsp,		/* IN - list of full-trees. */
int *			acount		/* OUT - count of array elements. */
)
{
struct full_set **	ap;
struct full_set *	p;
int			count;
struct full_set **	array;

	count = 0;
	for (p = fsp; p NE NULL; p = p -> next) {
		++count;
	}

	array = NEWA (count, struct full_set *);

	ap = array;
	for (p = fsp; p NE NULL; p = p -> next) {
		*ap++ = p;
	}

	*acount = count;

	return (array);
}
