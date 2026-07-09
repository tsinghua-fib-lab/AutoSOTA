/***********************************************************************

	$Id: mst.c,v 1.10 2022/11/19 13:45:53 warme Exp $

	File:	mst.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines to compute Minimum Spanning Trees.

************************************************************************

	Modification Log:

	a-1:	02/28/2001	warme
		: Created.  Gathered all copies of these routines
		:  into this one file.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.  Upgrade fatals.

************************************************************************/

#include "mst.h"

#include "dsuf.h"
#include "fatal.h"
#include "logic.h"
#include "steiner.h"


/*
 * Global Routines
 */

int		_gst_mst_edge_list (int			n,
				    int			nedges,
				    struct edge *	edge_list,
				    struct edge *	edges);
void		_gst_sort_edge_list (struct edge * a, int n);


/*
 * Local Routines
 */

	/* none */

/*
 * This routine computes the MST of a given list of edges.
 */

	int
_gst_mst_edge_list (

int			n,		/* IN - number of vertices */
int			nedges,		/* IN - number of edges */
struct edge *		edge_list,	/* IN - list of edges */
struct edge *		edges		/* OUT - MST edge list */
)
{
int		i;
int		mst_edge_count;
int		components;
int		max_vert;
struct edge *	ep;
struct edge *	ep_endp;
int		root1;
int		root2;
struct dsuf	sets;

	_gst_sort_edge_list (edge_list, nedges);

	/* Don't assume that the vertex numbers are well-behaved,	*/
	/* except that they must be non-negative.  We do a quick scan	*/
	/* to determine the largest vertex number and then allocate	*/
	/* a union-find data structure large enough to handle it.  Note	*/
	/* that we then use this union-find data structure in a		*/
	/* completely sparse way -- we only ever access set items for	*/
	/* vertices that are named by an edge.				*/

	max_vert = 1;		/* avoid zero-size union-find... */
	ep = edge_list;
	for (i = 0; i < nedges; i++, ep++) {
		if (ep -> p1 > max_vert) {
			max_vert = ep -> p1;
		}
		if (ep -> p2 > max_vert) {
			max_vert = ep -> p2;
		}
	}

	_gst_dsuf_create (&sets, max_vert + 1);

	/* Note that it is not a problem to "makeset" a vertex more	*/
	/* than once...							*/
	ep = edge_list;
	for (i = 0; i < nedges; i++, ep++) {
		_gst_dsuf_makeset (&sets, ep -> p1);
		_gst_dsuf_makeset (&sets, ep -> p2);
	}

	components = n;
	mst_edge_count = 0;
	ep = edge_list;
	ep_endp = (ep + nedges);

	while (components > 1) {
		if (ep >= ep_endp) {
			/* Ran out of edges before MST complete! */
			FATAL_ERROR;
		}
		root1 = _gst_dsuf_find (&sets, ep -> p1);
		root2 = _gst_dsuf_find (&sets, ep -> p2);
		if (root1 NE root2) {
			_gst_dsuf_unite (&sets, root1, root2);
			*edges = *ep;
			++edges;
			++mst_edge_count;
			--components;
		}
		++ep;
	}

	_gst_dsuf_destroy (&sets);

	return (mst_edge_count);
}

/*
 * This routine sorts the given edge list in INCREASING order by edge length.
 */

	void
_gst_sort_edge_list (

struct edge *		a,	/* IN/OUT - array of edges to be sorted. */
int			n	/* IN - number of elements in array. */
)
{
int		i, j, k;
int		v1a, v1b, v2a, v2b, tmpv;
struct edge	tmp;
struct edge *	p1;
struct edge *	p2;

	/* Construct the heap via sift-downs, in O(n) time. */
	for (k = n >> 1; k >= 0; k--) {
		j = k;
		for (;;) {
			i = (j << 1) + 1;
			if (i + 1 < n) {
				/* Increment i (to right subchild of j) */
				/* if the right subchild is greater. */
				p1 = &a [i];
				p2 = &a [i + 1];
				if (p2 -> len > p1 -> len) {
					++i;
				}
				else if (p2 -> len EQ p1 -> len) {
					/* Edges are the same length: */
					/* compare lexicographically. */
					v1a = p1 -> p1;
					v1b = p1 -> p2;
					if (v1a > v1b) {
						tmpv = v1a;
						v1a = v1b;
						v1b = tmpv;
					}
					v2a = p2 -> p1;
					v2b = p2 -> p2;
					if (v2a > v2b) {
						tmpv = v2a;
						v2a = v2b;
						v2b = tmpv;
					}
					if ((v2a > v1a) OR
					    ((v2a EQ v1a) AND (v2b > v1b))) {
						++i;
					}
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			p1 = &a [j];
			p2 = &a [i];
			if (p1 -> len > p2 -> len) {
				/* Greatest child is smaller.  Sift-	*/
				/* down is done. */
				break;
			}
			else if (p1 -> len EQ p2 -> len) {
				/* Edges are same length:	*/
				/* compare lexicographically.	*/
				v1a = p1 -> p1;
				v1b = p1 -> p2;
				if (v1a > v1b) {
					tmpv = v1a;
					v1a = v1b;
					v1b = tmpv;
				}
				v2a = p2 -> p1;
				v2b = p2 -> p2;
				if (v2a > v2b) {
					tmpv = v2a;
					v2a = v2b;
					v2b = tmpv;
				}
				if ((v1a > v2a) OR
				    ((v1a EQ v2a) AND (v1b > v2b))) {
					/* Greatest child is smaller.  Sift- */
					/* down is done. */
					break;
				}
			}
			/* Sift down and continue. */
			tmp = *p1;
			*p1 = *p2;
			*p2 = tmp;
			j = i;
		}
	}

	/* Now do actual sorting.  Exchange first/last and sift down. */
	while (n > 1) {
		/* Largest is at a [0], swap with a [n-1],	*/
		/* thereby putting it into final position.	*/
		--n;
		tmp = a [0];
		a [0] = a [n];
		a [n] = tmp;

		/* Now restore the heap by sifting down at index 0. */
		j = 0;
		for (;;) {
			i = (j << 1) + 1;
			if (i + 1 < n) {
				/* Increment i (to right subchild of j) */
				/* if the right subchild is greater. */
				p1 = &a [i];
				p2 = &a [i + 1];
				if (p2 -> len > p1 -> len) {
					++i;
				}
				else if (p2 -> len EQ p1 -> len) {
					/* Edges are the same length: */
					/* compare lexicographically. */
					v1a = p1 -> p1;
					v1b = p1 -> p2;
					if (v1a > v1b) {
						tmpv = v1a;
						v1a = v1b;
						v1b = tmpv;
					}
					v2a = p2 -> p1;
					v2b = p2 -> p2;
					if (v2a > v2b) {
						tmpv = v2a;
						v2a = v2b;
						v2b = tmpv;
					}
					if ((v2a > v1a) OR
					    ((v2a EQ v1a) AND (v2b > v1b))) {
						++i;
					}
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			p1 = &a [j];
			p2 = &a [i];
			if (p1 -> len > p2 -> len) {
				/* Greatest child is smaller.  Sift-	*/
				/* down is done. */
				break;
			}
			else if (p1 -> len EQ p2 -> len) {
				/* Edges are same length:	*/
				/* compare lexicographically.	*/
				v1a = p1 -> p1;
				v1b = p1 -> p2;
				if (v1a > v1b) {
					tmpv = v1a;
					v1a = v1b;
					v1b = tmpv;
				}
				v2a = p2 -> p1;
				v2b = p2 -> p2;
				if (v2a > v2b) {
					tmpv = v2a;
					v2a = v2b;
					v2b = tmpv;
				}
				if ((v1a > v2a) OR
				    ((v1a EQ v2a) AND (v1b > v2b))) {
					/* Greatest child is smaller.  Sift- */
					/* down is done. */
					break;
				}
			}
			/* Sift down and continue. */
			tmp = *p1;
			*p1 = *p2;
			*p2 = tmp;
			j = i;
		}
	}
}
