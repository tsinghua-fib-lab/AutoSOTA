/***********************************************************************

	$Id: rmst.c,v 1.10 2022/11/19 13:45:54 warme Exp $

	File:	rmst.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines to compute rectilinear Minimum Spanning Trees.
	Also contains the 1-Steiner heuristic of Kahng and Robins.

************************************************************************

	Modification Log:

	a-1:	02/20/93	warme
		: Created.
	b-1:	02/28/2001	warme
		: Changes for 3.1 release.
		: Use common Kruskal code in mst.c.
		: Rename build_edges to be build_rect_edges.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.  Upgrade fatals.

************************************************************************/

#include "rmst.h"

#include "dsuf.h"
#include "emptyr.h"
#include "fatal.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "mst.h"
#include "point.h"
#include <stdlib.h>
#include "steiner.h"
#include <string.h>


/*
 * Global Routines
 */

int		_gst_kahng_robins (struct pset *, dist_t, struct edge *);
dist_t		_gst_kahng_robins_length (struct pset *, dist_t);
int		_gst_rect_mst (struct pset *, struct edge *, bitmap_t *);
dist_t		_gst_rect_mst_length (struct pset *);


/*
 * External References
 */

	/* none */


/*
 * Local Routines
 */

static int		build_rect_edges (struct pset *,
					  struct edge **,
					  int,
					  bitmap_t *);
static dist_t		kr_main (struct pset *, dist_t);

/*
 * This routine computes the total length of the Minimum Spanning Tree
 * of the given set of points.
 */

	dist_t
_gst_rect_mst_length (

struct pset *		pts	/* IN - point set to find MST length of. */
)
{
int		i;
int		nedges;
dist_t		total;
struct edge *	ep;
struct edge *	edges;

	edges = NEWA (pts -> n - 1, struct edge);

	nedges = _gst_rect_mst (pts, &edges [0], NULL);

	FATAL_ERROR_IF (nedges NE pts -> n - 1);

	total = 0;
	ep = &edges [0];
	for (i = 0; i < nedges; i++) {
		total += ep -> len;
		++ep;
	}

	free ((char *) edges);

	return (total);
}

/*
 * This routine computes a rectilinear Minimum Spanning Tree for the
 * given point set.  The output is a list of edges.
 *
 * The caller may optionally provide an empty rectangle bit-matrix.
 * This info can substantially speed up the algorithm and decrease
 * memory requirements by limiting the number of edges stored and sorted.
 * If such info is unavailable, supply empty_rect = NULL.
 */

	int
_gst_rect_mst (

struct pset *		pts,		/* IN - point set. */
struct edge *		edges,		/* OUT - edge list. */
bitmap_t *		empty_rect	/* IN - empty rectangle info */
)
{
int		nedges;
int		mst_edge_count;
struct edge *	edge_array;

	nedges = build_rect_edges (pts, &edge_array, 0, empty_rect);

	mst_edge_count = _gst_mst_edge_list (pts -> n,
					     nedges,
					     &edge_array [0],
					     edges);

	free ((char *) edge_array);

	return (mst_edge_count);
}

/*
 * This routine builds an edge-list containing all pairs of points.
 *
 * If non-NULL, the empty-rectangle info is used to greatly reduce
 * the number of edges produced.
 */

	static
	int
build_rect_edges (

struct pset *		pts,		/* IN - set of points */
struct edge **		edges_out,	/* OUT - edge list */
int			at_least,	/* IN - allocate at least this many */
					/*	edges */
bitmap_t *		empty_rect	/* IN - empty rectangle info */
)
{
int		i;
int		j;
int		n;
int		nedges;
int		nalloc;
struct edge *	edges;
struct point *	p1;
struct point *	p2;

	n = pts -> n;

	if (empty_rect EQ NULL) {
		/* Generate all N choose 2 edges. */
		nedges = (n * (n - 1)) >> 1;
		nalloc = nedges;
		if (nalloc < at_least) {
			nalloc = at_least;
		}
		edges = NEWA (nalloc, struct edge);
		*edges_out = edges;

		p1 = &(pts -> a [0]);
		for (i = 0; i < n; i++, p1++) {
			p2 = (p1 + 1);
			for (j = i + 1; j < n; j++, p2++) {
				edges -> len	= RDIST (p1, p2);
				edges -> p1	= i;
				edges -> p2	= j;
				++edges;
			}
		}
	}
	else {
		/* Generate a sparse set of edges. */
		nedges = _gst_count_empty_rectangles (empty_rect, n);
		nalloc = nedges;
		if (nalloc < at_least) {
			nalloc = at_least;
		}
		edges = NEWA (nalloc, struct edge);
		*edges_out = edges;

		p1 = &(pts -> a [0]);
		for (i = 0; i < n; i++, p1++) {
			p2 = (p1 + 1);
			for (j = i + 1; j < n; j++, p2++) {
				if (NOT _gst_is_empty_rectangle (empty_rect, i, j)) continue;

				edges -> len	= RDIST (p1, p2);
				edges -> p1	= i;
				edges -> p2	= j;
				++edges;
			}
		}
	}

	return (nedges);
}

/*
 * This routine computes an approximate Steiner Minimal Tree using the
 * heuristic method of Kahng and Robins.
 *
 * Inputs:	pts	- the input point set -- MODIFIED to contain
 *			  additional Steiner points.
 *
 *		limit	- we quit if we ever get a tree shorter than this.
 *
 * Outputs:	edges	- The list of edges in the Steiner tree.
 *
 * Returns:		- Number of edges in Steiner tree.
 */

	int
_gst_kahng_robins (

struct pset *	pts,		/* IN/OUT - terminals and steiner points. */
dist_t		limit,		/* IN - quit if tree below this in length. */
struct edge *	edges		/* OUT - edge list for Steiner tree. */
)
{
int		nedges;

	/* Do the Kahng-Robins stuff -- ignore the resulting tree length... */
	(void) kr_main (pts, limit);

	nedges = _gst_rect_mst (pts, edges, NULL);

	return (nedges);
}

/*
 * This routine computes the LENGTH of an approximate Steiner Minimal Tree
 * using the heuristic method of Kahng and Robins.  We use this routine
 * when we don't care what the tree looks like and we only want to know
 * its length.
 *
 * Inputs:	pts	- the input point set.
 *		limit	- we quit if we ever get a tree shorter than this.
 *
 * Returns:		- length of Steiner tree obtained.
 */

	dist_t
_gst_kahng_robins_length (

struct pset *	in_pts,		/* IN - terminals. */
dist_t		limit		/* IN - quit if tree below this in length. */
)
{
int		n;
dist_t		len;
struct pset *	pts_copy;

	n = in_pts -> n;

	/* Make a temporary buffer big enough to handle the	*/
	/* additional Steiner points that KR will append...	*/

	pts_copy = NEW_PSET (2 * n);

	/* Make a local copy of the point set that we can modify... */
	COPY_PSET (pts_copy, in_pts);

	len = kr_main (pts_copy, limit);

	free ((char *) pts_copy);

	return (len);
}

/*
 * This routine is the guts of the Kahng-Robins heuristic for computing
 * an approximate Steiner Minimal Tree.
 *
 * Inputs:	pts	- the input point set -- MODIFIED to contain
 *			  additional Steiner points.
 *
 *		limit	- we quit if we ever get a tree shorter than this.
 *
 * Returns:		- length of approximate Steiner Minimal Tree.
 */

	static
	dist_t
kr_main (

struct pset *	pts,		/* IN/OUT - terminals and steiner points. */
dist_t		limit		/* IN - quit if tree below this in length. */
)
{
int		i;
int		j;
int		k;
int		m;
int		n;
int		nterms;
int		max_points;
int		max_edges;
int		components;
dist_t		last;
dist_t		best;
dist_t		len;
int		best_j;
int		best_k;
int		nedges;
struct point *	p1;
struct point *	p2;
struct point *	p3;
struct point *	newpt;
struct edge *	ep;
struct edge *	endp;
struct edge *	ep1;
struct edge *	ep2;
struct edge *	endp1;
struct edge *	endp2;
int		root1;
int		root2;
int		kmasks;
bitmap_t *	mark_rowp;
int		new_nedges;
struct point	bestpt;
struct edge *	new_edges;
struct edge *	edge_array;
bitmap_t *	mark;
struct dsuf	sets;

	nterms		= pts -> n;
	kmasks		= BMAP_ELTS (nterms);
	max_points	= nterms + nterms - 2;
	max_edges	= max_points * (max_points - 1) / 2;

	new_edges	= NEWA (max_points, struct edge);
	mark		= NEWA (nterms * kmasks, bitmap_t);

	nedges = build_rect_edges (pts, &edge_array, max_edges, NULL);

	_gst_sort_edge_list (&edge_array [0], nedges);

	/* We know that build-edges gives us edges having vertices	*/
	/* in the range of 0 through nterms - 1, so we don't have to	*/
	/* do the sparse makeset's.  We use a union-find big enough to	*/
	/* handle LOTS of added Steiner points!				*/

	_gst_dsuf_create (&sets, max_points);
	for (i = 0; i < nterms; i++) {
		_gst_dsuf_makeset (&sets, i);
	}

	components = nterms;
	ep = &edge_array [0];
	endp = ep + nedges;
	len = 0;

	while (components > 1) {
		if (ep >= endp) {
			/* Ran out of edges before MST built! */
			FATAL_ERROR;
		}
		root1 = _gst_dsuf_find (&sets, ep -> p1);
		root2 = _gst_dsuf_find (&sets, ep -> p2);
		if (root1 NE root2) {
			_gst_dsuf_unite (&sets, root1, root2);
			len += ep -> len;
			--components;
		}
		++ep;
	}
	best = len;

	memset ((char *) mark, 0, nterms * kmasks * sizeof (bitmap_t));

	/* Mark the diagonal to dis-allow terminals as steiners. */
	mark_rowp = &mark [0];
	for (i = 0; i < nterms; i++) {
		SETBIT (mark_rowp, i);
		mark_rowp += kmasks;
	}

	n = nterms;
	newpt = &(pts -> a [n]);
	for (i = 0; (i < nterms) AND (n < max_points); i++) {
		last = best;
		memset (newpt, 0, sizeof (*newpt));
		memset (&bestpt, 0, sizeof (bestpt));
		best_j = -1;
		best_k = -1;
		mark_rowp = &mark [0];
		p1 = &(pts -> a [0]);
		for (j = 0; j < nterms; mark_rowp += kmasks, p1++, j++) {
			newpt -> x = p1 -> x;
			p2 = &(pts -> a [0]);
			for (k = 0; k < nterms; p2++, k++) {
				if (BITON (mark_rowp, k)) continue;
				newpt -> y = p2 -> y;
				ep2 = &new_edges [0];
				p3 = &(pts -> a [0]);
				new_nedges = 0;
				for (m = 0; m < n; p3++, m++) {
					ep2 -> p1	= n;
					ep2 -> p2	= m;
					ep2 -> len	= RDIST (newpt, p3);
					++new_nedges;
					++ep2;
				}
				_gst_sort_edge_list (&new_edges [0],
						     new_nedges);

				for (m = 0; m <= n; m++) {
					_gst_dsuf_makeset (&sets, m);
				}

				components = n + 1;
				ep1	= &edge_array [0];
				endp1	= ep1 + nedges;
				ep2	= &new_edges [0];
				endp2	= ep2 + new_nedges;
				len	= 0;

				while (components > 1) {
					if (ep1 < endp1) {
						if (ep2 < endp2) {
							if (ep1 -> len < ep2 -> len) {
								ep = ep1++;
							}
							else {
								ep = ep2++;
							}
						}
						else {
							ep = ep1++;
						}
					}
					else if (ep2 < endp2) {
						ep = ep2++;
					}
					else {
						FATAL_ERROR;
					}
					root1 = _gst_dsuf_find (&sets, ep -> p1);
					root2 = _gst_dsuf_find (&sets, ep -> p2);
					if (root1 NE root2) {
						_gst_dsuf_unite (&sets, root1, root2);
						len += ep -> len;
						--components;
					}
				}
				if (len < best) {
					bestpt = *newpt;
					best = len;
					best_j = j;
					best_k = k;
					if (best < limit) break;
				}
			}
			if (best < limit) break;
		}
		if (best < limit) {
			/* Include this point in the output... */
			++n;
			break;
		}
		if (best >= last) break;

		/* Avoid trying this steiner point again... */
		SETBIT (&mark [best_j * kmasks], best_k);

		/* Re-create edges for best point, and merge into */
		/* list of all edges, maintaining sorted order... */
		*newpt = bestpt;
		p3 = &(pts -> a [0]);
		ep = &new_edges [0];
		new_nedges = 0;
		for (m = 0; m < n; p3++, m++) {
			ep -> p1	= n;
			ep -> p2	= m;
			ep -> len	= RDIST (newpt, p3);
			++ep;
			++new_nedges;
		}
		_gst_sort_edge_list (&new_edges [0], new_nedges);
		ep1	= &edge_array [0];
		endp1	= ep1 + nedges;
		ep2	= &new_edges [0];
		endp2	= ep2 + new_nedges;
		ep	= endp1 + new_nedges;
		nedges += new_nedges;

		while (ep2 < endp2) {
			if (ep1 < endp1) {
				if (endp1 [-1].len >= endp2 [-1].len) {
					*--ep = *--endp1;
				}
				else {
					*--ep = *--endp2;
				}
			}
			else {
				*--ep = *--endp2;
			}
		}

		/* Edge list updated.  Now add best point to	*/
		/* the set and try for another.			*/
		++n;
		++newpt;
	}

	pts -> n = n;

	_gst_dsuf_destroy (&sets);

	free ((char *) edge_array);
	free ((char *) mark);
	free ((char *) new_edges);

	return (best);
}
