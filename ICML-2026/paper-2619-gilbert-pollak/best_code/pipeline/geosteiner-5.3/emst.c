/***********************************************************************

	$Id: emst.c,v 1.13 2022/11/19 13:45:51 warme Exp $

	File:	emst.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by Martin Zachariasen & David M. Warme.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Routines to compute Euclidean Minimum Spanning Trees.

************************************************************************

	Modification Log:

	a-1:	11/10/98	martinz
		: Created.
	b-1:	02/28/2001	warme
		: Renamed build_edges, and eliminated its 3rd "at_least"
		:  parameter.  Build complete graph when N < 10.
		: Use common routines now in mst.c.
		: Fix duplicate points problem.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.

************************************************************************/

#include "emst.h"

#include "dsuf.h"
#include "dt.h"
#include "fatal.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "mst.h"
#include "point.h"
#include "sortfuncs.h"
#include <stdlib.h>
#include "steiner.h"
#include <string.h>


/*
 * Global Routines
 */

int		_gst_euclidean_mst (struct pset * pts, struct edge * edges);
dist_t		_get_euclidean_mst_length (struct pset * pts);


/*
 * External References
 */

	/* none */


/*
 * Local Routines
 */

static int		build_euclidean_edges (struct pset *,
					       struct edge **);

/*
 * This routine computes the total length of the Euclidean Minimum
 * Spanning Tree of the given set of points.
 */

	dist_t
_gst_euclidean_mst_length (

struct pset *		pts	/* IN - point set to find MST length of. */
)
{
int		i;
int		nedges;
dist_t		total;
struct edge *	ep;
struct edge *	edges;

	edges = NEWA (pts -> n - 1, struct edge);

	nedges = _gst_euclidean_mst (pts, &edges [0]);

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
 * This routine computes an Euclidean Minimum Spanning Tree for the
 * given point set.  The output is a list of edges.
 *
 * The "Triangle" package by Jonathan Richard Shewchuk is used
 * to provide the Delaunay triangulation for the set of points.
 */

	int
_gst_euclidean_mst (

struct pset *		pts,		/* IN - point set. */
struct edge *		edges		/* OUT - edge list. */
)
{
int		nedges;
int		mst_edge_count;
struct edge *	edge_array;

	nedges = build_euclidean_edges (pts, &edge_array);

	mst_edge_count = _gst_mst_edge_list (pts -> n,
					     nedges,
					     &edge_array [0],
					     edges);

	free ((char *) edge_array);

	return (mst_edge_count);
}

/*
 * This routine builds an edge-list containing all edges in
 * the Delaunay triangulation of the set of points.
 */

	static
	int
build_euclidean_edges (

struct pset *		pts,		/* IN - set of points */
struct edge **		edges_out	/* OUT - edge list */
)
{
int			i, i1;
int			j, j1;
int			k;
int			n;
int			nedges;
int			ndup;
int			numberofedges;
int *			edgelist;
struct edge *		edges;
struct point *		p1;
struct point *		p2;
struct pset *		newpts;
int *			order;
int *			orig_vnum;
bool *			dflags;
struct edge *		zedges;

	n = pts -> n;

	if (n < 10) {
		/* Build the complete graph... */
		nedges = n * (n - 1) / 2;
		edges = NEWA (nedges, struct edge);
		*edges_out = edges;
		p1 = &(pts -> a [0]);
		for (i = 0; i < n; i++, p1++) {
			p2 = p1 + 1;
			for (j = i + 1; j < n; j++, p2++) {
				edges -> len	= EDIST (p1, p2);
				edges -> p1	= i;
				edges -> p2	= j;
				++edges;
			}
		}
		return (nedges);
	}

	/* Triangle does a "good" job when given duplicate points -- it	*/
	/* emits edges to only one of the given coincident points.	*/
	/* This is bad for us, however, since we want a fully connected	*/
	/* MST, and cannot get one if there are vertices for which we	*/
	/* have no incident edges.  Therefore we must detect all	*/
	/* duplicate points and manually generate one zero-length edge	*/
	/* for each (copies 2 through K of a point each have an edge to	*/
	/* the first copy of the point).				*/

	order = _gst_heapsort_x (pts);

	dflags = NEWA (n, bool);
	memset (dflags, FALSE, n * sizeof (dflags [0]));

	zedges = NEWA (n, struct edge);
	memset (zedges, 0, n * sizeof (zedges));

	ndup = 0;
	for (i = 0; i < n - 1; ) {
		i1 = order [i];
		p1 = &(pts -> a [i1]);
		for (j = i; ; ) {
			++j;
			if (j >= n) break;

			j1 = order [j];
			p2 = &(pts -> a [j1]);
			if (p1 -> x NE p2 -> x) break;
			if (p1 -> y NE p2 -> y) break;

			/* Point j1 is a duplicate of point i1.  The	*/
			/* sort also guarantees that i1 < j1.  Omit	*/
			/* point j1.					*/

			dflags [j1] = TRUE;

			/* Generate zero-length edge (i1,j1). */

			zedges [ndup].len	= 0.0;
			zedges [ndup].p1	= i1;
			zedges [ndup].p2	= j1;
			++ndup;
		}
		i = j;
	}

	free (order);

	/* Get array to map duplicate-free points back to originals. */
	orig_vnum = NEWA (n, int);

	/* Set up data structure to call triangle. */
	newpts = NEW_PSET (n);
	j = 0;
	for (i = 0; i < n; i++) {
		if (dflags [i]) continue;
		newpts -> a [j].x	= pts -> a [i].x;
		newpts -> a [j].y	= pts -> a [i].y;
		orig_vnum [j] = i;
		++j;
	}
	newpts -> n	= j;

	free (dflags);

	_gst_delaunay_triangulation (
				newpts,
				&numberofedges,
				&edgelist,
				NULL, 
				NULL,
				NULL);
	free (newpts);

	nedges = numberofedges;

	edges = NEWA (nedges + ndup, struct edge);
	*edges_out = edges;

	for (k = 0; k < ndup; k++) {
		*edges++ = zedges [k];
	}

	free (zedges);

	for (k = 0; k < nedges; k++) {
		i  = orig_vnum [edgelist [2*k    ]];
		j  = orig_vnum [edgelist [2*k + 1]];
		p1 = &(pts -> a [i]);
		p2 = &(pts -> a [j]);

		edges -> len	= EDIST (p1, p2);
		edges -> p1	= i;
		edges -> p2	= j;
		++edges;
	}

	free (edgelist);
	free (orig_vnum);

	return (nedges + ndup);
}
