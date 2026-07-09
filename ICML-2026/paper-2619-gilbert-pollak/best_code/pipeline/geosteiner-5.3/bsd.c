/***********************************************************************

	$Id: bsd.c,v 1.17 2022/11/19 13:45:49 warme Exp $

	File:	bsd.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1997, 2022 by David M. Warme and Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Bottleneck Steiner Distance stuff.

************************************************************************

	Modification Log:

	a-1:	09/04/97	warme
		: Split off from coarse.c.
	a-2:	10/04/98	warme
		: Completely re-coded to conserve memory and provide
		:  multiple implementations that provide different
		:  space-time tradeoffs.
	a-3:	02/28/2001	warme
		: Changes for 3.1 release.  Migrate data from bsd
		:  structure to local vars.
	a-4:	08/01/2002	martinz
		: Implemented new linear space and logarithmic time
		: lookup data structure by Mandoiu et al.
		: Old data structure is method 0 and new is method 1.
	a-5:	11/22/2002	benny
		: Added dynamic choice between the two methods.
		: New method used when more than 100 terminals are given.
		: Method values are now symbolic (geosteiner.h).
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.

************************************************************************/

#include "bsd.h"

#include "fatal.h"
#include "geosteiner.h"
#include <limits.h>
#include "logic.h"
#include "memory.h"
#include <stdlib.h>
#include "steiner.h"
#include <string.h>


/*
 * Global Routines
 */

dist_t			_gst_bsd (struct bsd * bsdp, int i, int j);
struct bsd *		_gst_compute_bsd (int		nedges,
					  struct edge *	mst_edges,
					  int		method);
bool			_gst_is_mst_edge (struct bsd * bsdp, int i, int j);
void			_gst_shutdown_bsd (struct bsd * bsdp);


/*
 * Local Constants
 */

#define LOG_BSD_THRESHOLD	100


/*
 * Local Types
 */

	/* A structure for representing an MST in adjacency list format. */

struct mstadj {
	int		edge;		/* Edge number */
	int		vnum;		/* Index of other vertex */
};


/*
 * Local Routines
 */

static struct mstadj **	build_adjacency_list (int, int, int, struct edge *);
static void		walk (int,
			      int,
			      struct mstadj **,
			      bool *,
			      int *);

/*
 * This routine returns the Bottleneck Steiner Distance between
 * two terminals, specified by index.
 */

	dist_t
_gst_bsd (

struct bsd *	bsdp,		/* IN - BSD data structure */
int		i,		/* IN - first terminal */
int		j		/* IN - second terminal */
)
{
int		n;
int		index;
int		ei;
int		ej;

	n = bsdp -> n;
	FATAL_ERROR_IF ((i < 0) OR (i >= n) OR (j < 0) OR (j >= n));

	if (i EQ j) return (0.0);

	switch (bsdp -> method) {
	case GST_PVAL_BSD_METHOD_CONSTANT:
		/* Complete lower-triangular matrix. */
		if (i > j) {
			i = ((i * (i - 1)) >> 1) + j;
		}
		else {
			i = ((j * (j - 1)) >> 1) + i;
		}
		j = bsdp -> ematrix [i];
		return (bsdp -> mst_edges [j].len);

	case GST_PVAL_BSD_METHOD_LOGARITHMIC:
		/* Linear space, logarithmic time lookup. */
		index = -1;
		while (i NE j) {
			ei = bsdp -> edge[i];
			ej = bsdp -> edge[j];
			if (ei > index) index = ei;
			if (ej > index) index = ej;
			i = bsdp -> parent[i];
			j = bsdp -> parent[j];
		}
		return (bsdp -> mst_edges [index].len);

	default:
		FATAL_ERROR;
		break;
	}

	return (0.0);
}

/*
 * This routine initializes the BSD data structures according to the
 * given point set and implementation method.  The first thing is to
 * compute the actual minimum spanning tree.  The rest of the initialization
 * is method-specific.
 * Method 0: Dynamic choice between 1 and 2 based on the number of terminals.
 * Method 1: Quadratic space, constant time lookup.
 * Method 2: Linear space, logarithmic time loopkup.
 *
 * NOTE: This code (specifically Method 2) assumes that the given MST edges
 *	 have been sorted in non-decreasing order by length!!!
 */

	struct bsd *
_gst_compute_bsd (

int			nedges,		/* IN - number of MST edges */
struct edge *		mst_edges,	/* IN - edges of the MST */
int			method		/* IN - implementation method */
)
{
int			i;
int			j;
int			u;
int			v;
int			pu;
int			pv;
int			next;
int			nterms;
struct bsd *		bsdp;
int16u *		rowp;
int *			tmprow;
struct mstadj **	adj_list;
struct edge *		edges;
bool *			mark;

	nterms = nedges + 1;

	/* Find out which BSD method to use. */
	switch (method) {
	case GST_PVAL_BSD_METHOD_DYNAMIC:
		if (nterms < LOG_BSD_THRESHOLD) {
			method = GST_PVAL_BSD_METHOD_CONSTANT;
		}
		else {
			method = GST_PVAL_BSD_METHOD_LOGARITHMIC;
		}
		break;

	case GST_PVAL_BSD_METHOD_CONSTANT:
		if (nterms > USHRT_MAX) {
			/* We cannot use the constant time / quadratic	*/
			/* space method (unless we use "int" instead of	*/
			/* "unsigned short", but this would double the	*/
			/* already costly space complexity).		*/
			/* Gracefully force use of log time / linear	*/
			/* space method.				*/
			method = GST_PVAL_BSD_METHOD_LOGARITHMIC;
		}
		break;

	case GST_PVAL_BSD_METHOD_LOGARITHMIC:
		break;

	default:
		FATAL_ERROR;
		break;
	}

	/* Allocate and zero the BSD structure. */
	bsdp = NEW (struct bsd);
	memset (bsdp, 0, sizeof (*bsdp));

	bsdp -> method	= method;
	bsdp -> n	= nterms;
	edges		= NEWA (nedges + 1, struct edge); /* 1 extra! */

	/* Add initial zero-length edge, so that we have an edge number	*/
	/* that represents a BSD of zero.				*/
	edges [0].len	= 0.0;
	edges [0].p1	= 0;
	edges [0].p2	= 0;

	memcpy (&edges [1], mst_edges, nedges * sizeof (mst_edges [0]));

	bsdp -> mst_edges = edges;

	adj_list = build_adjacency_list (nterms, 1, nedges + 1, edges);

	switch (method) {
	case GST_PVAL_BSD_METHOD_CONSTANT:
		/* Complete lower triangular matrix. */

		mark = NEWA (nterms, bool);
		for (i = 0; i < nterms; i++) {
			mark [i] = FALSE;
		}

		bsdp -> ematrix = NEWA (nterms * (nterms - 1) >> 1, int16u);
		tmprow = NEWA (nterms, int);

		/* Fill in the matrix, one row at a time... */
		rowp = bsdp -> ematrix;
		for (i = 0; i < nterms; i++) {
			walk (i, 0, adj_list, mark, tmprow);
			for (j = 0; j < i; j++) {
				*rowp++ = tmprow [j];
			}
		}
		free ((char *) tmprow);
		free ((char *) mark);
		break;

	case GST_PVAL_BSD_METHOD_LOGARITHMIC:
		/* Linear space, logarithmic time lookup */
		bsdp -> parent = NEWA(2 * nterms - 1, int);
		bsdp -> edge = NEWA(2 * nterms - 1, int);

		/* Initialize */
		for (i = 0; i < 2*nterms - 1; i++) {
			bsdp -> parent[i] = -1;
			bsdp -> edge[i]	  = -1;
		}

		/* Create data structure */
		/* NOTE: This code assumes the MST edges are	*/
		/* sorted in non-decreasing order by length!!!	*/
		next = nterms;
		for (i = 1; i <= nterms - 1; i++) {
			u  = bsdp -> mst_edges[i].p1;
			v  = bsdp -> mst_edges[i].p2;
			pu = bsdp -> parent[u];
			pv = bsdp -> parent[v];

			while ((u NE v) AND (pu >= 0) AND (pv >= 0)) {
				u = pu;
				v = pv;
				pu = bsdp -> parent[u];
				pv = bsdp -> parent[v];
			}

			if ((pu < 0) AND (pv < 0)) {
				next++;
				bsdp -> parent[u] = next;
				bsdp -> parent[v] = next;
				bsdp -> edge[u]	  = i;
				bsdp -> edge[v]	  = i;
				continue;
			}

			if ((pu < 0) AND (pv >= 0)) {
				bsdp -> parent[u] = pv;
				bsdp -> edge[u]	  = i;
				continue;
			}

			if ((pu >= 0) AND (pv < 0)) {
				bsdp -> parent[v] = pu;
				bsdp -> edge[v]	  = i;
			}
		}
		break;

	default:
		FATAL_ERROR;
		break;
	}

	bsdp -> adj_list = adj_list;
	return (bsdp);
}

/*
 * This routine converts a list of edges into a full graph structure
 * represented in adjacency list form.  The adjacency list is in two parts:
 * an array of pointers indexed by node number, and an array of "adj"
 * structures indexed by these pointers.  To find all neighbors of node K,
 * look at every "mstadj" structure between adj_list [K] and adj_list [K+1].
 */

	static
	struct mstadj **
build_adjacency_list (

int			n,		/* IN - number of nodes */
int			first_edge,	/* IN - first edge number */
int			nedges,		/* IN - number of edges */
struct edge *		edges		/* IN - array of edges */
)
{
int			i;
struct edge *		ep;
struct mstadj *		ap;
int *			count;
struct mstadj **	tmp_ptr;
struct mstadj **	adj_list;

	adj_list	= NEWA (n + 1, struct mstadj *);
	ap		= NEWA (2 * nedges, struct mstadj);
	tmp_ptr		= NEWA (n, struct mstadj *);
	count		= NEWA (n, int);

	for (i = 0; i < n; i++) {
		count [i] = 0;
	}

	/* Count neighbors for each node. */
	ep = &edges [first_edge];
	for (i = first_edge; i < nedges; ep++, i++) {
		++(count [ep -> p1]);
		++(count [ep -> p2]);
	}

	/* Set up the pointers for each node. */
	for (i = 0; i < n; i++) {
		adj_list [i]	= ap;
		tmp_ptr [i]	= ap;
		ap += count [i];
	}
	adj_list [i] = ap;		/* set overall end of list. */

	/* Deposit neighbors into properly allocated lists. */
	ep = &edges [first_edge];
	for (i = first_edge; i < nedges; ep++, i++) {
		ap = tmp_ptr [ep -> p1]++;
		ap -> edge	= i;
		ap -> vnum	= ep -> p2;

		ap = tmp_ptr [ep -> p2]++;
		ap -> edge	= i;
		ap -> vnum	= ep -> p1;
	}

	free ((char *) count);
	free ((char *) tmp_ptr);

	return (adj_list);
}

/*
 * This routine recursively walks through the given Minimum Spanning Tree
 * starting at the given node, and determines the longest edge along the
 * path from the original given root node, to every other node in the tree.
 *
 * Note: this routine assumes that edges are listed in increasing order,
 * just as Kruskal's MST algorithm emits them.
 */

	static
	void
walk (

int			node,		/* IN - current node in tree */
int			longest,	/* IN - longest edge # in cur path */
struct mstadj **	adj_list,	/* IN - MST adjacency list */
bool *			mark,		/* IN - vertices visited */
int *			rowp		/* OUT - array of longest edge #'s */
)
{
struct mstadj *		ap;
struct mstadj *		endp;

	/* We now know the longest edge along the path from the root */
	/* to this current node!  Remember it in the matrix! */
	rowp [node] = longest;

	mark [node] = TRUE;
	ap	= adj_list [node];
	endp	= adj_list [node + 1];
	while (ap < endp) {
		if (NOT mark [ap -> vnum]) {
			walk (ap -> vnum,
			     (longest >= ap -> edge) ? longest : ap -> edge,
			      adj_list,
			      mark,
			      rowp);
		}
		++ap;
	}
	mark [node] = FALSE;
}

/*
 * This routine simply tells whether a given edge is part of the MST.
 */
	bool
_gst_is_mst_edge (

struct bsd *	bsdp,
int		i,
int		j
)
{
struct mstadj *		ap;
struct mstadj *		endp;

	ap	= bsdp -> adj_list [i];
	endp	= bsdp -> adj_list [i + 1];
	while (ap < endp) {
		if (ap -> vnum EQ j) {
			return TRUE;
		}
		++ap;
	}

	return FALSE;
};

/*
 * This routine destroys and deallocates the BSD information.
 */

	void
_gst_shutdown_bsd (

struct bsd *	bsdp		/* IN - BSD info to destroy */
)
{
	if (bsdp EQ NULL) {
		return;
	}

	if (bsdp -> ematrix NE NULL) {
		free ((char *) (bsdp -> ematrix));
		bsdp -> ematrix = NULL;
	}

	if (bsdp -> mst_edges NE NULL) {
		free ((char *) (bsdp -> mst_edges));
		bsdp -> mst_edges = NULL;
	}

	if (bsdp -> adj_list NE NULL) {
		if (bsdp -> adj_list [0] NE NULL) {
			free ((char *) bsdp -> adj_list [0]);
		}
		free ((char *) bsdp -> adj_list);
		bsdp -> adj_list = NULL;
	}

	if (bsdp -> edge NE NULL) {
		free ((char *) (bsdp -> edge));
		bsdp -> edge = NULL;
	}

	if (bsdp -> parent NE NULL) {
		free ((char *) (bsdp -> parent));
		bsdp -> parent = NULL;
	}

	free ((char *) bsdp);
}
