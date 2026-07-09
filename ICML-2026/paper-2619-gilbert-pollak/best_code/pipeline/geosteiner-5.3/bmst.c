/***********************************************************************

	$Id: bmst.c,v 1.10 2022/11/19 13:45:49 warme Exp $

	File:	bmst.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1997, 2022 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial  4.0 International License.

************************************************************************

	Routines to compute Minimum Spanning Trees using the
	Bottleneck Steiner Distance.

************************************************************************

	Modification Log:

	a-1:	09/04/97	warme
		: Created.
	b-1:	02/28/2001	martinz
		: Added new procedures "bmst" and "bmst_terms" that
		: return the edges of the MST (instead of only the
		: length)
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.

************************************************************************/

#include "bmst.h"

#include "bsd.h"
#include "fatal.h"
#include "logic.h"
#include "memory.h"
#include "mst.h"
#include <stdlib.h>
#include "steiner.h"


/*
 * Global Routines
 */

dist_t	_gst_bmst_terms_length (int * terms, int n, struct bsd * bsdp);
int	_gst_bmst_terms (int *		terms,
			 int		n,
			 struct bsd *	bsdp,
			 struct edge *	edges);


/*
 * External References
 */

	/* none */

/*
 * This routine computes the length of a Minimum Spanning Tree for
 * the given list of terminals as measured using the
 * Bottleneck Steiner Distance.
 */

	dist_t
_gst_bmst_terms_length (

int *			terms,		/* IN - array of terminal numbers */
int			n,		/* IN - number of terminals */
struct bsd *		bsdp		/* IN - BSD data structure */
)
{
int		i;
int		nedges;
struct edge *	ep;
struct edge *	edges;
dist_t		total;

	edges = NEWA (n - 1, struct edge);

	nedges = _gst_bmst_terms (terms, n, bsdp, edges);

	FATAL_ERROR_IF (nedges NE n - 1);

	total = 0;
	ep = edges;
	for (i = 0; i < nedges; i++, ep++) {
		total += ep -> len;
	}

	free ((char *) edges);

	return (total);
}

/*
 * This routine computes a Minimum Spanning Tree for
 * the given list of terminals as measured using the
 * Bottleneck Steiner Distance.
 */

	int
_gst_bmst_terms (

int *			terms,		/* IN - array of terminal numbers */
int			n,		/* IN - number of terminals */
struct bsd *		bsdp,		/* IN - BSD data structure */
struct edge *		edges		/* OUT - edge list */
)
{
int		i;
int		j;
int		t;
int		nedges;
int		mst_edge_count;
int *		ip;
int *		jp;
struct edge *	ep;
struct edge *	edge_array;

	nedges	= (n * (n - 1)) >> 1;

	edge_array = NEWA (nedges, struct edge);

	ep = edge_array;
	ip = &terms [0];
	for (i = 0; i < n; i++, ip++) {
		t = *ip;
		jp = ip + 1;
		for (j = i + 1; j < n; j++, jp++) {
			ep -> len	= _gst_bsd (bsdp, t, *jp);
			ep -> p1	= i;
			ep -> p2	= j;
			++ep;
		}
	}

	mst_edge_count = _gst_mst_edge_list (n, nedges, &edge_array [0], edges);

	free ((char *) edge_array);

	return (mst_edge_count);
}
