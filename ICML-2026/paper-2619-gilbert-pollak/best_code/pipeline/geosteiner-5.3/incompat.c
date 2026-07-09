/***********************************************************************

	$Id: incompat.c,v 1.9 2022/11/19 13:45:52 warme Exp $

	File:	incompat.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2001, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routine for accessing set of edges incompatible to
	a given edge.

************************************************************************

	Modification Log:

	a-1:	08/07/2001	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.

************************************************************************/

#include "incompat.h"

#include "fatal.h"
#include "logic.h"
#include "memory.h"
#include <stdlib.h>
#include "steiner.h"


/*
 * Global Routines
 */

int	_gst_get_incompat_edges (int *			incompat,
				 int			e,
				 struct inc_info *	tmp);
void	_gst_shutdown_incompat_edges (struct inc_info *	tmp);
void	_gst_startup_incompat_edges (struct inc_info *		tmp,
				     struct gst_hypergraph *	cip);

/*
 * Start up this package.  We simply allocate and initialize
 * a few scratch buffers.
 */

	void
_gst_startup_incompat_edges (

struct inc_info *	tmp,	/* OUT - various scratch buffers */
struct gst_hypergraph *	cip	/* IN - problem instance */
)
{
int		i;
int		n;
int		nedges;
int		kmasks;
int		nmasks;
bitmap_t *	bp1;

	nedges = cip -> num_edges;
	kmasks = cip -> num_vert_masks;
	nmasks = cip -> num_edge_masks;

	/* Allocate and zero all three buffers in one fell swoop. */

	n = 2 * nmasks + kmasks;
	bp1 = NEWA (n, bitmap_t);

	for (i = 0; i < n; i++) {
		bp1 [i] = 0;
	}

	/* Initialize collection of scratch buffers. */

	tmp -> incmask		= bp1;		bp1 += nmasks;
	tmp -> edges_seen	= bp1;		bp1 += nmasks;
	tmp -> vmask		= bp1;
	tmp -> seenp		= NEWA (nedges, int);
	tmp -> cip		= cip;
}


/*
 * Shut down this package.
 */

	void
_gst_shutdown_incompat_edges (

struct inc_info *	tmp	/* IN - scratch buffers */
)
{
	free ((char *) (tmp -> incmask));
	free ((char *) (tmp -> seenp));

	tmp -> incmask		= NULL;
	tmp -> edges_seen	= NULL;
	tmp -> vmask		= NULL;
	tmp -> seenp		= NULL;
}

/*
 * This routine retrieves a list of all edges that are incompatible
 * with the given edge.  This includes all incompatibilities (if any)
 * in the "inc_edges" list, as well as all "basic" incompatibilities.
 * In order to conserve memory, we do not store any of the "basic"
 * incompatibilities in the "inc_edges" list.  Instead we compute
 * these as needed right here.
 *
 * The number of edges produced is returned.
 * The "incompat" argument should be a buffer big enough to contain the
 * set of incompatible edges returned.
 *
 * Note: the edges are returned in an unspecified order!  (i.e., NOT sorted!)
 */

	int
_gst_get_incompat_edges (

int *			incompat,	/* OUT - list of incompatible edges */
int			e,		/* IN - edge to get inc. edges for */
struct inc_info *	tmp		/* IN - scratch buffers */
)
{
int			j;
int			k;
int			v;
int			t;
struct gst_hypergraph *	cip;
int *			outp;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			vp3;
int *			vp4;
int *			seenp1;
int *			seenp2;
bitmap_t *		edge_mask;
bitmap_t *		incmask;
bitmap_t *		edges_seen;
bitmap_t *		vmask;

	outp = incompat;

	incmask	   = tmp -> incmask;
	edges_seen = tmp -> edges_seen;
	vmask	   = tmp -> vmask;
	seenp1	   = tmp -> seenp;
	cip	   = tmp -> cip;

	seenp2	   = seenp1;

	edge_mask = cip -> initial_edge_mask;

	if (cip -> inc_edges NE NULL) {
		/* Copy all "non-basic" incompatibilies... */
		ep1 = cip -> inc_edges [e];
		ep2 = cip -> inc_edges [e + 1];
		while (ep1 < ep2) {
			j = *ep1++;
			if (NOT BITON (edge_mask, j)) continue;
			if (BITON (incmask, j)) continue;
			SETBIT (incmask, j);
			SETBIT (edges_seen, j);
			*outp++ = j;
			*seenp2++ = j;
		}
	}

	/* Set up mask of vertices in the edge being queried. */
	vp1 = cip -> edge [e];
	vp2 = cip -> edge [e + 1];
	while (vp1 < vp2) {
		v = *vp1++;
		SETBIT (vmask, v);
	}

	vp1 = cip -> edge [e];
	vp2 = cip -> edge [e + 1];
	while (vp1 < vp2) {
		v = *vp1++;
		ep1 = cip -> term_trees [v];
		ep2 = cip -> term_trees [v + 1];
		while (ep1 < ep2) {
			j = *ep1++;
			if (NOT BITON (edge_mask, j)) continue;
			if (BITON (edges_seen, j)) continue;
			if (j EQ e) continue;
			SETBIT (edges_seen, j);
			*seenp2++ = j;

#if 0
			if (BITON (incmask, j)) {
				/* Edge we haven't seen yet is incompat? */
				FATAL_ERROR;
			}
#endif

			/* Count vertices in common... */
			k = 0;
			vp3 = cip -> edge [j];
			vp4 = cip -> edge [j + 1];
			while (vp3 < vp4) {
				t = *vp3++;
				if (BITON (vmask, t)) {
					++k;
					if (k > 1) {
						/* "Basic" incompatibility! */
						SETBIT (incmask, j);
						*outp++ = j;
						break;
					}
				}
			}
		}
	}

	/* Tidy up: clear the masks back to all zeros. */

	k = outp - incompat;

	while (outp > incompat) {
		j = *--outp;
		CLRBIT (incmask, j);
	}

	while (seenp1 < seenp2) {
		j = *--seenp2;
		CLRBIT (edges_seen, j);
	}

	vp1 = cip -> edge [e];
	vp2 = cip -> edge [e + 1];
	while (vp1 < vp2) {
		v = *vp1++;
		CLRBIT (vmask, v);
	}

	return (k);
}
