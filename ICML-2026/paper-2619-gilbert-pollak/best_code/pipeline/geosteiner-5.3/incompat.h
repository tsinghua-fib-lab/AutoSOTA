/***********************************************************************

	$Id: incompat.h,v 1.8 2022/11/19 13:45:52 warme Exp $

	File:	incompat.h
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
		: Reorganized include files, apply prefixes.

************************************************************************/

#ifndef INCOMPAT_H
#define	INCOMPAT_H

#include "bitmaskmacros.h"

struct gst_hypergraph;

/*
 * The following structure is used to contain temporary buffers
 * used to retrieve incompatibility information.  This encapsulates
 * the details of allocating and initializing all the buffers needed
 * the the "get_incompat_edges" routine.
 */

struct inc_info {
	bitmap_t *	incmask;	/* Mask of incompatible edges found */
	bitmap_t *	edges_seen;	/* Mask of edges seen and checked */
	bitmap_t *	vmask;		/* Mask of vertices in queried edge */
	int *		seenp;		/* Scratch buffer for edges seen */
	struct gst_hypergraph *
			cip;		/* Problem instance */
};


/*
 * Function Prototypes
 */

extern int	_gst_get_incompat_edges (int *			incompat,
					 int			e,
					 struct inc_info *	tmp);
extern void	_gst_shutdown_incompat_edges (struct inc_info *	tmp);
extern void	_gst_startup_incompat_edges (
					struct inc_info *	tmp,
					struct gst_hypergraph *	cip);

#endif
