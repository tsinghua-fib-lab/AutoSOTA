/***********************************************************************

	$Id: sec_comp.h,v 1.11 2022/11/19 13:45:54 warme Exp $

	File:	sec_comp.h
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Data structures for decomposition to make separation of
	SEC's easier.

************************************************************************

	Modification Log:

	a-1:	10/05/96	warme
		: Created.
	b-1:	02/28/2001	warme
		: Changes for 3.1 release.
		: Renamed many members of "struct comp" -- we now
		:  prefer "vertex" over "terminal" and "edge" over
		:  "full set" or FST.
		: Replace "tmasks" and "kmasks" with new "rverts".
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
	e-4:	11/18/2022	warme
		: Simplify calling convention.
		: Added _gst_create_comp_masks() and
		:  _gst_free_congested_component_list().

************************************************************************/

#ifndef SEC_COMP_H
#define SEC_COMP_H

#include "bitmaskmacros.h"
#include "gsttypes.h"

struct bbinfo;
struct comp;
struct constraint;


/*
 * The following structure contains all of the information that we
 * need to represent a single component of an SEC separation problem.
 * We use a variety of techniques to split one separation problem into
 * several components, and to simplify those components.
 */

struct comp {
	struct comp *	next;		/* Linked list of components */
	int		num_verts;	/* Number of vertices in component */
	int		num_edges;	/* Number of edges in component */
	int8u		flags;		/* See below */
	/* Info about edges in the component... */
	double *	x;		/* LP assigned weight */
	int **		everts;		/* Vertices in each edge */
	/* Info about vertices in the component... */
	int **		vedges;		/* Edges incident to each vertex */
	double *	tviol;		/* Extra violation for each vertex */
	int **		rverts;		/* List of original vertices that */
					/* are represented by each vertex */
					/* (we simplify by merging multiple */
					/* vertices into one...) */
	bitmap_t *	vert_mask;	/* Mask of valid vertices */
	bitmap_t *	edge_mask;	/* Mask of valid edges */
	struct constraint * cp;		/* List of constraints for this */
					/* component */
};

/*
 * Various flags.  These are used to keep track of which hurdles a given
 * component has cleared.  By keeping track of this info we apply only
 * those transformations that have not previously been done to the component.
 * When a component is simplified or further divided, however, we can reset
 * these flags on the sub-component(s) -- permitting further decomposing or
 * simplification without fear of infinite looping or recursion.
 */

#define CFLG_CONG	0x01	/* Known to contain only congested vertices */
#define CFLG_CC		0x02	/* Known to be a connected component */
#define CFLG_BCC	0x04	/* Known to be a bi-connected component */
#define CFLG_CHAIN	0x08	/* Long integral chains merged */

#define CFLG_ALL	(CFLG_CONG | CFLG_CC | CFLG_BCC | CFLG_CHAIN)


extern struct constraint * _gst_check_component_subtour (
					bitmap_t *		S,
					struct comp *		comp,
					struct constraint *	cp,
					double *		x,
					struct bbinfo *		bbip);
extern void		_gst_create_comp_masks (struct comp *);
extern struct comp *	_gst_delete_vertex_from_component (
					int			t,
					struct comp *		comp,
					struct bbinfo *		bbip);
extern struct comp *	_gst_find_congested_components (
					double *	x,
					bitmap_t *	vert_mask,
					bitmap_t *	edge_mask,
					bool		print_flag,
					struct bbinfo * bbip);

extern int		_gst_find_least_congested_vertex (
					bitmap_t *		S,
					struct comp *		comp);
extern void		_gst_free_congested_component (struct comp * p);
extern void		_gst_free_congested_component_list (struct comp * p);

#endif
