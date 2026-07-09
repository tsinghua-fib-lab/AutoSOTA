/***********************************************************************

	$Id: steiner.h,v 1.43 2022/11/19 13:45:55 warme Exp $

	File:	steiner.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	General declarations for the Steiner Tree program.

************************************************************************

	Modification Log:

	a-1:	02/20/93	warme
		: Created.
	a-2:	08/31/98	warme
		: Split off metric dependent stuff.
	b-1:	02/28/2001	warme
		: Changes for 3.1 release.
		: Modified "struct numlist" to be partially scaled
		:  numeric representation instead of textual.
		: Added struct scale_info and UNSCALE macro.
		: Added tracef routine and struct tracef_control.
		: Pass new scale_info to all routines that need it.
		: Added several new routines.
	c-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Split off some stuff to general.h and logic.h.
		: Introduced metric structure.
		: Introduced PRELUDE/POSTLUDE macros.
		: Removed tracef control structure.
		: Changed some hypergraph values to properties.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
		: Added cost_extension.
		: Removed unused declaration.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Split most previous contents off into other files.

************************************************************************/

#ifndef	STEINER_H
#define	STEINER_H

#define _GNU_SOURCE

#include <stddef.h>

#include "bitmaskmacros.h"
#include "geomtypes.h"
#include "gsttypes.h"

struct environment;
struct gst_channel;
struct gst_param;
struct gst_parmdefs;
struct gst_solver;
struct point;
struct pset;


/*
 * A constant to represent the "infinite" distance.
 */

#define INF_DISTANCE	((dist_t) DBL_MAX)	/* "infinite" distance. */


/*
 * The folling represents a single edge in a weighted tree or graph.
 */

struct edge {
	dist_t		len;		/* Length of edge. */
	int		p1;		/* First endpoint of edge. */
	int		p2;		/* Second endpoint of edge. */
};


/*
 * The structure used to represent a single FST.
 */

struct full_set {
	struct full_set *	next;
	int			tree_num;
	dist_t			tree_len;
	int *			tlist;
	struct pset *		terminals;
	struct pset *		steiners;
	int			nedges;
	struct edge *		edges;
};

/*
 * The gst_hypergraph describes the basic problem instance, and also contains
 * some additional information (i.e., compatibility/incompatibility info).
 */

struct gst_hypergraph {
	/* The basic problem description. */
	int			num_verts;
	int			num_edges;
	int			num_vert_masks;
	int			num_edge_masks;
	int **			edge;		/* Terminals in each edge */
	int *			edge_size;	/* Cardinality of each edge */
	dist_t *		cost;		/* Length of each edge */
	bool *			tflag;		/* TRUE: vertex is terminal, */
						/* FALSE: vertex is Steiner. */

	struct gst_metric *	metric;		/* Metric information */
	struct gst_scale_info *	scale;		/* Problem scaling info */

	/* Initial problem bit-masks. */
	bitmap_t *		initial_vert_mask;
	bitmap_t *		initial_edge_mask;
	bitmap_t *		required_edges;
	/* Pre-initialized tables. */
	int **			term_trees;
	int **			inc_edges;

	/* Optional geometric information. */
	struct pset *		pts;
	struct full_set **	full_trees;

	/* Optional extension for specialized cost information. */
	struct cost_extension *	cost_extension;

	/* Additional information about the hypergraph can be obtained	*/
	/* through the property system -- see gst_get_hg_properties()	*/
	/* and GST_PROP_HG_... in geosteiner.h				*/
	struct gst_proplist *	proplist;

	/* Version numbers */
	int version;		/* The current version of the hypergraph */
	int requested_version;	/* Latest version requested/used by a solver */
};

/*
 * Function Prototypes.
 */

extern void	_gst_free_full_set (struct full_set * fsp);
extern struct full_set *
		_gst_remove_degree_two_steiner_points (struct full_set * fsp);

#endif
