/***********************************************************************

	$Id: ufst.h,v 1.27 2022/11/19 13:45:55 warme Exp $

	File:	ufst.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by PAwel Winter & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Uniform FST generator.

************************************************************************

	Modification Log:

	a-1:	05/03/2002	benny
		: Created.  Derived from efst.c and UniSteiner.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef UFST_H
#define UFST_H

#include "geomtypes.h"
#include "gsttypes.h"
#include "point.h"

struct bsd;
struct full_set;
struct gst_metric;
struct point;
struct pset;

/*
 * A structure to keep track of one UFST.  They are kept in a hash table
 * so that we can rapidly identify duplicates.	We also keep them all in
 * one big list, which is doubly-linked so that we can rapidly delete a
 * duplicate that is suboptimal.
 */

struct ulist {
	struct ulist *		forw;	/* Next UFST in saved list */
	struct ulist *		back;	/* Previous UFST in saved list */
	struct ulist *		next;	/* Next UFST in hash table */
	int			size;	/* Number of terminals */
	struct full_set *	fst;	/* The actual FST */
};

/*
 * Global information used by the UFST generator.
 */

struct uinfo {
	struct pset *	pts;		/* The set of terminals */
	int *		x_order;	/* Order of terminals by X, Y, index */
					/* directions (plus 3 wrap dirs) */
	dist_t		mst_length;	/* Length of the MST */
	struct bsd *	bsd;		/* Bottleneck Steiner distance info */

	struct ulist ** hash;		/* FST hash table */
	struct ulist	list;		/* Head of circular UFST list */

	int             num_term_masks; /* Size of terminal mask in each FST */

	int		ntrees;		/* Final number of FSTs */
	struct full_set * full_sets;	/* Final list of FSTs */
	struct full_set ** hookp;	/* For adding to end of FST list */

	bool *		term_check;	/* Used by the WedgeTest and the FST
					   comparison */

	double minx, maxx, miny, maxy;	/* Point set characteristics */
	double dmax;

	struct gst_metric *	metric;	/* Information about the metric */
	struct gst_param *	params;	/* Parameters */

	dist_t		eps;		/* Relative epsilon used for comparisons */

	struct point    mean;           /* Mean point of all terminals */
	struct pset *	pts_org;	/* Original set of terminals (not translated) */

	/* Concatenator related */
	int hFSTCount;
};

/*
 * Local Types
 */

enum {
	TYPE_STRAIGHT = 0,
	TYPE_CORNER,
	TYPE_CROSS
};

struct hFST
{
	struct point	root;	/* Coordinates of the root of the FST */

  	/* Local displacement representation of root coordinates.                */
	/* Same technique as used in Euclidean generator, see efst.h for details */ 
	int		origin_term;	/* Terminal used as origin for root */ 
	struct point	droot;		/* Local disp. vector of root */
 
	double		length;		/* Total length of FST */
	int		*terms;		/* Terminals included in the FST */
	int		S;		/* Number of terminals used in construction */
	int		type;		/* Type of the root of the FST */
					/* i.e. TYPE_STRAIGHT, TYPE_CORNER or TYPE_CROSS */

	struct hFST *	left_tree;	/* Pointer to left subtree */
	struct hFST *	right_tree;	/* Pointer to right subtree */

	int		ext_left;	/* Left leg's extension orientation */
	int		ext_right;	/* Right leg's extension orientation */
	int		ext;		/* Extension orientation of hFST */

	int		index;		/* Used by hFSTs that are just terminals */
	double		UB;		/* Upper bound on tree spanning root and terminals */
	double		BS;		/* Shortest BSD between terminals */

	/* Variables related to the canonical form */
	int		status;		/* State of first hFST (Clean, Mixed, ...) */
	/* Mixed tree values */
	int		mixed_index;	/* Offset when lambda != 3m */
	bool		mixable;	/* State when lambda == 3m */
	/* Clean tree values */
	bool		right_legs[3];	/* Used orient. in legs to the right */
	bool		all_left_used;

	struct hFST *	next;
	bool		prev_identical;	/* Same terminals as in previous hFST */
};

/*
 * Functions
 */

extern struct gst_hypergraph *
		gst_generate_ufsts (int			nterms,
				    double *		terminals,
				    int			lambda,
				    struct gst_param *	params,
				    int *		status);

/*
 * It is possible to get extra statistical info by defining the macro below.
 */

/* #define STATISTICS_ENABLED
*/

#endif
