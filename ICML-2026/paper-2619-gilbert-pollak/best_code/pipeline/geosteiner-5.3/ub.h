/***********************************************************************

	$Id: ub.h,v 1.9 2022/11/19 13:45:55 warme Exp $

	File:	ub.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1997, 2022 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Declarations pertaining to the heuristic upper bound code.

************************************************************************

	Modification Log:

	a-1:	09/06/97	warme
		: Created.
	b-1:	02/28/2001	warme
		: Add new "struct ubinfo" to better encapsulate
		:  global state info.
		: Add startup and shutdown routines.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/


#ifndef UB_H
#define	UB_H

#include "gsttypes.h"

struct gst_hypergraph;
struct gst_solver;

/*
 * The following structure contains information needed by the upper
 * bounding heuristics -- information that is computed only once, and
 * then used each time the heuristic is called.
 */

struct ubinfo {
	int	num_rankings;	/* Number of valid rankings of the FSTs */
	int *	rankings [2];	/* Various rankings of the FSTs */
	int *	mst_edges;	/* The MST edges, shortest to longest */
	double	best_z;		/* Best solution seen during heuristic */
};


extern bool		_gst_compute_heuristic_upper_bound (
					double *		x,
					struct gst_solver *	solver);
extern void		_gst_shutdown_heuristic_upper_bound (
						struct ubinfo *	ubip);
extern struct ubinfo *	_gst_startup_heuristic_upper_bound (
					struct gst_hypergraph *	cip);

#endif
