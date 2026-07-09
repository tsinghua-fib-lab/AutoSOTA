/***********************************************************************

	$Id: bsd.h,v 1.11 2022/11/19 13:45:49 warme Exp $

	File:	bsd.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1998, 2022 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Declarations for the Bottleneck Steiner Distance
	implementation.

************************************************************************

	Modification Log:

	a-1:	10/04/98	warme
		: Created.
	a-2:	02/28/2001	warme
		: Changes for 3.1 release.  Hide certain interfaces
		:  in the .c file where they belong.
	a-3:	02/28/2001	warme
		: Changes for 3.1 release.  Migrate data from bsd
		:  structure to local vars.
	a-4:	08/01/2002	martinz
		: Implemented new linear space and logarithmic time
		: lookup data structure by Mandoiu et al.
		: Old data structure is method 0 and new is method 1.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef	BSD_H
#define	BSD_H

#include "geomtypes.h"
#include "gsttypes.h"

struct bsd;
struct edge;


/*
 * A structure to encapsulate all the data we store for rapidly
 * computing the BSD of two terminals.
 */

struct bsd {
	int		method;		/* Implementation method to use */

	/* Stuff common to all implementations. */
	int		n;		/* Number of terminals */
	struct edge *	mst_edges;	/* List of MST edges */
	struct mstadj ** adj_list;	/* Adjacency list */

	/* Stuff for the lower triangular matrix of edge #'s implementation. */
	int16u *	ematrix;	/* The full matrix */

	/* Stuff for the linear space implementation. */
	int *		edge;		/* Edge index of nearest nb */
	int *		parent;		/* Parent of node */
};


extern dist_t		_gst_bsd (struct bsd * bsdp, int i, int j);
extern struct bsd *	_gst_compute_bsd (int		nedges,
					  struct edge *	mst_edges,
					  int		method);
extern bool		_gst_is_mst_edge (struct bsd * bsdp, int i, int j);
extern void		_gst_shutdown_bsd (struct bsd * bsdp);

#endif
