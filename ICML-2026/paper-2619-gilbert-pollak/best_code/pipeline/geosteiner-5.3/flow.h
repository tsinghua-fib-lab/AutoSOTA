/***********************************************************************

	$Id: flow.h,v 1.8 2022/11/19 13:45:51 warme Exp $

	File:	flow.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Data structures and declarations for the maximum
	flow network solver.

************************************************************************

	Modification Log:

	a-1:	07/05/96	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef FLOW_H
#define	FLOW_H

#include "bitmaskmacros.h"


/*
 * The value used to represent "infinite" flow.
 */

#define	INFINITE_FLOW		10000.0


/*
 * The following structure contains all of the information needed
 * to describe a single network flow problem instance.
 */

struct flow_prob {
	int		num_nodes;	/* Number of nodes */
	int		num_arcs;	/* Number of arcs */
	int		source;		/* Source node number */
	int		sink;		/* Sink node number */
	int **		out;		/* outgoing arcs for each node */
	int **		in;		/* incoming arcs for each node */
	int *		arc_src;	/* source node for each arc */
	int *		arc_dst;	/* destination node for each arc */
	double *	capacity;	/* capacity for each arc */
};


/*
 * The next structure contains all of the information that describes
 * a network max-flow solution.
 */

struct flow_soln {
	double		z;		/* total network src->sink flow */
	double *	flow;		/* flow for each arc */
	bitmap_t *	cut;		/* nodes on source side of cut */
};


/*
 * Finally, this structure contains temporary data structures used
 * internally by the network flow solver.  Its contents are unimportant
 * to the outside world.
 */

struct flow_temp {
	/* Pre-allocated temporary buffers used during a	*/
	/* single run of the flow solver -- may be reused for	*/
	/* several consecutive runs...				*/
	int *		pred_arc;	/* predecessor arc in path */
	double *	delta;		/* flow increment available at node */
	int *		queue;		/* breadth-first queue of nodes */
};


extern void	_gst_compute_max_flow (struct flow_prob *	prob,
				       struct flow_temp *	temp,
				       struct flow_soln *	soln);
extern void	_gst_create_flow_solution_data (
					struct flow_prob *	prob,
					struct flow_soln *	soln);
extern void	_gst_create_flow_temp_data (struct flow_prob *	prob,
					    struct flow_temp *	temp);
extern void	_gst_free_flow_solution_data (struct flow_soln * soln);
extern void	_gst_free_flow_temp_data (struct flow_temp *	temp);


#endif
