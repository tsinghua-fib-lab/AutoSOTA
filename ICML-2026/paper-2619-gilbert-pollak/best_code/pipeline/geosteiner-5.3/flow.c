/***********************************************************************

	$Id: flow.c,v 1.12 2023/04/03 17:40:50 warme Exp $

	File:	flow.c
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 1996, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	A standard network maximum flow solver.  Uses the augmenting
	path method with breadth-first search.

************************************************************************

	Modification Log:

	a-1:	05/13/96	warme
		: Created.
	a-2:	07/05/96	warme
		: Split off from sec.c.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.
	e-4:	11/18/2022	warme
		: Fix -Wall issue.
	e-5:	04/03/2023	warme
		: Fix a bug that made temp arrays be wrong size.

************************************************************************/

#include "flow.h"

#include "fatal.h"
#include "logic.h"
#include "memory.h"
#include <stdlib.h>
#include "steiner.h"


/*
 * Global Routines
 */

void	_gst_compute_max_flow (struct flow_prob *	prob,
			       struct flow_temp *	temp,
			       struct flow_soln *	soln);
void	_gst_create_flow_solution_data (
				struct flow_prob *	prob,
				struct flow_soln *	soln);
void	_gst_create_flow_temp_data (struct flow_prob *	prob,
				    struct flow_temp *	temp);
void	_gst_free_flow_solution_data (struct flow_soln * soln);
void	_gst_free_flow_temp_data (struct flow_temp *	temp);


/*
 * External References
 */

	/* none */


/*
 * Local Equates
 */


#define FUZZ			0.000001

/*
 * This routine computes the maximum flow in the given directed flow
 * graph.  It uses the augmenting-path approach, with breadth-first
 * search to assure that a shortest augmenting path is found at every
 * iteration.
 */

	void
_gst_compute_max_flow (

struct flow_prob *	prob,	/* IN - the problem instance to solve */
struct flow_temp *	temp,	/* IN/OUT - temporary buffers */
struct flow_soln *	soln	/* OUT - the maximum flow solution */
)
{
int			i;
int			j;
int			k;
int			num_nodes;
int			num_arcs;
int			nmasks;
int *			ip1;
int *			ip2;
int *			headp;
int *			tailp;
int *			arc_src;
int *			arc_dst;
int *			pred_arc;
double *		delta;
double *		x;
double *		c;
double			z;
double			deltai;
double			deltaj;

	num_nodes	= prob -> num_nodes;
	num_arcs	= prob -> num_arcs;
	arc_src		= prob -> arc_src;
	arc_dst		= prob -> arc_dst;
	x		= soln -> flow;
	c		= prob -> capacity;
	pred_arc	= temp -> pred_arc;
	delta		= temp -> delta;

	/* Standard augmenting path method.  Do a breadth-first	*/
	/* search looking for an augmenting path from the	*/
	/* source to the sink.  If none exists we are done.	*/
	/* Otherwise, use the augmenting path to increase the	*/
	/* total flow along the path.				*/

	/* initial feasible flow = 0. */
	z = 0;
	for (i = 0; i < num_arcs; i++) {
		x [i] = 0.0;
	}

	for (;;) {
		/* Do breadth-first search for an augmenting path. */
		for (i = 0; i < num_nodes; i++) {
			pred_arc [i]	= -1;
			delta [i]	= 0.0;
		}
		/* Label the source node and put it into the queue. */
		i = prob -> source;
		pred_arc [i]	= 0;
		delta [i]	= INFINITE_FLOW;

		headp = temp -> queue;
		tailp = headp;
		*tailp++ = i;
		deltai = delta [i];

		for (;;) {
			if (headp >= tailp) {
				/* Queue is empty.  No augmenting path	*/
				/* exists, so we are done!		*/
				soln -> z = z;

				/* Generate bit-mask of nodes visited. */
				nmasks = BMAP_ELTS (num_nodes);
				for (i = 0; i < nmasks; i++) {
					soln -> cut [i] = 0;
				}
				for (i = 0; i < num_nodes; i++) {
					if (temp -> pred_arc [i] >= 0) {
						SETBIT (soln -> cut, i);
					}
				}
				return;
			}
			/* Unqueue next node to scan. */
			i = *headp++;
			if (i EQ prob -> sink) {
				/* Found augmenting path! */
				break;
			}
			deltai = delta [i];
			/* Scan all outgoing edges... */
			ip1 = prob -> out [i];
			ip2 = prob -> out [i + 1];
			while (ip1 < ip2) {
				k = *ip1++;	/* k = outgoing arc #. */
				j = arc_dst [k];
				deltaj = c [k] - x [k];
				if ((deltaj > FUZZ) AND
				    (pred_arc [j] < 0)) {
					if (deltaj > deltai) {
						deltaj = deltai;
					}
					pred_arc [j]	= k + 1;
					delta [j]	= deltaj;
					*tailp++ = j;
				}
			}
			/* Scan all incoming edges... */
			ip1 = prob -> in [i];
			ip2 = prob -> in [i + 1];
			while (ip1 < ip2) {
				k = *ip1++;	/* k = incoming arc #. */
				j = arc_src [k];
				if ((x [k] > FUZZ) AND
				    (pred_arc [j] < 0)) {
					deltaj = x [k];
					if (deltaj > deltai) {
						deltaj = deltai;
					}
					pred_arc [j]	= k + 1;
					delta [j]	= deltaj;
					*tailp++ = j;
				}
			}
		}

		/* An augmenting path has been found!  Trace it	*/
		/* back from the sink to the source, increasing	*/
		/* the flow along forward edges and decreasing	*/
		/* it along backward edges...			*/
		/* (i EQ prob -> sink) at this point...		*/
		deltai = delta [i];
		while (i NE prob -> source) {
			k = pred_arc [i];
			FATAL_ERROR_IF ((k <= 0) OR (k > num_arcs));
			--k;
			j = arc_src [k];
			if (i EQ j) {
				/* backward edge... */
				x [k] -= deltai;
				j = arc_dst [k];
			}
			else {
				/* forward edge... */
				x [k] += deltai;
			}
			i = j;
		}
		z += deltai;
	}
}

/*
 * Given a properly-initialized problem instance, this routine
 * creates and initializes the given solution buffer.
 */

	void
_gst_create_flow_solution_data (

struct flow_prob *	prob,	/* IN - the flow problem instance */
struct flow_soln *	soln	/* OUT - solution buffer */
)
{
int		num_nodes;
int		num_arcs;
int		nmasks;

	num_nodes = prob -> num_arcs;
	num_arcs = prob -> num_arcs;

	nmasks = BMAP_ELTS (num_nodes);

	soln -> flow = NEWA (num_arcs, double);
	soln -> cut  = NEWA (nmasks, bitmap_t);
}


/*
 * This routine frees the memory associated with the given network
 * flow solution.
 */

	void
_gst_free_flow_solution_data (

struct flow_soln *	soln	/* IN - flow solution to free */
)
{
	free ((char *) (soln -> cut));
	free ((char *) (soln -> flow));
}

/*
 * Given a properly-initialized problem instance, this routine
 * creates and initializes the given temporary buffers.
 */

	void
_gst_create_flow_temp_data (

struct flow_prob *	prob,	/* IN - the flow problem instance */
struct flow_temp *	temp	/* OUT - temporary buffers */
)
{
int		num_nodes;

	num_nodes = prob -> num_nodes;

	temp -> delta	 = NEWA (num_nodes, double);
	temp -> pred_arc = NEWA (num_nodes, int);
	temp -> queue	 = NEWA (num_nodes, int);
}


/*
 * This routine frees the memory associated with the given network
 * flow temporary buffers.
 */

	void
_gst_free_flow_temp_data (

struct flow_temp *	temp	/* IN - flow temp buffers to free */
)
{
	free ((char *) (temp -> queue));
	free ((char *) (temp -> pred_arc));
	free ((char *) (temp -> delta));
}
