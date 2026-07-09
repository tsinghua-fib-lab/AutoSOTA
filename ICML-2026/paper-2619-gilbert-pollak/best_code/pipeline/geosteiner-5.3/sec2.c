/***********************************************************************

	$Id: sec2.c,v 1.14 2023/04/03 17:06:05 warme Exp $

	File:	sec2.c
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 1997, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Deterministic separation procedure for the "generalized SEC's"
	(Subtour Elimination Constraints).  This method reduces the
	problem to min-cut on a simple bipartite network.

************************************************************************

	Modification Log:

	a-1:	05/16/97	warme
		: Created.
	b-1:	02/28/2001	warme
		: Changes for 3.1 release.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.  Upgrade fatals.
	e-4:	11/18/2022	warme
		: Simplify calling convention of various functions.
		: Fix -Wall issues.
	e-5:	04/03/2023	warme
		: Fix extra parens in disabled code.

************************************************************************/

#include "sec2.h"

#include "bb.h"
#include "constrnt.h"
#include "fatal.h"
#include "flow.h"
#include "genps.h"
#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include "parmblk.h"
#include "sec_comp.h"
#include "sec_heur.h"
#include "steiner.h"
#include "utils.h"


/*
 * Global Routines
 */

struct constraint *	_gst_sec_flow_separator (
					struct comp **		comp_hookp,
					double *		x,
					struct bbinfo *		bbip,
					struct constraint *	cp);


/*
 * External References
 */

	/* None */


/*
 * Local Types
 */

struct sec_flow_info {

	/* Data used by the flow solver... */
	struct flow_prob	prob;	/* The network flow formulation */
	struct flow_soln	soln;	/* The network flow solution */
	struct flow_temp	temp;	/* Temporary data structures */
};



/*
 * Local Routines
 */

static void			build_SEC_flow_formulation (
					struct comp *		comp,
					int			t,
					struct sec_flow_info *	flowp);
static double			do_flow_problem (struct comp *	comp,
						 int		t,
						 bitmap_t *	S);
static void			free_SEC_flow_formulation (
					struct sec_flow_info *	flowp);


/*
 * Local Variables
 */

	/* none */

/*
 * This is the main routine for the NEW deterministic separation
 * procedure for the generalized Subtour Elimination Constraints.  This
 * method reduces the problem to min-cut on a simple bipartite network.
 *
 * We take a chain of congested components.  We separate and then destroy
 * each congested component in the chain.
 */

#if 1

	struct constraint *
_gst_sec_flow_separator (

struct comp **		comp_hookp,	/* IN/OUT - congested component(s) */
double *		x,		/* IN - the LP solution to separate */
struct bbinfo *		bbip,		/* IN - branch-and-bound info */
struct constraint *	cp		/* IN - existing constraints */
)
{
int			t;
struct comp *		comp;
struct gst_hypergraph *	cip;
double			z;
bitmap_t *		S;
gst_channel_ptr		trace;

	cip = bbip -> cip;
	trace = bbip -> params -> print_solve_trace;
	(void) trace;

#if 0
	/* Two problems here:						*/
	/* 1. This is a library resident file calling non-library	*/
	/*    routines (genps.c is not in the library), and		*/
	/* 2. The postscript we generate is going to be	stuck inside of	*/
	/*    postscript comments(!) unless we disable postscript	*/
	/*    commenting on the channel.				*/
	_gst_plot_lp_solution (trace, cip, "LP solution to separate", x, BIG_PLOT);
#endif

	S = NEWA (cip -> num_vert_masks, bitmap_t);

	for (;;) {
		comp = *comp_hookp;
		if (comp EQ NULL) break;
		if (comp -> num_verts <= 0) {
			/* Free up this component... */
			*comp_hookp = comp -> next;
			comp -> next = NULL;
			_gst_free_congested_component (comp);
			continue;
		}
		if (comp -> num_verts EQ 1) {
			/* Because of the total-degree constraint, this	 */
			/* cannot happen unless something is very wrong! */
			FATAL_ERROR;
		}

		if (comp -> num_verts <= bbip -> params -> sec_enum_limit) {
			/* We have whittled this component down to	*/
			/* something small.  Use brute force on the	*/
			/* rest...					*/
			cp = _gst_enumerate_all_subtours (comp, cp, bbip);

			/* Free up this component... */
			*comp_hookp = comp -> next;
			comp -> next = NULL;
			_gst_free_congested_component (comp);
			continue;
		}

		/* Find the LEAST congested vertex.  this is the one	*/
		/* we are going to try to force into the solution,	*/
		/* since that is the one we would like to delete from	*/
		/* the set afterward...					*/
		t = _gst_find_least_congested_vertex (NULL, comp);

#if 0
		gst_channel_printf (trace,
			" %% -------------------------"
			    "-------------------------\n"
			" %% separating comp with %d verts, %d edges,"
			    " forcing vertex %d\n"
			" %% -------------------------"
			    "-------------------------\n",
			comp -> num_verts, comp -> num_edges,
			comp -> rverts [t] [0]);
#endif

		/* Find worst SEC violation involving vertex t. */
		z = do_flow_problem (comp, t, S);

#if 0
#if 0
		{
			int i, kmasks, *ip1, *ip2;
			bitmap_t * RS;
			kmasks = cip -> num_vert_masks;
			RS = NEWA (kmasks, bitmap_t);
			for (i = 0; i < kmasks; i++) {
				RS [i] = 0;
			}
			for (i = 0; i < comp -> num_verts; i++) {
				if (NOT BITON (S, i)) continue;
				ip1 = comp -> rverts [i];
				ip2 = comp -> rverts [i + 1];
				while (ip1 < ip2) {
					int j = *ip1++;
					SETBIT (RS, j);
				}
			}
			_gst_print_mask (trace, " S =", RS, cip -> num_verts);
			free (RS);
		}
#else
		_gst_print_mask (trace, " S =", S, comp -> num_verts);
#endif
		gst_channel_printf (trace, "	f(S) = %-24.15g\n", z);
#endif

		if (z < (1.0 - FUZZ)) {
			/* Add new violated constraint to the list... */
			cp = _gst_check_component_subtour (S,
							   comp,
							   cp,
							   x,
							   bbip);
		}

		/* We have found the worst violation (if any) involving	*/
		/* vertex t.  We can now eliminate t from further	*/
		/* consideration...					*/

		*comp_hookp = _gst_delete_vertex_from_component (t, comp, bbip);
	}

	free ((char *) S);

	return (cp);
}

/*
 * This is the main routine for the NEW deterministic separation
 * procedure for the generalized Subtour Elimination Constraints.  This
 * method reduces the problem to min-cut on a simple bipartite network.
 *
 * We take a chain of congested components.  We separate and then destroy
 * each congested component in the chain.
 */

#else

	struct constraint *
_gst_sec_flow_separator (

struct comp **		comp_hookp,	/* IN/OUT - congested component(s) */
double *		x,		/* IN - the LP solution to separate */
struct bbinfo *		bbip,		/* IN - branch-and-bound info */
struct constraint *	cp		/* IN - existing constraints */
)
{
int			i;
int			j;
int			t;
int			kmasks;
struct comp *		comp;
struct gst_hypergraph *	cip;
int *			ip1;
int *			ip2;
double			z;
bitmap_t *		S;
bitmap_t *		RS;
struct constraint *	cp2;
struct constraint *	cp3;
gst_channel_ptr		trace;

	cip = bbip -> cip;
	trace = bbip -> params -> print_solve_trace;

#if 0
	/* Two problems here:						*/
	/* 1. This is a library resident file calling non-library	*/
	/*    routines (genps.c is not in the library), and		*/
	/* 2. The postscript we generate is going to be	stuck inside of	*/
	/*    postscript comments(!) unless we disable postscript	*/
	/*    commenting on the channel.				*/
	_gst_plot_lp_solution (trace, cip, "LP solution to separate", x, BIG_PLOT);
#endif

	S = NEWA (2 * cip -> num_vert_masks, bitmap_t);
	RS = S + cip -> num_vert_masks;

	for (;;) {
		comp = *comp_hookp;
		if (comp EQ NULL) break;
		if (comp -> num_verts <= 0) {
			/* Free up this component... */
			*comp_hookp = comp -> next;
			comp -> next = NULL;
			_gst_free_congested_component (comp);
			continue;
		}
		if (comp -> num_verts EQ 1) {
			/* Because of the total-degree constraint, this	 */
			/* cannot happen unless something is very wrong! */
			FATAL_ERROR;
		}

		if (comp -> num_verts <= bbip -> params -> sec_enum_limit) {
			/* We have whittled this component down to	*/
			/* something small.  Use brute force on the	*/
			/* rest...					*/
			cp = _gst_enumerate_all_subtours (comp, cp, bbip);

			/* Free up this component... */
			*comp_hookp = comp -> next;
			comp -> next = NULL;
			_gst_free_congested_component (comp);
			continue;
		}

		/* Find the LEAST congested vertex.  this is the one	*/
		/* we are going to try to force into the solution,	*/
		/* since that is the one we would like to delete from	*/
		/* the set afterward...					*/
		t = _gst_find_least_congested_vertex (NULL, comp);

#if 0
		gst_channel_printf (trace,
		        " %% -------------------------"
			    "-------------------------\n"
			" %% separating comp with %d verts, %d edges,"
			    " forcing vertex %d\n"
			" %% -------------------------"
			    "-------------------------\n",
			comp -> num_verts, comp -> num_edges,
			comp -> rverts [t] [0]);
#endif

		/* Find worst SEC violation involving vertex t. */
		z = do_flow_problem (comp, t, S);

#if 0
#if 0
		kmasks = cip -> num_vert_masks;
		for (i = 0; i < kmasks; i++) {
			RS [i] = 0;
		}
		for (i = 0; i < comp -> num_verts; i++) {
			if (NOT BITON (S, i)) continue;
			ip1 = comp -> rverts [i];
			ip2 = comp -> rverts [i + 1];
			while (ip1 < ip2) {
				j = *ip1++;
				SETBIT (RS, j);
			}
		}
		_gst_print_mask (trace, " S =", RS, cip -> num_verts);
#else
		_gst_print_mask (trace, " S =", S, comp -> num_verts);
#endif
		gst_channel_printf (trace, "	f(S) = %-24.15g\n", z);
#endif

		cp2 = NULL;
		if (z < (1.0 - FUZZ)) {
			/* Perform reductions (including greedy		*/
			/* strengthening) to find the strongest core	*/
			/* violations (that are subsets of this flow	*/
			/* solution) that we can.			*/
			cp2 = _gst_check_component_subtour (S,
							    comp,
							    NULL,
							    x,
							    bbip);
		}

		if (cp2 NE NULL) {
			/* Compute a single mask that is the union of	*/
			/* all subtours discovered.  Also add these	*/
			/* constraints onto the output list of		*/
			/* generated constraints.			*/
			kmasks = cip -> num_vert_masks;
			for (i = 0; i < kmasks; i++) {
				RS [i] = 0;
			}
			while (cp2 NE NULL) {
				cp3 = cp2;
				cp2 = cp3 -> next;
				cp3 -> next = cp;
				cp = cp3;
				FATAL_ERROR_IF (cp3 -> type NE CT_SUBTOUR);
				for (i = 0; i < kmasks; i++) {
					RS [i] |= cp3 -> mask [i];
				}
			}
			/* Delete all of these vertices from the	*/
			/* component at once.				*/
			create_masks (comp);
			for (i = 0; i < comp -> num_verts; i++) {
				ip1 = comp -> rverts [i];
				ip2 = comp -> rverts [i + 1];
				while (ip1 < ip2) {
					j = *ip1++;
					if (BITON (RS, j)) {
						CLRBIT (comp -> tmap, i);
						break;
					}
				}
			}
			*comp_hookp = _gst_delete_vertex_from_component (-1,
									 comp,
									 bbip);
		}
		else {
			/* No violation involving t.  Delete t from	*/
			/* the component and pick another.		*/
			*comp_hookp = _gst_delete_vertex_from_component (t,
									 comp,
									 bbip);
		}
	}

	free ((char *) S);

	return (cp);
}

#endif

/*
 * This routine performs a single flow sub-problem.  We are given a
 * vertex to FORCE into the solution.  We find the worst SEC violation
 * involving that vertex.
 */

	static
	double
do_flow_problem (

struct comp *		comp,		/* IN - congested component */
int			t,		/* IN - vertex to force */
bitmap_t *		S		/* OUT - a most-violated subtour */
)
{
int			i;
int			j;
int			k;
int			size;
int *			ip1;
int *			ip2;
double			sum;
double			z;
struct sec_flow_info	flow_info;

	build_SEC_flow_formulation (comp, t, &flow_info);

	_gst_compute_max_flow (&flow_info.prob,
			       &flow_info.temp,
			       &flow_info.soln);

	/* Construct the solution.  These are the vertices that are	*/
	/* on the FAR side of the cut.  Also, compute the z = f(S)	*/
	/* value to return to our caller.				*/
	k = BMAP_ELTS (comp -> num_verts);
	for (i = 0; i < k; i++) {
		S [i] = 0;
	}
	size = 0;
	for (i = 0; i < comp -> num_verts; i++) {
		if (BITON (flow_info.soln.cut, i)) continue;
		SETBIT (S, i);
		++size;
	}

	free_SEC_flow_formulation (&flow_info);

	sum = 0.0;
	for (i = 0; i < comp -> num_edges; i++) {
		ip1 = comp -> everts [i];
		ip2 = comp -> everts [i + 1];
		j = 0;
		while (ip1 < ip2) {
			k = *ip1++;
			if (BITON (S, k)) {
				++j;
			}
		}
		if (j >= 2) {
			sum += (j - 1) * (comp -> x [i]);
		}
	}

	z = ((double) size) - sum;

	return (z);
}

/*
 * This routine builds the network flow formulation for the
 * deterministic separation procedure.
 */

	static
	void
build_SEC_flow_formulation (

struct comp *		comp,		/* IN - congested component */
int			t,		/* IN - vertex to force */
struct sec_flow_info *	flowp		/* OUT - SEC flow formulation */
)
{
int			i;
int			j;
int			k;
int			nverts;
int			nedges;
int			nmasks;
int			num_arcs;
int			num_nodes;
int			e;
int			arc_num;
int			num_used_edges;
int			first_edge_node;
int *			used_edges;
bitmap_t *		unused_edge_mask;
struct flow_prob *	prob;
int *			ip1;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			outlist;
int *			inlist;
int *			srcp;
int *			dstp;
double *		capp;
int *			counts;
int **			ptrs;
double			sum;

	nverts	= comp -> num_verts;
	nedges	= comp -> num_edges;

	/* Compute a list of all component edges that	*/
	/* DO NOT contain the vertex "t".		*/
	nmasks		 = BMAP_ELTS (nedges);
	unused_edge_mask = NEWA (nmasks, bitmap_t);
	for (i = 0; i < nmasks; i++) {
		unused_edge_mask [i] = 0;
	}
	num_used_edges = nedges;
	ep1 = comp -> vedges [t];
	ep2 = comp -> vedges [t + 1];
	while (ep1 < ep2) {
		e = *ep1++;
		SETBIT (unused_edge_mask, e);
		--num_used_edges;
	}

	used_edges = NEWA (num_used_edges, int);
	ep1 = used_edges;
	for (i = 0; i < nedges; i++) {
		if (BITON (unused_edge_mask, i)) continue;
		*ep1++ = i;
	}
	free ((char *) unused_edge_mask);

	if (ep1 NE (used_edges + num_used_edges)) {
		/* lost count somewhere? */
		FATAL_ERROR;
	}

	/* Tally up the total number of nodes and arcs needed	*/
	/* for the flow graph.  For the sake of	simplicity, we	*/
	/* include a node for t, but there will be no arcs	*/
	/* associated with it...				*/

	/* One node per vertex.  One source and one sink	*/
	/* node.  One node per USED edge.			*/
	num_nodes	= nverts + 2 + num_used_edges;

	/* One arc per vertex, one arc per USED edge,		*/
	/* plus one arc for each vertex of every USED edge.	*/
	num_arcs	= nverts + num_used_edges;
	for (i = 0; i < num_used_edges; i++) {
		e = used_edges [i];
		num_arcs += (comp -> everts [e + 1] - comp -> everts [e]);
	}

	/* Start filling in the flow problem instance... */
	prob = &(flowp -> prob);
	prob -> num_nodes	= num_nodes;
	prob -> num_arcs	= num_arcs;

	/* Assign node numbers for the source and sink nodes... */
	prob -> source	= nverts;
	prob -> sink	= nverts + 1;
	first_edge_node	= nverts + 2;

	/* Now that we know how big the directed flow graph is,	*/
	/* allocate storage for the various data structures...	*/

	prob -> out		= NEWA (num_nodes + 1, int *);
	prob -> in		= NEWA (num_nodes + 1, int *);
	prob -> arc_src		= NEWA (num_arcs, int);
	prob -> arc_dst		= NEWA (num_arcs, int);
	prob -> capacity	= NEWA (num_arcs, double);

	outlist			= NEWA (num_arcs, int);
	inlist			= NEWA (num_arcs, int);

	/* Generate the arcs from the source node to each USED edge. */

	srcp = prob -> arc_src;
	dstp = prob -> arc_dst;
	capp = prob -> capacity;
	arc_num = 0;
	for (i = 0; i < num_used_edges; i++) {
		e = used_edges [i];
		j = first_edge_node + i;
		*srcp++		= prob -> source;
		*dstp++		= j;
		*capp++		= comp -> x [e];
		++arc_num;

		/* Generate the arcs from each edge node to the		*/
		/* corresponding vertex nodes.  These all have weight	*/
		/* 2, which is essentially infinite.			*/
		vp1 = comp -> everts [e];
		vp2 = comp -> everts [e + 1];
		while (vp1 < vp2) {
			k = *vp1++;
			*srcp++		= j;
			*dstp++		= k;
			*capp++		= 2.0;
			++arc_num;
		}
	}

	free ((char *) used_edges);

	/* Now generate one arc from each vertex node to the sink	*/
	/* node.  These all have weight (Bi - 1), where Bi is the	*/
	/* congestion level of vertex i.				*/
	for (i = 0; i < nverts; i++) {
		sum = -1.0;
		ep1 = comp -> vedges [i];
		ep2 = comp -> vedges [i + 1];
		while (ep1 < ep2) {
			e = *ep1++;
			sum += comp -> x [e];
		}
		*srcp++		= i;
		*dstp++		= prob -> sink;
		*capp++		= sum;
		++arc_num;
	}

	FATAL_ERROR_IF (arc_num NE num_arcs);

	/* We have now specified the directed flow graph as a	*/
	/* list of directed arcs.  Time to construct the	*/
	/* adjacency lists -- for each node we build a list of	*/
	/* outgoing and incoming arc numbers.  Do the outgoing	*/
	/* lists first...					*/

	counts	= NEWA (num_nodes, int);
	ptrs	= NEWA (num_nodes, int *);

	for (i = 0; i < num_nodes; i++) {
		counts [i] = 0;
	}
	for (i = 0; i < num_arcs; i++) {
		++(counts [prob -> arc_src [i]]);
	}
	ip1 = outlist;
	for (i = 0; i < num_nodes; i++) {
		ptrs [i] = ip1;
		prob -> out [i] = ip1;
		ip1 += counts [i];
	}
	prob -> out [i] = ip1;
	for (i = 0; i < num_arcs; i++) {
		j = prob -> arc_src [i];
		ip1 = ptrs [j]++;
		*ip1 = i;
	}

	/* Now do the incoming arc lists... */
	for (i = 0; i < num_nodes; i++) {
		counts [i] = 0;
	}
	for (i = 0; i < num_arcs; i++) {
		++(counts [prob -> arc_dst [i]]);
	}
	ip1 = inlist;
	for (i = 0; i < num_nodes; i++) {
		ptrs [i] = ip1;
		prob -> in [i] = ip1;
		ip1 += counts [i];
	}
	prob -> in [i] = ip1;
	for (i = 0; i < num_arcs; i++) {
		k = prob -> arc_dst [i];
		ip1 = ptrs [k]++;
		*ip1 = i;
	}

	/* Free temporary memory used to build things... */
	free ((char *) counts);
	free ((char *) ptrs);

	/* Initialize the buffers used to hold flow solutions */
	/* and temporary data... */
	_gst_create_flow_solution_data (prob, &(flowp -> soln));
	_gst_create_flow_temp_data (prob, &(flowp -> temp));
}

/*
 * This routine frees up the memory allocated by the given SEC max-flow
 * formulation.
 */

	static
	void
free_SEC_flow_formulation (

struct sec_flow_info *	flowp		/* IN - SEC flow formulation */
)
{
	/* Free up the buffers used to hold flow solutions */
	/* and temporary data... */
	_gst_free_flow_temp_data (&(flowp -> temp));
	_gst_free_flow_solution_data (&(flowp -> soln));

	/* Free up the problem formulation... */
	free ((char *) (flowp -> prob.in [0]));
	free ((char *) (flowp -> prob.out [0]));

	free ((char *) (flowp -> prob.capacity));
	free ((char *) (flowp -> prob.arc_dst));
	free ((char *) (flowp -> prob.arc_src));
	free ((char *) (flowp -> prob.in));
	free ((char *) (flowp -> prob.out));
}
