/***********************************************************************

	$Id: sec_heur.c,v 1.16 2023/04/03 17:03:50 warme Exp $

	File:	sec_heur.c
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 1996, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Heuristic separation procedure for SEC's (Subtour
	Elimination Constraints).  The heuristic is that we reduce
	the problem from weighted hypergraphs down to standard
	weighted graphs by computing a geometric MST for each
	hyperedge.

************************************************************************

	Modification Log:

	a-1:	05/13/96	warme
		: Created.
	b-1:	02/28/2001	warme
		: Numerous changes for 3.1 release.
		: Completely rewrote subtour enumeration code to
		:  use fast incremental update of subtour constraint
		:  LHS and RHS values.
	c-1:	08/05/2002	benny
		: Some changes for library release.
		: Uses parameters.
		: Uses channels for trace output.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Use better encapsulation for time conversions.
		: Upgrade fatals.
	e-4:	11/18/2022	warme
		: Simplify calling convention of various functions.
		: Fix -Wall issues.
	e-5:	04/03/2023	warme
		: Fixed bug wherein find_almost_integer_cycles()
		:  was generating subtours that are not violated,
		:  which was causing crashes up in
		:  _gst_branch_and_cut() when do_separations()
		:  returns (cp NE NULL) but no constraints get
		:  added to the LP.

************************************************************************/

#include "sec_heur.h"

#include "bb.h"
#include "constrnt.h"
#include "expand.h"
#include "fatal.h"
#include "flow.h"
#include "geosteiner.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "mst.h"
#include "parmblk.h"
#include "point.h"
#include "sec_comp.h"
#include <stdlib.h>
#include "steiner.h"
#include "utils.h"


/*
 * Global Routines
 */

struct constraint *	_gst_check_subtour (
					bitmap_t *		stour,
					struct constraint *	clist,
					double *		x,
					struct bbinfo *		bbip);
struct constraint *	_gst_enumerate_all_subtours (
					struct comp *		comp,
					struct constraint *	cp,
					struct bbinfo *		bbip);
struct constraint *	_gst_find_integer_cycles (
					double *		x,
					struct constraint *	cp,
					struct bbinfo *		bbip);
struct constraint *	_gst_find_small_subtours (
					struct comp *		comp,
					struct constraint *	cp,
					struct bbinfo *		bbip);
bool			_gst_is_equal (bitmap_t *	bp1,
				       bitmap_t *	bp2,
				       int		nmasks);
struct constraint *	_gst_sec_flow_heuristic (struct comp *		comp,
						 double *		x,
						 struct bbinfo *	bbip,
						 struct constraint *	cp);


/*
 * External References
 */

	/* None */


/*
 * Local Equates
 */

#define CYCLE_LIMIT	250


/*
 * Local Types
 */

struct sec_heur_info {

	/* Data used by the flow solver... */
	struct flow_prob	prob;	/* The network flow formulation */
	struct flow_soln	soln;	/* The network flow solution */
	struct flow_temp	temp;	/* Temporary data structures */

	/* Data used to set the arc capacities and modify the	*/
	/* flow network during SEC separation. */
	int		src_arc1;	/* First source arc */
	int		dst_arc1;	/* First destination arc */
	int **		edge_lists;	/* edges for each arc */
};



/*
 * Local Routines
 */

static void			build_heuristic_SEC_formulation (
					struct comp *		comp,
					struct gst_hypergraph *	cip,
					struct sec_heur_info *	hsecp);
static struct constraint *	check_unique_subtour (
					bitmap_t *		stour,
					int			kmasks,
					struct constraint *	cp,
					double *		x,
					struct bbinfo *		bbip);
static struct pset *		compute_vertex_coords (
					struct comp *,
					struct gst_hypergraph *);
static struct constraint *	cwalk (int,
				       int,
				       bitmap_t *,
				       bitmap_t *,
				       bitmap_t *,
				       bitmap_t *,
				       bitmap_t *,
				       bitmap_t *,
				       bitmap_t *,
				       int *,
				       int *,
				       int *,
				       double *,
				       struct constraint *,
				       struct bbinfo *);
#ifdef OLD_ENUMERATION
static struct constraint *	enumerate_subtours (int,
						    int,
						    double,
						    bitmap_t *,
						    int *,
						    int *,
						    struct comp *,
						    struct constraint *,
						    struct bbinfo *);
#endif
static struct constraint *	find_almost_integer_cycles (
						int,
						int *,
						bitmap_t *,
						bitmap_t *,
						bitmap_t *,
						double *,
						struct bbinfo *);
static void			free_heuristic_SEC_formulation (
					struct sec_heur_info *	hsecp);
static int			full_set_mst (struct comp *,
					      int,
					      struct pset *,
					      struct edge *);
static bool			fwalk (int,
				       int,
				       bitmap_t *,
				       bitmap_t *,
				       struct gst_hypergraph *);
static struct constraint *	heuristically_find_violated_secs (
					struct comp *		comp,
					struct bbinfo *		bbip,
					struct sec_heur_info *	hsecp,
					double *		x,
					struct constraint *	clist);
#ifndef OLD_ENUMERATION
static struct constraint *	recurse_enum (int	limit,
					      int	navail,
					      int *	avail,
					      int	nchosen,
					      int *	chosen,
					      int	nexcl,
					      int *	excluded,
					      int *	vstat,
					      double	lhs,
					      double	rhs,
					      bool	supersets,
					      struct comp *	comp,
					      struct constraint * cp,
					      struct bbinfo *	bbip);
#endif
static void			set_arc_capacities (struct comp *,
						    struct sec_heur_info *);
static void			sort_edges (int, struct edge *, int *);


/*
 * Local Variables
 */

	/* none */

/*
 * This routine heuristically reduces the weighted hypergraph of FSTs
 * to a conventional weighted graph by computing a minimum spanning
 * tree for each FST.  We then use the flow formulation of
 * Padberg ("Trees and Cuts" paper) to find SEC violations in the
 * weighted graph.  This method is heuristic since violations will be
 * discovered or not depending on the particular reduction from
 * hypergraph to graph that is chosen.
 */

	struct constraint *
_gst_sec_flow_heuristic (

struct comp *		comp,		/* IN - component to separate */
double *		x,		/* IN - current LP solution */
struct bbinfo *		bbip,		/* IN - branch and bound info */
struct constraint *	cp		/* IN - existing constraints */
)
{
struct gst_hypergraph *	cip;
struct sec_heur_info	sec_heur_info;

	cip	  = bbip -> cip;

	if (cip -> pts EQ NULL) {
		/* This heuristic needs coordinates for each vertex! */
		return (cp);
	}

	if (comp -> num_verts > 1) {
		/* Build an SEC formulation for this component... */
		build_heuristic_SEC_formulation (comp,
						 cip,
						 &sec_heur_info);

		/* Do the heuristic separation... */
		cp = heuristically_find_violated_secs (comp,
						       bbip,
						       &sec_heur_info,
						       x,
						       cp);

		free_heuristic_SEC_formulation (&sec_heur_info);
	}

	return (cp);
}

/*
 * This routine builds the initial formulation for the heuristic
 * separation procedure.
 */

	static
	void
build_heuristic_SEC_formulation (

struct comp *		comp,		/* IN - congested component. */
struct gst_hypergraph *	cip,		/* IN - compatibility info. */
struct sec_heur_info *	hsecp		/* OUT - SEC flow formulation. */
)
{
int			i;
int			j;
int			k;
int			n;
int			nverts;
int			nedges;
int			num_fst_edges;
int			num_unique_edges;
int			total_edges;
int			num_arcs;
int			num_nodes;
int			arc_num;
struct pset *		pts;
struct flow_prob *	prob;
struct edge *		fs_mst_edges;
int *			fs_num;
struct edge *		ep1;
struct edge *		ep2;
struct edge *		ep_endp;
int *			nump;
int *			outlist;
int *			inlist;
int *			fslist;
int *			srcp;
int *			dstp;
int *			fs_ip;
int *			ip;
int *			counts;
int **			ptrs;

	nverts	= comp -> num_verts;
	nedges	= comp -> num_edges;

	/* Compute actual X,Y coordinates for each vertex in	*/
	/* the congested component...				*/
	pts = compute_vertex_coords (comp, cip);

	/* Tally up the total number of edges needed to compute	*/
	/* a minimum spanning tree for the specified vertices	*/
	/* of each edge.  This is the sum of (|e| - 1) for	*/
	/* all edges e...					*/
	total_edges = (comp -> everts [nedges] - comp -> everts [0])
		      - nedges;

	/* Allocate temporary arrays to hold the edges of the	*/
	/* minimum spanning trees of each FST, together with	*/
	/* the FST number for the edge.				*/
	fs_mst_edges	= NEWA (total_edges, struct edge);
	fs_num		= NEWA (total_edges, int);

	/* Compute a Minimum Spanning Tree for each FST.	*/
	/* Record the edges for each, and the FST number	*/
	/* that the edge came from.				*/
	ep1 = fs_mst_edges;
	nump = fs_num;
	for (i = 0; i < nedges; i++) {
		num_fst_edges = full_set_mst (comp, i, pts, ep1);
		ep1 += num_fst_edges;
		for (j = 0; j < num_fst_edges; j++) {
			*nump++ = i;
		}
	}

	FATAL_ERROR_IF ((ep1 NE (fs_mst_edges + total_edges)) OR
			(nump NE (fs_num + total_edges)));

	free ((char *) pts);

	/* Sort the list of all MST edges into increasing order	*/
	/* (primary key = low endpoint, secondary key = high	*/
	/* endpoint).						*/
	sort_edges (total_edges, fs_mst_edges, fs_num);

	/* Scan through the sorted edges and count the number	*/
	/* of unique (a,b) pairs.				*/
	if (total_edges <= 0) {
		num_unique_edges = 0;
	}
	else {
		num_unique_edges = 1;
		ep1 = &fs_mst_edges [1];
		for (i = 1; i < total_edges; i++, ep1++) {
			if ((ep1 [-1].p1 NE ep1 [0].p1) OR
			    (ep1 [-1].p2 NE ep1 [0].p2)) {
				++num_unique_edges;
			}
		}
	}

	/* Compute the total number of directed arcs in the	*/
	/* flow graph.  Each undirected edge becomes a pair of	*/
	/* head-to-tail directed arcs, plus 2 extra arcs per	*/
	/* vertex -- one from the source node and one to the	*/
	/* sink node.						*/
	num_nodes		= nverts + 2;
	num_arcs		= 2 * num_unique_edges + 2 * nverts;

	/* Start filling in the flow problem instance... */
	prob = &(hsecp -> prob);
	prob -> num_nodes	= num_nodes;
	prob -> num_arcs	= num_arcs;

	/* Assign node numbers for the source and sink nodes... */
	prob -> source	= nverts;
	prob -> sink	= nverts + 1;

	/* Now that we know how big the directed flow graph is,	*/
	/* allocate storage for the various data structures...	*/

	prob -> out		= NEWA (num_nodes + 1, int *);
	prob -> in		= NEWA (num_nodes + 1, int *);
	hsecp -> edge_lists	= NEWA (2 * num_unique_edges + 1, int *);
	prob -> arc_src		= NEWA (num_arcs, int);
	prob -> arc_dst		= NEWA (num_arcs, int);

	outlist			= NEWA (num_arcs, int);
	inlist			= NEWA (num_arcs, int);
	fslist			= NEWA (2 * total_edges, int);

	prob -> capacity	= NEWA (num_arcs, double);

	/* Loop through all of the undirected edges, generating	*/
	/* the pair of directed arcs for each.  For each arc	*/
	/* generated, generate the list of edges from which	*/
	/* it draws capacity.					*/
	srcp = prob -> arc_src;
	dstp = prob -> arc_dst;

	ep1	= &fs_mst_edges [0];
	ep_endp	= &fs_mst_edges [total_edges];
	fs_ip	= fs_num;
	arc_num	= 0;
	for (i = 0; i < total_edges;) {
		/* Find number of consecutive "copies" of this edge... */
		ep2 = ep1 + 1;
		n = 1;
		for (;;) {
			if (ep2 >= ep_endp) break;
			if (ep1 -> p1 NE ep2 -> p1) break;
			if (ep1 -> p2 NE ep2 -> p2) break;
			++ep2;
			++n;
		}
		/* Record the A -> B arc... */
		*srcp++ = ep1 -> p1;
		*dstp++ = ep1 -> p2;
		hsecp -> edge_lists [arc_num] = fslist;
		for (j = 0; j < n; j++) {
			*fslist++ = fs_ip [j];
		}
		++arc_num;

		/* Record the B -> A arc... */
		*srcp++ = ep1 -> p2;
		*dstp++ = ep1 -> p1;
		hsecp -> edge_lists [arc_num] = fslist;
		for (j = 0; j < n; j++) {
			*fslist++ = fs_ip [j];
		}
		++arc_num;

		/* Advance to the next undirected edge. */
		ep1	+= n;
		fs_ip	+= n;
		i	+= n;
	}
	/* Terminate list of edges comprising each arc. */
	hsecp -> edge_lists [arc_num] = fslist;

	free ((char *) fs_num);
	free ((char *) fs_mst_edges);

	/* Record the source -> vertex arcs... */
	hsecp -> src_arc1 = arc_num;
	for (i = 0; i < nverts; i++) {
		*srcp++ = prob -> source;
		*dstp++ = i;
		++arc_num;
	}

	/* Record the vertex -> sink arcs... */
	hsecp -> dst_arc1 = arc_num;
	for (i = 0; i < nverts; i++) {
		*srcp++ = i;
		*dstp++ = prob -> sink;
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
	ip = outlist;
	for (i = 0; i < num_nodes; i++) {
		ptrs [i] = ip;
		prob -> out [i] = ip;
		ip += counts [i];
	}
	prob -> out [i] = ip;
	for (i = 0; i < num_arcs; i++) {
		j = prob -> arc_src [i];
		ip = ptrs [j]++;
		*ip = i;
	}

	/* Now do the incoming arc lists... */
	for (i = 0; i < num_nodes; i++) {
		counts [i] = 0;
	}
	for (i = 0; i < num_arcs; i++) {
		++(counts [prob -> arc_dst [i]]);
	}
	ip = inlist;
	for (i = 0; i < num_nodes; i++) {
		ptrs [i] = ip;
		prob -> in [i] = ip;
		ip += counts [i];
	}
	prob -> in [i] = ip;
	for (i = 0; i < num_arcs; i++) {
		k = prob -> arc_dst [i];
		ip = ptrs [k]++;
		*ip = i;
	}

	/* Free temporary memory used to build things... */
	free ((char *) counts);
	free ((char *) ptrs);

	/* Initialize the buffers used to hold flow solutions */
	/* and temporary data... */
	_gst_create_flow_solution_data (prob, &(hsecp -> soln));
	_gst_create_flow_temp_data (prob, &(hsecp -> temp));
}

/*
 * This routine computes an X,Y coordinate for each vertex in the
 * congested component.  This is tricky since there may be "vertices"
 * in the component that represent the MERGING of numerous "real"
 * vertices from the problem data.  We handle this by computing the
 * AVERAGE x,y coordinate of each real vertex in the component vertex.
 */

	static
	struct pset *
compute_vertex_coords (

struct comp *		comp,		/* IN - congested component */
struct gst_hypergraph *	cip		/* IN - compatibility info */
)
{
int			i;
int			j;
int			k;
int			nverts;
int *			ip1;
int *			ip2;
struct pset *		pts;
struct point *		p1;
coord_t			x;
coord_t			y;

	nverts = comp -> num_verts;

	pts = NEW_PSET (nverts);

	pts -> n = nverts;

	for (i = 0; i < nverts; i++) {
		/* Compute the "average" of the X,Y coordinates of each	*/
		/* "real" vertex embedded in this "fake" congested	*/
		/* component vertex.  We probably don't even really	*/
		/* care if these additions overflow -- we are simply	*/
		/* choosing SOME spanning tree for the FST...		*/
		x = 0;
		y = 0;
		k = 0;
		ip1 = comp -> rverts [i];
		ip2 = comp -> rverts [i + 1];
		while (ip1 < ip2) {
			j = *ip1++;
			p1 = &(cip -> pts -> a [j]);
			x += p1 -> x;
			y += p1 -> y;
			++k;
		}
		if (k > 0) {
			pts -> a [i].x = x / k;
			pts -> a [i].y = y / k;
		}
		else {
			pts -> a [i].x = 0;
			pts -> a [i].y = 0;
		}
	}

	return (pts);
}

/*
 * This routine computes a Minimum Spanning Tree for the given FST
 * from the given congested component.
 */

	static
	int
full_set_mst (

struct comp *		comp,		/* IN - congested component */
int			edge,		/* IN - edge number in component */
struct pset *		pts,		/* IN - X,Y coords of vertices */
struct edge *		elist		/* OUT - list of MST edges */
)
{
int			n;
int			t1;
int			t2;
int			nedges;
int			num_mst_edges;
int			num_mst_verts;
struct point *		p1;
struct point *		p2;
struct edge *		ep;
int *			ip1;
int *			ip2;
int *			endp;
struct edge *		edges;

	ip1  = comp -> everts [edge];
	endp = comp -> everts [edge + 1];

	n = endp - ip1;

	n = n * (n - 1) / 2;

	edges = NEWA (n, struct edge);

	/* Generate list of all edges, and compute the length	*/
	/* of each.  Note that we list the endpoints of each	*/
	/* edge in increasing order so that edges may be easily	*/
	/* compared and sorted...				*/
	nedges = 0;
	ep = &edges [0];
	num_mst_verts = endp - ip1;
	while (ip1 < endp) {
		t1 = *ip1++;
		p1 = &(pts -> a [t1]);
		ip2 = ip1;
		while (ip2 < endp) {
			t2 = *ip2++;
			p2 = &(pts -> a [t2]);
#if 1
			/* Rectilinear distance. */
			ep -> len = RDIST (p1, p2);
#else
			/* Euclidean distance. */
			ep -> len = EDIST (p1, p2);
#endif
			if (t1 < t2) {
				ep -> p1 = t1;
				ep -> p2 = t2;
			}
			else {
				ep -> p1 = t2;
				ep -> p2 = t1;
			}
			++ep;
			++nedges;
		}
	}

	if (nedges > 0) {
		num_mst_edges = _gst_mst_edge_list (num_mst_verts,
						    nedges,
						    edges,
						    elist);
	}
	else {
		num_mst_edges = 0;
	}

	free ((char *) edges);

	return (num_mst_edges);
}

/*
 * This routine sorts the given list of edges (and corresponding edge
 * numbers) on three keys -- first and second endpoint vertex numbers,
 * and then edge number.  Because p1 < p2 for every edge, this is a
 * well-defined ordering of edges for the undirected graph.
 */

	static
	void
sort_edges (

int			n,	/* IN - number of edges to sort. */
struct edge *		edges,	/* IN/OUT - edges to sort. */
int *			e_num	/* IN/OUT - edge number for each edge. */
)
{
int		h;
struct edge	etmp;
struct edge *	p1;
struct edge *	p2;
struct edge *	p3;
struct edge *	p4;
struct edge *	endp;
int		itmp;
int *		ip1;
int *		ip2;
int *		ip3;

	endp = &edges [n];

	for (h = 1; h <= n; h = 3*h+1) {
	}

	do {
		h = h / 3;
		p4 = &edges [h];
		p1 = p4;
		ip1 = &e_num [h];
		while (p1 < endp) {
			etmp = *p1;
			itmp = *ip1;
			p2 = p1;
			ip2 = ip1;
			while (TRUE) {
				p3 = (p2 - h);
				ip3 = (ip2 - h);
				if (p3 -> p1 < etmp.p1) break;
				if (p3 -> p1 EQ etmp.p1) {
					if (p3 -> p2 < etmp.p2) break;
					if (p3 -> p2 EQ etmp.p2) {
						if (*ip3 <= itmp) {
							break;
						}
					}
				}
				/* *p3 > *p2 */
				*p2 = *p3;
				p2 = p3;
				*ip2 = *ip3;
				ip2 = ip3;
				if (p2 < p4) break;
			}
			*p2 = etmp;
			*ip2 = itmp;
			++p1;
			++ip1;
		}
	} while (h > 1);
}

/*
 * This routine heuristically identifies violated Subtour Elimination
 * Constraints by finding max-flows (i.e. min-cuts) in the previously
 * constructed directed graph.  The given LP solution is used to set
 * the arc capacities before doing the max-flow.
 *
 * A list of violated SEC constraints is returned.  This may be NULL,
 * in which case no violations were detected.
 */

	static
	struct constraint *
heuristically_find_violated_secs (

struct comp *		comp,		/* IN - congested component */
struct bbinfo *		bbip,		/* IN - branch and bound info */
struct sec_heur_info *	hsecp,		/* IN - directed flow graph */
double *		x,		/* IN - LP solution to separate */
struct constraint *	clist		/* IN - existing constraints */
)
{
int			i;
int			j;
int			k;
int			n;
int			kmasks;
int			num_nodes;
int			src_arc1;
int			dst_arc1;
int *			ip1;
int *			ip2;
double *		c;
double			save;
bitmap_t *		stour;

	num_nodes	= hsecp -> prob.num_nodes;
	c		= hsecp -> prob.capacity;

	src_arc1	= hsecp -> src_arc1;
	dst_arc1	= hsecp -> dst_arc1;


	/* First we must set the arc capacities from the LP solution. */
	set_arc_capacities (comp, hsecp);

	/* Run one max-flow problem for each problem vertex... */

	n = num_nodes - 2;

	kmasks = bbip -> cip -> num_vert_masks;

	stour  = NEWA (kmasks, bitmap_t);

	for (i = 0; i < n; i++) {
		/* Modify problem to find worst SEC involving	*/
		/* vertex i.  Save capacity (for restore).	*/
		save = c [src_arc1 + i];
		c [src_arc1 + i] = INFINITE_FLOW;

		_gst_compute_max_flow (&(hsecp -> prob),
				       &(hsecp -> temp),
				       &(hsecp -> soln));

		/* Construct bitmask of "real" vertices... */
		for (j = 0; j < kmasks; j++) {
			stour [j] = 0;
		}
		for (j = 0; j < n; j++) {
			if (NOT BITON (hsecp -> soln.cut, j)) continue;
			ip1 = comp -> rverts [j];
			ip2 = comp -> rverts [j + 1];
			while (ip1 < ip2) {
				k = *ip1++;
				SETBIT (stour, k);
			}
		}

		/* Generate constraint if a violation is present. */
		clist = _gst_check_subtour (stour,
					    clist,
					    x,
					    bbip);

		/* Restore the modified capacity... */
		c [src_arc1 + i] = save;

		/* Force node i to the far side of the cut... */
		c [dst_arc1 + i] = INFINITE_FLOW;
	}

	free ((char *) stour);

	return (clist);
}

/*
 * This routine sets the all of the arc capacities from the LP solution.
 * There are 3 kinds of arcs to handle:  the "normal" arcs comprised of
 * flow taken from 1 or more edges, "source" arcs, and "sink" arcs.
 */

	static
	void
set_arc_capacities (

struct comp *		comp,		/* IN - congested component */
struct sec_heur_info *	hsecp		/* IN - directed flow graph */
)
{
int			i;
int			j;
int			num_nodes;
int			nverts;
int			src_arc1;
int			dst_arc1;
double *		c;
int *			ip1;
int *			ip2;
double			sum;
double *		x;

	num_nodes	= hsecp -> prob.num_nodes;
	nverts		= num_nodes - 2;
	c		= hsecp -> prob.capacity;

	src_arc1	= hsecp -> src_arc1;
	dst_arc1	= hsecp -> dst_arc1;

	x = comp -> x;

	/* All arcs before the first "source" arc are "normal" arcs. */
	for (i = 0; i < src_arc1; i++) {
		ip1 = hsecp -> edge_lists [i];
		ip2 = hsecp -> edge_lists [i + 1];
		sum = 0.0;
		while (ip1 < ip2) {
			/* Total weight of all edges that contribute */
			/* to this arc... */
			j = *ip1++;
			sum += x [j];
		}
		c [i] = 0.5 * sum;
	}

	/* Now initialize all "source" and "sink" arcs... */
	for (i = 0; i < nverts; i++) {
		ip1 = comp -> vedges [i];
		ip2 = comp -> vedges [i + 1];
		sum = 0.0;
		while (ip1 < ip2) {
			j = *ip1++;
			sum += x [j];
		}

		if (sum > 2.0) {
			c [src_arc1 + i] = 0.5 * sum - 1.0;
			c [dst_arc1 + i] = 0.0;
		}
		else {
			c [src_arc1 + i] = 0.0;
			c [dst_arc1 + i] = 1.0 - 0.5 * sum;
		}
	}
}

/*
 * This routine frees up the memory allocated by the given SEC max-flow
 * formulation.
 */

	static
	void
free_heuristic_SEC_formulation (

struct sec_heur_info *	hsecp		/* IN - SEC flow formulation */
)
{
	/* Free up the buffers used to hold flow solutions */
	/* and temporary data... */
	_gst_free_flow_temp_data (&(hsecp -> temp));
	_gst_free_flow_solution_data (&(hsecp -> soln));

	/* Free up the problem formulation... */
	free ((char *) (hsecp -> prob.out [0]));
	free ((char *) (hsecp -> prob.in [0]));
	free ((char *) (hsecp -> edge_lists [0]));

	free ((char *) (hsecp -> prob.out));
	free ((char *) (hsecp -> prob.in));
	free ((char *) (hsecp -> edge_lists));
	free ((char *) (hsecp -> prob.arc_src));
	free ((char *) (hsecp -> prob.arc_dst));
	free ((char *) (hsecp -> prob.capacity));
}

/*
 * This routine locates any cycles formed by hyperedges hyperedges having
 * weight 1.0 in the solution -- in other words, solid integer cycles.
 *
 * Each time we identify a connected component of solid integral edges,
 * we then look for "almost integer" cycles -- integral except for a single
 * fractional edge.  We do this by checking each fractional edge for two
 * or more vertices in common with the integral connected component.
 */

	struct constraint *
_gst_find_integer_cycles (

double *		x,		/* IN - LP solution to check */
struct constraint *	cp,		/* IN - previous constraints */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			j;
int			nverts;
int			nedges;
int			kmasks;
int			nmasks;
struct gst_hypergraph *	cip;
bitmap_t *		integral_edges;
bitmap_t *		edges_left;
bitmap_t *		cc_edges;
bitmap_t *		emark;
int *			stack;
int			num_frac;
int			num_cycles;
int *			frac_edge_nums;
struct constraint *	cp1;
struct constraint *	cp2;
bool			cycles_found;
bitmap_t *		vmark;
bitmap_t *		stour;
bitmap_t *		cc_verts;

	cip = bbip -> cip;
	nverts = cip -> num_verts;
	nedges = cip -> num_edges;
	kmasks = cip -> num_vert_masks;
	nmasks = cip -> num_edge_masks;

	stack			= NEWA (2 * nverts, int);
	frac_edge_nums		= NEWA (nedges, int);
	integral_edges		= NEWA (nmasks, bitmap_t);
	edges_left		= NEWA (nmasks, bitmap_t);
	cc_edges		= NEWA (nmasks, bitmap_t);
	emark			= NEWA (nmasks, bitmap_t);
	vmark			= NEWA (kmasks, bitmap_t);
	stour			= NEWA (kmasks, bitmap_t);
	cc_verts		= NEWA (kmasks, bitmap_t);

	for (i = 0; i < nmasks; i++) {
		integral_edges [i] = 0;
	}

	num_frac = 0;
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (bbip -> edge_mask, i)) continue;
		if (x [i] + FUZZ >= 1.0) {
			/* Edge is present with weight 1.0 */
			SETBIT (integral_edges, i);
		}
		else if (x [i] > FUZZ) {
			/* Edge is fractionally present. */
			frac_edge_nums [num_frac++] = i;
		}
	}

	for (i = 0; i < nmasks; i++) {
		edges_left [i] = integral_edges [i];
		emark [i] = 0;
	}

	for (i = 0; i < kmasks; i++) {
		vmark [i] = 0;
	}

	num_cycles = 0;
	for (i = 0; i < nedges; i++) {
		/* Find an edge that has not yet been traversed. */
		if (NOT BITON (edges_left, i)) continue;

		/* Prepare a place to hold the vertices and edges	*/
		/* of this integral connected-component...		*/
		for (j = 0; j < kmasks; j++) {
			cc_verts [j] = 0;
		}
		for (j = 0; j < nmasks; j++) {
			cc_edges [j] = 0;
		}

		/* Walk the component containing edge i, looking	*/
		/* for cycles...					*/
		cp1 = NULL;
		cp2 = cwalk (i,
			     -1,
			     integral_edges,
			     edges_left,
			     emark,
			     vmark,
			     stour,
			     cc_verts,
			     cc_edges,
			     stack,
			     stack,
			     &num_cycles,
			     x,
			     cp1,
			     bbip);

#if 0
		_gst_print_mask (bbip -> params -> print_solve_trace,
				 " Int CC verts:", cc_verts, nverts);
		_gst_print_mask (bbip -> params -> print_solve_trace,
				 " Int CC edges:", cc_edges, nedges);
#endif

		/* Add any cycles we found onto the main list... */
		cycles_found = FALSE;
		while (cp2 NE NULL) {
			cp1 = cp2 -> next;
			cp2 -> next = cp;
			cp = cp2;
			cp2 = cp1;
			cycles_found = TRUE;
		}

		if (num_cycles >= CYCLE_LIMIT) break;
		if (cycles_found) continue;

		/* The current connected-component does not contain any	*/
		/* solid integer cycles.  But now that we know all of	*/
		/* the vertices of this CC we can check for any		*/
		/* fractional edges that have at least two vertices in	*/
		/* common with the CC.  These represent "almost		*/
		/* integer" cycles...  The cycle is unique since the	*/
		/* integer CC is acyclic.				*/
		cp2 = find_almost_integer_cycles (num_frac,
						  frac_edge_nums,
						  cc_verts,
						  cc_edges,
						  stour,
						  x,
						  bbip);

		/* Add any "almost integer" cycles onto main list...	*/
		while (cp2 NE NULL) {
			cp1 = cp2 -> next;
			cp2 -> next = cp;
			cp = cp2;
			cp2 = cp1;
		}
	}

	free ((char *) cc_verts);
	free ((char *) stour);
	free ((char *) vmark);
	free ((char *) emark);
	free ((char *) cc_edges);
	free ((char *) edges_left);
	free ((char *) integral_edges);
	free ((char *) frac_edge_nums);
	free ((char *) stack);

	return (cp);
}

/*
 * This routine recursively walks connected components of hyperedges
 * looking for cycles.  It generates a Subtour Elimination Constraint
 * for every cycle found.
 */

	static
	struct constraint *
cwalk (

int			e,		/* IN - edge number */
int			vertex,		/* IN - vertex by which we entered */
					/*	this hyperedge (or -1) */
bitmap_t *		edge_mask,	/* IN - set of valid hyperedges */
bitmap_t *		edges_left,	/* IN - unvisited hyperedges */
bitmap_t *		emark,		/* IN - edges on stack */
bitmap_t *		vmark,		/* IN - vertices on stack */
bitmap_t *		stour,		/* IN - temporary vertex mask */
bitmap_t *		cc_verts,	/* OUT - vertices of connected comp */
bitmap_t *		cc_edges,	/* OUT - edges of connected comp */
int *			stack,		/* IN - base of vertex stack */
int *			sp,		/* IN - top of vertex stack */
int *			num_cycles,	/* IN/OUT - number of cycles found */
double *		x,		/* IN - LP solution to separate */
struct constraint *	cp,		/* IN - list of constraints */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			j;
int			v;
int			e2;
int			kmasks;
struct gst_hypergraph *	cip;
struct constraint *	tmp;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			vp3;

	/* Stop nearly-infinite loops on highly cyclic stuff... */
	if (*num_cycles >= CYCLE_LIMIT) return (cp);

	cip = bbip -> cip;

	if (BITON (emark, e)) {
		/* We have visited this edge before (higher up on	*/
		/* the stack).  We also know it was not the immediately	*/
		/* preceeding edge.  Therefore we have a cycle.  Read	*/
		/* the intervening vertices off the stack...		*/
		kmasks = cip -> num_vert_masks;
		for (i = 0; i < kmasks; i++) {
			stour [i] = 0;
		}
		if (sp < &stack [4]) {
			/* Stack too short for this! */
			FATAL_ERROR;
		}
		for (;;) {
			if (sp < &stack [2]) {
				/* Cycle edge not found on stack! */
				FATAL_ERROR;
			}
			sp -= 2;
			SETBIT (stour, sp [1]);
			if (sp [0] EQ e) break;
		}
		/* We should see every cycle twice.  Check if we have	*/
		/* seen this one before...				*/
		tmp = check_unique_subtour (stour, kmasks, cp, x, bbip);
#if 0
		if (tmp NE cp) {
			++(*num_cycles);
		}
#else
		/* Count duplicates too!  We don't want to	*/
		/* spend much time doing this...		*/
		++(*num_cycles);
#endif
		return (tmp);
	}

	/* This edge does NOT currently appear on the	*/
	/* stack.  Push it onto the stack now...	*/
	sp [0] = e;
	SETBIT (emark, e);
	SETBIT (cc_edges, e);

	/* Mark this edge as being traversed AT LEAST once... */
	CLRBIT (edges_left, e);

	/* Now walk to all OTHER vertices in this edge -- all except	*/
	/* for the one (if any) by which we entered this edge.		*/
	vp1 = cip -> edge [e];
	vp2 = cip -> edge [e + 1];
	while (vp1 < vp2) {
		v = *vp1++;
		if (v EQ vertex) continue;

		if (BITON (vmark, v)) {
			/* We have visited this vertex before (higher	*/
			/* up on the stack).  We also know it was not	*/
			/* the immediately preceeding vertex.		*/
			/* Therefore we have a cycle.  Read it off the	*/
			/* stack...					*/
			kmasks = cip -> num_vert_masks;
			for (j = 0; j < kmasks; j++) {
				stour [j] = 0;
			}
			SETBIT (stour, v);
			vp3 = sp;
			for (;;) {
				if (vp3 <= stack) {
					/* cycle vertex not found on	*/
					/* stack!			*/
					FATAL_ERROR;
				}
				vp3 -= 2;
				j = vp3 [1];
				if (j EQ v) break;
				SETBIT (stour, j);
			}

			tmp = check_unique_subtour (stour, kmasks, cp, x, bbip);
#if 0
			if (tmp NE cp) {
				++(*num_cycles);
			}
#else
			/* Count duplicates too!  We don't want to	*/
			/* spend much time doing this...		*/
			++(*num_cycles);
#endif
			cp = tmp;
			continue;
		}

		/* This vertex does NOT currently appear on the	*/
		/* stack.  Push it onto the stack now...	*/
		sp [1] = v;
		SETBIT (vmark, v);

		/* This vertex is part of the connected component... */
		SETBIT (cc_verts, v);

		/* Traverse every valid edge going out of vertex v,	*/
		/* (except the one via which we entered vertex v)...	*/
		ep1 = cip -> term_trees [v];
		ep2 = cip -> term_trees [v + 1];
		while (ep1 < ep2) {
			e2 = *ep1++;
			if (NOT BITON (edge_mask, e2)) continue;
			if (e2 EQ e) continue;
			/* Recursively walk subtree looking for cycles... */
			cp = cwalk (e2,
				    v,
				    edge_mask,
				    edges_left,
				    emark,
				    vmark,
				    stour,
				    cc_verts,
				    cc_edges,
				    stack,
				    sp + 2,
				    num_cycles,
				    x,
				    cp,
				    bbip);
			if (*num_cycles >= CYCLE_LIMIT) break;
		}
		CLRBIT (vmark, v);	/* Taking vertex v off the stack */
		if (*num_cycles >= CYCLE_LIMIT) break;
	}

	CLRBIT (emark, e);

	return (cp);
}

/*
 * This routine looks for "almost integer" cycles.  We are given an
 * acyclic connected component consisting of integral edges.  We are
 * also given a list of all of the fractional edges.  Each fractional
 * edge that shares two or more vertices in common with the integral
 * connected component therefore represents an SEC violation.  Given that
 * they intersect at vertices A and B there is a unique path from A to B
 * in the connected component.  We use a brute force DFS to recover the
 * actual vertices of the cycle.
 */

	static
	struct constraint *
find_almost_integer_cycles (

int			num_frac,	/* IN - number of fractional edges */
int *			frac_enums,	/* IN - list of fractional edges */
bitmap_t *		cc_verts,	/* IN - vertices of connected comp */
bitmap_t *		cc_edges,	/* IN - edges of connected comp */
bitmap_t *		stour,		/* IN - scratch vertex map */
double *		x,		/* IN - LP solution to separate */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			k;
int			kmasks;
int			e;
int			t1;
int			t2;
int *			vp1;
int *			vp2;
int *			vp3;
struct gst_hypergraph *	cip;
struct constraint *	cp;
bool			found;

	cip	= bbip -> cip;
	kmasks	= cip -> num_vert_masks;

	cp = NULL;

	while (num_frac > 0) {
		e = *frac_enums++;
		--num_frac;

		/* Find each vertex pair (if any) in the CC... */
		vp1 = cip -> edge [e];
		vp3 = cip -> edge [e + 1];
		while (vp1 < vp3) {
			t1 = *vp1++;
			if (NOT BITON (cc_verts, t1)) continue;
			vp2 = vp1;
			while (vp2 < vp3) {
				t2 = *vp2++;
				if (NOT BITON (cc_verts, t2)) continue;

				/* We have a new cycle!  Now identify it... */
				for (k = 0; k < kmasks; k++) {
					stour [k] = 0;
				}
				found = fwalk (t1,
					       t2,
					       cc_edges,
					       stour,
					       cip);
				FATAL_ERROR_IF (NOT found);

				/* Could have seen this same cycle before... */
				cp = check_unique_subtour (stour, kmasks, cp, x, bbip);
			}
		}
	}

	return (cp);
}

/*
 * This routine recursively finds a path from vertex t1 to vertex
 * t2 in the given (acyclic) connected component.  It returns TRUE
 * when the path has been found.
 */

	static
	bool
fwalk (

int			t1,		/* IN - first vertex */
int			t2,		/* IN - last vertex */
bitmap_t *		cc_edges,	/* IN - edges of connected comp */
bitmap_t *		path,		/* OUT - vertices along path */
struct gst_hypergraph *	cip		/* IN - compatibility info */
)
{
int			e;
int			t3;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
bool			found;

	if (BITON (path, t1)) {
		/* We've been here before... */
		return (FALSE);
	}

	SETBIT (path, t1);

	if (t1 EQ t2) return (TRUE);

	ep1 = cip -> term_trees [t1];
	ep2 = cip -> term_trees [t1 + 1];
	while (ep1 < ep2) {
		e = *ep1++;
		if (NOT BITON (cc_edges, e)) continue;

		CLRBIT (cc_edges, e);
		found = FALSE;
		vp1 = cip -> edge [e];
		vp2 = cip -> edge [e + 1];
		while (vp1 < vp2) {
			t3 = *vp1++;
			found = fwalk (t3, t2, cc_edges, path, cip);
			if (found) break;
		}
		SETBIT (cc_edges, e);
		if (found) return (TRUE);
	}

	CLRBIT (path, t1);

	return (FALSE);
}

/*
 * This routine checks the given subtour to see if it is both violated
 * and unique.  If so a new constraint is added to the given list.
 * Otherwise, the given constraint list is returned unchanged.
 */

	static
	struct constraint *
check_unique_subtour (

bitmap_t *		stour,		/* IN - subtour to check */
int			kmasks,		/* IN - number of vertex masks */
struct constraint *	cp,		/* IN - current constraint list */
double *		x,		/* IN - LP solution to separate */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
bool			violation;
struct cpool *		pool;
struct constraint *	p;
bitmap_t *		bp1;
bitmap_t *		bp2;
bitmap_t *		bp3;

	for (p = cp; p NE NULL; p = p -> next) {
		if (_gst_is_equal (stour, p -> mask, kmasks)) {
			/* Already seen this subtour... */
			return (cp);
		}
	}

	/* Not seen before -- record it. */
	p = NEW (struct constraint);
	bp1 = NEWA (kmasks, bitmap_t);

	p -> next	= NULL;
	p -> iteration	= 0;
	p -> type	= CT_SUBTOUR;
	p -> mask	= bp1;
	bp2 = stour;
	bp3 = bp2 + kmasks;
	while (bp2 < bp3) {
		*bp1++ = *bp2++;
	}

	/* See if it is violated. */
	pool	= bbip -> cpool;
	_gst_expand_constraint (p, pool -> cbuf, bbip);
	violation = _gst_is_violation (pool -> cbuf, x);
	if (NOT violation) {
		/* Constraint is not violated.  Ignore it. */
		free (p -> mask);
		free (p);
		return (cp);
	}

	p -> next	= cp;

	return (p);
}

/*
 * This routine returns TRUE if-and-only-if the two bit-masks
 * are identical.
 */

	bool
_gst_is_equal (

bitmap_t *	bp1,		/* IN - first set. */
bitmap_t *	bp2,		/* IN - second set. */
int		nmasks		/* IN - number of masks in set. */
)
{
int		i;

	for (i = 0; i < nmasks; i++) {
		if (*bp1 NE *bp2) return (FALSE);
		++bp1;
		++bp2;
	}

	return (TRUE);
}

/*
 * This routine finds violated SEC's by exhaustively enumerating all
 * subsets of size 2 or more.  We return a list of the constraints that
 * were found.
 */

#ifdef	OLD_ENUMERATION

	struct constraint *
_gst_enumerate_all_subtours (

struct comp *		comp,		/* IN - component to check. */
struct constraint *	cp,		/* IN - existing list of constraints */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			k;
int			nverts;
double			rhs;
bitmap_t *		chosen;
int *			clist;

	nverts = comp -> num_verts;

	k = BMAP_ELTS (nverts);

	clist  = NEWA (nverts, int);
	chosen = NEWA (k, bitmap_t);

	for (i = 0; i < k; i++) {
		chosen [i] = 0;
	}

#if 0
	gst_channel_printf (bbip -> params -> print_solve_trace,
		" Exhaustively enumerating %d congested vertices.\n",
		nverts);
#endif


	for (i = 2; i <= nverts; i++) {
		/* Look for subtours of size "i". */
		rhs = ((double) (i - 1)) + FUZZ;
		cp = enumerate_subtours (i,
					 0,
					 rhs,
					 chosen,
					 &clist [0],
					 &clist [0],
					 comp,
					 cp,
					 bbip);
	}

	free ((char *) chosen);
	free ((char *) clist);

	return (cp);
}

/*
 * This routine finds violated SEC's by explicitly enumerating subsets
 * of increasing cardinality.  Only the vertices of the given congested
 * component are examined.  We return a list of the constraints that
 * were found.
 */

	struct constraint *
_gst_find_small_subtours (

struct comp *		comp,		/* IN - component to check. */
struct constraint *	cp,		/* IN - existing list of constraints */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			k;
int			nverts;
int			klimit;
int			senum_limit;
cpu_time_t		ticks_limit;
cpu_time_t		t0;
cpu_time_t		t1;
bitmap_t *		chosen;
int *			clist;
double			est;
double			rhs;
char			tbuf [32];

	nverts = comp -> num_verts;

	k = BMAP_ELTS (nverts);

	clist  = NEWA (nverts, int);
	chosen = NEWA (k, bitmap_t);

	for (i = 0; i < k; i++) {
		chosen [i] = 0;
	}

	gst_channel_printf (bbip -> params -> print_solve_trace,
		"Enumerating %d congested vertices.\n", nverts);

	t0 = _gst_get_cpu_time ();

	/* Determine cardinality limit for partial enumeration. */

	senum_limit = bbip -> params -> sec_enum_limit;
	if (nverts <= senum_limit) {
		klimit = nverts;
	}
	else if (nverts <= 2 * senum_limit) {
		klimit = 5;
	}
	else if (nverts <= 3 * senum_limit) {
		klimit = 3;
	}
	else {
		klimit = 2;
	}

	/* Establish time limit for each cardinality. */
	ticks_limit = _gst_int_seconds_to_cpu_time_t (nverts);

	/* Don't need to check 2-vertex subtours if they are	*/
	/* already present in the constraint pool!		*/
	i = bbip -> params -> seed_pool_with_2secs ? 3 : 2;

	for (; i <= klimit; i++) {
#if 0
		gst_channel_printf (bbip -> params -> print_solve_trace,
			" Checking %d subtours\n", i);
#endif

		/* Look for subtours of size "i". */
		rhs = ((double) (i - 1)) + FUZZ;
		cp = enumerate_subtours (i,
					 0,
					 rhs,
					 chosen,
					 &clist [0],
					 &clist [0],
					 comp,
					 cp,
					 bbip);

		/* Determine how long this took... */
		t1 = _gst_get_cpu_time ();
#if 0
		_gst_convert_cpu_time (t1 - t0, tbuf);
		gst_channel_printf (bbip -> params -> print_solve_trace,
			"	%s seconds\n", tbuf);
#endif

		/* if we found any violations, get out! */
		if (cp NE NULL) break;

		/* Estimate how long the next pass will take... */
		est = ((double) (nverts - i)) / ((double) (i + 1));
		est *= (t1 - t0);
		if (est > ticks_limit) {
			/* Do not take more than specified	*/
			/* number of ticks...			*/
			break;
		}
		t0 = t1;
	}

	free ((char *) chosen);
	free ((char *) clist);

	return (cp);
}

/*
 * This routine recursively enumerates subsets of size k of the
 * congested vertices.
 */

	static
	struct constraint *
enumerate_subtours (

int			k,		/* IN - size of subset to choose */
int			t,		/* IN - next vertex to take/omit */
double			rhs,		/* IN - required right-hand-side */
bitmap_t *		chosen,		/* IN - component vertices taken */
int *			start,		/* IN - start of chosen vertices */
int *			end,		/* IN - end of chosen vertices */
struct comp *		comp,		/* IN - component to enumerate */
struct constraint *	cp,		/* IN - existing constraints */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			j;
int			nverts;
int			nedges;
int			kmasks;
int			limit;
int *			ip1;
int *			ip2;
int *			ip3;
bitmap_t *		bp1;
bitmap_t *		bp2;
struct constraint *	newp;
double			sum;

	nverts = comp -> num_verts;

	if (k <= 0) {
		/* We have chosen a complete subset of the desired	*/
		/* cardinality.  Check it for violations...		*/
		nedges = comp -> num_edges;
		sum = 0.0;
		for (i = 0; i < nedges; i++) {
			ip1 = comp -> everts [i];
			ip2 = comp -> everts [i + 1];
			k = -1;
			while (ip1 < ip2) {
				t = *ip1++;
				if (BITON (chosen, t)) {
					++k;
				}
			}
			if (k > 0) {
				sum += (k * (comp -> x [i]));
			}
		}
		ip1 = start;
		while (ip1 < end) {
			t = *ip1++;
			sum += comp -> tviol [t];
		}

		if (sum > rhs) {
			/* We have a violation!  Assemble a mask of the	*/
			/* actual vertices in the subtour from the	*/
			/* chosen vertices of the congested component.	*/
			kmasks = bbip -> cip -> num_vert_masks;
			bp1 = NEWA (kmasks, bitmap_t);

			for (i = 0; i < kmasks; i++) {
				bp1 [i] = 0;
			}
			ip1 = start;
			while (ip1 < end) {
				i = *ip1++;
				ip2 = comp -> rverts [i];
				ip3 = comp -> rverts [i + 1];
				while (ip2 < ip3) {
					j = *ip2++;
					SETBIT (bp1, j);
				}
			}

			newp = NEW (struct constraint);
			newp -> next		= cp;
			newp -> iteration	= 0;
			newp -> type		= CT_SUBTOUR;
			newp -> mask		= bp1;
			cp = newp;
#if 0
			_gst_print_mask (bbip -> params -> print_solve_trace,
					 " subtour:", bp1, kmasks * BPW);
#endif
		}

		return (cp);
	}

	limit = nverts - k;

	while (t <= limit) {
		/* Try taking the current vertex... */
		SETBIT (chosen, t);
		*end = t;
		cp = enumerate_subtours (k - 1,
					 t + 1,
					 rhs,
					 chosen,
					 start,
					 end + 1,
					 comp,
					 cp,
					 bbip);
		CLRBIT (chosen, t);
		++t;
	}

	return (cp);
}

/*
 * This routine finds violated SEC's by exhaustively enumerating all
 * subsets of size 2 or more.  We return a list of the constraints that
 * were found.
 */

#else

	struct constraint *
_gst_enumerate_all_subtours (

struct comp *		comp,		/* IN - component to check. */
struct constraint *	cp,		/* IN - existing list of constraints */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			nverts;
int *			avail;
int *			chosen;
int *			excluded;
int *			vstat;

	nverts = comp -> num_verts;

#if 0
	gst_channel_printf (bbip -> params -> print_solve_trace,
		" Exhaustively enumerating %d congested vertices.\n",
		nverts);
#endif

	avail	 = NEWA (4 * nverts, int);
	chosen	 = avail + nverts;
	excluded = chosen + nverts;
	vstat	 = excluded + nverts;

	/* Each vertex is initially free... */
	for (i = 0; i < nverts; i++) {
		vstat [i] = -1;
	}

	for (i = 0; i < nverts; i++) {

		chosen [0]	= i;
		vstat [i]	= 1;

		cp = recurse_enum (nverts,	/* limit */
				   0,		/* navail */
				   avail,
				   1,		/* nchosen */
				   chosen,
				   i,		/* nexcl */
				   excluded,
				   vstat,
				   0.0,		/* LHS */
				   FUZZ,	/* RHS */
				   TRUE,	/* do_supersets */
				   comp,
				   cp,
				   bbip);
		excluded [i] = i;
		vstat [i] = 2;
	}

	free ((char *) avail);

	return (cp);
}

/*
 * This routine finds violated SEC's by explicitly enumerating subsets
 * of increasing cardinality.  Only the vertices of the given congested
 * component are examined.  We return a list of the constraints that
 * were found.
 */

	struct constraint *
_gst_find_small_subtours (

struct comp *		comp,		/* IN - component to check. */
struct constraint *	cp,		/* IN - existing list of constraints */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			k;
int			nverts;
int			klimit;
int			senum_limit;
cpu_time_t		ticks_limit;
cpu_time_t		t0;
cpu_time_t		t1;
int *			avail;
int *			chosen;
int *			excluded;
int *			vstat;
double			est;

	nverts = comp -> num_verts;

	avail	 = NEWA (4 * nverts, int);
	chosen	 = avail + nverts;
	excluded = chosen + nverts;
	vstat	 = excluded + nverts;

	gst_channel_printf (bbip -> params -> print_solve_trace,
		"Enumerating %d congested vertices.\n", nverts);

	t0 = _gst_get_cpu_time ();

	/* Determine cardinality limit for partial enumeration. */

	senum_limit = bbip -> params -> sec_enum_limit;
	if (nverts <= senum_limit) {
		klimit = nverts;
	}
	else if (nverts <= 2 * senum_limit) {
		klimit = 5;
	}
	else if (nverts <= 3 * senum_limit) {
		klimit = 3;
	}
	else {
		klimit = 2;
	}

	/* Establish time limit for each cardinality. */
	ticks_limit = _gst_int_seconds_to_cpu_time_t (nverts);

	/* Don't need to check 2-vertex subtours if they are	*/
	/* already present in the constraint pool!		*/
	k = bbip -> params -> seed_pool_with_2secs ? 3 : 2;

	for (; k <= klimit; k++) {
#if 0
		gst_channel_printf (bbip -> params -> print_solve_trace,
			" Checking %d subtours\n", k);
#endif

		/* Each vertex is initially free... */
		for (i = 0; i < nverts; i++) {
			vstat [i] = -1;
		}

		for (i = 0; i < nverts; i++) {

			chosen [0]	= i;
			vstat [i]	= 1;

			cp = recurse_enum (k,		/* limit */
					   0,		/* navail */
					   avail,
					   1,		/* nchosen */
					   chosen,
					   i,		/* nexcl */
					   excluded,
					   vstat,
					   0.0,		/* LHS */
					   FUZZ,	/* RHS */
					   FALSE,	/* do_supersets */
					   comp,
					   cp,
					   bbip);
			excluded [i] = i;
			vstat [i] = 2;
		}

		/* Determine how long this took... */
		t1 = _gst_get_cpu_time ();
#if 0
		{ char	tbuf [64];
		_gst_convert_cpu_time (t1 - t0, tbuf);
		gst_channel_printf (bbip -> params -> print_solve_trace,
				    "	%s seconds\n", tbuf);
		}
#endif

		/* if we found any violations, get out! */
		if (cp NE NULL) break;

		/* Estimate how long the next pass will take... */
		est = ((double) (nverts - k)) / ((double) (k + 1));
		est *= (t1 - t0);
		if (est > ticks_limit) {
			/* Do not take more than specified	*/
			/* number of ticks...			*/
			break;
		}
		t0 = t1;
	}

	free ((char *) avail);

	return (cp);
}

/*
 * This routine recursively enumerates connected subsets of size k of the
 * congested vertices.
 */

	static
	struct constraint *
recurse_enum (

int			limit,		/* IN - max num verts to choose */
int			navail,		/* IN - num verts available */
int *			avail,		/* IN - unchosen verts available */
int			nchosen,	/* IN - num verts chosen */
int *			chosen,		/* IN - vertices chosen */
int			nexcl,		/* IN - num vertices excluded */
int *			excluded,	/* IN - verts excluded from choice */
int *			vstat,		/* IN - status of each vertex: */
					/* -1=free, 0=avail, 1=chosen, */
					/* 2=excluded. */
double			lhs,		/* IN - LHS of constraint */
double			rhs,		/* IN - RHS of constraint */
bool			do_supersets,	/* IN - do supersets of violations? */
struct comp *		comp,		/* IN - component to enumerate */
struct constraint *	cp,		/* IN - existing constraints */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			j;
int			k;
int			t;
int			e;
int			kmasks;
int			navail_save;
int			new_nexcl;
int *			vp1;
int *			vp2;
int *			ep1;
int *			ep2;
bitmap_t *		bp1;
struct constraint *	newp;
double			sum;

	/* Check if chosen vertices yield a violation: */
	if (lhs > rhs) {
		kmasks = bbip -> cip -> num_vert_masks;
		bp1 = NEWA (kmasks, bitmap_t);

		for (i = 0; i < kmasks; i++) {
			bp1 [i] = 0;
		}
		for (i = 0; i < nchosen; i++) {
			j = chosen [i];
			vp1 = comp -> rverts [j];
			vp2 = comp -> rverts [j + 1];
			while (vp1 < vp2) {
				k = *vp1++;
				SETBIT (bp1, k);
			}
		}

		newp = NEW (struct constraint);
		newp -> next		= cp;
		newp -> iteration	= 0;
		newp -> type		= CT_SUBTOUR;
		newp -> mask		= bp1;

		cp = newp;
#if 0
		_gst_print_mask (bbip -> params -> print_solve_trace,
				 " subtour:", bp1, bbip -> cip -> num_verts);
#endif

		if (NOT do_supersets) {
			/* Don't enumerate supersets of any violation... */
			return (cp);
		}
	}

	if (nchosen >= limit) {
		/* Don't recurse any deeper. */
		return (cp);
	}

	navail_save = navail;

	/* Find newly available vertices.  All free vertices adjacent */
	/* to the most-recently-chosen vertex now become available. */
	t = chosen [nchosen - 1];
	ep1 = comp -> vedges [t];
	ep2 = comp -> vedges [t + 1];
	while (ep1 < ep2) {
		e = *ep1++;
		vp1 = comp -> everts [e];
		vp2 = comp -> everts [e + 1];
		while (vp1 < vp2) {
			i = *vp1++;
			if (vstat [i] < 0) {
				avail [navail++] = i;
				vstat [i] = 0;
			}
		}
	}

	/* Choose each available vertex.  After recursing on	*/
	/* this choice, exclude it from future consideration.	*/

	new_nexcl = nexcl;
	while (navail > 0) {
		/* Choose (pop) last available vertex t. */
		t = avail [--navail];

		/* Compute weight of edges connecting vertex t to all	*/
		/* previously chosen vertices.  This is used to update	*/
		/* the LHS of the constraint.				*/
		sum = 0.0;
		ep1 = comp -> vedges [t];
		ep2 = comp -> vedges [t + 1];
		while (ep1 < ep2) {
			e = *ep1++;
			vp1 = comp -> everts [e];
			vp2 = comp -> everts [e + 1];
			while (vp1 < vp2) {
				j = *vp1++;
				if (vstat [j] EQ 1) {
					sum += comp -> x [e];
					break;
				}
			}
		}

		/* Vertex t is now CHOSEN. */
		chosen [nchosen]	= t;
		vstat [t]		= 1;

		cp = recurse_enum (limit,
				   navail,
				   avail,
				   nchosen + 1,
				   chosen,
				   new_nexcl,
				   excluded,
				   vstat,
				   lhs + sum,
				   rhs + 1.0,
				   do_supersets,
				   comp,
				   cp,
				   bbip);

		/* Now exclude vertex t from further consideration. */
		vstat [t]		= 2;
		excluded [new_nexcl++]	= t;
	}

	/* Restore those vertices that we excluded back to	*/
	/* their available state.				*/
	while (new_nexcl > nexcl) {
		t = excluded [--new_nexcl];
		vstat [t] = 0;
		avail [navail++] = t;
	}

	/* Restore those vertices that we made available back	*/
	/* to their original "free" state.			*/
	while (navail > navail_save) {
		t = avail [--navail];
		vstat [t] = -1;
	}

	return (cp);
}

#endif

/*
 * This routine checks the given subset of vertices to see if it defines
 * a violated subtour constraint.  If so it adds the constraint to the
 * current list of constraints.
 */

	struct constraint *
_gst_check_subtour (

bitmap_t *		stour,		/* IN - subset to check */
struct constraint *	clist,		/* IN - existing constraints */
double *		x,		/* IN - current LP solution */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			j;
int			kmasks;
int			nedges;
int			ssize;
int			count;
struct gst_hypergraph *	cip;
bitmap_t *		edge_mask;
int *			vp1;
int *			vp2;
bitmap_t *		bp1;
bitmap_t		mask;
double			total;
double			coeff;
struct constraint *	cp;

	cip		= bbip -> cip;
	edge_mask	= bbip -> edge_mask;
	nedges		= cip -> num_edges;
	kmasks		= cip -> num_vert_masks;

	/* Get size of subtour - 1... */
	ssize = -1;
	bp1 = stour;
	for (i = 0; i < kmasks; i++) {
		mask = *bp1++;
		ssize += NBITSON (mask);
	}

	if (ssize <= 0) return (clist);

	total = 0.0;
	for (j = 0; j < nedges; j++) {
		if (NOT BITON (edge_mask, j)) continue;
		count = -1;
		vp1 = cip -> edge [j];
		vp2 = cip -> edge [j + 1];
		while (vp1 < vp2) {
			i = *vp1++;
			if (BITON (stour, i)) {
				++count;
			}
		}
		if (count <= 0) continue;
		coeff = ((double) count);
		total += coeff * x [j];
	}

	if (total <= ((double) ssize) + FUZZ) {
		/* Constraint is not violated... */
		return (clist);
	}

	/* Create new constraint and add it to the list... */

	bp1 = NEWA (kmasks, bitmap_t);
	for (i = 0; i < kmasks; i++) {
		bp1 [i] = stour [i];
	}

	cp = NEW (struct constraint);
	cp -> next	= clist;
	cp -> iteration	= 0;
	cp -> type	= CT_SUBTOUR;
	cp -> mask	= bp1;

#if 0
	_gst_print_mask (bbip -> params -> print_solve_trace,
			 " subtour:", stour, cip -> num_verts);
#endif
#if 0
	_gst_debug_print_constraint (" % Subtour:  ",
				     " %\t",
				     cp,
				     x,
				     bbip);
#endif

	return (cp);
}
