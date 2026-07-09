/***********************************************************************

	$Id: sec_comp.c,v 1.18 2022/11/19 13:45:54 warme Exp $

	File:	sec_comp.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines to decompose SEC separation problems into smaller
	and simpler sub-problems.

************************************************************************

	Modification Log:

	a-1:	10/05/96	warme
		: Created.
	b-1:	02/28/2001	warme
		: Changes for 3.1 release.
		: Rename many fields in "struct comp".
		: Fix incorrect BCC routine.
		: Use consistent terminology: prefer "vertex" to
		:  "terminal" and "edge" to "full set".
		: Use rverts instead of tmasks.
		: Completely re-write of merge_chains routine.
		: Fix incomplete reduction bug.
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
		: Fix -Wall issues.  Upgrade fatals.
	e-4:	11/18/2022	warme
		: Implement new parameters to turn strengthening on/off,
		:  and whether to generate the un-strengthened subtour
		:  or not.
		: Fixed a nasty bug that threw all of the strengthened
		:  subtours away.  Code to actually emit the
		:  strengthened subtours was intermingled with code to
		:  plot significant reductions, which was disabled.
		:  Teased these two functions apart.
		: Switch to more efficient reductions in
		:  _gst_check_component_subtour().
		: Simplify calling convention of various functions.
		: Add _gst_free_congested_component_list(), and use
		:  it here.
		: Rename create_masks() to _gst_create_comp_masks().
		: Split off code into plot_significant_reductions().
		: Fix -Wall issues.

************************************************************************/

#include "sec_comp.h"

#include "bb.h"
#include "constrnt.h"
#include "dsuf.h"
#include "fatal.h"
#include "genps.h"
#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include "parmblk.h"
#include "sec_heur.h"
#include "steiner.h"
#include <string.h>
#include "utils.h"


/*
 * Global Routines
 */

struct constraint *	_gst_check_component_subtour (
					bitmap_t *		S,
					struct comp *		comp,
					struct constraint *	cp,
					double *		x,
					struct bbinfo *		bbip);
void		_gst_create_comp_masks (struct comp *);
struct comp *	_gst_delete_vertex_from_component (int			t,
						   struct comp *	comp,
						   struct bbinfo *	bbip);
struct comp *	_gst_find_congested_components (double *	x,
						bitmap_t *	vert_mask,
						bitmap_t *	edge_mask,
						bool		print_flag,
						struct bbinfo *	bbip);
int		_gst_find_least_congested_vertex (bitmap_t *	S,
						  struct comp *	comp);
void		_gst_free_congested_component (struct comp *	p);


/*
 * External References
 */

	/* None */

/*
 * Local Types
 */

struct bc {
	struct comp *	comp;		/* component being split */
	struct comp *	list;		/* output list of BCC's */
	int *		dfs;		/* DFS number of each vertex */
	int *		low;		/* lowest DFS num in component */
	int *		parent;		/* parents of vertices in DFS tree */
	int *		stack;		/* base-address of edge stack */
	int *		sp;		/* current stack pointer */
	int		counter;	/* DFS number generator */
	bitmap_t *	vert_mask;	/* scratch buffer for new components */
	bitmap_t *	edge_mask;	/* scratch buffer for new components */
	bitmap_t *	edges_seen;	/* edges already pushed */
	int		nvmasks;	/* size of vert_mask */
	int		nemasks;	/* size of edge_mask */
	int		ncomps;		/* number of BCC's split off */
	int		max_stack;	/* size of stack */
	struct bbinfo *	bbip;		/* branch and bound info */
};


/*
 * Local Routines
 */

static void		bcc (struct bc *, int);
static void		component_verts_to_real_verts (bitmap_t *,
						       struct comp *,
						       bitmap_t *,
						       int);
static struct comp *	copy_component_induced_by_vertices (
						struct comp *	comp,
						bitmap_t *	S);
static struct comp *	copy_subcomponent (struct comp *,
					   bitmap_t *,
					   bitmap_t *);
static struct comp *	find_first_component (double *,
					      bitmap_t *,
					      bitmap_t *,
					      struct gst_hypergraph *);
static struct comp *	greedily_improve_one_component (struct comp *,
							struct bbinfo *);
static struct comp *	greedy_improvement (struct comp *, struct bbinfo *);
static void		merge_chains (struct comp *, struct bbinfo *);
static void		plot_significant_reductions (
					int			ncomps,
					struct comp **		comps,
					struct constraint **	constraints,
					double *		x,
					bitmap_t *		orig_stour,
					struct bbinfo *		bbip);
static void		reduce_component_in_place (struct comp *);
static struct comp *	simplify_one_component (struct comp *,
						struct bbinfo *);
static struct comp *	split_biconnected_components (struct comp *,
						      struct bbinfo *);
static void		split_connected_components (struct comp *);
static void		strip_uncongested_vertices (struct comp *);
static int		vacuous_component (struct comp *);

/*
 * Compute the smallest possible subsets of the problem on which to
 * run the SEC separation routines.  We use the following principles
 * to prune down the vertices and edges examined:
 *
 *	- Let S be a set of vertices that define a violated SEC.  Let
 *	  t be a vertex in S with delta(t) <= 1.  Then there is a
 *	  subset S' of S that does not contain t, yet still defines a
 *	  violated SEC.  Therefore, all such vertices t can be
 *	  iteratively removed from consideration.  (Such vertices are
 *	  NOT "congested".)
 *
 *	- Let the hypergraph of vertices and edges be partitioned
 *	  into its connected components C1 = (V1,F1), C2 = (V2,F2), ...,
 *	  Ck = (Vk,Fk).  Let S be any violated SEC.  Then there is at
 *	  least one i in 1..k for which Si (a subset of
 *	  [Vi \intersect S]) is a violated SEC.
 *
 *	- [Same, but with C1, C2, ..., Ck being biconnected components.]
 *
 *	- Let F1 be an edge containing exactly two congested vertices
 *	  s and t.  Let S be an SEC violation containing s and t.  Define
 *	  a new hypergraph C' = (V - {s,t} + {u}, F') where u is a new
 *	  vertex not in V, and F' is derived from F by replacing every
 *	  reference to s or t with u.  [Resulting edges with fewer than
 *	  2 distinct vertices are deleted from F'.]  Let S be a
 *	  violation in C.  Then S' = S - {s,t} + ({u} if s in S or t in S)
 *	  defines a violation in C'.
 *
 * These rules may be applied recursively to arrive at a final set of
 * congested components that are each usually quit small.
 */

	struct comp *
_gst_find_congested_components (

double *		x,		/* IN - current LP solution */
bitmap_t *		vert_mask,	/* IN - set of valid vertices */
bitmap_t *		edge_mask,	/* IN - set of valid hyperedges */
bool			print_flag,	/* IN - TRUE ==> print info */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			n;
struct gst_hypergraph *	cip;
struct comp *		comp;
struct comp *		p;
gst_channel_ptr		trace;

	cip	  = bbip -> cip;

	/* Find the first component by iteratively applying the	*/
	/* delta(t) <= 1 rule...				*/

	comp = find_first_component (x, vert_mask, edge_mask, cip);

	if (comp EQ NULL) return (NULL);

	trace = bbip -> params -> print_solve_trace;

	if (print_flag AND (trace NE NULL)) {
		gst_channel_printf (trace,
			"initially %d congested vertices:\n",
			comp -> num_verts);
#if 0
		int i, j, k;
		k = 0;
		for (i = 0; i < comp -> num_verts; i++) {
			int * vp1 = comp -> rverts [i];
			int * vp2 = comp -> rverts [i + 1];
			while (vp1 < vp2) {
				j = *vp1++;
				if (k EQ 0) {
					gst_channel_printf (trace, "	");
				}
				gst_channel_printf (trace, " %3d", j);
				++k;
				if (k >= 16) {
					gst_channel_printf (trace, "\n");
					k = 0;
				}
			}
		}
		if (k > 0) {
			gst_channel_printf (trace, "\n");
		}
#endif
	}

	comp = simplify_one_component (comp, bbip);

	if (print_flag AND (trace NE NULL)) {

		n = 0;
		for (p = comp; p NE NULL; p = p -> next) {
			++n;
		}
		gst_channel_printf (trace,
			"_gst_find_congested_components found %d components:\n",
			n);
		n = 0;
		for (p = comp; p NE NULL; p = p -> next) {
			gst_channel_printf (trace,
				"\tcomponent %d:\t%d verts,\t%d edges\n",
				n++, p -> num_verts, p -> num_edges);
#if 0
			gst_channel_printf (trace, "\t   ");
			for (i = 0; i < p -> num_verts; i++) {
				int * vp1 = p -> rverts [i];
				int * vp2 = p -> rverts [i + 1];
				int k = (vp2 - vp1);
				if (k NE 1) {
					gst_channel_printf (trace, " {");
				}
				while (vp1 < vp2) {
					j = *vp1++;
					gst_channel_printf (trace, " %d", j);
				}
				if (k NE 1) {
					gst_channel_printf (trace, "}");
				}
			}
			gst_channel_printf (trace, "\n");
#endif
		}
	}

	return (comp);
}

/*
 * This routine finds the initial component by iteratively applying
 * the delta(t) <= 1 rule.  This normally shrinks the problem down
 * substantially.  We then convert the problem into "struct comp"
 * form -- after which we no longer need to refer to the gst_hypergraph
 * stuff.
 */

	static
	struct comp *
find_first_component (

double *		x,		/* IN - current LP solution */
bitmap_t *		vert_mask,	/* IN - set of valid vertices */
bitmap_t *		edge_mask,	/* IN - set of valid hyperedges */
struct gst_hypergraph *	cip		/* IN - compatibility info */
)
{
int			i;
int			j;
int			k;
int			e;
int			t;
int			nedges;
int			nverts;
int			kmasks;
int			ecount;
int			vcount;
int *			vp1;
int *			vp2;
int *			vp3;
int *			ep1;
int *			ep2;
int *			ep3;
double			sum;
int *			vleft;
int *			sp;
int *			new_vnum;
int *			new_enum;
int *			old_enum;
int *			ip1;
struct comp *		newp;
bitmap_t *		bp1;
bitmap_t *		bp2;
bitmap_t *		bp3;
bitmap_t *		bp4;
bitmap_t *		verts_stacked;
bitmap_t *		cvmask;
int *			stack;
double *		B;

	nedges = cip -> num_edges;
	nverts = cip -> num_verts;
	kmasks = cip -> num_vert_masks;

	verts_stacked = NEWA (2 * kmasks, bitmap_t);
	cvmask	      = verts_stacked + kmasks;

	for (i = 0; i < kmasks; i++) {
		verts_stacked [i] = 0;
	}

	/* Compute "vleft" for each valid hyperedge having non-zero	*/
	/* weight.  vleft is 1 less than the number of valid vertices	*/
	/* in the hyperedge.  We decrement vleft every time we delete a	*/
	/* vertex from the hyperedge.  When this count goes to zero, we	*/
	/* can delete the hyperedge.  Also count the valid hyperedges	*/
	/* with non-zero weight.					*/

	vleft = NEWA (nedges, int);
	ecount = 0;
	for (i = 0; i < nedges; i++) {
		vleft [i] = 0;
		if (BITON (edge_mask, i) AND (x [i] > FUZZ)) {
			k = 0;
			vp1 = cip -> edge [i];
			vp2 = cip -> edge [i + 1];
			while (vp1 < vp2) {
				j = *vp1++;
				if (BITON (vert_mask, j)) {
					++k;
				}
			}
			if (k >= 2) {
				vleft [i] = k - 1;
				++ecount;
			}
		}
	}

	/* Compute the congestion level Bi of each vertex i. */
	stack = NEWA (nverts, int);
	B = NEWA (nverts, double);
	sp = &stack [0];
	vcount = 0;
	for (i = 0; i < nverts; i++) {
		sum = 0.0;
		if (BITON (vert_mask, i)) {
			++vcount;
			ep1 = cip -> term_trees [i];
			ep2 = cip -> term_trees [i + 1];
			while (ep1 < ep2) {
				e = *ep1++;
				if (vleft [e] > 0) {
					sum += x [e];
				}
			}
		}
		B [i] = sum;
	}

	/* Add every vertex with weight <= 1 to a list of vertices	*/
	/* to be deleted...						*/
	for (i = 0; i < nverts; i++) {
		if (NOT BITON (vert_mask, i)) {
			/* Pretend this vertex has already been		*/
			/* stacked and deleted (already has weight 0).	*/
			SETBIT (verts_stacked, i);
		}
		else if (B [i] <= (1.0 + FUZZ)) {
			/* Schedule this vertex for deletion! */
			*sp++ = i;
			SETBIT (verts_stacked, i);
		}
	}

	/* Iteratively pop and delete vertices until stack is empty.	*/
	while (sp > stack) {
		t = *--sp;

		/* Prune vertex "t" from the remaining structure... */
		--vcount;
		B [t] = 0.0;
		/* Drop one vertex from every remaining edge that	*/
		/* contains vertex "t"...				*/
		ep1 = cip -> term_trees [t];
		ep2 = cip -> term_trees [t + 1];
		while (ep1 < ep2) {
			e = *ep1++;
			if (vleft [e] <= 0) continue;
			--(vleft [e]);
			if (vleft [e] > 0) continue;
			/* Count for this hyperedge has now gone to	*/
			/* zero, so we can delete it.  Find the edge's	*/
			/* one remaining vertex j and subtract edge e's	*/
			/* weight from Bj.				*/
			--ecount;
			vp1 = cip -> edge [e];
			vp2 = cip -> edge [e + 1];
			for (;;) {
				FATAL_ERROR_IF (vp1 >= vp2);
				j = *vp1++;
				if (B [j] > FUZZ) break;
			}
			B [j] -= x [e];

			if ((B [j] <= 1.0 + FUZZ) AND
			    (NOT BITON (verts_stacked, j))) {
				/* Schedule this vertex for deletion... */
				*sp++ = j;
				SETBIT (verts_stacked, j);
			}
		}
	}

	/* Construct a mask of the vertices that are left (the	*/
	/* congested vertices).					*/
	bp1 = cvmask;
	bp2 = bp1 + kmasks;
	bp3 = vert_mask;
	bp4 = verts_stacked;
	while (bp1 < bp2) {
		*bp1++ = (*bp3++ & ~(*bp4++));
	}

	if ((vcount EQ 0) OR (ecount EQ 0)) {
		/* Nothing congested left... no components! */
		free ((char *) B);
		free ((char *) stack);
		free ((char *) vleft);
		free ((char *) verts_stacked);
		return (NULL);
	}

	/* Now construct the component for what's left... */

	newp = NEW (struct comp);

	newp -> next		= NULL;
	newp -> num_verts	= vcount;
	newp -> num_edges	= ecount;
	newp -> flags		= CFLG_CONG;
	newp -> x		= NEWA (ecount, double);
	newp -> everts		= NEWA (ecount + 1, int *);
	newp -> vedges		= NEWA (vcount + 1, int *);
	newp -> tviol		= NEWA (vcount, double);
	newp -> rverts		= NEWA (vcount + 1, int *);
	newp -> vert_mask	= NULL;
	newp -> edge_mask	= NULL;
	newp -> cp		= NULL;

	new_vnum = NEWA (nverts, int);
	for (i = 0; i < nverts; i++) {
		new_vnum [i] = -1;
	}
	ip1 = NEWA (vcount, int);
	k = 0;
	for (i = 0; i < nverts; i++) {
		if (NOT BITON (cvmask, i)) continue;
		/* This vertex was retained... */
		new_vnum [i] = k;
		newp -> tviol [k] = 0.0;
		newp -> rverts [k] = ip1;
		*ip1++ = i;
		++k;
	}
	newp -> rverts [k] = ip1;
	FATAL_ERROR_IF (k NE vcount);

	old_enum = NEWA (nedges, int);
	new_enum = NEWA (nedges, int);
	for (i = 0; i < nedges; i++) {
		new_enum [i] = -1;
	}

	/* Construct hyperedge info, including edge -> vertices map. */
	j = 0;
	k = 0;
	for (i = 0; i < nedges; i++) {
		if (vleft [i] < 1) continue;
		new_enum [i] = j;
		old_enum [j] = i;
		newp -> x [j]	= x [i];
		k += (1 + vleft [i]);
		++j;
	}
	FATAL_ERROR_IF (j NE ecount);
	vp3 = NEWA (k, int);
	ep3 = NEWA (k, int);
	for (j = 0; j < ecount; j++) {
		newp -> everts [j] = vp3;
		i = old_enum [j];
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			t = *vp1++;
			if (NOT BITON (cvmask, t)) continue;
			t = new_vnum [t];
			FATAL_ERROR_IF (t < 0);
			*vp3++ = t;
		}
	}
	newp -> everts [ecount] = vp3;
	FATAL_ERROR_IF (vp3 NE newp -> everts [0] + k);
	for (i = 0; i < nverts; i++) {
		j = new_vnum [i];
		if (j < 0) continue;
		newp -> vedges [j] = ep3;
		ep1 = cip -> term_trees [i];
		ep2 = cip -> term_trees [i + 1];
		while (ep1 < ep2) {
			e = new_enum [*ep1++];
			if (e >= 0) {
				*ep3++ = e;
			}
		}
	}
	newp -> vedges [vcount] = ep3;
	FATAL_ERROR_IF (ep3 NE newp -> vedges [0] + k);

	free ((char *) new_enum);
	free ((char *) old_enum);
	free ((char *) new_vnum);
	free ((char *) B);
	free ((char *) stack);
	free ((char *) vleft);
	free ((char *) verts_stacked);

	return (newp);
}

/*
 * This routine simplifies a SINGLE COMPONENT, producing 0 or more
 * simplified components that are concatenated onto the front of the
 * successors that may follow the given component in the linked list.
 */

	static
	struct comp *
simplify_one_component (

struct comp *		comp,		/* IN - component to simplify */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
struct comp *		rest;
struct comp *		p;
struct comp *		temp;
struct comp *		done;
struct comp **		done_hookp;

	FATAL_ERROR_IF (comp EQ NULL);

	/* Disconnect this component from any others after it... */
	rest = comp -> next;

	p = comp;		/* one-and-only pending node... */
	p -> next  = NULL;
	done	   = NULL;	/* nothing completed yet... */
	done_hookp = &done;

	while (p NE NULL) {
		/* Process next pending component... */

		/* Create vertex and edge masks if necessary... */
		_gst_create_comp_masks (p);

		if ((p -> flags & CFLG_CONG) EQ 0) {
			/* Reduce down to congested minimum by removing	*/
			/* all uncongested vertices...			*/
			strip_uncongested_vertices (p);
		}
		else if ((p -> flags & CFLG_CC) EQ 0) {
			/* Break off connected components... */
			split_connected_components (p);
		}
		else if ((p -> flags & CFLG_BCC) EQ 0) {
			/* Break off biconnected components... */
			 p = split_biconnected_components (p, bbip);
		}
		else if ((p -> flags & CFLG_CHAIN) EQ 0) {
			/* Shrink down long chains of		*/
			/* two-vertex edges of weight 1.	*/
			merge_chains (p, bbip);
		}
		else {
			/* Component is now fully simplified.	*/
			/* Reduce/renumber it...		*/
			reduce_component_in_place (p);
			if (vacuous_component (p)) {
				/* Component disolved into nothing! */
				temp = p -> next;
				p -> next = NULL;
				_gst_free_congested_component (p);
				p = temp;
			}
			else {
				/* Component is DONE.  Move it off	*/
				/* to the "done" list...		*/
				*done_hookp = p;
				done_hookp = &(p -> next);
				p = p -> next;
				*done_hookp = NULL;
			}
		}
	}

	/* Concatenate the completed nodes onto the rest we started with. */
	*done_hookp = rest;

	return (done);
}

/*
 * This routine iteratively strips away all vertices t with delta(t) <= 1
 * until we are left with a "core" of congested vertices.
 */

	static
	void
strip_uncongested_vertices (

struct comp *		comp		/* IN - component to strip */
)
{
int			i;
int			j;
int			k;
int			e;
int			t;
int			nedges;
int			nverts;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
double			sum;
int *			vleft;
int *			sp;
bitmap_t *		verts_stacked;
int *			stack;
double *		B;
bool			changed;

	changed = FALSE;

	nverts = comp -> num_verts;
	nedges = comp -> num_edges;

	k = BMAP_ELTS (nverts);
	verts_stacked = NEWA (k, bitmap_t);

	for (i = 0; i < k; i++) {
		verts_stacked [i] = 0;
	}

	/* Compute a count of valid vertices for each edge. */
	vleft = NEWA (nedges, int);
	for (i = 0; i < nedges; i++) {
		vleft [i] = 0;
		if (NOT BITON (comp -> edge_mask, i)) continue;

		if (comp -> x [i] <= FUZZ) {
			/* Edge not really present... */
			CLRBIT (comp -> edge_mask, i);
			changed = TRUE;
			continue;
		}
		vp1 = comp -> everts [i];
		vp2 = comp -> everts [i + 1];
		k = 0;
		while (vp1 < vp2) {
			j = *vp1++;
			if (BITON (comp -> vert_mask, j)) {
				++k;
			}
		}
		if (k < 2) {
			/* Edge not really there any more... */
			CLRBIT (comp -> edge_mask, i);
			changed = TRUE;
			continue;
		}
		vleft [i] = k;
	}

	/* Compute the congestion level Bi of each vertex i. */
	B = NEWA (nverts, double);
	for (i = 0; i < nverts; i++) {
		/* Start with any weight inherent in the vertex itself... */
		sum = comp -> tviol [i];
		if (BITON (comp -> vert_mask, i)) {
			ep1 = comp -> vedges [i];
			ep2 = comp -> vedges [i + 1];
			while (ep1 < ep2) {
				e = *ep1++;
				/* Include weight only from edges with	*/
				/* at least one other valid vertex...	*/
				if (vleft [e] < 2) continue;
				sum += comp -> x [e];
			}
		}
		B [i] = sum;
	}

	/* Add every vertex with weight <= 1 to a list of vertices	*/
	/* to be deleted...						*/
	stack = NEWA (nverts, int);
	sp = &stack [0];
	for (i = 0; i < nverts; i++) {
		if (NOT BITON (comp -> vert_mask, i)) {
			/* Pretend this vertex has already been		*/
			/* stacked and deleted (already has weight 0).	*/
			SETBIT (verts_stacked, i);
		}
		else if (B [i] <= (1.0 + FUZZ)) {
			/* Schedule this vertex for deletion! */
			*sp++ = i;
			SETBIT (verts_stacked, i);
		}
	}

	/* Iteratively pop and delete vertices until stack is empty.	*/
	while (sp > stack) {
		t = *--sp;

		/* Prune vertex "t" from the remaining structure... */
		CLRBIT (comp -> vert_mask, t);
		B [t] = 0.0;
		changed = TRUE;

		/* Drop one vertex from every remaining edge that	*/
		/* contains vertex "t"...				*/
		ep1 = comp -> vedges [t];
		ep2 = comp -> vedges [t + 1];
		while (ep1 < ep2) {
			e = *ep1++;
			if (vleft [e] < 2) continue;
			--(vleft [e]);
			if (vleft [e] >= 2) continue;
			/* This edge now has fewer than 2 valid		*/
			/* vertices left.  The edge must now go	away --	*/
			/* but first we must find its last valid	*/
			/* vertex and subtract our weight from it...	*/
			CLRBIT (comp -> edge_mask, e);
			vp1 = comp -> everts [e];
			vp2 = comp -> everts [e + 1];
			for (;;) {
				FATAL_ERROR_IF (vp1 >= vp2);
				j = *vp1++;
				if (BITON (comp -> vert_mask, j)) break;
			}
			B [j] -= comp -> x [e];

			if ((B [j] <= 1.0 + FUZZ) AND
			    (NOT BITON (verts_stacked, j))) {
				/* Schedule this vertex for deletion... */
				*sp++ = j;
				SETBIT (verts_stacked, j);
			}
		}
	}

	free ((char *) stack);
	free ((char *) B);
	free ((char *) vleft);
	free ((char *) verts_stacked);

	/* This component now contains only congested vertices... */
	comp -> flags |= CFLG_CONG;

#if 1
	if (changed) {
		/* Need to look for CC/BCC's again... */
		comp -> flags &= ~(CFLG_CC | CFLG_BCC);
	}
#endif
}

/*
 * This routine determines the number of connected components within the
 * given component.  If there is only one, it is marked as connected.
 * Otherwise, one connected component is split off from this one.
 */

	static
	void
split_connected_components (

struct comp *		comp		/* IN - component to split */
)
{
int			i;
int			t;
int			e;
int			nverts;
int			nedges;
int			num_vert_masks;
int			num_edge_masks;
int *			sp;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
bitmap_t *		cc_edge_mask;
struct comp *		p2;
bitmap_t *		cc_vert_mask;
int *			stack;

	/* Get the masks, just in case... */
	_gst_create_comp_masks (comp);

	nverts		= comp -> num_verts;
	nedges		= comp -> num_edges;

	num_vert_masks	= BMAP_ELTS (nverts);
	num_edge_masks	= BMAP_ELTS (nedges);

	/* Make masks of vertices and edges in the connected	*/
	/* component...						*/
	cc_vert_mask = NEWA (num_vert_masks, bitmap_t);
	for (i = 0; i < num_vert_masks; i++) {
		cc_vert_mask [i] = 0;
	}
	cc_edge_mask = NEWA (num_edge_masks, bitmap_t);
	for (i = 0; i < num_edge_masks; i++) {
		cc_edge_mask [i] = 0;
	}

	/* Find a vertex to include in the connected component... */
	for (i = 0; ; i++) {
		if (i >= nverts) {
			/* This component has no vertices!  There is	*/
			/* definitely NOT more than one connected	*/
			/* component!  Reduce this guy, mark him as a	*/
			/* connected component, and get out.		*/
			reduce_component_in_place (comp);
			comp -> flags |= CFLG_CC;
			free ((char *) cc_edge_mask);
			free ((char *) cc_vert_mask);
			return;
		}
		if (BITON (comp -> vert_mask, i)) break;
	}

	/* This is the first vertex in the component.  Stack	*/
	/* initially contains this one vertex...		*/
	SETBIT (cc_vert_mask, i);

	stack = NEWA (nverts, int);
	sp = &stack [0];
	*sp++ = i;

	/* Scan all vertices reachable from this one via valid edges. */
	while (sp > &stack [0]) {
		t = *--sp;
		ep1 = comp -> vedges [t];
		ep2 = comp -> vedges [t + 1];
		while (ep1 < ep2) {
			e = *ep1++;
			if (NOT BITON (comp -> edge_mask, e)) continue;
			if (BITON (cc_edge_mask, e)) continue;
			SETBIT (cc_edge_mask, e);
			vp1 = comp -> everts [e];
			vp2 = comp -> everts [e + 1];
			while (vp1 < vp2) {
				i = *vp1++;
				if (NOT BITON (comp -> vert_mask, i)) continue;
				if (BITON (cc_vert_mask, i)) continue;
				SETBIT (cc_vert_mask, i);
				*sp++ = i;
			}
		}
	}

	/* The cc_vert_mask and cc_edge_mask now define a connected	*/
	/* component.  See if this is the entire component...		*/
	for (i = 0; ; i++) {
		if (i >= num_vert_masks) {
			/* This component is fully connected as-is. */
			comp -> flags |= CFLG_CC;
			free ((char *) stack);
			free ((char *) cc_edge_mask);
			free ((char *) cc_vert_mask);
			return;
		}
		if (cc_vert_mask [i] NE comp -> vert_mask [i]) break;
	}

	/* We have identified one of at least TWO connected components.	*/
	/* Copy off the one we identified, and remove it from the	*/
	/* current one...						*/
	p2 = copy_subcomponent (comp, cc_vert_mask, cc_edge_mask);
	p2 -> flags |= CFLG_CC;

	/* Insert new component right after this one... */
	p2 -> next = comp -> next;
	comp -> next = p2;

	/* Remove split-off vertices and edges from this component. */
	for (i = 0; i < num_vert_masks; i++) {
		comp -> vert_mask [i] &= ~cc_vert_mask [i];
	}
	for (i = 0; i < num_edge_masks; i++) {
		comp -> edge_mask [i] &= ~cc_edge_mask [i];
	}

	free ((char *) stack);
	free ((char *) cc_edge_mask);
	free ((char *) cc_vert_mask);
}

/*
 * This routine splits the given component into its bi-connected-components,
 * assuming that each hyperedge of n vertices is replaced with the complete
 * graph Kn.  It is essentially the standard algorithm for BCC, but modified
 * to work on a hypergraph -- hyperedges of degree 3 or more are assumed to
 * be inherently bi-connected.
 */

	static
	struct comp *
split_biconnected_components (

struct comp *		comp,		/* IN - component to split up */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			nverts;
int			nedges;
struct comp *		p;
struct bc		bc;

	/* First reduce the component.  This takes care of two issues	*/
	/* at the same time: we won't have to worry about vertex or	*/
	/* edge masks, and we can allocate smaller stacks when there	*/
	/* are no extraneous things lying around.			*/

	reduce_component_in_place (comp);

	nverts	= comp -> num_verts;
	nedges	= comp -> num_edges;

	if ((nverts <= 2) OR (nedges <= 1)) {
		/* Component is already bi-connected. */
		comp -> flags |= CFLG_BCC;
		return (comp);
	}

#if 0
	gst_channel_printf (bbip -> params -> print_solve_trace,
		" split_biconnected_components: nverts=%d, nedges=%d\n",
		nverts, nedges);
#endif

	/* We will consume (and dispose of) this component, and		*/
	/* produce a sequence of other sub-components that we prepend	*/
	/* onto the front of the list in place of this one.		*/
	bc.list		= comp -> next;
	bc.comp		= comp;
	comp -> next = NULL;

	bc.dfs		= NEWA (nverts, int);
	bc.low		= NEWA (nverts, int);
	bc.parent	= NEWA (nverts, int);

	for (i = 0; i < nverts; i++) {
		bc.dfs [i] = 0;
		bc.low [i] = 0;
		bc.parent [i] = -1;
	}

	bc.max_stack	= nedges;
	bc.stack	= NEWA (bc.max_stack, int);
	bc.sp		= bc.stack;
	bc.counter	= 0;

	bc.nvmasks	= BMAP_ELTS (nverts);
	bc.nemasks	= BMAP_ELTS (nedges);

	bc.vert_mask	= NEWA (bc.nvmasks, bitmap_t);
	bc.edge_mask	= NEWA (bc.nemasks, bitmap_t);
	bc.edges_seen	= NEWA (bc.nemasks, bitmap_t);
	bc.ncomps	= 0;
	bc.bbip		= bbip;

	for (i = 0; i < bc.nemasks; i++) {
		bc.edges_seen [i] = 0;
	}

	/* Traverse component starting with vertex 0, splitting off	*/
	/* the bi-connected components as we go...			*/
	bcc (&bc, 0);

	if (bc.ncomps > 1) {
		/* Since we broke stuff apart, it is possible that	*/
		/* some vertices that were previously congested have	*/
		/* now lost this property...  Clear the flag on each	*/
		/* generated BCC so that the congested test gets re-run	*/
		/* on them...						*/
		p = bc.list;
		for (i = 0; i < bc.ncomps; i++) {
			p -> flags &= ~CFLG_CONG;
			p = p -> next;
		}
	}

	free ((char *) bc.edges_seen);
	free ((char *) bc.edge_mask);
	free ((char *) bc.vert_mask);
	free ((char *) bc.stack);
	free ((char *) bc.parent);
	free ((char *) bc.low);
	free ((char *) bc.dfs);

	_gst_free_congested_component (comp);

	return (bc.list);
}

/*
 * This is the recursive part of the bi-connected-components algorithm.  It
 * is the standard method, with a few tweaks to work on hypergraphs instead.
 */

	static
	void
bcc (

struct bc *		bcp,		/* IN - global BCC data */
int			v		/* IN - current DFS vertex */
)
{
int			i;
int			e;
int			e2;
int			w;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			vp3;
int *			vp4;
int *			sp;
int *			stack;
int *			stack_endp;
struct comp *		comp;
struct comp *		newp;

	comp = bcp -> comp;

	FATAL_ERROR_IF ((v < 0) OR (v >= comp -> num_verts));

	++(bcp -> counter);
	bcp -> dfs [v] = bcp -> counter;
	bcp -> low [v] = bcp -> counter;
	ep1 = comp -> vedges [v];
	ep2 = comp -> vedges [v + 1];
	while (ep1 < ep2) {
		e = *ep1++;
		FATAL_ERROR_IF ((e < 0) OR (e >= comp -> num_edges));
		if (NOT BITON (bcp -> edges_seen, e)) {
			/* We haven't seen this edge before.  Push	*/
			/* it onto the stack...				*/
			stack_endp = &(bcp -> stack [bcp -> max_stack]);
			FATAL_ERROR_IF ((bcp -> sp < bcp -> stack) OR
					(bcp -> sp >= stack_endp));
			*(bcp -> sp)++ = e;
			SETBIT (bcp -> edges_seen, e);
		}
		/* Scan the vertices and process them... */
		vp1 = comp -> everts [e];
		vp2 = comp -> everts [e + 1];
		while (vp1 < vp2) {
			w = *vp1++;
			FATAL_ERROR_IF ((w < 0) OR (w >= comp -> num_verts));
			if (bcp -> dfs [w] EQ 0) {
				bcp -> parent [w] = v;
				bcc (bcp, w);
				if (bcp -> low [w] >= bcp -> dfs [v]) {
					/* We have a new BCC! */
					for (i = 0; i < bcp -> nvmasks; i++) {
						bcp -> vert_mask [i] = 0;
					}
					for (i = 0; i < bcp -> nemasks; i++) {
						bcp -> edge_mask [i] = 0;
					}
#if 0
					gst_channel_ptr trace;
					trace = bcp -> bbip -> params -> print_solve_trace;
					gst_channel_printf (trace, "bcc: popping edges");
#endif
					stack	= bcp -> stack;
					sp	= bcp -> sp;
					do {
						FATAL_ERROR_IF (sp <= stack);
						e2 = *--sp;
#if 0
						gst_channel_printf (trace,
								    " %d", e2);
#endif
						SETBIT (bcp -> edge_mask, e2);
						vp3 = comp -> everts [e2];
						vp4 = comp -> everts [e2 + 1];
						while (vp3 < vp4) {
							i = *vp3++;
							SETBIT (bcp -> vert_mask, i);
						}
					} while (e2 NE e);
					bcp -> sp = sp;
#if 0
					gst_channel_printf (trace, "\n");
					_gst_print_mask (trace,
							 " bcc: cverts =",
							 bcp -> vert_mask,
							 comp -> num_verts);
#endif
					newp = copy_subcomponent (comp,
								  bcp -> vert_mask,
								  bcp -> edge_mask);
					newp -> flags |= CFLG_BCC;
					newp -> next = bcp -> list;
					bcp -> list = newp;
					++(bcp -> ncomps);
				}
				else if (bcp -> low [w] < bcp -> low [v]) {
					bcp -> low [v] = bcp -> low [w];
				}
			}
			else if ((w NE bcp -> parent [v]) AND
				 (bcp -> dfs [w] < bcp -> low [v])) {
				bcp -> low [v] = bcp -> dfs [w];
			}
		}
	}
}

/*
 * This routine looks for long chains of edges where each link Li in
 * the chain is a group of edges with the following properties:
 *
 *	- Every edge in Li has exactly two vertices A and B.
 *	- Every edge that includes A and B is in Li.
 *	- The sum of the weights of the edges in Li is 1.
 *
 * Let L0, L1, L2, ..., Lk be such a chain of maximal length.  Let Li
 * contain vertices i-1 and i for i=1..k.  Whenever k > 1 we merge
 * vertices 1, 2, ..., k-1 together -- leaving us with a shorter chain
 * containing 3 vertices and 2 links.
 */

	static
	void
merge_chains (

struct comp *		comp,		/* IN - component to merge chains in */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			j;
int			k;
int			t1;
int			t2;
int			nverts;
int			nedges;
int			num_real_edges;
int			invalid_edge;
bool *			vmark;
int *			vcount;
int *			rvcount;
int *			elist;
int *			vlist;
int *			ep1;
int *			ep2;
int *			ep3;
int *			vp1;
int *			vp2;
int *			vp3;
int **			new_rverts;
int **			new_rptr;
int			merge_count;
gst_channel_ptr		trace;
bool			free_sets;
struct dsuf		sets;

	/* Create the masks, just in case. */
	_gst_create_comp_masks (comp);

	nverts = comp -> num_verts;
	nedges = comp -> num_edges;

	free_sets = FALSE;
	elist = NULL;
	vlist = NULL;
	vcount = NULL;
	rvcount = NULL;

	/* Mark each valid vertex having exactly 2 incident edges. */
	vmark = NEWA (nverts, bool);
	memset (vmark, FALSE, nverts * sizeof (bool));
	for (i = 0; i < nverts; i++) {
		if (NOT BITON (comp -> vert_mask, i)) continue;
		k = 0;
		ep1 = comp -> vedges [i];
		ep2 = comp -> vedges [i + 1];
		while (ep1 < ep2) {
			j = *ep1++;
			if (BITON (comp -> edge_mask, j)) {
				++k;
			}
		}
		if (k EQ 2) {
			vmark [i] = TRUE;
		}
	}

	/* Compute a list of edges that have weight 1 and cardinality 2. */
	/* At least one of the edge's two endpoints must be a vertex	 */
	/* residing in exactly 2 edges.					 */

	elist = NEWA (nedges, int);
	vlist = NEWA (2 * nedges, int);
	ep1 = elist;
	vp1 = vlist;
	num_real_edges = 0;
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (comp -> edge_mask, i)) continue;
		++num_real_edges;
		if (comp -> x [i] < 1.0 - FUZZ) continue;
		vp2 = comp -> everts [i];
		vp3 = comp -> everts [i + 1];
		k = 0;
		t1 = t2 = 0;
		while (vp2 < vp3) {
			j = *vp2++;
			if (BITON (comp -> vert_mask, j)) {
				t1 = t2;
				t2 = j;
				++k;
			}
		}
		if (k NE 2) continue;

		/* Edge has cardinality 2.  Its endpoints are t1 and t2. */

		if ((NOT vmark [t1]) AND (NOT vmark [t2])) continue;

		/* At least one endpoint is incident to exactly 2 edges	  */
		/* (including this edge).  Record edge and its endpoints. */

		*ep1++ = i;
		*vp1++ = t1;
		*vp1++ = t2;
	}

	trace = bbip -> params -> print_solve_trace;
	(void) trace;

	if (ep1 <= elist) {
		/* No suitable edges to contract.  Get out. */
#if 0
		gst_channel_printf (trace, " merge_chains: exit 1\n");
#endif
		goto all_done;
	}

	/* For each vertex, determine how many of our special edges are	*/
	/* incident.							*/
	vcount = NEWA (nverts, int);
	for (i = 0; i < nverts; i++) {
		vcount [i] = 0;
	}
	vp3 = vp1;
	vp1 = vlist;
	while (vp1 < vp3) {
		i = *vp1++;
		++(vcount [i]);
	}

	/* Now delete edges from the list that do not satisfy all of	*/
	/* the criteria for being contracted (merging their endpoints).	*/
	ep3 = ep1;
	ep1 = elist;
	ep2 = elist;
	vp1 = vlist;
	vp2 = vlist;
	while (ep2 < ep3) {
		i = *ep2++;
		t1 = *vp2++;
		t2 = *vp2++;
		if (NOT vmark [t1]) continue;
		if (NOT vmark [t2]) continue;
		if (vcount [t1] NE 2) continue;
		if (vcount [t2] NE 2) continue;

		*ep1++ = i;
		*vp1++ = t1;
		*vp1++ = t2;
	}

	if (ep1 <= elist) {
		/* No edges contracted.  Get out. */
#if 0
		gst_channel_printf (trace, " merge_chains: exit 2\n");
#endif
		goto all_done;
	}

	i = ep1 - elist;
	if (i >= num_real_edges) {
		/* Entire component is a cycle of 2-edges of weight 1! */
		if (i <= 3) {
#if 0
			gst_channel_printf (trace, " merge_chains: exit 3\n");
#endif
			goto all_done;
		}
		/* Remove any three edges list of edges to contract.	*/
		/* This will cause us to reduce the N-cycle down to a	*/
		/* 3-cycle.						*/
		ep1 -= 3;
		vp1 -= 6;
	}

	/* Remember an edge that will disappear... */
	invalid_edge = elist [0];

	/* Initialize a collection of disjoint sets, one per vertex. */
	_gst_dsuf_create (&sets, nverts);
	free_sets = TRUE;
	for (i = 0; i < nverts; i++) {
		_gst_dsuf_makeset (&sets, i);
	}

	merge_count = 0;
	ep3 = ep1;
	vp3 = vp1;
	ep1 = elist;
	vp1 = vlist;
	while (ep1 < ep3) {
		i = *ep1++;
		t1 = *vp1++;
		t2 = *vp1++;
		j = _gst_dsuf_find (&sets, t1);
		k = _gst_dsuf_find (&sets, t2);
		FATAL_ERROR_IF (j EQ k);
		/* Contract edge i, merging vertices t1 and t2. */
		_gst_dsuf_unite (&sets, j, k);
		CLRBIT (comp -> edge_mask, i);
		++merge_count;
	}

	if (merge_count <= 0) {
		/* No edges contracted.  Get out. */
#if 0
		gst_channel_printf (trace, " merge_chains: exit 4\n");
#endif
		goto all_done;
	}

	/* Count the number of "real" vertices for each component vertex. */
	rvcount = NEWA (nverts, int);
	k = 0;
	for (i = 0; i < nverts; i++) {
		j = 0;
		if (BITON (comp -> vert_mask, i)) {
			j = comp -> rverts [i + 1] - comp -> rverts [i];
		}
		rvcount [i] = j;
		k += j;
	}

	/* Update "real vertex" counts to consider effects of merging... */
	for (i = 0; i < nverts; i++) {
		j = _gst_dsuf_find (&sets, i);
		if (i NE j) {
			/* Vertex i will be donating to vertex j... */
			rvcount [j] += rvcount [i];
			rvcount [i] = 0;
		}
	}

	/* Create a new "real vertex" list to fill in. */
	new_rverts = NEWA (nverts + 1, int *);
	new_rptr   = NEWA (nverts, int *);
	vp1	   = NEWA (k, int);
	for (i = 0; i < nverts; i++) {
		new_rverts [i]	= vp1;
		new_rptr [i]	= vp1;
		vp1 += rvcount [i];
	}
	new_rverts [i] = vp1;

	/* Retain only the canonical member of each set of vertices. */
	for (i = 0; i < nverts; i++) {
		vcount [i] = 0;
		vmark [i] = FALSE;
	}
	for (i = 0; i < nverts; i++) {
		j = _gst_dsuf_find (&sets, i);
		if (i NE j) {
			/* Vertex j remains, vertex i disappears... */
			vmark [j] = TRUE;
			CLRBIT (comp -> vert_mask, i);
		}
		/* Copy i's "real" vertices into j's new bucket. */
		vp1 = comp -> rverts [i];
		vp2 = comp -> rverts [i + 1];
		vp3 = new_rptr [j];
		while (vp1 < vp2) {
			*vp3++ = *vp1++;
		}
		new_rptr [j] = vp3;
	}

	/* Get rid of old "real" vertices, keep the new... */
	free ((char *) (comp -> rverts [0]));
	free ((char *) (comp -> rverts));
	comp -> rverts = new_rverts;
	free ((char *) new_rptr);

	/* Fill edge list for each canonical vertex with edges that	*/
	/* disappeared...						*/
	for (i = 0; i < nverts; i++) {
		if (NOT vmark [i]) continue;
		ep1 = comp -> vedges [i];
		ep2 = comp -> vedges [i + 1];
		while (ep1 < ep2) {
			*ep1++ = invalid_edge;
		}
	}
	/* Renumber the vertices in each edge. */
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (comp -> edge_mask, i)) continue;
		vp1 = comp -> everts [i];
		vp2 = comp -> everts [i + 1];
		while (vp1 < vp2) {
			j = *vp1;
			k = _gst_dsuf_find (&sets, j);
			*vp1 = k;
			if (vmark [k]) {
				/* Update vertex to edge mapping. */
				j = vcount [k]++;
				comp -> vedges [k] [j] = i;
			}
			++vp1;
		}
	}

#if 0
	gst_channel_printf (trace,
		" merge_chains: reduced from %d vertices to %d\n",
		comp -> num_verts + merge_count, comp -> num_verts);
#endif

all_done:
	if (rvcount NE NULL) {
		free ((char *) rvcount);
	}
	if (free_sets) {
		_gst_dsuf_destroy (&sets);
	}
	if (vcount NE NULL) {
		free ((char *) vcount);
	}
	if (vlist NE NULL) {
		free ((char *) vlist);
	}
	if (elist NE NULL) {
		free ((char *) elist);
	}
	free ((char *) vmark);

	comp -> flags |= CFLG_CHAIN;
}

/*
 * This routine copies off a specified SUBSET of the given component
 * into a new, freshly created component that is returned.
 */

	static
	struct comp *
copy_subcomponent (

struct comp *		comp,		/* IN - component to copy from */
bitmap_t *		vert_mask,	/* IN - vertices to copy */
bitmap_t *		edge_mask	/* IN - hyperedges to copy */
)
{
int			i;
int			j;
int			k;
int			nverts;
int			nedges;
int			src;
int			dst;
int			new_num_verts;
int			new_num_edges;
int			rvcount;
struct comp *		newp;
int *			ip1;
int *			ip2;
int *			ip3;
int *			ip4;
int *			vert_renum;
int *			edge_renum;
int *			vlist;
int *			elist;
int *			rvlist;

	/* Force masks to be present... */
	_gst_create_comp_masks (comp);

	nverts = comp -> num_verts;
	nedges = comp -> num_edges;

	/* Get arrays used to renumber the vertices and edges... */
	vert_renum	= NEWA (nverts, int);
	edge_renum	= NEWA (nedges, int);

	/* Compute renumberings... */
	j = 0;
	for (i = 0; i < nverts; i++) {
		if (BITON (vert_mask, i)) {
			vert_renum [i] = j++;
		}
		else {
			vert_renum [i] = -1;
		}
	}
	new_num_verts = j;

	j = 0;
	for (i = 0; i < nedges; i++) {
		if (BITON (edge_mask, i)) {
			edge_renum [i] = j++;
		}
		else {
			edge_renum [i] = -1;
		}
	}
	new_num_edges = j;

	/* Compute size of new edge-to-vertices list... */
	k = 0;
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (edge_mask, i)) continue;
		ip1 = comp -> everts [i];
		ip2 = comp -> everts [i + 1];
		while (ip1 < ip2) {
			j = *ip1++;
			if (NOT BITON (vert_mask, j)) continue;
			++k;
		}
	}

	/* Compute size of new "real vertices" list... */
	rvcount = 0;
	for (i = 0; i < nverts; i++) {
		if (BITON (vert_mask, i)) {
			ip1 = comp -> rverts [i];
			ip2 = comp -> rverts [i + 1];
			rvcount += (ip2 - ip1);
		}
	}

	/* Create the new component... */
	newp = NEW (struct comp);

	vlist  = NEWA (k, int);
	elist = NEWA (k, int);
	rvlist = NEWA (rvcount, int);

	newp -> next		= NULL;
	newp -> num_verts	= new_num_verts;
	newp -> num_edges	= new_num_edges;
	newp -> flags		= comp -> flags;
	newp -> x		= NEWA (new_num_edges, double);
	newp -> everts		= NEWA (new_num_edges + 1, int *);
	newp -> vedges		= NEWA (new_num_verts + 1, int *);
	newp -> tviol		= NEWA (new_num_verts, double);
	newp -> rverts		= NEWA (new_num_verts + 1, int *);
	newp -> vert_mask	= NULL;
	newp -> edge_mask	= NULL;
	newp -> cp		= NULL;

	/* Remap/delete entries in the edge-to-vertices mapping... */
	dst = 0;
	ip1 = vlist;
	for (src = 0; src < nedges; src++) {
		if (edge_renum [src] < 0) continue;
		ip2 = comp -> everts [src];
		ip3 = comp -> everts [src + 1];
		newp -> everts [dst] = ip1;
		while (ip2 < ip3) {
			j = *ip2++;
			j = vert_renum [j];
			if (j >= 0) {
				*ip1++ = j;
			}
		}
		newp -> x [dst] = comp -> x [src];
		++dst;
	}
	newp -> everts [dst] = ip1;
	FATAL_ERROR_IF (dst NE new_num_edges);

	/* Remap/delete entries in the vertex-to-edges mapping... */
	dst = 0;
	ip1 = elist;
	ip4 = rvlist;
	for (src = 0; src < nverts; src++) {
		if (vert_renum [src] < 0) continue;
		ip2 = comp -> vedges [src];
		ip3 = comp -> vedges [src + 1];
		newp -> vedges [dst] = ip1;
		while (ip2 < ip3) {
			j = *ip2++;
			j = edge_renum [j];
			if (j >= 0) {
				*ip1++ = j;
			}
		}
		newp -> tviol [dst] = comp -> tviol [src];
		ip2 = comp -> rverts [src];
		ip3 = comp -> rverts [src + 1];
		newp -> rverts [dst] = ip4;
		while (ip2 < ip3) {
			*ip4++ = *ip2++;
		}
		++dst;
	}
	newp -> rverts [dst] = ip4;
	newp -> vedges [dst] = ip1;
	FATAL_ERROR_IF (dst NE new_num_verts);

	free ((char *) edge_renum);
	free ((char *) vert_renum);

	return (newp);
}

/*
 * This routine reduces the given component in-place by re-numbering
 * all of the verticess and edges so that the ones that are not
 * marked in the vertex and edge masks are gone.
 *
 * This functions as a kind of garbage-compaction effect, but we do
 * NOT actually re-allocate ourselves into smaller memory...
 */

	static
	void
reduce_component_in_place (

struct comp *		comp		/* IN - component to reduce */
)
{
int			i;
int			j;
int			nverts;
int			nedges;
int			src;
int			dst;
int			new_num_verts;
int			new_num_edges;
int *			ip1;
int *			ip2;
int *			ip3;
int *			ip4;
int *			vert_renum;
int *			edge_renum;

	if ((comp -> vert_mask EQ NULL) AND (comp -> edge_mask EQ NULL)) {
		/* Already reduced... */
		return;
	}

	_gst_create_comp_masks (comp);	/* in case one exists, but not the other... */

	nverts = comp -> num_verts;
	nedges = comp -> num_edges;

	if ((nverts <= 0) OR (nedges <= 0)) {
		comp -> num_verts = 0;
		comp -> num_edges = 0;
		free ((char *) (comp -> vert_mask));
		free ((char *) (comp -> edge_mask));
		comp -> vert_mask = NULL;
		comp -> edge_mask = NULL;
		return;
	}

	/* Get arrays used to renumber the vertices and edges... */
	vert_renum	= NEWA (nverts, int);
	edge_renum	= NEWA (nedges, int);

	/* Compute renumberings... */
	j = 0;
	for (i = 0; i < nverts; i++) {
		if (BITON (comp -> vert_mask, i)) {
			vert_renum [i] = j++;
		}
		else {
			vert_renum [i] = -1;
		}
	}
	new_num_verts = j;

	j = 0;
	for (i = 0; i < nedges; i++) {
		if (BITON (comp -> edge_mask, i)) {
			edge_renum [i] = j++;
		}
		else {
			edge_renum [i] = -1;
		}
	}
	new_num_edges = j;

	/* Remap/delete entries in the edge-to-vertices mapping... */
	dst = 0;
	ip1 = comp -> everts [0];
	for (src = 0; src < nedges; src++) {
		if (edge_renum [src] < 0) continue;
		ip2 = comp -> everts [src];
		ip3 = comp -> everts [src + 1];
		comp -> everts [dst] = ip1;
		while (ip2 < ip3) {
			j = *ip2++;
			j = vert_renum [j];
			if (j >= 0) {
				*ip1++ = j;
			}
		}
		comp -> x [dst] = comp -> x [src];
		++dst;
	}
	comp -> everts [dst] = ip1;
	FATAL_ERROR_IF (dst NE new_num_edges);
	comp -> num_edges = new_num_edges;

	/* Remap/delete entries in the vertex-to-hyperedges mapping... */
	dst = 0;
	ip1 = comp -> vedges [0];
	ip4 = comp -> rverts [0];
	for (src = 0; src < nverts; src++) {
		if (vert_renum [src] < 0) continue;
		ip2 = comp -> vedges [src];
		ip3 = comp -> vedges [src + 1];
		comp -> vedges [dst] = ip1;
		while (ip2 < ip3) {
			j = *ip2++;
			j = edge_renum [j];
			if (j >= 0) {
				*ip1++ = j;
			}
		}
		comp -> tviol [dst] = comp -> tviol [src];
		ip2 = comp -> rverts [src];
		ip3 = comp -> rverts [src + 1];
		comp -> rverts [dst] = ip4;
		while (ip2 < ip3) {
			*ip4++ = *ip2++;
		}
		++dst;
	}
	comp -> rverts [dst] = ip4;
	comp -> vedges [dst] = ip1;
	FATAL_ERROR_IF (dst NE new_num_verts);

	/* Component is now a different size... */
	comp -> num_verts	= new_num_verts;
	comp -> num_edges	= new_num_edges;

	free ((char *) edge_renum);
	free ((char *) vert_renum);

	/* Free the valid vertices and edges masks... */
	free ((char *) (comp -> edge_mask));
	free ((char *) (comp -> vert_mask));
	comp -> vert_mask	= NULL;
	comp -> edge_mask	= NULL;
}

/*
 * This routine determines whether or not the given component is
 * "vacuous" -- too small to contain a subtour violation.
 */

	static
	int
vacuous_component (

struct comp *		p		/* IN - component to test */
)
{
	if (p -> num_verts <= 0) return (TRUE);
	if (p -> num_edges <= 1) return (TRUE);
	if (p -> num_verts EQ 1) {
		/* Special check for exactly 1 vertex... */
		if (p -> tviol [0] > FUZZ) {
			/* We have a single vertex, but it is congested! */
			/* This is a valid component... */
			return (FALSE);
		}
		/* Only 1 vertex -- no congestion -- ignore this component */
		return (TRUE);
	}

	/* We have a component worth looking at for violated SEC's... */

	return (FALSE);
}

/*
 * Free up the given *list* of congested components.
 */

	void
_gst_free_congested_component_list (

struct comp *		p	/* IN: list of components to free */
)
{
struct comp *		tmp;

	while (p NE NULL) {
		tmp = p -> next;
		p -> next = NULL;

		_gst_free_congested_component (p);

		p = tmp;
	}
}

/*
 * This routine frees up the given congested component.
 */

	void
_gst_free_congested_component (

struct comp *		p	/* IN - component to free */
)
{
	if (p -> vert_mask NE NULL) {
		free ((char *) (p -> vert_mask));
	}
	if (p -> edge_mask NE NULL) {
		free ((char *) (p -> edge_mask));
	}
	free ((char *) (p -> vedges [0]));
	free ((char *) (p -> everts [0]));
	free ((char *) (p -> rverts [0]));
	free ((char *) (p -> rverts));
	free ((char *) (p -> tviol));
	free ((char *) (p -> vedges));
	free ((char *) (p -> everts));
	free ((char *) (p -> x));
	free ((char *) p);
}

/*
 * This routine deletes a specified vertex from the given congested
 * component and then does all possible further simplifications on what
 * is left.  Note that it is possible that the given component splits
 * into several when this is done...
 *
 * Also note that the given component may be MODIFIED -- even FREED!
 */

	struct comp *
_gst_delete_vertex_from_component (

int			t,		/* IN - vertex to delete */
struct comp *		comp,		/* IN - component to delete from */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
	FATAL_ERROR_IF (t >= comp -> num_verts);

	/* Create masks if not currently present... */
	_gst_create_comp_masks (comp);

	if (t >= 0) {
		if (NOT BITON (comp -> vert_mask, t)) {
			/* Deleting a vertex that is not present. */
			FATAL_ERROR;
		}

		/* Remove the vertex from the bit-mask... */
		CLRBIT (comp -> vert_mask, t);
	}
	else {
		/* Caller has deleted vertices.  Just clean up.	*/
	}

	/* Now that we have removed a vertex, we can no longer	*/
	/* be certain that the component is: fully congested,	*/
	/* connected, bi-connected, etc...			*/
	comp -> flags &= ~CFLG_ALL;

	comp = simplify_one_component (comp, bbip);

	return (comp);
}

/*
 * This routine adds the vertex and edge masks to the given
 * component, if they are NOT already present...
 */

	void
_gst_create_comp_masks (

struct comp *		p		/* IN - component to put masks on */
)
{
int		i;
int		n;
int		nmasks;
bitmap_t *	mask;

	if (p -> vert_mask EQ NULL) {
		/* Sprout a new vertex mask with all vertices present. */
		n = p -> num_verts;
		nmasks = BMAP_ELTS (n);
		if (nmasks < 1) {
			nmasks = 1;
		}
		mask = NEWA (nmasks, bitmap_t);
		for (i = 0; i < nmasks; i++) {
			mask [i] = 0;
		}
		for (i = 0; i < n; i++) {
			SETBIT (mask, i);
		}
		p -> vert_mask = mask;
	}

	if (p -> edge_mask EQ NULL) {
		/* Sprout a new edge mask with all edges present. */
		n = p -> num_edges;
		nmasks = BMAP_ELTS (n);
		if (nmasks < 1) {
			nmasks = 1;
		}
		mask = NEWA (nmasks, bitmap_t);
		for (i = 0; i < nmasks; i++) {
			mask [i] = 0;
		}
		for (i = 0; i < n; i++) {
			SETBIT (mask, i);
		}
		p -> edge_mask = mask;
	}
}

/*
 * This routine selects a vertex that is a member of the previous
 * solution S and has minimum weight.
 */

	int
_gst_find_least_congested_vertex (

bitmap_t *		S,	/* IN - previous separation solution */
struct comp *		comp	/* IN - congested component */
)
{
int			i;
int			j;
int			nverts;
int			nedges;
int *			ip1;
int *			ip2;
int			min_vert;
double			w;
double			min_weight;

	nverts	= comp -> num_verts;
	nedges	= comp -> num_edges;

	min_weight	= 2.0 * nedges;
	min_vert	= -1;

	for (i = 0; i < nverts; i++) {
		if ((S NE NULL) AND (NOT BITON (S, i))) continue;
		w = 0.0;
		ip1 = comp -> vedges [i];
		ip2 = comp -> vedges [i + 1];
		while (ip1 < ip2) {
			j = *ip1++;
			w += comp -> x [j];
		}
		if (w < min_weight) {
			min_weight = w;
			min_vert   = i;
		}
	}
	if (min_vert < 0) {
		/* Least congested vertex not found? */
		FATAL_ERROR;
	}

	return (min_vert);
}

/*
 * This routine produces the proper constraints that are represented
 * by the given subset of the given component.  We try to "clean up"
 * the constraint by taking only the congested portion of the
 * vertices in it.  This can strengthen the constraints considerably.
 */

	struct constraint *
_gst_check_component_subtour (

bitmap_t *		S,		/* IN - subtour (set of vertices) */
struct comp *		comp,		/* IN - congested component */
struct constraint *	cp,		/* IN - existing constraints */
double *		x,		/* IN - LP solution vector */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i, kmasks, ncomps;
bool			gen_orig, any_reductions;
struct comp *		vcomps;
struct comp *		tmp;
struct comp **		comp_array;
struct constraint **	constraint_array;
struct gst_hypergraph *	cip;
struct constraint *	cp2;
bitmap_t *		orig_stour;
bitmap_t *		stour;

	cip = bbip -> cip;
	kmasks = cip -> num_vert_masks;

	orig_stour = NEWA (2 * kmasks, bitmap_t);
	stour	   = orig_stour + kmasks;

	/* Construct a bitmask of the REAL vertices comprising	*/
	/* this subtour...					*/
	component_verts_to_real_verts (S, comp, orig_stour, kmasks);

	if (bbip -> params -> strengthen_reduce EQ
	    GST_PVAL_STRENGTHEN_REDUCE_DISABLE) {
		/* Strengthening via reductions is disabled.	*/
		/* Just generate the original subtour.		*/
		cp = _gst_check_subtour (orig_stour, cp, x, bbip);

		free (orig_stour);

		return (cp);
	}

	any_reductions	= FALSE;

	/* Silence compiler warnings. */
	(void) copy_component_induced_by_vertices;
	(void) greedy_improvement;

	/* Take the violated subtour and "clean" it up by computing the	*/
	/* congested components for it.  This will get rid of some of	*/
	/* the cruft, and help to strengthen the generated constraints.	*/

#if 0
	{ bool print_flag;
	  bitmap_t * edge_mask;
		print_flag = FALSE;
		edge_mask = cip -> initial_edge_mask;
		vcomps = _gst_find_congested_components (x,
							 orig_stour,
							 edge_mask,
							 print_flag,
							 bbip);
	}
#else
	/* A more efficient alternative to starting all over! */
	vcomps = copy_component_induced_by_vertices (comp, S);

	vcomps = simplify_one_component (vcomps, bbip);
#endif

#if 0
	/* Greedily remove vertices that do not cause us to	*/
	/* lose our violation, although the magnitude of the	*/
	/* violation may decrease.				*/

	vcomps = greedy_improvement (vcomps, bbip);
#endif

	/* Make an array of the components in vcomps. */
	ncomps = 0;
	for (tmp = vcomps; tmp NE NULL; tmp = tmp -> next) {
		++ncomps;
	}
	comp_array	 = NULL;
	constraint_array = NULL;
	if (ncomps > 0) {
		comp_array	 = NEWA (ncomps, struct comp *);
		constraint_array = NEWA (ncomps, struct constraint *);
		i = 0;
		for (tmp = vcomps; tmp NE NULL; tmp = tmp -> next) {
			comp_array [i++] = tmp;
		}
		FATAL_ERROR_IF (i NE ncomps);
	}

	/* Generate a constraint for each component.  Record the	*/
	/* constraint associated with each component (NULL if no	*/
	/* violation exists.						*/
	for (i = 0; i < ncomps; i++) {
		tmp = comp_array [i];

		component_verts_to_real_verts (tmp -> vert_mask,
					       tmp,
					       stour,
					       kmasks);

		/* Add it to the list (if it is a violation). */
		cp2 = _gst_check_subtour (stour, cp, x, bbip);

		if (cp2 NE cp) {
			/* Record violated constraint. */
			constraint_array [i] = cp2;
		}
		else {
			/* This one is not violated. */
			constraint_array [i] = NULL;
		}
		cp = cp2;
	}

	plot_significant_reductions (ncomps,
				     comp_array,
				     constraint_array,
				     x,
				     orig_stour,
				     bbip);

	if (constraint_array NE NULL) {
		free (constraint_array);
	}
	if (comp_array NE NULL) {
		free (comp_array);
	}

	_gst_free_congested_component_list (vcomps);

	gen_orig	= FALSE;
	if (NOT any_reductions) {
		gen_orig = TRUE;
	}
	else if (bbip -> params -> generate_unstrengthened EQ
	    GST_PVAL_GENERATE_UNSTRENGTHENED_ENABLE) {
		gen_orig = TRUE;
	}

	if (gen_orig) {
		/* Generate the original (unstrengthened) subtour. */
		cp = _gst_check_subtour (orig_stour, cp, x, bbip);
	}

	free ((char *) orig_stour);

	return (cp);
}

/*
 * Given a component, compute the sub-hypergraph *induced* by the given
 * subset S of the component's vertices.  Returns a new component that
 * must be freed.
 */

	static
	struct comp *
copy_component_induced_by_vertices (

struct comp *	comp,		/* IN: Original component */
bitmap_t *	S		/* IN: Inducing vertex subset */
)
{
int		i, j, k;
int		nmasks;
bitmap_t *	sedges;
bitmap_t *	vert_mask;
bitmap_t *	edge_mask;
int *		ip1;
int *		ip2;
struct comp *	new_comp;

	/* Compute edges induced by the subtour vertices. */
	nmasks = BMAP_ELTS (comp -> num_edges);
	sedges = NEWA (nmasks, bitmap_t);

	memset (sedges, 0, nmasks * sizeof (sedges [0]));

	for (i = 0; i < comp -> num_edges; i++) {
		ip1 = comp -> everts [i];
		ip2 = comp -> everts [i + 1];
		k = 0;
		while (ip1 < ip2) {
			j = *ip1++;
			if (BITON (S, j)) {
				++k;
				if (k >= 2) {
					SETBIT (sedges, i);
					break;
				}
			}
		}
	}

	vert_mask	= S;
	edge_mask	= sedges;

	new_comp = copy_subcomponent (comp, vert_mask, edge_mask);

	free (sedges);

	return (new_comp);
}

/*
 * This routine converts a vertex subset within a given component
 * back into the corresponding subset of the original vertices.
 * Specifying a NULL pointer for comp_S is a convenient way to specify
 * that the entire component is to be translated back out.
 */

	static
	void
component_verts_to_real_verts (

bitmap_t *		comp_S,	/* IN - subset of vertices in component */
struct comp *		comp,	/* IN - component those vertices are in */
bitmap_t *		real_S,	/* OUT - real set of vertices */
int			kmasks	/* IN - size of real_S */
)
{
int			i;
int			j;
int *			ip1;
int *			ip2;

	/* Zero out entire set of real vertices... */
	for (i = 0; i < kmasks; i++) {
		real_S [i] = 0;
	}

	/* Set the "real" vertex number bits for each component vertex. */
	for (i = 0; i < comp -> num_verts; i++) {
		if ((comp_S NE NULL) AND (NOT BITON (comp_S, i))) continue;
		ip1 = comp -> rverts [i];
		ip2 = comp -> rverts [i + 1];
		while (ip1 < ip2) {
			j = *ip1++;
			SETBIT (real_S, j);
		}
	}
}

/*
 * A greedy heuristic for strengthening subtour constraints.  During
 * each iteration, we remove a vertex t that causes the smallest decrease
 * in violation -- but only if removing t does not cause the violation
 * to disappear entirely.
 */

	static
	struct comp *
greedy_improvement (

struct comp *		comp,		/* IN - component to improve */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
struct comp *		done;
struct comp *		p1;
struct comp *		p2;

	done = NULL;
	while (comp NE NULL) {
		/* Disconnect first component from the rest. */
		p1 = comp;
		comp = p1 -> next;
		p1 -> next = NULL;

		/* Improve this component... */
		p1 = greedily_improve_one_component (p1, bbip);

		while (p1 NE NULL) {
			p2 = p1 -> next;
			p1 -> next = done;
			done = p1;
			p1 = p2;
		}
	}

	return (done);
}

/*
 * Greedily improve a single component.
 */

	static
	struct comp *
greedily_improve_one_component (

struct comp *		comp,		/* IN - component to improve */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			j;
int			k;
int			i1;
int			j1;
int			k1;
int			k2;
int			e;
int			t;
int			nedges;
int			nverts;
int			nheap;
int *			ip1;
int *			ip2;
int *			ip3;
int *			ip4;
int *			vleft;
int *			heap;
int *			hindex;
double *		B;
double			sum;
double			z;
bool			changed;

	FATAL_ERROR_IF (comp EQ NULL);
	FATAL_ERROR_IF (comp -> next NE NULL);

	_gst_create_comp_masks (comp);

	changed = FALSE;

	nverts = comp -> num_verts;
	nedges = comp -> num_edges;

	k = BMAP_ELTS (nverts);

	/* Compute "vleft" for each valid hyperedge having non-zero	*/
	/* weight.  vleft is 1 less than the number of valid vertices	*/
	/* in the hyperedge.  We decrement vleft every time we delete a	*/
	/* vertex from the hyperedge.  When this count goes to zero, we	*/
	/* can delete the hyperedge.					*/

	vleft = NEWA (nedges, int);
	for (i = 0; i < nedges; i++) {
		vleft [i] = 0;
		if (NOT BITON (comp -> edge_mask, i)) continue;

		if (comp -> x [i] <= FUZZ) {
			/* Edge not really present... */
			CLRBIT (comp -> edge_mask, i);
			changed = TRUE;
			continue;
		}
		ip1 = comp -> everts [i];
		ip2 = comp -> everts [i + 1];
		k = 0;
		while (ip1 < ip2) {
			j = *ip1++;
			if (BITON (comp -> vert_mask, j)) {
				++k;
			}
		}
		if (k < 2) {
			/* Hyperedge not really there... */
			CLRBIT (comp -> edge_mask, i);
			changed = TRUE;
			continue;
		}
		vleft [i] = k - 1;
	}

	/* Compute the conjestion level Bi of each vertex i. */
	B = NEWA (2 * nverts, double);
	k = 0;
	for (i = 0; i < nverts; i++) {
		/* Start with any weight inherent in the vertex itself... */
		sum = comp -> tviol [i];
		if (BITON (comp -> vert_mask, i)) {
			++k;
			ip1 = comp -> vedges [i];
			ip2 = comp -> vedges [i + 1];
			while (ip1 < ip2) {
				e = *ip1++;
				/* Include weight only from edges	*/
				/* with at least one other valid	*/
				/* vertex...				*/
				if (vleft [e] > 0) {
					sum += comp -> x [e];
				}
			}
		}
		B [i] = sum;
	}

	/* Compute the violation Z for the entire component.	*/
	/* Z > 0 implies a violation.				*/

	z = - (k - 1);
	for (i = 0; i < nedges; i++) {
		z += (vleft [i] * comp -> x [i]);
	}

	if (z < FUZZ) goto get_out;

	/* Put all of the vertices into a heap that is sorted	*/
	/* so as to give the smallest current B value.		*/
	/* Note that this requires only O(N) time.		*/

	heap = NEWA (2 * nverts, int);
	hindex = heap + nverts;
	nheap = 0;

	for (i = 0; i < nverts; i++) {
		hindex [i] = -1;
	}
	for (i = 0; i < nverts; i++) {
		if (NOT BITON (comp -> vert_mask, i)) continue;
		heap [nheap] = i;
		hindex [i] = nheap;
		++nheap;
	}
	for (i = nheap / 2 - 1; i >= 0; i--) {
		/* Sift down at node i. */
		j = i;
		for (;;) {
			/* Let k be j's left child. */
			k = 2 * j + 1;
			if (k >= nheap) break;
			k1 = heap [k];
			if (k + 1 < nheap) {
				k2 = heap [k + 1];
				if (B [k1] > B [k2]) {
					/* Right exists, and is smaller... */
					k1 = k2;
					++k;
				}
			}
			j1 = heap [j];
			if (B [j1] <= B [k1]) break;
			/* Swap parent and child.  Update hindex too. */
			heap [j] = k1;
			heap [k] = j1;
			hindex [j1] = k;
			hindex [k1] = j;
			/* Move down the heap. */
			j = k;
		}
	}

	/* Now use the heap to iteratively delete vertices. */
	while (nheap > 0) {
		/* Get the vertex t having the smallest B value. */
		t = heap [0];

		z = z - (B [t] - 1.0);
		if (z <= FUZZ) {
			/* Deleting vertex t would cause the violation	*/
			/* to disappear completely, so we are done.	*/
			break;
		}

		changed = TRUE;

		/* Delete vertex t from the heap. */

		FATAL_ERROR_IF (hindex [t] NE 0);
		hindex [t] = -1;
		--nheap;
		i1 = heap [nheap];
		heap [0] = i1;
		hindex [i1] = 0;
		/* Sift down at node 0. */
		j = 0;
		for (;;) {
			/* Let k be j's left child. */
			k = 2 * j + 1;
			if (k >= nheap) break;
			k1 = heap [k];
			if (k + 1 < nheap) {
				k2 = heap [k + 1];
				if (B [k1] > B [k2]) {
					/* Right exists, and is smaller... */
					k1 = k2;
					++k;
				}
			}
			j1 = heap [j];
			if (B [j1] <= B [k1]) break;
			/* Swap parent and child.  Update hindex too. */
			heap [j] = k1;
			heap [k] = j1;
			hindex [j1] = k;
			hindex [k1] = j;
			/* Move down the heap. */
			j = k;
		}

		/* Delete vertex t from the component. */
		CLRBIT (comp -> vert_mask, t);
		ip1 = comp -> vedges [t];
		ip2 = comp -> vedges [t + 1];
		while (ip1 < ip2) {
			e = *ip1++;
			if (vleft [e] <= 0) continue;
			--(vleft [e]);
			if (vleft [e] > 0) continue;
			/* Count for this hyperedge has now gone to	*/
			/* zero, so we can delete it.  Find the edge's	*/
			/* one remaining vertex j and subtract edge e's	*/
			/* weight from Bj.				*/
			CLRBIT (comp -> edge_mask, e);
			ip3 = comp -> everts [e];
			ip4 = comp -> everts [e + 1];
			for (;;) {
				FATAL_ERROR_IF (ip3 >= ip4);
				j = *ip3++;
				if (BITON (comp -> vert_mask, j)) break;
			}
			B [j] -= comp -> x [e];
			/* Vertex j's B value has been decreased. */
			/* We must therefore sift its node up in the heap. */
			j1 = j;
			j = hindex [j1];
			while (j > 0) {
				k = (j - 1) >> 1;
				k1 = heap [k];
				if (B [k1] <= B [j1]) break;
				/* Swap parent and child, update hindex. */
				heap [k] = j1;
				heap [j] = k1;
				hindex [j1] = k;
				hindex [k1] = j;
				j = k;
			}
		}
	}

	free ((char *) heap);

get_out:

	free ((char *) B);
	free ((char *) vleft);

	/* This component now contains only congested vertices... */
	comp -> flags |= CFLG_CONG;

#if 1
	if (changed) {
		/* Need to look for CC/BCC's again... */
		comp -> flags &= ~(CFLG_CC | CFLG_BCC);

		comp = simplify_one_component (comp, bbip);
	}
#endif

	return (comp);
}

/*
 * A routine to generate postscript plots describing cases wherein
 * the reductions split a violated subtour into a non-trivial number
 * of smaller, separate subtours (all likely much stronger than the
 * original subtour).
 *
 * At least two problems here:
 *
 *	1. This is a library resident file calling non-library
 *	   routines (genps.c is not in the library).
 *
 *	2. The postscript we generate is going to be stuck inside of
 *	   postscript comments(!) unless we disable postscript
 *	   commenting on the channel.
 */

	static
	void
plot_significant_reductions (

int			ncomps,		/* IN: Number of components */
struct comp **		comps,		/* IN: Array of components */
struct constraint **	constraints,	/* IN: Array of constraints */
double *		x,		/* IN: LP solution */
bitmap_t *		orig_stour,	/* IN: Original violated subtour */
struct bbinfo *		bbip		/* IN: branch-and-bound info */
)
{
int			i, nviol;
struct gst_hypergraph *	cip;
struct comp *		p;
struct constraint *	cp;
gst_channel_ptr		trace;

	cip	= bbip -> cip;
	trace	= bbip -> params -> print_solve_trace;

	(void) cip;
	(void) trace;

	/* Count and retain only violated components. */
	nviol = 0;
	for (i = 0; i < ncomps; i++) {
		cp = constraints [i];
		if (cp NE NULL) {
			p = comps [i];

			comps [nviol]		= p;
			constraints [nviol]	= cp;
			++nviol;
		}
	}

	if (nviol < 3) {
		/* This is not "significant." */
		return;
	}

#if 0
	_gst_plot_lp_solution (trace,
			       cip,
			       "Full LP Solution",
			       x,
			       BIG_PLOT);

	_gst_plot_subtour (trace,
			   "Separated Subtour",
			   x, cip, orig_stour, BIG_PLOT);

	for (i = 0; i < nviol; i++) {
		p = comps [i];
		cp = constraints [i];
		FATAL_ERROR_IF ((p EQ NULL) OR (cp EQ NULL));

		_gst_plot_subtour (trace,
				   "Strengthened Subtour",
				   x, cip, cp -> mask, BIG_PLOT);
	}
#endif
}
