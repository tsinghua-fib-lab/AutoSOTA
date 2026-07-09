/***********************************************************************

	$Id: localcut.c,v 1.33 2023/04/03 17:25:53 warme Exp $

	File:	localcut.c
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 1997, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	A special separation routine that can find violated facets
	in a small-dimensional projection of the space.  When lifted
	back up, they are certainly valid constraints, although they
	might not be true facets for the global problem.

************************************************************************

	Modification Log:

	a-1:	12/07/99	warme
		: Created.
	b-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Uses parameters.
		: Uses channels for trace output.
		: Uses gst_hgmst().
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganized include files, apply prefixes.
		: Fix -Wall issues.  Upgrade fatals.
		: Make features unconditional.
	e-4:	11/18/2022	warme
		: Simplify calling convention of various functions.
		: Use function to free list of components.
		: Fix -Wall issues.
	e-5:	04/03/2023	warme
		: Fix bad bug that caused invalid local cuts to be
		:  generated when the "Exact algorithm" for max-weight
		:  forest sub-problems was sufficiently small to use
		:  the backtrack search, which does not correctly
		:  handle edges with negative costs.
		: Add a new parameter that selects different
		:  trace output formats.

************************************************************************/

#include "localcut.h"

#include "bb.h"
#include "ckpt.h"
#include "constrnt.h"
#include "ddsuf.h"
#include "dsuf.h"
#include "fatal.h"
#include <float.h>
#include "geosteiner.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "p1read.h"
#include "parmblk.h"
#include "sec_comp.h"
#include "sec_heur.h"
#include "solver.h"
#include "sortfuncs.h"
#include "steiner.h"
#include <string.h>
#include "utils.h"


/*
 * Global Routines
 */

struct constraint *	_gst_find_local_cuts (
					double *		x,
					struct bbinfo *		bbip,
					struct constraint *	cp);
struct constraint *	_gst_find_local_cuts_in_component (
					struct comp *		comp,
					double *		x,
					struct bbinfo *		bbip,
					struct constraint *	cp);

int		_gst_find_forests (struct comp *	comp,
				   bitmap_t **		flist,
				   gst_channel_ptr	print_solve_trace);
void		_gst_print_forests (struct comp *	comp,
				    bitmap_t *		flist,
				    int			n,
				    gst_channel_ptr	print_solve_trace);


/*
 * Local Equates
 */

#define FORMULATE_DUAL		0

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
	bitmap_t *	fsets_seen;	/* edges already pushed */
	int		nvmasks;	/* size of vert_mask */
	int		nemasks;	/* size of edge_mask */
	int		ncomps;		/* number of BCC's split off */
	int		max_stack;	/* size of stack */
	struct bbinfo *	bbip;		/* branch and bound info */
};

struct ff {
	struct comp *	comp;
	int *		vbuf;
	bool *		vflag;
	int		nforests;
	bitmap_t *	ptr;
	struct ddsuf	sets;
};

/*
 * An object to encapsulate the printing of trace output from the
 * localcut algorithm.
 */

struct LCTrace {
	struct bbinfo *	bbip;	/* Branch-and-bound info */
	int		num_cv;
	int		num_af;
	int		num_exact;

	/* Pointers to functions, each of which encapsulates a	*/
	/* different type of trace message.			*/
	void		(*del_slack) (struct LCTrace *	tp,
				      int		ndel,
				      int		nleft);
	void		(*cut_violation) (struct LCTrace *	tp,
					  const char *		var,
					  double		val);
	void		(*add_forest) (struct LCTrace *	tp,
				       int			nf,
				       const int *		forest,
				       double			z);
	void		(*exact) (struct LCTrace * tp);
	void		(*lp_error) (struct LCTrace *		tp,
				     const char *		format,
				     int			status);
	void		(*finish) (struct LCTrace * tp);
};

/* The "classic" and "quiet" derived objects both operate	*/
/* directly with the LTrace base class.				*/

/*
 * Derived class for producing terse output.
 */

struct LCTrace_terse {
	struct LCTrace	base;	/* Base class */

	int	fcount;		/* # Forests toward next dot */
	int	f_per_dot;	/* Forests per dot */
	int	numdots;	/* # Dots on current line */
	char	dotchar;	/* '.' or 'x' (if Exact Algorithm used) */
};

/*
 * Local Routines
 */

static void		add_forest_to_lp (LP_t *, int, int *);
static void		bcc_fcomp (struct bc *, int);
static void		classic_trace_add_forest (
					struct LCTrace *	tp,
					int			nf,
					const int *		forest,
					double			z);
static void		classic_trace_cut_violation (
					struct LCTrace *	tp,
					const char *		var,
					double			val);
static void		classic_trace_del_slack (
					struct LCTrace *	tp,
					int			ndel,
					int			nleft);
static void		classic_trace_exact (struct LCTrace * tp);
static void		classic_trace_finish (struct LCTrace * tp);
static void		classic_trace_lp_error (
					struct LCTrace *	tp,
					const char *		format,
					int			status);
static int		compare_edge_ratios (struct comp *,
					     double *,
					     int *,
					     int,
					     int);
static int		compare_edges (struct comp *, int, int);
static int		compare_edges_lexicographically (struct comp *, int, int);
static int		compute_fake_edge_verts (struct comp *,
						 double *,
						 int *);
static struct comp *	copy_fsubcomp (struct comp *	comp,
				       bitmap_t *	vert_mask,
				       bitmap_t *	edge_mask);
static void		create_fcomp_masks (struct comp *);
static void		delete_slack (LP_t *, double *, struct LCTrace * tp);
static void		ff_recurse (int, int, int, bitmap_t, struct ff *);
static struct constraint * find_fcomp_cut (struct comp *,
					   struct bbinfo *,
					   struct constraint *);
static struct comp *	find_first_fcomp (double *		x,
					  struct bbinfo *	bbip);
static struct comp *	find_fractional_comps (double *		x,
					       bool		print_flag,
					       struct bbinfo *	bbip);
static int		find_large_weight_forest (
					struct comp *		comp,
					double *		cost,
					int *			edge_freq,
					int *			forest,
					gst_param_ptr		params,
					struct LCTrace *	tp);
static int		find_max_forest (struct comp *,
					 double *,
					 int *,
					 gst_param_ptr);
static int *		heapsort_edges (struct comp *);
static bool		is_failed_subproblem (struct comp *, struct bbinfo *);
static struct constraint * lift_constraint (struct comp *,
					    double *,
					    double,
					    struct bbinfo *);
static struct LCTrace *	make_classic_tracer (struct bbinfo * bbip);
static LP_t *		make_fcomp_lp (struct comp *, struct lpmem *);
static struct LCTrace *	make_quiet_tracer (struct bbinfo * bbip);
static struct LCTrace *	make_terse_tracer (struct bbinfo * bbip);
static struct LCTrace *	make_tracer (struct bbinfo * bbip);
static int		max_forest_heuristic (struct comp *,
					      double *,
					      int *,
					      int *);
static double		max_forest_kruskal (struct comp *,
					    int *,
					    bool *,
					    double *,
					    int *,
					    int *);
static struct comp *	merge_equivalent_edges (struct comp *);
static void		quiet_trace_add_forest (
					struct LCTrace *	tp,
					int			nf,
					const int *		forest,
					double			z);
static void		quiet_trace_cut_violation (
					struct LCTrace *	tp,
					const char *		var,
					double			val);
static void		quiet_trace_del_slack (
					struct LCTrace *	tp,
					int			ndel,
					int			nleft);
static void		quiet_trace_exact (struct LCTrace * tp);
static void		quiet_trace_finish (struct LCTrace * tp);
static void		quiet_trace_lp_error (
					struct LCTrace *	tp,
					const char *		format,
					int			status);
static void		record_failed_fcomp (struct comp *, struct bbinfo *);
static void		reduce_fcomp_in_place (struct comp *);
static struct comp *	simplify_one_fcomp (struct comp *, struct bbinfo *);
static int *		sort_edges_by_cost_ratio (struct comp *,
						  double *,
						  int *);
static struct comp *	split_biconnected_fcomps (struct comp *,
						  struct bbinfo *);
static struct comp *	subgraph_induced_by_vertices (double *,
						      bitmap_t *,
						      bitmap_t *,
						      struct gst_hypergraph *);
static void		terse_trace_add_forest (
					struct LCTrace *	tp,
					int			nf,
					const int *		forest,
					double			z);
static void		terse_trace_cut_violation (
					struct LCTrace *	tp,
					const char *		var,
					double			val);
static void		terse_trace_del_slack (
					struct LCTrace *	tp,
					int			ndel,
					int			nleft);
static void		terse_trace_exact (struct LCTrace * tp);
static void		terse_trace_finish (struct LCTrace * tp);
static void		terse_trace_lp_error (
					struct LCTrace *	tp,
					const char *		format,
					int			status);
static int		vacuous_fcomp (struct comp *);


/*
 * This routine takes all of the "real" vertices for the given
 * congested component, and computes a local cut for them -- if the resulting
 * number of vertices and edges is small enough.
 */

	struct constraint *
_gst_find_local_cuts_in_component (

struct comp *		comp,		/* IN - component to separate */
double *		x,		/* IN - LP solution to separate */
struct bbinfo *		bbip,		/* IN - branch-and-bound-info */
struct constraint *	cp		/* IN - existing constraints */
)
{
int			i;
int			j;
int			nverts;
int			nedges;
int			kmasks;
bitmap_t *		comp_vert_mask;
int *			ip1;
int *			ip2;
bitmap_t *		bp1;
bitmap_t *		bp2;
struct gst_hypergraph *	cip;
bitmap_t *		vert_mask;
bitmap_t *		edge_mask;
struct comp *		comp2;
gst_param_ptr		params;

	if (	bbip -> params -> local_cuts_mode	EQ GST_PVAL_LOCAL_CUTS_MODE_DISABLE
	    OR	bbip -> params -> local_cuts_max_depth	EQ GST_PVAL_LOCAL_CUTS_MAX_DEPTH_DISABLE) {
		return (cp);
	}

	nverts	= comp -> num_verts;
	nedges	= comp -> num_edges;

	_gst_create_comp_masks (comp);

	vert_mask = comp -> vert_mask;
	edge_mask = comp -> edge_mask;

	if (   (nverts > bbip -> params -> local_cuts_max_vertices)
	    OR (nedges > bbip -> params -> local_cuts_max_edges)) {
		/* Component already exceeds maximums -- even before	*/
		/* we expand it back to "real" vertices!		*/
		return (cp);
	}

	params = gst_create_param (NULL);
	gst_copy_param (params, bbip -> params);

	if (params -> local_cuts_trace_depth EQ 0) {
		params -> print_solve_trace = NULL;
	}
	else {
		params -> local_cuts_trace_depth  = params -> local_cuts_trace_depth - 1;
	}

	cip	= bbip -> cip;
	kmasks	= cip -> num_vert_masks;

	/* Compute a mask of all "real" vertices in the component. */
	comp_vert_mask = NEWA (kmasks, bitmap_t);
	for (i = 0; i < kmasks; i++) {
		comp_vert_mask [i] = 0;
	}
	ip1 = comp -> rverts [0];
	ip2 = comp -> rverts [nverts];
	while (ip1 < ip2) {
		j = *ip1++;
		SETBIT (comp_vert_mask, j);
	}
	/* Make sure the component did not contain any invalid vertices. */
	bp1 = comp_vert_mask;
	bp2 = vert_mask;
	for (i = 0; i < kmasks; i++) {
		if ((*bp1++ & ~*bp2++) NE 0) {
			FATAL_ERROR;
		}
	}

	comp2 = subgraph_induced_by_vertices (x,
					      comp_vert_mask,
					      edge_mask,
					      cip);

	free ((char *) comp_vert_mask);

	if ((comp2 -> num_verts <= params -> local_cuts_max_vertices) AND
	    (comp2 -> num_edges <= params -> local_cuts_max_edges)) {

		comp2 = merge_equivalent_edges (comp2);

		cp = find_fcomp_cut (comp2, bbip, cp);
	}

	_gst_free_congested_component (comp2);

	gst_free_param (params);
	return (cp);
}

/*
 * This routine produces the sub-hypergraph induced by the given subset
 * of the vertices.  In other words, given (V,E) it returns the
 * hypergraph (V,E') such that E' = { e \in E : |e \intersect V| >= 2}.
 */

	static
	struct comp *
subgraph_induced_by_vertices (

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
int			ecount;
int			vcount;
int *			ecard;
int *			vp1;
int *			vp2;
int *			vp3;
int *			ep1;
int *			ep2;
int *			ep3;
int *			ip1;
int *			new_vnum;
int *			new_enum;
int *			old_enum;
struct comp *		newp;

	nedges = cip -> num_edges;
	nverts = cip -> num_verts;

	/* Compute the cardinality (in V) of each edge.	*/
	/* Edges NOT in E have cardinality 0.  If its	*/
	/* cardinality ends up being less than 2, we	*/
	/* set it to 0.					*/

	ecard = NEWA (nedges, int);
	ecount = 0;
	for (i = 0; i < nedges; i++) {
		ecard [i] = 0;
		if (NOT BITON (edge_mask, i)) continue;

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
			ecard [i] = k;
			++ecount;
		}
	}

	/* Count the vertices in V. */

	vcount = 0;
	for (i = 0; i < nverts; i++) {
		if (BITON (vert_mask, i)) {
			++vcount;
		}
	}

	/* Now construct the component... */

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
		if (NOT BITON (vert_mask, i)) continue;
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

	/* Construct edge info, including edge -> vertices map. */
	j = 0;
	k = 0;
	for (i = 0; i < nedges; i++) {
		if (ecard [i] < 2) continue;
		new_enum [i] = j;
		old_enum [j] = i;
		newp -> x [j]	= x [i];
		k += ecard [i];
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
			if (NOT BITON (vert_mask, t)) continue;
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

	/* Sort the vertices of each edge into ascending order. */
	for (i = 0; i < ecount; i++) {
		vp1 = newp -> everts [i];
		vp2 = newp -> everts [i + 1];
		_gst_sort_ints (vp1, vp2 - vp1);
	}

	free ((char *) new_enum);
	free ((char *) old_enum);
	free ((char *) new_vnum);
	free ((char *) ecard);

	return (newp);
}

/*
 * A special separation routine that can find facets in a small-dimensional
 * projection of the polytope.  When lifted back up, they resulting
 * constraints are certainly valid, although they might not be true facets
 * for the global problem.
 *
 * In fact, we don't in general know *WHAT* they are: they could be
 * subtours, or something completely weird.
 */

	struct constraint *
_gst_find_local_cuts (

double *		x,		/* IN - LP solution to separate */
struct bbinfo *		bbip,		/* IN - branch-and-bound info */
struct constraint *	cp		/* IN - list of constraints */
)
{
struct comp *		comp;
struct comp *		p;
bool			print_flag;
gst_param_ptr		params;

	if (	bbip -> params -> local_cuts_mode	EQ GST_PVAL_LOCAL_CUTS_MODE_DISABLE
	    OR	bbip -> params -> local_cuts_max_depth	EQ GST_PVAL_LOCAL_CUTS_MAX_DEPTH_DISABLE) {
		return (cp);
	}

	params = gst_create_param (NULL);
	gst_copy_param (params, bbip -> params);

	if (params -> local_cuts_trace_depth EQ 0) {
		params -> print_solve_trace = NULL;
	}
	else {
		params -> local_cuts_trace_depth  = params -> local_cuts_trace_depth - 1;
	}

	print_flag = TRUE;
	comp = find_fractional_comps (x, print_flag, bbip);

	for (p = comp; p NE NULL; p = p -> next) {
		cp = find_fcomp_cut (p, bbip, cp);
	}

	_gst_free_congested_component_list (comp);

	gst_free_param (params);

	return (cp);
}

/*
 * This routine performs a set of reductions, similar to those that
 * are done by the SEC separator.  These reductions are less powerful
 * and constitute a subset of those done for SECs, however.
 *
 * Essentially, we compute the bi-connected components, and delete those
 * that consist entirely of integral-weight edges.  This leaves only
 * a set of fractional "blobs".
 */

	static
	struct comp *
find_fractional_comps (

double *		x,		/* IN - current LP solution */
bool			print_flag,	/* IN - TRUE ==> print info */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			n;
struct comp *		comp;
struct comp *		p;
gst_channel_ptr		trace;

	trace = bbip -> params -> print_solve_trace;

	/* Find the first component by iteratively applying the	*/
	/* delta(t) <= 1 rule...				*/

	comp = find_first_fcomp (x, bbip);

	if (comp EQ NULL) return (NULL);

	if (print_flag AND (trace NE NULL)) {
		gst_channel_printf (trace,
			"initially %d fractional component vertices:\n",
			comp -> num_verts);
	}

	comp = simplify_one_fcomp (comp, bbip);

	if (print_flag AND (trace NE NULL)) {

		n = 0;
		for (p = comp; p NE NULL; p = p -> next) {
			++n;
		}
		gst_channel_printf (trace,
			"find_fractional_comps found %d components:\n",
			n);
		n = 0;
		for (p = comp; p NE NULL; p = p -> next) {
			gst_channel_printf (trace,
				"\tfcomp %d:\t%d verts,\t%d edges\n",
				n++, p -> num_verts, p -> num_edges);
		}
	}

	return (comp);
}

/*
 * This routine finds the initial component by iteratively deleting
 * degree 1 vertices whose incident edge has integral weight.  This
 * normally shrinks the problem down substantially.  We then convert
 * the problem into "struct comp" form.
 */

	static
	struct comp *
find_first_fcomp (

double *		x,		/* IN - current LP solution */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
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
int *			ip1;
int *			vleft;
int *			sp;
int *			new_vnum;
int *			new_enum;
int *			old_enum;
struct comp *		newp;
bitmap_t *		bp1;
bitmap_t *		bp2;
bitmap_t *		bp3;
bitmap_t *		bp4;
bitmap_t *		verts_stacked;
bitmap_t *		verts_deleted;
bitmap_t *		cvmask;
int *			stack;
int *			degree;
struct gst_hypergraph *	cip;
bitmap_t *		vert_mask;
bitmap_t *		edge_mask;

	cip		= bbip -> cip;
	vert_mask	= bbip -> vert_mask;
	edge_mask	= bbip -> edge_mask;

	nedges = cip -> num_edges;
	nverts = cip -> num_verts;
	kmasks = cip -> num_vert_masks;

	verts_stacked = NEWA (3 * kmasks, bitmap_t);
	verts_deleted = verts_stacked + kmasks;
	cvmask	      = verts_deleted + kmasks;

	for (i = 0; i < kmasks; i++) {
		verts_stacked [i] = 0;
		verts_deleted [i] = 0;
	}

	/* Compute "vleft" for each edge.  After this many of its	*/
	/* vertices have been deleted, then the edge itself may be	*/
	/* deleted.							*/

	vleft = NEWA (nedges, int);
	for (i = 0; i < nedges; i++) {
		vleft [i] = 0;
		if (NOT BITON (edge_mask, i)) continue;
		if (x [i] <= FUZZ) continue;

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
		}
	}

	/* Compute the degree of each vertex... */
	stack = NEWA (nverts, int);
	degree = NEWA (nverts, int);
	sp = &stack [0];
	vcount = 0;
	for (i = 0; i < nverts; i++) {
		k = 0;
		if (BITON (vert_mask, i)) {
			++vcount;
			ep1 = cip -> term_trees [i];
			ep2 = cip -> term_trees [i + 1];
			while (ep1 < ep2) {
				e = *ep1++;
				if (vleft [e] > 0) {
					++k;
				}
			}
		}
		degree [i] = k;
	}

	/* Mark every vertex that is incident to a fractional edge as	*/
	/* being "already stacked".  This prevents these vertices from	*/
	/* being deleted by this algorithm.				*/

	for (i = 0; i < nedges; i++) {
		if (NOT BITON (edge_mask, i)) continue;
		if (x [i] <= FUZZ) continue;
		if (x [i] >= 1.0 - FUZZ) continue;

		/* edge i is fractional... */
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			j = *vp1++;
			SETBIT (verts_stacked, j);
		}
	}

	/* Push every vertex having degree <= 1 (unless it has been	*/
	/* marked already).  Vertices on the stack get deleted.		*/

	for (i = 0; i < nverts; i++) {
		if (NOT BITON (vert_mask, i)) {
			/* Pretend this vertex has already been		*/
			/* stacked and deleted (already has weight 0).	*/
			SETBIT (verts_stacked, i);
		}
		else if ((degree [i] <= 1) AND
			 (NOT BITON (verts_stacked, i))) {
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
		degree [t] = 0;
		SETBIT (verts_deleted, t);

		/* Decrement vleft for each edge containing vertex t.	*/
		ep1 = cip -> term_trees [t];
		ep2 = cip -> term_trees [t + 1];
		while (ep1 < ep2) {
			e = *ep1++;
			if (vleft [e] <= 0) continue;
			--(vleft [e]);
			if (vleft [e] > 0) continue;
			/* Time to delete this edge.  Find the edge's	*/
			/* remaining vertex and decrement its degree.	*/
			vp1 = cip -> edge [e];
			vp2 = cip -> edge [e + 1];
			for (;;) {
				FATAL_ERROR_IF (vp1 >= vp2);
				j = *vp1++;
				if (degree [j] > 0) break;
			}
			--(degree [j]);

			if ((degree [j] <= 1) AND
			    (NOT BITON (verts_stacked, j))) {
				/* Schedule this vertex for deletion... */
				*sp++ = j;
				SETBIT (verts_stacked, j);
			}
		}
	}

	/* Construct a mask of the vertices that are left.	*/
	bp1 = cvmask;
	bp2 = bp1 + kmasks;
	bp3 = vert_mask;
	bp4 = verts_deleted;
	while (bp1 < bp2) {
		*bp1++ = (*bp3++ & ~(*bp4++));
	}

	if (vcount EQ 0) {
		/* Nothing left... no components! */
		free ((char *) degree);
		free ((char *) stack);
		free ((char *) vleft);
		free ((char *) verts_stacked);
		return (NULL);
	}

	/* Tally the number of non-deleted vertices in each edge. */
	/* Count all edges having 2 or more retained vertices -- */
	/* EVEN IF THEY HAVE ZERO LP WEIGHT!!!			 */

	ecount = 0;
	for (i = 0; i < nedges; i++) {
		vleft [i] = 0;
		if (NOT BITON (edge_mask, i)) continue;
		k = 0;
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			j = *vp1++;
			if (NOT BITON (verts_deleted, j)) {
				++k;
			}
		}
		vleft [i] = k;
		if (k >= 2) {
			++ecount;
		}
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
		if (vleft [i] < 2) continue;
		new_enum [i] = j;
		old_enum [j] = i;
		newp -> x [j]	= x [i];
		k += vleft [i];
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

	/* Sort the vertices of each edge into ascending order. */
	for (i = 0; i < ecount; i++) {
		vp1 = newp -> everts [i];
		vp2 = newp -> everts [i + 1];
		_gst_sort_ints (vp1, vp2 - vp1);
	}

	free ((char *) new_enum);
	free ((char *) old_enum);
	free ((char *) new_vnum);
	free ((char *) degree);
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
simplify_one_fcomp (

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

	/* Process the pending component... */

	/* Create vertex and edge masks if necessary... */
	create_fcomp_masks (p);

	/* Merge all edges that span the same subset of vertices. */
	p = merge_equivalent_edges (p);

	/* Break off biconnected components... */
	p = split_biconnected_fcomps (p, bbip);

	while (p NE NULL) {
		/* Component is now fully simplified.	*/
		/* Reduce/renumber it...		*/
		reduce_fcomp_in_place (p);

		if (vacuous_fcomp (p)) {
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

	/* Concatenate the completed nodes onto the rest we started with. */
	*done_hookp = rest;

	return (done);
}

/*
 * This routine merges all edges that span the same subset of the
 * vertices.
 */

	static
	struct comp *
merge_equivalent_edges (

struct comp *		comp		/* IN - component to simplify */
)
{
int			i;
int			j;
int			k;
int			i1;
int			i2;
int			nedges;
int *			index;
bool			edges_merged;

	nedges = comp -> num_edges;

	if (nedges < 2) return (comp);

	create_fcomp_masks (comp);

	index = heapsort_edges (comp);

	edges_merged = FALSE;

	for (i = 0; i < nedges; ) {
		i1 = index [i];
		for (j = i + 1; j < nedges; j++) {
			i2 = index [j];
			k = compare_edges_lexicographically (comp, i1, i2);
			if (k > 0) {
				/* Edges aren't properly sorted! */
				FATAL_ERROR;
			}
			if (k < 0) break;
			/* Edges i1 and i2 span the same subset! */
			/* Merge i2 into i1. */
			comp -> x [i1] += comp -> x [i2];
			CLRBIT (comp -> edge_mask, i2);
			edges_merged = TRUE;
		}
		i = j;
	}

	free ((char *) index);

	if (edges_merged) {
		reduce_fcomp_in_place (comp);
	}

	return (comp);
}

/*
 * This routine creates an array of all the edge numbers of the given
 * component.  The array gives the order of the edges, lexicographically.
 * This sort is made stable by including the edge number as the final
 * sort key.
 */

	static
	int *
heapsort_edges (

struct comp *		comp		/* IN - component to sort edges of */
)
{
int			i;
int			j;
int			k;
int			n;
int			i1;
int			i2;
int *			index;

	n = comp -> num_edges;

	index = NEWA (n, int);
	for (i = 0; i < n; i++) {
		index [i] = i;
	}

	/* Construct the heap via sift-downs, in O(n) time. */
	for (k = n >> 1; k >= 0; k--) {
		j = k;
		for (;;) {
			i = (j << 1) + 1;
			if (i + 1 < n) {
				/* Increment i (to right subchild of j) */
				/* if the right subchild is greater. */
				i1 = index [i];
				i2 = index [i + 1];
				if (compare_edges (comp, i1, i2) < 0) {
					++i;
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			i1 = index [j];
			i2 = index [i];
			if (compare_edges (comp, i1, i2) >= 0) {
				/* Parent is >= greatest child, */
				/* Sift-down is done. */
				break;
			}
			/* Sift down and continue. */
			index [j] = i2;
			index [i] = i1;
			j = i;
		}
	}

	/* Now do actual sorting.  Exchange first/last and sift down. */
	while (n > 1) {
		/* Largest is at index [0], swap with index [n-1],	*/
		/* thereby putting it into final position.		*/
		--n;
		i = index [0];
		index [0] = index [n];
		index [n] = i;

		/* Now restore the heap by sifting index [0] down. */
		j = 0;
		for (;;) {
			i = (j << 1) + 1;
			if (i + 1 < n) {
				/* Increment i (to right subchild of j) */
				/* if the right subchild is greater. */
				i1 = index [i];
				i2 = index [i + 1];
				if (compare_edges (comp, i1, i2) < 0) {
					++i;
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			i1 = index [j];
			i2 = index [i];
			if (compare_edges (comp, i1, i2) >= 0) {
				/* Parent is >= greatest child, */
				/* Sift-down is done. */
				break;
			}
			/* Sift down and continue. */
			index [j] = i2;
			index [i] = i1;
			j = i;
		}
	}

	return (index);
}

/*
 * This routine compares two edges lexicographically.  If they are
 * equal, it uses the edge numbers to resolve the difference.
 */

	static
	int
compare_edges (

struct comp *		comp,		/* IN - component containing edges */
int			e1,		/* IN - first edge number */
int			e2		/* IN - second edge number */
)
{
int			status;

	status = compare_edges_lexicographically (comp, e1, e2);
	if (status EQ 0) {
		if (e1 < e2) {
			status = -1;
		}
		else if (e1 > e2) {
			status = 1;
		}
	}

	return (status);
}


/*
 * Compare two edges lexicographically.  We assume that the vertices of
 * each edge have been sorted.
 */

	static
	int
compare_edges_lexicographically (

struct comp *		comp,		/* IN - component containing edges */
int			e1,		/* IN - first edge number */
int			e2		/* IN - second edge number */
)
{
int		i1;
int		i2;
int *		p1;
int *		p2;
int *		end1;
int *		end2;

	p1	= comp -> everts [e1];
	end1	= comp -> everts [e1 + 1];

	p2	= comp -> everts [e2];
	end2	= comp -> everts [e2 + 1];

	for (;;) {
		if (p1 >= end1) {
			if (p2 >= end2) break;
			return (-1);
		}
		if (p2 >= end2) return (1);
		i1 = *p1++;
		i2 = *p2++;
		if (i1 < i2) return (-1);
		if (i1 > i2) return (1);
	}
	return (0);
}

/*
 * This routine splits the given component into its bi-connected-components,
 * assuming that each edge of size n is replaced with the complete graph
 * Kn.  It is essentially the standard algorithm for BCC, but modified to
 * work on a hypergraph -- hyperedges of degree 3 or more are assumed to be
 * inherently bi-connected.
 */

	static
	struct comp *
split_biconnected_fcomps (

struct comp *		comp,		/* IN - component to split up */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			nverts;
int			nedges;
struct comp *		p;
struct bc		bc;

#if 0
	/* First reduce the component.  This takes care of two issues	*/
	/* at the same time: we won't have to worry about vertex or	*/
	/* edge masks, and we can allocate smaller stacks when there	*/
	/* are no extraneous things lying around.			*/

	reduce_fcomp_in_place (comp);
#endif

	nverts	= comp -> num_verts;
	nedges	= comp -> num_edges;

	if ((nverts <= 2) OR (nedges <= 1)) {
		/* Component is already bi-connected. */
		comp -> flags |= CFLG_BCC;
		return (comp);
	}

#if 0
	gst_channel_printf (bbip -> params -> print_solve_trace,
		" split_biconnected_fcomps: nverts=%d, nedges=%d\n",
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
	bc.fsets_seen	= NEWA (bc.nemasks, bitmap_t);
	bc.ncomps	= 0;
	bc.bbip		= bbip;

	for (i = 0; i < bc.nemasks; i++) {
		bc.fsets_seen [i] = 0;
	}

	for (i = 0; i < nverts; i++) {
		if (bc.dfs [i] > 0) continue;
		/* Traverse component starting with vertex i, splitting	*/
		/* off the bi-connected components as we go...		*/
		bcc_fcomp (&bc, i);
	}

	if (bc.ncomps > 1) {
		/* Since we broke stuff apart, it is possible that	*/
		/* some vertices that were previously reduced have	*/
		/* now lost this property...  Clear the flag on each	*/
		/* generated BCC so that the congested test gets re-run	*/
		/* on them...						*/
		p = bc.list;
		for (i = 0; i < bc.ncomps; i++) {
			p -> flags &= ~CFLG_CONG;
			p = p -> next;
		}
	}

	free ((char *) bc.fsets_seen);
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
bcc_fcomp (

struct bc *		bcp,		/* IN - global BCC data */
int			v		/* IN - current DFS vertex */
)
{
int			i;
int			k;
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
gst_channel_ptr		trace;

	comp = bcp -> comp;
	trace = bcp -> bbip -> params -> print_solve_trace;
	(void) trace;

	FATAL_ERROR_IF ((v < 0) OR (v >= comp -> num_verts));

	++(bcp -> counter);
	bcp -> dfs [v] = bcp -> counter;
	bcp -> low [v] = bcp -> counter;
	ep1 = comp -> vedges [v];
	ep2 = comp -> vedges [v + 1];
	while (ep1 < ep2) {
		e = *ep1++;
		FATAL_ERROR_IF ((e < 0) OR (e >= comp -> num_edges));
		if (NOT BITON (bcp -> fsets_seen, e)) {
			/* We haven't seen this edge before.  Push	*/
			/* it onto the stack...				*/
			stack_endp = &(bcp -> stack [bcp -> max_stack]);
			FATAL_ERROR_IF ((bcp -> sp < bcp -> stack) OR
					(bcp -> sp >= stack_endp));
			*(bcp -> sp)++ = e;
			SETBIT (bcp -> fsets_seen, e);
		}
		/* Scan the vertices and process them... */
		vp1 = comp -> everts [e];
		vp2 = comp -> everts [e + 1];
		while (vp1 < vp2) {
			w = *vp1++;
			FATAL_ERROR_IF ((w < 0) OR (w >= comp -> num_verts));
			if (bcp -> dfs [w] EQ 0) {
				bcp -> parent [w] = v;
				bcc_fcomp (bcp, w);
				if (bcp -> low [w] >= bcp -> dfs [v]) {
					/* We have a new BCC! */
					for (i = 0; i < bcp -> nvmasks; i++) {
						bcp -> vert_mask [i] = 0;
					}
					for (i = 0; i < bcp -> nemasks; i++) {
						bcp -> edge_mask [i] = 0;
					}
#if 0
					gst_channel_printf (trace,
						" bcc_fcomp: popping edges");
#endif
					k = 0;
					stack	= bcp -> stack;
					sp	= bcp -> sp;
					do {
						FATAL_ERROR_IF (sp <= stack);
						e2 = *--sp;
#if 0
						gst_channel_printf (trace,
								    " %d",
								    e2);
#endif
						SETBIT (bcp -> edge_mask, e2);
						vp3 = comp -> everts [e2];
						vp4 = comp -> everts [e2 + 1];
						while (vp3 < vp4) {
							i = *vp3++;
							SETBIT (bcp -> vert_mask, i);
						}
						++k;
					} while (e2 NE e);
					bcp -> sp = sp;
#if 0
					gst_channel_printf (trace, "\n");
					_gst_print_mask (trace,
							 " bcc_fcomp: cverts =",
							 bcp -> vert_mask,
							 comp -> num_verts);
#endif
					if (k > 1) {
						newp = copy_fsubcomp (comp,
									  bcp -> vert_mask,
									  bcp -> edge_mask);
						newp -> flags |= CFLG_BCC;
						newp -> next = bcp -> list;
						bcp -> list = newp;
						++(bcp -> ncomps);
					}
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
 * This routine copies off a specified SUBSET of the given component
 * into a new, freshly created component that is returned.
 */

	static
	struct comp *
copy_fsubcomp (

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
	create_fcomp_masks (comp);

	if (vert_mask EQ NULL) {
		vert_mask = comp -> vert_mask;
	}
	if (edge_mask EQ NULL) {
		edge_mask = comp -> edge_mask;
	}

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
		newp -> rverts [dst] = ip4;
		ip2 = comp -> rverts [src];
		ip3 = comp -> rverts [src + 1];
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
 * all of the vertices and edges so that the ones that are not
 * marked in the vertex and edge masks are gone.
 *
 * This functions as a kind of garbage-compaction effect, but we do
 * NOT actually re-allocate ourselves into smaller memory...
 */

	static
	void
reduce_fcomp_in_place (

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

	/* in case one exists, but not the other... */
	create_fcomp_masks (comp);

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
vacuous_fcomp (

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
 * This routine adds the vertex and edge masks to the given fractional
 * component, if they are NOT already present...
 */

	static
	void
create_fcomp_masks (

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
 * This routine computes a local cut for the given fractional component --
 * which is really a projection of the original separation problem
 * into a much smaller dimensional space.
 *
 * We find exactly one constraint for the component.  If it happens that
 * this constraint is violated by the current LP solution, then we have
 * won.  The constraint *is* a facet in the low-dimensional space -- but
 * after we lift it back up to the full dimensionality it may or may not
 * be facet-defining -- all bets are off.  All we can be sure of is that
 * the constraint is valid and violated.
 */

#define USE_EUCLIDEAN_NORM	0

	static
	struct constraint *
find_fcomp_cut (

struct comp *		comp,		/* IN - component to separate */
struct bbinfo *		bbip,		/* IN - branch-and-bound info */
struct constraint *	cp		/* IN - existing constraints */
)
{
int			i;
int			nverts;
int			nedges;
int			nf;
int			status;
int			slack_size;
int *			forest;
double *		y;
double *		slack;
LP_t *			lp;
struct constraint *	newcp;
int *			edge_freq;
double			w;
double			z;
struct lpmem		lpmem;
gst_param_ptr		params;
gst_channel_ptr		print_solve_trace;
struct LCTrace *	tp;

#if USE_EUCLIDEAN_NORM
  double		dist2;
  double		prev_dist2;
  double		ynorm2;
#else
  double		best_w;
#endif

	params = bbip -> params;
	print_solve_trace = params -> print_solve_trace;

	nverts	= comp -> num_verts;
	nedges	= comp -> num_edges;

#if 0
	if (cp EQ NULL) goto pull_out_all_the_stops;
#endif

	if (   (nverts > params -> local_cuts_max_vertices)
	    OR (nedges > params -> local_cuts_max_edges)) {
		/* Problem is too big to attempt! */
		return (cp);
	}
#if 0
pull_out_all_the_stops:
#endif
	if (nverts > (params -> local_cuts_vertex_threshold * bbip -> cip -> num_verts)) {
		/* Sub-problem is too large a fraction */
		/* of the containing problem. */
		return (cp);
	}
	if (is_failed_subproblem (comp, bbip)) {
		/* Sub-problem was previously tried and failed. */
		return (cp);
	}

	gst_channel_printf (print_solve_trace,
		"Enter find_fcomp_cut with %d vertices and %d edges\n",
		nverts, nedges);

	lp = make_fcomp_lp (comp, &lpmem);

	y	= NEWA (nedges, double);
	forest	= NEWA (nverts, int);

	slack_size = 10 * GET_LP_NUM_ROWS (lp);
	slack = NEWA (slack_size, double);

	edge_freq = NEWA (nedges, int);
	for (i = 0; i < nedges; i++) {
		edge_freq [i] = 0;
	}

#if USE_EUCLIDEAN_NORM
	prev_dist2	= DBL_MAX;
#else
	best_w		= DBL_MAX;
#endif

	tp = make_tracer (bbip);

	for (;;) {
		i = GET_LP_NUM_ROWS (lp) + 1;
		if (slack_size < i) {
			/* Reallocate the slack vector. */
			slack_size = 2 * i;
			free ((char *) slack);
			slack = NEWA (slack_size, double);
		}

		/* Solve the LP instance... */

#ifdef CPLEX

#if FORMULATE_DUAL
		status = _MYCPX_primopt (lp);
		if (status NE 0) {
			tp -> lp_error (tp,
					" WARNING primopt: status = %d\n",
					status);
		}
		i = _MYCPX_solution (lp, &status, &w, NULL, y, slack, NULL);
#else
		status = _MYCPX_dualopt (lp);
		if (status NE 0) {
			tp -> lp_error (tp,
					" WARNING dualopt: status = %d\n",
					status);
		}
		i = _MYCPX_solution (lp, &status, &w, y, NULL, slack, NULL);
#endif
		FATAL_ERROR_IF (i NE 0);
		if (status NE _MYCPX_STAT_OPTIMAL) {
			tp -> lp_error (tp,
					" WARNING: solution status = %d\n",
					status);
		}
#endif

#ifdef LPSOLVE
		status = solve (lp);

		if (status NE OPTIMAL) {
			tp -> lp_error (tp,
					" WARNING: solution status = %d\n",
					status);
		}

		w = lp -> best_solution [0];

#if FORMULATE_DUAL
		/* Get LP dual variables... */
		for (i = 0; i < nedges; i++) {
			y [i] = lp -> duals [1 + i];
		}

		/* FIXME - need slack vars here... */
		for (i = 0; i <= lp -> rows; i++) {
			slack [i] = 0.0;
		}
#else
		/* Get LP solution variables... */
		for (i = 0; i < nedges; i++) {
			y [i] = lp -> best_solution [lp -> rows + i + 1];
		}

		/* FIXME - need slack vars here... */
		for (i = 0; i <= lp -> rows; i++) {
			slack [i] = 0.0;
		}
#endif
#endif

		/* Find a forest x that violates the constraint	*/
		/* y*x <= 1.  Try a fast heuristic first.  If	*/
		/* no violation is found, then use the entire	*/
		/* branch-and-cut recursively to find a forest	*/
		/* x that MAXIMIZES y*x.			*/
		nf = find_large_weight_forest (comp, y, edge_freq, forest, params, tp);

		z = 0.0;
		for (i = 0; i < nf; i++) {
			z += y [forest [i]];
		}

		if (z <= 1.0 + FUZZ) {
			/* All forests are now on the correct side of	*/
			/* hyperplane y*x = 1.  We are done!		*/
			break;
		}

#if USE_EUCLIDEAN_NORM
		/* See if it is OK to delete the slack forests from the LP. */
		ynorm2 = 0.0;
		for (i = 0; i < nedges; i++) {
			ynorm2 += y [i] * y [i];
		}
		dist2 = 1.0 - w;
		dist2 *= dist2;
		dist2 /= ynorm2;

		if (dist2 < (1.0 - 1.0e-10) * prev_dist2) {
			/* Delete all forests that are slack. */
			delete_slack (lp, slack, tp);
			prev_dist2 = dist2;
		}
		tp -> cut_violation (tp, "D^2", dist2);
#else
		/* See if it is OK to delete the slack forests from the LP. */
		if (w < (1.0 - 1.0e-10) * best_w) {
			/* Delete all forests that are slack. */
			delete_slack (lp, slack, tp);
			best_w = w;
		}
		tp -> cut_violation (tp, "W", w);
#endif

		/* Trace the forest we are adding. */
		tp -> add_forest (tp, nf, forest, z);

		add_forest_to_lp (lp, nf, forest);
	}

	/* Finish off trace output (e.g., partial line). */
	tp -> finish (tp);

	gst_channel_printf (print_solve_trace,
			    "\t%d cut LPs, %d forests, %d exact\n",
			    tp -> num_cv, tp -> num_af, tp -> num_exact);
	free (tp);

	/* See if the constraint is violated. */
	z = 0.0;
	for (i = 0; i < nedges; i++) {
		z += comp -> x [i] * y [i];
	}

	if (z <= 1.0 + FUZZ) {
		gst_channel_printf (print_solve_trace,
			"find_fcomp_cut failed\n");
		record_failed_fcomp (comp, bbip);
	}
	else {
		newcp = lift_constraint (comp, y, z, bbip);

		newcp -> next = cp;
		cp = newcp;
	}

	free ((char *) edge_freq);
	free ((char *) slack);
	free ((char *) forest);
	free ((char *) y);

#ifdef CPLEX
	_MYCPX_freeprob (&lp);

	free ((char *) lpmem.matval);
	free ((char *) lpmem.matind);
	free ((char *) lpmem.matcnt);
	free ((char *) lpmem.matbeg);
	free ((char *) lpmem.senx);
	free ((char *) lpmem.rhsx);
	free ((char *) lpmem.bdu);
	free ((char *) lpmem.bdl);
	free ((char *) lpmem.objx);
#endif

#ifdef LPSOLVE
	delete_lp (lp);
#endif

	return (cp);
}

/*
 * Make the initial LP instance for CPLEX (dual formulation).
 */

#if defined(CPLEX) AND FORMULATE_DUAL

	static
	LP_t *
make_fcomp_lp (

struct comp *		comp,
struct lpmem *		lpmem
)
{
int			i;
int			k;
int			nverts;
int			nedges;
int			mac;
int			mar;
int			macsz;
int			marsz;
int			matsz;
int			col;
int			objsen;
char *			senx;
int *			matbeg;
int *			matcnt;
int *			matind;
double *		objx;
double *		bdl;
double *		bdu;
double *		rhsx;
double *		matval;
LP_t *			lp;

	nverts	= comp -> num_verts;
	nedges	= comp -> num_edges;

	macsz = 32 * nedges;
	mac = nedges;

	/* Build the objective function... */
	objx = NEWA (macsz, double);
	for (i = 0; i < mac; i++) {
		objx [i] = 1.0;
	}
	objsen = _MYCPX_MIN;	/* Minimize */

	/* Build variable bound arrays... */
	bdl = NEWA (macsz, double);
	bdu = NEWA (macsz, double);
	for (i = 0; i < mac; i++) {
		bdl [i] = 0.0;
		bdu [i] = 1.0;
	}

	mar = nedges;
	marsz = mar;
	matsz = macsz * nverts;

	rhsx = NEWA (marsz, double);
	senx = NEWA (marsz, char);
	matbeg = NEWA (macsz, int);
	matcnt = NEWA (macsz, int);
	matind = NEWA (matsz, int);
	matval = NEWA (matsz, double);

	for (i = 0; i < mar; i++) {
		rhsx [i] = comp -> x [i];
		senx [i] = 'G';
	}

	/* Emit one column per 1-edge forest. */
	k = 0;
	col = 0;
	for (i = 0; i < nedges; i++) {
		matbeg [col] = k;
		matcnt [col] = 1;
		matind [k] = i;
		matval [k] = 1.0;
		++k;
		++col;
	}

	lp = _MYCPX_loadlp ("localcut",
			    mac,
			    mar,
			    objsen,
			    objx,
			    rhsx,
			    senx,
			    matbeg,
			    matcnt,
			    matind,
			    matval,
			    bdl,
			    bdu,
			    NULL,
			    macsz,
			    marsz,
			    matsz);

	FATAL_ERROR_IF (lp EQ NULL);

	lpmem -> objx		= objx;
	lpmem -> rhsx		= rhsx;
	lpmem -> senx		= senx;
	lpmem -> matbeg		= matbeg;
	lpmem -> matcnt		= matcnt;
	lpmem -> matind		= matind;
	lpmem -> matval		= matval;
	lpmem -> bdl		= bdl;
	lpmem -> bdu		= bdu;

	return (lp);
}

#endif

/*
 * Make the initial LP instance for CPLEX (primal formulation).
 */

#if defined(CPLEX) AND NOT FORMULATE_DUAL

	static
	LP_t *
make_fcomp_lp (

struct comp *		comp,
struct lpmem *		lpmem
)
{
int			i;
int			k;
int			nverts;
int			nedges;
int			mac;
int			mar;
int			macsz;
int			marsz;
int			matsz;
int			col;
int			objsen;
char *			senx;
int *			matbeg;
int *			matcnt;
int *			matind;
double *		objx;
double *		bdl;
double *		bdu;
double *		rhsx;
double *		matval;
LP_t *			lp;

	nverts	= comp -> num_verts;
	nedges	= comp -> num_edges;

	macsz = nedges;
	mac = nedges;

	/* Build the objective function... */
	objx = NEWA (macsz, double);
	for (i = 0; i < mac; i++) {
		objx [i] = comp -> x [i];
	}
	objsen = _MYCPX_MAX;	/* Maximize */

	/* Build variable bound arrays... */
	bdl = NEWA (macsz, double);
	bdu = NEWA (macsz, double);
	for (i = 0; i < mac; i++) {
		if (comp -> x [i] <= FUZZ) {
			bdl [i] = 0.0;
			bdu [i] = _MYCPX_INFBOUND;
		}
		else {
			bdl [i] = - _MYCPX_INFBOUND;
			bdu [i] = _MYCPX_INFBOUND;
		}
	}

	mar = nedges;
	marsz = 32 * nedges;
	matsz = marsz * nverts;

	rhsx = NEWA (marsz, double);
	senx = NEWA (marsz, char);
	matbeg = NEWA (macsz, int);
	matcnt = NEWA (macsz, int);
	matind = NEWA (matsz, int);
	matval = NEWA (matsz, double);

	for (i = 0; i < mar; i++) {
		rhsx [i] = 1.0;
		senx [i] = 'L';
	}

	/* Emit one row per 1-edge forest. */
	k = 0;
	col = 0;
	for (i = 0; i < nedges; i++) {
		matbeg [col] = k;
		matcnt [col] = 1;
		matind [k] = i;
		matval [k] = 1.0;
		++k;
		++col;
	}

	lp = _MYCPX_loadlp ("localcut",
			    mac,
			    mar,
			    objsen,
			    objx,
			    rhsx,
			    senx,
			    matbeg,
			    matcnt,
			    matind,
			    matval,
			    bdl,
			    bdu,
			    NULL,
			    macsz,
			    marsz,
			    matsz);

	FATAL_ERROR_IF (lp EQ NULL);

	lpmem -> objx		= objx;
	lpmem -> rhsx		= rhsx;
	lpmem -> senx		= senx;
	lpmem -> matbeg		= matbeg;
	lpmem -> matcnt		= matcnt;
	lpmem -> matind		= matind;
	lpmem -> matval		= matval;
	lpmem -> bdl		= bdl;
	lpmem -> bdu		= bdu;

	return (lp);
}

#endif

/*
 * Delete slack forests from the given CPLEX LP (primal formulation).
 */


#if defined(CPLEX) AND NOT FORMULATE_DUAL

	static
	void
delete_slack (

LP_t *			lp,
double *		slack,
struct LCTrace *	tp
)
{
int		i;
int		j;
int		nrows;
int		ncols;
int *		dflag;

	ncols = GET_LP_NUM_COLS (lp);
	nrows = GET_LP_NUM_ROWS (lp);

	dflag = NEWA (nrows, int);
	memset (dflag, 0, nrows * sizeof (dflag [0]));

	j = 0;
	for (i = ncols; i < nrows; i++) {
		if (slack [i] > FUZZ) {
			dflag [i] = 1;
			++j;
		}
	}

	if (j > 0) {
		i = _MYCPX_delsetrows (lp, dflag);
		FATAL_ERROR_IF (i NE 0);

		tp -> del_slack (tp, j, nrows - j);
	}
	free ((char *) dflag);
}

#endif

/*
 * Delete slack forests from the given lp_solve problem (dual formulation).
 */


#if defined(LPSOLVE) AND FORMULATE_DUAL

	static
	void
delete_slack (

LP_t *			lp,
double *		slack,
struct LCTrace *	tp
)
{
int		i;
int		j;
int		nrows;
int		ncols;

	ncols = GET_LP_NUM_COLS (lp);

	/* Grog!  This is totally gross... */

	j = 0;
	for (i = ncols - 1; i >= 0; i--) {
		if (slack [i] > FUZZ) {
			del_column (lp, i + 1);
			++j;
		}
	}

	if (j > 0) {
		tp -> del_slack (tp, j, ncols - j);
	}
}

#endif

/*
 * Delete slack forests from the given lp_solve problem (primal formulation).
 */


#if defined(LPSOLVE) AND NOT FORMULATE_DUAL

	static
	void
delete_slack (

LP_t *			lp,
double *		slack,
struct LCTrace *	tp
)
{
int		i;
int		j;
int		nrows;
int *		dflag;

	nrows = GET_LP_NUM_ROWS (lp);

	dflag = NEWA (nrows + 1, int);

	j = 0;
	dflag [0] = 0;
	for (i = 0; i < nrows; i++) {
		if (slack [i] > FUZZ) {
			dflag [i + 1] = 1;
			++j;
		}
		else {
			dflag [i + 1] = 0;
		}
	}

	if (j > 0) {
		delete_row_set (lp, dflag);

		tp -> del_slack (tp, j, nrows - j);
	}
	free ((char *) dflag);
}

#endif

/*
 * Add the given forest to the given CPLEX LP (dual formulation).
 */

#if defined(CPLEX) AND FORMULATE_DUAL

	static
	void
add_forest_to_lp (

LP_t *		lp,		/* IN - LP to add forest to */
int		nf,		/* IN - number of edges in forest */
int *		forest		/* IN - list of edges in forest */
)
{
int		j;
int		nedges;
int		cmatbeg;
int		status;
int *		cmatind;
double *	cmatval;
double		lowerbd;
double		upperbd;
double		cval;

	nedges = GET_LP_NUM_ROWS (lp);

	cmatval = NEWA (nedges, double);
	cmatind = NEWA (nedges, int);
	for (j = 0; j < nf; j++) {
		cmatval [j] = 1.0;
		cmatind [j] = forest [j];
	}

	lowerbd = 0.0;
	upperbd = 1.0;
	cval = 1.0;
	cmatbeg = 0;

	status = CPXaddcols (cplex_env,
			     lp,
			     1,
			     nf,
			     &cval,
			     &cmatbeg,
			     cmatind,
			     cmatval,
			     &lowerbd,
			     &upperbd,
			     NULL);

	FATAL_ERROR_IF (status NE 0);

	free ((char *) cmatind);
	free ((char *) cmatval);
}

#endif

/*
 * Add the given forest to the given CPLEX LP (primal formulation).
 */

#if defined(CPLEX) AND NOT FORMULATE_DUAL

	static
	void
add_forest_to_lp (

LP_t *		lp,		/* IN - LP to add forest to */
int		nf,		/* IN - number of edges in forest */
int *		forest		/* IN - list of edges in forest */
)
{
int		j;
int		nedges;
int		rmatbeg;
int		status;
int *		rmatind;
double *	rmatval;
double		rval;
char		sense;

	nedges = GET_LP_NUM_COLS (lp);

	rmatval = NEWA (nedges, double);
	rmatind = NEWA (nedges, int);
	for (j = 0; j < nf; j++) {
		rmatval [j] = 1.0;
		rmatind [j] = forest [j];
	}

	rval = 1.0;
	rmatbeg = 0;
	sense = 'L';

	status = _MYCPX_addrows (lp,
				 0,
				 1,
				 nf,
				 &rval,
				 &sense,
				 &rmatbeg,
				 rmatind,
				 rmatval,
				 NULL,
				 NULL);

	FATAL_ERROR_IF (status NE 0);

	free ((char *) rmatind);
	free ((char *) rmatval);
}

#endif

/*
 * Make the initial LP instance for lp_solve (dual formulation).
 */

#if defined(LPSOLVE) AND FORMULATE_DUAL

	static
	LP_t *
make_fcomp_lp (

struct comp *		comp,
struct lpmem *		lpmem
)
{
int			i;
int			nverts;
int			nedges;
int			ncols;
int			nrows;
short *			ctype;
int *			matbeg;
int *			matind;
double *		objx;
double *		rhs;
double *		matval;
LP_t *			lp;

	nverts	= comp -> num_verts;
	nedges	= comp -> num_edges;

	ncols = nedges;

	lp = make_lp (0, ncols);

	/* All variables (really dual variables) are 0-1 variables... */
	for (i = 1; i <= ncols; i++) {
		set_bounds (lp, i, 0.0, 1.0);
	}

	/* Minimize */
	set_minim (lp);

	/* Build the objective function... */
	objx = NEWA (ncols + 1, double);
	for (i = 1; i <= ncols; i++) {
		objx [i] = 1.0;
	}
#if 1
	inc_mat_space (lp, ncols + 1);
#endif
	set_obj_fn (lp, objx);
	free (objx);

	nrows = nedges;

	rhs = NEWA (nrows, double);
	ctype = NEWA (nrows, short);
	matbeg = NEWA (ncols + 1, int);
	matind = NEWA (nrows, int);
	matval = NEWA (nrows, double);

	for (i = 0; i < nrows; i++) {
		matbeg [i] = i;
		matind [i] = i;
		matval [i] = 1.0;
		rhs [i] = comp -> x [i];
		ctype [i] = RC_OP_GE;
	}
	matbeg [i] = i;

	add_rows (lp, 0, nrows, rhs, ctype, matbeg, matind, matval);

	free ((char *) matval);
	free ((char *) matind);
	free ((char *) matbeg);
	free ((char *) ctype);
	free ((char *) rhs);

	return (lp);
}

#endif

/*
 * Make the initial LP instance for lp_solve (primal formulation).
 */

#if defined(LPSOLVE) AND NOT FORMULATE_DUAL

	static
	LP_t *
make_fcomp_lp (

struct comp *		comp,
struct lpmem *		lpmem
)
{
int			i;
int			nedges;
int			ncols;
int			nrows;
short *			ctype;
int *			matbeg;
int *			matind;
double *		objx;
double *		rhs;
double *		matval;
LP_t *			lp;

	nedges	= comp -> num_edges;

	ncols = nedges;

	lp = make_lp (0, ncols);

	/* Set variable bounds... */
	for (i = 0; i < ncols; i++) {
		if (comp -> x [i] <= FUZZ) {
			/* Default 0 ... infinity bounds are OK. */
			continue;
		}
		/* We would really like -inf ... inf, but lp_solve does	*/
		/* not support unrestricted negative variables...	*/
		set_bounds (lp, i + 1, -128.0, 1.0e24);
	}

	/* Maximize */
	set_maxim (lp);

	/* Build the objective function... */
	objx = NEWA (ncols + 1, double);
	for (i = 0; i < ncols; i++) {
		objx [i + 1] = comp -> x [i];
	}
#if 1
	inc_mat_space (lp, ncols + 1);
#endif
	set_obj_fn (lp, objx);
	free (objx);

	nrows = nedges;

	rhs = NEWA (nrows, double);
	ctype = NEWA (nrows, short);
	matbeg = NEWA (ncols + 1, int);
	matind = NEWA (nrows, int);
	matval = NEWA (nrows, double);

	/* Emit one row per 1-edge forest. */

	for (i = 0; i < nedges; i++) {
		matbeg [i] = i;
		matind [i] = i;
		matval [i] = 1.0;
		rhs [i] = 1.0;
		ctype [i] = RC_OP_LE;
	}
	matbeg [i] = i;

	add_rows (lp, 0, nrows, rhs, ctype, matbeg, matind, matval);

	free ((char *) matval);
	free ((char *) matind);
	free ((char *) matbeg);
	free ((char *) ctype);
	free ((char *) rhs);

	return (lp);
}

#endif

/*
 * Add the given forest to the given lp_solve LP (dual formulation).
 */

#if defined(LPSOLVE) AND FORMULATE_DUAL

	static
	void
add_forest_to_lp (

LP_t *		lp,		/* IN - LP to add forest to */
int		nf,		/* IN - number of edges in forest */
int *		forest		/* IN - list of edges in forest */
)
{
int		j;
int		nedges;
double *	colvec;

	nedges = GET_LP_NUM_ROWS (lp);

	colvec = NEWA (nedges + 1, double);

	for (j = 0; j <= nedges; j++) {
		colvec [j] = 0.0;
	}

	colvec [0] = 1.0;	/* objective coefficient... */

	for (j = 0; j < nf; j++) {
		colvec [1 + forest [j]] = 1.0;
	}

	add_column (lp, colvec);

	free ((char *) colvec);

	set_bounds (lp, lp -> columns, 0.0, 1.0);
}

#endif

/*
 * Add the given forest to the given lp_solve LP (primal formulation).
 */

#if defined(LPSOLVE) AND NOT FORMULATE_DUAL

	static
	void
add_forest_to_lp (

LP_t *		lp,		/* IN - LP to add forest to */
int		nf,		/* IN - number of edges in forest */
int *		forest		/* IN - list of edges in forest */
)
{
int		j;
int		nedges;
double *	rowvec;

	nedges = GET_LP_NUM_COLS (lp);

	rowvec = NEWA (nedges + 1, double);

	for (j = 0; j <= nedges; j++) {
		rowvec [j] = 0.0;
	}

	for (j = 0; j < nf; j++) {
		rowvec [1 + forest [j]] = 1.0;
	}

	add_constraint (lp, rowvec, REL_LE, 1.0);

	free ((char *) rowvec);
}

#endif

/*
 * Lift the generated constraint back to one that is valid for
 * the entire problem.  This is almost trivial: each edge of the
 * original problem that projects down to an edge of the component
 * gets a copy of the component edge's coefficient.
 */

	static
	struct constraint *
lift_constraint (

struct comp *		comp,		/* IN - component with constraint */
double *		y,		/* IN - LHS of constraint */
double			z,		/* IN - LHS evaled @ x to separate */
struct bbinfo *		bbip		/* IN - branch-and-cut info */
)
{
int			i;
int			j;
int			k;
int			nverts;
int			nedges;
int			jmasks;
bitmap_t *		mask;
bitmap_t *		emasks;
bitmap_t *		bp1;
int *			orig_vnum;
int *			new_vnum;
int *			vp1;
int *			vp2;
struct gst_hypergraph *	cip;
struct rcoef *		rp;
struct rcoef *		newrp;
struct constraint *	newcp;
double			x;
gst_channel_ptr		print_solve_trace;

	print_solve_trace = bbip -> params -> print_solve_trace;

	nverts	= comp -> num_verts;
	nedges	= comp -> num_edges;

	/* First, multiply constraint by the smallest integer	*/
	/* that makes it purely integral.			*/

	for (i = 1; ; i++) {
		for (j = 0; ; j++) {
			if (j >= nedges) {
				goto mul_by_i;
			}
			x = y [j] * i;
			if (fabs (x - floor (x + 0.5)) > 1.0e-6) {
				/* Muptiplying by i does NOT yield an */
				/* integer.  Try next i value. */
				break;
			}
		}
	}

mul_by_i:

	x = i;
	z *= x;
	for (j = 0; j < nedges; j++) {
		y [j] *= x;
		y [j] = floor (y [j] + 0.5);
	}

#if 1
	if (print_solve_trace NE NULL) {
		gst_channel_printf (print_solve_trace, "\t");
		z = 0;
		for (j = 0; j < nedges; j++) {
			z += y [j] * comp -> x [j];
			if (y [j] EQ 0.0) continue;
			if (y [j] EQ 1.0) {
				gst_channel_printf (print_solve_trace, " +");
			}
			else if (y [j] EQ -1.0) {
				gst_channel_printf (print_solve_trace, " -");
			}
			else if (y [j] < 0) {
				gst_channel_printf (print_solve_trace, " - %g", -y [j]);
			}
			else {
				gst_channel_printf (print_solve_trace, " + %g", y [j]);
			}
			gst_channel_printf (print_solve_trace, " x");
			vp1 = comp -> everts [j];
			vp2 = comp -> everts [j + 1];
			while (vp1 < vp2) {
				gst_channel_printf (print_solve_trace, ",%d", *vp1++);
			}
		}
		gst_channel_printf (print_solve_trace, " <= %g (%g)\n", x, z);
	}
#endif

	/* Make a bitmask for each edge showing which vertices it contains. */

	jmasks = BMAP_ELTS (nverts);
	emasks = NEWA (nedges * jmasks, bitmap_t);
	memset (emasks, 0, nedges * jmasks * sizeof (bitmap_t));
	bp1 = emasks;
	for (i = 0; i < nedges; i++) {
		vp1 = comp -> everts [i];
		vp2 = comp -> everts [i + 1];
		while (vp1 < vp2) {
			j = *vp1++;
			SETBIT (bp1, j);
		}
		bp1 += jmasks;
	}

	cip = bbip -> cip;

	/* Determine the original vertex number of each vertex in the fcomp. */

	orig_vnum = NEWA (nverts, int);
	for (i = 0; i < nverts; i++) {
		orig_vnum [i] = comp -> rverts [i] [0];
	}

	new_vnum = NEWA (cip -> num_verts, int);
	for (i = 0; i < cip -> num_verts; i++) {
		new_vnum [i] = -1;
	}
	for (i = 0; i < nverts; i++) {
		new_vnum [orig_vnum [i]] = i;
	}

	rp = NEWA (cip -> num_edges + 1, struct rcoef);
	newcp = NEW (struct constraint);

	newcp -> next		= NULL;
	newcp -> iteration	= 0;
	newcp -> type		= CT_RAW;
	newcp -> mask		= (bitmap_t *) rp;

	mask = NEWA (jmasks, bitmap_t);

	for (i = 0; i < cip -> num_edges; i++) {
		memset (mask, 0, jmasks * sizeof (bitmap_t));
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		k = 0;
		while (vp1 < vp2) {
			j = *vp1++;
			j = new_vnum [j];
			if (j >= 0) {
				SETBIT (mask, j);
				++k;
			}
		}
		if (k <= 0) continue;
		for (j = 0; j < nedges; j++) {
			if (_gst_is_equal (mask, &emasks [j * jmasks], jmasks)) {
				if (y [j] NE 0.0) {
					rp -> var = RC_VAR_BASE + i;
					rp -> val = y [j];
					++rp;
				}
				break;
			}
		}
	}
	rp -> var = RC_OP_LE;
	rp -> val = x;
	++rp;

	/* Copy the constraint into a right-sized buffer. */

	k = (rp - ((struct rcoef *) (newcp -> mask)));

	newrp = NEWA (k, struct rcoef);
	memcpy (newrp, newcp -> mask, k * sizeof (*newrp));
	free ((char *) (newcp -> mask));
	newcp -> mask = ((bitmap_t *) newrp);

	free ((char *) mask);
	free ((char *) orig_vnum);
	free ((char *) new_vnum);
	free ((char *) emasks);

	return (newcp);
}

/*
 * See if the given fractional component has been tried before,
 * with no success.  When this happens, we save time by not trying
 * it over again.
 */

	static
	bool
is_failed_subproblem (

struct comp *		comp,
struct bbinfo *		bbip
)
{
int			i;
int			nverts;
int			nedges;
int			total_card;
struct comp *		p;
int *			vp1;
int *			vp2;
double			delta;

	nverts		= comp -> num_verts;
	nedges		= comp -> num_edges;
	total_card	= comp -> everts [nedges] - comp -> everts [0];

	for (p = bbip -> failed_fcomps; p NE NULL; p = p -> next) {
		if (p -> num_verts NE nverts) continue;
		if (p -> num_edges NE nedges) continue;
		vp1 = p -> everts [0];
		vp2 = p -> everts [nedges];
		if ((vp2 - vp1) NE total_card) continue;
		for (i = 0; i < nedges; i++) {
			delta = fabs (p -> x [i] - comp -> x [i]);
			if (delta > FUZZ) goto mismatch;
		}
		vp2 = comp -> everts [0];
		for (i = 0; i < total_card; i++) {
			if (vp1 [i] NE vp2 [i]) goto mismatch;
		}
		/* This problem matches a failed one! */
		return (TRUE);
mismatch:	;
	}

	return (FALSE);
}

/*
 * Record the given fractional component in the list of those
 * that have FAILED to find a violated constraint.
 */

	static
	void
record_failed_fcomp (

struct comp *		comp,		/* IN - component that failed */
struct bbinfo *		bbip
)
{
struct comp *		comp2;

	/* Make a copy of the component... */
	comp2 = copy_fsubcomp (comp, NULL, NULL);

	comp2 -> next = bbip -> failed_fcomps;
	bbip -> failed_fcomps = comp2;
}

/*
 * Find any forest x of the given component whose total cost is > 1
 * (i.e., cost*x > 1).  Try a fast heuristic first.  If no such
 * forest is found, then use the entire branch-and-cut recursively
 * to find a forest x that is an optimal maximum of cost*x.
 *
 * The number of edges in the forest is returned.
 */

	static
	int
find_large_weight_forest (

struct comp *		comp,		/* IN - component to find forest in */
double *		cost,		/* IN - edge costs */
int *			edge_freq,	/* IN - edge use frequencies */
int *			forest,		/* OUT - edges in max weight forest */
gst_param_ptr		params,		/* IN - algorithm parameters */
struct LCTrace *	tp		/* IN - LCTrace object */
)
{
int			i;
int			nf;
double			z;

	/* Use a fast heuristic first. */

	nf = max_forest_heuristic (comp, cost, edge_freq, forest);

	z = 0.0;
	for (i = 0; i < nf; i++) {
		z += cost [forest [i]];
	}

	if (z <= 1.0 + FUZZ) {
		/* The heuristic forest is NOT good enough for our purposes. */
		/* Use the branch-and-cut recursively. */

		/* Display message that we are using the Exact Algorithm. */
		tp -> exact (tp);

		nf = find_max_forest (comp, cost, forest, params);
	}

	return (nf);
}

/*
 * A fast heuristic to look for forests having large cost according
 * to the given set of edge costs.
 */

	static
	int
max_forest_heuristic (

struct comp *		comp,		/* IN - component to find forest in */
double *		cost,		/* IN - edge costs */
int *			edge_freq,	/* IN - edge use frequencies */
int *			forest		/* OUT - edges in max weight forest */
)
{
int			i;
int			j;
int			k;
int			nverts;
int			nedges;
int			nf;
int			best_nf;
int *			index;
int *			temp_edges;
int *			temp_forest;
bool *			used;
double			z;
double			best_z;

	nverts = comp -> num_verts;
	nedges = comp -> num_edges;

	index = sort_edges_by_cost_ratio (comp, cost, edge_freq);

	temp_edges = NEWA (nedges, int);
	temp_forest = NEWA (nverts - 1, int);

	used = NEWA (nedges, bool);
	memset (used, FALSE, nedges * sizeof (used [0]));

	best_nf = -1;
	best_z = -DBL_MAX;

	for (i = 0; i < nedges; i++) {
		j = index [i];
		if (used [j]) {
			/* We have already seen a forest using edge j. */
			continue;
		}
#if 1
		if (cost [j] <= FUZZ) break;
#else
		if (cost [j] < -FUZZ) break;
#endif

		/* Create copy of list with edge j placed first. */
		temp_edges [0] = j;
		k = 1;
		for (j = 0; j < i; j++) {
			temp_edges [k++] = index [j];
		}
		for (j = i + 1; j < nedges; j++) {
			temp_edges [k++] = index [j];
		}

		z = max_forest_kruskal (comp,
					temp_edges,
					used,
					cost,
					&nf,
					temp_forest);

		if ((nf > 1) AND (z > best_z)) {
			best_nf = nf;
			best_z = z;
			for (j = 0; j < best_nf; j++) {
				forest [j] = temp_forest [j];
			}
		}
	}

	/* Put the edges of the forest in order... */

	_gst_sort_ints (forest, best_nf);

	/* Update edge use frequencies. */
	for (i = 0; i < best_nf; i++) {
		j = forest [i];
		++(edge_freq [j]);
	}

	free ((char *) used);
	free ((char *) temp_forest);
	free ((char *) temp_edges);
	free ((char *) index);

	return (best_nf);
}

/*
 * Sort edges of the given component into DECREASING order by:
 *
 *	1.	cost [edge] / (edge_size [edge] - 1),
 *	2.	edge use frequency,
 *	3.	edge_size [edge],
 *	4.	edge index.
 */

	static
	int *
sort_edges_by_cost_ratio (

struct comp *		comp,		/* IN - component to find forest in */
double *		cost,		/* IN - edge costs */
int *			edge_freq	/* IN - edge use frequencies */
)
{
int			i;
int			j;
int			k;
int			n;
int			i1;
int			i2;
int *			index;

	n = comp -> num_edges;

	index = NEWA (n, int);
	for (i = 0; i < n; i++) {
		index [i] = i;
	}

	/* Construct the heap via sift-downs, in O(n) time. */
	for (k = n >> 1; k >= 0; k--) {
		j = k;
		for (;;) {
			i = (j << 1) + 1;
			if (i + 1 < n) {
				/* Increment i (to right subchild of j) */
				/* if the right subchild is less. */
				i1 = index [i];
				i2 = index [i + 1];
				if (compare_edge_ratios (comp, cost, edge_freq, i1, i2) > 0) {
					++i;
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			i1 = index [j];
			i2 = index [i];
			if (compare_edge_ratios (comp, cost, edge_freq, i1, i2) <= 0) {
				/* Parent is <= least child, */
				/* Sift-down is done. */
				break;
			}
			/* Sift down and continue. */
			index [j] = i2;
			index [i] = i1;
			j = i;
		}
	}

	/* Now do actual sorting.  Exchange first/last and sift down. */
	while (n > 1) {
		/* Smallest is at index [0], swap with index [n-1],	*/
		/* thereby putting it into final position.		*/
		--n;
		i = index [0];
		index [0] = index [n];
		index [n] = i;

		/* Now restore the heap by sifting index [0] down. */
		j = 0;
		for (;;) {
			i = (j << 1) + 1;
			if (i + 1 < n) {
				/* Increment i (to right subchild of j) */
				/* if the right subchild is less. */
				i1 = index [i];
				i2 = index [i + 1];
				if (compare_edge_ratios (comp, cost, edge_freq, i1, i2) > 0) {
					++i;
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			i1 = index [j];
			i2 = index [i];
			if (compare_edge_ratios (comp, cost, edge_freq, i1, i2) <= 0) {
				/* Parent is <= least child, */
				/* Sift-down is done. */
				break;
			}
			/* Sift down and continue. */
			index [j] = i2;
			index [i] = i1;
			j = i;
		}
	}

	return (index);
}

/*
 * Compare the cost ratios of two edges.
 */

	static
	int
compare_edge_ratios (

struct comp *		comp,		/* IN - component to find forest in */
double *		cost,		/* IN - edge costs */
int *			edge_freq,	/* IN - edge use frequencies */
int			e1,		/* IN - first edge number */
int			e2		/* IN - second edge number */
)
{
int			esize1;
int			esize2;
double			z;

	esize1 = comp -> everts [e1 + 1] - comp -> everts [e1];
	esize2 = comp -> everts [e2 + 1] - comp -> everts [e2];

#if 1
	--esize1;
	--esize2;
#endif

	z = cost [e1] * esize2 - cost [e2] * esize1;

	if (z < -FUZZ) return (-1);
	if (z > FUZZ) return (1);

#if 0
	/* Want less frequently used edges first. */
	if (edge_freq [e1] < edge_freq [e2]) return (1);
	if (edge_freq [e1] > edge_freq [e2]) return (-1);
#endif

	if (esize1 < esize2) return (1);	/* Want smaller edges first. */
	if (esize1 > esize2) return (-1);

	return (e2 - e1);	/* Edges equivalent.  Make sort be stable. */
}

/*
 * Use Kruskal's algorithm (extended for hypergraphs) to greedily construct
 * a forest from the given sequence of edges.  Stop when the forest has
 * the largest weight we can obtain from the given ordering.
 */

	static
	double
max_forest_kruskal (

struct comp *		comp,		/* IN - comp to build forest for */
int *			edge_list,	/* IN - ordered list of edges */
bool *			used,		/* IN - marks edges used (ever) */
double *		cost,		/* IN - edge costs */
int *			nf,		/* OUT - number of edges in forest */
int *			forest		/* OUT - edges in forest */
)
{
int			i;
int			j;
int			e;
int			nverts;
int			nedges;
int *			ep1;
int *			ep2;
int *			ep3;
int *			vp1;
int *			vp2;
int *			vp3;
int *			vp4;
int *			roots;
bool *			mark;
int			components;
double			z;
struct dsuf		sets;

	nverts		= comp -> num_verts;
	nedges		= comp -> num_edges;

	/* Initialize one disjoint subtree for each vertex. */
	_gst_dsuf_create (&sets, nverts);
	for (i = 0; i < nverts; i++) {
		_gst_dsuf_makeset (&sets, i);
	}

	mark = NEWA (nverts, bool);
	for (i = 0; i < nverts; i++) {
		mark [i] = FALSE;
	}
	roots = NEWA (nverts, int);

	/* Start the greedy Kruskal procedure... */
	components = nverts;
	z = 0.0;
	ep1 = forest;
	ep2 = edge_list;
	ep3 = edge_list + nedges;
	while (components > 1) {
		if (ep2 >= ep3) {
			/* Nothing more we could add to the forest. */
			break;
		}
		e = *ep2++;
		if ((cost [e] <= FUZZ) AND (ep1 >= &forest [2])) {
			/* This edge and all that follow it would only	*/
			/* reduce the cost of the forest.  Stop now.	*/
			break;
		}
		vp3 = roots;
		vp1 = comp -> everts [e];
		vp2 = comp -> everts [e + 1];
		for (;;) {
			if (vp1 >= vp2) {
				/* No cycle!  Include e in solution! */
				*ep1++ = e;
				z += cost [e];
				used [e] = TRUE;
				/* Unite all subtrees joined, and clear	*/
				/* out the mark bits...			*/
				vp4 = roots;
				i = *vp4++;
				mark [i] = FALSE;
				while (vp4 < vp3) {
					j = *vp4++;
					mark [j] = FALSE;
					_gst_dsuf_unite (&sets, i, j);
					i = _gst_dsuf_find (&sets, i);
					--components;
				}
				break;
			}
			j = _gst_dsuf_find (&sets, *vp1++);
			if (mark [j]) {
				/* This FST would form a cycle. */
				while (roots < vp3) {
					mark [*--vp3] = FALSE;
				}
				break;
			}
			/* This FST vertex is in distinct subtree... */
			mark [j] = TRUE;
			*vp3++ = j;
		}
	}

	free ((char *) roots);
	free ((char *) mark);
	_gst_dsuf_destroy (&sets);

	*nf = (ep1 - forest);

	return (z);
}

#if 1
/*
 * Find the maximum-weight forest within the given component, using
 * the given edge weights.  We solve this by adding 2-edges of zero
 * weight (if necessary so that a spanning tree exists that is composed
 * entirely of 2-edges) and then looking for a MINIMUM-weight spanning
 * tree using the NEGATIVES of the given edge weights.  The resulting
 * MST in hypergraph problem is solved by recursing the entire
 * branch-and-cut!
 *
 * The number of edges in the forest is returned.
 *
 * BKN: This routine uses gst_hgmst which is a bit slower than a brute force
 * method because of extra copying of structures and the inability to set the
 * metric to rectilinear (which apparently would give better bounds)...
 */

	static
	int
find_max_forest (

struct comp *		comp,		/* IN - component to find forest in */
double *		cost,		/* IN - edge costs */
int *			elist,		/* OUT - edges in max weight forest */
gst_param_ptr		params
)
{
int			i;
int			j;
int			k;
int			res;
int			nverts;
int			nedges;
int			new_nedges;
int			nforest;
int			nfake;
int			nmstedges;
int *			fake;
int *			renum;
int *			orig_edge;
int *			vp1;
int *			vp2;
int *			vp3;
int *			edge_sizes;
int *			edges;
int *			mstedges;
double *		costs;
gst_param_ptr		cparams;

	nverts = comp -> num_verts;
	nedges = comp -> num_edges;

	/* Compute the set of vertices between which fake edges	*/
	/* are needed.  This means one vertex per connected	*/
	/* component in the subgraph consisting of all 2-edges.	*/
	/* We will add fake edges that connect the first of	*/
	/* these vertices to every other.			*/
	fake = NEWA (nverts, int);
	nfake = compute_fake_edge_verts (comp, cost, fake);
	--nfake;

	/* Count given edges that transfer to the new problem.  We omit	*/
	/* all edges having cost <= 0, since we can delete these from	*/
	/* any solution to obtain another solution of equal or better	*/
	/* cost.  Also tally total cardinality of retained edges.	*/

	orig_edge = NEWA (nedges, int);
	renum = NEWA (nedges, int);
	j = 0;
	k = 0;
	for (i = 0; i < nedges; i++) {
		if (cost [i] <= FUZZ) {
			renum [i] = -1;
		}
		else {
			renum [i] = j;
			orig_edge [j] = i;
			j++;
			k += comp -> everts [i + 1] - comp -> everts [i];
		}
	}
	new_nedges = j;

	costs		= NEWA (new_nedges + nfake, dist_t);
	edges		= NEWA (k + 2 * nfake, int);
	edge_sizes	= NEWA (new_nedges + nfake, int);

	vp1 = edges;
	for (i = 0; i < nedges; i++) {
		j = renum [i];
		if (j < 0) continue;
		vp2 = comp -> everts [i];
		vp3 = comp -> everts [i + 1];
		edge_sizes [j] = (vp3 - vp2);
		costs [j] = -cost [i];
		while (vp2 < vp3) {
			*vp1++ = *vp2++;
		}
	}
	free ((char *) renum);
	for (i = 0; i < nfake; i++) {
		edge_sizes [new_nedges + i] = 2;
		costs [new_nedges + i] = 0.0;
		*vp1++ = fake [0];
		*vp1++ = fake [1 + i];
	}

	free ((char *) fake);

	/* Setup parameters */
	cparams = gst_create_param (NULL);
	cparams -> local_cuts_mode	   = params -> local_cuts_mode;
	cparams -> local_cuts_max_vertices = params -> local_cuts_max_vertices;
	cparams -> local_cuts_max_edges	   = params -> local_cuts_max_edges;
	cparams -> local_cuts_max_depth	   = params -> local_cuts_max_depth - 1;

	/* Note: the backtrack search does not correctly handle		*/
	/* instances having edges with negative cost, which are		*/
	/* definitely present in this sub-problem.  So force the use	*/
	/* of *only* the branch-and-cut algorithm.			*/
	cparams -> solver_algorithm = GST_PVAL_SOLVER_ALGORITHM_BRANCH_AND_CUT;

	if (params -> local_cuts_trace_depth NE 0) {
		cparams -> local_cuts_trace_depth  = params -> local_cuts_trace_depth - 1;
		cparams -> print_solve_trace = params -> print_solve_trace;
	}

	/* Default parameters makes checkpoint_filename EQ NULL */

	mstedges = elist;
	res = gst_hgmst (nverts, new_nedges + nfake, edge_sizes, edges, costs,
			 NULL, &nmstedges, mstedges, NULL, cparams);
	FATAL_ERROR_IF (res NE 0);

	/* Reorganize list to only include the original edges */
	nforest = 0;
	for (i = 0; i < nmstedges; i++) {
		k = mstedges[i];
		if (k < new_nedges) {
			*elist++ = orig_edge [k];
			++nforest;
		}
		else
			break;
	}

	gst_free_param (cparams);
	free ((char *) edges);
	free ((char *) edge_sizes);
	free ((char *) costs);
	free ((char *) orig_edge);

	return (nforest);
}
#else
/*
 * Find the maximum-weight forest within the given component, using
 * the given edge weights.  We solve this by adding 2-edges of zero
 * weight (if necessary so that a spanning tree exists that is composed
 * entirely of 2-edges) and then looking for a MINIMUM-weight spanning
 * tree using the NEGATIVES of the given edge weights.  The resulting
 * MST in hypergraph problem is solved by recursing the entire
 * branch-and-cut!
 *
 * The number of edges in the forest is returned.
 */

	static
	int
find_max_forest (

struct comp *		comp,		/* IN - component to find forest in */
double *		cost,		/* IN - edge costs */
int *			elist,		/* OUT - edges in max weight forest */
gst_param_ptr		params
)
{
int			i;
int			j;
int			k;
int			nverts;
int			nedges;
int			new_nedges;
int			kmasks;
int			nmasks;
int			nforest;
int			nfake;
int *			fake;
int *			renum;
int *			orig_edge;
int *			vp1;
int *			vp2;
int *			vp3;
#if 0
struct bbinfo *		bbip;
#endif
struct gst_hypergraph *	cip;
double			z;
gst_solver_ptr		solver;
bitmap_t *		smt;
gst_channel_ptr		trace;

	nverts = comp -> num_verts;
	nedges = comp -> num_edges;

	trace = params -> print_solve_trace;

	/* Compute the set of vertices between which fake edges	*/
	/* are needed.  This means one vertex per connected	*/
	/* component in the subgraph consisting of all 2-edges.	*/
	/* We will add fake edges that connect the first of	*/
	/* these vertices to every other.			*/
	fake = NEWA (nverts, int);
	nfake = compute_fake_edge_verts (comp, cost, fake);
	--nfake;

	/* Count given edges that transfer to the new problem.  We omit	*/
	/* all edges having cost <= 0, since we can delete these from	*/
	/* any solution to obtain another solution of equal or better	*/
	/* cost.  Also tally total cardinality of retained edges.	*/

	orig_edge = NEWA (nedges, int);
	renum = NEWA (nedges, int);
	j = 0;
	k = 0;
	for (i = 0; i < nedges; i++) {
		if (cost [i] <= FUZZ) {
			renum [i] = -1;
		}
		else {
			renum [i] = j;
			orig_edge [j] = i;
			j++;
			k += comp -> everts [i + 1] - comp -> everts [i];
		}
	}
	new_nedges = j;

	kmasks = BMAP_ELTS (nverts);
	nmasks = BMAP_ELTS (new_nedges + nfake);

	cip = gst_create_hg (nverts, NULL, NULL);
	cip -> num_edges	= new_nedges + nfake;
	cip -> num_edge_masks	= nmasks;
	cip -> edge		= NEWA (new_nedges + nfake + 1, int *);
	cip -> edge_size	= NEWA (new_nedges + nfake, int);
	cip -> cost		= NEWA (new_nedges + nfake, dist_t);
#if 0
	cip -> metric		= PURE_GRAPH;
#else
	/* Pretend its rectilinear.  Since we have MST edges, */
	/* the upper bounding heuristic should work fine. */
	cip -> metric		= RECTILINEAR;
#endif
	cip -> integrality_delta	= 0.0;
	cip -> initial_edge_mask	= NEWA (nmasks, bitmap_t);
	cip -> required_edges		= NEWA (nmasks, bitmap_t);
	cip -> term_trees		= NULL;
	cip -> inc_edges		= NULL;
	cip -> description		= NULL;
/*	cip -> p1time			= _gst_get_cpu_time (); */
	cip -> pts			= NULL;
	cip -> full_trees		= NULL;

	vp1 = NEWA (k + 2 * nfake, int);
	for (i = 0; i < nedges; i++) {
		j = renum [i];
		if (j < 0) continue;
		cip -> edge [j] = vp1;
		vp2 = comp -> everts [i];
		vp3 = comp -> everts [i + 1];
		cip -> edge_size [j] = (vp3 - vp2);
		cip -> cost [j] = -cost [i];
		while (vp2 < vp3) {
			*vp1++ = *vp2++;
		}
	}
	free ((char *) renum);
	for (i = 0; i < nfake; i++) {
		cip -> edge [new_nedges + i] = vp1;
		cip -> edge_size [new_nedges + i] = 2;
		cip -> cost [new_nedges + i] = 0.0;
		*vp1++ = fake [0];
		*vp1++ = fake [1 + i];
	}
	cip -> edge [new_nedges + nfake] = vp1;

	free ((char *) fake);

	for (i = 0; i < nmasks; i++) {
		cip -> initial_edge_mask [i] = 0;
		cip -> required_edges [i] = 0;
	}
	k = new_nedges + nfake;
	for (i = 0; i < k; i++) {
		SETBIT (cip -> initial_edge_mask, i);
	}

	_gst_init_term_trees (cip);

	/* Default parameters makes checkpoint_filename EQ NULL */
	solver = gst_create_solver (cip, NULL, NULL);
	gst_hg_solve (solver);		/* Le grande recurs... */

	smt = solver -> solutions [0].edge_mask;
	nforest = 0;
	for (i = 0; i < new_nedges; i++) {
		if (BITON (smt, i)) {
			*elist++ = orig_edge [i];
			++nforest;
		}
	}

#if 0
	gst_channel_printf (trace, "Sub-problem:\n");
	for (i = 0; i < new_nedges; i++) {
		gst_channel_printf (trace, " Edge %d (%d):  ",
				    i, orig_edge [i]);
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			gst_channel_printf (trace, " %d", *vp1++);
		}
		gst_channel_printf (trace, "\tCost = %g\n", cip -> cost [i]);
	}
	for ( ; i < cip -> num_edges; i++) {
		gst_channel_printf (trace, " Fake Edge %d:  ", i);
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			gst_channel_printf (trace, " %d", *vp1++);
		}
		gst_channel_printf (trace, "\tCost = %g\n", cip -> cost [i]);
	}
	gst_channel_printf (trace, " Solution:");
	z = 0.0;
	for (i = 0; i < cip -> num_edges; i++) {
		if (NOT BITON (smt, i)) continue;
		gst_channel_printf (trace, " %d", i);
		z += cip -> cost [i];
	}
	gst_channel_printf (trace, ", Z = %g\n", z);
#endif

	free ((char *) orig_edge);

	gst_free_solver (solver);
	gst_free_hg (cip);

	return (nforest);
}
#endif

/*
 * Choose one vertex per connected component in the subgraph consisting
 * of all 2-edges.  The caller will construct "fake" edges of weight 0
 * connecting the first of these vertices to each of the others.
 * These edges do not affect the objective function, and they guarantee
 * the existence of a spanning tree.  Finding a minimum spanning tree
 * in the resulting hypergraph, and then removing any such "fake" edges
 * from the solution yields a minimum weight forest.
 */

	static
	int
compute_fake_edge_verts (

struct comp *		comp,		/* IN - component to process */
double *		cost,		/* IN - cost of each edge */
int *			fake		/* OUT - fake vertices */
)
{
int		i;
int		j1;
int		j2;
int		k;
int		nverts;
int		nedges;
int		nfake;
int *		vp1;
int *		vp2;
struct dsuf	sets;

	nverts = comp -> num_verts;
	nedges = comp -> num_edges;

	_gst_dsuf_create (&sets, nverts);
	for (i = 0; i < nverts; i++) {
		_gst_dsuf_makeset (&sets, i);
	}

	/* Process all 2-edges. */
	for (i = 0; i < nedges; i++) {
		vp1 = comp -> everts [i];
		vp2 = comp -> everts [i + 1];
		k = (vp2 - vp1);
		if (k NE 2) continue;

		if (cost [i] <= FUZZ) {
			/* Ignore this 2-edge, since using it imposes	*/
			/* a penalty.  Must sprout a new zero-cost edge. */
			continue;
		}

		j1 = _gst_dsuf_find (&sets, vp1 [0]);
		j2 = _gst_dsuf_find (&sets, vp1 [1]);
		if (j1 NE j2) {
			_gst_dsuf_unite (&sets, j1, j2);
		}
	}

	nfake = 0;
	for (i = 0; i < nverts; i++) {
		j1 = _gst_dsuf_find (&sets, i);
		if (j1 EQ i) {
			fake [nfake++] = i;
		}
	}

	_gst_dsuf_destroy (&sets);

	return (nfake);
}

/*
 * This routine computes a list of all non-trivial forests.
 */

	int
_gst_find_forests (

struct comp *		comp,		/* IN - component to get forests of */
bitmap_t **		flist,		/* OUT - list of forests */
gst_channel_ptr		print_solve_trace
)
{
int			i;
int			nverts;
int *			vbuf;
bool *			vflag;
bitmap_t *		fbuf;
struct ff		ff;

	nverts = comp -> num_verts;

	vbuf = NEWA (nverts, int);
	vflag = NEWA (nverts, bool);
	for (i = 0; i < nverts; i++) {
		vflag [i] = FALSE;
	}

	ff.comp		= comp;
	ff.vbuf		= vbuf;
	ff.vflag	= vflag;
	ff.nforests	= 0;
	ff.ptr		= NULL;

	/* Create a collection of sets, one per vertex. */
	_gst_ddsuf_create (&ff.sets, nverts);
	for (i = 0; i < nverts; i++) {
		_gst_ddsuf_makeset (&ff.sets, i);
	}

	ff_recurse (0, 0, nverts, 0, &ff);

	gst_channel_printf (print_solve_trace,
		"find_forests: Total of %d forests.\n", ff.nforests);

	/* Now that we know how many forests we get, allocate a buffer	*/
	/* to hold them all. */

	fbuf = NEWA (ff.nforests, bitmap_t);

	ff.ptr		= fbuf;
	ff.nforests	= 0;

	ff_recurse (0, 0, nverts, 0, &ff);

	_gst_ddsuf_destroy (&ff.sets);

	free ((char *) vflag);
	free ((char *) vbuf);

	*flist = fbuf;

	return (ff.nforests);
}

/*
 * Recursively enumerate all forests.  We define a forest to be any
 * non-empty subset of the edges that is acyclic.
 */

	static
	void
ff_recurse (

int		e,		/* IN - current edge to include/exclude */
int		ne,		/* IN - num edges in forest */
int		ncc,		/* IN - number of connected components */
bitmap_t	emask,		/* IN - set of edges in current forest */
struct ff *	fp		/* IN/OUT - global recursion info */
)
{
int		i;
int		j;
int		k;
int		nedges;
int		state;
int *		vp1;
int *		vp2;
int *		vp3;
struct comp *	comp;
bool		cyclic;

	comp = fp -> comp;

	nedges = comp -> num_edges;

	if (e >= nedges) return;
	if (ncc <= 1) return;

	/* Now include edge e -- but only if it causes no cycle. */
	state = _gst_ddsuf_get_state (&(fp -> sets));

	cyclic = FALSE;
	vp1 = comp -> everts [e];
	vp2 = comp -> everts [e + 1];
	k = (vp2 - vp1);
	if (ncc >= k) {
		vp3 = fp -> vbuf;
		while (vp1 < vp2) {
			j = *vp1++;
			/* Determine which connected component this	*/
			/* vertex is in.				*/
			j = _gst_ddsuf_find (&(fp -> sets), j);
			if (fp -> vflag [j]) {
				cyclic = TRUE;
				break;
			}
			fp -> vflag [j] = TRUE;
			*vp3++ = j;
		}

		if (cyclic) {
			/* Restore the vbuf. */
			while (vp3 > fp -> vbuf) {
				j = *--vp3;
				fp -> vflag [j] = FALSE;
			}
		}
		else {
			if (ne > 0) {
				/* This constitutes a new forest! */
				/* We ignore forests with 1 or fewer edges */
				/* because they produce vacuous or simple */
				/* bound constraints, which we handle */
				/* differently. */
				++(fp -> nforests);
				if (fp -> ptr NE NULL) {
					/* Second pass -- save edge set. */
					*(fp -> ptr)++ = emask | (1 << e);
				}
			}

			/* Unite the sets joined by edge e */
			i = *--vp3;
			fp -> vflag [i] = FALSE;
			while (vp3 > fp -> vbuf) {
				j = *--vp3;
				fp -> vflag [j] = FALSE;
				_gst_ddsuf_unite (&(fp -> sets), i, j);
				i = _gst_ddsuf_find (&(fp -> sets), i);
			}

			/* Recurse, including edge e. */
			ff_recurse (e + 1,
				    ne + 1,
				    ncc - k + 1,
				    emask | (1 << e),
				    fp);
		}

		_gst_ddsuf_restore (&(fp -> sets), state);
	}

	/* Now exclude the edge. */
	ff_recurse (e + 1, ne, ncc, emask, fp);
}

/*
 * Print the entire list of forests in readable form.
 */

	void
_gst_print_forests (

struct comp *		comp,		/* IN - component owning forests */
bitmap_t *		flist,		/* IN - list of forests */
int			n,		/* IN - number of forests */
gst_channel_ptr		print_solve_trace
)
{
int			i;
int			j;
int			nedges;
int *			vp1;
int *			vp2;
bitmap_t		mask;

	gst_channel_printf (print_solve_trace, "List of all forests:\n");

	nedges = comp -> num_edges;

	for (i = 0; i < n; i++) {
		gst_channel_printf (print_solve_trace, "\t");
		mask = flist [i];
		for (j = 0; j < nedges; j++, mask >>= 1) {
			if ((mask & 1) EQ 0) continue;
			gst_channel_printf (print_solve_trace, " ");
			vp1 = comp -> everts [j];
			vp2 = comp -> everts [j + 1];
			while (vp1 < vp2) {
				gst_channel_printf (print_solve_trace, ",%d", *vp1++);
			}
		}
		gst_channel_printf (print_solve_trace, "\n");
	}
}

/*
 * Factory method for making LCTrace objects.
 */

	static
	struct LCTrace *
make_tracer (

struct bbinfo *		bbip	/* IN: branch-and-bound info */
)
{
gst_param_ptr		params;
gst_channel_ptr		trace;
struct LCTrace *	tp;

	params	= bbip -> params;
	trace	= params -> print_solve_trace;

	tp = NULL;
	if (trace EQ NULL) {
		/* Be completely silent. */
		tp = make_quiet_tracer (bbip);
	}
	else {
		switch (params -> local_cuts_trace_style) {
		case GST_PVAL_LOCAL_CUTS_TRACE_STYLE_NORMAL:
			/* The "new normal" is very terse output. */
			tp = make_terse_tracer (bbip);
			break;

		case GST_PVAL_LOCAL_CUTS_TRACE_STYLE_VERBOSE:
			/* Verbose gets us the "classic" trace output. */
			tp = make_classic_tracer (bbip);
			break;

		default:
			FATAL_ERROR;
			break;
		}
	}

	return (tp);
}

/*
 * Make an LCTrace object that is completely quiet.
 */

	static
	struct LCTrace *
make_quiet_tracer (

struct bbinfo *		bbip	/* IN: branch-and-bound-info */
)
{
struct LCTrace *	tp;

	tp = NEW (struct LCTrace);
	memset (tp, 0, sizeof (*tp));

	tp -> bbip		= bbip;
	tp -> del_slack		= quiet_trace_del_slack;
	tp -> cut_violation	= quiet_trace_cut_violation;
	tp -> add_forest	= quiet_trace_add_forest;
	tp -> exact		= quiet_trace_exact;
	tp -> lp_error		= quiet_trace_lp_error;
	tp -> finish		= quiet_trace_finish;

	return (tp);
}

/*
 * Quiet version of the "del slack" message.
 */

	static
	void
quiet_trace_del_slack (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
int			ndel,	/* IN: Num forests deleted */
int			nleft	/* IN: Num forests left */
)
{
}

/*
 * Quiet version of "cut violation" message.
 */

	static
	void
quiet_trace_cut_violation (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
const char *		var,	/* IN: Variable name */
double			val	/* IN: Variable value */
)
{
	/* Just count them. */
	++(tp -> num_cv);
}

/*
 * Quiet version of "add forest" message.
 */

	static
	void
quiet_trace_add_forest (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
int			nf,	/* IN: Number of edges in forest */
const int *		forest,	/* IN: List of edges in forest */
double			z	/* IN: Objective value of forest */
)
{
	/* Just count them. */
	++(tp -> num_af);
}

/*
 * Quiet version of "Exact algorithm" message.
 */

	static
	void
quiet_trace_exact (

struct LCTrace *	tp	/* IN/OUT: LCTrace object */
)
{
	/* Just count them. */
	++(tp -> num_exact);
}

/*
 * Quiet version of "LP solver error" message.
 */

	static
	void
quiet_trace_lp_error (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
const char *		format,	/* IN: Format string for error message */
int			status	/* IN: Status value to display */
)
{
	/* Do nothing! */
}

/*
 * Quiet version of the "finish" routine.
 */

	static
	void
quiet_trace_finish (

struct LCTrace *	tp	/* IN/OUT: LCTrace object */
)
{
}

/*
 * Make an LCTrace object that is very terse.
 */

	static
	struct LCTrace *
make_terse_tracer (

struct bbinfo *		bbip	/* IN: branch-and-bound-info */
)
{
struct LCTrace_terse *	tp;

	/* We need a derived class to hold additional data. */
	tp = NEW (struct LCTrace_terse);
	memset (tp, 0, sizeof (*tp));

	tp -> base.bbip			= bbip;
	tp -> base.del_slack		= terse_trace_del_slack;
	tp -> base.cut_violation	= terse_trace_cut_violation;
	tp -> base.add_forest		= terse_trace_add_forest;
	tp -> base.exact		= terse_trace_exact;
	tp -> base.lp_error		= terse_trace_lp_error;
	tp -> base.finish		= terse_trace_finish;

	/* Display one dot for every 10 forests added. */
	tp -> f_per_dot	= 10;

	return ((struct LCTrace *) tp);
}

/*
 * Terse version of the "del slack" message.
 */

	static
	void
terse_trace_del_slack (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
int			ndel,	/* IN: Num forests deleted */
int			nleft	/* IN: Num forests left */
)
{
}

/*
 * Terse version of "cut violation" message.
 */

	static
	void
terse_trace_cut_violation (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
const char *		var,	/* IN: Variable name */
double			val	/* IN: Variable value */
)
{
struct LCTrace_terse *	p;

	/* Access derived class data. */
	p = (struct LCTrace_terse *) tp;

	++(p -> base.num_cv);
}

/*
 * Terse version of "add forest" message.
 */

	static
	void
terse_trace_add_forest (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
int			nf,	/* IN: Number of edges in forest */
const int *		forest,	/* IN: List of edges in forest */
double			z	/* IN: Objective value of forest */
)
{
struct LCTrace_terse *	p;
gst_channel_ptr		trace;

	/* Access derived class data. */
	p = (struct LCTrace_terse *) tp;

	++(p -> base.num_af);

	++(p -> fcount);
	if (p -> fcount >= p -> f_per_dot) {
		/* Time to output the next '.' or 'x' char. */
		trace = p -> base.bbip -> params -> print_solve_trace;
		if (p -> numdots EQ 0) {
			/* Start of new line.  Indent it. */
			gst_channel_printf (trace, "\t");
		}
		gst_channel_printf (trace, "%c", p -> dotchar);
		p -> fcount	= 0;
		p -> dotchar	= '.';
		++(p -> numdots);
		if (p -> numdots >= 50) {
			/* End the current line of dots. */
			gst_channel_printf (trace, "\n");
			p -> numdots = 0;
		}
	}
}

/*
 * Terse version of "Exact algorithm" message.
 */

	static
	void
terse_trace_exact (

struct LCTrace *	tp	/* IN/OUT: LCTrace object */
)
{
struct LCTrace_terse *	p;

	/* Access derived class data. */
	p = (struct LCTrace_terse *) tp;

	++(p -> base.num_exact);

	p -> dotchar = 'x';	/* Indicate Exact Algorithm has been used. */
}

/*
 * Terse version of "LP solver error" message.
 */

	static
	void
terse_trace_lp_error (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
const char *		format,	/* IN: Format string for error message */
int			status	/* IN: Status value to display */
)
{
struct LCTrace_terse *	p;
gst_channel_ptr		trace;

	/* Access derived class data. */
	p = (struct LCTrace_terse *) tp;

	trace = p -> base.bbip -> params -> print_solve_trace;

	if (p -> numdots > 0) {
		/* Finish current line of output. */
		gst_channel_printf (trace, "\n");
	}

	/* Now display the error message. */
	gst_channel_printf (trace, format, status);

	/* Resume trace at new line of input. */
	p -> fcount	= 0;
	p -> numdots	= 0;
	p -> dotchar	= '.';
}

/*
 * Terse version of the "finish" routine.
 */

	static
	void
terse_trace_finish (

struct LCTrace *	tp	/* IN/OUT: LCTrace object */
)
{
struct LCTrace_terse *	p;
gst_channel_ptr		trace;

	/* Access derived class data. */
	p = (struct LCTrace_terse *) tp;

	trace = p -> base.bbip -> params -> print_solve_trace;

	if (p -> fcount > 0) {
		/* We have a "partial" dot.  Finish it. */
		gst_channel_printf (trace, "%c", p -> dotchar);
		++(p -> numdots);
	}
	if (p -> numdots > 0) {
		/* Finish current line of output. */
		gst_channel_printf (trace, "\n");
	}
}

/*
 * Make an LCTrace object that produces the verbose "classic" output.
 */

	static
	struct LCTrace *
make_classic_tracer (

struct bbinfo *		bbip	/* IN: branch-and-bound-info */
)
{
struct LCTrace *	tp;

	tp = NEW (struct LCTrace);
	memset (tp, 0, sizeof (*tp));

	tp -> bbip		= bbip;
	tp -> del_slack		= classic_trace_del_slack;
	tp -> cut_violation	= classic_trace_cut_violation;
	tp -> add_forest	= classic_trace_add_forest;
	tp -> exact		= classic_trace_exact;
	tp -> lp_error		= classic_trace_lp_error;
	tp -> finish		= classic_trace_finish;

	return (tp);
}

/*
 * Classic version of the "del slack" message.
 */

	static
	void
classic_trace_del_slack (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
int			ndel,	/* IN: Num forests deleted */
int			nleft	/* IN: Num forests left */
)
{
gst_channel_ptr		trace;

	trace = tp -> bbip -> params -> print_solve_trace;

	gst_channel_printf (trace,
			    "\tDeleted %d slack forests, %d left.\n",
			    ndel, nleft);
}

/*
 * Classic version of "cut violation" message.
 */

	static
	void
classic_trace_cut_violation (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
const char *		var,	/* IN: Variable name */
double			val	/* IN: Variable value */
)
{
gst_channel_ptr		trace;

	++(tp -> num_cv);

	trace = tp -> bbip -> params -> print_solve_trace;

	gst_channel_printf (trace, "\t\t\t\t%s = %.24g\n", var, val);
}

/*
 * Classic version of "add forest" message.
 */

	static
	void
classic_trace_add_forest (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
int			nf,	/* IN: Number of edges in forest */
const int *		forest,	/* IN: List of edges in forest */
double			z	/* IN: Objective value of forest */
)
{
int			i;
gst_channel_ptr		trace;

	++(tp -> num_af);

	trace = tp -> bbip -> params -> print_solve_trace;

	gst_channel_printf (trace, "Adding forest:");
	for (i = 0; i < nf; i++) {
		gst_channel_printf (trace, " %d", forest [i]);
	}
	gst_channel_printf (trace, " (z = %.24g)\n", z);
}

/*
 * Classic version of "Exact algorithm" message.
 */

	static
	void
classic_trace_exact (

struct LCTrace *	tp	/* IN/OUT: LCTrace object */
)
{
gst_channel_ptr		trace;

	++(tp -> num_exact);

	trace = tp -> bbip -> params -> print_solve_trace;

	gst_channel_printf (trace, "\tUsing Exact algorithm.\n");
}

/*
 * Classic version of "LP solver error" message.
 */

	static
	void
classic_trace_lp_error (

struct LCTrace *	tp,	/* IN/OUT: LCTrace object */
const char *		format,	/* IN: Format string for error message */
int			status	/* IN: Status value to display */
)
{
gst_channel_ptr		trace;

	trace = tp -> bbip -> params -> print_solve_trace;

	gst_channel_printf (trace, format, status);
}

/*
 * Classic version of the "finish" routine.
 */

	static
	void
classic_trace_finish (

struct LCTrace *	tp
)
{
}
