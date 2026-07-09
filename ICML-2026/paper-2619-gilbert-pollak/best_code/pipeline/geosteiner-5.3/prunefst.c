/***********************************************************************

	$Id: prunefst.c,v 1.51 2023/04/03 17:12:24 warme Exp $

	File:	prunefst.c
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 2000, 2023 by Martin Zachariasen.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Pruning of Euclidean and rectilinear FSTs using method
	originally proposed by Fossmeier & Kaufmann.
	Use implementation similar to the one suggested by Althaus,
	but with significant improvements in the compatibility tests.

************************************************************************

	Modification Log:

	a-1:	11/22/2000	martinz
		: Created.
	c-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Removed global variables.
		: Uses parameters.
		: Split off main function to prunefstmain.c.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Use better encapsulation for time conversions.
		: Fix -Wall issues.  Upgrade fatals.
	e-4:	11/18/2022	warme
		: Fixed -Wall issues.
	e-5:	04/03/2023	warme
		: Added check for final FSTs invalid.

************************************************************************/

#include "bb.h"
#include "bsd.h"
#include "bmst.h"
#include "btsearch.h"
#include "dsuf.h"
#include "emptyr.h"
#include "emst.h"
#include "fatal.h"
#include <float.h>
#include "geosteiner.h"
#include "greedy.h"
#include "incompat.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "metric.h"
#include "mst.h"
#include "parmblk.h"
#include "point.h"
#include "prepostlude.h"
#include "rmst.h"
#include "solver.h"
#include "steiner.h"
#include <string.h>

/*
 * Global Routines
 */

gst_hg_ptr		gst_hg_prune_edges (gst_hg_ptr, gst_param_ptr, int *);

/*
 * Local Macros
 */

#ifndef MIN
 #define MIN(a,b)	(((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
 #define MAX(a,b)	(((a) > (b)) ? (a) : (b))
#endif

#define INCOMPAT_STEP_SIZE		8

/*
 * Local Types
 */

struct pg_edge {
	dist_t				len; /* Length of edge */
	int				fst; /* Corresponding FST */
	int				p1;  /* First vertex */
	int				p2;  /* Second vertex */
};

struct clt_info {
	dist_t				dist;
	int				term;
	int				aterm1;
	int				aterm2;
};

struct pinfo {
	struct gst_hypergraph *		cip;
	int				num_pg_edges;
	int				num_pg_verts;
	struct pg_edge *		pg_edges;
	int *				steiner_index;
	struct clt_info **		clt_info;
	int *				clt_count;
	bitmap_t *			compat_mask;
	bitmap_t *			locally_subopt_mask;
	double				eps;
	double				integrality_delta;
	gst_param_ptr			params;
	gst_param_ptr			upper_bound_params;
	gst_metric_ptr			metric;
};

struct incompat {
	struct incompat *		next;
	int				fst;
};

struct bc3 {
	struct gst_hypergraph *
			cip;		/* problem data */
	int		kmasks;		/* size of vert_mask */
	int		nmasks;		/* size of edge_mask */
	int *		dfs;		/* DFS number of each vertex */
	int *		low;		/* lowest DFS num in component */
	int *		parent;		/* parents of vertices in DFS tree */
	int		max_stack;	/* size of stack */
	int *		stack;		/* base-address of edge stack */
	int *		sp;		/* current stack pointer */
	int		counter;	/* DFS number generator */
	bitmap_t *	bcc_vmask;	/* scratch buffer for new BCCs */
	bitmap_t *	bcc_emask;	/* scratch buffer for new BCCs */
	bitmap_t *	edges_seen;	/* edges already pushed */
	int *		bcc_vlist;	/* scratch vertex list buffer */
	int *		degree;		/* temp vertex degree counter */
	int *		made_req;	/* caller's list of required edges */
	int		req_count;	/* cur index into made-req */
};


/*
 * Local Routines
 */

static void		add_incompat (struct incompat **, int, int, int *);
static int		bcc_find_required (struct gst_hypergraph *,
					   int *,
					   int);
static void		bcc3 (struct bc3 *, int);
static void		check_for_invalid_fsts (struct gst_hypergraph * H);
static int		comp_ints (const void *, const void *);
static int		comp_pg_edges (const void *, const void *);
static int		comp_clt (const void *, const void *);
static void		compute_incompatibility_info (struct pinfo *,
						      struct bsd *,
						      int,
						      int,
						      bool *,
						      bool *);
static void		compute_pruning_info (struct gst_hypergraph *,
					      struct bsd *,
					      struct pinfo *);
static void		init_upper_bound_params (struct pinfo *);
static bool		passes_upper_bound_tests (struct pinfo *, struct bsd *, int, int,
						  int *, struct pset *, bitmap_t *, int *, bitmap_t *);
static void		process_bcc3 (struct bc3 *, int *, int *);
static void		prune_fsts (struct gst_hypergraph *,
				    struct bsd *,
				    double,
				    gst_param_ptr);
static bool		prune_this_fst (struct gst_hypergraph *,
					struct pinfo *,
					int);
static dist_t		terminal_edge_distance (struct gst_hypergraph *,
						struct point *,
						struct point *,
						struct point *,
						dist_t *,
						dist_t *);
static void		test_close_terminal (struct pinfo *, struct bsd *, struct full_set *,
					     int, int, struct clt_info **);
static void		zap_deleted_fsts (struct gst_hypergraph *);


/*
 *
 */
	gst_hg_ptr
gst_hg_prune_edges (

gst_hg_ptr	orig_hg,
gst_param_ptr	params,
int *		status
)
{
int			nedges;
int			code;
double			default_eps;
char			buf1 [32];
struct bsd *		BSD;
struct edge *		mst_edges;
bitmap_t *		empty_rect;
cpu_time_t		T0;
cpu_time_t		Tn;
cpu_time_t		Tzap;
gst_channel_ptr		timing;
gst_hg_ptr		H;
gst_metric_ptr		metric;
int			metric_type;
int			metric_parameter;
/* FIXME: The following declarations will probably not be needed later */
struct edge *		edges;
struct edge *		ep;
struct point *		p1;
struct point *		p2;
int			i;
int			j;

	GST_PRELUDE

	code = 0;

	H = NULL;

	_gst_begin_using_lp_solver ();

	H = gst_create_hg (NULL);
	gst_copy_hg (H, orig_hg);

	gst_get_hg_metric(H, &metric);
	metric_type = GST_METRIC_NONE;

	if (metric NE NULL) {
		gst_get_metric_info (metric, &metric_type, &metric_parameter);
	}

	FATAL_ERROR_IF (metric_type EQ GST_METRIC_NONE);

	if (params EQ NULL) {
		params = (gst_param_ptr) &_gst_default_parmblk;
	}
	timing = params -> detailed_timings_channel;

	T0 = _gst_get_cpu_time ();
	Tn = T0;

	/* Compute minimum spanning tree */

	mst_edges = NEWA (H -> pts -> n, struct edge);
	default_eps = (params -> eps_mult_factor) * DBL_EPSILON;
	nedges = 0;
	if (_gst_is_euclidean (H)) {
		nedges	= _gst_euclidean_mst (H -> pts, mst_edges);
	}
	else if (_gst_is_rectilinear (H)) {
		empty_rect = _gst_init_empty_rectangles (H -> pts, NULL);
		nedges	   = _gst_rect_mst (H -> pts, mst_edges, empty_rect);
		free ((char *) empty_rect);
	}
	else {
		/* General metric (FIXME: will be made more efficient later) */

		/* First make complete graph */
		nedges = (H -> pts -> n * (H -> pts -> n - 1))/2;
		edges = NEWA (nedges, struct edge);

		ep = edges;
		p1 = &(H -> pts -> a[0]);
		for (i = 0; i < H -> pts -> n; i++, p1++) {
			p2 = &(H -> pts -> a[0]);
			for (j = 0; j < i; j++, p2++, ep++) {
				ep -> p1 = i;
				ep -> p2 = j;
				ep -> len = gst_distance(metric, p1 -> x, p1 -> y, p2 -> x, p2 -> y);
			}
		}

		/* Then compute an MST */
		_gst_mst_edge_list (H -> pts -> n, nedges, edges, mst_edges);
		nedges = H -> pts -> n - 1;
		free ((char *) edges);
	}

	FATAL_ERROR_IF (nedges NE H -> pts -> n - 1);

	if (timing NE NULL) {
		_gst_convert_delta_cpu_time (buf1, &Tn);
		gst_channel_printf (timing, "Compute MST:            %s\n", buf1);
	}
	/* Compute bottleneck Steiner distances */

	BSD = _gst_compute_bsd (nedges, mst_edges, params -> bsd_method);
	free ((char *) mst_edges);

	if (timing NE NULL) {
		_gst_convert_delta_cpu_time (buf1, &Tn);
		gst_channel_printf (timing, "Compute BSD:            %s\n", buf1);
	}

	/* Prune FSTs */
	prune_fsts (H, BSD, default_eps, params);

	if (timing NE NULL) {
		_gst_convert_delta_cpu_time (buf1, &Tn);
		gst_channel_printf (timing, "Pruning FSTs:           %s\n", buf1);
	}

	/* Remove deleted FSTs permanently */
	zap_deleted_fsts (H);

	/* Measure zap time.  This also sets Tn so that Tn-T0 is   */
	/* the total processing time.				   */
	Tzap = _gst_get_delta_cpu_time (&Tn);

	if (timing NE NULL) {
		_gst_convert_cpu_time (Tzap, buf1);
		gst_channel_printf (timing, "Zap deleted FSTs:       %s\n", buf1);
		_gst_convert_cpu_time (Tn - T0, buf1);
		gst_channel_printf (timing, "Total:                  %s\n", buf1);
	}

	gst_set_dbl_property (H -> proplist,
			      GST_PROP_HG_PRUNING_TIME,
			      _gst_cpu_time_t_to_double_seconds (Tn - T0));

	/* Make sure that the final un-pruned FSTs span all terminals. */
	check_for_invalid_fsts (H);

	_gst_shutdown_bsd (BSD);
	_gst_stop_using_lp_solver ();

	if (status NE NULL) {
		*status = code;
	}

	GST_POSTLUDE
	return (H);
}

/*
 * Main pruning procedure. We use a method proposed by Fossmeier and Kaufmann
 * based on thesing whether is advantageous to extend a given FST with a
 * terminal not currently spanned. If so, the FST is discarded.
 */
	static
	void
prune_fsts (

struct gst_hypergraph *	cip,		/* IN/OUT - compatibility info */
struct bsd *		BSD,		/* IN	  - BSD data structure	*/
double			default_eps,	/* IN	  - default epsilon value */
gst_param_ptr		params		/* IN	  - parameters */
)
{
int			i, j, k, t, r1, r2;
int			tcomp;
int			fsave;
int			pruned_total;
int			required_total;
int			old_pruned_total;
int			scan;
int			nverts;
int			kmasks;
int			nedges;
int			nmasks;
int			adj_edge;
int			numinc;
int			min_pair;
int			max_pair;
int *			ep1;
int *			ep2;
int *			vp;
int *			vp1;
int *			vp2;
int *			comps_edge;
int *			made_req;
int			req_count;
bool			first_2edge;
bool			this_2edge;
bool			changed;
bool			all_pairs_tested;
struct dsuf		comps;
struct pinfo		pinfo;
int *			lvlist;
struct pset *		ltlist;
bitmap_t *		ltmask;
int *			lflist;
int *			inclist;
bitmap_t *		lfmask;
struct inc_info		inc_info;
gst_channel_ptr		timing;

	timing = params -> detailed_timings_channel;

	nverts = cip -> num_verts;
	kmasks = cip -> num_vert_masks;
	nmasks = cip -> num_edge_masks;
	nedges = cip -> num_edges;

	pinfo.cip		  = cip;
	pinfo.eps		  = default_eps;
	pinfo.params		  = params;
	pinfo.compat_mask	  = NEWA (nmasks, bitmap_t);
	pinfo.locally_subopt_mask = NEWA (nmasks, bitmap_t);
	memset (pinfo.compat_mask, 0, nmasks * sizeof (bitmap_t));
	memset (pinfo.locally_subopt_mask, 0, nmasks * sizeof (bitmap_t));
	gst_get_hg_metric(cip, &pinfo.metric);

	init_upper_bound_params (&pinfo);

	pinfo.integrality_delta = 0.0;
	gst_get_dbl_property (cip -> proplist,
			      GST_PROP_HG_INTEGRALITY_DELTA,
			      &(pinfo.integrality_delta));

	if (_gst_is_rectilinear (cip) AND (pinfo.integrality_delta EQ 1.0)) {
		/* Rectilinear metric with integral costs -- we can */
		/* use epsilon EQ zero! */
		pinfo.eps = 0.0;
	}

	/* The following are only used for calling upper bound procedure */
	lvlist	  = NEWA (nverts, int);
	ltlist	  = NEW_PSET (nverts);
	ltmask	  = NEWA (kmasks, bitmap_t);
	lflist	  = NEWA (nedges, int);
	lfmask	  = NEWA (nmasks, bitmap_t);
	inclist	  = NEWA (nedges, int);

	/* Initialize masks */
	for (i = 0; i < kmasks; i++) {
		ltmask [i] = 0;
	}
	for (i = 0; i < nmasks; i++) {
		lfmask [i] = 0;
	}

	_gst_startup_incompat_edges (&inc_info, cip);

	/* Perform thorough upper bound test for every not-yet pruned FST */
	pruned_total = 0;
	for (i = 0; i < nedges; i++) {

		if (BITON (cip -> initial_edge_mask, i)) {
			if (NOT passes_upper_bound_tests (&pinfo, BSD, i, -1,
							  lvlist, ltlist, ltmask, lflist, lfmask)) {
				CLRBIT (cip -> initial_edge_mask, i);
				SETBIT (pinfo.locally_subopt_mask, i);
				pruned_total++;
			}
			else {
				SETBIT (pinfo.compat_mask, i);
			}
		}
		else {
			pruned_total++;
		}
	}

	/* Compute basic compatibility */

	min_pair = 0;
	max_pair = INCOMPAT_STEP_SIZE;
	compute_incompatibility_info (&pinfo,
				      BSD,
				      min_pair,
				      max_pair,
				      &changed,
				      &all_pairs_tested);

	/* Compute pruning information */
	compute_pruning_info(cip, BSD, &pinfo);

	/* Build union-find structure */
	_gst_dsuf_create (&comps, cip -> num_verts);
	for (t = 0; t < cip -> num_verts; t++) {
		_gst_dsuf_makeset (&comps, t);
	}

	comps_edge = NEWA (cip -> num_verts, int);
	made_req   = NEWA (nedges, int);

	/* Perform actual pruning */
	if (timing) {
		char buf [32];
		_gst_convert_cpu_time (_gst_get_cpu_time (), buf);
		gst_channel_printf (timing,
			"- scan 0 finished. %6d FSTs pruned - %s\n",
			pruned_total, buf);
	}
	required_total = 0;
	scan = 1;
    for (;;) {
	for (; scan < nedges; scan++) {

		old_pruned_total   = pruned_total;
		for (i = 0; i < nedges; i++) {
			if ((BITON (cip -> initial_edge_mask, i)) AND
			   (NOT BITON (cip -> required_edges, i))) {

				/* Get list of incompatible edges */
				numinc = _gst_get_incompat_edges (inclist,
								  i,
								  &inc_info);

				/* Set up mask of compatible FSTs */
				for (k = 0; k < numinc; k++) {
					j = inclist [k];
					CLRBIT (pinfo.compat_mask, j);
				}

				/* Test if FST can be pruned */
				if (prune_this_fst(cip, &pinfo, i)) {
					CLRBIT (cip -> initial_edge_mask, i);
					CLRBIT (pinfo.compat_mask, i);
					pruned_total++;
				}

				/* Reset mask */
				for (k = 0; k < numinc; k++) {
					j = inclist [k];
					if (BITON (cip -> initial_edge_mask, j))
						SETBIT (pinfo.compat_mask, j);
				}
			}
		}

		/* Test if any connected component (initially one
		   for each terminal) only has one adjacent FST */

	try_again:
		req_count = 0;
		for (t = 0; t < cip -> num_verts; t++) {
			comps_edge [t] = -1;
		}

		/* First find the number of adjacent FSTs... */
		for (t = 0; t < cip -> num_verts; t++) {
			tcomp = _gst_dsuf_find (&comps, t); /* t's component */
			if (comps_edge [tcomp] EQ -2) continue;
			ep1 = cip -> term_trees [t];
			ep2 = cip -> term_trees [t + 1];
			while (ep1 < ep2) {
				i = *ep1++;
				if ((BITON (cip -> initial_edge_mask, i)) AND
				    (NOT BITON (cip -> required_edges, i))) {
					adj_edge = comps_edge [tcomp];
					if (adj_edge EQ -2) break;
					if (adj_edge EQ -1) {
						/* First FST */
						comps_edge [tcomp] = i;
					}
					else {
						/* Second (or more) FST */
						first_2edge = (cip -> edge_size [adj_edge] EQ 2);
						this_2edge  = (cip -> edge_size [i] EQ 2);

						/* If all adjacent edges have been 2-edges, */
						/* then pick the shortest (if equal then take edge */
						/* with minimum index) */

						if (first_2edge AND this_2edge) {
							if ((cip -> cost [i] < cip -> cost [adj_edge]) OR
							    ((cip -> cost [i] EQ cip -> cost [adj_edge]) AND
							     (i < adj_edge))) {
								comps_edge [tcomp] = i;
							}
						}
						else {
							comps_edge [tcomp] = -2; break;
						}
					}
				}
			}
		}

		/* ... then check the counts */
		for (t = 0; t < cip -> num_verts; t++) {
			fsave = comps_edge [t];
			if ((fsave >= 0) AND
			    (NOT BITON (cip -> required_edges, fsave))) {

				if (NOT BITON (cip -> initial_edge_mask, fsave)) {
					/* fatal: FST already removed */
					FATAL_ERROR;
				}

				SETBIT (cip -> required_edges, fsave);
				required_total++;
				made_req [req_count++] = fsave;
			}
		}

		i = req_count;
		req_count = bcc_find_required (cip, made_req, req_count);
		required_total += (req_count - i);

		/* Now update data structures for new required FSTs */
		for (j = 0; j < req_count; j++) {

			fsave = made_req [j];

			/* Unite vertices spanned */
			vp1 = cip -> edge [fsave];
			vp2 = cip -> edge [fsave + 1] - 1;
			while (vp1 < vp2) {
				r1 = _gst_dsuf_find (&comps, *vp1);
				r2 = _gst_dsuf_find (&comps, *vp2);

				if (r1 EQ r2) { /* fatal: cycle created */
					FATAL_ERROR;
				}
				_gst_dsuf_unite (&comps, r1, r2);
				vp1++;
			}

			/* Prune all incompatible FSTs */
			numinc = _gst_get_incompat_edges (inclist, fsave, &inc_info);
			for (k = 0; k < numinc; k++) {
				i = inclist [k];
				if (BITON (cip -> initial_edge_mask, i)) {

					if (BITON (cip -> required_edges, i)) {
						/* fatal: FST already required */
						FATAL_ERROR;
					}
					CLRBIT (cip -> initial_edge_mask, i);
					CLRBIT (pinfo.compat_mask, i);
					pruned_total++;
				}
			}
		}

		/* Remove FSTs making cycles among required FSTs */
		for (i = 0; i < nedges; i++) {
			if ((BITON (cip -> initial_edge_mask, i)) AND
			    (NOT BITON (cip -> required_edges, i))) {

				/* Check if a pair of vertices span the same component */
				vp1 = cip -> edge [i];
				vp2 = cip -> edge [i + 1];
				while (vp1 < vp2) {
					vp = vp1 + 1;
					while (vp < vp2) {
						r1 = _gst_dsuf_find (&comps, *vp1);
						r2 = _gst_dsuf_find (&comps, *vp);

						if (r1 EQ r2) { /* cycle created - remove FST */
							CLRBIT (cip -> initial_edge_mask, i);
							CLRBIT (pinfo.compat_mask, i);
							pruned_total++;
							vp = vp1 = vp2;
							break;
						}
						vp++;
					}
					vp1++;
				}
			}
		}

		if (req_count > 0) goto try_again;

		if (old_pruned_total EQ pruned_total) break;

		if (timing) {
			char buf [32];
			_gst_convert_cpu_time (_gst_get_cpu_time (), buf);
			gst_channel_printf (timing,
				"- scan %d finished. %6d FSTs pruned - %s\n",
				scan, pruned_total - old_pruned_total, buf);
		}
	}

	if (all_pairs_tested) break;

	/* Compute more of the incompatibility info. */
	do {
		min_pair = max_pair + 1;
		max_pair += INCOMPAT_STEP_SIZE;
		compute_incompatibility_info (&pinfo,
					      BSD,
					      min_pair,
					      max_pair,
					      &changed,
					      &all_pairs_tested);
	} while ((NOT changed) AND (NOT all_pairs_tested));

	if (NOT changed) break;

	/* We found new incompatibilities.  Prune those undecided ones	*/
	/* that are incompatible to a required FST.			*/

	for (fsave = 0; fsave < nedges; fsave++) {
		if (NOT BITON (cip -> required_edges, fsave)) continue;
		/* This FST is required.  Prune all incompatible FSTs. */
		numinc = _gst_get_incompat_edges (inclist, fsave, &inc_info);
		for (k = 0; k < numinc; k++) {
			i = inclist [k];
			if (BITON (cip -> initial_edge_mask, i)) {

				if (BITON (cip -> required_edges, i)) {
					/* fatal: FST already required */
					FATAL_ERROR;
				}
				CLRBIT (cip -> initial_edge_mask, i);
				CLRBIT (pinfo.compat_mask, i);
				pruned_total++;
			}
		}
	}
    }

	if (timing) {
		gst_channel_printf (timing, "- pruning finished: before: %d  after: %d  required: %d\n",
			nedges, nedges - pruned_total, required_total);
	}

	/* Free pruning data information... */

	_gst_shutdown_incompat_edges (&inc_info);

	for (i = 0; i < nedges; i++) {
		if (pinfo.clt_info[i] NE NULL) {
			free ((char *) pinfo.clt_info [i]);
		}
	}
	free ((char *) pinfo.clt_count);
	free ((char *) pinfo.clt_info);
	free ((char *) pinfo.steiner_index);
	free ((char *) pinfo.pg_edges);

	_gst_dsuf_destroy (&comps);

	free ((char *) made_req);
	free ((char *) comps_edge);

	free ((char *) inclist);
	free ((char *) lfmask);
	free ((char *) lflist);
	free ((char *) ltmask);
	free ((char *) ltlist);
	free ((char *) lvlist);

	gst_free_param (pinfo.upper_bound_params);

	free ((char *) pinfo.locally_subopt_mask);
	free ((char *) pinfo.compat_mask);
}

/*
 * Create special parameters for the exact upper bound routine that
 * are used for small subsets.
 */

	static
	void
init_upper_bound_params (

struct pinfo *		pip
)
{
gst_param_ptr		newp;

	newp = gst_create_param (NULL);
	gst_copy_param (newp, pip -> params);

	newp -> solver_algorithm	= GST_PVAL_SOLVER_ALGORITHM_AUTO;
	newp -> backtrack_max_verts	= 24;
	newp -> backtrack_max_edges	= 31;
	newp -> max_backtracks		= 10000;

	pip -> upper_bound_params = newp;
}

/*
 * Remove FSTs permanently that have been marked as not needed.
 * However, MST edges (2-terminal FSTs) are not deleted even if
 * marked as never needed.
 * It should be noted that no arrays are reallocated; data is packed in place.
 */
	static
	void
zap_deleted_fsts (

struct gst_hypergraph *	cip		/* IN/OUT - compatibility info */
)
{
int			i;
int			j;
int			k;
int			new_nedges;
int *			ni;
int *			vp;
int *			vp1;
int *			vp2;
int *			ep;
int *			ep1;
int *			ep2;

	/* First we count the number of FSTs that remain */
	/* and set up map from old to new edge index */

	new_nedges = 0;
	ni  = NEWA (cip -> num_edges, int);
	for (i = 0; i < cip -> num_edges; i++) {
		if ((BITON (cip -> initial_edge_mask, i)) OR (cip -> edge_size [i] EQ 2))
			ni[i] = new_nedges++;
		else
			ni[i] = -1;
	}

	/* Pack edge_size and cost arrays */
	for (i = 0; i < cip -> num_edges; i++) {
		if (ni[i] >= 0) {
			cip -> edge_size [ ni[i] ] = cip -> edge_size [i];
			cip -> cost	 [ ni[i] ] = cip -> cost [i];
		}
	}

	/* Pack edge array */
	vp = cip -> edge [0];
	for (i = 0; i < cip -> num_edges; i++) {
		if (ni[i] >= 0) {
			vp1 = cip -> edge [i];
			vp2 = cip -> edge [i + 1];
			cip -> edge [ ni[i] ] = vp;
			while (vp1 < vp2)
				*(vp++) = *(vp1++);
		}
	}
	cip -> edge [ new_nedges ] = vp;

	/* Pack term_trees array */
	ep = cip -> term_trees [0];
	for (j = 0; j < cip -> num_verts; j++) {
		vp1 = cip -> term_trees [j];
		vp2 = cip -> term_trees [j + 1];
		cip -> term_trees [j] = ep;
		while (vp1 < vp2) {
			i = *(vp1++);
			if (ni[i] >= 0)
				*(ep++) = ni[i];
		}
	}
	cip -> term_trees[ cip -> num_verts ] = ep;

	/* Pack inc_edges array */
	ep = cip -> inc_edges [0];
	for (i = 0; i < cip -> num_edges; i++) {
		if (ni[i] >= 0) {
			ep1 = cip -> inc_edges [i];
			ep2 = cip -> inc_edges [i + 1];
			cip -> inc_edges [ ni[i] ] = ep;
			while (ep1 < ep2) {
				k = *(ep1++);
				if (ni[k] >= 0)
					*(ep++) = ni [k];
			}
		}
	}
	cip -> inc_edges [ new_nedges ] = ep;

	/* Pack full_trees array */
	if (cip -> full_trees NE NULL) {
		for (i = 0; i < cip -> num_edges; i++) {
			if (ni[i] >= 0) {
				cip -> full_trees [ ni[i] ] = cip -> full_trees [i];
				cip -> full_trees [ ni[i] ] -> tree_num = ni[i];
			}
			else {
				/* Free FST data */
				_gst_free_full_set (cip -> full_trees [i]);
			}
		}
	}

	/* Pack bit maps */
	for (i = 0; i < cip -> num_edges; i++) {
		if (ni[i] >= 0) {
			if (BITON (cip -> initial_edge_mask, i))
				SETBIT (cip -> initial_edge_mask, ni[i]);
			else
				CLRBIT (cip -> initial_edge_mask, ni[i]);

			if (BITON (cip -> required_edges, i))
				SETBIT (cip -> required_edges, ni[i]);
			else
				CLRBIT (cip -> required_edges, ni[i]);
		}
	}

	/* Finally set edge count */
	cip -> num_edges      = new_nedges;
	cip -> num_edge_masks = BMAP_ELTS (new_nedges);

	free((char *) ni);
}

/*
 * Check that all of the final, un-pruned FSTs actually span all of
 * the terminals.  FATAL_ERROR if this is not the case.
 *
 * This does not happen very often, but this happens occasionally
 * on FSTs produced by the Uniform Directions FST generator.  This
 * FST generator does not have all of the same numeric stability
 * stuff that we added to the Euclidean FST generator.  The accumulated
 * numeric errors in these FSTs can trick this code into pruning too
 * many FSTs, which now trips this crash.
 *
 * We crash here because it is not good for the prunefst code to
 * terminate in an "apparently successful" manner, yet produce output
 * that is invalid.  These invalid FSTs currently cause crashes
 * downstream in the branch-and-cut code (when generating the initial
 * set of constraints): there is at least one vertex v for which the
 * constraint x({v}:V-{v}) >= 1 expands to 0 >= 1, and this FATALs
 * because of no non-zeros on the left-hand-side.  Crashing here
 * moves the crash closer to the source of the problem.
 */

	static
	void
check_for_invalid_fsts (

struct gst_hypergraph *	H	/* IN: Final pruned FSTs */
)
{
int		v, e, nverts;
bitmap_t *	edge_mask;
int **		term_trees;
int *		ep1;
int *		ep2;

	nverts		= H -> num_verts;
	edge_mask	= H -> initial_edge_mask;
	term_trees	= H -> term_trees;

	for (v = 0; v < nverts; v++) {
		/* Look for at least one un-pruned edge incident to	*/
		/* vertex v.						*/
		ep1	= term_trees [v];
		ep2	= term_trees [v + 1];
		do {
			/* There are no un-pruned edges incident to	*/
			/* vertex v!  The final FSTs are invalid!	*/
			FATAL_ERROR_IF (ep1 >= ep2);
			e = *ep1++;
		} while (NOT (BITON (edge_mask, e)));

		/* Vertex v has at least one incident un-pruned edge e.	*/
		/* Go on to the next vertex.				*/
	}

	/* The final un-pruned FSTs span all terminals. */
}

/*
 * Check if a given FST can be pruned
 */
	static
	bool
prune_this_fst (

struct gst_hypergraph *	cip,		/* IN - compatibility info */
struct pinfo *	pip,		/* IN - pruning data structure */
int		fst
)
{
int			i;
int			clt_count;
int			curr_clt;
int			root;
int			root1;
int			root2;
bool			prune_fst;
struct dsuf		comps;
struct pg_edge *	pg_edge;
struct clt_info *	clt;

	clt_count = pip -> clt_count [fst];
	if (clt_count == 0) return FALSE;

	/* Create disjoint set */
	_gst_dsuf_create (&comps, pip -> num_pg_verts + 1);
	for (i = 0; i < pip -> num_pg_verts + 1; i++)
		_gst_dsuf_makeset (&comps, i);

	/* Add pruning graph edges (in sorted order) */

	prune_fst = FALSE;
	curr_clt  = 0;
	clt = &(pip -> clt_info [fst][curr_clt]);
	for (i = 0; i < pip -> num_pg_edges; i++) {
		pg_edge = &(pip -> pg_edges[i]);
		while (pg_edge -> len > clt -> dist) {
			root  = _gst_dsuf_find (&comps, clt -> term);
			root1 = _gst_dsuf_find (&comps, clt -> aterm1);
			root2 = _gst_dsuf_find (&comps, clt -> aterm2);

			if ((root NE root1) AND (root NE root2)) {
				prune_fst = TRUE; /* This FST can be pruned! */
				goto prune_exit;
			}

			curr_clt++;
			if (curr_clt >= clt_count)
				goto prune_exit; /* This FST cannot be pruned... */
			clt = &(pip -> clt_info [fst][curr_clt]);
		}

		/* Add edge if FST is not deleted or incompatible */
		if (BITON (pip -> compat_mask, pg_edge -> fst)) {
			root1 = _gst_dsuf_find (&comps, pg_edge -> p1);
			root2 = _gst_dsuf_find (&comps, pg_edge -> p2);
			if (root1 NE root2)
				_gst_dsuf_unite (&comps, root1, root2);
		}

	}
	prune_exit:

	_gst_dsuf_destroy (&comps);
	return prune_fst;
}

/*
 * Add a pair of FSTs as incompatible
 */

	static
	void
add_incompat (

struct incompat **	incompat,	/* IN/OUT - incomp. data structure */
int			fst1,		/* IN	  - first FST */
int			fst2,		/* IN	  - second FST */
int*			counts		/* IN/OUT - incomp. counts */
)
{
struct incompat *	icp;

	icp = incompat [fst1];
	incompat [fst1] = NEW (struct incompat);
	incompat [fst1] -> fst	= fst2;
	incompat [fst1] -> next = icp;
	counts [fst1]++;
	counts [fst2]++;
}

/*
 * Computes for each FST, a list of those FSTs that are incompatible,
 * that is, fulfill one of the following conditions:
 * 1. The FSTs have one terminal in common and the BSD of their
 * terminals is shorter than the total length of the FSTs
 * 2. An heuristic tree spanning the terminals is shorter.
 * 3. Another pair of FSTs has shorther or equal length.
 * 4. The MSTHG for the FSTs within the terminals spanned
 *    together with BSD-MST edges is shorter.
 *
 * NOTE: We OMIT the trivial incompatibilities (FSTs that share
 *	 two or more terminals) from the list!
 */

	static
	void
compute_incompatibility_info (

struct pinfo *		pip,		/* IN/OUT - pruning info */
struct bsd *		BSD,		/* IN	  - BSD data structure	*/
int			min_pair,	/* IN	  - smallest pair to test */
int			max_pair,	/* IN	  - largest pair to test */
bool *			changed,	/* OUT	  - info was changed */
bool *			all_pairs_tested /* OUT	  - no more pairs to test */
)
{
int			i;
int			j;
int			k;
int			t;
int			fs;
int			common;
int			nverts;
int			nedges;
int			kmasks;
int			nmasks;
int			total;
int			isize;
int			jsize;
bitmap_t *		fmask;
bitmap_t *		lfmask;
bitmap_t *		edge_mask;
bitmap_t *		tmask;
bitmap_t *		ltmask;
bitmap_t *		incmask;
int *			lvlist;
struct pset *		ltlist;
struct incompat *	icp;
struct incompat *	icpn;
struct incompat **	incompat;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			flist;
int *			lflist;
int **			inc_edges;
int *			counts;
int **			ptrs;
struct gst_hypergraph *	cip;
gst_channel_ptr		timing;
gst_param_ptr		params;

	params	= pip -> params;
	timing	= params -> detailed_timings_channel;

	/* Initialize and allocate various variables and arrays */
	cip	  = pip -> cip;
	nverts	  = cip -> num_verts;
	kmasks	  = cip -> num_vert_masks;
	nedges	  = cip -> num_edges;
	nmasks	  = cip -> num_edge_masks;
	edge_mask = cip -> initial_edge_mask;

	inc_edges = NEWA (nedges + 1, int *);
	counts	  = NEWA (nedges, int);
	incompat  = NEWA (nedges, struct incompat *);
	flist	  = NEWA (nedges, int);
	fmask	  = NEWA (nmasks, bitmap_t);
	tmask	  = NEWA (kmasks, bitmap_t);
	lvlist	  = NEWA (nverts, int);
	ltlist	  = NEW_PSET (nverts);
	ltmask	  = NEWA (kmasks, bitmap_t);
	lflist	  = NEWA (nedges, int);
	lfmask	  = NEWA (nmasks, bitmap_t);
	incmask	  = NEWA (nmasks, bitmap_t);

	for (i = 0; i < kmasks; i++) {
		tmask [i] = 0;
		ltmask [i] = 0;
	}
	for (i = 0; i < nmasks; i++) {
		fmask [i] = 0;
		lfmask [i] = 0;
		incmask [i] = 0;
	}
	for (i = 0; i < nedges; i++) {
		incompat [i] = NULL;
		counts [i]   = 0;
	}

	/* Compute the list of (lists of) incomatible FSTs... */
	total = 0;

	if (timing) {
		char buf [32];
		_gst_convert_cpu_time (_gst_get_cpu_time (), buf);
		gst_channel_printf (timing,
			"- computing incompatible FSTs for each FST"
			" (size %d - %d) - %s\n",
			min_pair, max_pair, buf);
	}

	/* Copy existing incompatibilities, if any. */
	if (cip -> inc_edges NE NULL) {
		for (i = 0; i < nedges; i++) {
			ep1 = cip -> inc_edges [i];
			ep2 = cip -> inc_edges [i + 1];
			while (ep1 < ep2) {
				j = *ep1++;
				if (j <= i) continue;
				if (NOT BITON (edge_mask, j)) continue;
				add_incompat (incompat, i, j, counts);
				++total;
			}
		}
	}

	*changed = FALSE;
	*all_pairs_tested = TRUE;
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (edge_mask, i)) continue;
		if (cip -> edge_size [i] >= max_pair) {
			/* Must test this one later. */
			*all_pairs_tested = FALSE;
			continue;
		}

		/* Get mask of FSTs already known to be incompatible... */
		/* (All of these should be "non-basic".) */
		if (cip -> inc_edges NE NULL) {
			ep1 = cip -> inc_edges [i];
			ep2 = cip -> inc_edges [i + 1];
			while (ep1 < ep2) {
				j = *ep1++;
				if (NOT BITON (edge_mask, j)) continue;

				/* No need to test this pair! */
				SETBIT (incmask, j);
			}
		}

		/* Develop list of all FSTs adjacent to FST i... */
		k = 0;
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			t = *vp1++;
			SETBIT (tmask, t);
			ep1 = cip -> term_trees [t];
			ep2 = cip -> term_trees [t + 1];
			while (ep1 < ep2) {
				fs = *ep1++;
				if (BITON (fmask, fs)) continue;
				if (NOT BITON (edge_mask, fs)) continue;
				if (fs <= i) continue; /* test pairs once */
				SETBIT (fmask, fs);
				flist [k++] = fs;
			}
		}

		isize = cip -> edge_size [i] - 1;

		/* Now loop through all adjacent FSTs */
		ep1 = &flist [0];
		ep2 = &flist [k];
		while (ep1 < ep2) {
			fs = *ep1++;
			CLRBIT (fmask, fs);

			jsize = isize + cip -> edge_size [fs];
			if (jsize < min_pair) {
				/* Already tested this pair -- skip. */
				continue;
			}
			if (jsize > max_pair) {
				/* Must test this one later. */
				*all_pairs_tested = FALSE;
				continue;
			}

			if (BITON (incmask, fs)) {
				/* Already known to be incompatible. */
				/* Do not re-test. */
				continue;
			}

			/* Count number of vertices in common. */
			common = 0;
			vp1 = cip -> edge [fs];
			vp2 = cip -> edge [fs + 1];
			while (vp1 < vp2) {
				t = *vp1++;
				if (BITON (tmask, t)) {
					++common;
				}
			}
			if (common >= 2) {
				/* Too many - we don't record these... */
				continue;
			}

			/* One terminal in common. Perform thorough upper tests. */
			if (NOT passes_upper_bound_tests (pip, BSD, i, fs,
							  lvlist, ltlist, ltmask, lflist, lfmask)) {
				/* Did not pass - retain as incompatible */
				add_incompat(incompat, i, fs, counts);
				++total;
				*changed = TRUE;
			}
		}

		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			j = *vp1++;
			CLRBIT (tmask, j);
		}

		/* Reset mask of FSTs already known to be incompatible... */
		if (cip -> inc_edges NE NULL) {
			ep1 = cip -> inc_edges [i];
			ep2 = cip -> inc_edges [i + 1];
			while (ep1 < ep2) {
				j = *ep1++;
				CLRBIT (incmask, j);
			}
		}
	}

	/* Now allocate and copy into contiguous memory... */
	total *= 2;
	ep1 = NEWA (total, int);
	ptrs = NEWA (nedges, int *);
	for (i = 0; i < nedges; i++) {
		inc_edges [i]	= ep1;
		ptrs [i]	= ep1;
		ep1 += counts [i];
	}
	inc_edges [i] = ep1;

	FATAL_ERROR_IF (ep1 - inc_edges [0] NE total);

	for (i = 0; i < nedges; i++) {
		icp = incompat [i];
		while (icp NE NULL) {
			j = icp -> fst;		/* pair I < J is incompat */
			ep1 = (ptrs [i])++;
			*ep1++ = j;

			ep1 = (ptrs [j])++;
			*ep1++ = i;

			icpn = icp -> next;
			free ((char *) icp);
			icp = icpn;
		}
	}
	free ((char *) ptrs);
	for (i = 0; i < nedges; i++) {
		ep1 = inc_edges [i];
		ep2 = inc_edges [i + 1];
		qsort (ep1, ep2 - ep1, sizeof (*ep1), comp_ints);
	}
	if (cip -> inc_edges NE NULL) {
		if (cip -> inc_edges [0] NE NULL) {
			free ((char *) (cip -> inc_edges [0]));
		}
		free ((char *) (cip -> inc_edges));
	}
	cip -> inc_edges = inc_edges;

	/* Free allocated memory */

	free ((char *) incmask);
	free ((char *) lfmask);
	free ((char *) lflist);
	free ((char *) ltmask);
	free ((char *) ltlist);
	free ((char *) lvlist);
	free ((char *) tmask);
	free ((char *) fmask);
	free ((char *) flist);
	free ((char *) incompat);
	free ((char *) counts);
}

/*
 * Compute pruning information. For every FST identify a list
 * of "close" terminals and find their distance to the FST.
 */

	static
	void
compute_pruning_info (

struct gst_hypergraph *	cip,	/* IN	  - compatibility info */
struct bsd *		BSD,	/* IN	  - BSD info */
struct pinfo *		pip	/* IN/OUT - pruning data structure  */
)
{
int			i;
int			j;
int			k;
int			t;
int			total;
int			steiner_index;
int			nedges;
int			nverts;
int			kmasks;
int *			vp1;
int *			vp2;
bitmap_t *		tmask;
struct full_set *	fsp;
int *			tlist;
struct pset *		terms;
struct clt_info*	cli;
struct clt_info*	clip;
struct point *		p1;
struct point *		p2;
dist_t			l;
gst_param_ptr		params;

	params = pip -> params;
	nedges = cip -> num_edges;
	nverts = cip -> num_verts;
	kmasks = cip -> num_vert_masks;

	/* Terminal mask */
	tmask = NEWA (kmasks, bitmap_t);
	for (i = 0; i < kmasks; i++) {
		tmask [i] = 0;
	}

	/* Generate list of all edges in all FSTs */
	total = 0;
	for (i = 0; i < nedges; i++) {
		fsp = _gst_remove_degree_two_steiner_points (cip -> full_trees [i]);
		total += fsp -> nedges;
		_gst_free_full_set (fsp);
	}

	steiner_index = nverts;
	pip -> num_pg_edges	= total;
	pip -> pg_edges		= NEWA (total, struct pg_edge);
	memset (pip -> pg_edges, 0, total * sizeof (pip -> pg_edges [0]));
	pip -> steiner_index	= NEWA (nedges, int);
	k = 0;
	l = 0.0;
	for (i = 0; i < nedges; i++) {
		/* Remove corner points before using this FST...*/
		fsp = _gst_remove_degree_two_steiner_points (cip -> full_trees [i]);
		tlist = fsp -> tlist;
		terms = fsp -> terminals;
		pip -> steiner_index [i] = steiner_index;
		for (j = 0; j < fsp -> nedges; j++) {

			/* Compute length of this edge */
			p1 = (fsp -> edges[j].p1 < terms -> n)
			     ? &(terms -> a[ fsp -> edges[j].p1 ])
			     : &(fsp -> steiners -> a [fsp -> edges[j].p1 - terms -> n]);
			p2 = (fsp -> edges[j].p2 < terms -> n)
			     ? &(terms -> a[ fsp -> edges[j].p2 ])
			     : &(fsp -> steiners -> a [fsp -> edges[j].p2 - terms -> n]);
			l = gst_distance(pip -> metric, p1 -> x, p1 -> y, p2 -> x, p2 -> y) *
			    (1.0 - pip -> eps * ((double) cip -> edge_size [i]));

			pip -> pg_edges[k].fst = i;
			pip -> pg_edges[k].len = l;
			pip -> pg_edges[k].p1  =
				(fsp -> edges[j].p1 < terms -> n)
				? tlist [ fsp -> edges[j].p1 ]
				: fsp -> edges[j].p1 - terms -> n + steiner_index;
			pip -> pg_edges[k].p2  =
				(fsp -> edges[j].p2 < terms -> n)
				? tlist [ fsp -> edges[j].p2 ]
				: fsp -> edges[j].p2 - terms -> n + steiner_index;
			k++;
		}
		if (fsp -> steiners NE NULL) {
			steiner_index += fsp -> steiners -> n;
		}
		_gst_free_full_set (fsp);
	}
	pip -> num_pg_verts = steiner_index;

	/* Sort edge list */
	qsort(pip -> pg_edges, pip -> num_pg_edges,
	      sizeof(struct pg_edge), comp_pg_edges);

	/* Pruning graph has been constructed.
	   Now construct close terminal lists */

	pip -> clt_info	 = NEWA (nedges, struct clt_info *);
	pip -> clt_count = NEWA (nedges, int);
	cli = NEWA (nverts, struct clt_info);
	memset (cli, 0, nverts * sizeof (cli [0]));
	if (params -> detailed_timings_channel) {
		char buf [32];
		_gst_convert_cpu_time (_gst_get_cpu_time (), buf);
		gst_channel_printf (params -> detailed_timings_channel,
			"- constructing close terminal list for each FST"
			" - %s\n",
			buf);
	}
	for (i = 0; i < nedges; i++) {

		/* Mark terminals in current FST */
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			t = *vp1++;
			SETBIT (tmask, t);
		}

		/* Find all close terminals and add information
		   to close terminal data structure */
		fsp = _gst_remove_degree_two_steiner_points (cip -> full_trees [i]);
		clip = cli;
		for (t = 0; t < nverts; t++) {
			if (NOT BITON (tmask, t)) {
				test_close_terminal(pip, BSD, fsp, i, t, &clip);
			}
		}
		_gst_free_full_set (fsp);

		/* Unmark terminals in current FST */
		vp1 = cip -> edge [i];
		while (vp1 < vp2) {
			t = *vp1++;
			CLRBIT (tmask, t);
		}

		/* Sort close terminals */
		pip -> clt_count [i] = clip - cli;
		pip -> clt_info [i]  = NULL;
		if (pip -> clt_count [i] > 0) {
			qsort(cli, pip -> clt_count [i],
			      sizeof(struct clt_info), comp_clt);
			pip -> clt_info [i] = NEWA (pip -> clt_count [i],
						    struct clt_info);
			for (j = 0; j < pip -> clt_count [i]; j++)
				pip -> clt_info [i][j] = cli[j];
		}
	}

	free ((char *) cli);
	free ((char *) tmask);
}
/*
 * Computes upper bounds for single FST or pair of FSTs.
 * Returns TRUE if all upper bound tests are passed
 * and FALSE otherwise.
 */

	static
	bool
passes_upper_bound_tests (

struct pinfo *		pip,	/* IN - pruning info */
struct bsd *		BSD,	/* IN - BSD data structure */
int			fst1,	/* IN - first FST */
int			fst2,	/* IN - second FST */
int *			lvlist,	/* IN - local vertex list (should just be allocated) */
struct pset *		ltlist, /* IN - local terminal list (should just be allocated) */
bitmap_t *		ltmask, /* IN - local terminal mask (should just be cleared) */
int *			lflist, /* IN - local FST list (should just be allocated) */
bitmap_t *		lfmask	/* IN - local FST mask (should just be cleared) */
)
{
int			i;
int			j;
int			t;
int			hid;
int			nverts;
int			nedges;
int			lnedges;
int			lfs;
int			lfs2;
int			lsmtcount;
int			fstmaxind;
int			isterms;
int			status;
int			soln_status;
bool			all_spanned;
bool			shorter_pair_found;
struct edge *		bsdmst;
struct edge *		ep;
int *			hep;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			lterm;
int *			edges;
int *			edge_sizes;
double *		weights;
struct gst_hypergraph *	cip;
dist_t			l;
dist_t			bsdl;
dist_t			msthgl;
dist_t			minl;
dist_t			pairl;

	cip = pip -> cip;

	/* Construct list of terminals and set terminal mask */
	i = 0;
	vp1 = cip -> edge [fst1];
	vp2 = cip -> edge [fst1 + 1];
	while (vp1 < vp2) {
		t = *vp1++;
		lvlist [i] = t;
		ltlist -> a[i++] = cip -> pts -> a[t];
		SETBIT (ltmask, t);
	}

	if (fst2 >= 0) { /* negative means not defined */
		vp1 = cip -> edge [fst2];
		vp2 = cip -> edge [fst2 + 1];
		while (vp1 < vp2) {
			t = *vp1++;
			if (BITON (ltmask, t)) continue;
			lvlist [i] = t;
			ltlist -> a[i++] = cip -> pts -> a[t];
			SETBIT (ltmask, t);
		}
	}
	ltlist -> n = i;

	/* Reset terminal masks (in case we quit) */
	for (i = 0; i < ltlist -> n; i++) {
		CLRBIT (ltmask, lvlist [i]);
	}

	/* Lower bound on total length of these two FSTs (or this single FST)*/
	if (fst2 >= 0) {
		l = cip -> cost [fst1] + cip -> cost [fst2];
	}
	else {
		l = cip -> cost [fst1];
	}

	l *= (1.0 - pip -> eps * ((double) ltlist -> n));

	/* Compute BSD-MST */
	bsdmst = NEWA (ltlist -> n - 1, struct edge);
	if (_gst_bmst_terms (lvlist, ltlist -> n, BSD, bsdmst) NE ltlist -> n - 1) {
		FATAL_ERROR;
	}

	/* BSD-MST test... */
	bsdl = 0.0;
	ep   = bsdmst;
	for (i = 0; i < ltlist -> n - 1; i++, ep++) {
		bsdl += ep -> len;
	}
	if ((ltlist -> n > 3) AND (bsdl <= l)) {
		free ((char *) bsdmst);
		return FALSE;
	}

	/* Heuristic upper bound test (rectilinear) ... */
	if ((_gst_is_rectilinear (cip)) AND (ltlist -> n <= 15)) {
		if (_gst_kahng_robins_length (ltlist, l) < l) {
			free ((char *) bsdmst);
			return FALSE;
		}
	}

	/* Heuristic upper bound test (Euclidean) ... */
	if (_gst_is_euclidean (cip)) {
		/* FIXME -- use parameter to choose between SLL and greedy? */
		if (_gst_greedy_heuristic (ltlist, lvlist, BSD) < l) {
			free ((char *) bsdmst);
			return FALSE;
		}
	}

	/* If we are testing a pair of MST edges, we stop here */
	if (ltlist -> n EQ 3) {
		free ((char *) bsdmst);
		return TRUE;
	}

	/* Set terminal masks again */
	for (i = 0; i < ltlist -> n; i++) {
		SETBIT (ltmask, lvlist [i]);
	}

	/* Prepare for calling the branch-and-cut MSTHG procedure.	   */
	/* Construct list of FSTs spaning a subset of the given terminals. */
	/* We only need FSTs of cardinality 3 or larger.		   */

	lnedges = 0;
	for (i = 0; i < ltlist -> n; i++) {
		ep1 = cip -> term_trees [ lvlist [i] ];
		ep2 = cip -> term_trees [ lvlist [i] + 1 ];
		while (ep1 < ep2) {
			lfs = *ep1++;
			if (BITON (lfmask, lfs)) continue;
#if 1
			if (BITON (pip -> locally_subopt_mask, lfs)) continue;
#else
			if (NOT BITON (cip -> initial_edge_mask, lfs)) continue;
#endif
			if (cip -> edge_size [lfs] <= 2) continue;
			if ((lfs EQ fst1) AND (fst2 < 0)) continue; /* skip FST being tested */

			/* Does FST span a subset of given terminals? */
			all_spanned = TRUE;
			vp1 = cip -> edge [lfs];
			vp2 = cip -> edge [lfs + 1];
			while (vp1 < vp2) {
				t = *vp1++;
				if (NOT BITON (ltmask, t)) {
					all_spanned = FALSE;
					break;
				}
			}
			if (all_spanned) {
				SETBIT (lfmask, lfs);
				lflist [lnedges++] = lfs;
			}
		}
	}

	/* Reset terminal and FST masks */
	for (i = 0; i < ltlist -> n; i++) {
		CLRBIT (ltmask, lvlist [i]);
	}
	for (i = 0; i < lnedges; i++) {
		CLRBIT (lfmask, lflist [i]);
	}

	/* If no large FSTs then return (test cannot be performed) */
	if (lnedges EQ 0) {
		free ((char *) bsdmst);
		return TRUE;
	}

	/* Test if there is a PAIR of FSTs that has smaller or equal total length.   */
	/* In case the length is equal we only need to keep the "canonical"	     */
	/* par, i.e., for which the maximum index of (large) FSTs is minimized.	     */

	if (fst2 >= 0) {
		fstmaxind = -1;
		if ((cip -> edge_size [fst1] NE 2) AND (fst1 > fstmaxind)) fstmaxind = fst1;
		if ((cip -> edge_size [fst2] NE 2) AND (fst2 > fstmaxind)) fstmaxind = fst2;
	}
	else {
		fstmaxind = fst1;
	}

	nverts = ltlist -> n;
	shorter_pair_found = FALSE;
	for (i = 0; i < lnedges; i++) {
		lfs = lflist [i];

		/* Mark terminals in this FST */
		vp1 = cip -> edge [lfs];
		vp2 = cip -> edge [lfs + 1];
		while (vp1 < vp2) {
			t = *vp1++;
			SETBIT (ltmask, t);
		}

		if (cip -> edge_size [lfs] + 1 EQ nverts) {

			/* Find shortest BSD-MST edge to remaining terminal */
			ep   = bsdmst;
			minl = INF_DISTANCE;
			for (j = 0; j < nverts - 1; j++, ep++) {
				if ((NOT BITON (ltmask, lvlist [ep -> p1])) OR
				    (NOT BITON (ltmask, lvlist [ep -> p2]))) {
					if (ep -> len < minl) minl = ep -> len;
				}
			}
			pairl = cip -> cost [lfs] + minl;
			if  (pairl <  l)			shorter_pair_found = TRUE;
			if ((pairl <= l) AND (lfs < fstmaxind)) shorter_pair_found = TRUE;
		}
		else {
			/* Try to combine with another FST */
			for (j = i+1; j < lnedges; j++) {
				lfs2  = lflist [j];
				if (cip -> edge_size [lfs] + cip -> edge_size [lfs2] - 1 NE nverts) continue;
				pairl = cip -> cost [lfs] + cip -> cost [lfs2];
				if (pairl > l) continue;

				/* Count intersecting terminals */
				isterms = 0;
				vp1 = cip -> edge [lfs2];
				vp2 = cip -> edge [lfs2 + 1];
				while (vp1 < vp2) {
					t = *vp1++;
					if (BITON (ltmask, t)) isterms++;
				}
				if (isterms NE 1) continue; /* not spanning all terminals */

				if  (pairl <  l)	    shorter_pair_found = TRUE;
				if ((pairl <= l)       AND
				    (lfs  < fstmaxind) AND
				    (lfs2 < fstmaxind))	    shorter_pair_found = TRUE;
			}
		}

		/* Reset terminal mask */
		for (j = 0; j < nverts; j++) {
			CLRBIT (ltmask, lvlist [j]);
		}
		if (shorter_pair_found) break;
	}

	if (shorter_pair_found) {
		free ((char *) bsdmst);
		return FALSE;
	}

	/* Allocate structures needed for gst_hgmst */
	nedges =  lnedges + ltlist -> n - 1; /* we add BSD-MST edges */

	edges		= NEWA (cip -> edge[cip -> num_edges] -
				cip -> edge[0], int);
	edge_sizes	= NEWA (nedges, int);
	weights		= NEWA (nedges, dist_t);

	lterm		= NEWA (cip -> num_verts, int);

	for (i = 0; i < nverts; i++) {
		lterm [ lvlist [i] ] = i;
	}

	hid   = 0;
	hep   = edges;

	/* Add all identified FSTs to hyperedge list */
	for (i = 0; i < lnedges; i++) {
		lfs = lflist [i];

		/* Make it more attractive to seek an MST with many FSTs */
		weights [hid]	  = cip -> cost [lfs] -
				    pip -> integrality_delta / (double) nverts;
		edge_sizes [hid]  = cip -> edge_size [lfs];
		vp1 = cip -> edge [lfs];
		vp2 = cip -> edge [lfs + 1];
		while (vp1 < vp2) {
			*hep++ = lterm[ *vp1++ ];
		}
		hid++;
	}

	/* Add all BSD-MST edges to hyperedge list */
	ep = bsdmst;
	for (i = 0; i < ltlist -> n - 1; i++, ep++) {
		weights [hid]	  = ep -> len -
				    pip -> integrality_delta / (double) nverts;
		edge_sizes [hid]  = 2;
		*(hep++) = ep -> p1;
		*(hep++) = ep -> p2;
		hid++;
	}

	/* Upper bound target for HGMST algorithm */
	if (fst2 >= 0) {
		pip -> upper_bound_params -> upper_bound_target = l - 2 * pip -> integrality_delta;
	}
	else {
		pip -> upper_bound_params -> upper_bound_target = l - pip -> integrality_delta;
	}

	/* Finally, compute HGMST! */
	status = gst_hgmst (nverts, nedges, edge_sizes, edges, weights,
			    &msthgl, &lsmtcount, NULL, &soln_status,
			    pip -> upper_bound_params);

	FATAL_ERROR_IF (status NE 0);

	free ((char *) edges);
	free ((char *) edge_sizes);
	free ((char *) weights);
	free ((char *) lterm);
	free ((char *) bsdmst);

	/* Return result of final test */
	if (fst2 >= 0) {
		if (msthgl < l - pip -> integrality_delta) {
			return FALSE;
		}
		/* If equal length then there should be at least three FSTs */
		else if ((msthgl < l) AND (lsmtcount >= 3)) {
			return FALSE;
		}
	}
	else {
		if (msthgl <= l) {
			return FALSE;
		}
	}

	return TRUE; /* all tests passed */
}

/*
 * Find distance and attachment terminals for given terminal
 * to a specific FST
 */

	static
	void
test_close_terminal (

struct pinfo *		pip,	/* IN	  - pruning data structure  */
struct bsd *		BSD,	/* IN	  - BSD info */
struct full_set *	fsp,	/* IN	  - FST */
int			fst,	/* IN	  - FST index */
int			term,	/* IN	  - Terminal */
struct clt_info**	clip	/* IN/OUT - store close terminal info here */
)
{
int			j;
int			i1;
int			i2;
dist_t			d;
dist_t			d1;
dist_t			d2;
dist_t			pg_longest;
struct point *		pt;
struct point *		p1;
struct point *		p2;
int *			fsp_tlist;
struct pset *		fsp_terms;
struct pset *		fsp_steins;
struct gst_hypergraph *	cip;
struct clt_info		clt;

	cip	   = pip -> cip;
	pt	   = &(cip -> pts -> a [term]);
	fsp_tlist  = fsp -> tlist;
	fsp_terms  = fsp -> terminals;
	fsp_steins = fsp -> steiners;
	pg_longest = pip -> pg_edges [pip -> num_pg_edges-1].len;

	/* First a rough test to eliminate the terminal */
	if (EDIST(&(fsp_terms -> a[0]), pt) >
	    (fsp_terms -> n - 1) * pg_longest)
		return; /* this terminal is too far away */

	/* Now find the edge that is closest edge to this terminal */
	memset (&clt, 0, sizeof (clt));
	clt.term = term;
	clt.dist = INF_DISTANCE;
	for (j = 0; j < fsp -> nedges; j++) {

		/* Get coordinates and pruning graph indices for endpoints */
		if (fsp -> edges[j].p1 < fsp_terms -> n) {
			p1 = &(fsp_terms  -> a[ fsp -> edges[j].p1 ]);
			i1 = fsp_tlist [ fsp -> edges[j].p1 ];
		}
		else {
			p1 = &(fsp_steins -> a[ fsp -> edges[j].p1 -
						fsp_terms -> n ]);
			i1 = pip -> steiner_index [fst] +
				fsp -> edges[j].p1 - fsp_terms -> n;
		}

		if (fsp -> edges[j].p2 < fsp_terms -> n) {
			p2 = &(fsp_terms  -> a[ fsp -> edges[j].p2 ]);
			i2 = fsp_tlist [ fsp -> edges[j].p2 ];
		}
		else {
			p2 = &(fsp_steins -> a[ fsp -> edges[j].p2 -
						fsp_terms -> n ]);
			i2 = pip -> steiner_index [fst] +
				fsp -> edges[j].p2 - fsp_terms -> n;
		}

		/* Compute closest distance from terminal to edge */
		d = terminal_edge_distance(cip, pt, p1, p2, &d1, &d2) *
		    (1.0 + pip -> eps * ((double) cip -> edge_size [fst]));
		if (d < clt.dist) {
			clt.dist = d;
			clt.aterm1 = (d1 <= d) ? i1 : pip -> num_pg_verts;
			clt.aterm2 = (d2 <= d) ? i2 : pip -> num_pg_verts;
		}
	}

	/* Is distance smaller than longest edge in pruning graph? */
	if (clt.dist < pg_longest) {

		/* Is distance smaller longest BSD distance to FST terminal?
		   (this test may reduce the pruning effect but it significantly
		    reduces running time for clustered instances) */
		for (j = 0; j < fsp_terms -> n; j++) {
			if (clt.dist < 2.0 * _gst_bsd (BSD, fsp_tlist [j], term)) {

				*((*clip)++) = clt; /* save this terminal */
				return;
			}
		}
	}
}

/*
 * Distance from point to edge
 */

	static
	dist_t
terminal_edge_distance (

struct gst_hypergraph *	cip,	/* IN  - compatibility info */
struct point *		pt,	/* IN  - point */
struct point *		p1,	/* IN  - first edge point */
struct point *		p2,	/* IN  - second edge point */
dist_t *		d1,	/* OUT - distance from closest point to p1 */
dist_t *		d2	/* OUT - distance from closest point to p2 */
)
{
dist_t			d;
dist_t			l1;
dist_t			l2;
dist_t			l;
double			terms[6];
int			nsps;
double			sps[2];
int			nedges;
int			edges[6];


	d = 0.0;
	/* Construct SMT for the three points */

	terms[0] = pt -> x;
	terms[1] = pt -> y;
	terms[2] = p1 -> x;
	terms[3] = p1 -> y;
	terms[4] = p2 -> x;
	terms[5] = p2 -> y;
	gst_smt(3, terms, &l, &nsps, sps, &nedges, edges, NULL, cip -> metric, NULL);

	if (nsps EQ 1) {
		/* There was a Steiner point */

		d   = l - gst_distance(cip -> metric, p1 -> x, p1 -> y, p2 -> x, p2 -> y);
		*d1 = gst_distance(cip -> metric, p1 -> x, p1 -> y, sps[0], sps[1]);
		*d2 = gst_distance(cip -> metric, p2 -> x, p2 -> y, sps[0], sps[1]);
	}
	else {
		/* No Steiner point */
		l1 = gst_distance(cip -> metric, pt -> x, pt -> y, p1 -> x, p1 -> y);
		l2 = gst_distance(cip -> metric, pt -> x, pt -> y, p2 -> x, p2 -> y);

		if (l1 < l2) {
			d = l1;
			*d1 = 0.0;
			*d2 = gst_distance(cip -> metric, p1 -> x, p1 -> y, p2 -> x, p2 -> y);
		}
		else {
			d = l2;
			*d1 = gst_distance(cip -> metric, p1 -> x, p1 -> y, p2 -> x, p2 -> y);
			*d2 = 0.0;
		}
	}
	return d;
}

/*
 * Find all FSTs whose removal splits the problem.  Make these
 * all required.
 */

	static
	int
bcc_find_required (

struct gst_hypergraph *	cip,		/* IN - compatibility info */
int *			made_req,	/* IN/OUT - list of FSTs made REQUIRED */
int			req_count	/* IN - existing required count */
)
{
int			i;
int			j;
int			nverts;
int			nedges;
int			kmasks;
int			nmasks;
struct bc3		bc;

	nverts	= cip -> num_verts;
	nedges	= cip -> num_edges;

	if (nverts < 2) {
		/* No BCC's. */
		return (req_count);
	}

	kmasks	= BMAP_ELTS (nverts);
	nmasks	= BMAP_ELTS (nedges);

	bc.cip		= cip;
	bc.kmasks	= kmasks;
	bc.nmasks	= nmasks;
	bc.dfs		= NEWA (nverts, int);
	bc.low		= NEWA (nverts, int);
	bc.parent	= NEWA (nverts, int);

	for (i = 0; i < nverts; i++) {
		bc.dfs	[i] = 0;
		bc.low [i] = 0;
		bc.parent [i] = -1;
	}

	j = (nedges > nverts) ? nedges : nverts;
	bc.max_stack	= j;
	bc.stack	= NEWA (j, int);
	bc.sp		= bc.stack;
	bc.counter	= 0;

	bc.bcc_vmask	= NEWA (kmasks, bitmap_t);
	bc.bcc_emask	= NEWA (nmasks, bitmap_t);
	bc.edges_seen	= NEWA (nmasks, bitmap_t);
	bc.bcc_vlist	= NEWA (nverts, int);
	bc.degree	= NEWA (nverts, int);
	bc.made_req	= made_req;
	bc.req_count	= req_count;

	for (i = 0; i < bc.kmasks; i++) {
		bc.bcc_vmask [i] = 0;
	}
	for (i = 0; i < bc.nmasks; i++) {
		bc.bcc_emask [i] = 0;
		bc.edges_seen [i] = 0;
	}

	/* Traverse each connected component, identifying its BCC's as	*/
	/* we go.							*/
	for (i = 0; i < nverts; i++) {
		if (NOT BITON (cip -> initial_vert_mask, i)) continue;
		if (bc.dfs [i] > 0) continue;

		/* Traverse one connected component, finding	*/
		/* each of its BCCs as we go.			*/

		bcc3 (&bc, i);
	}

	free ((char *) bc.degree);
	free ((char *) bc.bcc_vlist);
	free ((char *) bc.edges_seen);
	free ((char *) bc.bcc_emask);
	free ((char *) bc.bcc_vmask);
	free ((char *) bc.stack);
	free ((char *) bc.parent);
	free ((char *) bc.low);
	free ((char *) bc.dfs);

	return (bc.req_count);
}

/*
 * This is the recursive part of the bi-connected-components algorithm.	 It
 * is the standard method, with a few tweaks to work on hypergraphs instead.
 * We process each bi-connected component individually.
 */

	static
	void
bcc3 (

struct bc3 *		bcp,		/* IN - global BCC data */
int			v		/* IN - current DFS vertex */
)
{
int			e;
int			e2;
int			w;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			stack_endp;
int *			sp;
int *			stack;
struct gst_hypergraph *	cip;

	cip = bcp -> cip;

	FATAL_ERROR_IF ((v < 0) OR (v >= cip -> num_verts));

	++(bcp -> counter);
	bcp -> dfs [v] = bcp -> counter;
	bcp -> low [v] = bcp -> counter;
	ep1 = cip -> term_trees [v];
	ep2 = cip -> term_trees [v + 1];
	while (ep1 < ep2) {
		e = *ep1++;
		FATAL_ERROR_IF ((e < 0) OR (e >= cip -> num_edges));
		if (NOT BITON (cip -> initial_edge_mask, e)) continue;
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
		vp1 = cip -> edge [e];
		vp2 = cip -> edge [e + 1];
		while (vp1 < vp2) {
			w = *vp1++;
			FATAL_ERROR_IF ((w < 0) OR (w >= cip -> num_verts));
			if (bcp -> dfs [w] EQ 0) {
				bcp -> parent [w] = v;
				bcc3 (bcp, w);
				if (bcp -> low [w] >= bcp -> dfs [v]) {
					/* We have a new BCC! */
					stack	= bcp -> stack;
					sp	= bcp -> sp;
					do {
						FATAL_ERROR_IF (sp <= stack);
						e2 = *--sp;
					} while (e2 NE e);

					/* Process the bi-connected comp. */
					process_bcc3 (bcp, sp, bcp -> sp);

					/* Pop BCC edges from stack */
					bcp -> sp = sp;
				}
				if (bcp -> low [w] < bcp -> low [v]) {
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
 * Process a single bi-connected component, specified as a list of edges.
 * We look for vertices having degree 1 within the component.  When this
 * happens, the incident edge is required.
 */

	static
	void
process_bcc3 (

struct bc3 *	bcp,		/* IN - global BCC data */
int *		edge_ptr,	/* IN - list of BCC edges */
int *		endp		/* IN - end of BCC edge list */
)
{
int			i;
int			e;
int			v;
int			n;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			vlp;
struct gst_hypergraph *	cip;

	cip = bcp -> cip;

	/* Gather a list of all vertices in this BCC.  Compute	*/
	/* their degrees (with respect to the BCC).		*/
	vlp = bcp -> bcc_vlist;

	ep1 = edge_ptr;
	ep2 = endp;
	while (ep1 < ep2) {
		e = *ep1++;
		SETBIT (bcp -> bcc_emask, e);
		vp1 = cip -> edge [e];
		vp2 = cip -> edge [e + 1];
		while (vp1 < vp2) {
			v = *vp1++;
			if (NOT BITON (cip -> initial_vert_mask, v)) continue;
			if (NOT BITON (bcp -> bcc_vmask, v)) {
				*vlp++ = v;
				SETBIT (bcp -> bcc_vmask, v);
				bcp -> degree [v] = 0;
			}
			++(bcp -> degree [v]);
		}
	}

	/* All of the vertices of this BCC are now known, as are their	*/
	/* degrees (relative to the component).	 Now look for vertices	*/
	/* of degree 1.							*/

	vp1 = bcp -> bcc_vlist;
	vp2 = vlp;
	while (vp1 < vp2) {
		v = *vp1++;
		CLRBIT (bcp -> bcc_vmask, v);		/* clean up as we go */

		n = bcp -> degree [v];
		if (n > 1) continue;

		ep1 = cip -> term_trees [v];
		ep2 = cip -> term_trees [v + 1];
		for (;;) {
			FATAL_ERROR_IF (ep1 >= ep2);
			e = *ep1++;
			if (BITON (bcp -> bcc_emask, e)) break;
		}
		if (BITON (cip -> required_edges, e)) continue;
		i = (bcp -> req_count)++;
		bcp -> made_req [i] = e;
		SETBIT (cip -> required_edges, e);
	}

	/* Clean up edge mark flags. */
	ep1 = edge_ptr;
	ep2 = endp;
	while (ep1 < ep2) {
		e = *ep1++;
		CLRBIT (bcp -> bcc_emask, e);
	}
}

/*
 * For sorting integers in place
 */

	static
	int
comp_ints (

const void *		p1,
const void *		p2
)
{
int			l1;
int			l2;

	l1 = *((int *) p1);
	l2 = *((int *) p2);

	if (l1 < l2) return (-1);
	if (l1 > l2) return (1);
	return (0);
}


/*
 * For sorting pruning graph in place
 */

	static
	int
comp_pg_edges (

const void *		p1,
const void *		p2
)
{
dist_t			l1;
dist_t			l2;

	l1 = ((struct pg_edge *) p1) -> len;
	l2 = ((struct pg_edge *) p2) -> len;

	if (l1 < l2) return (-1);
	if (l1 > l2) return (1);
	return (0);
}

/*
 * For sorting close terminals in place
 */

	static
	int
comp_clt (

const void *		p1,
const void *		p2
)
{
dist_t			l1;
dist_t			l2;

	l1 = ((struct clt_info *) p1) -> dist;
	l2 = ((struct clt_info *) p2) -> dist;

	if (l1 < l2) return (-1);
	if (l1 > l2) return (1);
	return (0);
}
