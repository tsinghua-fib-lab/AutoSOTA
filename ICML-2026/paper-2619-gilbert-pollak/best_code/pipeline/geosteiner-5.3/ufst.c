/***********************************************************************

	$Id: ufst.c,v 1.53 2023/04/03 16:54:06 warme Exp $

	File:	ufst.c
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 2002, 2023 by Pawel Winter & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Uniform FST generator.

************************************************************************

	Modification Log:

	a-1:	05/03/2002	benny
		: Created.  Derived from efst.c and UniSteiner.
	b-1:	04/24/2014	warme
		: Comment code out in a different manner.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Use better encapsulation for time conversions.
		: Fix -Wall issues.  Upgrade fatals.
	e-4:	11/18/2022	warme
		: Fix -Wall issues.
	e-5:	04/03/2023	warme
		: Fix handling of >= 64k terminals.  (Still
		:  suffers from O(n^2) memory thrash.)

************************************************************************/

#include "ufst.h"

#include "bsd.h"
#include "cputime.h"
#include "efuncs.h"
#include "fatal.h"
#include <float.h>
#include "fstfuncs.h"
#include "geosteiner.h"
#include <limits.h>
#include "logic.h"
#include "memory.h"
#include "metric.h"
#include "mst.h"
#include "parmblk.h"
#include "prepostlude.h"
#include "sortfuncs.h"
#include "steiner.h"
#include <string.h>

/*
 * Global functions
 */

gst_hg_ptr	gst_generate_ofsts (int			nterms,
				    double *		terminals,
				    struct gst_param *	params,
				    int *		status);

gst_hg_ptr	gst_generate_ufsts (int,
				    double *,
				    int,
				    struct gst_param *,
				    int *);

/*
 * Local functions
 */

static void		add_zero_length_fsts (struct uinfo *, int, int **);
static void		build_fst_list (struct uinfo *);
static double		closest_terminal (struct hFST *,
					  gst_metric_ptr,
					  struct point *);
static void		compute_ufsts_for_unique_terminals (struct uinfo *,
							    cpu_time_t *);
static void		destroy_uinfo (struct uinfo *);
static bool		edge_lune_test (struct uinfo *,
					struct hFST *,
					struct hFST *,
					struct hFST *,
					double, double);
static double		get_bsd (struct hFST *, struct hFST *, struct bsd *);
static int		get_steiner_points (struct hFST *, struct point *, int *);
static double		hfst_mst (struct uinfo *, struct hFST *, bool);
static double		hfst_root_distance (struct uinfo *, struct hFST *, struct point *);
static bool		identical_lists (const int *, const int *);
static void		initialize_uinfo (struct uinfo *);
static bool		is_disjoint (struct hFST *, struct hFST *);
static bool 		points_close (struct point *, struct point *, double);
static void		renumber_terminals (struct uinfo *,
					    struct pset *,
					    int *);
static void		set_member (struct hFST *, bool *membArray, bool flag);
static int		setup_extensions (struct hFST *,
					  gst_metric_ptr,
					  int);
static dist_t		test_and_save_fst (struct uinfo *, struct hFST *);
static void		update_terms (struct hFST *);
static bool		upper_bound_tests (struct uinfo *, struct hFST *, bool);
static bool		wedge_test (struct uinfo *, struct hFST *);

static struct time_it *	create_time_it (char *);
static void		start_timer (struct time_it *);
static void		stop_timer (struct time_it *, bool);

enum
{
	NO_BENT_EDGE = 0,
	WEDGE,
	GENERATOR,
	SUBPATH,
	DISJOINT,
	ANGLE,
	INTERSECT,
	BSDTEST,
	LUNE,
	FSTCHECK,
	UB1,
	UB2,
	UB3,
	UPDATE_UB,
	ADDTIME,
	MSTEDGE,
	LEGALFLIP,
	LENGTH,
	REPLACE,
	ADDNEW,
	NUM_OF_COUNTERS
};

struct time_it
{
	char *		name;
	int		elapsed_time;		/* Elapsed time in all subperiods */
	int		queries;		/* Number of 'start_timer ()' */
	int		pruned;			/* Number of 'stop_timer (true)' */

	int		temp_time;
};

static struct time_it *counters[NUM_OF_COUNTERS];

/*
 * Local Macros
 */

#define STATE_CLEAN	0
#define STATE_MIXED	1

#define SubpathTest	TRUE
#define DoWedgeTest	TRUE

#ifdef STATISTICS_ENABLED
#define Verbose TRUE
#else
#define Verbose FALSE
#endif

/*
 * An octilinear FST generator...
 */

	gst_hg_ptr
gst_generate_ofsts (

int			nterms,
double *		terminals,
gst_param_ptr		params,
int *			status
)
{
gst_hg_ptr		H;

	GST_PRELUDE

	H = gst_generate_ufsts (nterms, terminals, 4, params, status);

	GST_POSTLUDE
	return H;
}

/*
 * The general uniform FST generator...
 */

	gst_hg_ptr
gst_generate_ufsts (

int			nterms,
double *		terminals,
int			lambda,
gst_param_ptr		params,
int *			status
)
{
int			i;
int			j;
int			k;
int			ndg;
int			ntrees;
int			count;
int			code;
int **			dup_grps;
int *			fwd_map;
int *			rev_map;
int *			ip1;
struct full_set *	fsp;
int *			tlist;
struct uinfo		uinfo;
struct gst_hypergraph *	cip;
struct pset *		pts;
struct pset *		pts2;
cpu_time_t		T0;
cpu_time_t		Tn;
cpu_time_t		Trenum;
char			buf1 [32];
gst_channel_ptr		timing;
gst_proplist_ptr	plist;

	GST_PRELUDE;

	code = 0;

	cip = NULL;

	if (params EQ NULL) {
		params = (gst_param_ptr) &_gst_default_parmblk;
	}
	uinfo.params = params;
	timing = params -> detailed_timings_channel;

	uinfo.metric = gst_create_metric (GST_METRIC_UNIFORM,
					  lambda,
					  NULL);

	pts     = _gst_create_pset (nterms, terminals);

	T0 = _gst_get_cpu_time ();
	Tn = T0;

	uinfo.x_order = _gst_heapsort_x (pts);

	if (timing NE NULL) {
		_gst_convert_delta_cpu_time (buf1, &Tn);
		gst_channel_printf (timing, "Sort X:                 %s\n", buf1);
	}

	/* Find all duplicate terminals in the input. */
	ndg = _gst_generate_duplicate_terminal_groups (pts, uinfo.x_order, &dup_grps);

	uinfo.num_term_masks = BMAP_ELTS (pts -> n);

	/* Remove all but the first of each duplicate terminal. */
	/* Compute forward and reverse maps to renumber the terminals. */
	pts2 = _gst_remove_duplicates (pts, ndg, dup_grps, &fwd_map, &rev_map);

	/* Renumber the x_order list -- instead of re-sorting pts2. */
	j = 0;
	for (i = 0; i < pts -> n; i++) {
		k = uinfo.x_order [i];
		FATAL_ERROR_IF ((k < 0) OR (pts -> n < k));
		k = fwd_map [k];
		if (k < 0) continue;
		uinfo.x_order [j++] = k;
	}

	if (timing NE NULL) {
		_gst_convert_delta_cpu_time (buf1, &Tn);
		gst_channel_printf (timing, "Remove Duplicates:      %s\n", buf1);
	}

	/* From now on, we work only with the reduced terminal set, and */
	/* we assume that all terminals are unique. */

	uinfo.pts = pts2;

	compute_ufsts_for_unique_terminals (&uinfo, &Tn);

	/* Now put the terminal numbers back the way they were, */
	/* renumber the terminals within each UFST, etc. */

	renumber_terminals (&uinfo, pts, rev_map);

	/* Link the FSTs together into one long list, and number them. */
	build_fst_list (&uinfo);

	/* Add one FST for each duplicate terminal that was removed. */
	if (ndg > 0) {
		add_zero_length_fsts (&uinfo, ndg, dup_grps);
	}

	/* Measure renumber time.  This also sets Tn so that Tn-T0 is	*/
	/* the total processing time.					*/
	Trenum = _gst_get_delta_cpu_time (&Tn);

	if (timing NE NULL) {
		_gst_convert_cpu_time (Trenum, buf1);
		gst_channel_printf (timing, "Renumber Terminals:     %s\n", buf1);
		_gst_convert_cpu_time (Tn - T0, buf1);
		gst_channel_printf (timing, "Total:                  %s\n", buf1);
	}

	cip = gst_create_hg (NULL);
	gst_set_hg_number_of_vertices (cip, uinfo.pts -> n);
	plist = cip -> proplist;

	gst_free_metric (cip -> metric);
	cip -> metric = uinfo.metric;

	cip -> num_edges		= uinfo.ntrees;
	cip -> num_edge_masks		= BMAP_ELTS (cip -> num_edges);
	cip -> edge			= NEWA (uinfo.ntrees + 1, int *);
	cip -> edge_size		= NEWA (uinfo.ntrees, int);
	cip -> cost			= NEWA (uinfo.ntrees, dist_t);
	cip -> pts			= uinfo.pts;
	cip -> full_trees		= _gst_put_trees_in_array (
							uinfo.full_sets,
							&ntrees);

	gst_set_dbl_property (plist, GST_PROP_HG_INTEGRALITY_DELTA, 0);
	gst_set_dbl_property (plist, GST_PROP_HG_MST_LENGTH, uinfo.mst_length);
	gst_set_dbl_property (cip -> proplist,
			      GST_PROP_HG_GENERATION_TIME,
			      _gst_cpu_time_t_to_double_seconds (Tn - T0));
	gst_set_int_property (cip -> proplist,
			      GST_PROP_HG_HALF_FST_COUNT,
			      uinfo.hFSTCount);

	count = 0;
	for (i = 0; i < uinfo.ntrees; i++) {
		fsp = cip -> full_trees [i];
		k = fsp -> terminals -> n;
		cip -> edge_size [i]	= k;
		cip -> cost [i]		= fsp -> tree_len;
		count += k;
	}
	ip1 = NEWA (count, int);
	for (i = 0; i < uinfo.ntrees; i++) {
		cip -> edge [i] = ip1;
		fsp = cip -> full_trees [i];
		tlist = fsp -> tlist;
		k = fsp -> terminals -> n;
		for (j = 0; j < k; j++) {
			*ip1++ = tlist [j];
		}
	}
	cip -> edge [i] = ip1;

	/* Clean up all that is not in cip */

	free ((char *) pts2);
	free ((char *) rev_map);
	free ((char *) fwd_map);
	if (dup_grps NE NULL) {
		if (dup_grps [0] NE NULL) {
			free ((char *) (dup_grps [0]));
		}
		free ((char *) dup_grps);
	}
	free ((char *) (uinfo.x_order));

	/* Initialize any missing information in the hypergraph */
	_gst_initialize_hypergraph (cip);

	if (status NE NULL) {
		*status = code;
	}

	GST_POSTLUDE;

	return (cip);
}

/*
 * Compute the UFSTs for the given set of terminals, which are now
 * guaranteed to be unique.
 */

	static
	void
compute_ufsts_for_unique_terminals (

struct uinfo *		uip,	/* IN - global UFST info */
cpu_time_t *		Tn
)
{
int			i, j, k, ti, tj;
int			n;		/* Number of points */
int			K;		/* 2 * lambda */
int			lambda;
int			size;		/* Size of (h)FST */
int			hFSTcount;	/* Number of generated hFSTs */
int			max_fst_size;
int			max_angle;
int			min_angle;
dist_t			eps_factor;
int *			meetAngles;
bool *			isLeftRight;
char			buf1 [32];
struct point		droot;
struct point *		p;
struct point		dp;
struct pset *		pts;
struct bsd *		BSD;
struct hFST		hfstk;
struct hFST *		tLeft;
struct hFST *		tRight;
struct hFST *		last;
struct hFST **		hFST_Lists;
gst_metric_ptr		metric;
gst_param_ptr		params;
gst_channel_ptr		timing;

	if (Verbose) {
		counters [NO_BENT_EDGE] = create_time_it("No bent edge");
		counters [WEDGE]	= create_time_it("Wedge");
		counters [GENERATOR]	= create_time_it("Generator");
		counters [SUBPATH]	= create_time_it("SubPath");
		counters [DISJOINT]	= create_time_it("Disjoint");
		counters [ANGLE]	= create_time_it("Angle test");
		counters [INTERSECT]	= create_time_it("Intersection");
		counters [BSDTEST]	= create_time_it("BSD");
		counters [LUNE]		= create_time_it("Lune");
		counters [FSTCHECK]	= create_time_it("FST Check");
		counters [UB1]		= create_time_it("Upperbound 1");
		counters [UB2]		= create_time_it("Upperbound 2");
		counters [UB3]		= create_time_it("Upperbound 3");
		counters [UPDATE_UB]	= create_time_it("Update UB");
		counters [ADDTIME]	= create_time_it("Add Time");
		counters [MSTEDGE]	= create_time_it("Not MST Edge");
		counters [LEGALFLIP]	= create_time_it("Illegal flip");
		counters [LENGTH]	= create_time_it("Length");
		counters [REPLACE]	= create_time_it("Replaced");
		counters [ADDNEW]	= create_time_it("Added");
	}

	if (Verbose) start_timer(counters[GENERATOR]);

	initialize_uinfo (uip);
	hFSTcount = 0;

	/* Initialize various commonly used variables */
	params		= uip -> params;
	timing		= params -> detailed_timings_channel;
	metric		= uip -> metric;
	K		= metric -> K;
	lambda		= metric -> lambda;
	max_angle	= metric -> max_angle;
	min_angle	= metric -> min_angle;
	eps_factor	= 1.0 + uip -> eps;
	BSD	= uip -> bsd;
	pts	= uip -> pts;
	n	= pts -> n;

	/* Generate table for checking that meeting angle is valid
	   (might be a bending point on a non-straight edge!) */
	meetAngles = NEWA (K*K, int);
	for (i = 0; i < K; i++) {
		for (j = 0; j < K; j++) {
			int meetangle = MIN ( abs (i - j), abs (K - abs (i - j)) );
			if (meetangle NE (lambda - 1)) {
				if ((meetangle < min_angle)
				 OR (meetangle > max_angle)) {
					meetangle = 0;
				}
			}
			meetAngles[i*K+j] = meetangle;
		}
	}

	/* Table for finding out which tree is to the left and which is to the right
	   (given the ray orientations).
			     o
			    / \
			   /   \
			  /	\
			Left   Right
	*/
	isLeftRight = NEWA (K*K, bool);
	for (i = 0; i < K; i++) {
		for (j = 0; j < K; j++) {
			int oi = (i + lambda) % K;
			int oj = (j + lambda) % K;
			isLeftRight[i*K+j] =	((oj > oi) AND ((oj - oi) <= lambda))
					     OR ((oj < oi) AND ((oi - oj) >  lambda));
		}
	}

	/* Compute the mean of all terminals.  We translate the terminals */
	/* so that the mean is at the origin.  The coordinates of the */
	/* translated instance have smaller magnitude, which permits us */
	/* to compute eq-point coordinates with higher precision. */

	uip -> mean.x = 0.0;
	uip -> mean.y = 0.0;
	for (k = 0; k < n; k++) {
		uip -> mean.x += pts -> a[k].x;
		uip -> mean.y += pts -> a[k].y;
	}
	uip -> mean.x = floor( uip -> mean.x / (double) n);
	uip -> mean.y = floor( uip -> mean.y / (double) n);

	/* Make a copy of the original terminal set */
	uip -> pts_org = NEW_PSET (n);

	/* Translate terminal set */
	for (k = 0; k < n; k++) {
	  	uip -> pts_org -> a[k] = pts -> a[k];
		pts -> a[k].x -= uip -> mean.x;
		pts -> a[k].y -= uip -> mean.y;
	}

	if (timing NE NULL) {
		_gst_convert_delta_cpu_time (buf1, Tn);
		gst_channel_printf (timing, "Initialization:         %s\n", buf1);
	}

	/* Generate and initialize basic hFSTs (terminals with one extension) */

	hFST_Lists = NEWA (n+1, struct hFST *);
	for (i = 0; i < n+1; i++) {
		hFST_Lists[i] = NULL;
	}

	hfstk.length		= 0.0;
	hfstk.left_tree		= NULL;
	hfstk.right_tree	= NULL;
	hfstk.ext_left		= -1;
	hfstk.ext_right		= -1;
	hfstk.S			= 1;
	hfstk.UB		= 0.0;
	hfstk.status		= STATE_CLEAN;
	hfstk.mixable		= TRUE;
	hfstk.all_left_used	= FALSE;
	hfstk.prev_identical	= FALSE;

	for (j = 0; j < 3; j++) {
		hfstk.right_legs [j] = FALSE;
	}

	last = NULL;
	p = &pts -> a[0];
	for (k = 0; k < n; k++, p++) {
		int or;
		hfstk.index = k;
		hfstk.root  = *p;

		hfstk.origin_term = k;
		hfstk.droot.x = 0.0;
		hfstk.droot.y = 0.0;

		for (or=0; or < K; or++) {
			struct hFST *tmp;
			hfstk.terms = NEWA (2, int);
			hfstk.terms[0] = k;
			hfstk.terms[1] = -1;
			hfstk.ext = or;

			/* An early wedge test -- might not be
			   a good idea for small lambda values */
			if (DoWedgeTest) {
				if (wedge_test(uip, &hfstk)) {
					free (hfstk.terms);
					continue;
				}
			}

			tmp = NEW (struct hFST);
			*tmp = hfstk;
			tmp -> next = NULL;

			if (last) {
				last -> next = tmp;
			}
			else {
				hFST_Lists[1] = tmp;
			}

			last = tmp;

			hFSTcount++;
		}
	}

	/* The variable 'hfstk' is used repeatedly to generate new trees.
	   It is copied whenever a tree needs to be saved for later use. */
	hfstk.terms	= NULL;
	hfstk.ext	= -1;

	max_fst_size = params -> max_fst_size;
	if (max_fst_size > n) max_fst_size = n;
	for (size = 2; size <= max_fst_size; size++) {
		int isize, jsize;
		/* New terminal array size */
		hfstk.terms = NEWA (size + 1, int);
		hfstk.S	    = size;

		if (Verbose) {
			fprintf (stderr, "- starting hFST size %d (hFSTs: %d, FSTs: %d)\n",
				size, hFSTcount, uip -> ntrees);
		}

		isize = (size-1)/2 + 1;
		for ( ; isize < size; isize++) {
			struct hFST *hfsti;
			for (hfsti = hFST_Lists[isize]; hfsti; hfsti = hfsti -> next) {
				struct hFST *hfstj;
				bool degree4possible;
				bool hfsti_changed = TRUE;
				bool bsd_reusable = FALSE;
				jsize = size - isize;

				/* Setup special degree 4 flag */
				/* This might not be entirely correct... */
				degree4possible = ( (isize EQ 2)
						AND (jsize EQ 2)
						AND ((K EQ 4) OR (K EQ 8)));


				for (hfstj = hFST_Lists[jsize]; hfstj; hfstj = hfstj -> next) {
					int ii, jj;
					int meetangle;
					double dist_ik, dist_jk;
					struct point *iRoot, *jRoot;
					int extensions;
					double cachedMST;
					bool disjoint;

					bool size_condition;
#if 1
					size_condition = TRUE;
#else
					size_condition = (jsize EQ isize);
#endif
					if (size_condition AND (hfsti EQ hfstj)) {
						break;
					}

					if (bsd_reusable AND (NOT hfstj -> prev_identical)) {
						bsd_reusable = FALSE;
					}

					/* This test works for any lambda-value:
					   The subtree which does not contain the lowest
					   index can be pruned if it is in a mixed state. */
					if (SubpathTest) {
						if (Verbose) start_timer(counters[SUBPATH]);
						if (*hfsti -> terms < *hfstj -> terms) {
							if (hfstj -> status EQ STATE_MIXED) {
								if (Verbose) stop_timer (counters[SUBPATH], TRUE);
								continue;
							}
						}
						else {
							if (hfsti -> status EQ STATE_MIXED) {
								if (Verbose) stop_timer (counters[SUBPATH], TRUE);
								continue;
							}
						}
						if (Verbose) stop_timer (counters[SUBPATH], FALSE);
					}

					/* Check that the hFSTs are terminal disjoint */
					disjoint = TRUE;
					if (Verbose) start_timer(counters[DISJOINT]);
					if (size EQ 2) { /* Two terminals */
						if (hfsti -> index EQ hfstj -> index) {
							disjoint = FALSE;
						}
					}
					else {
						if (NOT is_disjoint(hfsti, hfstj)) {
							disjoint = FALSE;
						}
					}

					if (disjoint) {
						if (Verbose) stop_timer (counters[DISJOINT], FALSE);
					}
					else {
						/* Skip identical hFSTs - regarding terminals */
						while (hfstj -> next AND hfstj -> next -> prev_identical) {
							hfstj = hfstj -> next;
							if (hfsti EQ hfstj) {
								break;
							}
						}
						if (Verbose) stop_timer (counters[DISJOINT], TRUE);
						continue;
					}

					/* Check that meeting angle is valid
					   (might be a bending point on a non-straight edge!) */

					ii = hfsti -> ext;
					jj = hfstj -> ext;
					iRoot = &hfsti -> root;
					jRoot = &hfstj -> root;

					/* Is it a legal angle? (This test could be better if
					   it also considered the direction of the rays). */
					if (Verbose) start_timer(counters[ANGLE]);
					meetangle = meetAngles[ii*K + jj];
					if (NOT meetangle) {
						/* Illegal unless it is a degree 4
						   Steiner point, not parallel and
						   with (almost) identical Root
						   points */
						if (   degree4possible
						   AND ((ii - jj) % lambda EQ 0)
						   AND (points_close(iRoot, jRoot, uip -> eps)) ) {
							/* Save it as an FST */
							hfstk.length	 = hfsti -> length + hfstj -> length;
							hfstk.left_tree	 = hfsti;
							hfstk.right_tree = hfstj;
							hfstk.type	 = TYPE_CROSS;
							hfstk.root       = *iRoot;
							update_terms(&hfstk); /* Update array of terminals. */
							test_and_save_fst (uip, &hfstk);
							continue;
						}
						else {
							if (Verbose) stop_timer (counters[ANGLE], TRUE);
							continue;
						}
					}
					if (Verbose) stop_timer (counters[ANGLE], FALSE);

					/* Roots should not be equal (but be careful with MST edges!) */
					if (Verbose) start_timer(counters[INTERSECT]);
					if ( (points_close(iRoot, jRoot, uip -> eps)) AND
					     (NOT ((hfstk.S EQ 2) AND 
						   (_gst_is_mst_edge(BSD, hfsti->index, hfstj->index)))) ) {
						if (Verbose) stop_timer (counters[INTERSECT], TRUE);
						continue;
					}

					/* Compute root of combined hFST */
					/* We do it carefully using displacements */

					ti = hfsti -> origin_term;
					tj = hfstj -> origin_term;

					dp.x = pts -> a[tj].x - pts -> a[ti].x;
					dp.y = pts -> a[tj].y - pts -> a[ti].y;

					/* Add displacements */
					dp.x -= hfsti -> droot.x;
					dp.y -= hfsti -> droot.y;
					dp.x += hfstj -> droot.x;
					dp.y += hfstj -> droot.y;

					/* Compute intersection */
					if (NOT _gst_ray_intersection(&metric -> dirs[ii],
								      &dp,
								      &metric -> dirs[jj],
								      uip -> eps, &droot)) {
						if (Verbose) stop_timer (counters[INTERSECT], TRUE);
						continue;
					}

					hfstk.droot.x = hfsti -> droot.x + droot.x; 
					hfstk.droot.y = hfsti -> droot.y + droot.y; 
					hfstk.origin_term = hfsti -> origin_term;

					/* Move new root to final position */
					hfstk.root.x = pts -> a[ hfstk.origin_term ].x +
						       hfstk.droot.x;
					hfstk.root.y = pts -> a[ hfstk.origin_term ].y +
						       hfstk.droot.y;
					if (Verbose) stop_timer (counters[INTERSECT], FALSE);

					/* Bottleneck Steiner distances */

					if (Verbose) start_timer(counters[BSDTEST]);
					if ((NOT bsd_reusable) OR (hfsti_changed)) {
						hfstk.BS = get_bsd(hfsti, hfstj, BSD);
						hfsti_changed = FALSE;
						bsd_reusable = TRUE;
					}
					hfstk.UB = hfsti->UB + hfstj->UB + hfstk.BS;

					dist_ik = sqrt(droot.x * droot.x + droot.y * droot.y);
					if (dist_ik > eps_factor * hfstk.BS) {
						if (Verbose) stop_timer (counters[BSDTEST], TRUE);
						continue;
					}

					dist_jk = sqrt(sqr_dist(&droot, &dp));
					if (dist_jk > eps_factor * hfstk.BS) {
						if (Verbose) stop_timer (counters[BSDTEST], TRUE);
						continue;
					}
					if (Verbose) stop_timer (counters[BSDTEST], FALSE);

					/* Edge Lune Test */
					if (Verbose) start_timer(counters[LUNE]);
					if (NOT edge_lune_test(uip, hfsti, hfstj, &hfstk,
							       dist_ik, dist_jk)) {
						if (Verbose) stop_timer (counters[LUNE], TRUE);
						continue;
					}
					if (Verbose) stop_timer (counters[LUNE], FALSE);

					/* Figure out what is left and what is right */
					if (isLeftRight[ii*K+jj]) {
						tLeft = hfsti; tRight = hfstj;
					}
					else {
						tLeft = hfstj; tRight = hfsti;
					}

					hfstk.left_tree		= tLeft;
					hfstk.right_tree	= tRight;
					hfstk.ext_left		= tLeft -> ext;
					hfstk.ext_right		= tRight -> ext;

					update_terms(&hfstk); /* Update array of terminals. */
					hfstk.length = hfsti -> length + hfstj -> length + dist_ik + dist_jk;

					/* If the Steiner point is VERY close to one of the points then
					   it is defined as overlapping and should not be extended */
					hfstk.type = TYPE_CORNER;

					if ( (points_close(&hfstk.root, iRoot, uip -> eps)) OR 
					     (points_close(&hfstk.root, jRoot, uip -> eps)) ) {
						hfstk.type = TYPE_STRAIGHT;
					}

					/* Can this hFST be extended */
					extensions = 0;

					/* No need to extend if Steiner point overlaps with children
					   or the meeting angle is not legal at Steiner points */
					if ((meetangle <= max_angle) AND (hfstk.type == TYPE_CORNER)) {
						/* Setup the extensions */
						extensions = setup_extensions(&hfstk, metric, meetangle);

						/* Can the wedge test remove the extension(s) */
						if (DoWedgeTest AND extensions) {
							/* Check the first extension */
							if (wedge_test (uip, &hfstk)) {
								extensions--;
								if (extensions) {
									/* Check the second extension */
									hfstk.status = STATE_CLEAN;
									hfstk.ext = (hfstk.ext + 1) % K;
									if (wedge_test(uip, &hfstk)) {
										extensions--;
									}
								}
							}

						}
					}

					cachedMST = 0;

					if (extensions) {
						struct hFST *tmp;
						struct hFST *before = NULL, *after;
						int *termsk = hfstk.terms;
						if (NOT upper_bound_tests (uip, &hfstk, TRUE)) continue;
						/* Length of hFST is UB for tree spanning terminals */
						if (Verbose) start_timer(counters[UPDATE_UB]);
						if (size > 2) {
							bool first_found = FALSE;
							struct hFST *hfstl;

							if (hfstk.UB > eps_factor * hfstk.length) {
								hfstk.UB = hfstk.length;
							}

							/* Heuristic upper bound */
							if (NOT cachedMST) {
								cachedMST = hfst_mst (uip, &hfstk, FALSE);
							}
							if (hfstk.UB > eps_factor * cachedMST) {
								hfstk.UB = cachedMST;
							}

							/* Now check if there is another hFST spanning the same set of terminals */
							for (hfstl = hFST_Lists[size]; hfstl; hfstl = hfstl -> next) {
								int *terms = hfstl -> terms;
								if (identical_lists(terms, termsk)) {
									first_found = TRUE;
									if (hfstl->UB <= hfstk.UB) {
										hfstk.UB = hfstl->UB;
										/* This is deliberate (and correct) */
										break;
									}
									else {
										hfstl->UB = hfstk.UB;
									}
								}
								else if (first_found) {
									break;
								}
							}
						}
						if (Verbose) stop_timer (counters[UPDATE_UB], FALSE);

						/* Sorted insertion of the hFST */

						for (after = hFST_Lists[size]; after; after = after -> next) {
							int *terms = after -> terms;
							if (identical_lists(terms, termsk)) {
								after -> prev_identical = TRUE;
								break;
							}
							before = after;
						}

						tmp = NEW (struct hFST);
						*tmp = hfstk;
						if (before) {
							before -> next = tmp;
						}
						else {
							hFST_Lists[size] = tmp;
						}

						if (extensions EQ 2) {
							hfstk.ext = (hfstk.ext + 1) % K;
							hfstk.status = STATE_CLEAN;

							if (NOT (DoWedgeTest AND wedge_test(uip, &hfstk))) {
								struct hFST *tmp2 = NEW (struct hFST);
								*tmp2 = hfstk;
								tmp2 -> terms = NEWA (size+1, int);
								memcpy (tmp2 -> terms, tmp -> terms, (size + 1)*sizeof (int));
								tmp2 -> prev_identical = TRUE;
								tmp -> next = tmp2;
								tmp = tmp2;

								hFSTcount++;
							}
						}

						tmp -> next = after;

						hfstk.terms = NEWA (size+1, int);
						memcpy (hfstk.terms, tmp -> terms, (size + 1)*sizeof (int));

						hFSTcount++;
					}

					if (Verbose) start_timer(counters[FSTCHECK]);

					/* Make sure the bent edge has an angle of \pi - \omega */
					if (meetangle < lambda - 1) {
						if (Verbose) stop_timer (counters[FSTCHECK], TRUE);
						continue;
					}

					/* A subtle restriction... :-)
					   Gives us the correct primary/secondary shape
					   of the bent edge. */
					if (*tLeft -> terms < *tRight -> terms) {
						if (Verbose) stop_timer (counters[FSTCHECK], TRUE);
						continue;
					}

					if (size EQ 2) {
						if (Verbose) start_timer(counters[MSTEDGE]);
						if (NOT _gst_is_mst_edge(BSD, tLeft->index, tRight->index)) {
							if (Verbose) stop_timer (counters[FSTCHECK], TRUE);
							if (Verbose) stop_timer (counters[MSTEDGE], TRUE);
							continue;
						}
						if (Verbose) stop_timer (counters[MSTEDGE], FALSE);
					}
					else { /* size > 2 */
						if (tRight -> status EQ STATE_CLEAN) {
							/* The bent edge cannot appear
							   in a tree to the right */
							if (lambda % 3 EQ 0) {
								if (NOT tRight -> mixable) {
									if (Verbose) stop_timer (counters[FSTCHECK], TRUE);
									continue;
								}
							}
							else {
								if (tRight -> right_legs[0]) {
									if (Verbose) stop_timer (counters[FSTCHECK], TRUE);
									continue;
								}
							}
						}
						else { /* tRight->Status EQ STATE_MIXED */
							/* The mixed edge has to be the bent edge
							   when lambda NE 3m */
							if (lambda % 3 NE 0 AND tRight->mixed_index NE 0) {
								if (Verbose) stop_timer (counters[FSTCHECK], TRUE);
								continue;
							}
						}

						if (Verbose) start_timer(counters[LENGTH]);
						if (hfstk.length > eps_factor * hfstk.UB) {
							if (Verbose) stop_timer (counters[FSTCHECK], TRUE);
							if (Verbose) stop_timer (counters[LENGTH], TRUE);
							continue;
						}
						if (Verbose) stop_timer (counters[LENGTH], FALSE);

						if (dist_ik + dist_jk > eps_factor * hfstk.BS) {
							if (Verbose) stop_timer (counters[FSTCHECK], TRUE);
							continue;
						}

						if (NOT extensions) {
							if (NOT upper_bound_tests (uip, &hfstk, FALSE)) continue;
						}

						/* Compute MST for terminals and Steiner points */
						if (NOT cachedMST) {
							cachedMST = hfst_mst(uip, &hfstk, FALSE);
						}
						if (hfstk.length > eps_factor * cachedMST) {
							if (Verbose) stop_timer (counters[FSTCHECK], TRUE);
							continue;
						}
					}
					if (Verbose) stop_timer (counters[FSTCHECK], FALSE);

					test_and_save_fst (uip, &hfstk);
				}
			}
		}
		free (hfstk.terms);

		if (timing NE NULL) {
			_gst_convert_delta_cpu_time (buf1, Tn);
			gst_channel_printf (timing, "Size %3d generation:    %s\n", size, buf1);
		}
	}

	free (isLeftRight);
	free (meetAngles);

	if (Verbose) {
		int genTime;
		int iterations = counters[SUBPATH] -> queries;
		stop_timer (counters[GENERATOR], FALSE);
		genTime = counters[GENERATOR] -> elapsed_time;

		fprintf (stderr, "FST-count: %d, hFSTCount: %d\n",
			uip -> ntrees, hFSTcount);

		fprintf (stderr, "\n              Queries          Pruned                Left               Time\n");
		for (i=0; i<NUM_OF_COUNTERS; i++) {
			fprintf (stderr, "%-12s: %8d - %8d (%6.2f%%) = %8d (%6.2f%%),\t%3d (%6.2f%%)\n",
			 counters[i] -> name,
			 counters[i] -> queries,
			 counters[i] -> pruned,
			 100.0*counters[i] -> pruned
			  / counters[i] -> queries,
			 counters[i] -> queries - counters[i] -> pruned,
			 100.0 * (counters[i] -> queries - counters[i]->pruned)
			  / iterations,
			 counters[i] -> elapsed_time,
			 100.0*counters[i] -> elapsed_time/genTime);

			free (counters[i]);
		}
		fprintf (stderr, "\n");
	}

	uip -> hFSTCount = hFSTcount;

	for (i = 0; i <= n; i++) {
		if (hFST_Lists[i]) {
			struct hFST *hfst = hFST_Lists[i];
			while (hfst) {
				struct hFST *tmp = hfst -> next;
				free (hfst -> terms);
				free (hfst);
				hfst = tmp;
			}
		}
	}
	free (hFST_Lists);

	/* Translate terminals back */
	for (k = 0; k < n; k++) {
		pts -> a[k] = uip -> pts_org -> a[k];
	}
	free (uip -> pts_org);

	destroy_uinfo (uip);
}

/*
 * Initialize various stuff in the uinfo structure.
 */

	static
	void
initialize_uinfo (

struct uinfo *uip
)
{
int		i;
int		j;
int		n;
long		nedges;
double		minx;
double		maxx;
double		miny;
double		maxy;
double		l1;
double		l2;
dist_t		eps_factor;
struct point *	p;
struct point *	p1;
struct point *	p2;
struct pset *	pts;
struct edge *	edges;
struct edge *	mstedges;
struct edge *	ep;
gst_metric_ptr	metric;
gst_param_ptr	params;

	params = uip -> params;
	metric = uip -> metric;

	pts = uip -> pts;
	n = pts -> n;

	/* Set relative epsilon for floating point comparisons.
	   We use the value parameter EPS_MULT_FACTOR to indicate
	   the maximum relative error that is expected.
	*/
	uip -> eps   = (params -> eps_mult_factor) * DBL_EPSILON;
	eps_factor   = 1.0 + uip -> eps;

	/* Calculate dimensions */
	minx = DBL_MAX;		maxx = -DBL_MAX;
	miny = DBL_MAX;		maxy = -DBL_MAX;
	p = &pts -> a [0];
	for (i = 0; i < n; i++, p++) {
		minx = MIN(minx, p->x);
		maxx = MAX(maxx, p->x);
		miny = MIN(miny, p->y);
		maxy = MAX(maxy, p->y);
	}

	uip -> minx = minx;
	uip -> maxx = maxx;
	uip -> miny = miny;
	uip -> maxy = miny;
	uip -> dmax = MAX(maxx-minx, maxy-miny);

	/* Calculate MST and BSD */
	/* First make complete graph */
	nedges = n;
	nedges *= (n - 1);
	nedges /= 2;
	edges = NEWA (nedges, struct edge);

	ep = edges;
	p1 = &pts -> a[0];
	for (i = 0; i < n; i++, p1++) {
		p2 = &pts -> a[0];
		for (j = 0; j < i; j++, p2++, ep++) {
			l1 = gst_distance(metric, p1 -> x, p1 -> y, p2 -> x, p2 ->y);
#if 1
			/* Look out for problems
			   with the selection of two-point-fsts */
			l2 = gst_distance(metric, p2 -> x, p2 -> y, p1 -> x, p1 ->y);
			if (NOT (eps_factor*l1 > l2 AND eps_factor*l2 > l1)) {
				fprintf(stderr, "Major difference on symmetric lengths!!!\n");
				fprintf(stderr, "%.40f\n%.40f\n\n", l1, l2);
				abort();
			}
#endif
			ep -> p1 = i;
			ep -> p2 = j;
			ep -> len = l1;
		}
	}

	/* Then compute an MST */
	mstedges = NEWA (n-1, struct edge);
	_gst_mst_edge_list (n, nedges, edges, mstedges);
	uip -> mst_length = 0;
	for (i = 0; i < n-1; i++) {
		uip -> mst_length += mstedges[i].len;
	}

	/* And finally compute the BSD */
	uip -> bsd = _gst_compute_bsd (n-1, mstedges, params -> bsd_method);

	uip -> ntrees		= 0;
	uip -> hash		= NEWA (n, struct ulist *);
	uip -> term_check	= NEWA (n, bool);

	for (i = 0; i < n; i++) {
		uip -> term_check [i] = FALSE;
		uip -> hash [i] = NULL;
	}
	uip -> list.forw	= &(uip -> list);
	uip -> list.back	= &(uip -> list);

	free (mstedges);
	free (edges);
}

/*
 * This function simply frees various stuff in the uinfo structure.
 */

	static
	void
destroy_uinfo (

struct uinfo *uip
)
{
struct ulist *	up;

	/* Disconnect FSTs from hash table. */

	for (up = uip -> list.forw;
	     up NE &(uip -> list);
	     up = up -> forw) {
		up -> next = NULL;
	}

	/* Clean up. */

	free ((char *) (uip -> hash));		uip -> hash = NULL;
	free ((char *) (uip -> term_check));	uip -> term_check = NULL;

	_gst_shutdown_bsd (uip -> bsd);
}

/*
 * Test if two points are very close to each other
 * (This test should be reasonably numerical robust.)
 */
	static
	bool
points_close (

struct point *p1,	/* IN - first point */
struct point *p2,	/* IN - second point */
double epsilon		/* IN - (relative) epsilon to use when making comparison */
)
{
double	eps, m;

	eps = epsilon;
	m = fabs(p1 -> x) + fabs(p1 -> y) + fabs(p2 -> x) + fabs(p2 -> y);
	if (m > 1.0) eps *= m;
		 
	if (( fabs(p1 -> x - p2 -> x) <= eps ) AND
	    ( fabs(p1 -> y - p2 -> y) <= eps )) {
		return TRUE;
	}
	return FALSE;
}

/*
 * Compute MST for terminals and Steiner points in hFST.
 */

/* This is in many senses a non-optimal function. Far too much is copied
   and it is all quite messy - should be replaced by a simple Steiner tree
   heuristic */

	static
	double
hfst_mst (

struct uinfo *uip,
struct hFST *hfst,
bool include_root
)
{
int		i, j, nterms, nsps, npts;
int		nedges;
int *		terms;
int *		tindex;
int *		orgt;
double		length, dx, dy;
struct point *	sps;
struct point *	pts;
struct edge *	edges;
struct edge *	ep;
struct edge *	mstedges;

	nterms = hfst->S;
	sps  = NEWA (nterms, struct point);
	orgt = NEWA (nterms, int);
	nsps = get_steiner_points(hfst, sps, orgt);

	/* Should we not include the top-level root? */
	if (NOT include_root) {
		--nsps;
	}

	npts = nterms + nsps;
	pts	= NEWA (npts, struct point);
	tindex	= NEWA (npts, int);

	i = 0;
	terms = hfst -> terms;
	while (*terms >= 0) {
		tindex[i] = *terms;
		pts[i].x  = 0.0;
		pts[i].y  = 0.0; /* coordinate relative to itself */
			
		++i;
		++terms;
	}

	for (j = 0; j < nsps; j++) {
		tindex[i] = orgt[j]; 
		pts[i]    = sps[j];
		++i;
	}

	nedges = (npts*(npts-1))/2;
	edges = NEWA (nedges, struct edge);

	/* Make complete graph */
	ep = edges;
	for (i = 0; i < npts; i++)
	for (j = 0; j < i; j++, ep++) {
		ep -> p1 = i;
		ep -> p2 = j;

		if ((i < nterms) AND (j < nterms)) {
			ep -> len = _gst_bsd (uip -> bsd, tindex[i], tindex[j]);
		}
		else {
			/* Coordinate distance between origin terminals */
			dx = uip -> pts -> a[tindex[j]].x - uip -> pts -> a[tindex[i]].x;
			dy = uip -> pts -> a[tindex[j]].y - uip -> pts -> a[tindex[i]].y;

			/* Coordinate distance from source to its origin terminal */
			dx -= pts[i].x;
			dy -= pts[i].y;

			/* Coordinate distance from target to its origin terminal */
			dx += pts[j].x;
			dy += pts[j].y;

			ep -> len = gst_distance(uip -> metric, 0.0, 0.0, dx, dy);
		}
	}

	mstedges = NEWA (npts-1, struct edge);
	_gst_mst_edge_list (npts, nedges, edges, mstedges);
#if 0
  /* Prune away Steiner points with degree one */
#endif
	/* Compute final heuristic tree length */
	length = 0.0;
	for (i = 0; i < npts-1; i++) {
		length += mstedges[i].len;
	}

	free (mstedges);
	free (edges);
	free (tindex);
	free (pts);
	free (orgt);
	free (sps);

	return length;
}

/*
 * Various upperbound tests
 */

	static
	bool
upper_bound_tests (

struct uinfo *	uip,
struct hFST *	hfst,
bool		all_tests
)
{
double		upper_bound;
double		distr;
double		distl;
dist_t		eps_factor;

	/* Compute a conservative epsilon due to the number of */
	/* floating point operations performed */
	eps_factor = 1.0 + 10.0 * uip -> eps;

	/* 1. upper bound */

	if (Verbose) start_timer(counters[UB1]);
	distr = closest_terminal (hfst -> right_tree, uip -> metric, &hfst -> root);
	distl = closest_terminal (hfst -> left_tree,  uip -> metric, &hfst -> root);
	upper_bound = hfst -> right_tree -> UB + hfst -> left_tree -> UB + distr + distl;
	if (eps_factor * upper_bound < hfst -> length) {
		if (Verbose)	stop_timer (counters[UB1], TRUE);
		return FALSE;
	}
	if (Verbose)	stop_timer (counters[UB1], FALSE);

	/* 2. upper bound */

	if (Verbose) start_timer(counters[UB2]);
	upper_bound = hfst -> right_tree -> UB + hfst -> left_tree -> UB
			+ MIN(distr, distl) + hfst -> BS;
	if (eps_factor * upper_bound < hfst -> length) {
		if (Verbose) stop_timer (counters[UB2], TRUE);
		return FALSE;
	}
	if (Verbose)	stop_timer (counters[UB2], FALSE);

	/* 3. upper bound */

	if (all_tests) {
		if (Verbose) start_timer(counters[UB3]);
		upper_bound = hfst_mst (uip, hfst, TRUE);
		if (eps_factor * upper_bound < hfst -> length) {
			if (Verbose)	stop_timer (counters[UB3], TRUE);
			return FALSE;
		}
		if (Verbose)	stop_timer (counters[UB3], FALSE);
	}

	return TRUE;
}

/*
 * A simple wedge test. It could be stronger.
 */

	static
	bool
wedge_test (

struct uinfo *uip,
struct hFST *hfst
)
{
int		n;
bool		prunable;
struct pset *	pts;

	pts = uip -> pts;
	n = pts -> n;

	if (hfst -> S EQ n) {
		/* This hFST does definitely not need to be extended */
		return 0;
	}

	if (uip -> metric -> lambda EQ 2) {
		/* The wedge test is not correct in this case */
		return 0;
	}

	if (Verbose)	start_timer(counters[WEDGE]);

	/* Mark all terminals which are members of this hFST */
	set_member(hfst, uip -> term_check, TRUE);

	/* Useful values */
	{
	int i;
	int leftTerms = 0, rightTerms = 0;
	struct point *point;

	gst_metric_ptr metric = uip -> metric;
	struct point *root = &hfst -> root;
	struct point *lOrients = metric -> left_dirs;
	struct point *rOrients = metric -> right_dirs;
	struct point pLeft, pRight, pSmallLeft, pSmallRight;
	int k = metric->K;
	int lambda = metric -> lambda;
	int delta = lambda - metric -> min_angle;
	int or = hfst -> ext;
	int lor = hfst -> S > 1 ? hfst -> ext_right + 1	    : or + delta;
	int ror = hfst -> S > 1 ? hfst -> ext_left  - 1 + k : or - delta + k;

	/* A few simple manipulation rules. They could be stronger. */
	int or2l = 0;
	int or2r = 0;
	int ror2 = 0;
	if (hfst -> status EQ STATE_MIXED) {
		/* The extension leg must be primary */
		or2r = 1;
		if ((lambda % 3 NE 0) AND (hfst -> mixed_index NE 0)) {
			or2l = 1;
		}

		ror2 = 1;
	}

	/* Calculate points on legal orientations */
	pLeft.x		= root->x + lOrients[ lor ].x;
	pLeft.y		= root->y + lOrients[ lor ].y;
	pRight.x	= root->x + rOrients[ ror + ror2 ].x;
	pRight.y	= root->y + rOrients[ ror + ror2 ].y;

	pSmallLeft.x	= root->x + lOrients[ or + 1 - or2l ].x;
	pSmallLeft.y	= root->y + lOrients[ or + 1 - or2l ].y;
	pSmallRight.x	= root->x + rOrients[ or - 1 + or2r + k ].x;
	pSmallRight.y	= root->y + rOrients[ or - 1 + or2r + k ].y;

	prunable = TRUE;
	for (i = 0; i < n; i++) {
		if (uip -> term_check[i]) continue;

		point = &pts -> a[i];
		if ((point->x EQ root->x) AND (point->y EQ root->y)) {
			continue; /* Just a special case which would otherwise
				     be a point in the small wedge... */
		}

		if (( left_turn(root, &pSmallRight, point)) AND
		   (right_turn(root, &pSmallLeft,  point))) {
			/* Terminal within small wedge... */
			prunable = FALSE;
			break;
		}
		else if (( left_turn(root, &pRight, point)) AND
			(right_turn(root, &pLeft,  point))) {
			/* Terminal within large wedge and not in small wedge... */
			if (NOT right_turn(root, &pSmallLeft, point)) {
				leftTerms++; /* Terminal in left wedge */
			}
			else {
				rightTerms++; /* Terminal in right wedge */
			}

			if (leftTerms AND rightTerms) {
				prunable = FALSE;
				break;
			}
		}
	}
	}

	set_member(hfst, uip->term_check, FALSE);
	if (Verbose) stop_timer (counters[WEDGE], prunable);

	return (prunable);
}

/*
 * Compute distance between the root of an half FST and some other point.
 * Do this in a numerically careful manner.
 */

	static
	double
hfst_root_distance (

struct uinfo *	uip,		/* IN - UFST info */
struct hFST *	hfst,		/* IN - half FST */
struct point *	p)		/* IN - point */
{
double 		dx, dy;

	dx  = (uip -> pts -> a[ hfst -> origin_term ].x - p -> x);
	dy  = (uip -> pts -> a[ hfst -> origin_term ].y - p -> y);
	dx += hfst -> droot.x;
	dy += hfst -> droot.y;

	return gst_distance(uip -> metric, 0.0, 0.0, dx, dy);
}

/*
 * Given a half FST and its two children, this function checks that no 
 * other points are close enough to the edges connecting their roots to
 * make the construction non-optimal.
 */

	static
	bool
edge_lune_test (

struct uinfo *	uip,		/* IN - ... */
struct hFST *	hfsti,		/* IN - left half FST */
struct hFST *	hfstj,		/* IN - right half FST */
struct hFST *	hfstk,		/* IN - current half FST */
double		d1,		/* IN - distance from root of hfsti to root of hfstk */
double		d2)		/* IN - distance from root of hfstj to root of hfstk */
{
int		i;
int		n;
double		sDist, d, sqr_d;
dist_t		eps;
struct pset *	pts;
struct point *	p;

	if ((d1 EQ 0.0) OR (d2 EQ 0.0)) {
		return TRUE; /* trivially passed */
	}

	if ((hfstk -> S EQ 2) AND (_gst_is_mst_edge(uip -> bsd, hfsti -> index, hfstj -> index))) {
		return TRUE; /* this is a MST edge - trivially passed */
	}

	pts = uip -> pts;
	n = pts -> n;
	p = &pts -> a[0];

	d = MAX(d1, d2);
	sqr_d = d * d;

	/* Set up a conservative epsilon and adjust input distances */
	eps = 5.0 * uip -> eps;
	if (d > 1.0) eps *= d;
	d1 -= eps;
	d2 -= eps;

	for (i = 0; i < n; i++, p++) {
		/* Test if this terminal is too far away */
		if (sqr_dist(&hfstk -> root, p) > sqr_d) continue;

		sDist = hfst_root_distance(uip, hfstk, p);
		if (	(  (sDist < d1) AND (hfst_root_distance(uip, hfsti, p) < d1))
		     OR (  (sDist < d2)	AND (hfst_root_distance(uip, hfstj, p) < d2))
		  ) {
			return FALSE; /* did not pass lune test */
		}
	}
	return TRUE; /* passed */
}

/*
 * Check if the elements of two lists of terminals are the same.
 * The function assumes that the two lists given have the same length
 * and that their contents are sorted!
 */

	static
	bool
identical_lists (

const int *	list1,		/* IN - first list of terminals */
const int *	list2		/* IN - second list of terminals */
)
{
int x1;
int x2;

	while ((x1 = *list1++) EQ (x2 = *list2++)) {
		if (x1 EQ -1) {
			return TRUE;
		}
	}

	return FALSE;
}

/*
 * Map all terminal numbers back to their original values.  (i.e., with
 * the duplicate terminals reinstated.	We must renumber the terminals
 * of each RFST also.
 */

	static
	void
renumber_terminals (

struct uinfo *		uip,		/* IN/OUT - global RFST info */
struct pset *		to_pts,		/* IN - point set to map to (orig) */
int *			rev_map		/* IN - map from new to old terms */
)
{
int			i;
int			j;
int			from_n;
int			to_n;
struct pset *		from_pts;
struct ulist *		up1;
struct ulist *		up2;
struct full_set *	fsp;
int *			tlist;
struct pset *		terms;
struct point *		p1;

	from_pts	= uip -> pts;
	from_n		= from_pts -> n;
	to_n		= to_pts -> n;

	/* Restore original set of terminals. */
	uip -> pts = to_pts;

	/* Renumber terminals in each FST. */
	up2 = &(uip -> list);
	for (up1 = up2 -> forw; up1 NE up2; up1 = up1 -> forw) {
		fsp = up1 -> fst;
		tlist = fsp -> tlist;
		terms = fsp -> terminals;
		p1 = &(terms -> a [0]);
		for (i = 0; i < terms -> n; i++, p1++) {
			j = tlist [i];
			FATAL_ERROR_IF ((j < 0) OR (j >= from_n));
			j = rev_map [j];
			FATAL_ERROR_IF ((j < 0) OR (j >= to_n));
			tlist [i] = j;
		}
	}
}

/*
 * Link all of the FSTs together into one long list and number them
 * each sequentially.  Free the doubly-linked ulists as we go.
 */

	static
	void
build_fst_list (

struct uinfo *		uip		/* IN - global RFST info */
)
{
int			i;
struct ulist *		up1;
struct ulist *		up2;
struct ulist *		up3;
struct full_set *	fsp;
struct full_set **	hookp;

	hookp = &(uip -> full_sets);

	up2 = &(uip -> list);
	i = 0;
	for (up1 = up2 -> forw; up1 NE up2; ) {
		fsp = up1 -> fst;
		fsp -> tree_num = i++;
		*hookp = fsp;
		hookp = &(fsp -> next);
		up3 = up1;
		up1 = up1 -> forw;
		free ((char *) up3);
	}
	*hookp = NULL;

	uip -> list.forw = up2;
	uip -> list.back = up2;

	/* Make it easy to add zero-length FSTs onto the end. */
	uip -> ntrees	= i;
	uip -> hookp	= hookp;
}

/*
 * This routine adds one RFST (of zero length) connecting the first
 * terminal of each duplicate terminal group to each terminal that
 * was deleted during the major processing.
 */

	static
	void
add_zero_length_fsts (

struct uinfo *		uip,		/* IN - global RFST info */
int			ndg,		/* IN - number of duplicate groups */
int **			list		/* IN - list of duplicate groups */
)
{
int			i;
int			t;
int			u;
int *			ip1;
int *			ip2;
int *			tlist;
struct pset *		pts;
struct pset *		terms;
struct edge *		edges;
struct full_set *	fsp;

	pts	= uip -> pts;

	for (i = 0; i < ndg; i++) {
		ip1 = list [i];
		ip2 = list [i + 1];
		if (ip1 + 2 > ip2) {
			/* Group with fewer than two members! */
			FATAL_ERROR;
		}
		t = *ip1++;
		while (ip1 < ip2) {
			u = *ip1++;

			tlist = NEWA (2, int);
			tlist [0] = t;
			tlist [1] = u;

			terms = NEW_PSET (2);
			terms -> n = 2;
			terms -> a [0]	= pts -> a [t];
			terms -> a [1]	= pts -> a [u];

			edges = NEW (struct edge);
			edges -> p1	= 0;
			edges -> p2	= 1;
			edges -> len	= 0.0;

			fsp = NEW (struct full_set);

			fsp -> next	 = NULL;
			fsp -> tree_num	 = (uip -> ntrees)++;
			fsp -> tree_len	 = 0.0;
			fsp -> tlist	 = tlist;
			fsp -> terminals = terms;
			fsp -> steiners	 = NULL;
			fsp -> nedges	 = 1;
			fsp -> edges	 = edges;

			*(uip -> hookp) = fsp;
			uip -> hookp	= &(fsp -> next);
		}
	}
}

	static
	int
build_ufst_graph (

struct uinfo *		uip,
struct hFST *		hfst,		/* IN - half FST */
int *			new_tlist,
int *			term_num,
struct point *		new_steiners,
int *			steiner_num,
struct edge **		edges,
int			size,
bool			include_root
)
{
int	r;
int	l;
int	n;
dist_t	l1;
dist_t	l2;

	if (hfst -> S EQ 1) { /* We are at a terminal */
		n = *term_num;
		new_tlist[n] = hfst -> index;
		(*term_num)++;
	}
	else {
		r = build_ufst_graph (uip, hfst -> right_tree, new_tlist, term_num, new_steiners, steiner_num, edges, size, TRUE);
		l = build_ufst_graph (uip, hfst -> left_tree, new_tlist, term_num, new_steiners, steiner_num, edges, size, TRUE);
		l1 = sqrt(sqr_dist(&hfst -> root, &hfst -> right_tree -> root));
		l2 = sqrt(sqr_dist(&hfst -> root, &hfst -> left_tree -> root));
		n = *steiner_num + size;

		if (include_root) {
			(*edges) -> p1	= r;
			(*edges) -> p2	= n;
			(*edges) -> len	= l1;
			(*edges)++;

			(*edges) -> p1	= n;
			(*edges) -> p2	= l;
			(*edges) -> len	= l2;
			(*edges)++;

			/* Translate back */
			new_steiners[*steiner_num].x = hfst -> root.x + uip -> mean.x;
			new_steiners[*steiner_num].y = hfst -> root.y + uip -> mean.y;
			(*steiner_num)++;
		}
		else {
			(*edges) -> p1	= r;
			(*edges) -> p2	= l;
			(*edges) -> len	= l1 + l2;
			(*edges)++;
		}
	}

	return (n);
}

	static
	dist_t
test_and_save_fst (

struct uinfo *	uip,		/* IN/OUT - The global UFST info */
struct hFST *	fst		/* IN - FST to be saved */
)
{
int			i, j, k;
int			nstein;
int			nedges;
int			size;
int			steiner_num;
int			term_num;
int *			terms;
struct pset *		pts;
int *			tlist;
struct ulist *		up;
struct ulist **		hookp;
struct ulist *		up1;
struct ulist *		up2;
int *			new_tlist;
struct pset *		new_terms;
struct pset *		new_steiners;
struct full_set *	fsp;
struct edge *		edges;
struct edge *		ep;
gst_param_ptr		params;

	size	= fst -> S;
	pts	= uip -> pts;
	terms	= fst -> terms;

	/* General duplicate test.  We use a hash table, for speed.	*/
	/* For correctness, the hash function must not depend upon the	*/
	/* order of the terminals in the FST.  A simple checksum has	*/
	/* this property and tends to avoid favoring any one bucket.	*/

	/* Compute hash and prepare for rapid set comparison. */
	k = 0;
	for (i = 0; i < size; i++) {
		j = terms [i];
		uip -> term_check [j] = TRUE;
		k += j;
	}
	k %= pts -> n;

	hookp = &(uip -> hash [k]);
	for (;;) {
		up = *hookp;
		if (up EQ NULL) break;
		if (up -> size < size) {
			/* rest are smaller */
			up = NULL;
			break;
		}
		if (up -> size EQ size) {
			tlist = up -> fst -> tlist;
			for (i = 0; ; i++) {
				if (i >= size) goto found_ufst;
				if (NOT uip -> term_check [tlist [i]]) break;
			}
		}
		hookp = &(up -> next);
	}

found_ufst:

	for (i = 0; i < size; i++) {
		uip -> term_check [terms [i]] = FALSE;
	}

	if (up EQ NULL) {
		uip -> ntrees++;
	}
	else {
		/* An FST for these terminals already exists. */
		fsp = up -> fst;
		if (fsp -> tree_len <= fst -> length) {
			return (fsp -> tree_len);
		}
		/* The new one is shorter!  Delete the old one. */
		*hookp = up -> next;
		up2 = up -> forw;
		up1 = up -> back;
		up2 -> back = up1;
		up1 -> forw = up2;
		free ((char *) (fsp -> terminals));
		free ((char *) (fsp -> steiners));
		free ((char *) (fsp -> edges));
		free ((char *) fsp);
		free ((char *) up);
	}

	/* Build FST graph in edge list form. */

	new_tlist = NEWA (size, int);

	if (size <= 2) {
		nstein = 0;
		nedges = 1;
	}
	else if (fst -> type == TYPE_CROSS) {
		nstein = 1;
		nedges = 4;
	}
	else {
		nstein = size - 2;
		nedges = 2 * size - 3;
	}

	params = uip -> params;
	if (params -> include_corners AND fst -> type == TYPE_CORNER) {
		nedges++;
		nstein++;
	}

	new_steiners = (nstein EQ 0) ? NULL : NEW_PSET (nstein);
	edges = NEWA (nedges, struct edge);

	steiner_num = 0;
	ep = edges;

	if (fst -> type == TYPE_CROSS) {
		steiner_num = 1;

		/* Translate back */
		new_steiners -> a[0].x = fst -> root.x + uip -> mean.x;
		new_steiners -> a[0].y = fst -> root.y + uip -> mean.y;

		new_tlist[0] = fst -> right_tree -> right_tree -> index;
		new_tlist[1] =  fst -> right_tree -> left_tree -> index;
		new_tlist[2] =  fst -> left_tree -> right_tree -> index;
		new_tlist[3] =  fst -> left_tree -> left_tree -> index;

		for (i = 0; i < 4; i++) {
			ep -> p1 = i;
			ep -> p2 = 4;
			j = new_tlist [i];
			ep -> len = sqrt(sqr_dist(&(uip -> pts -> a [j]), &new_steiners -> a[0]));
			++ep;
		}
	}
	else {
		term_num = 0;
		build_ufst_graph (uip,
			fst,
		   	new_tlist,
			&term_num,
			&new_steiners -> a[0],
			&steiner_num,
			&ep,
			size,
			params -> include_corners AND (fst -> type == TYPE_CORNER));
	}

	if (new_steiners NE NULL) {
		new_steiners -> n = steiner_num;
	}

	new_terms = NEW_PSET (size);
	new_terms -> n = size;
	for (i = 0; i < size; i++) {
		j = new_tlist [i];

		/* Translate back */
		new_terms -> a[i] = uip -> pts_org -> a[j];
	}

#if 0
	printf ("FST dump:\n");
	printf ("Terminals:\n");
	for (i = 0; i < size; i++) {
		struct point *p = &new_terms -> a[i];
		printf ("(%f, %f) - %d\n", p->x, p->y, new_tlist [i]);
	}
	printf ("Steiner points:\n");
	for (i = 0; i < nstein; i++) {
		struct point *p = &new_steiners -> a[i];
		printf ("(%f, %f)\n", p->x, p->y);
	}
	printf ("Edges:\n");
	for (i = 0; i < nedges; i++) {
		struct edge *e = &edges[i];
		printf ("(%d, %d) - %f\n", e->p1, e->p2, e->len);
	}
	printf ("Length: %f\n", fst -> length);
#endif
	fsp = NEW (struct full_set);

	fsp -> next		= NULL;
	fsp -> tree_num		= 0;
	fsp -> tree_len		= fst -> length;
	fsp -> tlist		= new_tlist;
	fsp -> terminals	= new_terms;
	fsp -> steiners		= new_steiners;
	fsp -> nedges		= nedges;
	fsp -> edges		= edges;

	up = NEW (struct ulist);

	up2 = &(uip -> list);
	up1 = up2 -> back;
	up -> back	= up1;
	up -> forw	= up2;
	up -> next	= uip -> hash [k];
	up -> size	= size;
	up -> fst	= fsp;

	up1 -> forw	= up;
	up2 -> back	= up;
	uip -> hash [k] = up;

	return (fst -> length);
}

/*
 * Update list of terminals spanned by an hFST
 */

	static
	void
update_terms (

struct hFST *	hfst
)
{
int		i;
int		x1;
int		x2;
int		size;
int *		terms;
int *		lterms;
int *		rterms;

	size = hfst -> S;

	if (hfst->terms EQ NULL) {
		hfst->terms = NEWA (size + 1, int);
	}

	terms  = hfst -> terms;
	lterms = hfst -> left_tree -> terms;
	rterms = hfst -> right_tree -> terms;

	x1 = *lterms; x2 = *rterms;
	for (i=0; i < size; i++) {
		if (x1 < x2) {
			*(terms++) = x1;
			x1 = *(++lterms);
			if (x1 < 0) {
				x1 = INT_MAX;
			}
		}
		else {
			*(terms++) = x2;
			x2 = *(++rterms);
			if (x2 < 0) {
				x2 = INT_MAX;
			}
		}
	}
	*terms = -1;
}

/*
 * Return Steiner points spanned by hFST
 * Steiner point coordinates are given relative to their origin terminals.
 */

	static
	int
get_steiner_points (

struct hFST *		hfst,	/* IN - half FST */
struct point *		sps,	/* OUT - list of Steiner points in half FST */
int *			orgt	/* OUT - list origin terminal indicies */
)
{
int	r;
int	l;

	if (hfst -> right_tree EQ NULL) {
		return 0;
	}
	else {
		r = get_steiner_points (hfst -> right_tree, sps, orgt);
		sps  += r;
		orgt += r;
		l = get_steiner_points (hfst -> left_tree, sps, orgt);
		sps  += l;
		orgt += l; 

		*sps++  = hfst -> droot;       /* relative coordinate */
		*orgt++ = hfst -> origin_term; /* terminal origin */

		return 1 + r + l;
	}
}

/*
 * For each terminal in the hFST, the corresponding index in the membArray is
 * set to the value of the flag.
 */

	static
	void
set_member (

struct hFST *		hfst,		/* IN - half FST */
bool *			membArray,	/* IN/OUT - hFST members */
bool			flag		/* IN - member setting */
)
{
	if (hfst -> right_tree EQ NULL) {
		membArray [hfst -> index] = flag;
	}
	else {
		set_member (hfst -> right_tree, membArray, flag);
		set_member (hfst -> left_tree, membArray, flag);
	}
}

/*
 * Determine whether the two given half FSTs are disjoint (do not share any
 * terminals).
 */

	static
	bool
is_disjoint (

struct hFST *	first,
struct hFST *	second
)
{
int	x1;
int	x2;
int *	l1;
int *	l2;

	l1 = first -> terms; l2 = second -> terms;
	x1 = *l1; x2 = *l2;
	while (x1 NE x2) {
		if (x1 < x2) {
			x1 = *(++l1);
			if (x1 EQ -1) {
				return TRUE;
			}
		}
		else {
			x2 = *(++l2);
			if (x2 EQ -1) {
				return TRUE;
			}
		}
	}

	return FALSE;
}

/*
 * Compute distance to closest terminal spanned by a half FST.
 */

	static
	double
closest_terminal (

struct hFST *		hfst,		/* IN - half FST */
gst_metric_ptr		metric,		/* IN - metric information */
struct point *		p		/* IN - a point */
)
{
double rdist;
double ldist;

	if (hfst -> right_tree EQ NULL) {
		return gst_distance (metric, hfst -> root.x, hfst -> root.y, p -> x, p -> y);
	}
	else {
		rdist = closest_terminal (hfst -> right_tree, metric, p);
		ldist = closest_terminal (hfst -> left_tree, metric, p);
	}

	if (rdist < ldist) return rdist;
	return ldist;
}

/*
 * Get bottleneck Steiner distance between two half FSTs.
 */

	static
	double
get_bsd (

struct hFST *	first,
struct hFST *	second,
struct bsd *	BSD
)
{
double rbsd;
double lbsd;

	if (first -> right_tree EQ NULL) {
		if (second -> right_tree EQ NULL) {
			return _gst_bsd (BSD, first -> index, second -> index);
		}
		rbsd = get_bsd(first, second -> right_tree, BSD);
		lbsd = get_bsd(first, second -> left_tree, BSD);
	}
	else {
		rbsd = get_bsd(second, first -> right_tree, BSD);
		lbsd = get_bsd(second, first -> left_tree, BSD);
	}

	if (rbsd < lbsd) return rbsd;
	return lbsd;
}

/*
 * This function simply finds the legal extensions from a half FST, but
 * nevertheless it is currently quite chaotic...
 */

	static
	int
setup_extensions (
struct hFST *		hfst,
gst_metric_ptr		metric,
int			middleSpan
)
{
	/* This drawing might help to understand the tests.
	   The numbers indicate family relations.

			     |
			   0 | Ext
			     |
			     o
			    / \
		     lDir1 /   \ rDir2
	     lDir2	  /     \	rDir1
		-------Left   Right-------
			 |	 |
			 |	 |
			 |	 |
		   lDir0 |	 | rDir0
	*/

	/* Whenever two extensions are possible then 'ext' is set to the
	   first (counterclockwise) extension orientation and it is assumed
	   elsewhere in the code that the other extension follows just after.
	   The variable 'status' is set to the state of the hFST corresponding
	   to the first possible extension. */

	/* For convenience */
	int k = metric -> K;
	int lambda = metric -> lambda;
	int a120 = metric -> a120; /* Largest angle not greater than 120 degrees */
	struct hFST *ltree = hfst -> left_tree;
	struct hFST *rtree = hfst -> right_tree;

	/* Is the minimum index in the left subtree? */
	bool minIndexLeft = *(ltree -> terms) < *(rtree -> terms);

	int type = lambda % 3;
	switch (type) {
	case 0: { /* Lambda = 3m */
		/* This case is the easiest because an edge in a mixed tree
		   is always either primary or secondary - it cannot be both */

		if (middleSpan EQ a120) {
			/* The most general case */
			if (minIndexLeft) {
				hfst -> status = hfst -> left_tree -> status;
				if (hfst -> status EQ STATE_MIXED)
					return 0;

				hfst -> ext = (hfst -> ext_left + lambda + 2*a120) % k;
				hfst -> mixable = FALSE;
			}
			else {
				hfst -> ext = (hfst -> ext_right + lambda + a120) % k;
				hfst -> status = rtree -> status;
				hfst -> mixable = rtree -> mixable;
			}
		}
		else if (middleSpan EQ (a120-1)) {
			return 0;
		}
		else { /* (middleSpan EQ (a120+1)) */
			if (NOT minIndexLeft OR NOT (ltree -> mixable)) {
				return 0;
			}

			hfst -> ext = (hfst -> ext_left + lambda + 2*a120) % k;
			hfst -> status = STATE_MIXED;
			hfst -> mixable = TRUE;
		}
		return 1;
	}
	case 1:	  /* Lambda = 3m + 1 */
	case 2: { /* Lambda = 3m + 2 */
		struct hFST *mTree = minIndexLeft ? ltree : rtree;

		/* Find the possible extensions - one or two.
		   Remember that a120 is the largest angle not greater
		   than 120 degrees */
		int nExt, priExt, secExt;
		if ((middleSpan EQ a120) ^ (type EQ 2)) {
			nExt = 1;
			priExt = (hfst -> ext_right + lambda + a120 + (type EQ 1)) % k;
			secExt = priExt;
		}
		else {
			nExt = 2;
			priExt = (hfst -> ext_right + lambda + a120) % k;
			secExt = (priExt + 1) % k;
		}

		if (hfst -> S EQ 2) { /* Two terminals */
			/* The simple case - always clean extensions */
			hfst -> ext = priExt;
			hfst -> status = STATE_CLEAN;

			hfst -> right_legs[0] = FALSE;
			hfst -> right_legs[1] = FALSE;
			hfst -> right_legs[2] = FALSE;
			hfst -> all_left_used = FALSE;

			if (minIndexLeft) {
				hfst -> right_legs[2] = TRUE;
			}
			else {
				hfst -> all_left_used = TRUE;
			}

			return nExt;
		}
		else { /* At least one tree. */
			/* Gather some orientation info about the two subtrees */
			int nEquals, equals0, equals1, equals2;
			int lDir0, lDir1, lDir2, rDir0, rDir1, rDir2;
			lDir1 = ltree -> ext % lambda;
			rDir2 = rtree -> ext % lambda;
			if (ltree -> S NE 1) { /* A real tree */
				lDir0 = ltree -> ext_right % lambda;
				lDir2 = ltree -> ext_left % lambda;
			}
			else { /* 'Steal' some harmless directions */
				lDir0 = rtree -> ext_left % lambda;
				lDir2 = rDir2;
			}

			if (rtree -> S NE 1) { /* A real tree */
				rDir0 = rtree -> ext_left % lambda;
				rDir1 = rtree -> ext_right % lambda;
			}
			else { /* 'Steal' some harmless directions */
				rDir0 = ltree -> ext_right % lambda;
				rDir1 = lDir1;
			}

			/* If mixed then use the primary edges... */
			if (mTree -> status EQ STATE_MIXED) {
				if (minIndexLeft) {
					hfst -> mixed_index = (ltree -> mixed_index + 1) % 3;
					if (hfst -> mixed_index EQ 0) {
						lDir0 = (lDir2 + a120) % lambda;
					}
				}
				else {
					hfst -> mixed_index = (rtree -> mixed_index + 2) % 3;
					if (hfst -> mixed_index EQ 1) {
						rDir1 = (rDir0 + a120) % lambda;
					}
				}
			}

			/* How many differs */
			equals0 = (lDir0 EQ rDir0);
			equals1 = (lDir1 EQ rDir1);
			equals2 = (lDir2 EQ rDir2);
			nEquals = equals0 + equals1 + equals2;

			if (nEquals < 2) {
				/* Incompatible subtrees */
				return 0;
			}

			if (minIndexLeft) {
				hfst -> right_legs[2] = TRUE;
				if (rtree -> S > 1) {
					hfst -> right_legs[0] = TRUE;
					hfst -> right_legs[1] = TRUE;
				}
				else {
					hfst -> right_legs[0] = ltree -> right_legs[2];
					hfst -> right_legs[1] = ltree -> right_legs[0];
				}

				if (ltree -> S > 1) {
					hfst -> all_left_used = TRUE;
				}
				else {
					hfst -> all_left_used = FALSE;
				}
			}
			else {
				hfst -> all_left_used = TRUE;
				hfst -> right_legs[0] = rtree -> right_legs[1];
				hfst -> right_legs[1] = rtree -> right_legs[2];
				hfst -> right_legs[2] = rtree -> right_legs[0];
			}

			if (nEquals EQ 3) { /* All legs are equal */
				if (mTree -> status EQ STATE_CLEAN) {
					if ((priExt % lambda) EQ rDir0) {
						/* Clean tree with priExt */
						hfst -> status = STATE_CLEAN;
						hfst -> ext = priExt;
					}
					else {
						/* If the extension direction has
						   not been used earlier as primary
						   then a mixed tree is possible? */
						if (minIndexLeft AND NOT (mTree -> all_left_used)) {
							hfst -> status = STATE_MIXED;
							hfst -> ext = priExt;
							hfst -> mixed_index = 0;

							return 2;
						}
						else { /* Only the clean tree is legal. */
							hfst -> status = STATE_CLEAN;
							hfst -> ext = secExt;
						}
					}
					return 1;
				}
				else { /* STATE_MIXED */
					if (minIndexLeft) {
						/* Clean side is to the right. It must be primary! */
						if ((hfst -> mixed_index EQ 2) OR
						    (rtree -> S > 1)) {
							return 0;
						}
					}
					/* else:
						Clean side is to the left
						and must be primary (i.e. legal) */
				}
			}  /* nEquals EQ 3 */
			else { /* nEquals EQ 2 -- One leg differs */
				int lDir, rDir;
				int index;
				if (NOT equals0) {
					lDir = lDir0; rDir = rDir0; index = 0;
				}
				else if (NOT equals1) {
					lDir = lDir1; rDir = rDir1; index = 1;
				}
				else {
					lDir = lDir2; rDir = rDir2; index = 2;
				}

				if (minIndexLeft) {
					if (NOT (rDir EQ ((lDir + 1) % lambda))) {
						/* The secondary edge is not in the tree to
						   the right */
						return 0;
					}
				}
				else {
					if (rtree -> all_left_used) {
						return 0;
					}

					if (NOT (    rtree -> status EQ STATE_CLEAN
						 AND index EQ 1
						 AND (rDir EQ ((lDir + 1) % lambda)))) {
						return 0;
					}
				}

				if (mTree -> status EQ STATE_CLEAN) {
					if (minIndexLeft) {
						if (ltree -> right_legs[(index + 2) % 3]) {
							return 0;
						}
					}
					else {
						if (index NE 1 OR rtree -> all_left_used) {
							return 0;
						}
					}

					hfst -> mixed_index = index;
				}
				else { /* STATE_MIXED */
					/* It has to be the same index as earlier encountered */
					if (hfst -> mixed_index NE index) {
						return 0;
					}
				}
			} /* nEquals EQ 2 */
		}

		/* We only get to this point if we are in a mixed state.
		   This means that there is only one possible
		   extension (the primary) */

		if (hfst -> mixed_index EQ 0) {
			hfst -> ext = priExt;
		}
		else {
			hfst -> ext = ltree -> right_tree ? ltree -> right_tree -> ext
							  : rtree -> left_tree	-> ext;
		}

		/* The new extension must be a locally legal extension */
		if (NOT ((priExt EQ hfst -> ext) OR (secExt EQ hfst -> ext))) {
			return 0;
		}

		hfst -> status = STATE_MIXED;
		return 1;
	}
	default:
		FATAL_ERROR;
	}

	FATAL_ERROR;
	return 0;
}

/*
 *  Allocate and initialize a timer (used for statistics).
 */
	static
	struct time_it *
create_time_it (

char *		name		/* IN - name of timer */
)
{
struct time_it *	ti;

	ti = NEW (struct time_it);
	ti -> name		= name;
	ti -> queries		= 0;
	ti -> pruned		= 0;
	ti -> elapsed_time	= 0;

	return (ti);
}

/*
 *  Start a timer subperiod (used for statistics).
 */
	static
	void
start_timer (

struct time_it *	ti	/* IN/OUT - timer information */
)
{
	ti -> queries++;
	ti -> temp_time = _gst_get_cpu_time ();
}

/*
 *  Stop a timer subperiod (used for statistics). A boolean value indicates success.
 */
	static
	void
stop_timer (

struct time_it *	ti,	/* IN/OUT - timer information */
bool			pruned	/* IN - boolean to indicate success */
)
{
	ti -> elapsed_time += _gst_get_cpu_time () - ti -> temp_time;

	if(pruned) {
		ti -> pruned++;
	}
}

#if 0
/*
 *  For debug purposes.
 */
	static
	void
dump_hfst (

struct hFST *	hfst	/* IN - */
)
{
int	*terms;

	printf ("Terminals (%d): ", hfst -> S);
	terms = hfst -> terms;
	while (*terms >= 0) {
		printf ("%d ", *terms++);
	}
	printf ("\n");
	printf ("Root: (%f, %f)\n", hfst -> root.x, hfst -> root.y);
	printf ("Length: %f\n", hfst -> length);

	printf ("Left tree  (%d): ", hfst -> left_tree -> S);
	terms = hfst -> left_tree -> terms;
	while (*terms >= 0) {
		printf ("%d ", *terms++);
	}
	printf ("\n");
	printf ("Right tree (%d): ", hfst -> right_tree -> S);
	terms = hfst -> right_tree -> terms;
	while (*terms >= 0) {
		printf ("%d ", *terms++);
	}
	printf ("\n");

	printf ("ext: %d, ext_left: %d, ext_right: %d\n", hfst -> ext, hfst -> ext_left, hfst -> ext_right);
	printf ("UB: %f, BS: %f\n", hfst -> UB, hfst -> BS);
}
#endif
