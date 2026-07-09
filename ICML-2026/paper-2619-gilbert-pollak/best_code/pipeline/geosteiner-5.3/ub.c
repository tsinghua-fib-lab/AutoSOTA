/***********************************************************************

	$Id: ub.c,v 1.22 2022/11/19 13:45:55 warme Exp $

	File:	ub.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1996, 2022 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Heuristic for turning a fractional LP solution (a lower bound)
	into a feasible Steiner tree (an upper bound).  If such a
	Steiner tree can be found, and is the best seen so far, it
	becomes the new best feasible integer solution.

************************************************************************

	Modification Log:

	a-1:	07/17/96	warme
		: Created.
	a-2:	09/06/97	warme
		: Completely replaced this algorithm with a new one
		:  suggested by Martin Zachariasen.
	b-1:	02/28/2001	warme
		: Changes for 3.1 release.
		: Major upgrades to the upper bound heuristic.
		: Support multiple rankings.
		: Add Martins latest tweaks:
		:   Now doing diversification, a number of techniques
		:    are supported, including greedy local search (the
		:    default).
		:   Intensify search if getting close to best known
		:    solution.
	c-1:	08/05/2002	benny
		: Some changes for library release.
		: Uses parameters.
		: Saves multiple solutions.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.  Upgrade fatals.
	e-4:	11/18/2022	warme
		: Fix -Wall issues.

************************************************************************/

#include "ub.h"

#include "bb.h" /* Only to get FUZZ */
#include "dsuf.h"
#include "emst.h"
#include "fatal.h"
#include <float.h>
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "metric.h"
#include "rmst.h"
#include "solver.h"
#include "steiner.h"
#include <string.h>

/*
 * Global Routines
 */

bool			_gst_compute_heuristic_upper_bound (
					double *		x,
					struct gst_solver *	solver);
void			_gst_shutdown_heuristic_upper_bound (
						struct ubinfo *	ubip);
struct ubinfo *		_gst_startup_heuristic_upper_bound (
					struct gst_hypergraph *	cip);


/*
 * External References
 */

	/* none */


/*
 * Local Routines
 */

static int *		compute_FST_ranking (int, dist_t *, dist_t *);
static void		sort_fst_list_by_lp_and_rank (int *,
						      int,
						      double *,
						      int *);
static void		sort_fst_list_by_rank (int *, int, int *);
static int *		sorted_mst_edges (struct gst_hypergraph *);
static void		try_trees (int *, int, struct gst_solver *);
static dist_t		ub_kruskal (int *, int, bitmap_t *, struct gst_solver *);

/*
 * This routine pre-computes some useful information for the upper
 * bound heuristic:
 *
 *	- A set of rankings, each of which establishes a total ordering
 *	  of the FSTs.
 *
 *	- The list of all MST edges, in increasing order by length.
 */

	struct ubinfo *
_gst_startup_heuristic_upper_bound (

struct gst_hypergraph *	cip		/* IN - compatibility info */
)
{
int			i;
int			nranks;
int			n;
int *			ranking;
struct ubinfo *		ubip;
struct pset *		terms;
dist_t *		fst_len;
dist_t *		mst_len;

	ubip = NEW (struct ubinfo);

	n = cip -> num_edges;

	fst_len = NEWA (n, dist_t);
	mst_len = NEWA (n, dist_t);

	/* Get FST lengths into an array... */
	for (i = 0; i < n; i++) {
		fst_len [i] = cip -> cost [i];
	}

	/* No rankings yet... */
	nranks = 0;

	/* Compute MST length of each FST... */
	if (_gst_is_euclidean (cip) AND
	    (cip -> full_trees NE NULL)) {
		for (i = 0; i < n; i++) {
			terms = cip -> full_trees [i] -> terminals;
			mst_len [i] = _gst_euclidean_mst_length (terms);
		}
		ranking = compute_FST_ranking (n, fst_len, mst_len);
		ubip -> rankings [nranks++] = ranking;
	}
	else if (_gst_is_rectilinear (cip) AND
		 (cip -> full_trees NE NULL)) {
		for (i = 0; i < n; i++) {
			terms = cip -> full_trees [i] -> terminals;
			mst_len [i] = _gst_rect_mst_length (terms);
		}
		ranking = compute_FST_ranking (n, fst_len, mst_len);
		ubip -> rankings [nranks++] = ranking;
	}

	/* Pretend each edge in the MST has length 1... */
	for (i = 0; i < n; i++) {
		mst_len [i] = cip -> edge_size [i] - 1;
	}
	ranking = compute_FST_ranking (n, fst_len, mst_len);
	ubip -> rankings [nranks++] = ranking;

	ubip -> num_rankings = nranks;

	ubip -> mst_edges = sorted_mst_edges (cip);

	free ((char *) mst_len);
	free ((char *) fst_len);

	return (ubip);
}

/*
 * This routine produces a ranking of the given FSTs, sorted in increasing
 * order by the given (numerator / denominator) values.
 */

	static
	int *
compute_FST_ranking (

int			n,		/* IN - number of FSTs */
dist_t *		num,		/* IN - numerator of values */
dist_t *		den		/* IN - denominator of values */
)
{
int			i;
int			h;
int			t1;
int			t2;
int *			p1;
int *			p2;
int *			p3;
int *			p4;
int *			endp;
int *			array;
int *			ranking;
dist_t			key1;
dist_t			key2;
dist_t			key3;
dist_t			key4;

	array = NEWA (n, int);
	for (i = 0; i < n; i++) {
		array [i] = i;
	}

	endp = &array [n];

	for (h = 1; h <= n; h = 3*h+1) {
	}

	do {
		h = h / 3;
		p4 = &array [h];
		p1 = p4;
		while (p1 < endp) {
			t1 = *p1;
			key1 = num [t1];
			key2 = den [t1];
			p2 = p1;
			while (TRUE) {
				p3 = (p2 - h);
				t2 = *p3;
				key3 = num [t2];
				key4 = den [t2];
				if (key3 * key2 <= key1 * key4) break;
				*p2 = t2;
				p2 = p3;
				if (p2 < p4) break;
			}
			*p2 = t1;
			++p1;
		}
	} while (h > 1);

	/* Produce array that ranks each FST... */
	ranking = NEWA (n, int);

	for (i = 0; i < n; i++) {
		ranking [array [i]] = i;
	}

	free ((char *) array);

	return (ranking);
}

/*
 * This routine finds all of the MST edges (edges of cardinality 2) and
 * sorts them in increasing order by length.
 */

	static
	int *
sorted_mst_edges (

struct gst_hypergraph *	cip		/* IN - compatibility info */
)
{
int			i;
int			e;
int			n;
int			h;
int			t1;
int			t2;
int *			mst_edges;
int *			p1;
int *			p2;
int *			p3;
int *			p4;
int *			endp;
dist_t			key1;
struct dsuf		sets;

	/* Count the cardinality 2 edges. */
	n = 0;
	for (i = 0; i < cip -> num_edges; i++) {
		if (cip -> edge_size [i] EQ 2) {
			++n;
		}
	}

	mst_edges = NULL;

	if (n > 0) {
		/* Get the cardinality 2 edges. */
		mst_edges = NEWA (n, int);
		n = 0;
		p1 =  mst_edges;
		for (i = 0; i < cip -> num_edges; i++) {
			if (cip -> edge_size [i] EQ 2) {
				*p1++ = i;
				++n;
			}
		}

		endp = &mst_edges [n];

		for (h = 1; h <= n; h = 3*h+1) {
		}

		do {
			h = h / 3;
			p4 = &mst_edges [h];
			p1 = p4;
			while (p1 < endp) {
				t1 = *p1;
				key1 = cip -> cost [t1];
				p2 = p1;
				while (TRUE) {
					p3 = (p2 - h);
					t2 = *p3;
					if (cip -> cost [t2] <= key1) break;
					*p2 = t2;
					p2 = p3;
					if (p2 < p4) break;
				}
				*p2 = t1;
				++p1;
			}
		} while (h > 1);

		if (n >= cip -> num_verts) {
			/* Remove all non-MST edges from the list. */
			_gst_dsuf_create (&sets, cip -> num_verts);
			for (i = 0; i < cip -> num_verts; i++) {
				_gst_dsuf_makeset (&sets, i);
			}
			p1 = mst_edges;
			for (i = 0; i < n; i++) {
				e = mst_edges [i];
				p2 = cip -> edge [e];
				t1 = _gst_dsuf_find (&sets, p2 [0]);
				t2 = _gst_dsuf_find (&sets, p2 [1]);
				if (t1 NE t2) {
					*p1++ = e;
					_gst_dsuf_unite (&sets, t1, t2);
				}
			}
			_gst_dsuf_destroy (&sets);
			n = p1 - mst_edges;
			FATAL_ERROR_IF (n NE cip -> num_verts - 1);
			/* Create a properly-sized copy of the edges. */
			p1 = NEWA (n, int);
			for (i = 0; i < n; i++) {
				p1 [i] = mst_edges [i];
			}
			free ((char *) mst_edges);
			mst_edges = p1;
		}
	}

	return (mst_edges);
}

/*
 * Shut down the heuristic upper bound stuff.  Free its permanently
 * allocated memory.
 */

	void
_gst_shutdown_heuristic_upper_bound (

struct ubinfo *		ubip		/* IN - global upper bound data */
)
{
int			i;
int *			ranking;

	if (ubip EQ NULL) return;

	for (i = 0; i < ubip -> num_rankings; i++) {
		ranking = ubip -> rankings [i];
		if (ranking NE NULL) {
			free ((char *) ranking);
		}
	}
	if (ubip -> mst_edges NE NULL) {
		free ((char *) (ubip -> mst_edges));
	}

	free ((char *) ubip);
}

/*
 * This routine takes an existing LP solution (a fractional lower
 * bound for an optimal Steiner tree) and attempts to perturb it into
 * a feasible integer solution.  If this succeeds and we discover a
 * shorter solution than previously known, it becomes the current
 * best feasible integer solution.  Any existing nodes that exceed
 * this new upper bound are then cut off.
 *
 * We use a greedy heuristic: sort the valid hyperedges into decreasing
 * order by LP solution weight.  If two edges have the same LP solution
 * weight, the one with the smaller FST-length/MST-length ratio will be
 * placed first.
 *
 * Greedily add edges from this sorted list to an empty tree, discarding
 * any edge that would form a cycle.  This is just a variant of Kruskal's
 * algorithm for MST, extended for hypergraphs.
 *
 * The one real efficiency hack is that we use linear time to partition
 * the edges into those with weight 1, fractional weight, and weight 0.
 * We then sort each region independently.
 */

	bool
_gst_compute_heuristic_upper_bound (

double *		x,	/* IN - an LP solution to use */
struct gst_solver *	solver	/* IN - needed for upper bound info and hg */
)
{
int			i;
int			j;
int			e;
int			nedges;
int			count_1;
int			count_frac;
int			count_0;
int *			ep1;
int *			ep2;
int *			ep3;
int *			edge_list;
int *			temp_edges;
int *			edges_1;
int *			edges_frac;
int *			edges_0;
int *			ranking;
bool			do_free_x;
double			old_ub;
struct ubinfo *		ubip;
struct gst_hypergraph *	cip;

	cip		= solver -> H;

	if (solver -> ubip EQ NULL) {
		solver -> ubip = _gst_startup_heuristic_upper_bound (cip);
	}

	ubip		= solver -> ubip;
	ubip -> best_z	= old_ub = solver -> upperbound;

	nedges		= cip -> num_edges;

	/* This is a temporary solution. The algorithm should handle the
	   absence of x more directly */
	do_free_x = FALSE;
	if (x EQ NULL) {
		x = NEWA (nedges, double);
		memset (x, 0, nedges * sizeof(double));
		do_free_x = TRUE;
	}

	/* Generate the list of valid full sets... */
	edge_list = NEWA (nedges, int);

	/* Put edges of weight 0.0 last, all the rest at the front. */
	ep1 = edge_list;
	ep2 = edge_list + nedges;
	for (i = 0; i < nedges; i++) {
		if (x [i] <= FUZZ) {
			*--ep2 = i;
		}
		else {
			*ep1++ = i;
		}
	}
	FATAL_ERROR_IF (ep1 NE ep2);
	edges_0 = ep2;
	count_0 = nedges - (edges_0 - edge_list);

	/* Now re-scan the non-zero edges, putting those of	*/
	/* weight 1.0 first.					*/
	ep1 = edge_list;
	ep3 = ep2;
	while (ep1 < ep2) {
		/* Scan forward from beginning for a fractional edge. */
		i = *ep1;
		if (x [i] + FUZZ >= 1.0) {
			++ep1;
			continue;
		}
		/* i is a fractional edge.  Scan backward from end	*/
		/* for an edge of weight 1.0.				*/
		for (;;) {
			--ep2;
			if (ep1 >= ep2) break;
			j = *ep2;
			if (x [j] + FUZZ >= 1.0) {
				/* i is fractional, j is integral -- swap. */
				*ep1 = j;
				*ep2 = i;
				++ep1;
				break;
			}
		}
	}
	FATAL_ERROR_IF (ep1 NE ep2);
	edges_1 = edge_list;
	edges_frac = ep2;

	count_1		= edges_frac - edges_1;
	count_frac	= edges_0 - edges_frac;

	temp_edges = NEWA (nedges, int);

	/* Repeat the greedy heuristic once for each FST rank ordering. */

	for (i = 0; i < ubip -> num_rankings; i++) {
		ranking = ubip -> rankings [i];

		/* Sort the integral FSTs by rank only. */
		sort_fst_list_by_rank (edges_1, count_1, ranking);

		/* Sort the fractional FSTs by LP weight, and then by rank. */
		sort_fst_list_by_lp_and_rank (edges_frac,
					      count_frac,
					      x,
					      ranking);

		/* Sort the zero-weight FSTs by rank only. */
		sort_fst_list_by_rank (edges_0, count_0, ranking);

		/* Try several greedy trees using this ordering of FSTs. */
		try_trees (edge_list, nedges, solver);

		/* Create a second ordering by deleting all of the MST	*/
		/* edges and placing them last.				*/
		ep1 = edge_list;
		ep2 = edge_list + nedges;
		ep3 = temp_edges;
		while (ep1 < ep2) {
			e = *ep1++;
			if (cip -> edge_size [e] <= 2) continue;
			*ep3++ = e;
		}
		if (ubip -> mst_edges) {
			ep1 = ubip -> mst_edges;
			ep2 = ep1 + (cip -> num_verts - 1);
			while (ep1 < ep2) {
				*ep3++ = *ep1++;
			}
		}

		/* Try a few greedy trees using this ordering. */
		try_trees (temp_edges, ep3 - temp_edges, solver);
	}

	free ((char *) temp_edges);
	free ((char *) edge_list);

	if (do_free_x) {
		free ((char *) x);
	}

	return (ubip -> best_z < old_ub);
}

/*
 * This routine sorts the full sets of integral weight (both 0.0 and 1.0).
 * In this case, only the given ranking is used.
 */

	static
	void
sort_fst_list_by_rank (

int *		array,		/* IN - list of full set numbers to sort */
int		n,		/* IN - number of full sets to sort */
int *		ranking		/* IN - rank ordering of FSTs */
)
{
int		h;
int		t1;
int		t2;
int		key1;
int *		p1;
int *		p2;
int *		p3;
int *		p4;
int *		endp;

	endp = &array [n];

	for (h = 1; h <= n; h = 3*h+1) {
	}

	do {
		h = h / 3;
		p4 = &array [h];
		p1 = p4;
		while (p1 < endp) {
			t1 = *p1;
			key1 = ranking [t1];
			p2 = p1;
			while (TRUE) {
				p3 = (p2 - h);
				t2 = *p3;
				if (ranking [t2] <= key1) break;
				*p2 = t2;
				p2 = p3;
				if (p2 < p4) break;
			}
			*p2 = t1;
			++p1;
		}
	} while (h > 1);
}

/*
 * This routine sorts the fractional full sets.  The primary key is LP
 * solution weight, ties are broken using the given ranking.
 */

	static
	void
sort_fst_list_by_lp_and_rank (

int *		array,		/* IN - list of full set numbers to sort */
int		n,		/* IN - number of full sets to sort */
double *	x,		/* IN - LP solution weights */
int *		ranking		/* IN - rank ordering of FSTs */
)
{
int		h;
int		t1;
int		t2;
double		key1;
int *		p1;
int *		p2;
int *		p3;
int *		p4;
int *		endp;

	endp = &array [n];

	for (h = 1; h <= n; h = 3*h+1) {
	}

	do {
		h = h / 3;
		p4 = &array [h];
		p1 = p4;
		while (p1 < endp) {
			t1 = *p1;
			key1 = x [t1];
			p2 = p1;
			while (TRUE) {
				p3 = (p2 - h);
				t2 = *p3;
				if (x [t2] > key1) break;
				if ((x [t2] EQ key1) AND
				    (ranking [t2] <= ranking [t1])) break;
				*p2 = t2;
				p2 = p3;
				if (p2 < p4) break;
			}
			*p2 = t1;
			++p1;
		}
	} while (h > 1);
}

/*
 * This routine constructs several trees according to the given sorted
 * list of FSTs.  It uses the greedy Kruskal algorithm to build an
 * initial tree.  It then constructs a sequence of DIFFERENT trees as
 * follows:
 *
 *	1. Find an edge that has NOT been used in any of the trees
 *	   constructed previously by this routine.
 *	2. Build a new tree using the same list, but putting this
 *	   previously unused edge in FIRST.
 *
 * Given M edges, we repeat the edge-forcing operation until we have
 * tried O(log(M)) unused edges, or until we run out of edges to force.
 */

	static
	void
try_trees (

int *			edge_list,	/* IN - ordered list of edges */
int			nedges,		/* IN - number of edges in list */
struct gst_solver *	solver		/* IN - solver object */
)
{
int			i;
int			j;
int			e;
int			k;
int			org_limit;
int			nmasks;
int			limit;
dist_t			l;
dist_t			old_l;
dist_t			small_gap;
dist_t			curr_gap;
dist_t			fraction;
dist_t			quadratic;
bool			have_reset;
struct gst_hypergraph *	cip;
struct ubinfo *		ubip;
int *			temp_edges;
bitmap_t *		used;

	ubip	= solver -> ubip;
	cip	= solver -> H;
	nmasks	= cip -> num_edge_masks;

	/* None of the edges have been used yet. */
	used = NEWA (nmasks, bitmap_t);
	for (i = 0; i < nmasks; i++) {
		used [i] = 0;
	}

	/* Construct the initial tree.  Note which edges were used. */
	l = ub_kruskal (edge_list, nedges, used, solver);

	if (l EQ INF_DISTANCE) {
		return; /* No initial tree found */
	}

	/* Copy the list so that we can quickly add an edge	*/
	/* onto the front of the list...			*/
	temp_edges = NEWA (nedges + 1, int);
	for (i = 0; i < nedges; i++) {
		temp_edges [i + 1] = edge_list [i];
	}

	/* Determine the limit of edges to try. */
	for (limit = 2; (1 << limit) < nedges; limit++) {
	}

	/* Compute "small" absolute gap value */
	small_gap = 0.0001 * fabs (ubip -> best_z);
	if (small_gap < 1.0) {
		small_gap = 1.0;
	}

	/* Greedy local search. */
	have_reset = FALSE;
	org_limit  = limit;
	old_l	   = INF_DISTANCE;
	k = nedges + 1;

	for (i = 0; i < nedges; i++) {
		e = edge_list [i];
		if (BITON (used, e)) continue;

		/* Force edge e to be chosen FIRST. */
		temp_edges [0] = e;

		/* Clear old used map */
		for (j = 0; j < nmasks; j++) {
			used [j] = 0;
		}

		l = ub_kruskal (temp_edges, k, used, solver);

		/* If improved solution then replace temp_edges */
		if (l < old_l) {
			k = 1;
			for (j = 0; j < nedges; j++) {
				e = edge_list [j];
				if (BITON (used, e) OR
				    (cip -> edge_size [e] EQ 2)) {
					temp_edges [k++] = e;
				}
			}
			old_l = l;

			/* Compute "small" absolute gap value */
			small_gap = 0.0001 * fabs (ubip -> best_z);
			if (small_gap < 1.0) {
				small_gap = 1.0;
			}
		}

		/* If we are really close to optimum then restart	*/
		/* and intensify search.				*/

		curr_gap = l - ubip -> best_z;
		if ((NOT have_reset) AND (curr_gap <= small_gap)) {

#if 0
			gst_channel_printf (solver -> params -> print_solve_trace,
				" upper bound heuristic intensifying"
				" search (absolute gap is %.3f)\n",
				curr_gap);
#endif

			/* Let the new limit be double, plus an a*x**2	*/
			/* term whose coefficient is linearly dependent	*/
			/* on the gap.					*/
			fraction = (small_gap - curr_gap) / small_gap;
			quadratic = floor (fraction * (org_limit * org_limit));
			limit = 2 * org_limit + (int) quadratic;

			i = 0;
			have_reset = TRUE;
		}

		if (--limit <= 0) break;
	}

	free ((char *) temp_edges);
	free ((char *) used);
}

/*
 * Use Kruskal's algorithm (extended for hypergraphs) to greedily construct
 * a tree from the given sequence of FSTs.  If this yields a better
 * solution than previously known, record it and update the upper bound.
 */

	static
	dist_t
ub_kruskal (

int *			edge_list,	/* IN - ordered list of edges */
int			nedges,		/* IN - number of edges in list */
bitmap_t *		used,		/* IN - marks FSTs used (ever) */
struct gst_solver *	solver		/* IN - solver object */
)
{
int			i;
int			j;
int			e;
int			nverts;
int			nedges_solution;
int *			ep1;
int *			ep2;
int *			ep3;
int *			vp1;
int *			vp2;
int *			vp3;
int *			vp4;
int *			tree_edges;
struct gst_hypergraph *	cip;
struct ubinfo *		ubip;
int *			roots;
bool *			mark;
int			components;
dist_t			length;
struct dsuf		sets;

	ubip		= solver -> ubip;

	cip		= solver -> H;

	nverts		= cip -> num_verts;

	nedges_solution = 0;

	/* Initialize one disjoint subtree for each terminal. */
	_gst_dsuf_create (&sets, nverts);
	for (i = 0; i < nverts; i++) {
		_gst_dsuf_makeset (&sets, i);
	}

	tree_edges = NEWA (nverts - 1, int);

	mark = NEWA (nverts, bool);
	for (i = 0; i < nverts; i++) {
		mark [i] = FALSE;
	}
	roots = NEWA (nverts, int);

	/* Start the greedy Kruskal procedure... */
	components = nverts;
	length = 0.0;
	ep1 = tree_edges;
	ep2 = edge_list;
	ep3 = edge_list + nedges;
	while (components > 1) {
		if (ep2 >= ep3) {
			/* FSTs ran out before tree constructed! */
			/* FATAL ("Bug 1."); */
			length = INF_DISTANCE;
			break;
		}
		e = *ep2++;
		vp3 = roots;
		vp1 = cip -> edge [e];
		vp2 = cip -> edge [e + 1];
		for (;;) {
			if (vp1 >= vp2) {
				/* No cycle!  Include e in solution! */
				*ep1++ = e;
				length += cip -> cost [e];
				nedges_solution++;
				SETBIT (used, e);
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
			/* This FST terminal is in distinct subtree... */
			mark [j] = TRUE;
			*vp3++ = j;
		}
	}

	if (components EQ 1) {
		/* A solution was found */
		if (_gst_update_best_solution_set (solver,
					      NULL,
					      nedges_solution,
					      tree_edges,
					      NULL)) {
			ubip -> best_z = solver -> upperbound;
		}
	}

	free ((char *) roots);
	free ((char *) mark);
	free ((char *) tree_edges);
	_gst_dsuf_destroy (&sets);

	return (length);
}
