/***********************************************************************

	$Id: greedy.c,v 1.18 2023/04/03 17:30:39 warme Exp $

	File:	greedy.c
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 2000, 2023 by Martin Zachariasen.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Implementation of greedy O(n^2) heuristic for
	computing Euclidean Steiner trees
	(Algorithmica 25, 418-437, 1999)

************************************************************************

	Modification Log:

	a-1:	01/31/2000	martinz
		: Created.  Derived from SLL heuristic
	b-1:	08/05/2002	benny
		: Now uses heapsort instead of qsort.
	c-1:	01/26/2004	martinz
		: Removed direct call to Triangle
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
	e-4:	11/18/2022	warme
		: Fix -Wall issue.
	e-5:	04/03/2023	warme
		: Fix -Wall issue.

************************************************************************/

#include "greedy.h"

#include "bsd.h"
#include "dsuf.h"
#include "dt.h"
#include "efuncs.h"
#include "emst.h"
#include <float.h>
#include "logic.h"
#include "memory.h"
#include "sortfuncs.h"
#include <stdlib.h>
#include "steiner.h"

/*
 * Global Routines
 */

dist_t		_gst_greedy_heuristic (struct pset *	ptss,
				       int *		termidx,
				       struct bsd *	bsdp);

/*
 * Local Types
 */

struct greedy_edge {
	dist_t		len;
	dist_t		bsd;
	int		p1;
	int		p2;
	bool		mst;
};

struct greedy_fst {
	dist_t		len;	/* Length of FST */
	dist_t		ratio;	/* Ratio of length to length of MST spanning FST-terms */
	int		count;	/* Number of terminals */
	int		p [4];	/* List of terminals */
	bool		chosen; /* Is this FST chosen by the algorithm? */
};


/*
 * Local Routines
 */

static void		add_fst (struct pset *		pts,
				 int *			terms,
				 int			term_count,
				 struct greedy_fst *	greedy_fsts,
				 int *			fst_count,
				 dist_t			mst_l,
				 bool			mst_connected);
static dist_t		fst_length (struct pset *, int *, int);
static dist_t		mst_length (struct pset *, int *, int);
static int		compare_edge_length (int, int, void *);
static int		compare_fst_ratio (int, int, void *);



/*
 * Compute the length of a shortest full Steiner tree
 * for a set of terminals with up to 4 terminals.
 *  Special fast version which only computes length.
 * Returns 0.0 if no Steiner tree exists.
 * Assume terminals are ordered as they appear on Steiner polygon.
 */

	static
	dist_t
fst_length (

struct pset *		pts,		/* IN - terminal list */
int *			terms,		/* IN - indices of terminals to consider */
int			term_count	/* IN - number of terminals */
)
{
int			i;
struct point *		a;
struct point *		b;
struct point *		c;
struct point *		d;
struct point		e, ctr, e_ad, e_cb;
dist_t			l;
dist_t			min_length;

	if (term_count EQ 2) {
		return (EDIST (&(pts -> a [terms [0]]),
			       &(pts -> a [terms [1]])));
	}

	if (term_count EQ 3) {
		a = &(pts -> a [terms [0]]);
		b = &(pts -> a [terms [1]]);
		c = &(pts -> a [terms [2]]);

		eq_point (a, b, &e);
		eq_circle_center (a, b, &e, &ctr);

		if (right_turn (&e, a, c) AND
		    left_turn (&e, b, c) AND
		    (sqr_dist (&ctr, c) > sqr_dist (&ctr, a))) {
			return EDIST (&e, c);
		}
	}

	if (term_count EQ 4) {
		min_length = INF_DISTANCE;
		for (i = 0; i <= 1; i++) {
			/* Using Lemma 5.2 p. 64 in Hwang, Richards, Winter */
			a = &(pts -> a [terms [i]]);
			d = &(pts -> a [terms [i+1]]);
			c = &(pts -> a [terms [i+2]]);
			b = &(pts -> a [terms [(i+3) % 4]]);

			/* Find intersetion point between ac and bd.
			   It is the same in both iterations */

			if ((i EQ 0) AND
			    NOT segment_intersection (a, c, b, d, &ctr)) break;

			eq_point (a, d, &e_ad);
			eq_point (c, b, &e_cb);

			if (NOT wedge120 (d, a, &e_cb) AND
			    NOT wedge120 (&e_cb, d, a) AND
			    NOT wedge120 (&e_ad, b, c) AND
			    NOT wedge120 (b, c, &e_ad) AND
			    NOT wedge120 (a, &ctr, d)) {
				l = EDIST (&e_ad, &e_cb);
				if (l < min_length) {
					min_length = l;
				}
			}
		}
		if (min_length < INF_DISTANCE) return (min_length);
	}

	return (0.0);
}

/*
 * Compute the length of a MST for a set of terminals.
 * Simply call general procedure (which might not be
 * the most effective to thing to do)
 */

	static
	dist_t
mst_length (

struct pset *		pts,		/* IN - terminal list */
int *			terms,		/* IN - indices of terminals to consider */
int			term_count	/* IN - number of terminals */
)
{
int			t;
struct pset *		tpts;
dist_t			mst_l;

	tpts = NEW_PSET (term_count);
	tpts -> n = term_count;
	for (t = 0; t < term_count; t++) {
		tpts -> a[t] = pts -> a[terms[t]];
	}
	mst_l  = _gst_euclidean_mst_length (tpts);
	free (tpts);
	return mst_l;
}

/*
 * Compares edges by length. Returns TRUE if the first edge is shorter than
 * the second edge or if they are equal in length but with first index smaller
 * than the second index. Otherwise FALSE is returned.
 */
	static
	int
compare_edge_length
(

int	i1,	/* IN - first index */
int	i2,	/* IN - second index */
void *	array	/* IN - array to be sorted */
)
{
dist_t			key1, key2;

	key1 = ((struct greedy_edge *)array)[i1].len;
	key2 = ((struct greedy_edge *)array)[i2].len;

	if ((key1 < key2) OR
	    ((key1 EQ key2) AND (i1 < i2)))
		return (-1);

	return (1);
}

/*
 * Compare fsts by ratio.
 */
	static
	int
compare_fst_ratio
(

int	i1,	/* IN - first index */
int	i2,	/* IN - second index */
void *	array	/* IN - array to be sorted */
)
{
dist_t			key1, key2;

	key1 = ((struct greedy_fst *)array)[i1].ratio;
	key2 = ((struct greedy_fst *)array)[i2].ratio;

	if ((key1 < key2) OR
	    ((key1 EQ key2) AND (i1 < i2)))
		return (-1);

	return (1);
}

/*
 * Add generated FST to list of FSTs
 */

	static
	void
add_fst (

struct pset *	pts,		/* IN - terminal list */
int *		terms,		/* IN - indices of terminals in FST */
int		term_count,	/* IN - number of terminals */
struct greedy_fst * greedy_fsts,/* IN/OUT - list of FSTs */
int *		fst_count,	/* IN/OUT - current FST count */
dist_t		mst_l,		/* Length of MST spanning FST-terms */
bool		mst_connected	/* Is induced subgraph of MST connected? */
)
{
int		j;
dist_t		smt_l;
dist_t		ratio;

	smt_l = fst_length (pts, terms, term_count);

	if (smt_l > 0) {
		ratio = smt_l / mst_l;
		if (ratio < 1.0) {
			for (j = 0; j < 4; j++) {
				greedy_fsts [*fst_count].p [j] = terms [j];
			}
			greedy_fsts [*fst_count].len	= smt_l;
			greedy_fsts [*fst_count].ratio	= ratio;
			greedy_fsts [*fst_count].count	= term_count;
			greedy_fsts [*fst_count].chosen = mst_connected;
			++(*fst_count);
		}
	}
}

/*
 * Greedy heuristic by Zachariasen and Winter
 */

	dist_t
_gst_greedy_heuristic (

struct pset *	ptss,		/* IN - point list for which heuristic tree
					should be constructed */
int *		termidx,	/* IN - terminal number for each point
					(or -1 if point is not a terminal) */
struct bsd *	bsdp		/* IN - BSD data structure */
)
{
int			i, j, k, jj, ei, fi, fk;
int			ni, nj, nt, np1, np2, p1, p2, p4;
int			fst_count, mst_count1, mst_count2;
int			neighbour_edge, neighbour_edge_idx;
int			term_count;
int			root [4];
int			terms [4];
int			numberofedges;
int			numberoftriangles;
int *			trianglelist;
int *			neighborlist;
int *			triangleedges;
int *			sortededges;
int *			sortedfsts;
bool *			new_chosen;
bool			convex_region;
bool			in_same_block;
dist_t			mst_l, mst_l1, total_mst_l;
dist_t			total_smt_l, best_smt_l, l[5], mx, my;
struct point		minp, maxp;
struct dsuf		mstsets, fstsets;
struct pset *		pts;
struct greedy_edge  *	greedy_edges;
struct greedy_fst  *	greedy_fsts;

	/* Special cases */

	if (ptss -> n <= 1) return (0.0);
	if (ptss -> n EQ 2) return (EDIST (&(ptss -> a [0]),
					   &(ptss -> a [1])));

	/* Compute mean of all terminals and translate in order */
	/* to improve the precision of computed eq-points.	*/
	mx = 0.0;
	my = 0.0;
	for (i = 0; i < ptss -> n; i++) {
		mx += ptss -> a[i].x;
		my += ptss -> a[i].y;
	}
	mx = floor(mx / ((dist_t) ptss -> n));
	my = floor(my / ((dist_t) ptss -> n));

	/* Compute Delaunay triangulation */

	pts  = NEW_PSET(ptss -> n);
	pts  -> n = ptss -> n;
	for (i = 0; i < pts -> n; i++) {
		pts  -> a[i].x	 = ptss -> a[i].x - mx;
		pts  -> a[i].y	 = ptss -> a[i].y - my;
	}

	_gst_delaunay_triangulation (
				ptss,
				&numberofedges,
				NULL,
				&numberoftriangles,
				&trianglelist,
				&neighborlist);

	if (numberoftriangles EQ 0) {

		/* There are no triangles (all input points co-linear).
		   Compute SMT as straight line segment between points */
		minp.x = INF_DISTANCE; maxp.x = -INF_DISTANCE;
		minp.y = INF_DISTANCE; maxp.y = -INF_DISTANCE;
		for (i = 0; i < pts -> n; i++) {
			minp.x = MIN(minp.x, pts -> a [i].x);
			maxp.x = MAX(maxp.x, pts -> a [i].x);
			minp.y = MIN(minp.y, pts -> a [i].y);
			maxp.y = MAX(maxp.y, pts -> a [i].y);
		}

		free (trianglelist);
		free (neighborlist);
		return (EDIST(&minp, &maxp));
	}

	/* Construct edge information */
	greedy_edges	= NEWA (numberofedges, struct greedy_edge);
	triangleedges	= NEWA (3*numberoftriangles, int);
	for (i = 0; i < 3*numberoftriangles; i++) {
		triangleedges [i] = -1;
	}

	ei = 0;
	for (i = 0; i < numberoftriangles; i++) {
		for (j = 0; j < 3; j++) {
			p1 = trianglelist [3*i + j];
			p2 = trianglelist [3*i + ((j+1) % 3)];
			if (triangleedges [3*i + j] EQ -1) {
				/* only add once */
				greedy_edges [ei].len	 = EDIST (&(pts -> a [p1]),
								  &(pts -> a [p2]));
				/* Get BSD info if it is there */
				if ((bsdp NE NULL) AND
				    (termidx NE NULL) AND
				    (termidx [p1] >= 0) AND
				    (termidx [p2] >= 0)) {
					greedy_edges [ei].bsd = _gst_bsd (bsdp,
									  termidx [p1],
									  termidx [p2]);
				}
				else {
					greedy_edges [ei].bsd = greedy_edges [ei].len;
				}

				greedy_edges [ei].p1	 = p1;
				greedy_edges [ei].p2	 = p2;
				greedy_edges [ei].mst	 = FALSE;
				triangleedges [3*i + j]	 = ei;

				/* Now go through neighbouring triangles */
				/* and add information about the new edge */
				for (ni = 0; ni < 3; ni++) {
					nt = neighborlist [3*i + ni];
					if (nt EQ -1) continue;
					for (nj = 0; nj < 3; nj++) {
						np1 = trianglelist [3*nt + nj];
						np2 = trianglelist [3*nt + ((nj+1) % 3)];
						if ((np1 EQ p2) AND
						    (np2 EQ p1)) {
							 /* found reverse edge */
							triangleedges [3*nt + nj] = ei;
						}
					}
				}
				++ei;
			}
		}
	}

	/* Sort edges */

	sortededges = _gst_heapsort (numberofedges,
				     greedy_edges,
				     compare_edge_length);

	/* Use Kruskal to find MST */

	_gst_dsuf_create (&mstsets, pts -> n);
	for (i = 0; i < pts -> n; i++) {
		_gst_dsuf_makeset (&mstsets, i);
	}
	total_mst_l = 0.0;
	for (i = 0; i < numberofedges; i++) {
		/* go through edges in sorted order */
		ei = sortededges [i];
		root [1] = _gst_dsuf_find (&mstsets, greedy_edges [ei].p1);
		root [2] = _gst_dsuf_find (&mstsets, greedy_edges [ei].p2);

		if (root [1] NE root [2]) {
			_gst_dsuf_unite (&mstsets, root [1], root [2]);
			total_mst_l += greedy_edges [ei].len;
			greedy_edges [ei].mst = TRUE; /* remember this edge */
		}
	}
	(void) total_mst_l;	/* Silence compiler warning. */

	/* Generate FSTs with 3 and 4 terminals */

	greedy_fsts = NEWA (4*numberoftriangles, struct greedy_fst);
	fst_count = 0;
	for (i = 0; i < numberoftriangles; i++) {

		/* Count the number of MST edges and find their length sum */
		mst_count1 = 0;
		mst_l1	   = 0.0;
		for (j = 0; j < 3; j++) {
			ei  = triangleedges [3*i + j];
			l[j] = greedy_edges [ei].len;
			if (greedy_edges [ei].mst) {
				mst_count1++;
				mst_l1 += l[j];
			}
		}

		for (j = 0; j < 3; j++) {
			terms [j] = trianglelist [3*i + j];
		}
		terms [3] = -1;

		/* Do the three terminals induced a connected dubgraph of the MST? */
		if (mst_count1 EQ 2) {
			/* Yes, so we know the length of the corresponding MST */
			mst_l = mst_l1;
		}
		else {
			/* No, they do not. Compute MST for these three terminals */
			mst_l = MIN(l[0]+l[1], MIN(l[0]+l[2], l[1]+l[2]));
		}

		/* Add 3-terminal FST */
		add_fst (pts, terms, 3, greedy_fsts, &fst_count, mst_l, (mst_count1 EQ 2));

		/* Now go through neighbouring triangles and add */
		/* valid FSTs */
		for (ni = 0; ni < 3; ni++) {

			nt = neighborlist [3*i + ni]; /* get triangle index */
			if (nt <= i) continue;		  /* only consider once... */

			/* Find neighbouring edge and outgoing MST */
			/* edge (note that there can be at most one) */
			neighbour_edge	   = -1;
			neighbour_edge_idx = -1;
			mst_count2	   =  0;

			for (nj = 0; nj < 3; nj++) {
				ei = triangleedges [3*nt + nj];
				if (greedy_edges [ei].mst) {
					mst_count2++;
				}

				/* Is this the neighouring edge? */
				for (j = 0; j < 3; j++) {
					if (triangleedges [3*i + j] EQ ei) {
						neighbour_edge	   = ei;
						neighbour_edge_idx = nj;
					}
				}
			}


			/* Idenfify fourth terminal */

			p4 = trianglelist [3*nt + ((neighbour_edge_idx + 2) % 3)];

			/* Make list of points on border of triangles */

			j = -1;
			jj = -1;
			do {
				terms [++jj] = trianglelist [3*i + (++j)];
			} while (triangleedges [3*i + j] NE neighbour_edge);
			terms [++jj] = p4;
			while (j < 2) {
				terms [++jj] = trianglelist [3*i + (++j)];
			}

			/* Make sure that the two triangles make up a */
			/* convex region */

			convex_region = TRUE;
			for (j = 0; j < 4; j++) {
				if (right_turn (&(pts -> a [terms [j]]),
						&(pts -> a [terms [(j+1) % 4]]),
						&(pts -> a [terms [(j+2) % 4]]))) {
					convex_region = FALSE;
					break;
				}
			}
			if (NOT convex_region) continue; /* do not try to construct FST */

			/* Compute MST for terminals */
			mst_l = mst_length(pts, terms, 4);

			/* Add 4-terminal FST */
			add_fst (pts,
				 terms,
				 4,
				 greedy_fsts,
				 &fst_count,
				 mst_l,
				 (mst_count1 + mst_count2 >= 3));
		}
	}

	/* Sort FSTs */

	sortedfsts = _gst_heapsort (fst_count, greedy_fsts, compare_fst_ratio);

	/* Use greedy concatenation algorithm:
	   1. Use FSTs that span connected subgraphs of MST
	   2. Insert not yet tried FSTs into existing tree
	*/

	new_chosen = NEWA (fst_count, bool);
	best_smt_l = INF_DISTANCE;
	for (k = -1; k < fst_count; k++) { /* k=-1 means use MST FSTs */

		/* Check that FST is not already in current tree */

		if ((k >= 0) AND greedy_fsts [ sortedfsts[k] ].chosen) continue;

		/* Build union-find structure */
		_gst_dsuf_create (&fstsets, pts -> n);
		for (i = 0; i < pts -> n; i++) {
			_gst_dsuf_makeset (&fstsets, i);
		}

		for (i = 0; i < fst_count; i++)
			new_chosen[i] = greedy_fsts[i].chosen;

		if (k >= 0) {
			/* Insert FST number k */
			fk = sortedfsts [k];
			term_count = greedy_fsts [fk].count;

			for (j = 1; j < term_count; j++) {
				_gst_dsuf_unite (&fstsets,
						 _gst_dsuf_find (&fstsets, greedy_fsts [fk].p [0]),
						 _gst_dsuf_find (&fstsets, greedy_fsts [fk].p [j]));
			}
			total_smt_l = greedy_fsts [fk].len;
			new_chosen[fk] = TRUE;
		}
		else
			total_smt_l = 0.0;

		/* Go through chosen FSTs (in sorted order) */
		for (i = 0; i < fst_count; i++) {
			fi = sortedfsts [i];
			if (NOT greedy_fsts [fi].chosen) continue;
			term_count = greedy_fsts [fi].count;

			/* check that no two terminals are in the same block */
			in_same_block = FALSE;
			for (j = 0; j < term_count; j++) {
				root [j] = _gst_dsuf_find (&fstsets, greedy_fsts [fi].p [j]);
			}
			for (j = 0; j < term_count-1; j++) {
				for (jj = j+1; jj < term_count; jj++) {
					if (root [j] EQ root [jj]) {
						in_same_block = TRUE;
						break;
					}
				}
			}
			if (in_same_block) {
				new_chosen [fi] = FALSE;
				continue;
			}
			for (j = 1; j < term_count; j++) {
				_gst_dsuf_unite (&fstsets,
						 _gst_dsuf_find (&fstsets, greedy_fsts [fi].p [0]),
						 _gst_dsuf_find (&fstsets, greedy_fsts [fi].p [j]));
			}
			total_smt_l += greedy_fsts [fi].len;
		}

		/* Go through edges in sorted order */
		for (i = 0; i < numberofedges; i++) {
			ei = sortededges [i];
			root [1] = _gst_dsuf_find (&fstsets, greedy_edges [ei].p1);
			root [2] = _gst_dsuf_find (&fstsets, greedy_edges [ei].p2);
			if (root [1] NE root [2]) {
				_gst_dsuf_unite (&fstsets, root [1], root [2]);
				/* Use BSD length here */
				total_smt_l += greedy_edges [ei].bsd;
			}
		}
		_gst_dsuf_destroy (&fstsets);

		if (total_smt_l < best_smt_l) {
			best_smt_l = total_smt_l;
			for (i = 0; i < fst_count; i++)
				greedy_fsts [i].chosen = new_chosen [i];
		}

	}

	/* Free all allocated arrays, including those allocated by Triangle */

	_gst_dsuf_destroy (&mstsets);

	free (pts);
	free (trianglelist);
	free (neighborlist);
	free (triangleedges);
	free (greedy_edges);
	free (greedy_fsts);
	free (sortededges);
	free (sortedfsts);
	free (new_chosen);

	return (best_smt_l);
}
