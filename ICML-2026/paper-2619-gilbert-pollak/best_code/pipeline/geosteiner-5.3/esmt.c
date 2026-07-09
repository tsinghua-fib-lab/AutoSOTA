/***********************************************************************

	$Id: esmt.c,v 1.28 2022/11/19 13:45:51 warme Exp $

	File:	esmt.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Function to solve Euclidean SMT problems. It only uses library
	functions.

************************************************************************

	Modification Log:

	a-1:	04/30/02	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files.

************************************************************************/

#include "efuncs.h"
#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include "prepostlude.h"
#include "steiner.h"

/* Special fast code to handle 3-terminal case */

	int
gst_esmt3 (

double *		terms,
double *		length,
int *			nsps,
double *		sps,
int *			nedges,
int *			edges
)
{
int		ab_swapped;
struct point	p1;
struct point	p2;
struct point	pt;
struct point	e;
struct point	ctr;
struct point	sp;
struct point *	a;
struct point *	b;

	pt.x = terms[0];
	pt.y = terms[1];
	p1.x = terms[2];
	p1.y = terms[3];
	p2.x = terms[4];
	p2.y = terms[5];

	/* Make sure points are oriented nicely */
	ab_swapped = FALSE;
	if (left_turn(&p1, &p2, &pt)) {
		a = &p1; b = &p2;
	}
	else {
		a = &p2; b = &p1;
		ab_swapped = TRUE;
	}

	eq_point (a, b, &e);
	eq_circle_center (a, b, &e, &ctr);

	if (nsps NE NULL) {
		*nsps = 0;
	}
	if (nedges NE NULL) {
		*nedges = 2;
	}
	if (right_turn (&e, a, &pt)) {
		if (left_turn (&e, b, &pt)) {
			if (sqr_dist(&ctr, &pt) > sqr_dist(&ctr, &e)) {

				/* We have a Steiner point */

				if (nsps NE NULL) {
					*nsps = 1;
				}
				if (sps NE NULL) {
					project_point(&e, &ctr, &pt, &sp);
					sps[0] = sp.x;
					sps[1] = sp.y;
				}
				if (nedges NE NULL) {
					*nedges = 3;
				}
				if (edges NE NULL) {
					edges[0] = 0;	edges[1] = 3;
					edges[2] = 1;	edges[3] = 3;
					edges[4] = 2;	edges[5] = 3;
				}
				if (length NE NULL) {
					*length = EDIST(&e, &pt);
				}
			}
			else {
				/* pt is degree 2 terminal in MST */
				if (edges NE NULL) {
					edges[0] = 0;	edges[1] = 1;
					edges[2] = 0;	edges[3] = 2;
				}
				if (length NE NULL) {
					*length = EDIST(a, &pt) + EDIST(b, &pt);
				}
			}
		}
		else {
			/* b is degree 2 terminal in MST */
			if (edges NE NULL) {
				if (ab_swapped) {
					edges[0] = 0;	edges[1] = 1;
					edges[2] = 1;	edges[3] = 2;
				}
				else {
					edges[0] = 0;	edges[1] = 2;
					edges[2] = 1;	edges[3] = 2;
				}
			}
			if (length NE NULL) {
				*length = EDIST(a, b) + EDIST(&pt, b);
			}
		}
	}
	else {
		/* a is degree 2 terminal in MST */
		if (edges NE NULL) {
			if (NOT ab_swapped) {
				edges[0] = 0;	edges[1] = 1;
				edges[2] = 1;	edges[3] = 2;
			}
			else {
				edges[0] = 0;	edges[1] = 2;
				edges[2] = 1;	edges[3] = 2;
			}
		}
		if (length NE NULL) {
			*length = EDIST(b, a) + EDIST(&pt, a);
		}
	}

	return (0);
}

/*
 * Compute a Euclidean Steiner minimal tree for a given set of points.
 */

	int
gst_esmt (

int			nterms,
double *		terms,
double *		length,
int *			nsps,
double *		sps,
int *			nedges,
int *			edges,
int *			soln_status,
gst_param_ptr		params
)
{
int		nfsts;
int		status;
int		res;
int *		fsts;
gst_hg_ptr	hg;
gst_hg_ptr	hg2;
gst_solver_ptr	solver;

	GST_PRELUDE

	res = 0;

	/* Set default value of all output parameters. */
	if (length NE NULL)	 *length	= 0.0;
	if (nsps NE NULL)	 *nsps		= 0;
	if (nedges NE NULL)	 *nedges	= 0;
	if (soln_status NE NULL) *soln_status	= GST_STATUS_NO_SOLUTION;

	if (nterms < 0) {
		res = GST_ERR_INVALID_NUMBER_OF_TERMINALS;
		goto end;
	}

	if (soln_status NE NULL) {
		/* default value */
		*soln_status = GST_STATUS_OPTIMAL;
	}

	/* Handle small cases quickly */
	if (nterms <= 1) {
		/* Current values of outputs are correct. */
	}
	else if ((nterms EQ 2) AND (params EQ NULL)) {
		if (length NE NULL) {
			*length = hypot (fabs (terms[2] - terms[0]),
					 fabs (terms[3] - terms[1]));
		}
		if (nedges NE NULL) {
			*nedges = 1;
		}
		if (edges NE NULL) {
			edges[0] = 0;
			edges[1] = 1;
		}
	}
	else if ((nterms EQ 3) AND (params EQ NULL)) {
		gst_esmt3 (terms, length, nsps, sps, nedges, edges);
	}
	else {
		/* General case: Generate FSTs and concatenate */

		hg = gst_generate_efsts (nterms, terms, params, &status);

		if (hg EQ NULL) {
			res = status;
			if (soln_status NE NULL) {
				*soln_status = GST_STATUS_NO_SOLUTION;
			}
			goto end;
		}

		/* Prune FSTs when instance is large */

		if (nterms >= 2000) {
			hg2 = gst_hg_prune_edges (hg, params, &status);
			if (hg2 NE NULL) {
				gst_free_hg (hg);
				hg = hg2;
			}
			else {
				/* Pruner failed.  Just ignore this	*/
				/* and proceed with the unpruned FSTs.	*/
			}
		}

		/* Find MST in FST hypergraph */

		solver = gst_create_solver (hg, params, &status);
		res = gst_hg_solve (solver, NULL);

		if (res EQ 0) {
			/* Find length */
			if (length NE NULL) {
				gst_hg_solution (solver, &status, NULL, length, 0);
			}

			if (soln_status NE NULL) {
				gst_get_solver_status (solver, soln_status);
			}

			/* Other info... */
			if ((sps NE NULL) OR (nsps NE NULL) OR
			    (nedges NE NULL) OR (edges NE NULL)) {

				/* Get the number of FSTs in the optimal solution */
				gst_hg_solution (solver, &nfsts, NULL, NULL, 0);

				/* Get a list of indices to the FSTs in the optimal solution */
				fsts = NEWA (nfsts, int);
				gst_hg_solution (solver, NULL, fsts, NULL, 0);

				gst_get_hg_edge_embedding (hg, nfsts, fsts, nsps, sps, nedges, edges);
				free (fsts);
			}
		}

		/* Clean up */
		gst_free_hg (hg);
		gst_free_solver (solver);
	}

end:
	GST_POSTLUDE

	return res;
}
