/***********************************************************************

	$Id: rsmt.c,v 1.23 2022/11/19 13:45:54 warme Exp $

	File:	rsmt.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Function to solve rectilinear SMT problems. It only uses library
	functions.

************************************************************************

	Modification Log:

	a-1:	04/30/2002	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, fix -Wall issues.

************************************************************************/

#include "geosteiner.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "prepostlude.h"
#include "steiner.h"

/* Special fast code to handle 3-terminal case */

	int
gst_rsmt3 (

double *		terms,
double *		length,
int *			nsps,
double *		sps,
int *			nedges,
int *			edges
)
{
double		spx;
double		spy;
int		xorder[3];
int		yorder[3];
int		t;

	/* Find order of terminals in x-direction */
	xorder[0] = 0; xorder[1] = 2; xorder[2] = 4;
	if (terms[xorder[0]] > terms[xorder[1]]) {
		t = xorder[0]; xorder[0] = xorder[1]; xorder[1] = t;
	}
	if (terms[xorder[1]] > terms[xorder[2]]) {
		t = xorder[1]; xorder[1] = xorder[2]; xorder[2] = t;
	}
	if (terms[xorder[0]] > terms[xorder[1]]) {
		t = xorder[0]; xorder[0] = xorder[1]; xorder[1] = t;
	}

	/* Find order of terminals in y-direction */
	yorder[0] = 1; yorder[1] = 3; yorder[2] = 5;
	if (terms[yorder[0]] > terms[yorder[1]]) {
		t = yorder[0]; yorder[0] = yorder[1]; yorder[1] = t;
	}
	if (terms[yorder[1]] > terms[yorder[2]]) {
		t = yorder[1]; yorder[1] = yorder[2]; yorder[2] = t;
	}
	if (terms[yorder[0]] > terms[yorder[1]]) {
		t = yorder[0]; yorder[0] = yorder[1]; yorder[1] = t;
	}

	/* Do we really need to compute a Steiner point? */
	if ((nsps NE NULL) OR (sps NE NULL) OR
	    (nedges NE NULL) OR (edges NE NULL)) {
		/* Coordinates of possible Steiner point */
		spx = terms[xorder[1]];
		spy = terms[yorder[1]];

		if (nsps NE NULL) {
			*nsps = 0;
		}
		if (nedges NE NULL) {
			*nedges = 2;
		}
		if ((spx EQ terms[0]) AND (spy EQ terms[1])) {
			/* Steiner point overlaps with first terminal */
			if (edges NE NULL) {
				edges[0] = 0; edges[1] = 1;
				edges[2] = 0; edges[3] = 2;
			}
		}
		else if ((spx EQ terms[2]) AND (spy EQ terms[3])) {
			/* Steiner point overlaps with second terminal */
			if (edges NE NULL) {
				edges[0] = 0; edges[1] = 1;
				edges[2] = 1; edges[3] = 2;
			}
		}
		else if ((spx EQ terms[4]) AND (spy EQ terms[5])) {
			/* Steiner point overlaps with third terminal */
			if (edges NE NULL) {
				edges[0] = 0; edges[1] = 2;
				edges[2] = 1; edges[3] = 2;
			}
		}
		else {
			/* This is really a Steiner point */
			if (nsps NE NULL) {
				*nsps = 1;
			}
			if (sps NE NULL) {
				sps[0] = spx; sps[1] = spy;
			}
			if (nedges NE NULL) {
				*nedges = 3;
			}
			if (edges NE NULL) {
				edges[0] = 0; edges[1] = 3;
				edges[2] = 1; edges[3] = 3;
				edges[4] = 2; edges[5] = 3;
			}
		}
	}
	if (length NE NULL) {
		*length = terms[xorder[2]] - terms[xorder[0]] +
			  terms[yorder[2]] - terms[yorder[0]];
	}

	return (0);
}

/*
 * High-level function to compute a rectilinear Steiner tree
 */

	int
gst_rsmt (

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
			*length = fabs(terms[2] - terms[0]) +
				  fabs(terms[3] - terms[1]);
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
		gst_rsmt3 (terms, length, nsps, sps, nedges, edges);
	}
	else {
		/* General case: Generate FSTs and concatenate */

		hg = gst_generate_rfsts (nterms, terms, params, &status);

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
