/***********************************************************************

	$Id: smt.c,v 1.37 2022/11/19 13:45:55 warme Exp $

	File:	smt.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winer, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Function to solve general SMT problems. It only uses library
	functions.

************************************************************************

	Modification Log:

	a-1:	08/05/2002	benny
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
#include "memory.h"
#include "metric.h"
#include "prepostlude.h"
#include "steiner.h"

/* Fast specialized functions for small instances */

extern int gst_esmt3 (double*, double*, int*, double*, int*, int*);
extern int gst_rsmt3 (double*, double*, int*, double*, int*, int*);


	int
gst_smt (

int			nterms,
double *		terms,
double *		length,
int *			nsps,
double *		sps,
int *			nedges,
int *			edges,
int *			soln_status,
gst_metric_ptr		metric,
gst_param_ptr		params
)
{
int		nfsts;
int		status;
int		res;
int		type;
int		parameter;
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

	type = metric -> type;
	parameter = metric -> parameter;

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
			*length = gst_distance (metric, terms[0], terms[1], terms[2], terms[3]);
		}
		if (nedges NE NULL) {
			*nedges = 1;
		}
		if (edges NE NULL) {
			edges[0] = 0;
			edges[1] = 1;
		}
	}
	else if ((nterms EQ 3) AND (params EQ NULL) AND
		 (((type EQ GST_METRIC_L) AND (parameter EQ 1)) OR
		  ((type EQ GST_METRIC_UNIFORM) AND (parameter EQ 2)))) {
		/* Use specialized fast code for rectilinear problem */
		gst_rsmt3 (terms, length, nsps, sps, nedges, edges);
	}
	else if ((nterms EQ 3) AND (params EQ NULL) AND
		 ((type EQ GST_METRIC_L) AND (parameter EQ 2))) {
		/* Use specialized fast code for Euclidean problem */
		gst_esmt3 (terms, length, nsps, sps, nedges, edges);
	}
	else {
		/* General case: Generate FSTs and concatenate */

		hg = gst_generate_fsts (nterms, terms, metric, params, &status);
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
				status = gst_hg_solution (solver, NULL, NULL, length, 0);
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
