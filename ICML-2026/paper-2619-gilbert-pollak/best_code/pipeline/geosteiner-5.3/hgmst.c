/***********************************************************************

	$Id: hgmst.c,v 1.17 2023/04/03 17:29:33 warme Exp $

	File:	hgmst.c
	Rev:	e-4
	Date:	04/03/2023

	Copyright (c) 2002, 2023 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	A simple function to solve hypergraph MST problems.

************************************************************************

	Modification Log:

	a-1:	04/30/02	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, fix -Wall issues.
	e-4:	04/03/2023	warme
		: Always set solution status (if requested).

************************************************************************/

#include "geosteiner.h"
#include "logic.h"
#include "prepostlude.h"
#include "steiner.h"

/*
 * Given hypergraph information this function finds the MST. If any of the
 * pointers length, nmstedges or mstedges is NULL then the corresponding
 * information is not returned.
 */

	int
gst_hgmst (

int		nterms,
int		nedges,
int *		edge_sizes,
int *		edges,
double *	weights,
double *	length,
int *		nmstedges,
int *		mstedges,
int *		soln_status,
gst_param_ptr	params
)
{
int		res;
gst_hg_ptr	H;
gst_solver_ptr	solver;

	GST_PRELUDE

	res = 0;

	if (soln_status NE NULL) {
		*soln_status = GST_STATUS_NO_SOLUTION;
	}

	/* Setup hypergraph */
	H = gst_create_hg (NULL);
	gst_set_hg_number_of_vertices (H, nterms);
	gst_set_hg_edges (H, nedges, edge_sizes, edges, weights);

	/* Setup solver and find solution */
	solver = gst_create_solver (H, params, NULL);

	res = gst_hg_solve (solver, NULL);

	if (res EQ 0) {
		/* No serious problems encountered */
		/* Update the wanted information */
		gst_hg_solution (solver, nmstedges, mstedges, length, 0);
		gst_get_solver_status (solver, soln_status);
	}

	/* Free the temporary structures */
	gst_free_hg (H);
	gst_free_solver (solver);

	GST_POSTLUDE

	return res;
}
