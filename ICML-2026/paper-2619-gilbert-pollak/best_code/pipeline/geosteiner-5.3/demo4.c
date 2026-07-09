/***********************************************************************

	$Id: demo4.c,v 1.12 2022/11/19 13:45:50 warme Exp $

	File:	demo4.c
	Rev:	e-2
	Date:	09/05/2016

	Copyright (c) 2003, 2022 by David M. Warme, Pawel Winter.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************
	GeoSteiner callable library demo program.

	Generate a random instance with NUM_TERMS terminals.
	Generate rectilinear FSTs and solve concatenation problem to
	within MAX_EXCESS of optimality. Display status of solution
	process every TIME_INTERVAL seconds.
	(An absolute minimum of error-checking is performed.)
************************************************************************

	Modification Log:

	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.

***********************************************************************/

#include <math.h>
#include <stdlib.h>
#include "geosteiner.h"

#define NUM_TERMS	1000
#define TIME_INTERVAL	2
#define MAX_EXCESS	0.1

int main (int argc, char** argv)
{
	int j, status, soln_status;
	double terms[2*NUM_TERMS], lb, ub, cpu;
	gst_hg_ptr hg;	gst_solver_ptr solver;	gst_param_ptr params;

	/* Open GeoSteiner environment */
	if (gst_open_geosteiner () != 0) {
		printf ("Could not open GeoSteiner.\n");
		exit (1);
	}
	/* Generate random terminals with coordinates in range 0..9999 */
	srand48 (1);
	for (j = 0; j < 2*NUM_TERMS; j++)
		terms[j] = floor (drand48 () * 10000.0);
	/* Generate full Steiner trees (default parameters) */
	hg = gst_generate_rfsts (NUM_TERMS, terms, NULL, &status);
	/* Set up solver and its parameters */
	params = gst_create_param (&status);
	gst_set_dbl_param (params, GST_PARAM_CPU_TIME_LIMIT, TIME_INTERVAL);
	solver = gst_create_solver (hg, params, &status);

	for (;;) {
		gst_hg_solve (solver, NULL);
		gst_get_solver_status (solver, &soln_status);
		switch (soln_status) {
			case GST_STATUS_OPTIMAL:
			case GST_STATUS_FEASIBLE:
				gst_get_dbl_property (gst_get_solver_properties (solver),
						      GST_PROP_SOLVER_LOWER_BOUND, &lb);
				gst_get_dbl_property (gst_get_solver_properties (solver),
						      GST_PROP_SOLVER_CPU_TIME, &cpu);
				gst_hg_solution (solver, NULL, NULL, &ub, 0);

				printf ("Time: %.2f. LB = %f, UB = %f, ratio = %f\n",
					 cpu, lb, ub, ub/lb); break;
			case GST_STATUS_INFEASIBLE:
				printf ("Problem is infeasible!\n"); break;
			case GST_STATUS_NO_FEASIBLE:
				gst_get_dbl_property (gst_get_solver_properties (solver),
						      GST_PROP_SOLVER_CPU_TIME, &cpu);
				printf ("Time: %.2f. No feasible solution found yet.\n", cpu);
		}
		if (soln_status == GST_STATUS_OPTIMAL) break;
		if ((soln_status == GST_STATUS_FEASIBLE) &&
		    (ub/lb < 1.0 + (MAX_EXCESS / 100.0))) break;
	}
	/* Clean up */
	gst_free_solver (solver);  gst_free_hg (hg);
	gst_free_param (params);   gst_close_geosteiner ();
	exit (0);
}
