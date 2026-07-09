/***********************************************************************

	$Id: demo2.c,v 1.15 2022/11/19 13:45:50 warme Exp $

	File:	demo2.c
	Rev:	e-2
	Date:	09/05/2016

	Copyright (c) 2003, 2022 by David M. Warme, Pawel Winter.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************
	GeoSteiner callable library demo program.

	Generate 10 random point sets, each having 50 terminals,
	and compute a Steiner tree for each instance.
	The metric (default rectilinear) and max. excess from optimum
	can be set by command line parameters to the program.
	Display length of each instance and total length.
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

#define NUM_INSTANCES	10
#define NUM_TERMS	50

int main (int argc, char** argv)
{
	int i, j, lambda = 2;
	double terms[2*NUM_TERMS], length, total_length = 0.0, max_excess = 0.0;
	gst_metric_ptr metric;
	gst_param_ptr params;

	/* Read command line parameters (metric and max. excess in percent) */
	if (argc >= 2) lambda	  = atoi (argv[1]);
	if (argc >= 3) max_excess = atof (argv[2]);

	/* Open GeoSteiner environment */
	if (gst_open_geosteiner () != 0) {
		printf ("Could not open GeoSteiner.\n");
		exit (1);
	}

	/* Set up metric */
	switch (lambda) {
	case 0: /* Euclidean metric */
		metric = gst_create_metric (GST_METRIC_L, 2, NULL); break;
	case 2: /* Rectilinear metric */
		metric = gst_create_metric (GST_METRIC_L, 1, NULL); break;
	default:/* General uniform metric */
		metric = gst_create_metric (GST_METRIC_UNIFORM, lambda, NULL);
	}

	/* Set up parameter set */
	params = gst_create_param (NULL);
	gst_set_dbl_param (params, GST_PARAM_GAP_TARGET, 1.0 + (max_excess/100.0));

	/* Generate NUM_INSTANCES random instances with NUM_TERMS terminals */
	srand48 (1);
	for (i = 1; i <= NUM_INSTANCES; i++) {

		/* Generate random points with coordinates in range 0..9999 */
		for (j = 0; j < 2*NUM_TERMS; j++)
			terms[j] = floor (drand48() * 10000.0);

		/* Compute Steiner tree and print length */
		gst_smt (NUM_TERMS, terms, &length, NULL, NULL, NULL, NULL, NULL,
			 metric, params);
		printf ("Instance %2d has length %f\n", i, length);
		total_length += length;
	}
	printf ("\nTotal length of all instances is %f\n", total_length);

	/* Clean up */
	gst_free_metric (metric);
	gst_free_param (params);
	gst_close_geosteiner ();

	exit (0);
}
