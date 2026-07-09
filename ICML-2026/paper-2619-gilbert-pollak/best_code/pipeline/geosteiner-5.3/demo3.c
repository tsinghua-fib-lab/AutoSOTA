/***********************************************************************

	$Id: demo3.c,v 1.20 2022/11/19 13:45:50 warme Exp $

	File:	demo3.c
	Rev:	e-2
	Date:	09/05/2016

	Copyright (c) 2003, 2022 by David M. Warme, Pawel Winter.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************
	GeoSteiner callable library demo program.

	Takes input from OR-library file and computes Steiner trees
	for each instance in the input.
	The metric (default rectilinear) and max. FST size
	can be set by command line parameters to the program.
	Display length of each instance and total length.
	(An absolute minimum of error-checking is performed.)
************************************************************************

	Modication Log:

	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.

***********************************************************************/

#include <stdlib.h>
#include "geosteiner.h"

int main (int argc, char** argv)
{
	int i, lambda = 2, num_instances, num_terms;
	double * terms, length, total_length = 0.0, max_fst_size = 0;
	gst_metric_ptr metric;
	gst_param_ptr params;

	/* Read command line parameters (metric and max. FST size) */
	if (argc >= 2) lambda	    = atoi (argv [1]);
	if (argc >= 3) max_fst_size = atof (argv [2]);

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
	if (max_fst_size >= 2)
		gst_set_int_param (params, GST_PARAM_MAX_FST_SIZE, max_fst_size);

	/* Read the number of instances and then the instances thenselves */
	scanf ("%d", &num_instances);
	for (i = 1; i <= num_instances; i++) {

		/* Read instance from stdin */
		scanf ("%d", &num_terms);
		terms = (double *) malloc (2*num_terms*sizeof(double));
		gst_get_points (stdin, num_terms, &terms, NULL);

		/* Compute Steiner tree */
		gst_smt (num_terms, terms, &length, NULL, NULL, NULL, NULL, NULL,
			 metric, params);

		printf ("Instance %5d has %5d terminals and length %f\n",
			i, num_terms, length);
		total_length += length;
		free (terms);
	}
	printf ("\nTotal length of all instances is %f\n", total_length);

	/* Clean up */
	gst_free_metric (metric);
	gst_free_param (params);
	gst_close_geosteiner ();

	exit (0);
}
