/***********************************************************************

	$Id: hgmstmain.c,v 1.16 2022/11/19 13:45:52 warme Exp $

	File:	hgmstmain.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************


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

#include "memory.h" /* Only because of memory checking... */
#include "geosteiner.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * A simple example of finding the MST in a hypergraph using GeoSteiner.
 */

	int
main (

int		argc,
char **		argv)
{
int		i;
double		length;
int		nterms = 5;
int		nedges = 5;
int		edge_sizes[] = {2, 2, 2, 2, 3};
int		edges[] = {0,1, 1,2, 2,3, 3,4, 0,3,4 };
double		weights[] = {10, 20, 30, 15, 10};
int		nmstedges;
int		mstedges[5];

gst_param_ptr	params;

	/* Initialize geosteiner */
	if (gst_open_geosteiner () != 0) {
		fprintf (stderr, "%s: Unable to open geosteiner.\n", argv [0]);
		exit (1);
	}

	params = gst_create_param (NULL);
#if 0
	gst_set_int_param (params,
			   GST_PARAM_HYPERGRAPH_SOLVER,
			   GST_PVAL_HYPERGRAPH_SOLVER_BACKTRACK_SEARCH);
#endif
	/* A single call to the library and we have our hypergraph MST */
	gst_hgmst (nterms, nedges, edge_sizes, edges, weights,
		   &length, &nmstedges, mstedges, NULL, params);

	printf ("Length: %lf\n", length);
	printf ("Number of edges: %d\n", nmstedges);
	for (i = 0; i < nmstedges; i++) {
		printf (" Edge number %d\n", mstedges[i]);
	}

	/* Clean up */
	gst_free_param (params);
	gst_close_geosteiner ();

	CHECK_MEMORY

	exit (0);
}
