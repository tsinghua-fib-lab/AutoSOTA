/***********************************************************************

	$Id: demo1.c,v 1.10 2022/11/19 13:45:50 warme Exp $

	File:	demo1.c
	Rev:	e-2
	Date:	09/05/2016

	Copyright (c) 2003, 2022 by David M. Warme, Pawel Winter.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	GeoSteiner callable library demo program.

	Construct an Euclidean Steiner tree for the point set
	(0,0), (0,1), (1,0) and (1,1). Display length and Steiner
	points.

************************************************************************

	Modification Log:

	a-1:	02/02/2014	warme
		: Added include file.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.

***********************************************************************/

#include "geosteiner.h"
#include "stdlib.h"

int main (int argc, char** argv)
{
	double terms [8] = { 0, 0,
			     0, 1,
			     1, 0,
			     1, 1 };

	int i, nsps;
	double length, sps [4];

	/* Open GeoSteiner environment */
	if (gst_open_geosteiner () != 0) {
		printf ("Could not open GeoSteiner.\n");
		exit (1);
	}

	/* Compute Euclidean Steiner tree */
	gst_esmt (4, terms, &length, &nsps, sps, NULL, NULL, NULL, NULL);

	/* Display information about solution */
	printf ("Steiner tree has length %f\n", length);

	for (i = 0; i < nsps; i++) {
		printf ("Steiner point: (%f, %f)\n", sps[2*i], sps[2*i+1]);
	}

	/* Close GeoSteiner environment */
	gst_close_geosteiner ();

	exit (0);
}
