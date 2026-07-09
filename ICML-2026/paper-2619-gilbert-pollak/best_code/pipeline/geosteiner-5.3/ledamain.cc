/***********************************************************************

        $Id: ledamain.cc,v 1.6 2022/11/19 13:45:53 warme Exp $

	File:	ledamain.cc
	Rev:	e-2
	Date:	09/05/2016

	Copyright (c) 2003, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

************************************************************************

        Modification Log:

        a-1:    01/30/03        benny
                : Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.

************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include <LEDA/array.h>
#include <LEDA/basic.h>
#include <LEDA/graph.h>
#include <LEDA/list.h>
#include <LEDA/point.h>

#include "geosteiner.h"

/*
 * Examples of LEDA-fied interfaces for GeoSteiner
 */

double	GST_ESMT (list<point>& L, GRAPH<point,double>& T);
double	GST_RSMT (list<point>& L, GRAPH<point,double>& T);
double	GST_OSMT (list<point>& L, GRAPH<point,double>& T);
double	GST_USMT (list<point>& L, GRAPH<point,double>& T, int lambda = 4);

/*
 * A very simple main function using the LEDA interfaces to find lengths of
 * SMTs
 */
	int
main ()
{
int			i;
list<point>		L;
GRAPH<point,double>	T;
point p[] = { point(100,100), point(100,200), point(500,100), point(500,200) };

	for (i = 0; i < 4; i++) {
		L.push (p[i]);
	}

	/* Initialize GeoSteiner */
	gst_open_geosteiner ();

	printf ("The length is: %f\n", GST_ESMT (L, T));
	printf ("The length is: %f\n", GST_RSMT (L, T));
	printf ("The length is: %f\n", GST_OSMT (L, T));
	printf ("The length is: %f\n", GST_USMT (L, T, 8));

	/* Close GeoSteiner */
	gst_close_geosteiner ();
}

/*
 * A general interface to the gst_smt() function in GeoSteiner
 */
	double
GST_USMT (

list<point>&		L,
GRAPH<point,double>&	T,
int			lambda = 4
)
{
int		i;
int		nterms;
int		nedges;
int		nsps;
int *		edges;
double		length;
double *	sps;
double *	terms;
double *	tmp;

	nterms = L.size();
	if(nterms < 2) {
		// Error.
	}

	/* Collect points in a simple array */
	tmp = terms = (double *)malloc (4 * nterms * sizeof(double));
	point p;
	forall(p, L) {
		*tmp++ = p.xcoord();
		*tmp++ = p.ycoord();
	}

	/* Allocate memory for the SMT and then compute it */
	sps = &terms[2*nterms];
	edges = (int *)malloc (2 * nterms * 2 * sizeof(int));

	/* Use GeoSteiner to find a solution */
	gst_metric_ptr metric;
	switch(lambda)
	{
	case 0:
		metric = gst_create_metric(GST_METRIC_L, 2, NULL);
		break;
	case 1:
		metric = gst_create_metric(GST_METRIC_L, 1, NULL);
		break;
	default:
		metric = gst_create_metric(GST_METRIC_UNIFORM, lambda, NULL);
		break;
	}

	gst_smt (nterms, terms, &length, &nsps, sps, &nedges, edges, NULL, metric, NULL);

	/* Build LEDA graph */
	for (int i = 0; i < nterms; i++) {
		T.new_node (point(terms[2*i], terms[2*i+1]));
	}
	for (int i = 0; i < nsps; i++) {
		T.new_node (point(sps[2*i], sps[2*i+1]));
	}

	/* Get a list of all nodes */
	list<node> A = T.all_nodes();

	for (int i = 0; i < nedges; i++) {
		int a = edges[2*i];
		int b = edges[2*i+1];
		double length = gst_distance (metric,
					      terms[2*a], terms[2*a+1],
					      terms[2*b], terms[2*b+1]);

/* This is inefficient (quadratic) !!!!!!!!!!!!!!!!!!!!!!!!! */
		T.new_edge (A[A[a]], A[A[b]], length);
	}

	/* Clean up */
	gst_free_metric(metric);
	free (edges);
	free (terms);

	return length;
}

/*
 * Various wrappers for more specific functions.
 */

	double
GST_ESMT (

list<point>& L,
GRAPH<point,double>& T
)
{
	return GST_USMT(L, T, 0);
}

	double
GST_RSMT (

list<point>& L,
GRAPH<point,double>& T
)
{
	/* Note the use of 1 instead of 2. This is to allow the use of two
	   different rectilinear generators. See the interpretation of
	   this value in GST_USMT */
 	return GST_USMT(L, T, 1);
}

	double
GST_OSMT (

list<point>& L,
GRAPH<point,double>& T
)
{
	return GST_USMT(L, T);
}
