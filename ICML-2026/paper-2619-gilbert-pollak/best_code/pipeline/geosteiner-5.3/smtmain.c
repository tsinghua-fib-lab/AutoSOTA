/***********************************************************************

	$Id: smtmain.c,v 1.22 2022/11/19 13:45:55 warme Exp $

	File:	smtmain.c
	Rev:	e-2
	Date:	09/05/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	04/30/2002	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.

************************************************************************/

#include "logic.h"
#include "memory.h"	/* Only because of memory checking... */
#include "geosteiner.h"
#include <stdlib.h>
#include <stdio.h>

/*
 * Local Routines
 */

static void		decode_params (int, char **, gst_param_ptr);
static void		usage (void);

/*
 * Local Variables
 */

static char *		me;
static int		Lambda = 0;

/*
 * The main routine for the "smt" program.  It reads a point set
 * from standard input, generates the wanted SMT (Euclidean, rectilinear, ...)
 * and outputs solution information to standard output.
 */

	int
main (

int		argc,
char **		argv)
{
int		nterms;
double		length;
int		nsps;
int		nedges;
int *		edges;
double *	terms;
double *	sps;
gst_metric_ptr	metric;
gst_param_ptr	params;

	me = argv [0];

	/* Initialize geosteiner */
	if (gst_open_geosteiner () NE 0) {
		fprintf (stderr, "%s: Unable to open geosteiner.\n", me);
		exit (1);
	}

	/* Parse arguments and setup the parameters */
	params = gst_create_param (NULL);
	decode_params (argc, argv, params);

	/* Read points from stdin */
	nterms = gst_get_points (stdin, 0, &terms, NULL);

	printf("Number of terminals: %d\n", nterms);

	/* Allocate memory and find the SMT */
	sps = NEWA (2 * nterms, double);
	edges = NEWA (2 * nterms * 2, int);
	switch(Lambda) {
	case 0:
		metric = gst_create_metric (GST_METRIC_L, 2, NULL);
		break;
	case 2:
		metric = gst_create_metric (GST_METRIC_L, 1, NULL);
		break;
	default:
		metric = gst_create_metric (GST_METRIC_UNIFORM, Lambda, NULL);
	}

	gst_smt (nterms, terms, &length, &nsps, sps, &nedges, edges, NULL, metric, params);

	printf("Length of SMT: %f\n", length);

#if 1
	{ int i;
	printf("Number of Steiner points: %d\n", nsps);
	printf("Steiner points:\n");
	for(i = 0; i < nsps; i++)
		printf(" (%f, %f)\n", sps[2*i], sps[2*i+1]);

	printf("Number of edges: %d\n", nedges);
	for(i = 0; i < nedges; i++)
		printf(" (%d, %d)\n", edges[2*i], edges[2*i+1]);
	}
#endif
	/* Clean up */
	gst_free_metric (metric);
	gst_free_param (params);
	free (edges);
	free (sps);
	free (terms);

	/* Close geosteiner */
	gst_close_geosteiner ();

	CHECK_MEMORY
	exit (0);
}

/*
 * This routine decodes the various command-line arguments.
 */

	static
	void
decode_params (

int		argc,
char **		argv,
gst_param_ptr	params
)
{
char *		ap;
char		c;

	--argc;
	me = *argv++;
	while (argc > 0) {
		ap = *argv++;
		if (*ap NE '-') {
			usage ();
		}
		++ap;
		while ((c = *ap++) NE '\0') {
			switch (c) {
			case 'l':
				/* Number of uniformly distributed orientations */
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					--argc;
				}
				Lambda = atoi (ap);
				ap = "";
				break;

			case 'k':
				/* Max number of terminals in FSTs */
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					--argc;
				}
				gst_set_int_param(params, GST_PARAM_MAX_FST_SIZE, atoi(ap));
				ap = "";
				break;

			default:
				usage ();
				break;
			}
		}
		--argc;
	}
}


/*
 * This routine prints out the proper usage and exits.
 */

static char *	arg_doc [] = {
	"",
	"\t-l L\tOnly use L uniformly distributed orientations, 0 is Euclidean",
	"\t\tand 2 is rectilinear (default: 0)",
	"\t-k K\tOnly generate FSTs spanning up to K terminals.",
	"",
	NULL
};

	static
	void
usage (void)

{
char **		pp;
char *		p;

	(void) fprintf (stderr,
			"\nUsage: %s"
			" [-l K] [-k K]"
			" <points-file\n",
			me);
	pp = &arg_doc [0];
	while ((p = *pp++) NE NULL) {
		(void) fprintf (stderr, "%s\n", p);
	}
	exit (1);
}
