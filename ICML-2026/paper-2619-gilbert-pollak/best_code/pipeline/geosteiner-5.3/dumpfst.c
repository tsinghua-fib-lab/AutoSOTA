/***********************************************************************

	$Id: dumpfst.c,v 1.20 2022/11/19 13:45:51 warme Exp $

	File:	dumpfst.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Main routine for a utility to dump a set of FSTs.

************************************************************************

	Modification Log:

	a-1:	10/22/98	warme
		: Created.
	b-1:	01/21/2001	warme
		: Added -d and -h options to output only statistical
		:  summaries of the given FSTs.
		: Used global sort_ints() function.
	c-1:	08/05/2002	benny
		: Numerous changes for library release.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files.

************************************************************************/

#include "geosteiner.h"
#include "gsttypes.h"
#include "logic.h"
#include "memory.h"
#include <stdlib.h>


/*
 * Global Routines
 */

int			main (int, char **);


/*
 * Local Routines
 */

static int		comp_ints (const void *, const void *);
static void		decode_params (int, char **);
static void		usage (void);


/*
 * Local Variables
 */

static bool		aflag = FALSE;
static bool		dflag = FALSE;
static bool		hflag = FALSE;
static bool		lflag = FALSE;
static char *		me;
static bool		sflag = FALSE;

/*
 * For sorting integers in place
 */
	int
comp_ints (

const void *		p1,
const void *		p2
)
{
int			l1;
int			l2;

	l1 = *((int *) p1);
	l2 = *((int *) p2);

	if (l1 < l2) return (-1);
	if (l1 > l2) return (1);
	fprintf (stderr, "Warning: Identical terminals in fst\n");
	return (0);
}

/*
 * The main routine for the dumpfst utility.
 */

	int
main (

int		argc,
char **		argv
)
{
int			i;
int			j;
int			m;
int			n;
int			p;
int			nedges;
int			nverts;
int			nignore;
int			nterms;
int			nsteins;
int			npruned;
int			nrequired;
int			nmaybe;
int			total;
int *			edges;
int *			edge_sizes;
int *			bucket;
int *			fterms;
int *			pruned;
int *			required;
double *		weights;
gst_hg_ptr		H;
gst_metric_ptr		metric;

	me = argv [0];

	setbuf (stdout, NULL);

	decode_params (argc, argv);

	if (gst_open_geosteiner () NE 0) {
		fprintf (stderr, "%s: Unable to open geosteiner.\n", me);
		exit (1);
	}

	H = gst_load_hg (stdin, NULL, NULL);

	gst_get_hg_terminals (H, &nverts, NULL);
	gst_get_hg_edges (H, &nedges, NULL, NULL, NULL);
	edge_sizes	= NEWA (nedges, int);
	weights		= NEWA (nedges, double);
	gst_get_hg_edges (H, NULL, edge_sizes, NULL, weights);

	pruned = NEWA (nedges, int);
	required = NEWA (nedges, int);
	for (i = 0; i < nedges; i++) {
		gst_get_hg_edge_status (H, i, &pruned[i], &required[i]);
	}

	if (dflag) {

		/* Display statistics on the set of FSTs -- */
		/* summary info only. */
		gst_get_hg_metric (H, &metric);
		gst_get_metric_info (metric, &m, &p);
		if (((m EQ GST_METRIC_L) AND (p EQ 1))
			OR ((m EQ GST_METRIC_UNIFORM) AND (p EQ 2)))
			printf ("Metric:		Rectilinear\n");
		else if ((m EQ GST_METRIC_L) AND (p EQ 2))
			printf ("Metric:		Euclidean\n");
		else if (m EQ GST_METRIC_UNIFORM)
			printf ("Metric:		%d-metric\n", p);
		else if (m EQ GST_METRIC_NONE)
			printf ("Metric:		Graph\n");
		else
			printf ("Metric:		???\n");


		nignore = 0;
		nterms	= nverts;
		nsteins = 0;
#if 0
		for (i = 0; i < nverts; i++) {
			if (NOT BITON (vert_mask, i)) {
				++nignore;
			}
			else if (H -> tflag [i]) {
				++nterms;
			}
			else {
				++nsteins;
			}
		}
#endif
		printf ("Vertices:		%d\n", nverts);
		printf ("  Unused:		%d\n", nignore);
		printf ("  Terminals:		%d\n", nterms);
		printf ("  Steiners:		%d\n", nsteins);

		npruned	  = 0;
		nrequired = 0;
		nmaybe	  = 0;
		for (i = 0; i < nedges; i++) {
			if (pruned[i]) {
				++npruned;
			}
			else if (required[i]) {
				++nrequired;
			}
			else {
				++nmaybe;
			}
		}

		printf ("Edges:			%d\n", nedges);
		printf ("  Pruned:		%d\n", npruned);
		printf ("  Required:		%d\n", nrequired);
		printf ("  Undecided:		%d\n", nmaybe);
	}
	if (hflag) {
		bucket = NEWA (nverts + 1, int);
		for (i = 0; i <= nverts; i++) {
			bucket [i] = 0;
		}
		for (i = 0; i < nedges; i++) {
			if ((NOT aflag) AND pruned[i]) continue;
			j = edge_sizes [i];
			++(bucket [j]);
		}
		printf ("Size\tCount\n----\t-----\n");
		for (i = 0; i <= nverts; i++) {
			if (bucket [i] <= 0) continue;
			printf ("%d\t%d\n", i, bucket [i]);
		}
		free ((char *) bucket);
	}

	if ((NOT dflag) AND (NOT hflag)) {

		/* Not a summary mode -- dump the FSTs. */

		total = 0;
		for (i = 0; i < nedges; i++)
			total += edge_sizes [i];

		edges = NEWA (total, int);
		gst_get_hg_edges (H, NULL, NULL, edges, NULL);

		fterms = edges;
		for (i = 0; i < nedges; i++) {
			n = edge_sizes [i];
			if ((NOT aflag) AND pruned[i]) {
				fterms += n;
				continue;
			}
			if (sflag) {
				qsort (fterms, n, sizeof (*fterms), comp_ints);
			}
			for (j = 0; j < n; j++) {
				printf (" %d", *fterms++);
			}
			if (lflag) {
				printf (" %.16f", weights[i]);
			}
			printf ("\n");
		}

		free (edges);
	}

	free (pruned);
	free (required);
	free (weights);
	free (edge_sizes);
	gst_free_hg (H);
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
char **		argv
)
{
char *		ap;
char		c;
bool		full_dump;

	full_dump = FALSE;

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
			case 'a':
				aflag = TRUE;
				break;

			case 'd':
				dflag = TRUE;
				break;

			case 'h':
				hflag = TRUE;
				break;

			case 'l':
				lflag = TRUE;
				full_dump = TRUE;
				break;

			case 's':
				sflag = TRUE;
				full_dump = TRUE;
				break;

			default:
				usage ();
				break;
			}
		}
		--argc;
	}

	if ((dflag OR hflag) AND full_dump) {
		/* -d and -h are summary modes that contradict */
		/* the flags pertaining to full dumps of the FSTs. */
		usage ();
	}
	if (dflag AND (NOT hflag) AND aflag) {
		/* Attempting "dumpfst -d -a"... */
		usage ();
	}
}

/*
 * This routine prints out the proper usage and exits.
 */

static char *	arg_doc [] = {
	"",
	"\t-a\tDump all FSTs, even those marked as never used.",
	"\t-d\tDisplay statistics about FSTs.",
	"\t-h\tDisplay histogram of FST sizes.",
	"\t\t(-a includes never used FSTs in histogram.)",
	"\t-l\tInclude length of each FST.",
	"\t-s\tSorts the terminals of each FST.",
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
			"\nUsage:\n"
			"\t%s [-d] [-h [-a]] <FST_file\n"
			"\t%s [-als] <FST_file\n",
			me,
			me);
	pp = &arg_doc [0];
	while ((p = *pp++) NE NULL) {
		(void) fprintf (stderr, "%s\n", p);
	}
	exit (1);
}
