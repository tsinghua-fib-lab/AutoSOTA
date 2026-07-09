/***********************************************************************

	$Id: fst2graphmain.c,v 1.21 2022/11/19 13:45:52 warme Exp $

	File:	fst2graphmain.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	08/05/2002	benny
		: Created. Split off from fst2graph.c.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
		: Change command line arguments to support
		:  SteinLib "integer" format.
	e-3:	09/24/2016	warme
		: Reorganize include files, fix -Wall issues.

************************************************************************/

#include "geosteiner.h"
#include "gsttypes.h"
#include "logic.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>

/*
 * Global Routines
 */

int			main (int, char **);


/*
 * Local Routines
 */

static void		decode_params (int, char **, gst_param_ptr);
static void		usage (void);


/*
 * Local Variables
 */

static char *		description = NULL;
static char *		me;

static bool	      Print_Unscaled	= FALSE;

/*
 * This is the main routine for computing Reduced Grid-Graphs.
 */

	int
main (

int		argc,
char **		argv
)
{
int			m;
int			p;
gst_hg_ptr		H;
gst_hg_ptr		H2;
gst_scale_info_ptr	scinfo;
gst_param_ptr		params;
gst_metric_ptr		metric;

	me = argv [0];

	setbuf (stdout, NULL);

	if (gst_open_geosteiner () NE 0) {
		fprintf (stderr, "%s: Unable to open geosteiner.\n", me);
		exit (1);
	}

	params = gst_create_param (NULL);
	gst_set_int_param (params, GST_PARAM_SAVE_FORMAT,
			   GST_PVAL_SAVE_FORMAT_ORLIBRARY);
	decode_params (argc, argv, params);

	H = gst_load_hg (stdin, NULL, NULL);
	H2 = gst_hg_to_graph (H, params, NULL);
	gst_free_hg (H);

	/* Examine the metric */
	if (NOT Print_Unscaled) {
		gst_get_hg_metric (H2, &metric);
		gst_get_metric_info (metric, &m, &p);
		if (   ((m EQ GST_METRIC_L) AND (p EQ 1))
		    OR ((m EQ GST_METRIC_UNIFORM) AND (p EQ 2))) {
			/* Force printing of integer data. */
			scinfo = gst_create_scale_info (NULL);
			gst_set_hg_scale_info (H2, scinfo);
			gst_free_scale_info (scinfo);
#if 0 /* FIXME:
	 The above code does not work as well as the code below. The above
	 sets min_precision to DBL_DIG + 1. But is the method below really
	 safe even if it was possible to do it? */

			H2 -> scale.min_precision = 0;
			if (H2 -> scale.scale > 0) {
				H2 -> scale.scale = 0;
			}
#endif
		}
	}

	if (description NE NULL) {
		gst_set_str_property (gst_get_hg_properties (H2),
	                              GST_PROP_HG_NAME,
	                              description);
	}

	gst_save_hg (stdout, H2, params);

	gst_free_hg (H2);
	gst_free_param (params);
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
int		v;
int		min;
int		max;
int		num_bits;

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
			case 'b':
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					--argc;
				}
				if ((*ap < '0') OR (*ap > '9')) {
					usage ();
				}
				num_bits = atoi (ap);
				gst_query_int_param (params,
						     GST_PARAM_SAVE_INT_NUMBITS,
						     NULL, NULL, &min, &max);
				if ((num_bits < min) OR
				    (num_bits > max)) {
					fprintf (stderr,
						 "%s: Bad number of bits `%s'."
						 "  Valid range is"
						 " from %d to %d.\n",
						 me,
						 ap,
						 min,
						 max);
					usage ();
				}
				gst_set_int_param (params,
						   GST_PARAM_SAVE_INT_NUMBITS,
						   num_bits);
				ap = "";
				break;

			case 'd':
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					--argc;
				}
				if (strlen (ap) >= 80) {
					fprintf (stderr,
						"Description must be less"
						" than 80 characters.\n");
					usage ();
				}
				description = ap;
				/* Change newlines to spaces... */
				for (;;) {
					ap = strchr (ap, '\n');
					if (ap EQ NULL) break;
					*ap++ = ' ';
				}
				ap = "";
				break;

			case 'e':
				gst_set_int_param (params,
						   GST_PARAM_GRID_OVERLAY,
						   GST_PVAL_GRID_OVERLAY_DISABLE);
				break;

			case 'u':
				Print_Unscaled = TRUE;
				break;

			case 'v':
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					--argc;
				}
				if ((*ap < '0') OR (*ap > '9')) {
					usage ();
				}
				v = atoi (ap);
				gst_query_int_param (params,
						     GST_PARAM_SAVE_FORMAT,
						     NULL, NULL, &min, &max);
				if ((v < min) OR
				    (v > max)) {
					fprintf (stderr,
						 "%s: Bad version `%s'."
						 "  Valid versions range"
						 " from %d to %d.\n",
						 me,
						 ap,
						 min,
						 max);
					usage ();
				}
				gst_set_int_param (params, GST_PARAM_SAVE_FORMAT, v);
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
	"Reads FST info from stdin. Produces an ordinary graph",
	"on stdout which is either:",
	" - graph of all line segments in all FSTs (default for Euclidean problem)",
	" - reduced grid graph (default for rectilinear problem)",
	"Output data is printed in OR-Library format.",
	"Distances in the rectilinear problem are scaled to integers."
	"",
	"\t-d txt\tDescription of problem instance.",
	"\t-e\tGenerate edge graph for the rectilinear problem.",
	"\t-u\tOutput unscaled (fractional) data.",
	"\t-v N\tGenerates version N output data format.",
	"",
	NULL
};

	static
	void
usage (void)

{
char **		pp;
char *		p;

	(void) fprintf (stderr, "\nUsage: %s [-eu]"
				" [-d description]"
				" [-v N]"
				"\n",
				me);
	pp = &arg_doc [0];
	while ((p = *pp++) NE NULL) {
		(void) fprintf (stderr, "%s\n", p);
	}
	exit (1);
}
