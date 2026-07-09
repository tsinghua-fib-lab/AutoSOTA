/***********************************************************************

	$Id: prunefstmain.c,v 1.25 2022/11/19 13:45:54 warme Exp $

	File:	prunefstmain.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Martin Zachariasen.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Removed global variables.
		: Uses parameters.
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
#include <string.h>

/*
 * Global Routines
 */

int			main (int, char **);

/*
 * External References
 */

	/* none */

/*
 * Local Routines
 */

static void		decode_params (int, char**, gst_param_ptr);
static void		usage ();

/*
 * Local Variables
 */

static char *		description;
static char *		me;
static bool		Print_Detailed_Timings = FALSE;

/*
 * The main routine for the "prunefst" program.	 It takes the output from
 * the Euclidean or rectilinear FST generator, prunes FSTs that cannot
 * appear in an optimal solution and outputs the pruned set of FSTs
 * to the backend (FST concatenation procedure).
 */

	int
main (

int		argc,
char **		argv
)
{
int		res;
int		status;
gst_param_ptr	params;
gst_hg_ptr	H;
gst_hg_ptr	H2;
gst_channel_ptr	chan;

	me = argv [0];

	res = 0;

	setbuf (stdout, NULL);

	/* Initialize geosteiner */
	if (gst_open_geosteiner () NE 0) {
		fprintf (stderr, "%s: Unable to open geosteiner.\n", me);
		exit (1);
	}

	/* Parse arguments and setup the parameters */
	params = gst_create_param (NULL);
	decode_params (argc, argv, params);

	/* Setup output channel if needed */
	chan = NULL;
	if (Print_Detailed_Timings) {
		chan = gst_create_channel (NULL, NULL);
		gst_channel_add_file (chan, stderr, NULL);
		gst_set_chn_param (params,
				   GST_PARAM_DETAILED_TIMINGS_CHANNEL,
				   chan);
	}

	/* Read FST data */
	H = gst_load_hg (stdin, NULL, NULL);

	/* Prune edges */
	H2 = gst_hg_prune_edges (H, params, &status);
	if (H2 NE NULL) {
		/* Print pruned FST data */
		gst_set_str_property (gst_get_hg_properties (H2),
				      GST_PROP_HG_NAME,
				      description);

		gst_save_hg (stdout, H2, params);
	}
	else {
		res = 1;
		fprintf (stderr, "Prune edges returned status = %d\n",
			 status);
	}

	gst_free_hg (H2);
	gst_free_hg (H);
	gst_free_channel (chan);
	gst_free_param (params);
	gst_close_geosteiner ();

	CHECK_MEMORY
	exit (res);
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
char *		pname;
int		v;
int		min;
int		max;
int		rv;

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
				gst_set_int_param (params,
						   GST_PARAM_BSD_METHOD,
						   GST_PVAL_BSD_METHOD_LOGARITHMIC);
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

			case 't':
				Print_Detailed_Timings = TRUE;
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

			case 'Z':
				if (*ap EQ '\0') {
					if (argc <= 0) {
						usage ();
					}
					ap = *argv++;
					--argc;
				}
				pname = ap;
				if (argc <= 1) {
					usage ();
				}
				ap = *argv++;
				--argc;

				rv = gst_set_param (params, pname, ap);
				switch (rv) {
				case 0:
					/* The parameter was correctly set */
					break;
				case GST_ERR_UNKNOWN_PARAMETER_ID:
					fprintf(stderr,
					    "Parameter '%s' does not exist.\n",
					    pname);
					usage ();
					break;
				case GST_ERR_PARAMETER_VALUE_OUT_OF_RANGE:
					fprintf(stderr,
					    "Parameter value, %s, for '%s' is out of range.\n",
					    ap, pname);
					usage ();
					break;
				default:
					usage ();
				}

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
	"\t-b\tUse linear space and logarithmic time lookup for BSDs.",
	"\t-d txt\tDescription of problem instance.",
	"\t-t\tPrint detailed timings on stderr.",
	"\t-v N\tGenerates version N output data format.",
	"\t-Z P V\tSet parameter P to value V.",
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
			"\nUsage: %s [-bt] [-d description] [-v N] [-Z P V]",
			me);
	pp = &arg_doc [0];
	while ((p = *pp++) NE NULL) {
		(void) fprintf (stderr, "%s\n", p);
	}
	exit (1);
}

