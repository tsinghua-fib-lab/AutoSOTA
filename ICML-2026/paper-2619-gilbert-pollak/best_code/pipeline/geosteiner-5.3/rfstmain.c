/***********************************************************************

	$Id: rfstmain.c,v 1.33 2022/11/19 13:45:54 warme Exp $

	File:	rfstmain.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Martin Zachariasen.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	04/30/2002	benny
		: Created. Split off from rfst.c.
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
 * Local Routines
 */

static void		decode_params (int, char **, gst_param_ptr);
static void		usage (void);

/*
 * Local Variables
 */

static char *		description;
static char *		me;
static bool		Print_Detailed_Timings = FALSE;

/*
 * The main routine for the "rfst" program.  It reads a point set
 * from standard input, generates all of the rectilinear FSTs,
 * and outputs them to standard output in our special "phase 1 I/O"
 * format.
 */

	int
main (

int		argc,
char **		argv
)
{
int			n;
int			res;
int			status;
double *		terms;
gst_param_ptr		params;
gst_hg_ptr		hg;
gst_channel_ptr		chan;
gst_scale_info_ptr	scinfo;

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
	gst_set_int_param (params,
		GST_PARAM_INCLUDE_CORNERS,
		GST_PVAL_INCLUDE_CORNERS_ENABLE);
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

	/* Read the points from stdin and generate the RFSTs */
	scinfo = gst_create_scale_info (NULL);
	n = gst_get_points (stdin, 0, &terms, scinfo);

	/* Scan the terminals and update the scale_info structure. This
	   will make later output much nicer. */
	gst_compute_scale_info_digits (n, terms, scinfo);
	hg = gst_generate_rfsts (n, terms, params, &status);

	if (hg NE NULL) {
		gst_set_hg_scale_info (hg, scinfo);

		/* Print all of the data we have generated to stdout... */
		gst_set_str_property (gst_get_hg_properties (hg),
				      GST_PROP_HG_NAME,
				      description);
		gst_save_hg (stdout, hg, params);
	}
	else {
		fprintf (stderr, "RFST generator returned status = %d\n",
			 status);
		res = 1;
	}

	/* Clean up. */
	free (terms);
	gst_free_hg (hg);
	gst_free_channel (chan);
	gst_free_param (params);
	gst_free_scale_info (scinfo);
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
	"\t-d txt\tDescription of problem instance.",
	"\t-k K\tOnly generate FSTs spanning up to K terminals.",
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
			"\nUsage: %s [-t] [-d txt] [-k K] [-v N] [Z P V]"
			" <points-file\n",
			me);
	pp = &arg_doc [0];
	while ((p = *pp++) NE NULL) {
		(void) fprintf (stderr, "%s\n", p);
	}
	exit (1);
}

