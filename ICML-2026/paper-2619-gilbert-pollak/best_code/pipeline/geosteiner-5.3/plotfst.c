/***********************************************************************

	$Id: plotfst.c,v 1.22 2022/11/19 13:45:53 warme Exp $

	File:	plotfst.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Main routine for a utility to plot the FSTs in
	various ways.

************************************************************************

	Modification Log:

	b-1:	01/10/97	warme
		: Split off from old_bs.c.
		: Reading in the phase 1 data.
	c-1:	01/21/2001	warme
		: Changed name from "fsplot" to "plotfst".
		: Added -p argument to plot the point set.
	c-2:	08/05/2002	benny
		: Numerous changes for library release.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.

************************************************************************/

#include "genps.h"
#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>


/*
 * Global Routines
 */

int			main (int, char **);

bool			Print_Grouped_Full_Sets		= FALSE;
bool			Print_Overlaid_Full_Sets	= FALSE;


/*
 * External References
 */

	/* none */


/*
 * Local Routines
 */

static void		decode_params (int, char **);
static void		usage (void);


/*
 * Local Variables
 */

static char *		me;
static bool		Print_Full_Sets = FALSE;
static bool		Print_Points = FALSE;

/*
 * The main routine for the plotfst utility.
 */

	int
main (

int		argc,
char **		argv
)
{
int			i;
int			nedges;
int			nverts;
int *			all_edges;
double			p1time;
double *		verts;
char			tbuf [20];
char			title [128];
gst_hg_ptr		H;
gst_channel_ptr		chan;
gst_proplist_ptr	hgprop;
gst_scale_info_ptr	sip;

	me = argv [0];

	setbuf (stdout, NULL);

	decode_params (argc, argv);

	if (gst_open_geosteiner () NE 0) {
		fprintf (stderr, "%s: Unable to open geosteiner.\n", me);
		exit (1);
	}

	/* Setup a channel for stdout */
	chan = gst_create_channel (NULL, NULL);
	gst_channel_add_file (chan, stdout, NULL);

	/* Parse the hypergraph from stdin */
	H = gst_load_hg (stdin, NULL, NULL);
	hgprop = gst_get_hg_properties (H);

	if (gst_get_dbl_property (hgprop,
				  GST_PROP_HG_GENERATION_TIME,
				  &p1time)) {
		sprintf (tbuf, "Unknown");
	}
	else {
		sprintf (tbuf, "%.2f", p1time);
	}

	printf (" %% Phase 1: %s seconds\n", tbuf);

	gst_get_hg_scale_info (H, &sip);

	/* Define all terminals. */
	gst_get_hg_terminals (H, &nverts, NULL);
	verts = NEWA (2*nverts, double);
	gst_get_hg_vertex_embedding (H, NULL, verts);
	_gst_define_Plot_Terminals (chan, nverts, verts, sip);

	gst_get_hg_edges (H, &nedges, NULL, NULL, NULL);

	if (Print_Points) {
		int length;
		char *tmp;
		tmp = NULL;
		gst_get_str_property (gst_get_hg_properties (H),
				      GST_PROP_HG_NAME,
				      &length,
				      NULL);
		tmp = NEWA (length + 1, char);
		gst_get_str_property (gst_get_hg_properties (H),
				      GST_PROP_HG_NAME,
				      NULL,
				      tmp);

		if ((tmp NE NULL) AND
		    (tmp [0] NE '\0')) {
			strcpy (title, tmp);
		}
		else {
			sprintf (title, "%u points", (int32u) nverts);
		}
		_gst_overlay_plot_subset (chan, H, title, 0, NULL, BIG_PLOT);
		free (tmp);
	}

	all_edges = NEWA (nedges, int);
	for (i = 0; i < nedges; i++) {
		all_edges [i] = i;
	}

	if (Print_Full_Sets) {
		_gst_plot_full_sets (chan, H, nedges, all_edges, SMALL_PLOT);
	}
	if (Print_Grouped_Full_Sets) {
		_gst_plot_full_sets_grouped (chan, H, nedges, all_edges, SMALL_PLOT);
	}
	if (Print_Overlaid_Full_Sets) {
		sprintf (title,
			 "All FSTs:  %u points,  %s seconds",
			 (int32u) nverts, tbuf);
		_gst_overlay_plot_subset (chan, H, title, nedges, all_edges, BIG_PLOT);
	}

	free (all_edges);
	free (verts);
	gst_free_channel (chan);
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
			case 'f':
				Print_Full_Sets = TRUE;
				break;

			case 'g':
				Print_Grouped_Full_Sets = TRUE;
				break;

			case 'o':
				Print_Overlaid_Full_Sets = TRUE;
				break;

			case 'p':
				Print_Points = TRUE;
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
	"\t-f\tPrints all full-sets in \"fly specks\" fashion.",
	"\t-g\tPrints full sets in \"grouped fly specks\" fashion.",
	"\t-o\tPrints all full-sets in overlaid fashion.",
	"\t-p\tPrints the point set, no FSTs.",
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
			"\nUsage: %s [-fgop]\n",
			me);
	pp = &arg_doc [0];
	while ((p = *pp++) NE NULL) {
		(void) fprintf (stderr, "%s\n", p);
	}
	exit (1);
}
