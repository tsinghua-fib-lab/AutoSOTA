/***********************************************************************

	$Id: analyze_ckpt.c,v 1.19 2022/11/19 13:45:49 warme Exp $

	File:	analyze_ckpt.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1995, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	A program that determines the subtour S associated with
	each constraint in the constraint pool.

************************************************************************

	Modification Log:

	a-1:	02/25/2000	warme
		: Created.
	b-1:	08/05/2002	benny
		: Adapted for library release.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Make features unconditional.

************************************************************************/

#include "analyze.h"
#include "bb.h"
#include "ckpt.h"
#include "fputils.h"
#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include "steiner.h"


/*
 * Global Routines
 */

int			main (int, char **);


/*
 * Local Routines
 */

static void		usage (void);


/*
 * Local Variables
 */

static char *		me;

/*
 * The main routine for the "analyze_ckpt" program.
 */

	int
main (

int		argc,
char **		argv
)
{
int			status;
char *			checkpoint_filename;
struct bbinfo *		bbip;
gst_hg_ptr		H;
gst_param_ptr		params;
struct fpsave		fpsave;

	me = argv [0];

	_gst_set_floating_point_configuration (&fpsave);

	setbuf (stdout, NULL);

	if (argc < 2) {
		usage ();
	}

	if (gst_open_geosteiner () NE 0) {
		fprintf (stderr, "%s: Unable to open geosteiner.\n", me);
		exit (1);
	}

	params = gst_create_param (NULL);
	checkpoint_filename = argv [1];
	gst_set_str_param (params, GST_PARAM_CHECKPOINT_FILENAME, checkpoint_filename);

	H = gst_load_hg (stdin, NULL, NULL);

	gst_open_lpsolver ();

	/* Restore a checkpoint, if available... */
	bbip = _gst_restore_checkpoint (H, params);
	if (bbip EQ NULL) {
		fprintf (stderr, "Error restoring checkpoint %s\n", argv [1]);
		status = 1;
	}
	else {

		printf ("**** CHECKPOINT RESTORED ****\n");

		_gst_analyze_constraints (bbip, FALSE);
		status = 0;
	}

	gst_close_lpsolver ();
	gst_free_param (params);
	gst_close_geosteiner ();

	_gst_restore_floating_point_configuration (&fpsave);

	CHECK_MEMORY
	exit (status);
}

/*
 * This routine prints out the proper usage and exits.
 */

	static
	void
usage (void)

{
	fprintf (stderr,
		 "\nUsage: %s file.chk\n",
		 me);
	exit (1);
}
