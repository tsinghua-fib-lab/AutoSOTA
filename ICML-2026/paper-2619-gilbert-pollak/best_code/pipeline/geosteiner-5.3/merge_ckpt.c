/***********************************************************************

	$Id: merge_ckpt.c,v 1.19 2022/11/19 13:45:53 warme Exp $

	File:	merge_ckpt.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1995, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	A program that restores a given checkpoint file (including
	its saved nodes), merges the constraints from one or more
	other checkpoint files, and then write the resulting state
	out to a new checkpoint file.

************************************************************************

	Modification Log:

	a-1:	01/07/2000	warme
		: Created.
	b-1:	08/05/2002	benny
		: Some changes for library release.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Make features unconditional.

************************************************************************/

#include "bb.h"
#include "ckpt.h"
#include "fputils.h"
#include "geosteiner.h"
#include "logic.h"
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
 * The main routine for the "merge_ckpt" program.  It takes three or more
 * pathnames as arguments, and reads FST data from stdin.
 * The first pathname is a checkpoint file to restore.  The last pathname
 * is a new checkpoint file to create.  The intervening pathnames are
 * for checkpoints files from which to load the constraints and merge
 * them into the restored checkpoint (before writing it out again).
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

	if (argc < 4) {
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

		checkpoint_filename = argv [argc - 1];
		argv [argc - 1] = NULL;

		/* Merge in constraints from specified files. */
		_gst_merge_constraints (bbip, &argv [2]);

		_gst_write_checkpoint (bbip);

		status = 0;
	}

	gst_close_lpsolver ();
	gst_free_param (params);
	gst_close_geosteiner ();

	_gst_restore_floating_point_configuration (&fpsave);

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
		 "\nUsage: %s restore.chk merge1.chk merge2.chk... new.chk\n",
		 me);
	exit (1);
}
