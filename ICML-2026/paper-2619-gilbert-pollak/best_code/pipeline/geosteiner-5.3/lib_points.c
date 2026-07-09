/***********************************************************************

	$Id: lib_points.c,v 1.10 2022/11/19 13:45:53 warme Exp $

	File:	lib_points.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2000, 2022 by Martin Zachariasen.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Convert OR-LIBRARY or TSPLIB file into a "clean" set of
	points that can be read by "rfst" or "efst".
	The coordinates themselves are just dumped
	exactly as they appear in the instance file.
	It is a filter with one optional parameter that specifies
	the instance number in an OR-LIBRARY file.
	The program identifies the instance file type automatically.

************************************************************************

	Modification Log:

	a-1:	01/22/2000	martinz
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files.

************************************************************************/

#include "gsttypes.h"
#include "logic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Global Routines
 */

int			main (int, char **);


/*
 * Local Routines
 */

static void		decode_params (int, char **);
static void		usage (void);


/*
 * Local Variables
 */
static char *		me;
static int		orlib_instance = 1;

/*
 * The main routine lib_points utility.
 */

	int
main (

int		argc,
char **		argv
)
{
int			i;
int			j;
int			n;
int			num_instances;
bool			found;
char			buf [256];

	setbuf (stdout, NULL);

	decode_params (argc, argv);

	scanf ("%s", buf);

	/* Is this a TSPLIB or OR-LIBRARY file? */
	if (strncmp (buf, "NAME", 4) EQ 0) {

		/* This appears to be a TSPLIB file */

		/* Search for dimension field */
		found = FALSE;
		while (NOT feof (stdin)) {
			scanf ("%s", buf);
			if (strncmp (buf, "DIMENSION:", 10) EQ 0) {
				found = TRUE;
				break;
			}

			if (strncmp (buf, "DIMENSION", 9) EQ 0) {
				scanf ("%s", buf); /* read the : */
				found = TRUE;
				break;
			}
		}

		if (NOT found) {
			fprintf (stderr,"\nError: Cannot find dimension.\n\n");
			exit (1);
		}

		scanf ("%s", buf);
		n = atoi (buf);
		if (n < 1) {
			fprintf (stderr,"\nError: Bad dimension.\n\n");
			exit (1);
		}

		/* Search for one of the valid coordinate types */
		found = FALSE;
		while (NOT feof (stdin)) {
			scanf ("%s", buf);
			if ((strncmp (buf, "EUC_2D",  6) EQ 0) OR
			    (strncmp (buf, "MAX_2D",  6) EQ 0) OR
			    (strncmp (buf, "CEIL_2D", 7) EQ 0) OR
			    (strncmp (buf, "GEO",     3) EQ 0) OR
			    (strncmp (buf, "ATT",     3) EQ 0)) {
				found = TRUE;
				break;
			}
		}

		if (NOT found) {
			fprintf (stderr, "\nError: TSPLIB input file is not\n"
					 "a 2-D plane instance.\n\n");
			exit (1);
		}

		/* Search for the coordinate section */
		found = FALSE;
		while (NOT feof (stdin)) {
			scanf ("%s", buf);
			if (strncmp (buf, "NODE_COORD_SECTION", 18) EQ 0) {
				found = TRUE;
				break;
			}
		}

		if (NOT found) {
			fprintf (stderr,
				 "\nError: Cannot find node coordinates.\n\n");
			exit (1);
		}

		/* Now start dumping out the coordinates */
		for (i = 1; i <= n; i++) {
			scanf ("%s", buf);

			if (i NE atoi (buf)) {
				fprintf (stderr,
					 "\nError: Bad node number.\n\n");
				exit (1);
			}

			scanf ("%s", buf); /* x-coordinate */
			printf ("%s ", buf);
			scanf ("%s", buf); /* y-coordinate */
			printf ("%s\n", buf);
		}
	}
	else {
		num_instances = atoi (buf);

		if (num_instances < 1) {
			fprintf (stderr,
				 "\nError: Input file is neither OR-LIBRARY\n"
				 "or TSPLIB file.\n\n");
			exit (1);
		}

		if (orlib_instance > num_instances) {
			fprintf (stderr,
				 "\nError: Specified instance number (%d)\n"
				 "does not exist in OR-LIBRARY file.\n\n",
				 orlib_instance);
			exit (1);
		}

		/* Now dump out the correct instance */
		for (j = 1; j <= orlib_instance; j++) {
			scanf ("%s", buf);
			n = atoi (buf);

			if (n < 1) {
				fprintf (stderr,
					 "\nError: Bad number of points in instance %d.\n\n",
					 j);
				exit (1);
			}

			if (j EQ orlib_instance) {
				for (i = 1; i <= n; i++) {
					scanf ("%s", buf);
					printf ("%s ", buf);
					scanf ("%s", buf);
					printf ("%s\n", buf);
				}
			}
			else {
				for (i = 1; i <= n; i++) {
					scanf ("%s%s", buf, buf);
				}
			}
		}
	}

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

	--argc;
	me = *argv++;
	while (argc > 0) {
		ap = *argv++;
		if (*ap NE '-') {
			orlib_instance = atoi (ap);
			if (orlib_instance < 1) {
				usage ();
				break;
			}
		}
		else {
			usage ();
			break;
		}
		--argc;
	}
}

/*
 * This routine prints out the proper usage and exits.
 */

static char *	arg_doc [] = {
	"",
	"\tN\tRead instance number N from OR-LIBRARY file.",
	"\t\t(default: 1).",
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
			"\nUsage: %s [N]\n",
			me);
	pp = &arg_doc [0];
	while ((p = *pp++) NE NULL) {
		(void) fprintf (stderr, "%s\n", p);
	}
	exit (1);
}
