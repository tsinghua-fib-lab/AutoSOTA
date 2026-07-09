/***********************************************************************

	$Id: fatal.c,v 1.4 2022/11/19 13:45:51 warme Exp $

	File:	fatal.c
	Rev:	e-2
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Stuff to report fatal software errors.

************************************************************************

	Modification Log:

	e-1:	09/05/2016	warme
		: Split off from utils.c.
		: Support FATAL_ERROR and FATAL_ERROR_IF().
	e-2:	09/24/2016	warme
		: Apply prefixes.
		: Remove support for obsolete FATAL() macro.

************************************************************************/

#include "fatal.h"

#include "logic.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>


/*
 * Global Routines
 */

void			_gst_fatal (const char *	filename,
				    int			lineno);


/*
 * This routine displays a fatal message and then dies!
 */

	void
_gst_fatal (

const char *	file,		/* IN - the source file calling us. */
int		lineno		/* IN - line number calling us. */
)
{
	fflush (stdout);

	fprintf (stderr,
		 "\nFatal software error at %s, line %d\n",
		 file, lineno);
	fflush (stderr);
	abort ();
}
