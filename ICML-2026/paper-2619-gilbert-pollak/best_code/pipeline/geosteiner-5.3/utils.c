/***********************************************************************

	$Id: utils.c,v 1.22 2022/11/19 13:45:55 warme Exp $

	File:	utils.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Various utility routines.

************************************************************************

	Modification Log:

	a-1:	04/18/93	warme
		: Created.  Collected these routines into this file
		:  from other places.
	b-1:	02/28/2001	warme
		: Changes for 3.1 release.
		: Added tracef and struct tracef_control.
		: Added routines for Intel floating point precision fix.
		: Added _gst_strdup and store_double.
	c-1:	08/05/2002	benny
		: Some changes for library release.
		: Moved some initialization functions to environment.c.
		: Removed tracef and struct tracef_control.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
		: Split fatal() and new() into separate files.
	e-3:	09/24/2016	warme
		: Reorganized include files.
		: Fix -Wall issues.
		: Split all floating-point stuff off into fputils.[ch].

************************************************************************/

#include "utils.h"

#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include <string.h>


/*
 * Global Routines
 */

void		_gst_print_mask (gst_channel_ptr	chan,
				 char *			msg,
				 bitmap_t *		bp1,
				 int			n);
char *		_gst_strdup (const char * s);


/*
 * External References
 */

	/* none */

/*
 * This routine prints out a given subset.
 */

	void
_gst_print_mask (

gst_channel_ptr chan,		/* IN - output channel */
char *		msg,		/* IN - header message to display */
bitmap_t *	bp1,		/* IN - subset to display */
int		n		/* IN - number of bits in mask */
)
{
int		i;
int		count;

	(void) gst_channel_printf (chan, "%s", msg);
	count = 0;
	for (i = 0; i < n; i++) {
		if (BITON (bp1, i)) {
			if (++count > 20) {
				gst_channel_printf (chan, "\n%s	", msg);
				count = 0;
			}
			gst_channel_printf (chan, " %u", (int32u) i);
		}
	}
	gst_channel_printf (chan, "\n");
}

/*
 * This routine is our own version of the "strdup" function, which
 * is not available everywhere.  This version also uses "new" instead
 * of "malloc".
 */

	char *
_gst_strdup (

const char *	s
)
{
size_t		n;
char *		p;

	if (s EQ NULL) return (NULL);

	n = strlen (s) + 1;

	p = NEWA (n, char);

	strcpy (p, s);

	return (p);
}
