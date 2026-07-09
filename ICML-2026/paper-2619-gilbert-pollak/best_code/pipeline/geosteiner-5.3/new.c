/***********************************************************************

	$Id: new.c,v 1.4 2022/11/19 13:45:53 warme Exp $

	File:	new.c
	Rev:	e-2
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Various utility routines.

************************************************************************

	Modification Log:

	e-1:	09/05/2016	warme
		: Split off from utils.c.
	e-2:	09/24/2016	warme
		: Apply prefixes.

************************************************************************/

#include "memory.h"

#include "logic.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>


/*
 * This routine performs all dynamic memory allocation for the program.
 * We test for out of memory condition here.
 */

#ifndef WATCH_MEMORY

	void *
_gst_new (

size_t		size		/* IN - size of chunk in bytes. */
)
{
void *		p;

	if (size EQ 0) {
		/* Avoid implementation-defined bahavior of malloc! */
		size = 1;
	}

	p = malloc (size);
	if (p EQ NULL) {
		(void) fprintf (stderr, "Out of memory!\n");
		exit (1);
	}

	return (p);
}

#endif
