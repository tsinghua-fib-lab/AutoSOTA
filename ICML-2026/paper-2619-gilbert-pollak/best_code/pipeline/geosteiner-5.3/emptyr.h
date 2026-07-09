/***********************************************************************

	$Id: emptyr.h,v 1.8 2022/11/19 13:45:51 warme Exp $

	File:	emptyr.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1998, 2022 by David M. Warme and Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Routines for efficiently determining whether or not two
	terminals define an empty rectangle.  We precompute this
	information and store it compactly.

************************************************************************

	Modification Log:

	a-1:	09/28/98	warme
		: Created.  Implemented Zachariasen's algorithm
		:  using Warme's infrastructure.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef	EMPTYR_H
#define	EMPTYR_H

#include "bitmaskmacros.h"
#include "gsttypes.h"

struct pset;


/*
 * Global Routines
 */

extern int		_gst_count_empty_rectangles (bitmap_t * bits, int n);
extern bitmap_t *	_gst_init_empty_rectangles (struct pset *	pts,
						    int *		succ0);
extern bool		_gst_is_empty_rectangle (bitmap_t *	bits,
						 int		i,
						 int		j);
extern void		_gst_shutdown_empty_rectangles (bitmap_t * bits);

#endif
