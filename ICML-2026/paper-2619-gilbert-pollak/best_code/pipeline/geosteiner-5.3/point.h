/***********************************************************************

	$Id: point.h,v 1.2 2022/11/19 13:45:54 warme Exp $

	File:	point.h
	Rev:	e-1
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Points and Point Sets.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from steiner.h.

************************************************************************/

#ifndef	POINT_H
#define	POINT_H

#include "geomtypes.h"

/*
 * The following object represents a single point.
 */

struct point {
	coord_t		x;
	coord_t		y;
};


/*
 * The following represents an input set of points, with the count.
 * We always allocate these dynamically with an appropriate number
 * of points in the "a" array.
 */

struct pset {
	int		n;
	struct point	a [1];
};


/*
 * Useful macros.
 */

#define DELTAX(p1,p2)	(fabs((p1) -> x - (p2) -> x))
#define DELTAY(p1,p2)	(fabs((p1) -> y - (p2) -> y))

#define RDIST(p1,p2)	(DELTAX ((p1), (p2)) + DELTAY ((p1), (p2)))
#define EDIST(p1,p2)	(hypot (DELTAX ((p1), (p2)), DELTAY ((p1), (p2))))

#define NULL_PSET	((struct pset *) 0)
#define PSET_SIZE(n)	(offsetof (struct pset, a [n]))
#define NEW_PSET(n)	((struct pset *) _gst_new (PSET_SIZE (n)))
#define COPY_PSET(dp, sp)	(void) memcpy ((dp), (sp), PSET_SIZE ((sp) -> n))
#define ZERO_PSET(dp, n)	(void) memset ((dp), 0, PSET_SIZE (n))

#endif
