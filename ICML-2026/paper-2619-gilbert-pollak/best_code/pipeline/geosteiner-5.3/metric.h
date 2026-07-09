/***********************************************************************

	$Id: metric.h,v 1.14 2022/11/19 13:45:53 warme Exp $

	File:	metric.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	General metric functions.

************************************************************************

	Modification Log:

	a-1:	08/05/2002	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Move two inline functions into .c file.

************************************************************************/

#ifndef METRIC_H
#define METRIC_H

#include "geomtypes.h"
#include "gsttypes.h"

struct gst_hypergraph;
struct point;

/*
 * Structure to contain metric related information.
 */

struct gst_metric {
	int		type;		/* L-metric, uniform, none... */
	int		parameter;	/* Eg. lambda value */

	/* Information related to lambda metrics only */
	int		lambda;		/* Number of allowed orientations */
	int		K;		/* 2*Lambda */
	int		quadrant;	/* No. of dirs. in first quadrant */
	int		min_angle;	/* Min/max angles as measured */
	int		max_angle;	/* in lambda/pi units */
	int		a120;		/* Largest legal angle <= 120 */

	struct point *	dirs;		/* Allowed directions as vectors */
	struct point *	left_dirs;	/* Slightly 'rotated' directions */
	struct point *	right_dirs;	/* vectors compared to 'dirs' */
	double *	tangents;	/* Tangent values for lambda/2 */
					/* direction vectors */
};

/*
 * Metric functions.
 */

extern bool	_gst_is_euclidean (struct gst_hypergraph * H);
extern bool	_gst_is_rectilinear (struct gst_hypergraph * H);
extern bool	_gst_ray_intersection (const struct point *,
				       const struct point *,
				       const struct point *,
				       dist_t,
				       struct point *);
#endif
