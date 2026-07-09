/***********************************************************************

	$Id: metric.c,v 1.20 2022/11/19 13:45:53 warme Exp $

	File:	metric.c
	Rev:	e-4
	Date:	09/30/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	General metric functions.

************************************************************************

	Modification Log:

	a-1:	08/05/2002	benny
		: Created.  Derived from efst.c and UniSteiner.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix issues in gst_copy_metric and centralize
		:  the clearing out of a metric object.
		: Move two inline functions here from .h file.
		: Fix -Wall issues.  Upgrade fatals.
	e-4:	09/30/2016	warme
		: Removed unused variable.

************************************************************************/

#include "metric.h"

#include "efuncs.h"
#include "fatal.h"
#include <float.h>
#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include "prepostlude.h"
#include "steiner.h"
#include <string.h>

/*
 * Global functions
 */

gst_metric_ptr		gst_create_metric (int, int, int *);
int			gst_copy_metric (gst_metric_ptr, gst_metric_ptr);
double			gst_distance (gst_metric_ptr, double, double, double, double);
int			gst_free_metric (gst_metric_ptr);

bool			_gst_ray_intersection (const struct point *,
					       const struct point *,
					       const struct point *,
					       dist_t,
					       struct point *);
/*
 * Local functions
 */

static void	clear_metric (gst_metric_ptr	metric);
static void	ray_intersection_always (const struct point *,
				  	 const struct point *,
				  	 const struct point *,
				  	 struct point *);

/*
 * Intersection between two rays: The first ray is always assumed to have 
 * its source in the origin. The orientations of the two rays are 
 * given by vectors.
 * Adapted from the segment intersection code.
 * The intersection is only calculated if 'is != NULL' (makes it possible
 * to avoid a division.)
 */

	bool
_gst_ray_intersection (

const struct point *	vec1,	/* IN - vector describing ray from origin */
const struct point *	src2,	/* IN - starting point of second ray */
const struct point *	vec2,	/* IN - vector describing ray from second ray */
dist_t			epsilon,
struct point *		is	/* OUT - intersection point (NULL is allowed) */
)
{
double		alpha, beta, alpha1, alpha2, beta1, beta2;
double		A, B, C, D, E, F;
double		denom, denom1, denom2;
double		eps, m;

	A = vec1 -> x;
	B = - vec2 -> x;
	C = src2 -> x; 
	D = vec1 -> y;
	E = - vec2 -> y;
	F = src2 -> y;

	/* Compute alpha, beta and denominator. Use factors to determine */
        /* a conservative epsilon to be used in comparisons.             */
	/* We use a variant of Mehlhorn and Nahers floating point        */
	/* filter to compute an appropriate epsilon.                     */

	denom1 = B * D;
	denom2 = A * E;
	denom  = denom1 - denom2;
	
	eps = 10.0 * epsilon; /* 10 is an approximation of the socalled index */

	m = fabs(denom1) + fabs(denom2);
	if (m > 1.0) eps *= m;

	alpha1 = B * F;
	alpha2 = C * E;
	alpha  = alpha1 - alpha2;

	beta1  = C * D;
	beta2  = A * F;
	beta   = beta1 - beta2;

	if (((denom > 0.0) AND (alpha < 0.0)) OR
	    ((denom < 0.0) AND (alpha > 0.0))) {
		m = fabs(alpha1) + fabs(alpha2);
		if (m > 1.0) eps *= m;
	}
	else {
		m = fabs(beta1) + fabs(beta2);
		if (m > 1.0) eps *= m;
	}

	if (denom NE 0.0) {
		/* Easy case -- lines are NOT parallel. */
		if (denom > 0.0) {
			if (alpha < 0.0) {
				/* First ray does not hit second rays line */
				if ( alpha + eps < 0.0 ) {
					return (FALSE);
				}
				else {
					/* alpha = 0.0; */
					if (is NE NULL) {
						is -> x = 0.0;
						is -> y = 0.0;
					}
					return (TRUE);
				}
			}

			if (beta < 0.0) {
				/* Second ray does not hit first rays line */
				if (beta + eps < 0.0 ) {
					return (FALSE);
				}
				else {
					/* beta = 0.0; */
					if (is NE NULL) {
						is -> x = src2 -> x;
						is -> y = src2 -> y;
					}
					return (TRUE);
				}
			}
		}
		else { /* denom < 0.0 */
			if (alpha > 0.0) {
				/* First ray does not hit second rays line */
				if (alpha > eps) {
					return (FALSE);
				}
				else {
					/* alpha = 0.0; */
					if (is NE NULL) {
						is -> x = 0.0;
						is -> y = 0.0;
					}
					return (TRUE);
				}
			}

			if (beta > 0.0) { /* beta > 0 */
				/* Second ray does not hit first rays line */
				if (beta > eps) {
					return (FALSE);
				}
				else {
					/* beta = 0.0; */
					if (is NE NULL) {
						is -> x = src2 -> x;
						is -> y = src2 -> y;
					}
					return (TRUE);
				}
			}
		}

		/* We have a single point of intersection! */

		if (is NE NULL) {
			alpha /= denom;
			is -> x = alpha * vec1 -> x;
			is -> y = alpha * vec1 -> y;
		}
		return (TRUE);
	}

	/* The two rays are parallel.
	   Report no (unique) intersection point. */

	return (FALSE);
}

/*
 * Intersection between two rays: The first ray is always assumed to have 
 * its source in the origin. The orientations of the two rays are 
 * given by vectors.
 * Adapted from the segment intersection code.
 * This version of the procedure ALWAYS assumes that an intersection
 * exists. Hence no epsilon is needed in comparisons.
 */

	static
	void
ray_intersection_always (

const struct point *	vec1,	/* IN - vector describing ray from origin */
const struct point *	src2,	/* IN - starting point of second ray */
const struct point *	vec2,	/* IN - vector describing ray from second ray */
struct point *		is	/* OUT - intersection point (NULL is allowed) */
)
{
double		alpha, beta;
double		A, B, C, D, E, F;
double		denom;

	A = vec1 -> x;
	B = - vec2 -> x;
	C = src2 -> x; 
	D = vec1 -> y;
	E = - vec2 -> y;
	F = src2 -> y;

	denom = B * D - A * E;
	if (denom NE 0.0) {
		/* Easy case -- lines are NOT parallel. */
		alpha = B * F - C * E;
		if (denom > 0) {
			if (alpha < 0.0) {
				/* First ray does not hit second rays line */
				/* Force alpha = 0. */
				is -> x = 0.0;
				is -> y = 0.0;
				return;
			}

			beta = C * D - A * F;
			if (beta < 0.0) {
				/* Second ray does not hit first rays line */
				/* Force beta = 0. */
				is -> x = src2 -> x;
				is -> y = src2 -> y;
				return;
			}
		}
		else { /* denom < 0 */
			if (alpha > 0.0) {
				/* First ray does not hit second rays line */
				/* Force alpha = 0. */ 
				is -> x = 0.0;
				is -> y = 0.0;
				return;
			}

			beta = C * D - A * F;
			if (beta > 0.0) { /* beta > 0 */
				/* Second ray does not hit first rays line */
				/* Force beta = 0. */
				is -> x = src2 -> x;
				is -> y = src2 -> y;
				return;
			}
		}

		/* We have a single point of intersection! */

		alpha /= denom;
		is -> x = alpha * vec1 -> x;
		is -> y = alpha * vec1 -> y;
	}
}

/*
 * Creation of a metric. Actual work is only done if it is a uniform orientation
 * metric.
 */

	gst_metric_ptr
gst_create_metric (

int		type,		/* IN - metric type (NONE, L, UNIFORM, ...) */
int		parameter,	/* IN - metric parameter */
int *		status		/* OUT - status */
)
{
int			i;
int			K;
int			lambda;
int			quadrant;
struct point *		dirs;
struct point *		left_dirs;
struct point *		right_dirs;
double *		tangents;
gst_metric_ptr		m;

	GST_PRELUDE

	m = NEW (struct gst_metric);
	memset (m, 0, sizeof (struct gst_metric));

	switch (type) {
	case GST_METRIC_L:
		FATAL_ERROR_IF ((parameter NE 1) AND (parameter NE 2));
		break;
	case GST_METRIC_UNIFORM:
		FATAL_ERROR_IF (parameter < 2);
		break;
	case GST_METRIC_NONE:
		break;
	default:
		FATAL_ERROR;
		break;
	}

	m -> type	= type;
	m -> parameter	= parameter;

	if (type EQ GST_METRIC_UNIFORM) {
		/* Some basic values */
		lambda	= m -> lambda	= parameter;
		K	= m -> K	= 2*lambda;
		m -> a120 = K/3;
		quadrant = m -> quadrant = (int)ceil(lambda/2.0);

		/* Various allocations */
		dirs	   = m -> dirs	     = NEWA (4*lambda, struct point);
		left_dirs  = m -> left_dirs  = NEWA (4*lambda, struct point);
		right_dirs = m -> right_dirs = NEWA (4*lambda, struct point);
		tangents   = m -> tangents   = NEWA (quadrant + 1, double);

		/* Calculate table of orientations */
		switch(lambda) {
		case 2:
			dirs[0].x =  1; dirs[0].y =  0;
			dirs[1].x =  0; dirs[1].y =  1;
			dirs[2].x = -1; dirs[2].y =  0;
			dirs[3].x =  0; dirs[3].y = -1;
			break;
		case 4:
			dirs[0].x =  1; dirs[0].y =  0;
			dirs[1].x =  1; dirs[1].y =  1;
			dirs[2].x =  0; dirs[2].y =  1;
			dirs[3].x = -1; dirs[3].y =  1;
			dirs[4].x = -1; dirs[4].y =  0;
			dirs[5].x = -1; dirs[5].y = -1;
			dirs[6].x =  0; dirs[6].y = -1;
			dirs[7].x =  1; dirs[7].y = -1;
			break;
		default:
			for (i = 0; i < lambda; i++) {
				dirs[i].x
				 = cos( ((double) i / (double) lambda) * PI);
				dirs[i].y
				 = sin( ((double) i / (double) lambda) * PI);
				dirs[i+lambda].x = -dirs[i].x;
				dirs[i+lambda].y = -dirs[i].y;
			}
			break;
		}

		for (i = 0; i < K; i++) {
			right_dirs[i].x = dirs[i].x + dirs[i].y/(1<<20);
			right_dirs[i].y = dirs[i].y - dirs[i].x/(1<<20);
			left_dirs[i].x	= dirs[i].x - dirs[i].y/(1<<20);
			left_dirs[i].y	= dirs[i].y + dirs[i].x/(1<<20);
		}

		for (i = K; i < 2*K; i++) {
			dirs[i].x = dirs[i - K].x;
			dirs[i].y = dirs[i - K].y;
			left_dirs[i].x = left_dirs[i - K].x;
			left_dirs[i].y = left_dirs[i - K].y;
			right_dirs[i].x = right_dirs[i - K].x;
			right_dirs[i].y = right_dirs[i - K].y;
		}

		/* Calculate table of tangents for fast calculation of
		   fixed orientations distances */
		for (i = 0; i < quadrant; i++)
			tangents[i] = tan(i*PI/lambda);
		tangents[quadrant] = DBL_MAX;

		/* Setup minimum and maximum angle for orientations */
		m -> min_angle = (int) ceil (K/3.0 - 1);
		m -> max_angle = (int) floor(K/3.0 + 1);
	}

	if (status NE NULL) {
		*status = 0;
	}

	GST_POSTLUDE
	return (m);
}

/*
 * Copy metric (inefficient implementation which simple creates a new metric,
 * and copies the content to the existing metric).
 */

	int
gst_copy_metric (

gst_metric_ptr		dest,		/* IN/OUT - destination metric */
gst_metric_ptr		src		/* IN - original metric */
)
{
int			status;
int			type;
int			parameter;
gst_metric_ptr		metric;

	GST_PRELUDE

	FATAL_ERROR_IF (dest EQ NULL);

	if (src EQ NULL) {
		type		= GST_METRIC_NONE;
		parameter	= 0;
	}
	else {
		type		= src -> type;
		parameter	= src -> parameter;
	}

	clear_metric (dest);

	metric = gst_create_metric (type, parameter, &status);

	if (status EQ 0) {
		memcpy (dest, metric, sizeof (struct gst_metric));
	}
	free (metric);

	GST_POSTLUDE
	return status;
}

/*
 * Free any space allocated for a metric.
 */

	int
gst_free_metric (

gst_metric_ptr		metric		/* IN - metric to be freed */
)
{
	GST_PRELUDE

	clear_metric (metric);

	free (metric);

	GST_POSTLUDE
	return 0;
}

/*
 * Free up any memory for the given metric, and force to "NONE" type.
 */

	static
	void
clear_metric (

gst_metric_ptr		metric		/* IN: metric to clear out */
)
{
	switch (metric -> type) {
	case GST_METRIC_NONE:
		break;

	case GST_METRIC_L:
		break;

	case GST_METRIC_UNIFORM:
		if (metric -> dirs NE NULL) {
			free (metric -> dirs);
		}
		if (metric -> left_dirs NE NULL) {
			free (metric -> left_dirs);
		}
		if (metric -> right_dirs NE NULL) {
			free (metric -> right_dirs);
		}
		if (metric -> tangents NE NULL) {
			free (metric -> tangents);
		}
		break;

	default:
		FATAL_ERROR;
	}

	metric -> type		= GST_METRIC_NONE;
	metric -> parameter	= 0;
}

/*
 * This function finds the distance between two points in the given lambda
 * metric. It first translates the problem to the origin AND the first
 * quadrant. It then finds the needed directions by using a pre-calculated
 * array of tangents. Finally it uses ray_intersection () to find the
 * corner point of the edge. The corner point is used to calculate
 * Euclidean distances.
 */

#define SQRT2_MINUS1 0.41421356237309514547462185873882845044

	double
gst_distance (

gst_metric_ptr		metric,		/* IN - metric */
double			px,		/* IN - x-coordinate of first point */
double			py,		/* IN - y-coordinate of first point */
double			qx,		/* IN - x-coordinate of second point */
double			qy		/* IN - y-coordinate of second point */
)
{
int		pdir;
int		qdir;
int		type;
int		parameter;
double		dx;
double		dy;
double		dxpow;
double		dypow;
struct point	temp;
struct point	is;

	type = metric -> type;
	parameter = metric -> parameter;

	/* Translate to the origin and move to first quadrant */
	dx = fabs(qx - px);
	dy = fabs(qy - py);

	switch (type) {
	case GST_METRIC_NONE:
		return (0.0);

	case GST_METRIC_UNIFORM:

		if (parameter EQ 2) {		/* Rectilinear metric */
			return dx + dy;
		}
		if (parameter EQ 4) {		/* Octilinear metric */
			return (MAX (dx, dy) + (SQRT2_MINUS1) * MIN (dx, dy));
		}
		break;

	case GST_METRIC_L:
		if (parameter EQ 1) {		/* Rectilinear metric */
			return dx + dy;
		}
		if (parameter EQ 2) {		/* Euclidean metric */
			return hypot (dx, dy);
		}
		/* General L_p metric. */
		if (dx <= 0) return (dy);
		if (dy <= 0) return (dx);
		dxpow = exp (parameter * log (dx));
		dypow = exp (parameter * log (dy));
		return (exp (log (dxpow + dypow) / parameter));
	}

	/* Find the needed directions */
	pdir = 1;
	if (dx EQ 0) {
		pdir = metric -> quadrant;
	}
	else if (dy NE 0) {
		double *tangents = metric -> tangents;
		int quadrant = metric -> quadrant;
		while (pdir < quadrant) {
			if (dy <= dx*tangents[pdir])
				break;
			pdir++;
		}
	}
	qdir = pdir - 1 + metric -> lambda;

	/* Now find the intersection point */
	temp.x = dx;  temp.y = dy;
	is.x   = 0.0; is.y   = 0.0;
	ray_intersection_always (&metric->dirs[pdir],
				 &temp,   
				 &metric->dirs[qdir],
				 &is);

	/* Calculate and return the distance */
	return (sqrt(is.x*is.x + is.y*is.y) + sqrt(sqr_dist(&temp, &is)));
}

/*
 *
 */

	int
gst_get_metric_info (

gst_metric_ptr	metric,		/* IN - metric */
int *		type,		/* OUT - metric type */
int *		p		/* OUT - parameter */
)
{
int	res;

	GST_PRELUDE

	res = 0;

	do {		/* Used only for "break"... */
		if (metric EQ NULL) {
			res = GST_ERR_INVALID_METRIC;
			break;
		}

		if (type NE NULL) {
			*type = metric -> type;
		}

		if (p NE NULL) {
			*p = metric -> parameter;
		}
	} while (FALSE);

	GST_POSTLUDE

	return res;
}

/*
 * Return TRUE if-and-only-if the given hypergraph has rectilinear metric.
 */

	bool
_gst_is_rectilinear (

struct gst_hypergraph *	H
)
{
int			m, p;
struct gst_metric *	metric;

	metric	= H -> metric;
	m	= metric -> type;
	p	= metric -> parameter;

	return (((m EQ GST_METRIC_L      ) AND (p EQ 1)) OR
		((m EQ GST_METRIC_UNIFORM) AND (p EQ 2)));
}

/*
 * Return TRUE if-and-only-if the given hypergraph has Euclidean metric.
 */

	bool
_gst_is_euclidean (

struct gst_hypergraph *	H
)
{
int			m, p;
struct gst_metric *	metric;

	metric	= H -> metric;
	m	= metric -> type;
	p	= metric -> parameter;

	return ((m EQ GST_METRIC_L) AND (p EQ 2));
}
