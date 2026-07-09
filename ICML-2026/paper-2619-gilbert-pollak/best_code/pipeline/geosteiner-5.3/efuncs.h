/***********************************************************************

	$Id: efuncs.h,v 1.10 2022/11/19 13:45:51 warme Exp $

	File:	efuncs.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2001, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Basic plane geometry functions used by Euclidean FST generator
	and pruning algorithms.
	This file is merely intended for inlining these speed-critical
	functions in various c-files.

************************************************************************

	Modification Log:

	a-1:	01/22/00	martinz
		: Created.  Split off from efst.c
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files.
		: Make these functions static inline to fix -Wall.

************************************************************************/

#ifndef EFUNCS_H
#define EFUNCS_H

#include "geomtypes.h"
#include "gsttypes.h"
#include "logic.h"
#include <math.h>
#include "point.h"


/*
 * Local Constants
 */

#define PI		(3.1415926535897932384626433832795028841971693)
#define RAD120		(2.0*PI/3.0)
#define SQRT3		(1.73205080756887729352)
#define HALF_SQRT3	(0.86602540378443864676)


/*
 * Local Macros
 */

#undef MIN
#define MIN(a,b)	(((a) < (b)) ? (a) : (b))

#undef MAX
#define MAX(a,b)	(((a) > (b)) ? (a) : (b))


/*
 * Does the sequence p1, p2, p3 make a right turn?
 */

	static
	inline
	bool
right_turn (

struct point *	p1,	/* IN - first point */
struct point *	p2,	/* IN - second point */
struct point *	p3	/* IN - third point */
)
{
	return (  (p1 -> x - p2 -> x) * (p1 -> y - p3 -> y) <
		  (p1 -> y - p2 -> y) * (p1 -> x - p3 -> x) );
}


/*
 * Does the sequence p1, p2, p3 make a left turn?
 */

	static
	inline
	bool
left_turn (

struct point *	p1,	/* IN - first point */
struct point *	p2,	/* IN - second point */
struct point *	p3	/* IN - third point */
)
{
	return (  (p1 -> x - p2 -> x) * (p1 -> y - p3 -> y) >
		  (p1 -> y - p2 -> y) * (p1 -> x - p3 -> x) );
}


/*
 * Square of distance between points p1 and p2
 */

	static
	inline
	dist_t
sqr_dist (

struct point *	p1,	/* IN - first point */
struct point *	p2	/* IN - second point */
)
{
double	dx, dy;

	dx = p1 -> x - p2 -> x;
	dy = p1 -> y - p2 -> y;

	return (dx*dx + dy*dy);
}


/*
 * Rotate vector 60 degrees counter-clockwise
 */

	static
	inline
	void
rotate60 (

double		dx,	/* IN - x-coordinate of vector */
double		dy,	/* IN - y-coordinate of vector */
double *	rdx,	/* OUT - x-coordinate of rotated vector */
double *	rdy	/* OUT - y-coordinate of rotated vector */
)
{
	*rdx = 0.5*dx - HALF_SQRT3*dy;
	*rdy = 0.5*dy + HALF_SQRT3*dx;
}

/*
 * Find equilateral point for points p1 and p2
 */

	static
	inline
	void
eq_point (

struct point *	p1,	/* IN - first point */
struct point *	p2,	/* IN - second point */
struct point *	e	/* OUT - equilateral point */
)
{
double	dx, dy;

	dx = p1 -> x - p2 -> x;
	dy = p1 -> y - p2 -> y;

	e -> x = p2 -> x + 0.5*dx - HALF_SQRT3*dy;
	e -> y = p2 -> y + 0.5*dy + HALF_SQRT3*dx;
}


/*
 * Find equilateral point for points p1 and p2 (alternative version)
 */

#if 0

	static
	inline
	void
eq_point (

struct point *	p1,	/* IN - first point */
struct point *	p2,	/* IN - second point */
struct point *	e	/* OUT - equilateral point */
)
{
	e -> x = 0.5*(p1 -> x + p2 -> x) - HALF_SQRT3*(p1 -> y - p2 -> y);
	e -> y = 0.5*(p1 -> y + p2 -> y) + HALF_SQRT3*(p1 -> x - p2 -> x);
}

#endif


/*
 * Find center of equilateral circle
 */

	static
	inline
	void
eq_circle_center (

struct point *	p1,	/* IN - first point on circle */
struct point *	p2,	/* IN - second second on circle */
struct point *	e,	/* IN - third point on circle */
struct point *	c	/* OUT - center of circle */
)
{
	c -> x = (p1 -> x + p2 -> x + e -> x) / 3.0;
	c -> y = (p1 -> y + p2 -> y + e -> y) / 3.0;
}


/*
 * Compute vector corresponding to angle between points a, b and c
 */

	static
	inline
	void
get_angle_vector (

struct point *	a,	/* IN - first point */
struct point *	b,	/* IN - second point */
struct point *	c,	/* IN - third point */
double *	dx,	/* OUT - x-coordinate of vector */
double *	dy	/* OUT - y-coordinate of vector */
)
{
double	abx, aby, cbx, cby;

	abx = a -> x - b -> x;
	aby = a -> y - b -> y;
	cbx = c -> x - b -> x;
	cby = c -> y - b -> y;

	*dx = abx*cbx + aby*cby;
	*dy = abx*cby - cbx*aby;
}


/*
 * Rotate point P around C at an angle v whose sin(v)
 * and cos(v) are given.
 */

	static
	inline
	void
rotate (

struct point *	P,	/* IN - point to be rotated */
struct point *	C,	/* IN - center of rotation */
double		sinv,	/* IN - sin(v) of angle */
double		cosv,	/* IN - cos(v) of angle */
struct point *	PN	/* OUT - rotated point */
)
{
double	ox, oy, dx, dy;

	ox	   = C -> x;
	oy	   = C -> y;
	dx	   = P -> x  -	ox;
	dy	   = P -> y  -	oy;
	PN -> x	   = ox + dx*cosv - dy*sinv;
	PN -> y	   = oy + dx*sinv + dy*cosv;
}


/*
 * Rotate point P around C by an angle 2v, where
 * sin(v) and cos(v) are given.
 */

	static
	inline
	void
rotate2 (

struct point *	P,	/* IN - point to be rotaed */
struct point *	C,	/* IN - center of rotation */
double		sinv,	/* IN - sin(v) of angle */
double		cosv,	/* IN - cos(v) of angle */
struct point *	PN	/* OUT - rotated point */
)
{
double	cosv2, sin2v, cos2v, ox, oy, dx, dy;

	if (cosv EQ 2.0) { /* cosv not defined */

	  	if (fabs(sinv) > 1.0) {
			/* this can only happen if we have numerical problems... */
			sinv = 0.0; 
		}
		cosv2 = 1.0 - sinv*sinv;
		cosv  = sqrt(cosv2);
	}
	else {
		cosv2 = cosv*cosv;
	}

	sin2v	   = 2.0*sinv*cosv;
	cos2v	   = 2.0*cosv2 - 1.0;
	ox	   = C -> x;
	oy	   = C -> y;
	dx	   = P -> x  -	ox;
	dy	   = P -> y  -	oy;
	PN -> x	   = ox + dx*cos2v - dy*sin2v;
	PN -> y	   = oy + dx*sin2v + dy*cos2v;
}


/*
 * Given a point E on a circle with center C the procedure returns
 * the projection EN of E onto the same circle in direction ER.
 */

	static
	inline
	void
project_point (

struct point *	E,	/* IN - point to be projected */
struct point *	C,	/* IN - center of circle */
struct point *	R,	/* IN - direction of projection */
struct point *	EN	/* OUT - projected point */
)
{
double	Ax, Ay, Bx, By, AdotB, BdotB, lambda;

	Ax = C -> x  -	E -> x;
	Ay = C -> y  -	E -> y;
	Bx = R -> x  -	E -> x;
	By = R -> y  -	E -> y;
	AdotB = Ax * Bx + Ay * By;
	BdotB = Bx * Bx + By * By;
	if (BdotB EQ 0.0) {
		*EN = *E;
	}
	else {
		lambda = AdotB / BdotB;
		lambda += lambda; /* multiply by 2 */
		EN -> x	   =  E -> x + lambda * Bx;
		EN -> y	   =  E -> y + lambda * By;
	}
}


/*
 * Given a point E on a circle with center C the procedure returns
 * the projection of E onto the same circle in a direction perpendicular
 * to CR.
 */

	static
	inline
	void
project_point_perp (

struct point *	E,	/* IN - point to be projected */
struct point *	C,	/* IN - center of circle */
struct point *	R,	/* IN - perpendicular direction of projection */
struct point *	EN	/* OUT - projected point */
)
{
double	Ax, Ay, Bx, By, AdotB, BdotB, lambda;

	Ax = C -> x  -	E -> x;
	Ay = C -> y  -	E -> y;
	Bx = C -> y  -	R -> y;
	By = R -> x  -	C -> x;
	AdotB = Ax * Bx + Ay * By;
	BdotB = Bx * Bx + By * By;

	if (BdotB EQ 0.0) {
		*EN = *E;
	}
	else {
		lambda = AdotB / BdotB;
		lambda += lambda; /* multiply by 2 */
		EN -> x	   = E -> x + lambda * Bx;
		EN -> y	   = E -> y + lambda * By;
	}
}


/*
 * Test if vector corresponds to an angle greater than 120 degrees.
 * Special fast version which avoids using trigonometric functions.
 */

	static
	inline
	bool
angle_geq_120 (

double		dx,	/* IN - x-coordinate of vector */
double		dy	/* IN - y-coordinate of vector */
)
{
	if  (dy	 < 0.0)						return TRUE;
	if ((dy EQ 0.0) AND (dx <= 0.0))			return TRUE;
	if ((dy	 > 0.0) AND (dx <  0.0) AND ( dy <= -SQRT3*dx)) return TRUE;
	return FALSE;
}


/*
 * Test if vector corresponds to an angle greater than 60 degrees.
 * Special fast version which avoids using trigonometric functions.
 */

	static
	inline
	bool
angle_geq_60 (

double		dx,	/* IN - x-coordinate of vector */
double		dy	/* IN - y-coordinate of vector */
)
{
	if ((dx	 < 0.0) OR (dy <  0.0))		return TRUE;
	if ((dy EQ 0.0) AND (dx <= 0.0))	return TRUE;
	if ((dy	 > 0.0) AND (dy >= SQRT3*dx))	return TRUE;
	return FALSE;
}


/*
 * Test if the points p1,p2,p3 make up a wedge
 * that is greater than 120 degrees
 */

	static
	inline
	bool
wedge120 (

struct point *		p1,	/* IN - first point */
struct point *		p2,	/* IN - second point */
struct point *		p3	/* IN - third point */
)
{
struct point	e;
struct point	c;

	if (left_turn (p1,p3,p2)) {
		eq_point (p1, p3, &e);
		eq_circle_center (p1, p3, &e, &c);
	}
	else {
		eq_point (p3, p1, &e);
		eq_circle_center (p3, p1, &e, &c);
	}

	if (sqr_dist (&c, p2) <= sqr_dist (&c, p1)) {
		/* p2 is inside circle -> angle greater than 120 deg */
		return (TRUE);
	}

	/* less than 120 deg */

	return (FALSE);
}


/*
 * Intersection between two line segments pq and rs.
 * Adapted from Dave's more generic intersection code.
 */

	static
	inline
	bool
segment_intersection (

struct point *		p,	/* IN - first endpoint of first segment */
struct point *		q,	/* IN - second endpoint of first segment */
struct point *		r,	/* IN - first endpoint of second segment */
struct point *		s,	/* IN - second endpoint of second segment */
struct point *		is	/* OUT - intersection point */
)
{
	/* This routine takes two line segments PQ and RS and determines
	   their intersection (if any).	 It makes use of a coordinate
	   transformation to reduce the number of special cases to a
	   minimum.  Line segment PQ runs from point P to point Q, and
	   line segment RS runs from point R to point S.  We create a
	   parametric form for line segment PQ as:

		XPQ(alpha) = (1 - alpha) * P.x + alpha * Q.x
		YPQ(alpha) = (1 - alpha) * P.y + alpha * Q.y

	   As alpha varies from 0.0 to 1.0, the parametric point follows
	   line segment PQ from P to Q.	 Similarly, we parameterize line
	   segment RS as a function of beta:

		XRS(beta) = (1 - beta) * R.x + beta * S.x
		YRS(beta) = (1 - beta) * R.y + beta * S.y

	   The (infinite) lines containing segments PQ and RS intersect when:

		[XPQ(alpha) = XRS(beta)] AND [YPQ(alpha) = YRS(beta)]

	   Note that if (0 <= alpha <= 1) AND (0 <= beta <= 1) then the
	   line segments actually intersect, not just the lines.  This is
	   the really nifty property of this coordinate transformation!

	   Expanding out the simultaneous equations gives:

		(1 - alpha) * P.x + alpha * Q.x = (1 - beta) * R.x + beta * S.x
		(1 - alpha) * P.y + alpha * Q.y = (1 - beta) * R.y + beta * S.y

	   or:

		(Q.x - P.x) * alpha + (R.x - S.x) * beta = (R.x - P.x)
		(Q.y - P.y) * alpha + (R.y - S.y) * beta = (R.y - P.y)

	   which we write as:

		A * alpha + B * beta = C
		D * alpha + E * beta = F

	   Unless this system is singular, its solutions are:

			 B*F - C*E		C*D - A*F
		alpha = -----------	beta = -----------
			 B*D - A*E		B*D - A*E

	   The system is singular if either line segment is zero-length, or
	   if the lines have the same slope (parallel or colinear).  These
	   cases are handled separately.
	*/

double		alpha, beta;
double		A, B, C, D, E, F;
double		denom;
struct point *	rp;
struct point *	sp;
struct point *	tmp;

	A = q -> x - p -> x;
	B = r -> x - s -> x;
	C = r -> x - p -> x;
	D = q -> y - p -> y;
	E = r -> y - s -> y;
	F = r -> y - p -> y;

	rp = r;
	sp = s;

	denom = B * D - A * E;

	if (denom < 0) {
		/* switch points R and S... */
		tmp = rp;
		rp = sp;
		sp = tmp;

		denom = - denom;
		B = - B;
		C = rp -> x - p -> x;
		E = - E;
		F = rp -> y - p -> y;
	}

	if (denom NE 0.0) {
		/* Easy case -- lines are NOT parallel. */
		alpha = (B * F - C * E);
		if ((alpha < 0.0) OR (denom < alpha)) {
			/* First segment does not reach to second */
			/* line segment. */
			return (FALSE);
		}

		beta = C * D - A * F;
		if ((beta < 0.0) OR (denom < beta)) {
			/* Second line segment does not reach to first */
			/* line segment. */
			return (FALSE);
		}

		/* We have a single point of intersection! */

		alpha /= denom;
		is -> x = (1.0 - alpha) * p -> x + alpha * q -> x;
		is -> y = (1.0 - alpha) * p -> y + alpha * q -> y;
		return (TRUE);
	}

	/* The two line segments are parallel.
	   Report no (unique) intersection point. */

	return (FALSE);
}

#endif
