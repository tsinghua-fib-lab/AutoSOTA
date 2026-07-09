/***********************************************************************

	$Id: dt.c,v 1.6 2022/11/19 13:45:51 warme Exp $

	File:	dt.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2005, 2022 by David M. warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines to compute Delaunay Triangulations

************************************************************************

	Modification Log:

	a-1:	09/17/2005	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.  Upgrade fatals.
		: Must define REAL in order to USE_TRIANGLE.
		: Fix some GMP-specific code.

************************************************************************/

#include "dt.h"

#include "config.h"
#include "fatal.h"
#include <float.h>
#include "fstfuncs.h"
#include "genps.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "point.h"
#include "sortfuncs.h"
#include <stdlib.h>
#include "steiner.h"
#include <string.h>

#ifdef USE_TRIANGLE
 #define REAL	double
 #include "triangle.h"
#endif

#ifdef HAVE_GMP
 #include <gmp.h>
#endif

#define	TEST_DRIVER	0


/*
 * Global Routines
 */

void		_gst_delaunay_triangulation (
				struct pset *	pts,
				int *		numberofedges,
				int **		edgelist,
				int *		numberoftriangles,
				int **		trianglelist,
				int **		neighborlist);

/*
 * Local Types
 */

#ifndef USE_TRIANGLE

struct E {				/* A directed edge */
	int		src;		/* Source node */
	int		dst;		/* Destination node */
	struct E *	vnext;		/* Next edge CCW around src vertex */
	struct E *	vprev;		/* Prev edge CCW around src vertex */
	struct E *	rev;		/* Oppositely directed edge */
	struct E *	snext;		/* Next edge on stack */
	int		flags;		/* Various flags */
	int		face;		/* Face number of edge */
};

/* Flags */

#define	XF		0x01		/* Edge is part of exterior face */
#define ONX		0x02		/* Edge is on exterior of triangulation */
#define INSTACK		0x04		/* Edge is on the stack */
#define MARK		0x08		/* Edge has been traversed */

struct DTinfo {
	struct pset *	pts;		/* The input point set */
	int		num_edges;	/* Number of edges */
	int		num_tri;	/* Number of triangles */
	struct E *	edges;		/* Array of edge structures */
	struct E **	vfirst;		/* First edge for each vertex */
};

typedef struct E *	Eptr;

#ifdef HAVE_GMP
 /* A point structure that uses GMP arbitrary precision integers */
 /* in a "fixed point" mode. */

 struct gmp_point {
	mpz_t	x;
	mpz_t	y;
 };
#endif


/*
 * Local Routines
 */

static bool	bends_left (struct pset * pts, int p1, int p2, int p3);
static bool	bends_right (struct pset * pts, int p1, int p2, int p3);
static int	bend_primitive (struct pset * pts, int p1, int p2, int p3);
static int	circle_test (struct pset * pts, int p1, int p2, int p3, int p4);
static void	clean_up (struct DTinfo * dt);
static void	delaunay_flip (struct DTinfo * dt);
static void	triangulate (struct DTinfo * dt);

#ifdef HAVE_GMP
 static int	bend_primitive_gmp (struct pset * pts, int p1, int p2, int p3);
 static int	circle_test_gmp (struct pset * pts, int p1, int p2, int p3, int p4);
 static void	convert_into_gmp_fixed_point (struct point **	  dbl_pts,
					      struct gmp_point ** gmp_pts,
					      int		  n);
#endif

#if TEST_DRIVER
 #define PLOT(dt,title)	plot_triangulation (dt, title)
 static void	plot_triangulation (struct DTinfo * dt, const char * title);
 static gst_channel_ptr	chan;
#else
 #define PLOT(dt,title)
#endif

#endif	/* NOT USE_TRIANGLE */

/*
 * Construction of Delaunay triangulation (DT) using "Triangle"
 */

#ifdef USE_TRIANGLE

	void
_gst_delaunay_triangulation (

struct pset *	pts,			/* IN - point set to triangulate */
int *		numberofedges,		/* OUT - number of edges in DT */
int **		edgelist,		/* OUT - list of edge indices */
int *		numberoftriangles, 	/* OUT - number of triangles in DT */
int **		trianglelist,		/* OUT - list of triangle edges */
int **		neighborlist		/* OUT - neighbouring triangles */
)
{
struct triangulateio	in, out, vorout;

	/* Transfer to triangulate data structure and call "Triangle" */

	in.numberofpoints 		= pts -> n;
	/* This is a bit of a kludge, but it works fine as long	*/
	/* as we don't add any new members to 'struct point'.	*/
	in.pointlist 	  		= &(pts -> a [0].x);
	in.numberofpointattributes	= 0;
	in.pointattributelist		= NULL;
	in.pointmarkerlist		= NULL;
	in.numberofsegments		= 0;
	in.numberofholes		= 0;
	in.numberofregions		= 0;
	in.regionlist			= NULL;

	out.pointlist			= NULL;
	out.edgelist			= NULL;
	out.trianglelist		= NULL;
	out.neighborlist		= NULL;

	if (edgelist EQ NULL) {
		/* We only need information about triangles */

		triangulate ("znNBQ", &in, &out, &vorout);

		free (out.pointlist);
		free (out.edgelist);
	
		*numberofedges 		= out.numberofedges;
		*numberoftriangles	= out.numberoftriangles;
		*trianglelist 		= out.trianglelist;
		*neighborlist 		= out.neighborlist;
	} 
	else {
		/* We only need information about edges */

		triangulate ("zeNBQ", &in, &out, &vorout);

		free (out.pointlist);
		free (out.trianglelist);
		free (out.neighborlist);

		*numberofedges 		= out.numberofedges;
		*edgelist		= out.edgelist;
	}
}

#endif

/*
 * Construct a (correct) Delaunay triangulation of the given point set.
 *
 * NOTE: THIS CODE ASSUMES THE INPUT CONTAINS NO DUPLICATE POINTS!!!
 *
 * This version is our own version, not as "souped up" as Shewchuk's
 * "triangle" package.  We use a simple two-phase algorithm:
 *
 *	1. Compute a valid planar triangulation,
 *	2. Perform edge flips until it becomes Delaunay.
 *
 * This algorithm is not asymptotically fast, but it is relatively
 * simple, and achieves good average-case performance.
 *
 * We use two "geometric primitives":
 *	a. Do 3 points A, B, C bend left, right, or are they collinear?
 *	b. Is a point D inside, outside or on the circle inscribed through
 *	   points A, B and C (which are specified in CCW order).
 *
 * We implement these primitives using floating point, but with a "filter"
 * that determines when the results should not be trusted.  Whenever this
 * happens, we switch over to using GMP arbitrary precision arithmetic.
 *
 * The parameters pts, and numberofedges must always be non-NULL.
 * Regarding the remaining parameters, there are two valid ways to call
 * this routine:
 *
 * (edgelist EQ NULL) AND (remaining parameters are all non-NULL)
 *
 *	OR
 *
 * (edgelist NE NULL) AND (remaining parameters are all NULL)
 */

#ifndef USE_TRIANGLE

	void
_gst_delaunay_triangulation (

struct pset *	pts,			/* IN - point set to triangulate */
int *		numberofedges,		/* OUT - number of edges in DT */
int **		edgelist,		/* OUT - list of edge indices */
int *		numberoftriangles, 	/* OUT - number of triangles in DT */
int **		trianglelist,		/* OUT - list of triangle edges */
int **		neighborlist		/* OUT - neighbouring triangles */
)
{
int		k;
int		npts;
int		nedges;
int		ntri;
int		tindex;
int		face;
Eptr		e1, e2, e3;
int *		ip1;
int *		ip2;
struct DTinfo	dtinfo;

	npts = pts -> n;

	*numberofedges = 0;
	if (edgelist NE NULL) {
		*edgelist = NULL;
	}
	else {
		*numberoftriangles	= 0;
		*trianglelist		= NULL;
		*neighborlist		= NULL;
	}
	if (npts <= 1) {
		return;
	}
	if (npts EQ 2) {
		*numberofedges	= 1;
		if (edgelist NE NULL) {
			ip1 = NEWA (2, int);
			ip1 [0] = 0;
			ip1 [1] = 1;
			*edgelist = ip1;
		}
		return;
	}

	/* Initialize global data. */
	memset (&dtinfo, 0, sizeof (dtinfo));
	dtinfo.pts = pts;

	triangulate (&dtinfo);
	PLOT (&dtinfo, "Initial Triangulation");

	/* Now perform Delaunay flips until we are fully Delaunay. */

	delaunay_flip (&dtinfo);
	PLOT (&dtinfo, "Delaunay Triangulation");

	nedges = dtinfo.num_edges;
	*numberofedges = nedges / 2;
	if (edgelist NE NULL) {
		/* Retrieving the list of edges. */
		ip1 = NEWA (nedges, int);
		e1 = dtinfo.edges;
		e2 = e1 + nedges;
		*edgelist = ip1;
		for (; e1 < e2; e1 += 2) {
			*ip1++ = e1 -> src;
			*ip1++ = e1 -> dst;
		}
	}
	else {
		/* Retrieving the triangles and neighbors. */
		ntri = dtinfo.num_tri;
		*numberoftriangles = ntri;

		/* Traverse each edge and identify all the faces. */
		e1 = dtinfo.edges;
		e3 = e1 + nedges;
		tindex = 0;
		for (; e1 < e3; e1++) {
			if ((e1 -> flags & MARK) NE 0) continue;
			face = -1;
			if ((e1 -> flags & XF) EQ 0) {
				face = tindex++;
			}
			e2 = e1;
			k = 0;
			do {
				e2 -> face = face;
				e2 -> flags |= MARK;
				/* e2 = next_face_edge (e2); */
				e2 = e2 -> rev -> vprev;
				++k;
			} while (e2 NE e1);
			if ((face >= 0) AND (k NE 3)) {
				/* All faces except the exterior must */
				/* be triangles! */
				FATAL_ERROR;
			}
		}
		if (tindex NE ntri) {
			/* Found wrong number of triangles! */
			FATAL_ERROR;
		}

		/* Allocate buffers */
		ip1 = NEWA (3 * ntri, int);
		ip2 = NEWA (3 * ntri, int);
		*trianglelist = ip1;
		*neighborlist = ip2;

		/* Traverse each edge again, writing the triangles	*/
		/* and neighbors.					*/
		e1 = dtinfo.edges;
		tindex = 0;
		for (; e1 < e3; e1++) {
			if ((e1 -> flags & MARK) EQ 0) continue;
			if (e1 -> face < 0) continue;
			face = tindex++;
			e2 = e1;
			k = 0;
			do {
				*ip1++ = e2 -> src;
				*ip2++ = e2 -> rev -> face;
				e2 -> flags &= ~MARK;
				/* e2 = next_face_edge (e2); */
				e2 = e2 -> rev -> vprev;
				++k;
			} while (e2 NE e1);
			if (k NE 3) {
				/* All interior faces must be triangles! */
				FATAL_ERROR;
			}
		}
		if (tindex NE ntri) {
			/* Wrong number of triangles! */
			FATAL_ERROR;
		}
	}

	clean_up (&dtinfo);
}

/*
 * Construct an initial triangulation using a simple sweep-line algorithm.
 */

	static
	void
triangulate (

struct DTinfo *		dt	/* IN/OUT - global Delaunay triang data */
)
{
int		i;
int		j;
int		n;
int		idx;
int		p;
int		prev_p;
int		max_edges;
int		ntri;
Eptr		e1, e2, e3, e4, e5, e6;
struct pset *	pts;
struct E *	edges;
struct E **	vfirst;
int *		order;

	pts = dt -> pts;
	n   = pts -> n;

	/* Compute the maximum number of edges and triangles we can have. */
	/* This routine is only called when N >= 3.			  */
	max_edges	= 3 * n - 6;
	max_edges	= 2 * max_edges;	/* edges are bi-directed */

	edges  = NEWA (max_edges, struct E);
	vfirst = NEWA (n, struct E *);

	dt -> edges	= edges;
	dt -> vfirst	= vfirst;

	/* Initialize each vertex to be incident to an empty list of edges. */
	for (i = 0; i < n; i++) {
		vfirst [i]	= NULL;
	}

	/* Use a sweep-line algorithm to triangulate the point set. */
	order = _gst_heapsort_x (pts);

	/* Find the first two points (skip any duplicates). */
	i = order [0];
	idx = 1;
	for (;;) {
		j = order [idx++];
		if (pts -> a [i].x NE pts -> a [j].x) break;
		if (pts -> a [i].y NE pts -> a [j].y) break;
		if (idx >= n) {
			/* All input points are identical! */
			dt -> num_edges	= 0;
			dt -> num_tri	= 0;
			return;
		}
	}

	/* Initialize first two points with a single edge between them. */
		 
	e1 = edges++;
	e2 = edges++;

	e1 -> src	= i;
	e1 -> dst	= j;
	e1 -> vnext	= e1;
	e1 -> vprev	= e1;
	e1 -> rev	= e2;
	e1 -> flags	= 0;

	e2 -> src	= j;
	e2 -> dst	= i;
	e2 -> vnext	= e2;
	e2 -> vprev	= e2;
	e2 -> rev	= e1;
	e2 -> flags	= 0;

	vfirst [i]	= e1;
	vfirst [j]	= e2;

	ntri = 0;

	prev_p = j;

	while (idx < n) {
		p = order [idx++];
		if ((pts -> a [p].x EQ pts -> a [prev_p].x) AND
		    (pts -> a [p].y EQ pts -> a [prev_p].y)) {
			/* Skip duplicate points. */
			continue;
		}

		/* Starting from last vertex, traverse up to non-left bend. */
		e1 = vfirst [prev_p] -> vprev;
		prev_p = p;
		do {
			/* e1 = prev_face_edge (e1); */
			e1 = e1 -> vnext -> rev;
		} while (bends_left (pts, p, e1 -> src, e1 -> dst));

		/* Traverse down to non-left bend, triangulating as we go. */
		do {
			/* e2 = next_face_edge (e1); */
			e2 = e1 -> rev -> vprev;
			j = e2 -> src;
			/* Create a new edge (2 opposing edges) from p	*/
			/* to j.  This may or may not produce a new	*/
			/* triangle (we don't really care right now).	*/

			e3 = edges++;
			e4 = edges++;

			e3 -> src	= p;
			e3 -> dst	= j;
			e3 -> rev	= e4;
			e3 -> flags	= 0;

			e4 -> src	= j;
			e4 -> dst	= p;
			e4 -> rev	= e3;
			e4 -> flags	= 0;

			if (vfirst [p] EQ NULL) {
				/* First edge to/from p... */
				vfirst [p]	= e3;
				e3 -> vnext	= e3;
				e3 -> vprev	= e3;
				e5 = e1 -> rev;
				e4 -> vnext	= e5;
				e4 -> vprev	= e2;
				e2 -> vnext	= e4;
				e5 -> vprev	= e4;
			}
			else {
				e6 = vfirst [p];
				e5 = e6 -> vprev;
				e3 -> vnext	= e6;
				e3 -> vprev	= e5;
				e6 -> vprev	= e3;
				e5 -> vnext	= e3;

				e5 = e1 -> rev;
				e4 -> vnext	= e5;
				e4 -> vprev	= e2;
				e2 -> vnext	= e4;
				e5 -> vprev	= e4;

				++ntri;
			}

			e1 = e2;
		} while (bends_left (pts, p, e1 -> src, e1 -> dst));
	}

	FATAL_ERROR_IF (edges > &(dt -> edges [max_edges]));

	dt -> num_edges = edges - dt -> edges;
	dt -> num_tri	= ntri;

	free (order);

	/* Mark the edges of the exterior face.  The last edge added to	*/
	/* the last vertex is one of the edges on this face.  Mark the	*/
	/* opposing edges as being "on the exterior".			*/
	e1 = vfirst [prev_p] -> vprev;
	e2 = e1;
	do {
		e2 -> flags |= (XF | ONX);
		e2 -> rev -> flags |= ONX;
		/* e2 = next_face_edge (e2); */
		e2 = e2 -> rev -> vprev;
	} while (e2 NE e1);
}

/*
 * Perform Delaunay flips on edges until the triangulation is fully Delaunay.
 */

	static
	void
delaunay_flip (

struct DTinfo *		dt	/* IN/OUT - global Delaunay triang data */
)
{
int		i;
int		nedges;
int		p1, p2, p3, p4;
Eptr		e1, e2, e3, e4, e5, e6, e7, e8;
Eptr		stack;
struct E **	vfirst;
struct pset *	pts;

#define CONDITIONAL_PUSH(e)					\
	if (((e) -> flags & INSTACK) EQ 0) {			\
		(e) -> snext = stack;				\
		stack = (e);					\
		(e) -> flags |= INSTACK;			\
	}

	pts	= dt -> pts;
	vfirst	= dt -> vfirst;

	nedges = dt -> num_edges;

	/* Make a stack that initially contains all edges that are	*/
	/* NOT on the exterior boundary of the triangulation.		*/
	stack = NULL;
	e1 = dt -> edges;
	for (i = 0; i < nedges; i++, e1++) {
		if ((e1 -> flags & ONX) NE 0) continue;
		e1 -> snext = stack;
		stack = e1;
		e1 -> flags |= INSTACK;
	}

	while (stack NE NULL) {
		e1 = stack;
		stack = stack -> snext;
		FATAL_ERROR_IF ((e1 -> flags & INSTACK) EQ 0);
		e1 -> flags &= ~INSTACK;
		e2 = e1 -> rev;
		/* e3 = next_face_edge (e1); */
		e3 = e1 -> rev -> vprev;
		/* e5 = next_face_edge (e2); */
		e5 = e2 -> rev -> vprev;

		p1 = e3 -> src;
		p2 = e3 -> dst;
		p3 = e5 -> src;
		p4 = e5 -> dst;

		if (NOT (bends_left (pts, p2, p4, p1) AND
			 bends_right (pts, p2, p4, p3))) {
			/* This is not a well-formed quadrilateral... */
			continue;
		}

		i = circle_test (pts, p1, p2, p3, p4);

		if (i <= 0) {
			/* This diagonal fine -- don't flip it. */
			continue;
		}

		/* e4 = next_face_edge (e3); */
		e4 = e3 -> rev -> vprev;
		/* e6 = next_face_edge (e5); */
		e6 = e5 -> rev -> vprev;

		CONDITIONAL_PUSH (e3);
		CONDITIONAL_PUSH (e4);
		CONDITIONAL_PUSH (e5);
		CONDITIONAL_PUSH (e6);

		/* Unthread edge e1 */
		e7 = e1 -> vprev;
		e8 = e1 -> vnext;
		e7 -> vnext = e8;
		e8 -> vprev = e7;
		if (vfirst [p3] EQ e1) {
			vfirst [p3] = e8;
		}

		/* Unthread edge e2 */
		e7 = e2 -> vprev;
		e8 = e2 -> vnext;
		e7 -> vnext = e8;
		e8 -> vprev = e7;
		if (vfirst [p1] EQ e2) {
			vfirst [p1] = e8;
		}

		/* Edge e1 was p3->p1.  Make it be p2->p4 by inserting	*/
		/* it into p2's list between e4 and e3->rev.		*/
		e1 -> src	= p2;
		e1 -> dst	= p4;
		e7 = e3 -> rev;
		e1 -> vprev	= e4;
		e1 -> vnext	= e7;
		e4 -> vnext	= e1;
		e7 -> vprev	= e1;

		/* Edge e2 was p1->p3.  Make it be p4->p2 by inserting	*/
		/* it into p4's list between e6 and e5->rev.		*/
		e2 -> src	= p4;
		e2 -> dst	= p2;
		e7 = e5 -> rev;
		e2 -> vprev	= e6;
		e2 -> vnext	= e7;
		e6 -> vnext	= e2;
		e7 -> vprev	= e2;
	}

#undef CONDITIONAL_PUSH
}

/*
 * Free up all of the memory we might have allocated.
 */

	static
	void
clean_up (

struct DTinfo *		dt	/* IN/OUT - global Delaunay triang data */
)
{
	if (dt -> edges NE NULL) {
		free (dt -> edges);
	}
	if (dt -> vfirst NE NULL) {
		free (dt -> vfirst);
	}
}

/*
 * Return TRUE if-and-only-if the point sequence p1, p2, p3 bends to
 * the left.
 */

	static
	bool
bends_left (

struct pset *		pts,	/* IN - the point set */
int			p1,	/* IN - index of point p1 */
int			p2,	/* IN - index of point p2 */
int			p3	/* IN - index of point p3 */
)
{
	return (bend_primitive (pts, p1, p2, p3) > 0);
}


/*
 * Return TRUE if-and-only-if the point sequence p1, p2, p3 bends to
 * the right.
 */

	static
	bool
bends_right (

struct pset *		pts,	/* IN - the point set */
int			p1,	/* IN - index of point p1 */
int			p2,	/* IN - index of point p2 */
int			p3	/* IN - index of point p3 */
)
{
	return (bend_primitive (pts, p1, p2, p3) < 0);
}

/*
 * Return +1 if-and-only-if p1, p2, p3 bends to the left.
 * Return -1 if-and-only-if p1, p2, p3 bends to the right.
 * Return  0 if-and-only-if p1, p2, p3 are collinear.
 */

	static
	int
bend_primitive (

struct pset *		pts,	/* IN - the point set */
int			p1,	/* IN - index of point p1 */
int			p2,	/* IN - index of point p2 */
int			p3	/* IN - index of point p3 */
)
{
int			result;
struct point *		pp1;
struct point *		pp2;
struct point *		pp3;
double			ax, ay, bx, by;
double			prod1, prod2, Z;

#ifdef HAVE_GMP
	/* We hope the compiler does this computation at compile time	*/
	/* and puts the result into memory that is read-only at run	*/
	/* time...							*/
	/* This is the tolerance that Shewchuk provides in his robust	*/
	/* geometric primitives paper.  Note that he defines epsilon to	*/
	/* be DBL_EPS/2.  We use the "ANSI-C" epsilon, just to be a	*/
	/* "bit" more careful.  :^>					*/
	static const double	tolerance = (3.0 + 16.0 * DBL_EPSILON) * DBL_EPSILON;
#endif

	pp1 = &(pts -> a [p1]);
	pp2 = &(pts -> a [p2]);
	pp3 = &(pts -> a [p3]);

	/* These subtractions always yield the correct sign, at least... */
	ax = pp2 -> x - pp1 -> x;
	ay = pp2 -> y - pp1 -> y;
	bx = pp3 -> x - pp1 -> x;
	by = pp3 -> y - pp1 -> y;

	prod1 = ax*by;
	prod2 = ay*bx;
	Z = prod1 - prod2;

	/* Get tentative result. */
	if (Z > 0.0) {
		result = +1;
	}
	else if (Z < 0.0) {
		result = -1;
	}
	else {
		result = 0;
	}

#if HAVE_GMP
	{ double norm, bound;
		if (prod1 > 0.0) {
			if (prod2 <= 0.0) {
				return (result);	/* tentative result OK. */
			}
			norm = prod1 + prod2;
		}
		else if (prod1 < 0.0) {
			if (prod2 >= 0.0) {
				return (result);	/* tentative result OK. */
			}
			norm = -prod1 - prod2;
		}
		else {
			return (result);	/* tentative result OK. */
		}

		/* When we get here, cancellation is occurring in the final	*/
		/* subtraction.  Need to check numeric tolerances...		*/
		bound = norm * tolerance;

		if (Z >  bound) return (+1);
		if (Z < -bound) return (-1);
	}
	result = bend_primitive_gmp (pts, p1, p2, p3);
#endif

	return (result);
}

/*
 * Exact "bend primitive" using GMP arithmetic.
 */

#ifdef HAVE_GMP

	static
	int
bend_primitive_gmp (

struct pset *		pts,	/* IN - the point set */
int			p1,	/* IN - index of point p1 */
int			p2,	/* IN - index of point p2 */
int			p3	/* IN - index of point p3 */
)
{
int			result;
struct gmp_point	A, B, C;
struct point *		dbl_pts [3];
struct gmp_point *	gmp_pts [3];

	dbl_pts [0] = &(pts -> a [p1]);
	dbl_pts [1] = &(pts -> a [p2]);
	dbl_pts [2] = &(pts -> a [p3]);

	gmp_pts [0] = &A;
	gmp_pts [1] = &B;
	gmp_pts [2] = &C;

	convert_into_gmp_fixed_point (&dbl_pts [0], &gmp_pts [0], 3);

	/* At this point, we are working with exact integers. */

	mpz_sub (B.x, B.x, A.x);
	mpz_sub (B.y, B.y, A.y);
	mpz_sub (C.x, C.x, A.x);
	mpz_sub (C.y, C.y, A.y);

	mpz_mul (A.x, B.x, C.y);
	mpz_mul (A.y, B.y, C.x);
	mpz_sub (A.x, A.x, A.y);

	result = mpz_sgn (A.x);

	mpz_clear (C.y);	mpz_clear (C.x);
	mpz_clear (B.y);	mpz_clear (B.x);
	mpz_clear (A.y);	mpz_clear (A.x);

	return (result);
}

#endif

/*
 * Consider the *directed* circle through points p1, p2 and p3.
 *
 * Return +1 if-and-only-if p4 lies to the *left* of this directed circle.
 * Return -1 if-and-only-if p4 lies to the *right* of this directed circle.
 * Return  0 if-and-only-if p4 lies *on* the circle.
 *
 * If the caller specifies p1, p2, and p3 in counter-clockwise order,
 * then +1 means p4 is inside the circle, -1 means outside, and 0 means
 * "on" the circle.
 *
 * Derivation of this expression:
 *
 * Let points p1=(ax,ay), p2=(bx,by), p3=(cx,cy), p4=(dx,dy).
 * Compute line L1 to be the perpendicular bisector of p1 and p2:
 *
 *				(ax^2 + ay^2) - (bx^2+by^2)
 *	(by-ay)*y + (bx-ax)*x + --------------------------- = 0
 *					      2
 *
 * Compute line L2 to be the perpendicular bisector of p2 and p3:
 *
 *				(Bx^2 + by^2) - (cx^2+cy^2)
 *	(cy-by)*y + (cx-bx)*x + --------------------------- = 0
 *					      2
 *
 * We arbitrarily translate the problem instance so that p1 is at the
 * origin (substitute ax = 0 and ay = 0).  Let point O by the intersection
 * of lines L1 and L2.  Let points B = p2-p1, C = p3-p1, D = p4-p1 be the
 * translated versions of p2, p3 and p4, respectively.
 *
 * Let Z = (D - O) dot (D - O) - (B - O) dot (B - 0).
 * Doing the algebra we discover that:
 *
 *     (B dot B)*(D cross C) + (C dot C)*(B cross D) + (D dot D)*(C cross B)
 * Z = ---------------------------------------------------------------------
 *				  (B cross C)
 *
 * where "dot" is the 2-dimensional dot-product, and "cross" is the "Z"
 * component of the vector cross product (where the Z component of the
 * argument vectors is assumed to be zero).
 *
 * We note that if points p1, p2 and p3 are specified in counter-clockwise
 * order, then the denominator (B cross C) will always be positive, and it
 * suffices to consider only the sign of the numerator.  Alternatively, we
 * can consider the circle to be "directed" from p1 to p2 to p3, in which
 * case Z is positive, zero, or negative depending upon whether point p4 is
 * to the "left", on, or to the "right" of the directed circle through
 * p1, p2 and p3.
 *
 * In our application, p1, p2 and p3 are always specified in counter-
 * clockwise order.
 */

	static
	int
circle_test (

struct pset *		pts,	/* IN - the point set */
int			p1,	/* IN - index of point p1 */
int			p2,	/* IN - index of point p2 */
int			p3,	/* IN - index of point p3 */
int			p4	/* IN - index of point p4 */
)
{
struct point *		pp1;
struct point *		pp2;
struct point *		pp3;
struct point *		pp4;
double			ax, bx, cx, dx;
double			ay, by, cy, dy;
double			bmag2, cmag2, dmag2;
double			DxC, BxD, CxB;
double			Z;

#ifdef HAVE_GMP
	/* We hope the compiler does this computation at compile time	*/
	/* and puts the result into memory that is read-only at run	*/
	/* time...							*/
	static double	tolerance1 = (19.0 + 159.0 * DBL_EPSILON) * DBL_EPSILON;
	static double	tolerance2 = (22.0 + 223.0 * DBL_EPSILON) * DBL_EPSILON;
#endif

	pp1 = &(pts -> a [p1]);
	pp2 = &(pts -> a [p2]);
	pp3 = &(pts -> a [p3]);
	pp4 = &(pts -> a [p4]);

	ax = pp1 -> x;
	ay = pp1 -> y;

	bx = pp2 -> x - ax;
	by = pp2 -> y - ay;

	cx = pp3 -> x - ax;
	cy = pp3 -> y - ay;

	dx = pp4 -> x - ax;
	dy = pp4 -> y - ay;

	bmag2 = bx*bx + by*by;
	cmag2 = cx*cx + cy*cy;
	dmag2 = dx*dx + dy*dy;

	DxC = dx*cy - dy*cx;
	BxD = bx*dy - by*dx;
	CxB = cx*by - cy*bx;

	Z = bmag2 * DxC + cmag2 * BxD + dmag2 * CxB;

#ifdef HAVE_GMP
	/* If we assume the following relative error for primitive ops:	*/
	/*								*/
	/*		A+B		2*eps				*/
	/*		A-B		2*eps				*/
	/*		A+B+C		4*eps				*/
	/*		A*B		eps				*/
	/*		sqrt()		eps				*/
	/*								*/
	/* then the total relative error of the above formula for Z is:	*/
	/*								*/
	/* R =	256*eps^10 + 1600*eps^9 + 4416*eps^8 + 7088*eps^7	*/
	/*	+ 7328*eps^6 + 5100*eps^5 + 2420*eps^4 + 773*eps^3	*/
	/*	+159*eps^2 + 19*eps					*/
	/*								*/

	/* We start with a coarse bound -- it is not a very tight	*/
	/* bound, but it does not contain any square-roots, so it is	*/
	/* pretty fast to compute, and it screens out the vast majority	*/
	/* of cases with little effort.  This bound is essentially an	*/
	/* "L-infinity" bound that computes the maximum magnitude of	*/
	/* the result if no cancellation occurs.			*/

	ax = fabs (ax);
	ay = fabs (ay);

	bx = fabs (pp2 -> x) + ax;
	by = fabs (pp2 -> y) + ay;

	cx = fabs (pp3 -> x) + ax;
	cy = fabs (pp3 -> y) + ay;

	dx = fabs (pp4 -> x) + ax;
	dy = fabs (pp4 -> y) + ay;

	DxC = dx*cy + dy*cx;
	BxD = bx*dy + by*dx;
	CxB = cx*by + cy*bx;

	{ double bound, Zmax, dmag, cmag, bmag;
	  int result;

		Zmax = bmag2 * DxC + cmag2 * BxD + dmag2 * CxB;
		Zmax += (Zmax * tolerance1);

		bound = Zmax * tolerance1;

		if (Z >  bound) return (+1);
		if (Z < -bound) return (-1);

		/* The result is too close to call using the first numeric	*/
		/* bound, so we now apply a second bound that is virtually	*/
		/* always much tighter -- but it requires 3 square roots, so it	*/
		/* is more costly to compute.  This bound derives from the	*/
		/* observation that the formula for Z contains "cross products"	*/
		/* -- for which we have:					*/
		/*								*/
		/*	|A x B| = |A|*|B|*|sin(theta)| <= |A|*|B|.		*/
		/*								*/
		/* The exact relative error of this bound is:			*/
		/*								*/
		/* R = (1+eps)^6 * (1+2*eps)^8 - 1				*/
		/*   =	256*eps^14 + 2560*eps^13 + 11776*eps^12 + 33024*eps^11	*/
		/*	+ 63072*eps^10 + 86784*eps^9 + 88720*eps^8		*/
		/*	+ 68464*eps^7 + 40081*eps^6 + 17718*eps^5 + 5823*eps^4	*/
		/*	+ 1380*eps^3 + 223*eps^2 + 22*eps.			*/

		bmag = sqrt (bmag2);
		cmag = sqrt (cmag2);
		dmag = sqrt (dmag2);

		Zmax =	  bmag2 * cmag  * dmag
			+ bmag  * cmag2 * dmag
			+ bmag  * cmag  * dmag2;
		Zmax += (Zmax * tolerance2);

		bound = Zmax * tolerance2;

		if (Z >  bound) return (+1);
		if (Z < -bound) return (-1);

		/* Both floating point filters failed -- use GMP to get it right. */

		result = circle_test_gmp (pts, p1, p2, p3, p4);

		return (result);
	}
#else
	/* We do not have GMP.  Use raw floating point result. */
	if (Z > 0.0) return (+1);
	if (Z < 0.0) return (-1);

	return (0);
#endif
}

/*
 * Exact "circle test primitive" using GMP arithmetic.
 */

#ifdef HAVE_GMP

	static
	int
circle_test_gmp (

struct pset *		pts,	/* IN - the point set */
int			p1,	/* IN - index of point p1 */
int			p2,	/* IN - index of point p2 */
int			p3,	/* IN - index of point p3 */
int			p4	/* IN - index of point p4 */
)
{
int			result;
struct gmp_point	A, B, C, D;
mpz_t			bmag2, cmag2, dmag2;
mpz_t			DxC, BxD, CxB;
mpz_t			Z;
struct point *		dbl_pts [4];
struct gmp_point *	gmp_pts [4];

	dbl_pts [0] = &(pts -> a [p1]);
	dbl_pts [1] = &(pts -> a [p2]);
	dbl_pts [2] = &(pts -> a [p3]);
	dbl_pts [3] = &(pts -> a [p4]);

	gmp_pts [0] = &A;
	gmp_pts [1] = &B;
	gmp_pts [2] = &C;
	gmp_pts [3] = &D;

	convert_into_gmp_fixed_point (&dbl_pts [0], &gmp_pts [0], 4);

	/* At this point, we are working with exact integers. */

	mpz_sub (B.x, B.x, A.x);
	mpz_sub (B.y, B.y, A.y);
	mpz_sub (C.x, C.x, A.x);
	mpz_sub (C.y, C.y, A.y);
	mpz_sub (D.x, D.x, A.x);
	mpz_sub (D.y, D.y, A.y);

	mpz_init (bmag2);
	mpz_init (cmag2);
	mpz_init (dmag2);
	mpz_init (DxC);
	mpz_init (BxD);
	mpz_init (CxB);
	mpz_init (Z);

	mpz_mul (bmag2, B.x, B.x);
	mpz_mul (Z, B.y, B.y);
	mpz_add (bmag2, bmag2, Z);

	mpz_mul (cmag2, C.x, C.x);
	mpz_mul (Z, C.y, C.y);
	mpz_add (cmag2, cmag2, Z);

	mpz_mul (dmag2, D.x, D.x);
	mpz_mul (Z, D.y, D.y);
	mpz_add (dmag2, dmag2, Z);

	mpz_mul (DxC, D.x, C.y);
	mpz_mul (Z, D.y, C.x);
	mpz_sub (DxC, DxC, Z);

	mpz_mul (BxD, B.x, D.y);
	mpz_mul (Z, B.y, D.x);
	mpz_sub (BxD, BxD, Z);

	mpz_mul (CxB, C.x, B.y);
	mpz_mul (Z, C.y, B.x);
	mpz_sub (CxB, CxB, Z);

	mpz_mul (bmag2, bmag2, DxC);
	mpz_mul (cmag2, cmag2, BxD);
	mpz_mul (dmag2, dmag2, CxB);

	mpz_add (Z, bmag2, cmag2);
	mpz_add (Z, Z, dmag2);

	result = mpz_sgn (Z);

	mpz_clear (Z);
	mpz_clear (CxB);
	mpz_clear (BxD);
	mpz_clear (DxC);
	mpz_clear (dmag2);
	mpz_clear (cmag2);
	mpz_clear (bmag2);

	mpz_clear (D.y);	mpz_clear (D.x);
	mpz_clear (C.y);	mpz_clear (C.x);
	mpz_clear (B.y);	mpz_clear (B.x);
	mpz_clear (A.y);	mpz_clear (A.x);

	return (result);
}

#endif

/*
 * Convert the given array of "double precision points" into an
 * equivalent array of "GMP arbitrary precision integer points".
 * We choose a smallest reasonable integer K so that when every
 * "double" coordinate value is multiplied by 2**K we get a value that
 * is an integer.
 *
 * In the current implementation, K is always a multiple of 32 (or 64 --
 * whatever the GMP "limb" size is).  We do not attempt to cancel extra
 * powers of two by shifting the GMP integers to the right to eliminate
 * trailing zero bits -- this take too much time to justify the very rare
 * cases in which it might actually help.
 */

#ifdef HAVE_GMP

	static
	void
convert_into_gmp_fixed_point (

struct point **		dbl_pts,	/* IN - array of "double" points */
struct gmp_point **	gmp_pts,	/* OUT - array of "GMP" points */
int			n		/* IN - number of points to convert */
)
{
int			i;
int			size;
mp_exp_t		expon;
mp_exp_t		min_scale;
struct point *		dp;
struct gmp_point *	gp;
mpf_t			coord;
mpz_t			tmp;
mp_exp_t		xscale [4];
mp_exp_t		yscale [4];
bool			scale_seen;

	/* Set a precision of 3 GMP limbs.  This should give us up to	*/
	/* 96 bits of mantissa (at least 65 bits).  Since an IEEE 754	*/
	/* double is 53 bits, this should never lose any significant	*/
	/* bits.							*/

	mpf_init2 (coord, 3);

	scale_seen = FALSE;
	min_scale = 0;

	for (i = 0; i < n; i++) {
		dp = dbl_pts [i];
		gp = gmp_pts [i];

		/* Convert X coordinate... */
		if (dp -> x EQ 0.0) {
			/* Initialize to zero. */
			mpz_init (gp -> x);
		}
		else {
			mpf_set_d (coord, dp -> x);

			/* Access mantissa as an mpz_t... */
			tmp [0]._mp_alloc	= coord [0]._mp_prec + 1;
			tmp [0]._mp_size	= coord [0]._mp_size;
			tmp [0]._mp_d		= coord [0]._mp_d;

			/* Copy off mantissa as an integer. */
			mpz_init_set (gp -> x, tmp);

			/* Compute scale factor for this coordinate. */
			/* The mantissa is fractional (i.e., binary point */
			/* is at the most-significant end of the buffer), */
			/* so we need to adjust the exponent by the size  */
			/* of the mantissa.				  */
			size = coord [0]._mp_size;
			if (size < 0) {
				/* This just means the mantissa is negative. */
				size = -size;
			}
			expon = coord [0]._mp_exp - size;
			xscale [i] = expon;
			if ((NOT scale_seen) OR (expon < min_scale)) {
				min_scale = expon;
				scale_seen = TRUE;
			}
		}

		/* Convert Y coordinate... */
		if (dp -> y EQ 0.0) {
			/* Initialize to zero. */
			mpz_init (gp -> y);
		}
		else {
			mpf_set_d (coord, dp -> y);

			/* Access mantissa as an mpz_t... */
			tmp [0]._mp_alloc	= coord [0]._mp_prec + 1;
			tmp [0]._mp_size	= coord [0]._mp_size;
			tmp [0]._mp_d		= coord [0]._mp_d;

			/* Copy off mantissa as an integer. */
			mpz_init_set (gp -> y, tmp);

			/* Compute scale factor for this coordinate. */
			/* The mantissa is fractional (i.e., binary point */
			/* is at the most-significant end of the buffer), */
			/* so we need to adjust the exponent by the size  */
			/* of the mantissa.				  */
			size = coord [0]._mp_size;
			if (size < 0) {
				/* This just means the mantissa is negative. */
				size = -size;
			}
			expon = coord [0]._mp_exp - size;
			yscale [i] = expon;
			if ((NOT scale_seen) OR (expon < min_scale)) {
				min_scale = expon;
				scale_seen = TRUE;
			}
		}
	}

	mpf_clear (coord);

	/* Pass 2 -- properly scale all of the integer coordinates. */
	for (i = 0; i < n; i++) {
		dp = dbl_pts [i];
		gp = gmp_pts [i];

		if (dp -> x NE 0.0) {
			expon = xscale [i] - min_scale;
			if (expon > 0) {
				mpz_mul_2exp (gp -> x,
					      gp -> x,
					      GMP_LIMB_BITS * expon);
			}
		}

		if (dp -> y NE 0.0) {
			expon = yscale [i] - min_scale;
			if (expon > 0) {
				mpz_mul_2exp (gp -> y,
					      gp -> y,
					      GMP_LIMB_BITS * expon);
			}
		}
	}
}

#endif

/*
 * A quick test driver.
 */

#if TEST_DRIVER

	int
main (

int	argc,
char **	argv
)
{
int		i, n;
int		nedges;
int		ntri;
double *	terms;
struct pset *	pts;
int *		edges;
int *		tri;
int *		neigh;

	gst_open_geosteiner ();

	/* Setup a channel for stdout */
	chan = gst_create_channel (NULL, NULL);
	gst_channel_add_file (chan, stdout, NULL);

	/* Read in a point set. */
	n = gst_get_points (stdin, 0, &terms, NULL);

	_gst_define_Plot_Terminals (chan, n, terms, NULL);

	pts = NEW_PSET (n);
	pts -> n = n;
	for (i = 0; i < n; i++) {
		pts -> a [i].x = terms [2*i];
		pts -> a [i].y = terms [2*i + 1];
	}
	free (terms);

	if (argc <= 1) {
		/* Get DT edges only. */
		delaunay_triangulation_2 (pts, &nedges, &edges, NULL, NULL, NULL);
	}
	else {
		/* Get DT triangles and neighbors only. */
		delaunay_triangulation_2 (pts, &nedges, NULL, &ntri, &tri, &neigh);
	}

	gst_free_channel (chan);

	gst_close_geosteiner ();

	exit (0);
}

/*
 * Generate a postscript plot of the given triangulation.
 */

	static
	void
plot_triangulation (

struct DTinfo *		dt,	/* IN/OUT - global Delaunay triang data */
const char *		title	/* IN - title of the plot */
)
{
Eptr		e1, e2, e3;

	/* Plot the initial triangulation. */
	_gst_begin_plot (chan, BIG_PLOT);

	e1 = dt -> edges;
	e3 = e1 + dt -> num_edges;

	/* Print internal edges lightly. */
	gst_channel_printf (chan, "\t0.8 setgray\n"); 
	for (e2 = e1; e2 < e3; e2 += 2) {
		if ((e2 -> flags & ONX) NE 0) continue;
		gst_channel_printf (chan, "\t%d T %d T S\n",
				    e2 -> src, e2 -> dst);
	}

	/* Print external face in bold. */
	gst_channel_printf (chan, "\t0 setgray\n");
	for (e2 = e1; e2 < e3; e2 += 2) {
		if ((e2 -> flags & ONX) EQ 0) continue;
		gst_channel_printf (chan, "\t%d T %d T S\n",
				    e2 -> src, e2 -> dst);
	}
	gst_channel_printf (chan, "\tPlot_Terminals\n");

	_gst_end_plot (chan, title);
}

#endif	/* TEST_DRIVER */

#endif	/* NOT USE_TRIANGLE */
