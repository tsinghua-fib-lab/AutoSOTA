/***********************************************************************

	$Id: rfst.h,v 1.11 2022/11/19 13:45:54 warme Exp $

	File:	rfst.h
	Rev:	e-4
	Date:	10/09/2016

	Copyright (c) 1998, 2022 by Martin Zachariasen.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Declarations for the Rectilinear FST generator.

************************************************************************

	Modification Log:

	a-1:	08/31/98	warme
		: Created.  Implemented Zachariasen's algorithm
		:  using Warme's infrastructure.
	b-1:	02/02/2014	warme
		: Fix compiler errors from extern/static mismatch.
		: Note that this fix assumes that this include file
		:  is only included by rfst.c!
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files.
	e-4:	10/09/2016	warme
		: Fix more -Wall issues.

************************************************************************/

#ifndef	RFST_H
#define	RFST_H

#include "bitmaskmacros.h"
#include "geomtypes.h"
#include "gsttypes.h"
#include <stddef.h>

struct bsd;
struct full_set;
struct point;
struct pset;


/*
 * A type to represent which side of the backbone a particular point is on.
 */

enum side {
	UNKNOWN,
	LEFT,
	RIGHT
};


/*
 * Types of Hwang Topologies
 */

enum treetype {
	TYPE_1,
	TYPE_2,
	STRAIGHT,
	CROSS,
};

/*
 * A structure to keep track of one RFST.  They are kept in a hash table
 * so that we can rapidly identify duplicates.  We also keep them all in
 * one big list, which is doubly-linked so that we can rapidly delete a
 * duplicate that is suboptimal.
 */

struct rlist {
	struct rlist *		forw;	/* Next RFST in saved list */
	struct rlist *		back;	/* Previous RFST in saved list */
	struct rlist *		next;	/* Next RFST in hash table */
	int			size;	/* Number of terminals */
	struct full_set *	fst;	/* The actual FST */
};

/*
 * A type for "pointer to a distance computation function".
 */

typedef dist_t	(*dist_funcp) (struct point *, struct point *);


/*
 * Global information used by the RFST generator.
 */

struct rinfo {
	struct pset *	pts;		/* The set of terminals */
	int *		x_order;	/* Order of terminals by X, Y, index */
	int *		y_order;	/* Order of terminals by Y, X, index */
	int *		succ [7];	/* Successor terminal in each of 4 */
					/* directions (plus 3 wrap dirs) */
	bitmap_t *	empty_rect;	/* Which terms i,j form empty
					   rectangles */
	dist_t		mst_length;	/* Length of the MST */
	dist_t *	ub0 [7];	/* The upper bounds UB0 */
	dist_t *	ub1 [7];	/* The upper bounds UB1 */
	int **		zt [7];		/* Short leg candidate terminals */
	struct bsd *	bsd;		/* Bottleneck Steiner distance info */

	/* Arrays/variables used in recursive growing of FSTs. */
	int *		terms;
	int *		longterms;
	dist_t *	maxedges;
	int *		shortterm;
	int *		lrindex;

	int *		dirsucc;

	int		fsts_checked;	/* Num FSTs sent to screening tests */

	bool *		term_check;	/* To compare FST terminal sets */

	int		num_term_masks;	/* Size of terminal mask in each FST */

	struct rlist **	hash;		/* FST hash table */
	struct rlist	list;		/* Head of circular RFST list */

	int		ntrees;		/* Final number of FSTs */
	struct full_set * full_sets;	/* Final list of FSTs */
	struct full_set ** hookp;	/* For adding to end of FST list */
};

/*
 * Useful macros.
 */

/* DSTDIR(p,q,dir) gives the distance between p and q in direction dir.	*/
/* (effectively deltaX if dir is even, deltaY if dir is odd.)		*/
/* DSTDIRP(p,q,dir) gives the distence between p and q in the direction	*/
/* *perpendicular* to direction dir.					*/
/* We offer several implementations here.  Your mileage may vary on	*/
/* which is fastest for your machine.					*/
/* For the sake of efficiency, we also define a type and two macros	*/
/* that permit us to move part of the computation out of loops:		*/
/*	dstdiroff_t	diroff = DSTDIR_GETOFFSET(dir);			*/
/*	...								*/
/*		dist = DSTDIR_OFFSET(p, q, diroff);			*/

#if 0

	/* The simple implementation. */
#define	DSTDIR(p,q,dir)	 (((dir) & 1) ? DELTAY((p1),(p2)) : DELTAX((p1),(p2)))
#define	DSTDIRP(p,q,dir) (((dir) & 1) ? DELTAX((p1),(p2)) : DELTAY((p1),(p2)))
typedef int	dstdiroff_t;
#define	DSTDIR_GETOFFSET(dir)	(dir)
#define	DSTDIR_OFFSET(p,q,dir)	DSTDIR(p,q,dir)

#elif 1
	/* This implementation uses no conditional branches.  It will	*/
	/* be better on machines where pipeline flushes are an issue.	*/
	/* We could really use the pointer-to-member feature of C++ here! */
/* extern const int8u	dstdir_offsets []; */
typedef size_t	dstdiroff_t;
#define	DSTDIR(p,q,dir)	DSTDIR_OFFSET ((p), (q), dstdir_offsets [dir])
#define	DSTDIRP(p,q,dir) DSTDIR((p), (q), (dir) + 1)
#define	DSTDIR_GETOFFSET(dir)	(dstdir_offsets [dir])
#define	DSTDIR_OFFSET(p,q,off)	(fabs (_DEREF((p),(off)) - _DEREF((q),(off))))
#ifdef __GNUC__
	/* A slightly more type-safe version for GCC... */
#define	_DEREF(p,off)	(* ({ struct point * _p_ = (p); \
			      ((coord_t *) (((char *) _p_) + (off))); }) )
#else
#define	_DEREF(p,off)	(* ((coord_t *) (((char *) (p)) + (off))))
#endif

#elif 0
	/* Another implementation using an array of functions. */
typedef dist_t		(*dstdiroff_t) (struct point *, struct point *);
extern const dstdiroff_t	dstdir_funcs [];
#define NEED_DSTDIR_FUNCS
#define	DSTDIR(p,q,dir)	DSTDIR_OFFSET ((p), (q), dstdir_funcs [dir])
#define	DSTDIRP(p,q,dir) DSTDIR_OFFSET ((p), (q), (dir) + 1)
#define	DSTDIR_GETOFFSET(dir)	(dstdir_funcs [dir])
#define	DSTDIR_OFFSET(p,q,off)	((*(off)) ((p), (q)))

#endif	/* end of DSTDIR, DSTDIRP implementations */

/*
 * The macro SPOINT(s, r, p, dir) sets the XY coordinates of Steiner
 * point s.  The Steiner point is located along the ray emanating from
 * point r in the given direction.  Its position along that ray is
 * given by point p.
 */

#if 0
	/* A simple implementation.  Too many pipeline flushes. */
#define	SPOINT(s,r,p,dir) \
	((((dir) & 1) EQ 0) \
	 ? ((s) -> x = (p) -> x, (s) -> y = (r) -> y) \
	 : ((s) -> x = (r) -> x, (s) -> y = (p) -> y))

#elif 1
	/* A faster implementation.  Needs the "dstdir_offsets" and	*/
	/* "_DEREF" stuff above...					*/

#ifdef __GNUC__
	/* A slightly more type-safe version for GCC... */
#define	SPOINT(s,r,p,dir) \
	({ struct point *_s = (s), *_r = (r), *_p = (p); \
	   int _dir = (dir); \
	   dstdiroff_t	_doff = dstdir_offsets [_dir], \
			_dpoff = dstdir_offsets [_dir+1]; \
	   _DEREF (_s, _doff)  = _DEREF (_p, _doff); \
	   _DEREF (_s, _dpoff) = _DEREF (_r, _dpoff); })

#else /* not GCC */
#define	SPOINT(s,r,p,dir) \
	(_DEREF (s, dstdir_offsets [dir]) \
		= _DEREF (p, dstdir_offsets [dir]), \
	 _DEREF (s, dstdir_offsets [dir+1]) \
		= _DEREF (r, dstdir_offsets [dir+1]))
#endif

#endif

/*
 * The macro LRINDEX(p,q,dir) returns LEFT=0 or RIGHT=1, depending on
 * whether the point q lies strictly to the left of the ray emanating
 * from point p in direction dir.  Points exactly ON the ray are
 * arbitrarily considered to be on the RIGHT of it, although the code
 * should not be using this macro in such cases.
 *
 * We implement this using an array of pointers to functions, each of which
 * does the appropriate comparison.
 * Once again, we provide a typedef and two macros so that we can move
 * portions of the computation out of loops.
 */

typedef int	(* lrindex_funcp) (struct point *, struct point *);
/* extern const lrindex_funcp	lrindex_funcs []; */
typedef lrindex_funcp	lrdiroff_t;

#define	LRINDEX(p,q,dir)	LRINDEX_OFFSET((p), (q), lrindex_funcs [dir])
#define	LRINDEX_GETOFFSET(dir)	(lrindex_funcs [dir])
#define	LRINDEX_OFFSET(p,q,off)	((*(off)) ((p), (q)))

#endif
