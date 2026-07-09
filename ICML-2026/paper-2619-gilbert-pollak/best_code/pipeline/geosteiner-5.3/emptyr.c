/***********************************************************************

	$Id: emptyr.c,v 1.10 2022/11/19 13:45:51 warme Exp $

	File:	emptyr.c
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
	b-1:	01/31/2000	martinz
		: Successor array is computed within initialization
		: if given as a NULL pointer.
	b-2:	12/18/2014	warme
		: Fix several 64-bit architecture issues (use size_t).
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.

************************************************************************/

#include "emptyr.h"

#include <float.h>
#include "logic.h"
#include "memory.h"
#include "point.h"
#include "sortfuncs.h"
#include <stdlib.h>
#include "steiner.h"


/*
 * Global Routines
 */

int		_gst_count_empty_rectangles (bitmap_t * bits, int n);
bitmap_t *	_gst_init_empty_rectangles (struct pset *pts, int * succ0);
bool		_gst_is_empty_rectangle (bitmap_t * bits, int i, int j);
void		_gst_shutdown_empty_rectangles (bitmap_t * bits);


/*
 * Local Macros
 */

#define _POS(i,j)	((((size_t) i * (i-1)) >> 1) + j)
#define POS(i,j)	((i >= j) ? _POS(i,j) : _POS(j,i))


/*
 * Local Routines
 */

static void		set_bit (bitmap_t *, int, int);

/*
 * Pre-compute an N by N boolean matrix whose (i,j)-th element is
 * TRUE if-and-only-if the interior of the rectangle defined by
 * terminals i and j is devoid of terminals.
 *
 * Since this matrix is symmetric and the diagonal elements are
 * all TRUE, we store only the lower triangle in a compact bit-vector
 * form.
 *
 * The succ0 array may be given as the NULL pointer in which case it is
 * computed by the the procedure.
 */

	bitmap_t *
_gst_init_empty_rectangles (

struct pset *		pts,	/* IN - point set to use */
int *			succ0	/* IN - next "dir 0" successor for each term */
)
{
int		i;
int		j;
int		n;
bitmap_t *	bits;
size_t		nbits;
size_t		nwords;
int *		x_order;
int *		succ;
struct point *	p;
struct point *	q;
double		dx, dy;
double		x, y;
double		top_dist, bot_dist;
double		old_top_dist, old_bot_dist;
double		top_x, bot_x;

	n = pts -> n;

	x_order = NULL;
	succ	= NULL;

	/* Do we need to compute succ array? */
	if (succ0 EQ NULL) {
		x_order = _gst_heapsort_x (pts);
		succ	= NEWA (n, int);

		for (i = 1; i < n; i++) {
			succ [x_order [i - 1]] = x_order [i];
		}
		succ [x_order [i - 1]] = -1;
	}
	else {
		succ = succ0;
	}

	/* Create the lower-triangular bit-vector and zero it. */
	nbits = (((size_t) n) * (n - 1)) >> 1;
	nwords = BMAP_ELTS (nbits);

	bits = NEWA (nwords, bitmap_t);
	for (i = 0; i < nwords; i++) {
		bits [i] = 0;
	}

	p = &(pts -> a [0]);
	for (i = 0; i < n; i++, p++) {
		x = p -> x;
		y = p -> y;

		top_dist	= INF_DISTANCE;
		bot_dist	= INF_DISTANCE;
		old_top_dist	= INF_DISTANCE;
		old_bot_dist	= INF_DISTANCE;
		top_x		= x;
		bot_x		= x;
		for (j = succ [i]; j >= 0; j = succ [j]) {
			q = &(pts -> a [j]);
			dx = q -> x - x;
			if (dx EQ 0.0) {
				/* Q is exactly on vertical line through P. */
				set_bit (bits, i, j);
				continue;
			}

			dy = q -> y - y;
			if (dy EQ 0.0) {
				/* Q is exactly on horiz line through Q. */
				set_bit (bits, i, j);
				continue;
			}

			if (dy > 0.0) {
				/* Q is on top (above P). */
				if (dy <= top_dist) {
					set_bit (bits, i, j);
					if (q -> x > top_x) {
						old_top_dist = top_dist;
						top_x = q -> x;
					}
					top_dist = dy;
				}
				else if ((q -> x EQ top_x) AND
					 (dy <= old_top_dist)) {
					set_bit (bits, i, j);
				}
			}
			else {
				/* Q is on bottom (below P). */
				dy = - dy;
				if (dy <= bot_dist) {
					set_bit (bits, i, j);
					if (q -> x > bot_x) {
						old_bot_dist = bot_dist;
						bot_x = q -> x;
					}
					bot_dist = dy;
				}
				else if ((q -> x EQ bot_x) AND
					 (dy <= old_bot_dist)) {
					set_bit (bits, i, j);
				}
			}
		}
	}

	if (succ0 EQ NULL) {
		free ((char *) x_order);
		free ((char *) succ);
	}

	return (bits);
}

/*
 * Set bit (i,j) in the bit matrix.
 */

	static
	void
set_bit (

bitmap_t *		bits,	/* IN - bit vector for matrix */
int			i,	/* IN - row number */
int			j	/* IN - column number */
)
{
size_t			pos;

	/* The diagonal is not stored! */
	if (i EQ j) return;

	pos = POS (i, j);
	SETBIT (bits, pos);
}

/*
 * Clean up the empty rectangles data structure.
 */

	void
_gst_shutdown_empty_rectangles (

bitmap_t *		bits	/* IN - empty rectangle bit matrix */
)
{
	free ((char *) bits);
}

/*
 * Determine if the rectangle defined by terminals i and j is empty.
 */

	bool
_gst_is_empty_rectangle (

bitmap_t *		bits,	/* IN - empty rectangle bit matrix */
int			i,	/* IN - first terminal number */
int			j	/* IN - second terminal number */
)
{
size_t			pos;

	if (i EQ j) {
		/* The rectangle is zero by zero, and therefore has	*/
		/* no interior.	 There cannot be any terminals inside!	*/
		return (TRUE);
	}

	pos = POS (i, j);

	return (BITON (bits, pos));
}

/*
 * This routine counts the total number of 1 bits in the bit matrix.
 * This represents the number of pairs of terminals that define
 * rectangles whose INTERIORS are devoid of terminals.
 */

	int
_gst_count_empty_rectangles (

bitmap_t *		bits,	/* IN - empty rectangle bit matrix */
int			n	/* IN - number of terminals */
)
{
int		i;
size_t		nbits;
size_t		nwords;
int		count;
bitmap_t	mask1;
bitmap_t	mask2;
bitmap_t	limit;

	nbits = ((size_t) n * (n - 1)) >> 1;

	nwords = nbits / BPW;
	nbits  = nbits % BPW;

	count = 0;
	for (i = 0; i < nwords; i++) {
		mask1 = bits [i];
		while (mask1 NE 0) {
			mask1 ^= (mask1 & -mask1);
			++count;
		}
	}

	if (nbits > 0) {
		/* Count straggling bits in last word. */
		mask1 = bits [nwords];
		limit = 1 << nbits;	/* first bit to EXCLUDE from count */

		while (mask1 NE 0) {
			mask2 = (mask1 & -mask1);
			if (mask2 >= limit) break;
			mask1 ^= mask2;
			++count;
		}
	}

	return (count);
}
