/***********************************************************************

	$Id: heapsortx.c,v 1.6 2022/11/19 13:45:52 warme Exp $

	File:	heapsortx.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2000, 2022 by David M. warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Sort points by X,Y coordinate using heapsort.

************************************************************************

	Modification Log:

	a-1:	09/17/2005	warme
		: Created.  Moved here from fstfuncs.c file.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#include "sortfuncs.h"

#include "logic.h"
#include "memory.h"
#include "point.h"

/*
 * Global Routines
 */

int *		_gst_heapsort_x (struct pset * pts);

/*
 * Use the heapsort algorithm to sort the given terminals in increasing
 * order by the following keys:
 *
 *	1.	X coordinate
 *	2.	Y coordinate
 *	3.	index (i.e., position within input data)
 *
 * Of course, we do not move the points, but rather permute an array
 * of indexes into the points.
 */

	int *
_gst_heapsort_x (

struct pset *		pts		/* IN - the terminals to sort */
)
{
int			i, i1, i2, j, k, n;
struct point *		p1;
struct point *		p2;
int *			index;

	n = pts -> n;

	index = NEWA (n, int);
	for (i = 0; i < n; i++) {
		index [i] = i;
	}

	/* Construct the heap via sift-downs, in O(n) time. */
	for (k = n >> 1; k >= 0; k--) {
		j = k;
		for (;;) {
			i = (j << 1) + 1;
			if (i + 1 < n) {
				/* Increment i (to right subchild of j) */
				/* if the right subchild is greater. */
				i1 = index [i];
				i2 = index [i + 1];
				p1 = &(pts -> a [i1]);
				p2 = &(pts -> a [i2]);
				if ((p2 -> x > p1 -> x) OR
				    ((p2 -> x EQ p1 -> x) AND
				     ((p2 -> y > p1 -> y) OR
				      ((p2 -> y EQ p1 -> y) AND
				       (i2 > i1))))) {
					++i;
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			i1 = index [j];
			i2 = index [i];
			p1 = &(pts -> a [i1]);
			p2 = &(pts -> a [i2]);
			if ((p1 -> x > p2 -> x) OR
			    ((p1 -> x EQ p2 -> x) AND
			     ((p1 -> y > p2 -> y) OR
			      ((p1 -> y EQ p2 -> y) AND
			       (i1 > i2))))) {
				/* Greatest child is smaller.  Sift-	*/
				/* down is done. */
				break;
			}
			/* Sift down and continue. */
			index [j] = i2;
			index [i] = i1;
			j = i;
		}
	}

	/* Now do actual sorting.  Exchange first/last and sift down. */
	while (n > 1) {
		/* Largest is at index [0], swap with index [n-1],	*/
		/* thereby putting it into final position.		*/
		--n;
		i = index [0];
		index [0] = index [n];
		index [n] = i;

		/* Now restore the heap by sifting index [0] down. */
		j = 0;
		for (;;) {
			i = (j << 1) + 1;
			if (i + 1 < n) {
				/* Increment i (to right subchild of j) */
				/* if the right subchild is greater. */
				i1 = index [i];
				i2 = index [i + 1];
				p1 = &(pts -> a [i1]);
				p2 = &(pts -> a [i2]);
				if ((p2 -> x > p1 -> x) OR
				    ((p2 -> x EQ p1 -> x) AND
				     ((p2 -> y > p1 -> y) OR
				      ((p2 -> y EQ p1 -> y) AND
				       (i2 > i1))))) {
					++i;
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			i1 = index [j];
			i2 = index [i];
			p1 = &(pts -> a [i1]);
			p2 = &(pts -> a [i2]);
			if ((p1 -> x > p2 -> x) OR
			    ((p1 -> x EQ p2 -> x) AND
			     ((p1 -> y > p2 -> y) OR
			      ((p1 -> y EQ p2 -> y) AND
			       (i1 > i2))))) {
				/* Greatest child is smaller.  Sift-	*/
				/* down is done. */
				break;
			}
			/* Sift down and continue. */
			index [j] = i2;
			index [i] = i1;
			j = i;
		}
	}

	return (index);
}
