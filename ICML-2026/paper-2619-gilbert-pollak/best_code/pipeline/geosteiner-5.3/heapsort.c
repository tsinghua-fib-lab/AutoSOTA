/***********************************************************************

	$Id: heapsort.c,v 1.6 2022/11/19 13:45:52 warme Exp $

	File:	heapsort.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2000, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	A general purpose heapsort function.

************************************************************************

	Modification Log:

	a-1:	09/17/2005	warme
		: Created.  Moved here from greedy.c file.
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

/*
 * Global Routines
 */

int *		_gst_heapsort (int			n,
			       void *			array,
			       gst_compare_func_ptr	compare);

/*
 * Use the heapsort algorithm to sort the given array according to
 * the following keys:
 *
 *	1.	comparison-function
 *	2.	index (i.e., position within input data)
 *
 * Of course, we do not move the elements, but rather permute an array
 * of indexes into the array.
 */
	int *
_gst_heapsort (

int			n,	/* IN - number of elements in array */
void *			array,	/* IN - array to be sorted */
gst_compare_func_ptr	compare
)
{
int			i, i1, i2, j, k;
int *			index;

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
				if (compare (i1, i2, (void *)array) < 0) {
					++i;
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			i1 = index [j];
			i2 = index [i];
			if (compare (i2, i1, array) < 0) {
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
				if (compare (i1, i2, array) < 0) {
					++i;
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			i1 = index [j];
			i2 = index [i];
			if (compare (i2, i1, array) < 0) {
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
