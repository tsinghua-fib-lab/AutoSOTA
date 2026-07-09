/***********************************************************************

	$Id: sortfuncs.h,v 1.6 2022/11/19 13:45:55 warme Exp $

	File:	sortfuncs.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Generally useful sorting functions.

************************************************************************

	Modification Log:

	a-1:	09/17/2005	warme
		: Created.  Gathered all the various generally
		:  reusable sorting functions here.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
		: Included rename.h.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef	SORTFUNCS_H
#define	SORTFUNCS_H

struct pset;

typedef int (*gst_compare_func_ptr) (int, int, void *);

extern int *	_gst_heapsort (int			n,
			  void *		array,
			  gst_compare_func_ptr	compare);
extern int *	_gst_heapsort_x (struct pset *);
extern int *	_gst_heapsort_y (struct pset *);
extern void	_gst_sort_ints (int * array, int n);

#endif
