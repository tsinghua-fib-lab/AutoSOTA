/***********************************************************************

	$Id: sec_heur.h,v 1.11 2023/04/03 17:00:30 warme Exp $

	File:	sec_heur.h
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 1996, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Data structures for the various heuristic separation
	procedures for Strong SEC's (Subtour Elimination
	Constraints).

************************************************************************

	Modification Log:

	a-1:	05/13/96	warme
		: Created.
	b-1:	02/28/2001	warme
		: Changes for 3.1 release.
		: Made check_unique_subtour be global.
		: Change calling convention of enumerate_all_subtours
		:  and find_small_subtours.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganized include files, apply prefixes.
	e-4:	11/18/2022	warme
		: Simplify calling convention.
	e-5:	04/03/2023	warme
		: Removed _gst_check_unique_subtour().

************************************************************************/

#ifndef SEC_HEUR_H
#define	SEC_HEUR_H

#include "bitmaskmacros.h"

struct bbinfo;
struct bbnode;
struct comp;
struct constraint;

/*
 * Function Prototypes
 */

extern struct constraint *	_gst_check_subtour (
						bitmap_t *		stour,
						struct constraint *	clist,
						double *		x,
						struct bbinfo *		bbip);
extern struct constraint *	_gst_enumerate_all_subtours (
						struct comp *		comp,
						struct constraint *	cp,
						struct bbinfo *		bbip);
extern struct constraint *	_gst_find_integer_cycles (
						double *		x,
						struct constraint *	cp,
						struct bbinfo *		bbip);
extern struct constraint *	_gst_find_small_subtours (
						struct comp *		comp,
						struct constraint *	cp,
						struct bbinfo *		bbip);
extern bool			_gst_is_equal (bitmap_t *		bp1,
					       bitmap_t *		bp2,
					       int			nmasks);
extern struct constraint *	_gst_sec_flow_heuristic (
						struct comp *		comp,
						double *		x,
						struct bbinfo *		bbip,
						struct constraint *	cp);


#endif
