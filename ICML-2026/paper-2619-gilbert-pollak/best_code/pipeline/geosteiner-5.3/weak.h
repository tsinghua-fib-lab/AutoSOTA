/***********************************************************************

	$Id: weak.h,v 1.9 2022/11/19 13:45:55 warme Exp $

	File:	weak.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1996, 2022 by Dvaid M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	A subtour separator that finds weak connectivity.

************************************************************************

	Modification Log:

	a-1:	07/05/2000	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef WEAK_H
#define	WEAK_H

struct bbinfo;
struct constraint;

extern struct constraint * _gst_find_weak_connectivity (
						double *		x,
						struct constraint *	cp,
						struct bbinfo *		bbip);

#endif
