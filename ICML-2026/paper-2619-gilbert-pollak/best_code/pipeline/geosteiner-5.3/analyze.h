/***********************************************************************

	$Id: analyze.h,v 1.10 2022/11/19 13:45:49 warme Exp $

	File:	analyze.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1997, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Declarations pertaining to the subtour constraint
	analyzer (reverse-engineers subtour constraints).

************************************************************************

	Modification Log:

	a-1:	02/25/2000	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/


#ifndef ANALYZE_H
#define	ANALYZE_H

#include "gsttypes.h"


struct bbinfo;

extern void	_gst_analyze_constraints (struct bbinfo * bbip, bool lp_only);

#endif
