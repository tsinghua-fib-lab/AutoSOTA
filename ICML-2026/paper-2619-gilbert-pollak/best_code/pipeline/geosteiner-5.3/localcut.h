/***********************************************************************

	$Id: localcut.h,v 1.10 2022/11/19 13:45:53 warme Exp $

	File:	localcut.h
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1997, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Declarations pertaining to the local cut generator.

************************************************************************

	Modification Log:

	a-1:	12/07/99	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Make features unconditional.
	e-4:	11/18/2022	warme
		: Simplify calling conventions.

************************************************************************/

#ifndef LOCALCUT_H
#define	LOCALCUT_H

#include "bitmaskmacros.h"

struct bbinfo;
struct comp;
struct constraint;
struct gst_channel;

extern struct constraint *	_gst_find_local_cuts (
					double *		x,
					struct bbinfo *		bbip,
					struct constraint *	cp);
extern struct constraint *	_gst_find_local_cuts_in_component (
					struct comp *		comp,
					double *		x,
					struct bbinfo *		bbip,
					struct constraint *	cp);
extern void			_gst_print_forests (
					struct comp *		comp,
					bitmap_t *		flist,
					int			n,
					struct gst_channel *	chan);

#endif
