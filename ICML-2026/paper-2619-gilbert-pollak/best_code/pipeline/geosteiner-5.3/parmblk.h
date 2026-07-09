/***********************************************************************

	$Id: parmblk.h,v 1.15 2022/11/19 13:45:53 warme Exp $

	File:	parmblk.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	The parameter block structure generated with the help
	of a cryptic set of macros.

************************************************************************

	Modification Log:

	a-1:	04/30/2002	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef PARMBLK_H
#define PARMBLK_H

#include "parmdefs.h"

struct gst_channel;
struct gst_param;

#define INTROW(sym,val,var,min,max,dflt) int var;
#define DBLROW(sym,val,var,min,max,dflt) double var;
#define STRROW(sym,val,var,func,dflt) char * var;
#define CHNROW(sym,val,var,func) struct gst_channel * var;

/* Define the parameter block.  Note that this version does not */
/* segregate variables into sub-structures... */

struct gst_param {
	int		magic;
	int		pad;
	DBLPARMS(DBLROW)
	INTPARMS(INTROW)
	STRPARMS(STRROW)
	CHNPARMS(CHNROW)
};

#undef INTROW
#undef DBLROW
#undef STRROW
#undef CHNROW

extern const struct gst_param		_gst_default_parmblk;

#endif
