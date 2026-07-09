/***********************************************************************

	$Id: sec2.h,v 1.9 2022/11/19 13:45:54 warme Exp $

	File:	sec2.h
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1997, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Declarations for the deterministic separation procedure
	for the "generalized SEC's" that uses a reduction to min-cut
	in a bipartite network.

************************************************************************

	Modification Log:

	a-1:	05/16/1997	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
	e-4:	11/18/2022	warme
		: Simplify calling convention.

************************************************************************/

#ifndef SEC2_H
#define	SEC2_H

struct bbinfo;
struct comp;
struct constraint;

/*
 * Function Prototypes
 */

extern struct constraint *	_gst_sec_flow_separator (
					struct comp **		comp_hookp,
					double *		x,
					struct bbinfo *		bbip,
					struct constraint *	cp);

#endif
