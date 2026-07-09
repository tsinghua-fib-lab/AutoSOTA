/***********************************************************************

	$Id: lpinit.h,v 1.6 2022/11/19 13:45:53 warme Exp $

	File:	lpinit.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1995, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines to start up and shut down the LP solver.

************************************************************************

	Modification Log:

	a-1:	09/17/2005	warme
		: Split off from bbsubs.h.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Apply prefixes.

************************************************************************/

#ifndef	LPINIT_H
#define	LPINIT_H

extern void		_gst_shutdown_lp_solver (void);
extern void		_gst_startup_lp_solver (void);

#endif
