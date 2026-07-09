/***********************************************************************

	$Id: fpdebug.h,v 1.7 2022/11/19 13:45:51 warme Exp $

	File:	fpdebug.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2003, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Floating point debugging stuff.

************************************************************************

	Modification Log:

	a-1:	08/13/2003	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Apply prefixes.

************************************************************************/

#ifndef FPDEBUG_H
#define FPDEBUG_H

/*
 * If we are on an x86/Linux architecture, install a signal handler for
 * floating point exceptions that prints detailed information about the
 * FPU state at the time the exception was thrown.
 *
 * NOTE: This routine should never be called from library code because
 * the library must never establish or alter any signal handlers!
 */

extern	void _gst_establish_debug_floating_point_signal_handler ();

#endif
