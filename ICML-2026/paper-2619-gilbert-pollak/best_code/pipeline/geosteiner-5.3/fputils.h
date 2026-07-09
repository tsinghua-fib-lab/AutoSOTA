/***********************************************************************

	$Id: fputils.h,v 1.4 2023/04/03 17:38:03 warme Exp $

	File:	fputils.h
	Rev:	e-3
	Date:	04/03/2023

	Copyright (c) 1993, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Various floating-point utility routines.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from steiner.h.
	e-2:	09/30/2016	warme
		: Added control of long double arithmetic.
	e-3:	04/03/2023	warme
		: Add support for __arm64__ by splitting this up
		:  into CPU-specific files.

************************************************************************/

#ifndef	FPUTILS_H
#define	FPUTILS_H

#if defined (__arm64__)
 #include "fputils_arm64.h"
#elif defined (__x86_64__)
 #include "fputils_x86.h"
#else
 /* Unsupported architecture.  Provide dummy object.		*/
 /* We don't generate any error here, only in fputils.c.	*/
 struct fpsave {
	double	dummy;
 };
#endif

/*
 * Function Prototypes.
 */

extern int	_gst_enable_long_double_precision ();
extern void	_gst_restore_long_double_precision (int prevState);

extern void	_gst_restore_floating_point_configuration (struct fpsave * sp);
extern void	_gst_save_floating_point_configuration (struct fpsave * sp);
extern void	_gst_set_floating_point_configuration (struct fpsave * sp);
extern void	_gst_store_double (double * dp, double x);

#endif
