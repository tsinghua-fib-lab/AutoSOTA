/***********************************************************************

	$Id: fputils_arm64.h,v 1.1 2023/04/03 17:35:57 warme Exp $

	File:	fputils_arm64.h
	Rev:	e-1
	Date:	04/03/2023

	Copyright (c) 1993, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Floating-point stuff for __arm64__.

************************************************************************

	Modification Log:

	e-1:	04/03/2023	warme
		: Created.

************************************************************************/

#ifndef	FPUTILS_ARM64_H
#define	FPUTILS_ARM64_H

#include <fenv.h>

/*
 * The following structure is used to save floating point information
 * on the ARM64 CPU.
 * Throughout most of GeoSteiner, this is just uninterpreted black-box
 * data, but we need the CPU-specific details here.
 */

struct fpsave {
	fenv_t	fpu_state;
};

#endif
