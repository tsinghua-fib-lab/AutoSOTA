/***********************************************************************

	$Id: fputils_x86.h,v 1.1 2023/04/03 17:33:38 warme Exp $

	File:	fputils_x86.h
	Rev:	e-3
	Date:	04/03/2023

	Copyright (c) 1993, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Floating-point stuff for __x86_64__.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from steiner.h.
	e-2:	09/30/2016	warme
		: Added control of long double arithmetic.
	e-3:	04/03/2023	warme
		: Split off from fputils.h.

************************************************************************/

#ifndef	FPUTILS_X86_H
#define	FPUTILS_X86_H

/*
 * The following structure is used to save floating point information.
 * Throughout most of GeoSteiner, this is just uninterpreted black-box
 * data, but we need the CPU-specific details here.
 * We use an array of doubles to guarantee proper alignment.
 */

struct fpsave {
	double	data [4];
};

#endif
