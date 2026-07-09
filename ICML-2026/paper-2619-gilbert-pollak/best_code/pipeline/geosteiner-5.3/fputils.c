/***********************************************************************

	$Id: fputils.c,v 1.5 2023/04/03 17:39:34 warme Exp $

	File:	fputils.c
	Rev:	e-4
	Date:	04/03/2023

	Copyright (c) 1993, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Various floating-point utility routines.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from utils.c.
		: Support both x87 and SSE/SSE2 FPUs.
	e-2:	09/30/2016	warme
		: Added control of long double arithmetic.
	e-3:	10/09/2016	warme
		: Fix more -Wall issues.
	e-4:	04/03/2023	warme
		: Add support for __arm64__ by splitting this up
		:  into CPU-specific files.

************************************************************************/

#include "fputils.h"

#if defined (__arm64__)
	#include "fputils_arm64_code.h"
#elif defined (__x86_64__)
	#include "fputils_x86_code.h"
#else
	#error "Unsupported platform!"
#endif


/*
 * Global Routines
 */

void	_gst_store_double (double * dp, double x);


/*
 * External References
 */

	/* none */

/*
 * Special routine to work around problems such as the Intel floating
 * point implementation -- where registers have more precision than
 * memory.  This routine FORCES a value to be stored into memory, from
 * which we can then re-load a value that has exactly the precision of
 * a double, and no more.
 */

	void
_gst_store_double (

double *	dp,		/* OUT - double variable to store into */
double		x		/* IN - double value to store */
)
{
	*dp = x;
}
