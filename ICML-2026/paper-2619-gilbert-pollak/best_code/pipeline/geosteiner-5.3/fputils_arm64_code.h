/***********************************************************************

	$Id: fputils_arm64_code.h,v 1.1 2023/04/03 17:35:57 warme Exp $

	File:	fputils_arm64_code.h
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

#ifndef	FPUTILS_ARM64_CODE_H
#define	FPUTILS_ARM64_CODE_H

#include "fputils_arm64.h"

#include "fatal.h"
#include <fenv.h>
#include "gsttypes.h"
#include "logic.h"
#include <string.h>


/*
 * Global Routines
 */

void	_gst_restore_floating_point_configuration (struct fpsave * sp);
void	_gst_save_floating_point_configuration (struct fpsave * sp);
void	_gst_set_floating_point_configuration (struct fpsave * sp);


/*
 * External References
 */

	/* none */

#if	NOT defined(__arm64__)
	#error "Unsupported platform!"
#endif

/*
 * Save the floating-point modes.
 */

	void
_gst_save_floating_point_configuration (

struct fpsave *		sp	/* IN - buffer to save FPU state into */
)
{
	fegetenv (&(sp -> fpu_state));
}

/*
 * Save off the caller's FPU modes, and force the modes that we
 * want to use within Geosteiner.
 */

	void
_gst_set_floating_point_configuration (

struct fpsave *		sp	/* IN - buffer to save FPU state into */
)
{
unsigned long long	fpcr, fpsr;
fenv_t			temp;

	fegetenv (&temp);

	sp -> fpu_state = temp;

	/* Enable the following:					*/
	/*	RMode	= Round to Nearest (RN) mode.			*/
	/*	OFE	= Overflow exception trap.			*/
	/*	DZE	= Divide by Zero exception trap.		*/
	/*	IOE	= Invalid Operation exception.			*/
	/* NOTE: On MacOS, these generate SIGILL and not SIGFPE!!!	*/

	/* On MacOS, all of these API bits match the hardware bits and	*/
	/* can just be ORed together.					*/
	fpcr = temp.__fpcr;

	fpcr &= ~(FE_TONEAREST | FE_UPWARD | FE_DOWNWARD | FE_TOWARDZERO);
	fpcr |= (FE_TONEAREST
		 | __fpcr_trap_invalid
		 | __fpcr_trap_divbyzero
		 | __fpcr_trap_overflow);

	temp.__fpcr = fpcr;

	/* Clear all pending exceptions. */
	temp.__fpsr &= ~FE_ALL_EXCEPT;

	fesetenv (&temp);
}


	void
_gst_restore_floating_point_configuration (

struct fpsave *		sp	/* IN - buffer to restore FPU state from */
)
{
	fesetenv (&(sp -> fpu_state));
}

/*
 * Note: GeoSteiner does not use "long double" arithmetic, but the
 * LP solver might.
 *
 * The ARM64 architecture does not provide any hardware instructions
 * for IEEE 754 "extendeded precision."  When you do use "long double",
 * what you actually get is 128-bit floating-point, implemented in
 * software.  We avoid using "long double" on this platform.
 */

	int
_gst_enable_long_double_precision ()

{
	return (0);
}

	void
_get_restore_long_double_precision (

int		prevState
)
{
}

#endif
