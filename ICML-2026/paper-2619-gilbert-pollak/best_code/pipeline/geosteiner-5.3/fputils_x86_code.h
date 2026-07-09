/***********************************************************************

	$Id: fputils_x86_code.h,v 1.1 2023/04/03 17:33:38 warme Exp $

	File:	fputils_x86_code.h
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
		: Split off from fputils.c.

************************************************************************/

#ifndef	FPUTILS_X86_CODE_H
#define	FPUTILS_X86_CODE_H

#include "fputils_x86.h"

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

/*
 * Determine our platform
 */

#undef	PLATFORM_INTEL_32_BIT
#undef	PLATFORM_INTEL_64_BIT

#if defined(__i386) OR defined(__i386) OR defined(__WIN32__)
	#define PLATFORM_INTEL_32_BIT	1
#elif defined(__x86_64__)
	#define PLATFORM_INTEL_64_BIT	1
#else
	#error "Unsupported platform!"
#endif

/*
 * Determine which FPUs we are using.
 */

#undef USING_X87_FPU
#undef USING_SSE_SSE2_FPU

#if defined (PLATFORM_INTEL_32_BIT)
	#define USING_X87_FPU		1
#elif defined (PLATFORM_INTEL_64_BIT)
	/* This platform uses both FPU's. */
	#define USING_X87_FPU		1
	#define USING_SSE_SSE2_FPU	1
#else
	#error "Unsupported FPU architecture!"
#endif

/*
 * The following structure defines the format of the data that is
 * loaded and stored by the Intel x86 FPU instructions FLDENV and FSTENV.
 * Very hardware specific, and only correct in 32-bit protected mode.
 */

#ifdef USING_X87_FPU

struct x87_fpusave {
	unsigned int		cw;	/* FPU control word */
	unsigned int		sw;	/* FPU status word */
	unsigned int		tag;	/* FPU tag word */
	unsigned int		ip;	/* FPU instruction pointer offset */
	unsigned short		opcode;	/* FPU opcode */
	unsigned short		cs;	/* FPU instruction pointer selector */
	unsigned int		oprnd;	/* FPU operand pointer offset */
	unsigned int		ds;	/* FPU operand pointer selector */
};

/*
 * Read the x87 FPU Control Word register.
 */

	static
	inline
	int
read_x87_FPU_control_word ()

{
volatile unsigned short	cw;

	/* Get current floating-point Control Word register value. */
	__asm__ volatile ("fnstcw %0" : "=m" (*&cw));

	return (cw);
}

/*
 * Store the given value into the x87 FPU Control Word register.
 * Previous value is forgotten.
 */

	static
	inline
	void
set_x87_FPU_control_word (

int		value		/* IN: FPU Control Word value */
)
{
volatile unsigned short	cw;

	cw = value;

	__asm__ volatile ("fldcw %0" : : "m" (*&cw));
}

/*
 * Inline routine to update the x87 FPU Control Word.
 * Returns the original state of the Control Word.
 */

	static
	inline
	int
update_x87_FPU_control_word (

int	dst_mask,		/* IN: Bits to assign */
int	val_mask		/* IN: Value to assign to these bits */
)
{
int	prev_cw, new_cw;

	prev_cw = read_x87_FPU_control_word ();

	/* Form new register value from old. */
	new_cw = (prev_cw & ~dst_mask) | (val_mask & dst_mask);

	set_x87_FPU_control_word (new_cw);

	return (prev_cw);
}

/*
 * Execute the x87 FSTENV instruction.
 */

	static
	inline
	void
x87_fstenv_instruction (

int32u *	dst		/* IN/OUT: 7-word buffer to receive FPU	*/
				/*	   environment info		*/
)
{
	__asm volatile ("fstenv %0" : "=m" (*dst));
}

/*
 * Execute the x87 FLDENV instruction.
 */

	static
	inline
	void
x87_fldenv_instruction (

const int32u *	src		/* IN: 7-word buffer containing FPU	 */
				/*     environment info to load into FPU */
)
{
	__asm__ volatile ("fldenv %0" : : "m" (*src));
}

/*
 * Restore the x87 FPU state, being careful not to alter the tag field.
 */

	static
	inline
	void
restore_x87_fpu_state (

const int32u *	src		/* IN: 7-word buffer containing FPU	*/
				/*     environment info to restore	*/
)
{
struct x87_fpusave	temp1;
struct x87_fpusave	temp2;

	/* Get a local copy of the FPU state to restore. */
	memcpy (&temp1, src, 28);

	/* Get the current FPU state. */
	x87_fstenv_instruction ((int32u *) &temp2);

	/* DO NOT restore the tag word!  Must use current state!	*/
	temp1.tag	= temp2.tag;

	/* Compute new FPU status word: restore the exception flags,	*/
	/* stack fault and error summary status from saved state, but	*/
	/* all other bits are from the current FPU state.		*/
	temp1.sw	= ((temp1.sw & 0x00FF) | (temp2.sw & 0xFF00));

	x87_fldenv_instruction ((int32u *) &temp1);
}

#endif

#if USING_SSE_SSE2_FPU

/*
 * Inline routine to read the MXCSR Control/Status Register.
 */

	static
	inline
	int
read_mxcsr ()

{
volatile unsigned int	cw;

	/* Get current MXCSR value. */
	__asm volatile ("stmxcsr %0" : "=m" (*&cw));

	return (cw);
}

/*
 * Inline routine to store the given value into the MXCSR
 * Control/Status Register.  Previous value is forgotten.
 */

	static
	inline
	void
set_mxcsr (

int		value		/* IN: FPU Control Word value */
)
{
volatile unsigned int	cw;

	cw = value;

	__asm__ volatile ("ldmxcsr %0" : : "m" (*&cw));
}

/*
 * Inline routine to update the MXCSR Control/Status Word.
 * Returns the original state of the Control Word.
 */

	static
	inline
	int
update_mxcsr (

int	dst_mask,		/* IN: Bits to assign			*/
int	val_mask		/* IN: Value to assign to these bits	*/
)
{
int		prev_cw, new_cw;

	/* Get current MXCSR value. */
	prev_cw = read_mxcsr ();

	/* Form new register value from old. */
	new_cw = (prev_cw & ~dst_mask) | (val_mask & dst_mask);

	/* Put new value into MXCSR. */
	set_mxcsr (new_cw);

	return (prev_cw);
}

#endif

/*
 * On processors that support such a mode (currently only Intel x86):
 * 1 - return the current floating point configuration, including:
 *	- rounding control
 *	- precision
 *	- exception masks
 * 2 - force the current rounding mode to be "round to nearest",
 *     force the current floating point precision to "double" (returning
 *     the previous setting), and force the exceptions masks as follows:
 *	- IM: Invalid operation exception		Enabled
 *	- DM: Denormalized operand exception		Masked
 *	- ZM: Zero-divide exception			Enabled
 *	- OM: Overflow exception			Enabled
 *	- UM: Underflow exception			Masked
 *	- PM: Precision (inexact result) exception	Masked
 * 3 - restore a previously sampled floating point precision setting.
 */

	void
_gst_save_floating_point_configuration (

struct fpsave *		sp	/* IN - buffer to save FPU state into */
)
{
#if defined(PLATFORM_INTEL_32_BIT)
	/* This platform uses the x87 FPU for everything.	*/

	/* Save off the caller's FPU modes. */
	int32u * words = (int32u *) sp;
	x87_fstenv_instruction (&words [0]);
#elif defined (__x86_64__)
	/* The x86_64 architecture uses SSE/SSE2 instructions for	*/
	/* "float" and "double" arithmetic, but uses x87 instructions	*/
	/* for "long double".  Save both FPU's				*/

	/* Save off the caller's x87 FPU modes. */
	int32u * words = (int32u *) sp;
	x87_fstenv_instruction (&words [0]);

	/* Save off the caller's MXCSR (SSE/SSE2 FPU control/status).	*/
	words [7] = read_mxcsr ();
#endif
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
#if defined(PLATFORM_INTEL_32_BIT)
	/* This platform uses the x87 FPU for everything.	*/

	/* Save off the caller's FPU modes. */
	int32u * words = (int32u *) sp;
	x87_fstenv_instruction (&words [0]);

	/* Intel 32-bit x87 FPU					*/
	/*							*/
	/* Modes:						*/
	/*   _FPU_DOUBLE | _FPU_RC_NEAREST			*/
	/* | _FPU_MASK_DM | _FPU_MASK_UM | _FPU_MASK_PM		*/

	set_x87_FPU_control_word (0x0232);
#elif defined (__x86_64__)
	/* The x86_64 architecture uses SSE/SSE2 instructions for	*/
	/* "float" and "double" arithmetic, but uses x87 instructions	*/
	/* for "long double".  We can therefore leave the x87 FPU	*/
	/* set permanently to the "extended" precision setting, and	*/
	/* there is no need to enable/disable "long double" arithmetic.	*/

	/* Save off the caller's FPU modes. */
	int32u * words = (int32u *) sp;
	x87_fstenv_instruction (&words [0]);

	/* Set modes in the x87 FPU (used only for long double):	*/
	/*	     							*/
	/*   _FPU_EXTENDED | _FPU_RC_NEAREST				*/
	/* | _FPU_MASK_DM  | _FPU_MASK_UM | _FPU_MASK_PM		*/

	set_x87_FPU_control_word (0x0332);

	/* Set modes in the MXCSR (controls SSE/SSE2, used for double,	*/
	/* float):							*/
	/*								*/
	/* Flush to Zero:	off (turning on loses IEEE 754		*/
	/*			    compatibility)			*/
	/* Rounding Control:	Round-to-nearest "even"			*/
	/* Denormals are Zero:	off (turning on loses IEEE 754		*/
	/*			    compatibility)			*/
	/* Exceptions:							*/
	/*	Precision:	disabled				*/
	/*	Underflow:	disabled				*/
	/*	Overflow:	enabled					*/
	/*	Divide-by-zero:	enabled					*/
	/*	Denormal:	disabled				*/
	/*	Invalid op:	enabled					*/
	{ int prev_mxcsr;
		prev_mxcsr = update_mxcsr (0x0000FFFF, 0x00001900);
		/* Use the 8th doubleword to store MXCSR. */
		((int32u *) sp) [7] = prev_mxcsr;
	}
#endif
}


	void
_gst_restore_floating_point_configuration (

struct fpsave *		sp	/* IN - buffer to restore FPU state from */
)
{
#if defined(PLATFORM_INTEL_32_BIT)
	/* This platform uses the x87 FPU for everything.	*/

	/* Restore the caller's FPU modes. */
	int32u * words = (int32u *) sp;
	x87_fldenv_instruction (&words [0]);
#elif defined(__x86_64__)
	/* The x86_64 architecture uses SSE/SSE2 instructions for	*/
	/* "float" and "double" arithmetic, but uses x87 instructions	*/
	/* for "long double".  Restore both FPU's			*/

	/* Restore the caller's x87 FPU modes. */
	int32u * words = (int32u *) sp;
	restore_x87_fpu_state (&words [0]);

	/* Restore the caller's MXCSR (SSE/SSE2 FPU control/status).	*/
	set_mxcsr (words [7]);
#endif
}

/*
 * On 32-bit Intel CPUs, all floating-point arithmetic is performed
 * using the x87 FPU.  Since most arithmetic is "double", we set
 * the "Precision Control" field to "double" (instead of the default
 * of "extended") to eliminate most of the problems with the FPU
 * registers having more precision than actual operands in memory.
 *
 * This means that when we actually *want* to do "long double" arithmetic,
 * we must explicitly "enable" the FPU to do this by setting the
 * "Precision Control" field back to "extended", and then restore it
 * when we are done.  This is the purpose of the next two routines.
 *
 * On 64-bit CPUs, all float/double arithmetic is performed using the
 * SSE/SSE2 FPU, and the X87 FPU is used only for "long double."
 * There is therefore no need to do anything special in order to do
 * "long double" arithmetic -- the compiler simply uses x87 opcodes
 * and we leave the x87 FPU alone.
 */

	int
_gst_enable_long_double_precision ()

{
#if defined(PLATFORM_INTEL_32_BIT)
	/* Intel 32-bit / x87 architecture. */

	/* Force Precision Control to "extended" precision (64-bit). */
	int prev;
	prev = update_x87_FPU_control_word (0x0300, 0x0300);

	return (prev);
#elif defined(PLATFORM_INTEL_64_BIT)
	/* x87_64 architecture: long double is always enabled in x87 FPU */
	/* because this FPU is used only for long double arithmetic.	 */
	/* No need to enable/disable.					 */

	/* Muzzle compiler diagnostic regarding unreferenced function. */
	(void) update_x87_FPU_control_word;

	return (0);
#else
	#error "Unsupported FPU architecture!"
	return (0);
#endif
}

	void
_get_restore_long_double_precision (

int		prevState
)
{
#if defined(PLATFORM_INTEL_32_BIT)
	/* Intel 32-bit, x87 FPU. */
	/* Restore the Precision Control field. */
	update_x87_FPU_control_word (0x0300, prevState);
#elif defined(PLFORM_INTEL_64_BIT)
	/* x87_64 architecture: long double is always enabled in x87 FPU */
	/* because this FPU is used only for long double arithmetic.	 */
	/* No need to enable/disable.					 */
#else
	/* Not x87 / x87 architecture.  Do nothing. */
#endif
}

#endif
