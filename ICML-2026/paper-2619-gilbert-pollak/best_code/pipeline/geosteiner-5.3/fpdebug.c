/***********************************************************************

	$Id: fpdebug.c,v 1.9 2022/11/19 13:45:51 warme Exp $

	File:	fpdebug.c
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
	b-1:	02/02/2014	warme
		: Added an include file.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.

************************************************************************/

#include "fpdebug.h"

#include "fatal.h"
#include "logic.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__) && (defined(__i386__) OR defined(__i386))
#include <ucontext.h>
#endif

static void handleSIGFPE(int code, siginfo_t* info, void*);

/*
 * If we are on an x86/Linux architecture, install a signal handler for
 * floating point exceptions that prints detailed information about the
 * FPU state at the time the exception was thrown.
 *
 * NOTE: This routine should never be called from library code because
 * the library must never establish or alter any signal handlers!
 */

	void
_gst_establish_debug_floating_point_signal_handler ()

{
int			i;
struct sigaction	actionDefault;
struct sigaction	actionSIGFPE;

	/* install a signal handler that reports the type of floating	*/
	/* point exception encountered					*/

	memset(&actionSIGFPE, 0, sizeof(actionSIGFPE));
	memset(&actionDefault, 0, sizeof(actionDefault));

	/* Get the default action for SIGFPE and momentarily install	*/
	/* a NULL action.						*/
	i = sigaction(SIGFPE, &actionSIGFPE, &actionDefault);
	FATAL_ERROR_IF (i NE 0);

	/* Copy the default action into the customized action. */
	memcpy(&actionSIGFPE, &actionDefault, sizeof(actionSIGFPE));

	/* Set the new SIGFPE handler. */
	actionSIGFPE.sa_sigaction = handleSIGFPE;
	actionSIGFPE.sa_flags	  = SA_SIGINFO;

	/* Install the new action. */
	i = sigaction(SIGFPE, &actionSIGFPE, NULL);
	FATAL_ERROR_IF (i NE 0);
}

/*
 * Signal handler for floating point exceptions.
 */

	static
	void
handleSIGFPE(

int		code,
siginfo_t*	info,
void*		data
)
{
#if defined(__linux__) AND (defined(__i386__) OR defined(__i386))
int			i;
int			j;
int			k;
int			top;
int			phyreg;
struct ucontext *	uc;
fpregset_t		fp;
struct _libc_fpreg *	r;
#endif

	fprintf (stderr, "\n\nFloating Point Exception: ");

	if (info EQ NULL) {
		fprintf (stderr, "unspecified SIGFPE\n");
		abort ();
	}

	switch (info->si_code) {
#ifdef FPE_INTDIV
	case FPE_INTDIV:
		fprintf (stderr, "integer divide by zero\n");
		break;
#endif

#ifdef FPE_INTOVF
	case FPE_INTOVF:
		fprintf (stderr, "integer overflow\n");
		break;
#endif

#ifdef FPE_FLTDIV
	case FPE_FLTDIV:
		fprintf (stderr, "floating point divide by zero\n");
		break;
#endif

#ifdef FPE_FLTOVF
	case FPE_FLTOVF:
		fprintf (stderr, "floating point overflow\n");
		break;
#endif

#ifdef FPE_FLTUND
	case FPE_FLTUND:
		fprintf (stderr, "floating point underflow\n");
		break;
#endif

#ifdef FPE_FLTRES
	case FPE_FLTRES:
		fprintf (stderr, "floating point inexact result\n");
		break;
#endif

#ifdef FPE_FLTINV
	case FPE_FLTINV:
		fprintf (stderr, "floating point invalid operation\n");
		break;
#endif

	default:
		fprintf (stderr, "unknown error code!\n");
	}

#if defined(__linux__) AND (defined(__i386__) OR defined(__i386))
	/* This code is specific to the x86-linux configuration... */
	uc = (struct ucontext *) data;
	fp = uc -> uc_mcontext.fpregs;

	fprintf (stderr,
		 "Exception thrown by instruction at address 0x%08lx,\n"
		 "referencing operand at address 0x%08lx\n"
		 "Floating Point Register Stack:\n",
		 fp -> ipoff,
		 fp -> dataoff);

	top = (fp -> sw >> 11) & 0x07;
	for (i = 0; i < 8; i++) {
		static const char * tags [] = {
			"valid", "zero", "invalid/infin", "empty"
		};
		phyreg = (top + i) & 0x07;
		r = &(fp -> _st [phyreg]);
		j = (r->significand[3] >> 15) & 0x01;
		fprintf (stderr,
			 "  FP%c: Mantissa=%c%c.",
			 '0' + phyreg,
			 (((r->exponent & 0x8000) NE 0) ? '-' : '+'),
			 '0' + j);

		for (k = 3; k >= 0; k--) {
			j = ((r -> significand[k] << 1) & 0xFFFE)
			    |
			    ((k <= 0) ? 0
				      : ((r->significand[k-1]>>15) & 0x0001));
			fprintf (stderr, "%04X", j);
		}
		j = r -> exponent & 0x7FFF;
		fprintf (stderr,
			 ", Exponent=%04X (-3FFF = %04X) (%s)\n",
			 j,
			 (j - 0x3FFF) & 0x7FFF,
			 tags [(fp->tag >> (2*phyreg)) & 0x03]);
	}
#endif

	/* Floating-point exception. */
	FATAL_ERROR;
}
