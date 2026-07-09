/***********************************************************************

	$Id: cra.c,v 1.11 2022/11/19 13:45:50 warme Exp $

	File:	cra.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Closest Rational Approximation.

************************************************************************

	Modification Log:

	a-1:	08/09/96	warme
		: Created.
	a-2:	07/28/2000	warme
		: Added banner and improved packaging.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#include "cra.h"

#include "logic.h"
#include <math.h>
#include <stdio.h>


/*
 * Global Routines
 */

double		_gst_cra (double	z,
			  double *	numout,
			  double *	denout);


/*
 * Local Constants
 */

#ifdef __STANDALONE
 #define CRA_PRINT	5
#else
 #define CRA_PRINT	0
#endif	/* __STANDALONE */

#define CRA_FUZZ	1.0e-9


/*
 * Local Types
 */

struct cf {
	double		xi;
	struct cf *	next;
};


/*
 * Local Routines
 */

static void		cra_recurse (double		orig_z,
				     double		frac,
				     struct cf *	cfp,
				     double *		numout,
				     double *		denout);

/*
 * Closest Rational Approximation to the given floating point number.
 * Perform continued fraction expansion on the number.  Each time a
 * new term is added to the expansion, see if the result is close
 * enough to the original.  Once the CF expansion is close enough,
 * we are done.
 */

	double
_gst_cra (

double		z,
double *	numout,
double *	denout
)
{
double		sign, num, den, frac;
struct cf	cf;

	sign = 1.0;
	if (z < 0.0) {
		sign = -1.0;
		z = -z;
	}

	if (z <= CRA_FUZZ) return (0.0);

	frac = fmod (z, 1.0);

#if CRA_PRINT >= 1
printf ("Initial fraction: %24.15g\n", frac);
#endif

	/* First term of CF expansion is the integer part. */

	cf.xi	= z - frac;
	cf.next = NULL;

	/* Perform rest of CF expansion. */

	cra_recurse (z, frac, &cf, &num, &den);

#if CRA_PRINT >= 1
	printf ("%24.15g ~ %24.15g = %f / %f\n",
		z, sign * num / den, sign * num, den);
#endif

	if (numout NE NULL) {
		*numout = sign * num;
	}
	if (denout NE NULL) {
		*denout = den;
	}

	return (sign * num / den);
}

/*
 * Recursive portion of the closest rational approximation code.
 * We represent the continued fraction expansion using a linked list.
 * Rather than allocating these from the heap, we allocate them on
 * the stack and simply recurse deeper to allocate a new one.
 *
 * It is indeed unfortunate that to reevaluate the newly extended
 * CF expansion, it is necessary to start all over again.  Were
 * there a simple way to incrementally update the numerator and
 * denominator values, then this would not be necessary.
 */

	static
	void
cra_recurse (

double		orig_z,		/* IN - original Z */
double		frac,		/* IN - 0 <= frac < 1 */
struct cf *	cfp,		/* IN - continued fraction expansion */
double *	numout,		/* OUT - numerator */
double *	denout		/* OUT - denominator */
)
{
double		num, den, a, err, inv, tmp;
struct cf *	p;
struct cf	cf;

	/* Unravel the current continued fraction into a simple	*/
	/* numerator and denominator				*/

	num = 1.0;
	den = 0.0;
	for (p = cfp; p != NULL; p = p -> next) {
		tmp = den + (num * p -> xi);
		den = num;
		num = tmp;
	}
	*numout = num;
	*denout = den;

	a = num / den;

#if CRA_PRINT >= 2
	printf ("%24.15g ~ %24.15g = %f / %f\n", orig_z, a, num, den);
#endif

	err = (orig_z - a) / orig_z;
	if (fabs (err) <= CRA_FUZZ) return;

	inv = 1.0 / frac;
	frac = fmod (inv, 1.0);

	cf.xi	= inv - frac;
	cf.next	= cfp;

	cra_recurse (orig_z, frac, &cf, numout, denout);
}

/*
 * Main program for use when testing this routine in a stand-alone fashion.
 */

#ifdef	__STANDALONE

main ()

{
double		a, b;

	for (;;) {
		scanf (" %lg", &a);
		b = _gst_cra (a);

		printf ("%24.15g ===> %24.15g\n", a, b);
	}
}

#endif
