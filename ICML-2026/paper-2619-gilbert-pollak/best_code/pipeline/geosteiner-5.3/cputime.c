/***********************************************************************

	$Id: cputime.c,v 1.17 2022/11/19 13:45:50 warme Exp $

	File:	cputime.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	This file contains routines for computing CPU time
	consumption for the program.

************************************************************************

	Modification Log:

	a-1:	04/16/93	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Add new interfaces to encapsulate conversions
		:  to/from seconds to cpu_time_t units.
		: Fix -Wall issues.  Upgrade fatals.

************************************************************************/

#include "cputime.h"

#include "config.h"
#include "environment.h"
#include "fatal.h"
#include "logic.h"
#include "steiner.h"

#ifdef UNIX_CPU_TIME
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>
#endif

#include <math.h>
#include <time.h>


/*
 * Global Routines
 */

void		_gst_convert_cpu_time (cpu_time_t time, char * out_buf);
void		_gst_convert_delta_cpu_time (char * buf, cpu_time_t * Tn);
double		_gst_cpu_time_t_to_double_seconds (cpu_time_t ticks);
cpu_time_t	_gst_double_seconds_to_cpu_time_t (double seconds);
cpu_time_t	_gst_get_cpu_time (void);
cpu_time_t	_gst_get_delta_cpu_time (cpu_time_t * Tn);
void		_gst_initialize_cpu_time (struct environment * env);
cpu_time_t	_gst_int_seconds_to_cpu_time_t (unsigned int seconds);

/*
 * Define API_CLOCKS_PER_SECOND to be the tick rate presented to the
 * program by the chosen API.
 */

#ifdef UNIX_CPU_TIME
 #define API_CLOCKS_PER_SECOND	sysconf(_SC_CLK_TCK) /* The Posix value */
#else
 #define API_CLOCKS_PER_SECOND	CLOCKS_PER_SEC	/* The ANSI C value */
#endif

/*
 * This routine gets the current amount of CPU time that the program has
 * used.  It is returned in a format we define that is independent of the
 * operating system (i.e. hundreth's of a second).
 *
 * Because the times we are measuring can be quite long (several days),
 * we can run into overflow problems if the operating system gives us
 * a CPU timer with too high a resolution (i.e. microseconds).  This
 * code can be compiled to use either the standard interfaces defined
 * by ANSI C, or the native UNIX interface.  Choose the one that has
 * a resolution closest to our own TICKS_PER_SEC units.
 *
 * The basic conversion formula is:
 *
 *			   sys_ticks * our_ticks_per_sec    1
 *	our_ticks = floor (----------------------------- + ---)
 *				  sys_ticks_per_sec	    2
 *
 * which can be re-written
 *
 *			   2 * sys_ticks * our_tps + sys_tps
 *	our_ticks = floor (---------------------------------)
 *				     2 * sys_tps
 *
 * We compute this using one of several methods, depending upon the
 * relationship between sys_ticks_per_sec and our_ticks_per_sec.
 * The idea of it all is to minimize the chance of arithmetic overflow.
 */
	void
_gst_initialize_cpu_time (

struct environment *	env	/* IN/OUT - GeoSteiner environment */
)
{
clock_t		clocks_per_sec;
int32u		Q, R, method;

	/* On some systems this is a system call.  Do only once! */
	clocks_per_sec = API_CLOCKS_PER_SECOND;

	/* Now pick the conversion method and parameters. */
	if (TICKS_PER_SEC >= clocks_per_sec) {
		Q = TICKS_PER_SEC / clocks_per_sec;
		R = TICKS_PER_SEC % clocks_per_sec;
		method = (R EQ 0) ? 0 : 1;
	}
	else {
		R = clocks_per_sec % TICKS_PER_SEC;
		if (R EQ 0) {
			Q = clocks_per_sec / TICKS_PER_SEC;
			method = 2;
		}
		else {
			Q = 0;
			method = 3;
		}
	}

	env -> clocks_per_sec = clocks_per_sec;
	env -> Q = Q;
	env -> R = R;
	env -> method = method;
}

	cpu_time_t
_gst_get_cpu_time (void)

{
clock_t			total;
clock_t			clocks_per_sec;
int32u			seconds;
int32u			ticks;
cpu_time_t		cpu_time;
struct environment *	env;

#ifdef UNIX_CPU_TIME
	{ struct tms	t;

		times (&t);

		total	= t.tms_utime  + t.tms_stime
			+ t.tms_cutime + t.tms_cstime;
	}
#else
	/* Using the ANSI C defined interface... */
	total = clock ();
#endif

	env = gst_env;

	clocks_per_sec = env -> clocks_per_sec;

	seconds	= total / clocks_per_sec;
	ticks	= total % clocks_per_sec;

	/* Convert the fractions of a second ticks into .01 second units. */
	switch (env -> method) {
	case 0:
		ticks *= env -> Q;
		break;

	case 1:
		ticks = ticks * env -> Q
			+ (2 * ticks * env -> R + clocks_per_sec) /
			  (2 * clocks_per_sec);
		break;

	case 2:
		ticks = (2 * ticks + env -> Q) / (2 * env -> Q);
		break;

	case 3:
		ticks = ((2 * TICKS_PER_SEC) * ticks + clocks_per_sec)
			/ (2 * clocks_per_sec);
		break;

	default:
		FATAL_ERROR;
		break;
	}

	cpu_time = seconds * TICKS_PER_SEC + ticks;

	return (cpu_time);
}

/*
 * This routine will convert the given CPU time into a printable
 * null-terminated ASCII string.  The answer contains two decimal places.
 */

	void
_gst_convert_cpu_time (

cpu_time_t	time,
char *		out_buf
)
{
cpu_time_t	secs;
cpu_time_t	ticks;
char *		p;
char		buf [20];

#define ZERO	((cpu_time_t) 0)
#define TEN	((cpu_time_t) 10)

	secs	= time / TICKS_PER_SEC;
	ticks	= time % TICKS_PER_SEC;

	p = &buf [0];
	if (secs <= ZERO) {
		*p++ = '0';
	}
	else {
		while (secs > ZERO) {
			*p++ = (secs % TEN) + '0';
			secs /= TEN;
		}
	}
	while (p > &buf [0]) {
		*out_buf++ = *--p;
	}
	*out_buf++ = '.';
	ticks *= TEN;
	*out_buf++ = (ticks / TICKS_PER_SEC) + '0';
	ticks %= TICKS_PER_SEC;
	ticks *= TEN;
	*out_buf++ = (ticks / TICKS_PER_SEC) + '0';
	*out_buf++ = '\0';

#undef ZERO
#undef TEN
}

/*
 * Compute the CPU time used since the last time we called this routine.
 */

	cpu_time_t
_gst_get_delta_cpu_time (

cpu_time_t *	Tn
)
{
cpu_time_t		now;
cpu_time_t		delta;

	now = _gst_get_cpu_time ();

	delta = now - *Tn;

	*Tn = now;

	return (delta);
}

/*
 * Compute and format the CPU time used since the last time we called
 * this routine.
 */

	void
_gst_convert_delta_cpu_time (

char *		buf,		/* OUT - ASCII time string, CPU seconds */
cpu_time_t *	Tn
)
{
cpu_time_t	delta;

	delta = _gst_get_delta_cpu_time (Tn);
	_gst_convert_cpu_time (delta, buf);
}

/*
 * Convert an integer duration (in seconds) to the corresponding
 * cpu_time_t duration.
 */

	cpu_time_t
_gst_int_seconds_to_cpu_time_t (

unsigned int		seconds		/* IN: duration in seconds */
)
{
	return (seconds * TICKS_PER_SEC);
}


/*
 * Convert a double duration (in seconds) to the corresponding
 * cpu_time_t duration.
 */

	cpu_time_t
_gst_double_seconds_to_cpu_time_t (

double			seconds		/* IN: duration in seconds */
)
{
double		fticks;
	fticks = seconds * TICKS_PER_SEC;
	fticks = floor (fticks + 0.5);

	/* Must fit in unsigned 32-bit integer. */
	FATAL_ERROR_IF (fticks > 4294967295.0);

	return ((cpu_time_t) fticks);
}

/*
 * Convert the given cpu_time_t value into seconds as a "double" value.
 */

	double
_gst_cpu_time_t_to_double_seconds (

cpu_time_t		ticks		/* IN: tick value to convert */
)
{
double		dbl_ticks, seconds;

	dbl_ticks = ticks;

	seconds = dbl_ticks / TICKS_PER_SEC;

	return (seconds);
}
