/***********************************************************************

	$Id: polltime.h,v 1.8 2022/11/19 13:45:54 warme Exp $

	File:	polltime.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1996, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Facilities to enforce time limits via polling.

************************************************************************

	Modification Log:

	b-1:	: 08/05/2002	benny
		: Initial version.
	b-2:	: 06/16/2003	warme
		: Split off into separate file.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Apply prefixes.

************************************************************************/

#ifndef POLLTIME_H
#define	POLLTIME_H

#include "cputime.h"


/* This will generate debug output for CPU polling */
/* #define DEBUG_CPU_POLL */


/*
 * Structure used for CPU and wall time polling.
 */

struct cpu_poll {
	cpu_time_t	last;		/* Time at last CPU poll */
	cpu_time_t	end_time;	/* Wall time */
	int		frequency;	/* Iterations between polls */
	int		iteration;	/* Number of iterations done */
#ifdef DEBUG_CPU_POLL
	char *		name;
#endif
};


/*
 * Extern declarations
 */

extern bool		_gst_poll_cpu (struct cpu_poll * poll);


/*
 * Macro to test the time limit.
 */

#define TIME_LIMIT_EXCEEDED(cpu_time_limit, poll)	\
	((cpu_time_limit NE 0) AND _gst_poll_cpu (poll))

#endif
