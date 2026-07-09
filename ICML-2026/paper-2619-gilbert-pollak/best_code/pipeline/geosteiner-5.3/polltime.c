/***********************************************************************

	$Id: polltime.c,v 1.8 2022/11/19 13:45:54 warme Exp $

	File:	polltime.c
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
		: Reporganize include files, apply prefixes.

************************************************************************/

#include "polltime.h"

#include "logic.h"

#define POLL_FREQUENCY	10	/* Seconds between polls */


#ifdef DEBUG_CPU_POLL /* Debug output*/
 #define cpuprintf(a,b) printf("%% CPU: "); printf(a,b)
#else
 #define cpuprintf(a,b)
#endif

/*
 * This function polls the CPU and compares it to a specified wall time.
 * Using a counting scheme it tries to only poll the CPU every
 * POLL_FREQUENCY seconds.
 */

	bool
_gst_poll_cpu (

struct cpu_poll *	poll		/* IN - CPU polling structure */
)
{
cpu_time_t		now;
cpu_time_t		diff_polls;	/* Time between polls */
cpu_time_t		time_to_end;	/* Time to wall time */

	++(poll -> iteration);
	cpuprintf ("Polling cpu:\t\t%s\n", poll -> name);
	cpuprintf (" Iteration:\t\t%d\n", poll -> iteration);
	cpuprintf (" Frequency:\t\t%d\n", poll -> frequency);

	if (poll -> iteration >= poll -> frequency) {
		poll -> iteration = 0;
		now = _gst_get_cpu_time ();

		cpuprintf (" Time now:\t\t%d\n", now);
		if (now > poll -> end_time) {
			cpuprintf (" Wall time passed:\t%d\n", poll -> end_time);
			return TRUE;
		}

		diff_polls = now - poll -> last;
		time_to_end = poll -> end_time - now;
		cpuprintf (" Time passed:\t\t%d\n", diff_polls);
		cpuprintf (" Time to end:\t\t%d\n", time_to_end);
		if (diff_polls < (time_to_end >> 2)) {
			poll -> frequency <<= 1;
			cpuprintf (" Increased frequency:\t%d\n", poll -> frequency);
		}
		else if (diff_polls > (time_to_end >> 1)) {
			poll -> frequency = (time_to_end / diff_polls) >> 1;
			if (poll -> frequency < 1) {
				poll -> frequency = 1;
			}
			cpuprintf (" Computed frequency:\t%d\n", poll -> frequency);
		}
		poll -> last = now;
	}

	return FALSE;
}
