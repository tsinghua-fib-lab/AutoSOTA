/***********************************************************************

	$Id: cputime.h,v 1.6 2022/11/19 13:45:50 warme Exp $

	File:	cputime.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Facilities for measuring CPU time usage.

************************************************************************

	Modification Log:

	a-1:	09/19/2004	warme
		: Split off into separate file.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Add new iterfaces to encapsulate conversions
		:  to/from seconds to cpu_time_t units.

************************************************************************/

#ifndef CPUTIME_H
#define	CPUTIME_H

#include "gsttypes.h"

struct environment;

/*
 * Type used to represent CPU time usage.
 */

typedef int32u		cpu_time_t;

/*
 * This defines the tick rate that our cpu_time_t type uses.
 */

#define	TICKS_PER_SEC	100		/* This is the units WE use! */

extern void		_gst_convert_cpu_time (cpu_time_t, char *);
extern void		_gst_convert_delta_cpu_time (char *, cpu_time_t *);
extern double		_gst_cpu_time_t_to_double_seconds (cpu_time_t ticks);
extern cpu_time_t	_gst_double_seconds_to_cpu_time_t (double seconds);
extern cpu_time_t	_gst_get_cpu_time (void);
extern cpu_time_t	_gst_get_delta_cpu_time (cpu_time_t *);
extern void		_gst_initialize_cpu_time (struct environment *);
extern cpu_time_t	_gst_int_seconds_to_cpu_time_t (unsigned int seconds);

#endif
