/***********************************************************************

	$Id: environment.h,v 1.27 2023/04/03 17:45:35 warme Exp $

	File:	environment.h
	Rev:	e-4
	Date:	04/03/2023

	Copyright (c) 2002, 2023 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************



************************************************************************

	Modification Log:

	a-1:	04/30/02	benny
		: Created.
	b-1:	02/02/2014	warme
		: Removed unnecessary include file.
	b-2:	02/19/2015	warme
		: Add include file to get CPXENVptr.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
	e-4:	04/03/2023	warme
		: Added table initialization routine.

************************************************************************/

#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "config.h"
#include "gsttypes.h"
#include <time.h>

#include "lpsolver.h"	/* Needed for CPXENVptr */

struct gst_parmdefs;

struct environment {
	int		opencount;	/* Open count for GeoSteiner */
	bool		solver_open;	/* TRUE iff solver is open */
	int		solver_refcount; /* Reference count for LP solver */
	char *		machine_string;	/* String describing the machine */
	bool		fp_saved;
#ifdef CPLEX
	int		cplex_status;	/* CPLEX attached/unattached */
 #if CPLEX >= 40
	CPXENVptr	_cplex_env;	/* CPLEX environment pointer */
 #endif
#endif
	/* Two arrays to speed up the task of iterating over	*/
	/* all "1" bits in a bit mask.  This is used by the	*/
	/* backtrack search.					*/
	int8u *		one_bits_in_byte [256 + 1];
	int8u		one_bits_in_byte_array [1024];

	/* Some variables used for timing. */
	clock_t		clocks_per_sec;
	int32u		Q, R, method;

	/* Access to parameter definition info. */
	struct gst_parmdefs *	parmdefs;
};

extern struct environment * gst_env;

extern void	_gst_begin_using_lp_solver (void);
extern void	_gst_stop_using_lp_solver (void);

extern void	_gst_initialize_one_bits_in_byte_tables (void);

#endif /* ENVIRONMENT_H */
