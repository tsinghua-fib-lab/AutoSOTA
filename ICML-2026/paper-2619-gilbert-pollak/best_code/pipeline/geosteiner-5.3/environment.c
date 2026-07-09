/***********************************************************************

	$Id: environment.c,v 1.34 2023/04/03 17:43:34 warme Exp $

	File:	environment.c
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
		: Removed unnecessary include files.
	b-2:	04/24/2014	warme
		: Add include file for LP solver startup/shutdown.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.
	e-4:	04/03/2023	warme
		: Added table initialization routine.

************************************************************************/

#define	_GNU_SOURCE

#include "environment.h"

#include "config.h"
#include "cputime.h"
#include "fatal.h"
#include "geosteiner.h"
#include "logic.h"
#include "lpinit.h"
#include "machine.h"
#include "memory.h"
#include "parms.h"
#include "prepostlude.h"
#include "steiner.h"
#include <string.h>

#ifndef CPLEX
#include "lp_solve_2.3/patchlevel.h"
#endif

#ifdef NEED_CTYPE_C
 /* Need to work around old versions of CPLEX that reference stuff in	*/
 /* <ctype.h> that newer glibc's do not define.				*/
 #include "ctype.c"
#endif

/*
 * Global Routines
 */

int		gst_open_geosteiner (void);
int		gst_close_geosteiner (void);
const char *	gst_version_string (void);
int		gst_version (void);
int		gst_open_lpsolver (void);
int		gst_close_lpsolver (void);
const char *	gst_lpsolver_version_string (void);

#ifdef CPLEX
void		gst_attach_cplex (CPXENVptr envp);
CPXENVptr	gst_detach_cplex (void);
#endif


/*
 * Global Variables
 */

struct environment *	gst_env;

enum { CPLEX_UNATTACHED, CPLEX_ATTACHED };


/*
 * Local Routines
 */

static void	free_environment (struct environment *);

/*
 * Define macros to implement various portions of the opening /
 * initialization of GeoSteiner.
 */

#ifdef CPLEX
 #if CPLEX >= 40
  #define INIT_LP_SOLVER(p) \
	{	p -> cplex_status	= CPLEX_UNATTACHED;	\
		p -> _cplex_env		= NULL;			\
	}
 #else
  #define INIT_LP_SOLVER(p) \
	{	p -> cplex_status	= CPLEX_UNATTACHED;	}
 #endif
#else
 #define INIT_LP_SOLVER(p)
#endif

/*
 * Transition the GeoSteiner environment from "closed" to "open".
 */

	int
gst_open_geosteiner (void)
{
int			code;
struct environment *	p;

	code = 0;
	p = gst_env;
	if (p EQ NULL) {
		p = NEW (struct environment);
		memset (p, 0, sizeof (*p));

		/* Setup the environment */
		p -> solver_open	= FALSE;
		p -> solver_refcount	= 0;
		p -> machine_string	= _gst_get_machine_string ();
		p -> fp_saved		= FALSE;

		INIT_LP_SOLVER (p);
		_gst_initialize_cpu_time (p);
		_gst_initialize_parameters (p);

		p -> opencount = 1;
		gst_env = p;
	}
	else if (p -> opencount <= 0) {
		FATAL_ERROR;
	}
	else {
		++(p -> opencount);
	}

	return code;
}

	int
gst_close_geosteiner (void)
{
struct environment *	env;

	env = gst_env;
	if ((env EQ NULL) OR (env -> opencount <= 0)) {
		return GST_ERR_ALREADY_CLOSED;
	}
	else if (env -> opencount EQ 1) {
		if ((env -> solver_open) AND
		    (env -> solver_refcount > 0)) {
			return GST_ERR_LP_SOLVER_ACTIVE;
		}
		gst_close_lpsolver ();

		gst_env = NULL;
		/* GeoSteiner is now closed */

		free_environment (env);
		env -> opencount = 0;
		free (env);
	}
	else {
		--(env -> opencount);
	}

	return 0;
}

/*
 * Free up an environment.  It might not be completely constructed.
 */

	static
	void
free_environment (

struct environment *	p	/* IN - environment to free */
)
{
struct gst_parmdefs *	pdefs;

	pdefs = p -> parmdefs;
	p -> parmdefs = NULL;
	if (pdefs NE NULL) {
		_gst_shutdown_parameters (pdefs);
	}
	free (p -> machine_string);
	p -> machine_string = NULL;
}

/*
 * Explicitly open the lp solver.
 */

	int
gst_open_lpsolver (void)
{

	GST_PRELUDE

	if (NOT (gst_env -> solver_open)) {
		_gst_startup_lp_solver ();

		gst_env -> solver_open = TRUE;
	}
	GST_POSTLUDE
	return 0;
}

/*
 * Explicitly close the lp solver.
 */

	int
gst_close_lpsolver (void)
{
int	res;

	GST_PRELUDE

	res = 0;

	if (gst_env -> solver_open) {
		if (gst_env -> solver_refcount > 0) {
			res = GST_ERR_LP_SOLVER_ACTIVE;
		}
		else {
#ifdef CPLEX
			if (gst_env -> cplex_status EQ CPLEX_ATTACHED) {
				gst_detach_cplex ();
			}
			else {
				_gst_shutdown_lp_solver ();
			}
#else
			_gst_shutdown_lp_solver ();
#endif
			gst_env -> solver_open = FALSE;
		}
	}

	GST_POSTLUDE
	return res;
}

/*
 * Start using the LP solver.  We open it if necessary, and increment
 * the reference count in any case.
 */

	void
_gst_begin_using_lp_solver (void)

{
	/* This prevents it from being closed... */
	++(gst_env -> solver_refcount);

	gst_open_lpsolver ();
}

/*
 * Stop using the LP solver.  This just decrements the ref count.
 * Even if the reference count goes to zero, we keep the solver open
 * so that we don't have to start it up for the next problem.  Basically
 * the (refcount EQ 0) state means that it is OK to close the LP solver.
 */

	void
_gst_stop_using_lp_solver (void)

{
	if (--(gst_env -> solver_refcount) < 0) {
		FATAL_ERROR;
	}
}

/*
 * Returns the version of GeoSteiner as a string.
 */

	const char *
gst_version_string (void)
{

	GST_PRELUDE

	GST_POSTLUDE
	return (GEOLIB_VERSION_STRING);
}

/*
 * Returns the version as an integer, XXXYYYZZZ, where XXX is
 * the major version, YYY is the minor version and ZZZ is the patchlevel.
 */
	int
gst_version (void)
{
int	version;

	GST_PRELUDE

	version = 1000000 * GEOLIB_VERSION_MAJOR
		+    1000 * GEOLIB_VERSION_MINOR
		+	    GEOLIB_VERSION_PATCH;

	GST_POSTLUDE
	return (version);
}

/*
 * Returns the version of the lp solver as a string.
 */
	const char *
gst_lpsolver_version_string (void)
{
char *	version_string;

	GST_PRELUDE

#ifdef CPLEX
	version_string = "CPLEX " CPLEX_VERSION_STRING;
#else
	version_string = "lp_solve " PATCHLEVEL;
#endif

	GST_POSTLUDE
	return (version_string);
}

#if defined(CPLEX) AND (CPLEX >= 40)
	void
gst_attach_cplex (

CPXENVptr	envp	/* IN - CPLEX environment pointer */
)
{

	GST_PRELUDE

#ifdef CPLEX
	if (envp EQ NULL) {
		gst_detach_cplex ();
	}
	else {
		gst_env -> _cplex_env	= envp;
		gst_env -> cplex_status	= CPLEX_ATTACHED;
		gst_env -> solver_open	= TRUE;
	}
#endif
#ifdef LPSOLVE
	/* GeoSteiner is compiled to use lp_solve. */
	FATAL_ERROR;
#endif

	GST_POSTLUDE
}
#endif

#if defined(CPLEX) AND (CPLEX >= 40)
	CPXENVptr
gst_detach_cplex (void)
{
CPXENVptr tmp;

	GST_PRELUDE

#ifdef CPLEX
	tmp = gst_env -> _cplex_env;
	gst_env -> _cplex_env = NULL;
	gst_env -> cplex_status = CPLEX_UNATTACHED;
	gst_env -> solver_open = FALSE;
#endif
#ifdef LPSOLVE
	/* GeoSteiner is compiled to use lp_solve. */
	FATAL_ERROR;
#endif

	GST_POSTLUDE
	return tmp;
}
#endif


/* For debugging purposes... */
#if 1
void _gst_fp_saved()
{
	if (gst_env -> fp_saved) {
		fprintf(stderr, "FPU not restored!!\n");
	}
}
#endif

/*
 * Assure that the one-bits-in-byte tables have been initialized.
 */

	void
_gst_initialize_one_bits_in_byte_tables (void)

{
int		i, j, byte;
int8u *		p;
struct environment *	env;

	env = gst_env;

	FATAL_ERROR_IF (env EQ NULL);

	if (env -> one_bits_in_byte [0] NE NULL) {
		/* Tables have already been initialized. */
		return;
	}

	p = &(env -> one_bits_in_byte_array [0]);
	for (i = 0; i < 256; i++) {
		env -> one_bits_in_byte [i] = p;
		byte = i;
		j = 0;
		while (byte > 0) {
			if ((byte & 0x01) NE 0) {
				*p++ = j;
			}
			++j;
			byte >>= 1;
		}
	}

	/* Terminate last table entry. */
	env -> one_bits_in_byte [i] = p;
}
