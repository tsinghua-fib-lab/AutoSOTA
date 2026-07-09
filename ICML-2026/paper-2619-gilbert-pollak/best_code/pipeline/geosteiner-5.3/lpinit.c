/***********************************************************************

	$Id: lpinit.c,v 1.8 2022/11/19 13:45:53 warme Exp $

	File:	lpinit.c
	Rev:	e-4
	Date:	09/24/2016

	Copyright (c) 1995, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines to start up and shut down the LP solver.

************************************************************************

	Modification Log:

	a-1:	09/17/2005	warme
		: Split off from bbsubs.c so that the branch-and-cut
		:  is not linked into everything that references the
		:  GeoSteiner environment!
	b-1:	02/02/2014	warme
		: Removed an unnecessary include file.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	12/12/2015	warme
		: Fix memory leak on CPLEX log file.
	e-3:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-4:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.

************************************************************************/

#include "config.h"
#include "fatal.h"
#include "environment.h"
#include "logic.h"
#include "steiner.h"
#include <string.h>


/*
 * Global Routines
 */

void			_gst_shutdown_lp_solver (void);
void			_gst_startup_lp_solver (void);


/*
 * Local Routines
 */

#ifdef CPLEX
static void		startup_cplex (void);
#endif

/*
 * This routine does everything needed to start up whichever
 * LP solver we are using.
 */

	void
_gst_startup_lp_solver (void)

{
#ifdef CPLEX
	startup_cplex ();
#endif

	/* Nothing for lp_solve... */
}



/*
 * This routine does everything needed to shut down whichever
 * LP solver we are using.
 */

	void
_gst_shutdown_lp_solver (void)

{
#if defined (CPLEX) AND (CPLEX >= 40)
int		i;
CPXFILEptr	fp;

	/* Close our CPLEX logfile. */
	i = CPXflushstdchannels (cplex_env);
	FATAL_ERROR_IF (i NE 0);
	i = CPXgetlogfile (cplex_env, &fp);
	FATAL_ERROR_IF (i NE 0);
	i = CPXsetlogfile (cplex_env, NULL);
	FATAL_ERROR_IF (i NE 0);
	if (fp NE NULL) {
		CPXfclose (fp);
	}

	/* Shut down CPLEX... */
	if (CPXcloseCPLEX (&cplex_env) NE 0) {
		fprintf (stderr, "Warning: Unable to close CPLEX.\n");
	}
#endif

	/* Nothing for lp_solve... */
}

/*
 * This routine performs everything needed to start up the
 * newer versions of CPLEX.
 */

#if defined(CPLEX) AND (CPLEX >= 40)

	static
	void
startup_cplex (void)

{
int		status;
FILE *		fp;
CPXFILEptr	cfp;
CPXCHANNELptr	_res, _warn, _error, _log;
char		msg [512];

#ifdef HAVE_STDERR_IS_LVALUE
    {	FILE * esave;
	/* Flush the #@!$% CPLEX startup banner! */
	esave = stderr;
	stderr = fopen ("/dev/null", "w");
#endif

	cplex_env = _MYCPX_openCPLEX (&status);

#ifdef HAVE_STDERR_IS_LVALUE
	/* Undo flushing of stderr... */
	fp = stderr;
	stderr = esave;
	fclose (fp);
    }
#endif

	if (cplex_env EQ NULL) {
		if (CPXgeterrorstring (NULL, status, msg) EQ NULL) {
			strcpy (msg, "No CPLEX error message.");
		}
		fprintf (stderr, "%s\n", msg);
		goto shutdown;
	}

	/* Get rid of CPLEX's default message destinations. */
	CPXgetchannels (cplex_env, &_res, &_warn, &_error, &_log);
	CPXdisconnectchannel (cplex_env, _res);
	CPXdisconnectchannel (cplex_env, _warn);
	CPXdisconnectchannel (cplex_env, _error);
	CPXdisconnectchannel (cplex_env, _log);

	CPXsetintparam (cplex_env, CPX_PARAM_SCRIND, 0);

	cfp = CPXfopen ("cplex.log", "a");
	if (cfp EQ NULL) {
		perror ("cplex.log");
		goto shutdown;
	}

	/* Send all log stuff to the cplex.log file. */
	CPXsetlogfile (cplex_env, cfp);

	/* But discard the results stuff... */
	CPXdisconnectchannel (cplex_env, _res);

	/* CPLEX is now ready to roll! */
	return;

	/* Something didn't work -- shut down CPLEX and get out. */
shutdown:
	CPXcloseCPLEX (&cplex_env);
	exit (1);
}

#endif

/*
 * This routine performs everything needed to start up older
 * versions of CPLEX.
 */

#if defined(CPLEX) AND (CPLEX < 40)

	static
	void
startup_cplex (void)

{
	/* Older CPLEX library isn't as noisy! */
	_MYCPX_setlogfile (stderr);
}

#endif
