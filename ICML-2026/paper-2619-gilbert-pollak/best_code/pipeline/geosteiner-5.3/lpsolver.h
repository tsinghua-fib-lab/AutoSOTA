/***********************************************************************

	$Id: lpsolver.h,v 1.8 2022/11/19 13:45:53 warme Exp $

	File:	lpsolver.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Declarations for LP solver interfaces.

************************************************************************

	Modification Log:

	a-1:	09/19/2004	warme
		: Split off from other files.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, fix -Wall issues.

************************************************************************/

#ifndef LPSOLVER_H
#define	LPSOLVER_H

#define _GNU_SOURCE

#include "config.h"
#include "logic.h"


#ifdef CPLEX
 #if CPLEX < 40
  #define PROTOTYPE_MAX
  #include "cpxdefs.inc"
 #else
  #define CPX_PROTOTYPE_ANSI
  #include "cplex.h"
 #endif
#endif

#ifdef LPSOLVE
#include "lpkit.h"
#endif


#include "environment.h"	/* must come after cplex.h! */


/*
 * The type used to represent an LP problem instance depends upon
 * whieh LP-solver we are using...
 */

#ifdef CPLEX
typedef struct cpxlp		LP_t;
#endif

#ifdef LPSOLVE
typedef lprec			LP_t;
#endif

/*
 * Here are a bunch of macros that we use to insulate us from the
 * calling convention differences between CPLEX versions 3.0 and 4.0.
 */

#ifdef CPLEX
 #if CPLEX >= 40
  /* Version 4.0 or greater...					*/

  /* For simplicity, make the CPLEX environment be a global.	*/
   #if 0
     extern CPXENVptr	cplex_env;
   #endif
#define cplex_env gst_env -> _cplex_env

  /* These changed in 5.0...					*/
  #if CPLEX >= 50
   #define _MYCPX_copybase(lp, cstat, rstat) \
		(CPXcopybase (cplex_env, lp, cstat, rstat))
   #define _MYCPX_primopt(lp)	(CPXprimopt (cplex_env, lp))
   #define _MYCPX_INFBOUND	CPX_INFBOUND
  #else
   #define _MYCPX_copybase(lp, cstat, rstat) \
		(CPXloadbase (cplex_env, lp, cstat, rstat))
   #define _MYCPX_primopt(lp)	(CPXoptimize (cplex_env, lp))
   #define _MYCPX_INFBOUND	INFBOUND
  #endif

  /* Unchanged since 4.0...					*/

  #define _MYCPX_addrows(lp, ccnt, rcnt, nzcnt, rhs, sense, \
			 rmatbeg, rmatind, rmatval, colname, rowname) \
		(CPXaddrows (cplex_env, lp, ccnt, rcnt, nzcnt, rhs, sense, \
			     rmatbeg, rmatind, rmatval, colname, rowname))
  #define _MYCPX_chgbds(lp, cnt, index, lu, bd) \
		(CPXchgbds (cplex_env, lp, cnt, index, lu, bd))
  #define _MYCPX_delsetrows(lp, delstat) \
		(CPXdelsetrows (cplex_env, lp, delstat))
  #define _MYCPX_dualopt(lp)	(CPXdualopt (cplex_env, lp))
  #define _MYCPX_freeprob(lpp)	(CPXfreeprob (cplex_env, lpp))
  #define _MYCPX_getbase(lp, cstat, rstat) \
		(CPXgetbase (cplex_env, lp, cstat, rstat))
  #define _MYCPX_getdj(lp, dj, begin, end) \
		(CPXgetdj (cplex_env, lp, dj, begin, end))
  #define _MYCPX_getnumcols(lp)	(CPXgetnumcols (cplex_env, lp))
  #define _MYCPX_getnumnz(lp)	(CPXgetnumnz (cplex_env, lp))
  #define _MYCPX_getnumrows(lp)	(CPXgetnumrows (cplex_env, lp))
  #define _MYCPX_getnzspace(lp)	(CPXgetnzspace (cplex_env, lp))
  #define _MYCPX_getobj(lp,obj,b,e) (CPXgetobj (cplex_env, lp, obj, b, e))
  #define _MYCPX_getrowspace(lp) (CPXgetrowspace (cplex_env, lp))
  #define _MYCPX_getslack(lp, slack, begin, end) \
		(CPXgetslack (cplex_env, lp, slack, begin, end))
  #define _MYCPX_lpwrite(lp, fname) (CPXlpwrite (cplex_env, lp, fname))
  #define _MYCPX_openCPLEX(stp) (CPXopenCPLEX (stp))
  #define _MYCPX_setadvind(value) \
		(CPXsetintparam (cplex_env, CPX_PARAM_ADVIND, value))
  #define _MYCPX_setlogfile(stream) (CPXsetlogfile (cplex_env, stream))
  #define _MYCPX_setobjulim(limit, small, big) \
		((void) (small),	/* ignore obsolete arg */ \
		 (void) (big),		/* ignore obsolste arg */ \
		 CPXsetdblparam (cplex_env, CPX_PARAM_OBJULIM, limit))
  #define _MYCPX_setscaind(flag, small, big) \
		((void) (small),	/* ignore obsolete arg */ \
		 (void) (big),		/* ignore obsolete arg */ \
		 CPXsetintparam (cplex_env, CPX_PARAM_SCAIND, flag))
  #define _MYCPX_solution(lp, stat, z, x, pi, slack, dj) \
		(CPXsolution (cplex_env, lp, stat, z, x, pi, slack, dj))

  #define _MYCPX_MIN	CPX_MIN
  #define _MYCPX_MAX	CPX_MAX

  /* Not sure when these changed.  Check which are available. */
  #ifdef CPX_STAT_OPTIMAL
    #define _MYCPX_STAT_OPTIMAL		CPX_STAT_OPTIMAL
    #define _MYCPX_STAT_UNBOUNDED	CPX_STAT_UNBOUNDED
    #define _MYCPX_STAT_INFEASIBLE	CPX_STAT_INFEASIBLE
    #define _MYCPX_STAT_OPTIMAL_INFEAS	CPX_STAT_OPTIMAL_INFEAS
    #define _MYCPX_STAT_ABORT_OBJ_LIM	CPX_STAT_ABORT_OBJ_LIM
  #elif defined (CPX_OPTIMAL)
    #define _MYCPX_STAT_OPTIMAL		CPX_OPTIMAL
    #define _MYCPX_STAT_UNBOUNDED	CPX_UNBOUNDED
    #define _MYCPX_STAT_INFEASIBLE	CPX_INFEASIBLE
    #define _MYCPX_STAT_OPTIMAL_INFEAS	CPX_OPTIMAL_INFEAS
    #define _MYCPX_STAT_ABORT_OBJ_LIM	CPX_OBJ_LIM
  #else
    #error "CPLEX interfaces are missing!"
  #endif

  /* Later versions of CPLEX have ELIMINATED functions such as		*/
  /* CPXloadlp().  The only way to create a new LP problem instance in	*/
  /* these newer versions is via the CPXcreateprob() function, which	*/
  /* creates a new "empty" LP instance (zero rows and zero columns).	*/
  /* Since the vendor has decided that this is the preferred interface,	*/
  /* we will use it if it is available.  In this case, we provide our	*/
  /* own implementation of the "loadlp" routine that uses only the	*/
  /* new interfaces.							*/
  #ifdef CPLEX_HAS_CREATEPROB
		static
		__inline__
		CPXLPptr
	_gst_loadlp (

	CPXENVptr	env,
	char *		probname,
	int		numcols,
	int		numrows,
	int		objsen,
	double *	obj,
	double *	rhs,
	char *		sense,
	int *		matbeg,
	int *		matcnt,
	int *		matind,
	double *	matval,
	double *	lb,
	double *	ub,
	double *	rngval,
	int		colspace,
	int		rowspace,
	int		nzspace
	)
	{
	int		status;
	CPXLPptr	lp;

		(void) colspace;
		(void) rowspace;
		(void) nzspace;

		lp = CPXcreateprob (env, &status, probname);
		if (lp NE NULL) {
			status = CPXcopylp (env,
					    lp,
					    numcols,
					    numrows,
					    objsen,
					    obj,
					    rhs,
					    sense,
					    matbeg,
					    matcnt,
					    matind,
					    matval,
					    lb,
					    ub,
					    rngval);
			if (status NE 0) {
				CPXfreeprob (env, &lp);
				lp = NULL;
			}
		}
		return (lp);
	}
  #define _MYCPX_loadlp(probname, numcols, numrows, objsen, obj, rhs, \
			 sense, matbeg, matcnt, matind, matval, lb, ub, \
			 rngval, colspace, rowspace, nzspace) \
		(_gst_loadlp (cplex_env, probname, numcols, numrows, objsen, \
			      obj, rhs, sense, matbeg, matcnt, matind, \
			      matval, lb, ub, rngval, colspace, rowspace, \
			      nzspace))
  #else
    #define _MYCPX_loadlp(probname, numcols, numrows, objsen, obj, rhs, \
			  sense, matbeg, matcnt, matind, matval, lb, ub, \
			  rngval, colspace, rowspace, nzspace) \
		(CPXloadlp (cplex_env, probname, numcols, numrows, objsen, \
			    obj, rhs, sense, matbeg, matcnt, matind, matval, \
			    lb, ub, rngval, colspace, rowspace, nzspace))
  #endif

 #else
  /* version 3.0 */

  #define _MYCPX_addrows(lp, ccnt, rcnt, nzcnt, rhs, sense, \
			 rmatbeg, rmatind, rmatval, colname, rowname) \
		(addrows (lp, ccnt, rcnt, nzcnt, rhs, sense, \
			  rmatbeg, rmatind, rmatval, colname, rowname))
  #define _MYCPX_chgbds(lp, cnt, index, lu, bd) \
		(chgbds (lp, cnt, index, lu, bd))
  #define _MYCPX_delsetrows(lp, delstat) (delsetrows (lp, delstat))
  #define _MYCPX_dualopt(lp)	(dualopt (lp))
  #define _MYCPX_freeprob(lpp)	(freeprob (lpp), 0)
  #define _MYCPX_getbase(lp, cstat, rstat) \
		(getbase (lp, cstat, rstat))
  #define _MYCPX_getdj(lp, dj, begin, end) \
		(getdj (lp, dj, begin, end))
  #define _MYCPX_getnumcols(lp)	(getmac (lp))
  #define _MYCPX_getnumnz(lp)	(getmat (lp))
  #define _MYCPX_getnumrows(lp) (getmar (lp))
  #define _MYCPX_getnzspace(lp) (getmatsz (lp))
  #define _MYCPX_getobj(lp,obj,b,e) (getobj (lp, obj, b, e))
  #define _MYCPX_getrowspace(lp) (getmarsz (lp))
  #define _MYCPX_getslack(lp, slack, begin, end) \
		(getslack (lp, slack, begin, end))
  #define _MYCPX_copybase(lp, cstat, rstat) \
		(loadbase (lp, cstat, rstat))
  #define _MYCPX_loadlp(probname, numcols, numrows, objsen, obj, rhs, \
			 sense, matbeg, matcnt, matind, matval, lb, ub, \
			 rngval, colspace, rowspace, nzspace) \
		(loadprob (probname, numcols, numrows, 0, objsen, \
			    obj, rhs, sense, matbeg, matcnt, matind, matval, \
			    lb, ub, rngval, \
			    NULL, NULL, NULL, NULL, NULL, \
			    NULL, NULL, NULL, NULL, NULL, \
			    NULL, NULL, NULL, NULL, NULL, \
			    NULL, NULL, \
			    colspace, rowspace, nzspace, 0, 0, 0, 0, 0))
  #define _MYCPX_lpwrite(lp, fname) (lpwrite (lp, fname))
  #define _MYCPX_primopt(lp)	(optimize (lp))
  #define _MYCPX_setadvind(value) (setadvind (value))
  #define _MYCPX_setlogfile(stream) (setlogfile (stream))
  #define _MYCPX_setobjulim(limit, small, big) \
		(setobjulim (limit, small, big))
  #define _MYCPX_setscaind(flag, small, big) \
		(setscaind (flag, small, big))
  #define _MYCPX_solution(lp, stat, z, x, pi, slack, dj) \
		(solution (lp, stat, z, x, pi, slack, dj))

  #define _MYCPX_MIN	1
  #define _MYCPX_MAX	-1
  #define _MYCPX_INFBOUND	INFBOUND
 #endif
#endif

/*
 * Some macros to do common things to LP's
 */

#ifdef CPLEX
#define	GET_LP_NUM_COLS(lp)	(_MYCPX_getnumcols (lp))
#define	GET_LP_NUM_ROWS(lp)	(_MYCPX_getnumrows (lp))
#define	GET_LP_NUM_NZ(lp)	(_MYCPX_getnumnz (lp))
#endif

#ifdef LPSOLVE
#define	GET_LP_NUM_COLS(lp)	((lp) -> columns)
#define	GET_LP_NUM_ROWS(lp)	((lp) -> rows)
#define	GET_LP_NUM_NZ(lp)	((lp) -> non_zeros)
#endif


/*
 * A structure to keep track of dynamic memory used by an LP.
 */

#ifdef CPLEX
struct lpmem {
	double *	objx;
	double *	rhsx;
	char *		senx;
	int *		matbeg;
	int *		matcnt;
	int *		matind;
	double *	matval;
	double *	bdl;
	double *	bdu;
	int		obj_scale;	/* objective scale factor */
};
#endif

#ifdef LPSOLVE
struct lpmem {
	/* lp_solve_2.0 dynamically manages the LP tableaux memory... */
	int	dummy;
};
#endif

#endif
