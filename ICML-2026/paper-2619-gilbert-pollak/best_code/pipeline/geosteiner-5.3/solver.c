/***********************************************************************

	$Id: solver.c,v 1.74 2023/04/03 16:55:49 warme Exp $

	File:	solver.c
	Rev:	e-6
	Date:	04/03/2023

	Copyright (c) 2002, 2023 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	04/30/2002	benny
		: Created.
	a-2:	05/12/2014	warme
		: Eliminate excess precision (when using x87 FPU).
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	12/12/2015	warme
		: Fix uninitialized variable.
		: Fix memory leak.
	e-3:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-4:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Use better encapsulation for time conversions.
		: Fixed -Wall issues.  Upgrade fatals.
		: Make features unconditional.
	e-5:	11/18/2022	warme
		: Added gst_get_solver_hypergraph(),
		:  gst_get_solver_param() and gst_get_solver_status().
		: Removed bogus "gst_set_hook" function (undocumented).
		: Added gst_set_bb_callback_func() which provides
		:  a good (and documented) callback functionality.
		: Fix -Wall issues.
	e-6:	04/03/2023	warme
		: Handle cases where the backtrack search rejects
		:  instances having negative edge costs.

************************************************************************/

#include "solver.h"

#include "bb.h"
#include "bbsubs.h"
#include "btsearch.h"
#include "ckpt.h"
#include "constrnt.h"
#include "fatal.h"
#include <float.h>
#include "fputils.h"
#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include "parmblk.h"
#include "prepostlude.h"
#include "steiner.h"
#include <string.h>
#include "ub.h"


/*
 * Global routines
 */

gst_solver_ptr	gst_create_solver(gst_hg_ptr, gst_param_ptr, int *);
int		gst_free_solver (gst_solver_ptr);
gst_proplist_ptr gst_get_solver_properties (gst_solver_ptr);
gst_hg_ptr	gst_get_solver_hypergraph (gst_solver_ptr solver);
gst_param_ptr	gst_get_solver_param (gst_solver_ptr solver);
int		gst_get_solver_status (gst_solver_ptr	solver,
				       int *		soln_status);
int		gst_hg_solve (gst_solver_ptr, int *);
int		gst_hg_solution (gst_solver_ptr, int *, int *, double *, int);
void		gst_deliver_signals (gst_solver_ptr, int);

bool		_gst_update_best_solution_set (
					struct gst_solver *	solver,
					double *		x,
					int			nedges,
					int *			edges,
					bitmap_t *		smt);
void		_gst_update_solver_properties (struct gst_solver * solver);


/*
 * Local Routines
 */

static void	check_solution_set_resize (gst_solver_ptr);
static void	clear_solver (gst_solver_ptr);
static void	discard_solver_properties (gst_solver_ptr);
static void	free_files_to_merge (char **);
static char **	get_files_to_merge (gst_param_ptr);
static bool	problem_was_modified (gst_solver_ptr);
static void	truncate_upper_bound_list (gst_solver_ptr solver, int n);
static int	verify_hypergraph (gst_hg_ptr);

#define get_version(hg) \
	((hg) -> requested_version = (hg) -> version, (hg) -> version)

/*
 * Local Constants
 */

/* The optimizer latest in use. */
#define NO_OPTIMIZER		0
#define BACKTRACK_SEARCH	1
#define BRANCH_AND_CUT		2
#define HEURISTIC		3

/*
 * Create a solver object and initialize it
 */

	gst_solver_ptr
gst_create_solver (

gst_hg_ptr		H,
gst_param_ptr		params,
int *			status
)
{
int			nfs;
int			stat;
int			res;
gst_solver_ptr		solver;
struct gst_hg_solution * sol;

	GST_PRELUDE

	res = 0;

	if (params EQ NULL) {
		params = (gst_param_ptr) &_gst_default_parmblk;
	}

	solver = NEW (struct gst_solver);
	memset (solver, 0, sizeof (*solver));

	solver -> H			= H;
	solver -> params		= params;
	solver -> p2time		= 0;
	solver -> latest_optimizer_run	= NO_OPTIMIZER;
	solver -> solution_version = -1;

	nfs = params -> num_feasible_solutions;

	sol = NEWA (nfs, struct gst_hg_solution);
	memset (sol, 0, nfs * sizeof (*sol));
	solver -> solutions	= sol;
	solver -> solsize	= nfs;

	solver -> proplist = gst_create_proplist (&stat);
	if (stat NE 0) {
		res = stat;
	}

	if (status NE NULL) {
		*status = res;
	}

	GST_POSTLUDE
	return solver;
}

/*
 * Free the space taken by a solver object
 */

	int
gst_free_solver (

gst_solver_ptr		solver
)
{
	GST_PRELUDE

	if (solver NE NULL) {
		clear_solver (solver);

		gst_free_proplist (solver -> proplist);
		solver -> proplist = NULL;

		free (solver -> solutions);
		solver -> solutions = NULL;

		free (solver);
	}

	GST_POSTLUDE
	return 0;
}

/*
 * This routine returns a pointer to the solver's property list.
 */

	gst_proplist_ptr
gst_get_solver_properties (

gst_solver_ptr		sol	/* IN - solver */
)
{
gst_proplist_ptr	plist;

	GST_PRELUDE

	plist = NULL;
	if (sol NE NULL) {
		plist = sol -> proplist;
	}

	GST_POSTLUDE
	return plist;
}

/*
 * Try to solve whatever the parameters tells us to solve...
 */

	int
gst_hg_solve (

gst_solver_ptr	solver,		/* IN - the solver object */
int *		reason		/* OUT - reason to exit solver GST_SOLVE_... */
)
{
int		i;
int		nmasks;
int		res;
int		save_opt;
bool		use_backtrack_search;
cpu_time_t	t1;
struct bbinfo *	bbip;
gst_hg_ptr	H;
gst_param_ptr	params;
char **		merge_files;

	GST_PRELUDE

	H	= solver -> H;
	params	= solver -> params;
	bbip	= solver -> bbip;

	res = 0;

	if (problem_was_modified (solver)) {
		/* FIXME - (warme) Do we also need to clear the solver	*/
		/* if the *parameters* were modified since last time?	*/

		/* Scrap anything and everything we did before and	*/
		/* start from scratch (possibly even with a different	*/
		/* algorithm!).						*/

		clear_solver (solver);
		solver -> upperbound	= params -> initial_upper_bound;
		solver -> lowerbound	= -DBL_MAX;

		solver -> solution_version = get_version (solver -> H);

		i = verify_hypergraph (H);
		if (i NE 0) {
			res = i;
			goto error;
		}
	}

	discard_solver_properties (solver);

	check_solution_set_resize (solver);

	solver -> feasible_updates	= 0;
	solver -> preempt 		= GST_SOLVE_NORMAL;	/* Reason for exiting */
	solver -> t0	  		= _gst_get_cpu_time ();

	do {	/* Used only for "break". */

		if (bbip NE NULL) {
			/* Resuming a previous computation.	*/
			/* Don't restore any checkpoint!	*/
			break;
		}

		/* Restore a checkpoint, if available... */
		bbip = _gst_restore_checkpoint (H, params);
		if (bbip EQ NULL) {
			/* No checkpoint to restore. */
			break;
		}

		gst_channel_printf (params -> print_solve_trace, "\n\n");
		gst_channel_printf (params -> print_solve_trace,
			"**** RESTORING FROM CHECKPOINT ****\n\n");

		/* Build the initial LP formulation. */

		_gst_begin_using_lp_solver ();

		bbip -> lpmem = NEW (struct lpmem);
		bbip -> lp = _gst_build_initial_formulation (bbip -> cpool,
							     bbip -> vert_mask,
							     bbip -> edge_mask,
							     H,
							     bbip -> lpmem,
							     params);

		bbip -> params = params;
		solver -> bbip = bbip;
		bbip -> solver = solver;
	} while (FALSE);

	use_backtrack_search = FALSE;

	if (bbip EQ NULL) {

		/* Solving a problem from scratch.	*/

		/* If not searching for optimality a simple heuristic can be
		   attempted */
		if ((	 (params -> gap_target NE 1)
		     AND (solver -> lowerbound > -DBL_MAX))
		    OR (params -> upper_bound_target > -DBL_MAX)) {
			_gst_compute_heuristic_upper_bound (NULL, solver);
/* fprintf(stderr, "Calculated heuristic upper bound: %f\n", solver -> upperbound);
   fprintf(stderr, "Feasible updates: %d\n", solver -> feasible_updates);
*/
			if (solver -> upperbound <
			    params -> gap_target * solver -> lowerbound) {
				PREEMPT_SOLVER (solver, GST_SOLVE_GAP_TARGET);
			}

			if (solver -> upperbound <= params -> upper_bound_target) {
				PREEMPT_SOLVER (solver,
						GST_SOLVE_UPPER_BOUND_TARGET);
			}

			solver -> latest_optimizer_run = HEURISTIC;
		}

		/* First choose the algorithm...	*/
		switch (params -> solver_algorithm) {
		case GST_PVAL_SOLVER_ALGORITHM_AUTO:
			if ((H -> num_verts <= params -> backtrack_max_verts) AND
			    (H -> num_edges <= params -> backtrack_max_edges)) {
				use_backtrack_search = TRUE;
			}
			break;

		case GST_PVAL_SOLVER_ALGORITHM_BRANCH_AND_CUT:
			break;

		case GST_PVAL_SOLVER_ALGORITHM_BACKTRACK_SEARCH:
			use_backtrack_search = TRUE;
			break;

		default:
			FATAL_ERROR;
		}
	}

	nmasks = H -> num_edge_masks;

	if (	use_backtrack_search
	    AND (solver -> preempt EQ GST_SOLVE_NORMAL)) {
		if ((nmasks NE 1) OR (H -> num_edges > 32)) {
			/* The user ASKED for backtrack search on an	*/
			/* instance that is too big!			*/
			res = GST_ERR_BACKTRACK_OVERFLOW;
			goto error;
		}

		_gst_backtrack_search (solver);

		save_opt = solver -> latest_optimizer_run;
		solver -> latest_optimizer_run = BACKTRACK_SEARCH;

		if (solver -> preempt EQ GST_SOLVE_MAX_BACKTRACKS) {
			/* Backtrack search exceeded max-backtracks.	*/
			/* If the algorithm is "auto", we should fall	*/
			/* into the branch-and-cut instead.		*/
			solver -> latest_optimizer_run = save_opt;
			if (params -> solver_algorithm EQ
			    GST_PVAL_SOLVER_ALGORITHM_AUTO) {
				use_backtrack_search = FALSE;
				solver -> preempt = GST_SOLVE_NORMAL;
			}
		}
		else if (solver -> preempt EQ GST_SOLVE_BACKTRACK_BAD_COSTS) {
			/* Backtrack search was given an instance	*/
			/* having negative cost edges, which it cannot	*/
			/* handle, and rejected.			*/
			solver -> latest_optimizer_run = save_opt;
			if (solver -> params -> solver_algorithm EQ
			    GST_PVAL_SOLVER_ALGORITHM_BACKTRACK_SEARCH) {
				/* Caller requested that *only* the	*/
				/* backtrack search be used.  Pass this	*/
				/* error back to the caller.		*/
				res = GST_ERR_BACKTRACK_BAD_COSTS;
				goto error;
			}
			/* Fall back to branch-and-cut. */
			use_backtrack_search = FALSE;
			solver -> preempt = GST_SOLVE_NORMAL;
		}
	}

	if (	NOT use_backtrack_search
	    AND (solver -> preempt EQ GST_SOLVE_NORMAL)) {
		if (bbip EQ NULL) {
			_gst_begin_using_lp_solver ();
			bbip = _gst_create_bbinfo (solver);
			solver -> bbip = bbip;
		}
		else {
			/* Restarting a previous branch-and-cut... */
		}

		/* Merge in constraints from specified files. */
		merge_files = get_files_to_merge (params);
		_gst_merge_constraints (bbip, merge_files);
		free_files_to_merge (merge_files);

		/* Do the branch-and-cut... */
		_gst_branch_and_cut (solver);

		solver -> latest_optimizer_run = BRANCH_AND_CUT;
	}

	if (solver -> preempt EQ GST_SOLVE_NORMAL) {
		/* The search was not preempted and was therefore exhaustive */
		if (solver -> nsols > 0) {
			/* At least one solution was found and so it must be optimal */
			solver -> lowerbound = solver -> upperbound;
		}
	}

	/* Measure the time spent */
	t1 = _gst_get_cpu_time ();
	solver -> p2time += t1 - solver -> t0;

	_gst_update_solver_properties (solver);

	if (reason NE NULL) {
		*reason = solver -> preempt;
	}
error:
	GST_POSTLUDE
	return res;
}

/*
 * Query one of the best solutions. Rank 0 is the best solution available.
 */

	int
gst_hg_solution (

gst_solver_ptr		solver,
int *			nedges,
int *			edges,
double *		length,
int			rank
)
{
int				res;
struct gst_hg_solution *	sol;

	GST_PRELUDE

	res = 0;

	if (problem_was_modified (solver)) {
		truncate_upper_bound_list (solver, 0);
	}

	if (nedges NE NULL) {
		*nedges = 0;
	}

	do {		/* Used only for "break"... */
		if (solver -> nsols <= 0) {
			res = GST_ERR_RANK_OUT_OF_RANGE;
			break;
		}

		if ((rank < 0) OR
		    (rank >= solver -> solsize) OR
		    (rank >= solver -> nsols)) {
			res = GST_ERR_RANK_OUT_OF_RANGE;
			break;
		}

		sol = &(solver -> solutions [rank]);
		if (sol -> edge_mask EQ NULL) {
			res = GST_ERR_RANK_OUT_OF_RANGE;
			break;
		}

		if (nedges NE NULL) {
			*nedges = sol -> nedges;
		}
		if (edges NE NULL) {
			memcpy (edges,
				sol -> edges,
				sol -> nedges * sizeof (int));
		}
		if (length NE NULL) {
			*length = sol -> length;
		}
	} while (FALSE);

	GST_POSTLUDE
	return res;
}

/*
 * Return the status of the solution (if any) obtained by the given solver
 * object.  This status will be one of the following:
 *
 *  GST_STATUS_OPTIMAL	   - Optimal solution is available
 *  GST_STATUS_INFEASIBLE  - Problem is infeasible
 *  GST_STATUS_FEASIBLE	   - Search incomplete, feasible solution(s) known
 *  GST_STATUS_NO_FEASIBLE - Search incomplete, no feasible solutions known
 *  GST_STATUS_NO_SOLUTION - Solver never invoked, or problem change
 *			     invalidated prior solution
 *
 * This information is also available indirectly by using a combination of the
 * value returned by gst_hg_solve() and by trying to get the best solution
 * with gst_hg_solution().
 */

	int
gst_get_solver_status (

gst_solver_ptr		solver,		/* IN - a solver object */
int *			soln_status	/* OUT - status of the solution */
)
{
int	ss;

	GST_PRELUDE

	ss = GST_STATUS_NO_SOLUTION;

	if (problem_was_modified (solver)) {
		solver -> latest_optimizer_run = NO_OPTIMIZER;
	}

	switch (solver -> latest_optimizer_run) {
	case NO_OPTIMIZER: /* No solver has been run */
		ss = GST_STATUS_NO_SOLUTION;
		break;

	case HEURISTIC:
		if (solver -> nsols > 0) {
			ss = GST_STATUS_FEASIBLE;
		}
		else {
			ss = GST_STATUS_NO_FEASIBLE;
		}
		break;

	case BACKTRACK_SEARCH:
	case BRANCH_AND_CUT:
		if (solver -> preempt EQ GST_SOLVE_NORMAL) { /* The search was exhaustive */
			if (solver -> nsols > 0) {
				/* The best existing solution must be optimal */
				ss = GST_STATUS_OPTIMAL;
			}
			else {
				/* No solutions found -> the problem is infeasible */
				ss = GST_STATUS_INFEASIBLE;
			}
		}
		else { /* Non exhaustive search */
			if (solver -> nsols > 0) {
				ss = GST_STATUS_FEASIBLE;
			}
			else {
				ss = GST_STATUS_NO_FEASIBLE;
			}
		}
		break;

	default:
		FATAL_ERROR;
	}

	if (soln_status NE NULL) {
		*soln_status = ss;
	}

	GST_POSTLUDE
	return 0;
}

/*
 * Return the hypergraph associated with the given solver.
 */

	gst_hg_ptr
gst_get_solver_hypergraph (

gst_solver_ptr	solver		/* IN: Solver object */
)
{
gst_hg_ptr	cip;

	cip = NULL;
	if (solver NE NULL) {
		cip = solver -> H;
	}
	return (cip);
}

/*
 * Return the parameter object associated with the given solver.
 */

	gst_param_ptr
gst_get_solver_param (

gst_solver_ptr	solver		/* IN: Solver object */
)
{
gst_param_ptr	param;

	param = NULL;
	if (solver NE NULL) {
		param = solver -> params;
	}
	return (param);
}

/*
 * Set the callback function and data.
 * For simplicity there is one callback function for just about everything.
 */

	int
gst_set_bb_callback_func (

gst_solver_ptr		solver,		/* IN: Solver to set callback for */
gst_bb_callback_func_t *
			cb_func,	/* IN: Callback function */
void *			cb_data		/* IN: Data to pass to callback func */
)
{
int	retcode;

	if (solver NE NULL) {
		solver -> bb_callback_func	= cb_func;
		solver -> bb_callback_data	= cb_data;
		retcode = 0;
	}
	else {
		retcode = GST_ERR_INVALID_SOLVER;
	}

	return (retcode);
}

/*
 * This function is designed to be safely callable from a signal
 * handler.  The given signals are delivered to the given solver,
 * which responds to them at some point in the near future.
 * The signals parameter is the bit-wise OR of one or more special
 * signal values defined below.  For example, the user can establish a
 * signal handler for SIGINT that calls this routine with signals =
 * GST_SIGNAL_ABORT.
 */

	void
gst_deliver_signals (

gst_solver_ptr		solver,
int			signals
)
{
struct bbinfo *		bbip;

	/* We do *not* do GST_PRELUDE and GST_POSTLUDE here because:	*/
	/*	1. This routine does not do any floating point, so	*/
	/*	   there is no need to force FPU configuration.		*/
	/*	2. This is a signal handler, so it should be as		*/
	/*	   efficient as possible...				*/

	if (solver NE NULL) {
		if ((signals & GST_SIG_ABORT) NE 0) {
			PREEMPT_SOLVER (solver, GST_SOLVE_ABORT_SIGNAL);
		}
		bbip = solver -> bbip;
		if (bbip NE NULL) {
			if ((signals & GST_SIG_FORCE_BRANCH) NE 0) {
				bbip -> force_branch_flag = TRUE;
			}
			if ((signals & GST_SIG_STOP_TEST_BVAR) NE 0) {
				/* FIXME */
			}
			if ((signals & GST_SIG_STOP_SEP) NE 0) {
				/* FIXME */
			}
		}
	}
}

/*
 * This function helps to maintain a list of the n best feasible solutions.
 * It can receive solutions in three ways. As an LP solution (a list
 * of doubles), as an explicit list of edge numbers or as a bitmap array.
 * The returned value is a boolean indicating whether or not it was a new best
 * solution.
 * The function can also copy the solution to a given bitmap array when this is
 * needed.
 */

	bool
_gst_update_best_solution_set (

struct gst_solver *	solver,	/* IN - branch-and-bound info */
double *		x,	/* IN - LP solution */
int			nedges_solution, /* IN - number of edges in solution */
int *			edges,	/* IN - explicit edge numbers */
bitmap_t *		smt	/* IN/OUT - it is either the solution or where
				   we will store the solution */
)
{
int		i;
int		j;
int		e;
int		nsols;
int		solsize;
int		nedges;
int		nmasks;
int *		ep;
bitmap_t *	tmp_mask;
bitmap_t *	edge_mask;
struct gst_hg_solution *sols;
gst_param_ptr	params;
double *	cost;
double		length;
bool		res;

	sols	= solver -> solutions;
	nsols	= solver -> nsols;
	solsize	= solver -> solsize;
	params	= solver -> params;

	nedges = solver -> H -> num_edges;
	nmasks = solver -> H -> num_edge_masks;
	edge_mask = NEWA (nmasks, bitmap_t);
	cost   = solver -> H -> cost;

	for (i = 0; i < nmasks; i++) {
		edge_mask [i] = 0;
	}

	if (x NE NULL) {
		/* We are given an LP solution */
		nedges_solution = 0;
		for (i = 0; i < nedges; i++) {
			if (x [i] >= 0.5) {
				SETBIT (edge_mask, i);
				++nedges_solution;
			}
		}
	}
	else if (edges NE NULL) {
		/* We are given an explicit set of edges */
		ep = edges;
		for (i = 0; i < nedges_solution; i++) {
			e = *ep++;
			SETBIT (edge_mask, e);
		}
	}
	else if (smt NE NULL) {
		memcpy (edge_mask, smt, nmasks * sizeof (bitmap_t));
		smt = NULL;

		nedges_solution = 0;
		for (i = 0; i < nedges; i++) {
			if (BITON (edge_mask, i)) {
				++nedges_solution;
			}
		}
	}
	else {
		FATAL_ERROR;
	}

	/* Compute length of this solution.  Note: we always add up the	*/
	/* edge costs *in edge order*.  If we do NOT do this, then we	*/
	/* can get different lengths depending upon the order in which	*/
	/* we sum the costs!  This leads to nasty subtle bugs!		*/
	length = 0;
	for (i = 0; i < nedges; i++) {
		if (BITON (edge_mask, i)) {
			length += cost [i];
		}
	}

	/* Force result to double precision. */
	_gst_store_double (&length, length);

	if ((nsols >= solsize) AND (length >= sols [solsize - 1].length)) {
		/* Solution not among the K best solutions */
		free (edge_mask);
		return FALSE;
	}

	/* Check for duplicates (naive method) -- should use the length
	   to narrow the search */
	for (i = 0; i < nsols; i++) {
		if (nedges_solution NE sols [i].nedges) continue;
		tmp_mask = sols [i].edge_mask;
		for (j = 0; ; j++) {
			if (j >= nmasks) {
				/* Duplicate found */
				free (edge_mask);
				return FALSE;
			}
			if (tmp_mask [j] NE edge_mask [j]) {
				break;
			}
		}
	}

	/* Find the spot for the new solution */
	for (i = 0; i < nsols; i++) {
		if (length < sols [i].length) break;
	}

	res = i EQ 0; /* True if we have a new best solution (and better) */

	if (nsols >= solsize) {
		/* We don't have room to save another solution -- we	*/
		/* must discard the worst in order to make room.	*/
		FATAL_ERROR_IF (nsols > solsize);

		/* Free the last (worst) solution in the set */
		j = solsize - 1;
		free (sols [j].edges);
		sols [j].edges = NULL;

		free (sols [j].edge_mask);
		sols [j].edge_mask = NULL;
	}
	else {
		j = nsols;
		++nsols;
		solver -> nsols = nsols;
	}

	if (j > i) {
		/* Make room for the new solution */
		memmove (&sols [i + 1], &sols [i], (j - i) * sizeof (*sols));
	}

	/* Update the values for the new i'th best solution */
	ep = NEWA (nedges_solution, int);
	sols [i].length		= length;
	sols [i].nedges		= nedges_solution;
	sols [i].edges		= ep;
	sols [i].edge_mask	= edge_mask;

	/* Always generate from edge_mask, and never from edges	*/
	/* so that we can guarantee we list the edges in order.	*/
	for (j = 0; j < nedges; j++) {
		if (BITON (edge_mask, j)) {
			*ep++ = j;
		}
	}

	/* Count the number of updates */
	++(solver -> feasible_updates);

	if (	(params -> max_feasible_updates NE 0)
	    AND (solver -> feasible_updates >= params -> max_feasible_updates)) {
		PREEMPT_SOLVER (solver, GST_SOLVE_MAX_FEASIBLE_UPDATES);
	}

#if 0
	/* Dump solution set */
	printf ("Solution set:\n");
	for (i = 0; i < nsols; i++) {
		printf ("%20.16g: ", sols [i].length EQ DBL_MAX ? 0 : sols [i].length);
		if (sols [i].edges NE NULL) {
			for (j = 0; j < sols [i].nedges; j++) {
				printf ("%d ", sols [i].edges[j]);
			}
		}
		printf ("\n");
	}
#endif

	if (res) { /* A new best solution has been found */
		if (smt NE NULL) {
			memcpy (smt, sols [0].edge_mask,
				nmasks * sizeof (bitmap_t));
		}

		solver -> upperbound = length;
	}

	return res;
}

/*
 * Determine if the given problem instance was modified since the
 * last time the solver was called.
 */

	static
	bool
problem_was_modified (

gst_solver_ptr		solver
)
{
	if ((get_version (solver -> H)) > (solver -> solution_version)) {
		return (TRUE);
	}

	return (FALSE);
}

/*
 * Update the solver properties...
 */

	void
_gst_update_solver_properties (

struct gst_solver *	solver
)
{
struct bbstats *	statp;
gst_proplist_ptr	plist;

#define SETINT(a,b)	gst_set_int_property (plist, a, b)
#define SETDBL(a,b)	gst_set_dbl_property (plist, a, b)

	plist = solver -> proplist;
	SETDBL (GST_PROP_SOLVER_LOWER_BOUND, solver -> lowerbound);
	SETDBL (GST_PROP_SOLVER_CPU_TIME,
		_gst_cpu_time_t_to_double_seconds (solver -> p2time));

	if (	(solver -> bbip NE NULL)
	    AND ((statp = solver -> bbip -> statp) NE NULL)) {
		SETDBL (GST_PROP_SOLVER_ROOT_TIME,
			_gst_cpu_time_t_to_double_seconds (statp -> root_time));
		SETDBL (GST_PROP_SOLVER_ROOT_LENGTH,  statp -> root_z);

		SETINT (GST_PROP_SOLVER_ROOT_OPTIMAL, statp -> root_opt);
		SETINT (GST_PROP_SOLVER_ROOT_LPS,     statp -> root_lps);
		SETINT (GST_PROP_SOLVER_NUM_NODES,    statp -> num_nodes);
		SETINT (GST_PROP_SOLVER_NUM_LPS,      statp -> num_lps);
		SETINT (GST_PROP_SOLVER_INIT_PROWS,   statp -> cs_init.num_prows);
		SETINT (GST_PROP_SOLVER_INIT_PNZ,     statp -> cs_init.num_pnz);
		SETINT (GST_PROP_SOLVER_INIT_LPROWS,  statp -> cs_init.num_lprows);
		SETINT (GST_PROP_SOLVER_INIT_LPNZ,    statp -> cs_init.num_lpnz);
		SETINT (GST_PROP_SOLVER_ROOT_PROWS,   statp -> cs_root.num_prows);
		SETINT (GST_PROP_SOLVER_ROOT_PNZ,     statp -> cs_root.num_pnz);
		SETINT (GST_PROP_SOLVER_ROOT_LPROWS,  statp -> cs_root.num_lprows);
		SETINT (GST_PROP_SOLVER_ROOT_LPNZ,    statp -> cs_root.num_lpnz);
		SETINT (GST_PROP_SOLVER_FINAL_PROWS,  statp -> cs_final.num_prows);
		SETINT (GST_PROP_SOLVER_FINAL_PNZ,    statp -> cs_final.num_pnz);
		SETINT (GST_PROP_SOLVER_FINAL_LPROWS, statp -> cs_final.num_lprows);
		SETINT (GST_PROP_SOLVER_FINAL_LPNZ,   statp -> cs_final.num_lpnz);
	}
}

/*
 * Discard and and all values for the solver properties...
 */

	static
	void
discard_solver_properties (

gst_solver_ptr	solver
)
{
int			i;
gst_proplist_ptr	plist;
int *			ip;

static int ids [] = {
	GST_PROP_SOLVER_LOWER_BOUND,
	GST_PROP_SOLVER_CPU_TIME,
	GST_PROP_SOLVER_ROOT_TIME,
	GST_PROP_SOLVER_ROOT_LENGTH,
	GST_PROP_SOLVER_ROOT_OPTIMAL,
	GST_PROP_SOLVER_ROOT_LPS,
	GST_PROP_SOLVER_NUM_NODES,
	GST_PROP_SOLVER_NUM_LPS,
	GST_PROP_SOLVER_INIT_PROWS,
	GST_PROP_SOLVER_INIT_PNZ,
	GST_PROP_SOLVER_INIT_LPROWS,
	GST_PROP_SOLVER_INIT_LPNZ,
	GST_PROP_SOLVER_ROOT_PROWS,
	GST_PROP_SOLVER_ROOT_PNZ,
	GST_PROP_SOLVER_ROOT_LPROWS,
	GST_PROP_SOLVER_ROOT_LPNZ,
	GST_PROP_SOLVER_FINAL_PROWS,
	GST_PROP_SOLVER_FINAL_PNZ,
	GST_PROP_SOLVER_FINAL_LPROWS,
	GST_PROP_SOLVER_FINAL_LPNZ,
	-1
};

	plist = solver -> proplist;

	ip = &ids [0];
	for (;;) {
		i = *ip++;
		if (i < 0) break;
		gst_delete_property (plist, i);
	}
}

/*
 *  Clears all information related to any previous solution process.
 */

	static
	void
clear_solver (

gst_solver_ptr		solver	/* IN - an existing solver object */
)
{
	if (solver -> ubip NE NULL) {
		_gst_shutdown_heuristic_upper_bound (solver -> ubip);
		solver -> ubip = NULL;
	}
	if (solver -> bbip NE NULL) {
		_gst_destroy_bbinfo (solver -> bbip);
		solver -> bbip = NULL;
	}

	/* Discard all known upper bounds. */
	truncate_upper_bound_list (solver, 0);
}

/*
 * Check to see if the size of the "best solution set" has been
 * changed since the last time we solved.  If so, modify the
 * buffer accordingly.
 */

	static
	void
check_solution_set_resize (

gst_solver_ptr		solver	/* IN - an existing solver object */
)
{
int				n;
int				newsize;
int				cursize;
struct gst_hg_solution *	newsols;

	newsize	= solver -> params -> num_feasible_solutions;
	cursize	= solver -> solsize;

	if (newsize EQ cursize) {
		return;
	}

	if (newsize < cursize) {
		/* Buffer is being shrunk.  We might have to	*/
		/* throw away existing solutions.		*/
		truncate_upper_bound_list (solver, newsize);
	}

	newsols = NEWA (newsize, struct gst_hg_solution);

	/* Determine how much to copy from existing buffer. */
	n = solver -> nsols;
	if (n > newsize) {
		n = newsize;
	}
	memcpy (newsols, solver -> solutions, n * sizeof (*newsols));

	if (n < newsize) {
		/* Clear out the rest. */
		memset (&newsols [n], 0, (newsize - n) * sizeof (*newsols));
	}

	free (solver -> solutions);
	solver -> solutions	= newsols;
	solver -> solsize	= newsize;
}

/*
 * Truncate the list of upper bounds by discarding all
 * but the first N in the list.
 */

	static
	void
truncate_upper_bound_list (

gst_solver_ptr		solver,	/* IN - an existing solver object */
int			n	/* IN - truncate all but the first */
				/*	N upper bounds */
)
{
int				i;
int				nsols;
struct gst_hg_solution *	sols;

	FATAL_ERROR_IF (n < 0);

	/* Discard all but the first N upper bounds. */

	sols = solver -> solutions;
	if (sols NE NULL) {
		nsols = solver -> nsols;
		for (i = n; i < nsols; i++) {
			if (sols [i].edges NE NULL) {
				free (sols [i].edges);
				sols [i].edges = NULL;
			}
			if (sols [i].edge_mask NE NULL) {
				free (sols [i].edge_mask);
				sols [i].edge_mask = NULL;
			}
		}
	}
	if (solver -> nsols > n) {
		solver -> nsols = n;
	}
}

/*
 *  Various verifications of a hypergraph...
 */

	static
	int
verify_hypergraph (

gst_hg_ptr	H
)
{
int	i;
int	nedges;
int	nverts;

	nverts = gst_get_hg_number_of_vertices (H);
	if (nverts <= 0) {
		return (GST_ERR_INVALID_NUMBER_OF_VERTICES);
	}

	gst_get_hg_edges (H, &nedges, NULL, NULL, NULL);
	if (nedges <= 0) {
		return (GST_ERR_INVALID_NUMBER_OF_EDGES);
	}

	/* All vertices must be terminals i.e. no Steiner points */
	for (i = 0; i < nverts; i++) {
		if (H -> tflag [i] EQ FALSE) {
			return (GST_ERR_NO_STEINERS_ALLOWED);
		}
	}
	return (0);
}

/*
 * Get the list of checkpoint file pathnames from the CHECKPOINT_MERGE_FILES
 * parameter and convert it into a dynamically allocated NULL-terminated array
 * of pointers to dynamically allocated null-terminated strings.
 */

	static
	char **
get_files_to_merge (

gst_param_ptr		params		/* IN - parameters to get file list from */
)
{
int		i;
int		n;
int		propid;
int		length;
char **		list;
char **		newlist;
char *		value;
char *		p;
char *		q;
char *		s;

	n = 0;
	list = NEWA (1, char *);
	list [n] = NULL;

	propid = GST_PARAM_MERGE_CONSTRAINT_FILES;

	value = NULL;

	do {		/* Used only for "break"... */
		i = gst_get_str_param (params, propid, &length, NULL);
		if (i NE 0) break;

		/* The parameter is defined.  Allocate a buffer	*/
		/* and retrieve the value.			*/
		value = NEWA (length + 1, char);
		i = gst_get_str_param (params, propid, NULL, value);
		if (i NE 0) break;

		if (value [0] EQ '\0') {
			/* String is empty, so is the list of pathnames... */
			break;
		}

		/* We have the parameter value.  Construct list of strings. */
		p = value;
		while (*p NE '\0') {
			q = p;
			for (;;) {
				if (*q EQ '\0') break;
				if (*q EQ ':') break;
				++q;
			}
			newlist = NEWA (n + 1, char *);
			for (i = 0; i < n; i++) {
				newlist [i] = list [i];
			}
			free (list);
			list = newlist;
			s = NEWA ((q - p) + 1, char);
			list [n++] = s;
			list [n] = NULL;
			while (p < q) {
				*s++ = *p++;
			}
			*s++ = '\0';
			if (*p EQ ':') {
				++p;
			}
		}

	} while (FALSE);

	if (value NE NULL) {
		free (value);
	}

	return (list);
}

/*
 * Free up a dynamically allocated NULL-terminated array of pointers to
 * dynamically allocated null-terminated strings.
 */

	static
	void
free_files_to_merge (

char **		path_list		/* IN - array of strings to free */
)
{
char **		p;
char *		s;

	p = path_list;
	for (;;) {
		s = *p++;
		if (s EQ NULL) break;
		free (s);
	}

	free (path_list);
}
