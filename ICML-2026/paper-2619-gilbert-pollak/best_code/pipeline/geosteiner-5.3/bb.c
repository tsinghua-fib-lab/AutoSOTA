/***********************************************************************

	$Id: bb.c,v 1.63 2023/04/03 17:53:03 warme Exp $

	File:	bb.c
	Rev:	e-6
	Date:	04/03/2023

	Copyright (c) 1995, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	The guts of the branch-and-cut.

************************************************************************

	Modification Log:

	a-1:	11/17/95	warme
		: Created.
	b-1:	11/14/96	warme
		: Renamed this program.
		: Split off the cutset stuff into another file.
		: Other cleanups for release.
	b-2:	02/28/2001	warme
		: Numerous changes for 3.1 release.
		: Split the main() routine off into bbmain.c.
		: Split certain utility routines off into bbsubs.c.
		: Major improvements to branch var selection.
		: New create_bbinfo routine does setup for
		:  branch_and_cut.
		: Enable variable fixing code.
	c-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Global variables removed.
		: Introduced the use of parameters.
		: Added CPU polling.
		: Added saving of multiple solutions.
		: Channels used for trace output.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Clean up time unit conversions.
		: Fix -Wall issues.  Upgrade fatals.
		: Make features unconditional.
	e-4:	09/30/2016	warme
		: Fix another -Wall issue.
	e-5:	11/18/2022	warme
		: Add callbacks and new parameters.
		: Added _gst_get_lb_status().
		: Initialize bbnode owner and delrow_z fields.
		: Simplify calling conventions of several functions.
		: Fix looping bug in component-level local cuts.
		: Fix -Wall issues.
	e-6:	04/03/2023	warme
		: Fix bbip->lp pointer becoming invalid after handling
		:  CPLEX "unscaled infeasibility" issue.
		: Added additional comments regarding this.
		: Fix memory leak in cut_off_existing_nodes().
		: Print number of fractional variables on all
		:  feasible LP solutions, even if cutoff.

************************************************************************/

#include "bb.h"

#include "bbsubs.h"
#include "callback.h"
#include "channels.h"
#include "ckpt.h"
#include "config.h"
#include "constrnt.h"
#include "cputime.h"
#include "cra.h"
#include "cutset.h"
#include "fatal.h"
#include <float.h>
#include "genps.h"
#include "incompat.h"
#include "io.h"
#include "localcut.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "parmblk.h"
#include "polltime.h"
#include "sec2.h"
#include "sec_comp.h"
#include "sec_heur.h"
#include "solver.h"
#include "steiner.h"
#include <string.h>
#include "ub.h"
#include "weak.h"


/*
 * Global Routines
 */

void			_gst_branch_and_cut (gst_solver_ptr solver);
bool			_gst_check_for_better_IFS (double *	x,
					      struct bbinfo *	bbip,
					      double *		true_z);
struct bbinfo *		_gst_create_bbinfo (gst_solver_ptr	solver);
void			_gst_new_upper_bound (double ub, struct bbinfo * bbip);


/*
 * Local Equates
 */


	/* Lower bound outcomes... */
#define	LB_INFEASIBLE		1
#define	LB_CUTOFF		2
#define	LB_INTEGRAL		3
#define	LB_FRACTIONAL		4
#define LB_PREEMPTED		5

	/* Variable fixing outcomes... */
#define	VFIX_NOTHING_FIXED	0
#define	VFIX_VARIABLES_FIXED	1
#define	VFIX_FIXED_FRACTIONAL	2
#define	VFIX_INFEASIBLE		3

#define	UP_FIRST	TRUE


/*
 * Local Types
 */

struct bvar {			/* Info about best branch var seen */
	int	var;		/* Best branch var, or -1 */
	double	z0;		/* Objective when Xj=0 */
	double	z1;		/* Objective when Xj=1 */
	double	test_2nd_val;	/* Only check 2nd branch if 1st > this. */
};

#ifdef CPLEX

struct basis_save {	/* Structure to save basis state for CPLEX */
	int *		cstat;
	int *		rstat;
};

#endif


/*
 * Local Routines
 */

static int		carefully_choose_branching_variable (struct bbinfo *,
							     double *,
							     double *);
static void		change_var_bounds (LP_t *, int, double, double);
static void		check_root_constraints (struct bbinfo *);
static int		choose_branching_variable (struct bbinfo *,
						   double *,
						   double *);
static bool		compare_branch_vars (struct bbinfo *,
					     int,
					     struct bvar *);
static int		compute_good_lower_bound (struct bbinfo *);
static void		cut_off_existing_nodes (double		best_z,
						struct bbinfo *	bbip);
static struct constraint * do_separations (struct bbinfo *,
					   cpu_time_t **);
static bool		eval_branch_var (struct bbinfo *,
					 int,
					 int,
					 struct basis_save *,
					 double);
static int		fix_variables (struct bbinfo *,
				       int *, int,
				       int *, int);
static bool		integer_feasible_solution (double *		x,
						   struct bbinfo *	bbip);
static void		new_lower_bound (double, struct bbinfo *);
static int		num_fractional_vars (double * x, struct bbinfo * bbip);
static int		reduced_cost_var_fixing (struct bbinfo *);
static struct bbnode *	select_next_node (struct bbtree *);
static void		sort_branching_vars (int *, int, double *);
static void		trace_node (struct bbinfo *, char, char *);
static void		update_node_preempt_value (struct bbinfo *);

#ifdef CPLEX
static void		destroy_LP_basis (struct basis_save *);
static double		try_branch (LP_t *,
				    int,
				    int,
				    double *,
				    double,
				    struct basis_save *,
				    struct bbinfo *);
static void		save_LP_basis (LP_t *, struct basis_save *);
#endif

/*
 * Set up the initial branch-and-bound problem, including the root
 * node and the initial constraint pool.
 */

	struct bbinfo *
_gst_create_bbinfo (

gst_solver_ptr		solver		/* IN - the solver structure */
)
{
struct bbinfo *		bbip;
int			i;
int			j;
int			k;
int			n;
int			nmasks;
int			nedges;
LP_t *			lp;
bitmap_t *		vert_mask;
bitmap_t *		edge_mask;
bitmap_t *		req_edges;
bitmap_t *		fixed;
bitmap_t *		value;
struct cpool *		cpool;
struct bbstats *	statp;
struct bbtree *		bbtree;
struct bbnode *		root;
struct lpmem *		lpmem;
struct rcon *		rcp;
struct gst_hypergraph *	cip;
gst_param_ptr		params;

	/* Create the global branch-and-bound info structure... */
	bbip = NEW (struct bbinfo);
	memset (bbip, 0, sizeof (*bbip));

	cip	= solver -> H;
	params = solver -> params;

	nmasks = cip -> num_edge_masks;
	nedges = cip -> num_edges;

	vert_mask	= cip -> initial_vert_mask;
	edge_mask	= cip -> initial_edge_mask;
	req_edges	= cip -> required_edges;

	INDENT (params -> print_solve_trace);
	/* initialize global pool of constraints. */
	cpool = NEW (struct cpool);
	_gst_initialize_constraint_pool (cpool, vert_mask, edge_mask, cip, params);

	/* Build initial formulation. */
	lpmem = NEW (struct lpmem);
	lp = _gst_build_initial_formulation (cpool,
					     vert_mask,
					     edge_mask,
					     cip,
					     lpmem,
					     params);
	UNINDENT (params -> print_solve_trace);

	/* Initialize the branch-and-bound tree... */
	bbtree = _gst_create_bbtree (nmasks);

	/* Create vectors to describe the current problem... */
	fixed	= NEWA (nmasks, bitmap_t);
	value	= NEWA (nmasks, bitmap_t);

	for (i = 0; i < nmasks; i++) {
		fixed [i] = 0;
		value [i] = 0;
	}
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (edge_mask, i)) {
			/* variables that are outside of the problem */
			/* are fixed at zero... */
			SETBIT (fixed, i);
			change_var_bounds (lp, i, 0.0, 0.0);
		}
		else if (BITON (req_edges, i)) {
			/* Front-end has determined that this hyperedge	*/
			/* MUST be present in an optimal solution!	*/
			SETBIT (fixed, i);
			SETBIT (value, i);
			change_var_bounds (lp, i, 1.0, 1.0);
		}
	}

	/* Create the root node... */
	root = NEW (struct bbnode);
	memset (root, 0, sizeof (*root));

	root -> owner	= bbip;
	root -> z	= -DBL_MAX;
	root -> delrow_z = - DBL_MAX;
	root -> optimal	= FALSE;
	root -> num	= (bbtree -> snum)++;
	root -> iter	= 0;
	root -> parent	= -1;
	for (i = 0; i < NUM_BB_HEAPS; i++) {
		root -> index [i] = -1;
	}
	root -> var	= -1;
	root -> dir	= 0;
	root -> depth	= 0;
	root -> br1cnt	= 0;
	root -> x	= NEWA (nedges, double);
	root -> cpiter	= -1;		/* x is not current. */
	root -> zlb	= NEWA (2 * nedges, double);
	root -> fixed	= fixed;
	root -> value	= value;
	root -> n_uids	= 0;
	root -> bc_uids	= NULL;
	root -> bc_row	= NULL;
	root -> rstat	= NULL;
	root -> cstat	= NULL;
	root -> bheur	= NEWA (nedges, double);
	root -> next	= NULL;
	root -> prev	= NULL;

	for (i = 0; i < nedges; i++) {
		root -> bheur [i] = 0.0;
	}

	for (i = 0; i < 2*nedges; i++) {
		root -> zlb [i] = -DBL_MAX;
	}

	/* Create the branch-and-bound statistics structure... */
	statp = NEW (struct bbstats);
	memset (statp, 0, sizeof (*statp));

	statp -> num_nodes	= 0;
	statp -> num_lps	= 0;

	statp -> cs_init.num_prows	= cpool -> nrows;
	statp -> cs_init.num_lprows	= GET_LP_NUM_ROWS (lp);
	statp -> cs_init.num_pnz	= cpool -> num_nz;
	statp -> cs_init.num_lpnz	= GET_LP_NUM_NZ (lp);

	/* Fill in the global branch-and-bound info structure... */
	bbip -> cip		= cip;
	bbip -> solver		= solver;
	bbip -> params		= params;
	bbip -> vert_mask	= vert_mask;
	bbip -> edge_mask	= edge_mask;
	bbip -> lp		= lp;
	bbip -> lpmem		= lpmem;
	bbip -> cpool		= cpool;
	bbip -> bbtree		= bbtree;
	bbip -> csip		= NULL;
	bbip -> preempt_z	= params -> initial_upper_bound;
	bbip -> best_z		= solver -> upperbound;
	bbip -> _smt		= NULL;
	bbip -> node		= root;
	bbip -> slack_size	= 0;
	bbip -> slack		= NULL;
	bbip -> dj		= NEWA (nedges, double);
	bbip -> fixed		= NULL;
	bbip -> value		= NULL;
	bbip -> statp		= statp;
	bbip -> prevlb		= -DBL_MAX;
	bbip -> ubip		= NULL;
	bbip -> rcfile		= NULL;
	bbip -> failed_fcomps	= NULL;
	bbip -> next_ckpt_time	= 0;

	/* Make the root node inactive by putting it in the bbtree... */

	n = cpool -> nlprows;
	root -> n_uids	= n;
	root -> bc_uids	= NEWA (n, int);
	root -> bc_row	= NEWA (n, int);

	j = 0;
	rcp = &(cpool -> rows [0]);
	for (i = 0; i < cpool -> nrows; i++, rcp++) {
		k = rcp -> lprow;
		if (k < 0) continue;
		++(rcp -> refc);
		root -> bc_uids [j] = rcp -> uid;
		root -> bc_row [j]  = k;
		++j;
	}
	FATAL_ERROR_IF (j NE n);

	root -> next = NULL;
	root -> prev = NULL;
	bbtree -> first = root;

	_gst_bbheap_insert (root, bbtree, BEST_NODE_HEAP);
	_gst_bbheap_insert (root, bbtree, WORST_NODE_HEAP);

	return (bbip);
}

/*
 * This routine is the top-level of the branch-and-cut.
 */

	void
_gst_branch_and_cut (

gst_solver_ptr		solver	/* IN/OUT - Solver information */
)
{
int			i;
int			j;
int			nmasks;
int			nedges;
int			status;
bitmap_t *		fixed;
bitmap_t *		value;
bitmap_t *		delta;
cpu_time_t		cpu_time_limit;
double *		x;
double			z0;
double			z1;
double			tmpz;
double			best;
cpu_time_t		t1;
struct bbinfo *		bbip;
struct gst_hypergraph *	cip;
LP_t *			lp;
struct cpool *		cpool;
struct bbtree *		bbtree;
struct bbstats *	statp;
struct bbnode *		node;
struct bbnode *		node_to_free;
struct bbnode *		node2;
gst_param_ptr		params;
gst_channel_ptr		trace;

#ifdef CPLEX
int *			b_index;
char *			b_lu;
double *		b_bd;
double			objlim;
double			save_objlim;
#endif

	bbip		= solver -> bbip;
	bbip -> t0	= solver -> t0;
	params		= bbip -> params;

	if (params -> cpu_time_limit > 0) {
		cpu_time_limit = _gst_double_seconds_to_cpu_time_t (
						params -> cpu_time_limit);
		if (cpu_time_limit <= 0) {
			/* Time limit was too small.  Make it	*/
			/* be one "tick", whatever that is...	*/
			cpu_time_limit = 1;
		}
		/* Initialize timing structures */
		bbip -> mainpoll.last		= bbip -> t0;
		bbip -> mainpoll.end_time	= bbip -> t0 + cpu_time_limit;
		bbip -> mainpoll.frequency	= 1;
		bbip -> mainpoll.iteration	= 0;
		bbip -> cglbpoll		= bbip -> mainpoll; /* Copy */

#ifdef DEBUG_CPU_POLL
		bbip -> mainpoll.name = "_gst_branch_and_cut";
		bbip -> cglbpoll.name = "compute_good_lower_bound";
#endif
	}

	cip	= bbip -> cip;
	cpool	= bbip -> cpool;
	lp	= bbip -> lp;
	statp	= bbip -> statp;
	bbtree	= bbip -> bbtree;

	nmasks = cip -> num_edge_masks;
	nedges = cip -> num_edges;

	trace = params -> print_solve_trace;
	if ((params -> check_root_constraints EQ
	     GST_PVAL_CHECK_ROOT_CONSTRAINTS_ENABLE) AND
	    (bbip -> rcfile EQ NULL)) {
		bbip -> rcfile = fopen ("/tmp/lp.x", "w");
		FATAL_ERROR_IF (bbip -> rcfile EQ NULL);
	}

#if CPLEX
	/* Save the existing objective limit, and set it to infinity. */
	CPXgetdblparam (cplex_env, CPX_PARAM_OBJULIM, &save_objlim);
	objlim = DBL_MAX;
	CPXsetdblparam (cplex_env, CPX_PARAM_OBJULIM, objlim);
#endif

	/* Restore upper bound, if available. */
	x = _gst_restore_upper_bound_checkpoint (bbip);
	if (x NE NULL) {
		gst_channel_printf (trace,
			"\tRESTORED UPPER BOUND\n");
		(void) _gst_check_for_better_IFS (x, bbip, &tmpz);
		free ((char *) x);
		x = NULL;
	}

	bbip -> force_branch_flag = FALSE;

	/* Create vectors to describe the current problem... */
	fixed	= NEWA (nmasks, bitmap_t);
	value	= NEWA (nmasks, bitmap_t);
	delta	= NEWA (nmasks, bitmap_t);

	/* No variables have been fixed yet... */
	for (i = 0; i < nmasks; i++) {
		fixed [i] = 0;
		value [i] = 0;
	}
	bbip -> fixed = fixed;
	bbip -> value = value;

#ifdef CPLEX
	/* Create arrays for changing variable bounds... */
	b_index	= NEWA (2 * nedges, int);
	b_lu	= NEWA (2 * nedges, char);
	b_bd	= NEWA (2 * nedges, double);
#endif

#if 0
	/* Build cutset separation formulation.  This is not checkpointed. */
	_gst_build_cutset_separation_formulation (vert_mask,
						  edge_mask,
						  bbip);
#endif

	/* Init the heuristic upper bound.  This is not checkpointed... */

	if (NOT solver -> ubip) {
		solver -> ubip = _gst_startup_heuristic_upper_bound (cip);
	}
	bbip -> ubip = solver -> ubip;

	if (params -> initial_primal_heuristic EQ GST_PVAL_INITIAL_PRIMAL_HEURISTIC_ENABLE) {
		/* Invoke primal heuristic immediately, with no LP	*/
		/* solution as a hint.					*/
		_gst_compute_heuristic_upper_bound (NULL, solver);

		/* Display message indicating new upper bound. */
		_gst_new_upper_bound (bbip -> solver -> upperbound, bbip);

		if (params -> initial_primal_heur_stop EQ GST_PVAL_INITIAL_PRIMAL_HEUR_STOP_ENABLE) {
			solver -> preempt = GST_SOLVE_BB_STOP_REQUESTED;
		}
	}

	/* At this point, all nodes are inactive. */
	for (;;) {
		if (bbip -> solver -> preempt NE 0) {
			/* Computation has been terminated for some reason. */
			break;
		}
		/* Test time limit */
		if (TIME_LIMIT_EXCEEDED (params -> cpu_time_limit,
					 &(bbip -> mainpoll))) {
			bbip -> solver -> preempt = GST_SOLVE_TIME_LIMIT;
			break;
		}

		/* Select the next node to process. */
		node = select_next_node (bbtree);
		if (node EQ NULL) break;

		/* This is perhaps a new lower bound... */
		new_lower_bound (node -> z, bbip);

		if (node -> z > -DBL_MAX) {
			gst_channel_printf (trace,
					    "Resuming node %d at %24.20f\n",
					    node -> num,
					    UNSCALE (node -> z, cip -> scale));
		}
		else {
			gst_channel_printf (trace,
					    "Resuming node %d\n",
					    node -> num);
		}

		INDENT (trace);

		/* Restore the LP tableaux and basis for this node.	*/
		/* Decrement the reference counts on this node's	*/
		/* binding rows.  Since it is now the ACTIVE node,	*/
		/* there is no reason to continue protecting these rows	*/
		/* from deletion by other nodes.			*/
		_gst_restore_node_basis (node, bbip);

		/* Determine new preemption value (i.e. the objective	*/
		/* value of the next-best node).			*/
		update_node_preempt_value (bbip);

		/* Modify LP to represent problem from new node. */
		for (i = 0; i < nmasks; i++) {
			delta [i] =   (fixed [i] ^ node -> fixed [i])
				    | (value [i] ^ node -> value [i]);
		}
#ifdef CPLEX
		j = 0;
		for (i = 0; i < nedges; i++) {
			if (NOT BITON (delta, i)) continue;
			/* Force bounds for variable 'i' to be correct... */
			b_index [j]	= i;	/* variable i, */
			b_lu [j]	= 'L';	/*	lower bound */
			b_index [j+1]	= i;	/* variable i, */
			b_lu [j+1]	= 'U';	/*	upper bound */
			if (NOT BITON (node -> fixed, i)) {
				/* new variable is NOT fixed... */
				b_bd [j]	= 0.0;
				b_bd [j+1]	= 1.0;
			}
			else if (NOT BITON (node -> value, i)) {
				/* new variable is fixed to 0 */
				b_bd [j]	= 0.0;
				b_bd [j+1]	= 0.0;
			}
			else {
				/* new variable is fixed to 1 */
				b_bd [j]	= 1.0;
				b_bd [j+1]	= 1.0;
			}
			j += 2;
		}
		if (j > 0) {
			if (_MYCPX_chgbds (lp, j, b_index, b_lu, b_bd) NE 0) {
				FATAL_ERROR;
			}
#if 0
			++(bbip -> cpool -> uid);
#endif
		}
#endif

#ifdef LPSOLVE
		j = 0;
		for (i = 0; i < nedges; i++) {
			if (NOT BITON (delta, i)) continue;
			++j;
			/* Force bounds on variable 'i' to be correct... */
			if (NOT BITON (node -> fixed, i)) {
				/* variable is NOT fixed... */
				set_bounds (lp, i + 1, 0.0, 1.0);
			}
			else if (NOT BITON (node -> value, i)) {
				/* variable is fixed to 0 */
				set_bounds (lp, i + 1, 0.0, 0.0);
			}
			else {
				/* variable is fixed to 1 -- must set	*/
				/* bounds in this order to avoid	*/
				/* lb > ub condition between calls...	*/
				set_bounds (lp, i + 1, 1.0, 1.0);
			}
		}
		if (j > 0) {
#if 0
			++(bbip -> cpool -> uid);
#endif
		}
#endif

		for (i = 0; i < nmasks; i++) {
			fixed [i] = node -> fixed [i];
			value [i] = node -> value [i];
		}

		/* Set up new node to be processed */
		bbip -> node = node;
		if (node -> iter <= 0) {
			/* Haven't processed this node before	*/
			/* -- tally another node...		*/
			++(statp -> num_nodes);
		}

		/* Process the current node... */
		status = compute_good_lower_bound (bbip);

		/* Note: The previous call changes bbip->lp when	*/
		/* handling the CPLEX "unscaled infeasibility" issue.	*/
		lp = bbip -> lp;

		/* Save for potential access by callbacks. */
		node -> lb_status = status;

		if (node -> depth EQ 0) {
			/* Finished the root node... */
			statp -> root_z = node -> z;
			statp -> root_lps = statp -> num_lps;

			/* Slack rows should have already been deleted... */
			statp -> cs_root.num_prows = cpool -> nrows;
			statp -> cs_root.num_lprows =
				GET_LP_NUM_ROWS (lp);
			statp -> cs_root.num_pnz = cpool -> num_nz;
			statp -> cs_root.num_lpnz =
				GET_LP_NUM_NZ (lp);
			statp -> root_opt = node -> optimal;

			t1 = _gst_get_cpu_time ();
			statp -> root_time = t1 - bbip -> t0;

			if (bbip -> rcfile NE NULL) {
				check_root_constraints (bbip);
			}
		}

		node_to_free = NULL;	/* Default is no node to free... */

		switch (status) {
		case LB_INFEASIBLE:
			/* Node is fathomed! */
			trace_node (bbip, ' ', "infeasible");
			_gst_invoke_bb_callback (GST_CALLBACK_BB_NODE_COMPLETED,
						 node);
			node_to_free = node;
			break;

		case LB_CUTOFF:
			/* Node is fathomed! */
			trace_node (bbip, ' ', "cutoff");
			_gst_invoke_bb_callback (GST_CALLBACK_BB_NODE_COMPLETED,
						 node);
			node_to_free = node;
			break;

		case LB_INTEGRAL:
			best = bbip -> best_z;
			if (_gst_update_best_solution_set (solver, node -> x, 0, NULL, NULL)) {
				_gst_new_upper_bound (node -> z, bbip);
			}
			if (node -> z >= best) {
				trace_node (bbip, ' ', "cutoff");
			}
			else {
				trace_node (bbip, '*', NULL);
			}
			_gst_invoke_bb_callback (GST_CALLBACK_BB_NODE_COMPLETED,
						 node);
			node_to_free = node;
			break;

		case LB_FRACTIONAL:
			/* Check for node limit *before* choosing a	*/
			/* branch variable, which can be expensive.	*/
			if ((params -> bb_node_limit > 0) AND
			    (statp -> num_nodes >= params -> bb_node_limit) AND
			    (bbip -> solver -> preempt EQ 0)) {
				bbip -> solver -> preempt = GST_SOLVE_BB_NODE_LIMIT;
				goto suspend;
			}

			z0 = - DBL_MAX;
			z1 = - DBL_MAX;
			j = choose_branching_variable (bbip, &z0, &z1);
			if (j < 0) {
				/* At least one variable was fixed due	*/
				/* to cutoff or infeasibility.  It is	*/
				/* possible that the entire node is now	*/
				/* cutoff...				*/
				if (node -> z >= bbip -> best_z) {
					trace_node (bbip, ' ', "cutoff");
					_gst_invoke_bb_callback (GST_CALLBACK_BB_NODE_COMPLETED,
								 node);
					node_to_free = node;
					break;
				}
				goto suspend;
			}

			_gst_invoke_bb_callback (GST_CALLBACK_BB_NODE_COMPLETED,
						 node);

			/* Create two nodes... */
			if (z0 < z1) {
				_gst_add_bbnode (bbip, j, 0, z0);
				_gst_add_bbnode (bbip, j, 1, z1);
			}
			else if (z1 < z0) {
				_gst_add_bbnode (bbip, j, 1, z1);
				_gst_add_bbnode (bbip, j, 0, z0);
			}
			else if (UP_FIRST) {	/* To break ties... */
				_gst_add_bbnode (bbip, j, 0, z0);
				_gst_add_bbnode (bbip, j, 1, z1);
			}
			else {
				_gst_add_bbnode (bbip, j, 1, z1);
				_gst_add_bbnode (bbip, j, 0, z0);
			}
			trace_node (bbip, ' ', NULL);

			/* This node is done (became 2 children), free it. */
			node_to_free = node;
			break;

		case LB_PREEMPTED:
suspend:
			gst_channel_printf (trace,
				"suspending node %d at %24.20f\n",
				node -> num,
				UNSCALE (node -> z, cip -> scale));
			/* This node is no longer the best.  Put it	*/
			/* back into the heap and get another one...	*/
			node2 = bbtree -> first;
			if (node2 NE NULL) {
				node2 -> prev = node;
			}
			node -> next = node2;
			node -> prev = NULL;
			bbtree -> first = node;

			_gst_bbheap_insert (node, bbtree, BEST_NODE_HEAP);
			_gst_bbheap_insert (node, bbtree, WORST_NODE_HEAP);

			/* Deactivating this node -- remember the basis */
			_gst_save_node_basis (node, bbip);

			if (_gst_checkpoint_needed (bbip)) {
				/* No nodes are active, write out */
				/*the checkpoint file.		  */
				_gst_write_checkpoint (bbip);
			}

			/* Do NOT free this node! */
			break;
		}

		/* If there is a node to free, do so now... */
		if (node_to_free NE NULL) {
			/* Free up saved basis info and decrement	*/
			/* constraint reference counts before freeing.	*/
			_gst_destroy_node_basis (node_to_free, bbip);

			node_to_free -> next = bbtree -> free;
			bbtree -> free = node_to_free;

			if ((params -> bb_node_limit > 0) AND
			    (statp -> num_nodes >= params -> bb_node_limit) AND
			    (bbip -> solver -> preempt EQ 0)) {
				bbip -> solver -> preempt = GST_SOLVE_BB_NODE_LIMIT;
			}
		}

		UNINDENT (trace);
	}

	statp -> cs_final.num_prows	= cpool -> nrows;
	statp -> cs_final.num_lprows	= GET_LP_NUM_ROWS (lp);
	statp -> cs_final.num_pnz	= cpool -> num_nz;
	statp -> cs_final.num_lpnz	= GET_LP_NUM_NZ (lp);

	if (bbip -> best_z < params -> initial_upper_bound) {
		/* Feasible solution found */
		if (solver -> preempt EQ 0) { /* If no preemption occured */
			new_lower_bound (bbip -> best_z, bbip);
		}
	}

	solver -> lowerbound = bbip -> prevlb;

#if CPLEX
	free ((char *) b_bd);
	free ((char *) b_lu);
	free ((char *) b_index);
#endif

	free ((char *) delta);
	free ((char *) value);
	free ((char *) fixed);

#if CPLEX
	CPXsetdblparam (cplex_env, CPX_PARAM_OBJULIM, save_objlim);
#endif
}

/*
 * This routine selects the next node to process from the given
 * branch-and-bound tree.  This is where we implement the specific
 * search policy.
 */

	static
	struct bbnode *
select_next_node (

struct bbtree *		tp	/* IN - branch-and-bound tree */
)
{
struct bbnode *		p;

	p = NULL;

	if (tp -> first EQ NULL) {
		/* No more nodes! */
		return (NULL);
	}

	switch (tp -> node_policy) {
	case NN_DEPTH_FIRST:
		/* Get node created most recently... */
		p = tp -> first;
		break;

	case NN_BEST_NODE:
		/* Get node with lowest objective function value... */
		p = tp -> heap [BEST_NODE_HEAP].array [0];
		break;

	default:
		FATAL_ERROR;
	}

	_gst_delete_node_from_bbtree (p, tp);

	return (p);
}

/*
 * This routine traces the result for a given node.
 */

	static
	void
trace_node (

struct bbinfo *		bbip,	/* IN - the branch-and-bound info */
char			c1,	/* IN - space or * char */
char *			msg	/* IN - message OR NULL */
)
{
struct bbnode *		p;
struct bbtree *		tp;
struct bbheap *		hp;
char *			p1;
char *			p2;
char			c2;
char			buf1 [32];
char			buf2 [32];
char			buf3 [32];
char			buf4 [32];
char			buf5 [32];
char			buf6 [32];
char			line [256];

	p	= bbip -> node;
	tp	= bbip -> bbtree;

	if (msg EQ NULL) {
		(void) sprintf (buf1, "%14.4f", p -> z);
	}
	else {
		(void) sprintf (buf1, "%14s", msg);
	}
	if (bbip -> best_z EQ DBL_MAX) {
		buf2 [0] = '\0';
	}
	else {
		(void) sprintf (buf2, "%14.4f", bbip -> best_z);
	}
	hp = &(tp -> heap [BEST_NODE_HEAP]);
	if (hp -> nheap > 0) {
		(void) sprintf (buf3,
				"%14.4f",
				hp -> array [0] -> z);
	}
	else {
		buf3 [0] = '\0';
	}
	if (p -> var < 0) {
		buf4 [0] = '\0';
		c2 = ' ';
	}
	else {
		(void) sprintf (buf4, "x%d", p -> var);
		c2 = (p -> dir EQ 0) ? 'D' : 'U';
	}
	if (p -> parent < 0) {
		buf5 [0] = '\0';
	}
	else {
		(void) sprintf (buf5, "%d", p -> parent);
	}
	if (p -> depth <= 0) {
		buf6 [0] = '\0';
	}
	else {
		(void) sprintf (buf6, "%d", p -> depth);
	}

	(void) sprintf (line,
			"%c%6d%6d%14s%14s%14s%6s %c%6s%6s",
			c1,		/* space or * */
			p -> num,	/* node number */
			hp -> nheap,	/* nodes left */
			buf1,		/* objective/cutoff/infeas */
			buf2,		/* best integer soln */
			buf3,		/* best node */
			buf4,		/* variable name */
			c2,		/* branch direction */
			buf5,		/* parent node number */
			buf6);		/* node depth */

	p1 = line;
	for (p2 = p1; *p2 NE '\0'; p2++) {
	}
	while ((p2 > p1) AND (p2 [-1] EQ ' ')) {
		--p2;
	}
	*p2 = '\0';
	(void) gst_channel_printf (bbip -> params -> print_solve_trace, "%s\n", line);
}

/*
 * This routine chooses the next variable to branch on at this
 * node.
 */

	static
	int
choose_branching_variable (

struct bbinfo *		bbip,		/* IN - branch-and-bound info */
double *		z0,		/* OUT - value to give Xi=0 node */
double *		z1		/* OUT - value to give Xi=1 node */
)
{
int			i;
int			nedges;
int			best_var;
struct gst_hypergraph *	cip;
struct bbnode *		nodep;
double *		x;
bitmap_t *		edge_mask;
double			xi;
double			max_infeas;
double			infeas;

	if (bbip->params->branch_var_policy != GST_PVAL_BRANCH_VAR_POLICY_WEAK) {
		/* Do it very carefully! */
		int var;
		INDENT (bbip -> params -> print_solve_trace);
		var = carefully_choose_branching_variable (bbip, z0, z1);
		UNINDENT (bbip -> params -> print_solve_trace);
		return (var);
	}

	/* There are LOTS of things that we MIGHT do here in	*/
	/* the FUTURE!  For right now, just take the variable	*/
	/* that is closest to 1/2 -- take larger variables in	*/
	/* the event of a tie...				*/

	cip		= bbip -> cip;
	nodep		= bbip -> node;
	x		= nodep -> x;
	edge_mask	= bbip -> edge_mask;

	nedges	= cip -> num_edges;

	best_var	= -1;
	max_infeas	= 0.0;

	for (i = nedges - 1; i >= 0; i--) {
		if (NOT BITON (edge_mask, i)) continue;
		xi = x [i];
		if (xi <= FUZZ) continue;
		if (xi + FUZZ >= 1.0) continue;
		infeas = 1.0 - xi;
		if (xi < infeas) {
			infeas = xi;
		}
		if (infeas > max_infeas) {
			best_var = i;
			max_infeas = infeas;
		}
	}
	FATAL_ERROR_IF (best_var < 0);

	/* Give both nodes the same value... */
	*z0	= nodep -> z;
	*z1	= nodep -> z;

	return (best_var);
}

/*
 * This routine does a very careful job of choosing the next variable
 * to branch on.  For each fractional variable Xi, we solve the LP
 * first with Xi=0 and then Xi=1, yielding objective values Zi0 and Zi1
 * correspondingly.  We then choose variable Xi for which the value
 * min(Zi0, Zi1) is MAXIMIZED.  This provides us with the best possible
 * "one-branch/no-cuts" improvement in the lower bound.
 */

	static
	int
carefully_choose_branching_variable (

struct bbinfo *		bbip,		/* IN - branch-and-bound info */
double *		node_z0,	/* OUT - value for Xi=0 node */
double *		node_z1		/* OUT - value for Xi=1 node */
)
{
int			i;
int			j;
int			n;
int			nedges;
int			nfrac;
int			limit;
int			new_limit;
int			logn;
int			failure_limit;
int			num_failures;
struct gst_hypergraph *	cip;
struct bbnode *		nodep;
double *		x;
double *		rank;
bitmap_t *		edge_mask;
int *			fvars;
bool			fixed;
bool			cur_var_is_better;
double			xi;
double			z0;
double			z1;
double			z;
double			test_2nd_val;
double			num;
double			den;
struct bvar		best;
struct basis_save	bsave;
gst_channel_ptr		param_print_solve_trace;

	param_print_solve_trace = bbip -> params -> print_solve_trace;

	cip		= bbip -> cip;
	nodep		= bbip -> node;
	x		= nodep -> x;
	edge_mask	= bbip -> edge_mask;

	nedges	= cip -> num_edges;

	fvars = NEWA (nedges, int);

#if 0
 start_all_over:
#endif

	nfrac = 0;
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (edge_mask, i)) continue;
		xi = x [i];
		if (xi <= FUZZ) continue;
		if (xi + FUZZ >= 1.0) continue;
		fvars [nfrac++] = i;
	}
#if 1
	gst_channel_printf (param_print_solve_trace,
		"\n Carefully choosing branching variable, nfrac = %d\n",
		nfrac);
#endif

	/* Compute final heuristic ranking of the candidate branch	*/
	/* variables.  We multiply the node's "bheur" values by a	*/
	/* "complexity factor" that is 1 for variables sitting at 0.5,	*/
	/* and increases as the variable's "rational" value becomes	*/
	/* more complicated.  Thus, we prefer vars stuck at 1/2, and	*/
	/* then prefer vars stuck at 1/3 or 2/3, vars stuck at 1/4 or	*/
	/* 3/4, etc.							*/

	rank = NEWA (nedges, double);
	memcpy (rank, nodep -> bheur, nedges * sizeof (double));

	for (j = 0; j < nfrac; j++) {
		i = fvars [j];

		/* Compute closest rational approximation. */

		(void) _gst_cra (x [i], &num, &den);

		/* Factor is 1 + the denominator's "distance" from 1/2	*/
		/* + the numerator's "distance" from 1/2.		*/

		rank [i] *= ((den - 1.0) + fabs (num - 0.5 * den));
	}

	/* Sort the fractional variables so that good branch choices	*/
	/* appear first with high probability.				*/

	sort_branching_vars (fvars, nfrac, rank);

#if 0
	gst_channel_printf (param_print_solve_trace,
		" %% Branch Variable Info:\n");
	for (j = 0; j < nfrac; j++) {
		i = fvars [j];
		gst_channel_printf (param_print_solve_trace,
			" %% %d:\tx%d\t= %.6f,\tbheur = %g,\trank = %g\n",
			j, i, x [i], nodep -> bheur [i], rank [i]);
	}
#endif

	free ((char *) rank);

	/* Snapshot the current basis so that we can quickly	*/
	/* get back to it each time...				*/
	save_LP_basis (bbip -> lp, &bsave);

	/* Compute the non-improvement limit.  When we have tested this	*/
	/* many consecutive variables without finding a better choice,	*/
	/* we punt.  We use 2 * log(N), where N is the number of	*/
	/* fractional vars, and log(x) is the floor of the base-2 log.	*/

	failure_limit = nfrac;
	if (nfrac >= 20) {
		logn = 0;
		for (n = nfrac; n > 1; n >>= 1) {
			++logn;
		}
		failure_limit = 2 * bbip->params->check_branch_vars_thoroughly * logn;
	}

	best.var	= -1;
	best.z0		= nodep -> z;
	best.z1		= nodep -> z;

	test_2nd_val	= -DBL_MAX;

	/* Do a quick scan without forcing anything to determine the	*/
	/* best initial choice of branch variable.			*/

	for (j = 0; j < nfrac; j++) {
		i = fvars [j];
		if (i < 0) continue;	/* var was fixed! */

		cur_var_is_better = compare_branch_vars (bbip, i, &best);
		if (cur_var_is_better) {
			test_2nd_val = best.test_2nd_val;
		}
	}

#if 1
	if (best.var >= 0) {
		gst_channel_printf (param_print_solve_trace,
			"Initial guess is x%d,"
			" Z0 = %-24.15g, Z1 = %-24.15g\n\n",
			best.var,
			best.z0,
			best.z1);
	}
#endif

	/* Now do the expensive part -- testing one or both branches	*/
	/* of good candidate variables.					*/

	new_limit = -1;
	limit = nfrac;

	num_failures = 0;

again:

	for (j = 0; j < limit; j++) {

		if (bbip -> force_branch_flag) {
			if (best.var >= 0) goto get_out;

			/* Ignore this interrupt! */
			bbip -> force_branch_flag = FALSE;
		}

		i = fvars [j];
		if (i < 0) continue;	/* var was fixed! */

		xi = x [i];

		z0 = nodep -> zlb [2 * i + 0];
		z1 = nodep -> zlb [2 * i + 1];

		if (((z0 > nodep -> z) AND (z0 < z1)) OR
		    ((z0 EQ z1) AND (xi <= 0.5))) {
			/* Check the Xi=0 branch, and then the Xi=1 branch. */
			fixed = eval_branch_var (bbip,
						 i,
						 0,	/* Xi=0, then Xi=1 */
						 &bsave,
						 test_2nd_val);
		}
		else {
			/* Check the Xi=1 branch, and then the Xi=0 branch. */
			fixed = eval_branch_var (bbip,
						 i,
						 1,	/* Xi=1, then Xi=0 */
						 &bsave,
						 test_2nd_val);
		}

		if (fixed) {
#if 1
			/* Special return code that says to try */
			/* re-solving the LP again.		*/
			destroy_LP_basis (&bsave);
			free ((char *) fvars);
			return (-1);
#elif 0
			goto start_all_over;
#else
			fvars [j] = -1;
			new_limit = j;
			limit = nfrac;
			num_failures = 0;
#endif
		}

		/* Test the current var to see if it is better than the	*/
		/* best seen so var.					*/

		cur_var_is_better = compare_branch_vars (bbip, i, &best);

		if (cur_var_is_better) {
			/* Establish a new threshold for testing 2nd branch. */
			test_2nd_val = best.test_2nd_val;

			z0 = nodep -> zlb [2 * i + 0];
			z1 = nodep -> zlb [2 * i + 1];

			z = z0;
			if (z1 < z) {
				z = z1;
			}

#if 1
			gst_channel_printf (param_print_solve_trace,
				"  New best:  x%d, Z = %-24.15g\n",
				best.var, z);
#endif

			if (z >= bbip -> best_z) {
				/* Nice deal!  This variable	*/
				/* forces a cutoff on both	*/
				/* branches!  No need to look	*/
				/* at the rest...		*/
				new_limit = -1;
				break;
			}
			num_failures = 0;
		}
		else {
			++num_failures;
			if (num_failures >= failure_limit) {
#if 1
				gst_channel_printf (param_print_solve_trace,
					"%d consecutive failures: giving up.\n",
					num_failures);
#endif
				goto get_out;
			}
		}
	}

	FATAL_ERROR_IF (best.var < 0);

	if (new_limit >= 0) {
		/* We have fixed some variables.  Must retest. */
		limit = new_limit;
		new_limit = -1;
		goto again;
	}

get_out:

#if 1
	gst_channel_printf (param_print_solve_trace,
		"Best branch is x%d, Z0 = %-24.15g, Z1 = %-24.15g\n\n",
		best.var, best.z0, best.z1);
#endif

	destroy_LP_basis (&bsave);

	free ((char *) fvars);

	*node_z0 = best.z0;
	*node_z1 = best.z1;

	return (best.var);
}

/*
 * Sort the given list of fractional variables so that the best branching
 * variables appear early in the list with high probability.
 */

	static
	void
sort_branching_vars (

int *			fvars,	/* IN/OUT - fractional variables to sort */
int			n,	/* IN - number of fractional vars */
double *		rank	/* IN - heuristic rank of each variable */
)
{
int			i, i1, i2, j, k;

	/* Construct the heap via sift-downs, in O(n) time. */
	for (k = n >> 1; k >= 0; k--) {
		j = k;
		for (;;) {
			i = (j << 1) + 1;
			if (i + 1 < n) {
				/* Increment i (to right subchild of j) */
				/* if the right subchild is larger. */
				i1 = fvars [i];
				i2 = fvars [i + 1];
				if (rank [i2] > rank [i1]) {
					++i;
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			i1 = fvars [j];
			i2 = fvars [i];
			if (rank [i2] < rank [i1]) {
				/* Largest child is smaller.  Sift-	*/
				/* down is done. */
				break;
			}
			/* Sift down and continue. */
			fvars [j] = i2;
			fvars [i] = i1;
			j = i;
		}
	}

	/* Now do actual sorting.  Exchange first/last and sift down. */
	while (n > 1) {
		/* Largest is at fvars [0], swap with fvars [n-1],	*/
		/* thereby putting it into final position.		*/
		--n;
		i = fvars [0];
		fvars [0] = fvars [n];
		fvars [n] = i;

		/* Now restore the heap by sifting fvars [0] down. */
		j = 0;
		for (;;) {
			i = (j << 1) + 1;
			if (i + 1 < n) {
				/* Increment i (to right subchild of j) */
				/* if the right subchild is larger. */
				i1 = fvars [i];
				i2 = fvars [i + 1];
				if (rank [i2] > rank [i1]) {
					++i;
				}
			}
			if (i >= n) {
				/* Hit bottom of heap, sift-down is done. */
				break;
			}
			i1 = fvars [j];
			i2 = fvars [i];
			if (rank [i2] < rank [i1]) {
				/* Largest child is smaller.  Sift-	*/
				/* down is done. */
				break;
			}
			/* Sift down and continue. */
			fvars [j] = i2;
			fvars [i] = i1;
			j = i;
		}
	}
}

/*
 * This routine does the guts of carefully testing a single fractional
 * branching variable.  The caller indicates which direction should
 * be tested first.
 */

	static
	bool
eval_branch_var (

struct bbinfo *		bbip,		/* IN - branch-and-bound info */
int			var,		/* IN - variable to branch */
int			dir1,		/* IN - first branch direction */
struct basis_save *	basp,		/* IN - basis to restore when done */
double			test_2nd_val	/* IN - test 2nd if 1st is > this */
)
{
int			i;
int			nedges;
int			dir2;
struct gst_hypergraph *	cip;
LP_t *			lp;
struct bbnode *		nodep;
bool			found;
bool			fixed;
double *		x;
double			z;

	cip	= bbip -> cip;
	lp	= bbip -> lp;
	nodep	= bbip -> node;

	nedges = cip -> num_edges;

	x = NEWA (nedges, double);

	dir2 = 1 - dir1;

	fixed = FALSE;

	/* Try the first branch direction... */
#if CPLEX
	z = try_branch (lp, var + 1, dir1, x, DBL_MAX, basp, bbip);
#else
	z = try_branch (lp, var + 1, dir1, x, DBL_MAX, basp);
#endif

	/* Check for a better integer feasible solution... */
	found = _gst_check_for_better_IFS (x, bbip, &z);

	/* Update per-variable lower bounds. */
	i = 2 * var + dir1;
	if (z > nodep -> zlb [i]) {
		nodep -> zlb [i] = z;
	}
	else {
		z = nodep -> zlb [i];
	}

#if 1
	gst_channel_printf (bbip -> params -> print_solve_trace,
		"%s\tx%d = %d,\tZ%d = %-24.15g\n",
		found ? " !!!" : "",
		var, dir1, dir1, z);
#endif

	/* Try finding a good heuristic solution on the branched solution. */
	if (_gst_compute_heuristic_upper_bound (x, bbip -> solver)) {
		_gst_new_upper_bound (bbip -> solver -> upperbound, bbip);
	}

#if 1
	if (z >= bbip -> best_z + 1.0e-8 * fabs (bbip -> best_z)) {
		/* Cutoff or infeasible.  Var must be fixed to	*/
		/* other direction.  Reoptimize and get new	*/
		/* basis.					*/
		SETBIT (bbip -> fixed, var);
		SETBIT (bbip -> node -> fixed, var);
		if (dir1) {
			CLRBIT (bbip -> value, var);
			CLRBIT (bbip -> node -> value, var);
		}
		else {
			SETBIT (bbip -> value, var);
			SETBIT (bbip -> node -> value, var);
		}
		change_var_bounds (lp,
				   var,
				   (double) dir2,
				   (double) dir2);

		nodep -> cpiter = -1;	/* force re-solve of LP */

		_gst_solve_LP_over_constraint_pool (bbip);

		/* Note: The previous call changes bbip->lp when	*/
		/* handling the CPLEX "unscaled infeasibility" issue.	*/
		lp = bbip -> lp;

		destroy_LP_basis (basp);
		save_LP_basis (lp, basp);

		/* Try finding a good heuristic solution on the	*/
		/* new fixed solution...			*/
		if (_gst_compute_heuristic_upper_bound (bbip -> node -> x,
							bbip -> solver)) {
			_gst_new_upper_bound (bbip -> solver -> upperbound, bbip);
		}

		/* The variable has been fixed! */
		fixed = TRUE;
		goto all_done;
	}
#endif

	if (z <= test_2nd_val) {
		/* No need to test the second branch. */
		goto all_done;
	}

	/* Try the second branch direction... */
#if CPLEX
	z = try_branch (lp, var + 1, dir2, x, DBL_MAX, basp, bbip);
#else
	z = try_branch (lp, var + 1, dir2, x, DBL_MAX, basp);
#endif

	/* Check for better integer feasible solution... */
	found = _gst_check_for_better_IFS (x, bbip, &z);

	/* Update per-variable lower bounds. */
	i = 2 * var + dir2;
	if (z > nodep -> zlb [i]) {
		nodep -> zlb [i] = z;
	}
	else {
		z = nodep -> zlb [i];
	}

#if 1
	gst_channel_printf (bbip -> params -> print_solve_trace,
		"%s\tx%d = %d,\tZ%d = %-24.15g\n",
		found ? " !!!" : "",
		var, dir2, dir2, z);
#endif

	/* Try finding a good heuristic solution on the branched solution. */
	if (_gst_compute_heuristic_upper_bound (x, bbip -> solver)) {
		_gst_new_upper_bound (bbip -> solver -> upperbound, bbip);
	}

#if 1
	if (z >= bbip -> best_z + 1.0e-8 * fabs (bbip -> best_z)) {
		/* Cutoff or infeasible.  Var must be fixed to	*/
		/* other direction.  Reoptimize and get new	*/
		/* basis.					*/
		SETBIT (bbip -> fixed, var);
		SETBIT (bbip -> node -> fixed, var);
		if (dir2) {
			CLRBIT (bbip -> value, var);
			CLRBIT (bbip -> node -> value, var);
		}
		else {
			SETBIT (bbip -> value, var);
			SETBIT (bbip -> node -> value, var);
		}
		change_var_bounds (lp,
				   var,
				   (double) dir1,
				   (double) dir1);

		nodep -> cpiter = -1;	/* force re-solve of LP */

		_gst_solve_LP_over_constraint_pool (bbip);

		/* Note: The previous call changes bbip->lp when	*/
		/* handling the CPLEX "unscaled infeasibility" issue.	*/
		lp = bbip -> lp;

		destroy_LP_basis (basp);
		save_LP_basis (lp, basp);

		/* Try finding a good heuristic solution on the	*/
		/* new fixed solution...			*/
		if (_gst_compute_heuristic_upper_bound (bbip -> node -> x,
							bbip -> solver)) {
			_gst_new_upper_bound (bbip -> solver -> upperbound, bbip);
		}

		/* The variable has been fixed! */
		fixed = TRUE;
	}
#endif

all_done:

	free ((char *) x);

	return (fixed);
}

/*
 * See if one candidate branch variable is better than another.
 * We implement various policies here.
 */

	static
	bool
compare_branch_vars (

struct bbinfo *		bbip,		/* IN - branch and bound info */
int			i1,		/* IN - first branch var */
struct bvar *		bvp		/* IN/OUT - current best branch var */
)
{
struct bbnode *		nodep;
bool			cur_var_is_better;
double			z;
double			z0;
double			z1;
double			zmin;
double			zmax;
double			best_z0;
double			best_z1;
double			best_zmin;
double			best_zmax;
double			ub;
double			gap;
double			delta;
double			prod;
double			best_prod;
double			test2;

#define TOLERANCE	1.0e-10

	nodep	= bbip -> node;

	ub	= bbip -> best_z;
	z	= nodep -> z;

	if (i1 < 0) {
		return (FALSE);
	}

	z0	= nodep -> zlb [2 * i1 + 0];
	z1	= nodep -> zlb [2 * i1 + 1];

	if (z0 < z1) {
		zmin = z0;
		zmax = z1;
	}
	else {
		zmin = z1;
		zmax = z0;
	}

	if (bvp -> var < 0) {
		/* New branch var is better.  Still have to provide a	*/
		/* test_2nd_val threshold, which is policy specific.	*/

		test2 = - DBL_MAX;
		switch (bbip->params->branch_var_policy) {
		case GST_PVAL_BRANCH_VAR_POLICY_NAIVE:
			test2 = zmin;
			break;

		case GST_PVAL_BRANCH_VAR_POLICY_SMART:
			test2 = zmin - TOLERANCE * fabs (zmin);
			break;

		case GST_PVAL_BRANCH_VAR_POLICY_PROD:
			gap = fabs (ub - z);
			FATAL_ERROR_IF (gap <= 0.0);
			prod = fabs ((z0 - z) * (z1 - z));
			test2 = z + prod / gap;
			break;

		default:
			FATAL_ERROR;
			break;
		}

		bvp -> var		= i1;
		bvp -> z0		= z0;
		bvp -> z1		= z1;
		bvp -> test_2nd_val	= test2;

		return (TRUE);
	}

	best_z0	= bvp -> z0;
	best_z1	= bvp -> z1;

	if (best_z0 < best_z1) {
		best_zmin = best_z0;
		best_zmax = best_z1;
	}
	else {
		best_zmin = best_z1;
		best_zmax = best_z0;
	}

	cur_var_is_better = FALSE;

	switch (bbip->params->branch_var_policy) {
	case GST_PVAL_BRANCH_VAR_POLICY_NAIVE:
		/* Naive max of mins. */
		cur_var_is_better = FALSE;
		if (zmin > best_zmin) {
			cur_var_is_better = TRUE;
			test2 = zmin;
		}
		break;

	case GST_PVAL_BRANCH_VAR_POLICY_SMART:
		/* Smarter lexicographic max of mins.  If the mins	*/
		/* are "about equal", use the max values to decide.	*/
fuzzy_lexical:
		cur_var_is_better = FALSE;

		delta = TOLERANCE * fabs (best_zmin);
		if ((zmin - best_zmin) > delta) {
			cur_var_is_better = TRUE;
		}
		else if ((fabs (zmin - best_zmin) <= delta) AND
			 (zmax > best_zmax)) {
			cur_var_is_better = TRUE;
		}
		test2 = zmin - TOLERANCE * fabs (zmin);
		break;

	case GST_PVAL_BRANCH_VAR_POLICY_PROD:
		/* Product of improvements.  Uses method 1 to break	*/
		/* close ties.						*/
		prod = fabs ((z0 - z) * (z1 - z));
		best_prod = fabs ((best_z0 - z) * (best_z1 - z));

		/* Compute tolerance factor that is a fraction of the	*/
		/* current gap.						*/
		gap = fabs (ub - z);
		FATAL_ERROR_IF (gap <= 0.0);
		delta = 1.0E-5 * gap;
		if (fabs (prod - best_prod) <= delta) {
			/* Products are nearly equal.  Use fuzzy	*/
			/* lexicographic max of mins.			*/
			goto fuzzy_lexical;
		}
		cur_var_is_better = FALSE;
		if (prod > best_prod) {
			cur_var_is_better = TRUE;
			test2 = z + prod / gap;
		}
		break;

	default:
		FATAL_ERROR;
		break;
	}

	if (cur_var_is_better) {

		bvp -> var		= i1;
		bvp -> z0		= z0;
		bvp -> z1		= z1;
		bvp -> test_2nd_val	= test2;
	}

	return (cur_var_is_better);

#undef TOLERANCE
}

/*
 * This routine checks to see if the result of doing a "test-branch" on
 * a variable JUST HAPPENS to result in an integer feasible solution
 * that is the best so far.  Although this would be total serendipity,
 * we do it anyway.  The check takes virtually no time because we almost
 * always locate a fractional variable within the first few probes.
 * And besides, it would be a shame to NOT notice something this important
 * -- especially if we don't have ANY upper bound yet!
 */

	bool
_gst_check_for_better_IFS (

double *		x,		/* IN - solution to test */
struct bbinfo *		bbip,		/* IN - branch and bound info */
double *		true_z		/* OUT - true Z value, if integral */
)
{
int			i;
int			nedges;
struct gst_hypergraph *	cip;
double			z;

	cip = bbip -> cip;
	nedges	= cip -> num_edges;

	/* The special try_branch code for lp_solve can leave us with	*/
	/* an LP solution that isn't primal feasible.  In fact, it may	*/
	/* not even satisfy the variable bounds!  Detect this case	*/
	/* right up front.						*/

	for (i = 0; i < nedges; i++) {
		if (x [i] < -FUZZ) return (FALSE);
		if (x [i] > 1.0 + FUZZ) return (FALSE);
	}

	if (NOT integer_feasible_solution (x, bbip)) {
		/* Not integer feasible -- get out. */
		return (FALSE);
	}

	/* We literally stumbled across an Integer Feasible Solution!	*/
	/* Store it away if it is sufficiently good, and update the	*/
	/* upper bound if it happens to be the best so far.		*/

	if (_gst_update_best_solution_set (bbip -> solver, x, 0, NULL, NULL)) {
		/* Give caller the correct Z value... */
		z = bbip -> solver -> solutions [0].length;
		*true_z = z;
		/* We have a new best solution! */
		_gst_new_upper_bound (z, bbip);
		return (TRUE);
	}

	return (FALSE);
}

/*
 * This routine saves the current basis of the given LP.
 */

#ifdef CPLEX

	static
	void
save_LP_basis (

LP_t *			lp,		/* IN - LP to save basis for */
struct basis_save *	basp		/* OUT - saved basis info */
)
{
int		rows;
int		cols;

	rows = _MYCPX_getnumrows (lp);
	cols = _MYCPX_getnumcols (lp);

	basp -> cstat = NEWA (cols, int);
	basp -> rstat = NEWA (rows, int);

	if (_MYCPX_getbase (lp, basp -> cstat, basp -> rstat) NE 0) {
		FATAL_ERROR;
	}
}

/*
 * Destroy the saved basis info...
 */

	static
	void
destroy_LP_basis (

struct basis_save *	basp		/* IN - basis info to free up */
)
{
#if CPLEX >= 40
	free ((char *) (basp -> rstat));
	free ((char *) (basp -> cstat));
#endif
}

#endif

/*
 * This routine tries the given branch by solving the LP.  It
 * returns the resulting objective value, or INF_DISTANCE if something
 * goes wrong (like infeasible).
 */

#ifdef CPLEX

	static
	double
try_branch (

LP_t *			lp,		/* IN - LP to re-optimize */
int			var,		/* IN - variable to try branching */
int			dir,		/* IN - branch direction, 0 or 1 */
double *		x,		/* OUT - LP solution obtained */
double			ival,		/* IN - value to give if infeasible */
struct basis_save *	basp,		/* IN - basis to restore when done */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int		status;
gst_param_ptr	params;
double		z;
int		b_index [2];
char		b_lu [2];
double		b_bd [2];

	params = bbip -> params;

	--var;		/* vars are zero-origined in CPLEX... */

	b_index [0] = var;	b_lu [0] = 'L';
	b_index [1] = var;	b_lu [1] = 'U';
	if (dir EQ 0) {
		b_bd [0] = 0.0;
		b_bd [1] = 0.0;
	}
	else {
		b_bd [0] = 1.0;
		b_bd [1] = 1.0;
	}
	if (_MYCPX_chgbds (lp, 2, b_index, b_lu, b_bd) NE 0) {
		FATAL_ERROR;
	}

	/* Solve the current LP instance... */
	status = _MYCPX_dualopt (lp);
	if (status NE 0) {
		gst_channel_printf (params -> print_solve_trace,
			" WARNING dualopt: status = %d\n", status);
	}

	/* Get current LP solution... */
	if (_MYCPX_solution (lp, &status, &z, x, NULL, NULL, NULL) NE 0) {
		FATAL_ERROR;
	}

	/* Determine type of LP result... */
	switch (status) {
	case _MYCPX_STAT_OPTIMAL:
	case _MYCPX_STAT_OPTIMAL_INFEAS:
		/* Unscale the objective value. */
		z = ldexp (z, bbip -> lpmem -> obj_scale);
		break;

	case _MYCPX_STAT_INFEASIBLE:
	case _MYCPX_STAT_UNBOUNDED:
			/* (CPLEX 3.0 sometimes gives us infeasible!) */
	case _MYCPX_STAT_ABORT_OBJ_LIM:	/* Objective limit exceeded. */
		z = ival;
		break;

	default:
		gst_channel_printf (params -> print_solve_trace, "Status = %d\n", status);
		_MYCPX_lpwrite (lp, "core.lp");
		FATAL_ERROR;
		break;
	}

	b_bd [0] = 0.0;
	b_bd [1] = 1.0;
	if (_MYCPX_chgbds (lp, 2, b_index, b_lu, b_bd) NE 0) {
		FATAL_ERROR;
	}

	/* Restore the basis... */
	status = _MYCPX_copybase (lp, basp -> cstat, basp -> rstat);
	if (status NE 0) {
		fprintf (stderr, "try_branch: status = %d\n", status);
		FATAL_ERROR;
	}

	return (z);
}

#endif

/*
 * This routine computes the lower-bound for the current node, which
 * consists of solving the LP and generating violated constraints
 * until either:
 *
 *	- LP becomes infeasible
 *	- LP objective meets or exceeds cutoff value
 *	- LP solution is integral
 *	- separation finds no more violated constraints
 */

	static
	int
compute_good_lower_bound (

struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i, n;
int			num_const;
int			status;
struct gst_hypergraph *	cip;
struct bbnode *		nodep;
double *		x;
double			z;
struct constraint *	cp;
struct constraint *	tmp;
int			iteration;
int			fix_status;
int			num_fractional;
cpu_time_t		Tlp;
cpu_time_t *		Tp;
bool			is_int;
char *			tsp;
char *			tsp_endp;
char			time_str [128];
char			title [256];
cpu_time_t		Tn [20];
gst_param_ptr		params;
gst_channel_ptr		param_print_solve_trace;

	params = bbip -> params;
	param_print_solve_trace = params -> print_solve_trace;

	cip	  = bbip -> cip;
	nodep	  = bbip -> node;
	x	  = nodep -> x;

	Tp = &Tn [0];

	*Tp++ = _gst_get_cpu_time ();

	iteration = 1;
	num_const = 0;

	for (;;) {
		status = _gst_solve_LP_over_constraint_pool (bbip);

		/* Note: The previous call changes bbip->lp when	*/
		/* handling the CPLEX "unscaled infeasibility" issue.	*/

		z = nodep -> z;

		Tlp = _gst_get_cpu_time ();

		++(bbip -> node -> iter);
		++(bbip -> statp -> num_lps);

#if 0
		/* Display LP solution vector in machine-readable form... */
		for (i = 0; i < cip -> num_edges; i++) {
			gst_channel_printf (param_print_solve_trace,
				" %% %08lx %08lx\n",
				((bitmap_t *) &x [i]) [0],
				((bitmap_t *) &x [i]) [1]);
		}
#endif

		if ((bbip -> rcfile NE NULL) AND
		    (bbip -> node -> depth EQ 0)) {
			fwrite (x,
				1,
				cip -> num_edges * sizeof (*x),
				bbip -> rcfile);
		}

		_gst_invoke_bb_callback (GST_CALLBACK_BB_LP_SOLVED,
					 bbip -> node);

#if 1
		tsp = &time_str [0];
		tsp_endp = tsp + sizeof (time_str);
		n = Tp - &Tn [0];
		Tn [n] = Tlp;
		_gst_convert_cpu_time (Tn [1] - Tn [0], tsp);
		tsp = strchr (tsp, '\0');
		for (i = 2; i <= n; i++) {
			*tsp++ = '/';
			_gst_convert_cpu_time (Tn [i] - Tn [i - 1], tsp);
			tsp = strchr (tsp, '\0');
			FATAL_ERROR_IF (tsp >= tsp_endp);
		}
		(void) sprintf (title,
				"Node %d LP %d Solution, length = %f, %s %d",
				bbip -> node -> num, bbip -> node -> iter,
				z, time_str, num_const);
#if 0
		/* Two problems here:					*/
		/* 1. This is a library resident file calling		*/
		/*    non-library routines (genps.c is not in the	*/
		/*    library), and					*/
		/* 2. The postscript we generate is going to be stuck	*/
		/*    inside of postscript comments(!) unless we	*/
		/*    disable postscript commenting on the channel.	*/
		_gst_plot_lp_solution (param_print_solve_trace,
				       cip, title, x, BIG_PLOT);
#else
		INDENT (param_print_solve_trace);
		(void) gst_channel_printf (bbip -> params -> print_solve_trace, "%s\n", title);
		UNINDENT (param_print_solve_trace);
#endif
#endif

		if (status NE BBLP_INFEASIBLE) {
			/* Have a feasible LP.  Display number of	*/
			/* fractional variables.			*/
			num_fractional = num_fractional_vars (x, bbip);

			gst_channel_printf (param_print_solve_trace,
				"%d fractional variables\n", num_fractional);
		}

		switch (status) {
		case BBLP_OPTIMAL:
			if (z >= bbip -> best_z) {
				nodep -> z = bbip -> best_z;
				return (LB_CUTOFF);
			}
			break;

		case BBLP_CUTOFF:
			nodep -> z = bbip -> best_z;
			return (LB_CUTOFF);

		case BBLP_INFEASIBLE:
			nodep -> z = bbip -> best_z;
			return (LB_INFEASIBLE);

		default:
			gst_channel_printf (param_print_solve_trace,
				"solve status = %d\n", status);
			FATAL_ERROR;
		}

#ifdef CPLEX
		/* Now get rid of any rows that have become	*/
		/* slack.  (We don't lose these constraints:	*/
		/* they're still sitting around in the		*/
		/* constraint pool.)				*/
		_gst_delete_slack_rows_from_LP (bbip);
#endif

		/* Solution is feasible, check for integer-feasible... */
		is_int = integer_feasible_solution (x, bbip);

		if (is_int) {
			/* All vars are either 0 or 1 and the	*/
			/* solution is connected -- we have a	*/
			/* Steiner tree!			*/

			/* Re-calculate the final objective	*/
			/* function to eliminate numerical	*/
			/* errors in the value of Z...		*/
			z = 0.0;
			for (i = 0; i < cip -> num_edges; i++) {
				if (x [i] + FUZZ < 1.0) continue;
				z += cip -> cost [i];
			}
			nodep -> z = z;
			if (z >= bbip -> best_z) {
				/* probably a repeat performance... */
				nodep -> z = bbip -> best_z;
				return (LB_CUTOFF);
			}
			bbip -> node -> optimal = TRUE;
			return (LB_INTEGRAL);
		}

		/* Check to see if this node's objective value	*/
		/* is now high enough to be preempted...	*/
		if (nodep -> z > bbip -> preempt_z) {
			/* Node's value is no longer the lowest...	*/
			/* Preempt this one in favor of another.	*/
			return (LB_PREEMPTED);
		}

		/* Perhaps we have a new lower bound? */
		new_lower_bound (z, bbip);

		if (TIME_LIMIT_EXCEEDED (params -> cpu_time_limit,
					 &(bbip -> cglbpoll))) {
			bbip -> solver -> preempt = GST_SOLVE_TIME_LIMIT;
			return LB_PREEMPTED;
		}

		if (_gst_checkpoint_needed (bbip)) {
			/* Need to write out a checkpoint file.  We	*/
			/* only do this when all nodes are inactive, so	*/
			/* pretend that this node was preempted by	*/
			/* another.  The checkpoint will be taken after	*/
			/* the current node is deactivated.		*/
			return (LB_PREEMPTED);
		}

		if (bbip -> solver -> preempt NE 0) {
			/* Something has happened that says we should	*/
			/* stop / suspend the computation.		*/
			return (LB_PREEMPTED);
		}

		Tp = &Tn [0];
		*Tp++ = _gst_get_cpu_time ();

		if (_gst_compute_heuristic_upper_bound (x, bbip -> solver)) {
			_gst_new_upper_bound (bbip -> solver -> upperbound, bbip);
		}

		/* Check if LP limit reached.  (Note that we defer this	*/
		/* test until after we have run the upper bound		*/
		/* heuristic on this LP solution.)			*/
		if ((params -> bb_lp_limit > 0) AND
		    (bbip -> statp -> num_lps >= params -> bb_lp_limit) AND
		    (bbip -> solver -> preempt EQ 0)) {
			bbip -> solver -> preempt = GST_SOLVE_BB_LP_LIMIT;
		}

		if (bbip -> solver -> preempt NE 0) {
			/* Something has happened that says we should	*/
			/* stop / suspend the computation.		*/
			return (LB_PREEMPTED);
		}

		/* If we have improved the upper bound, it is possible	*/
		/* that this node can now be cutoff...			*/
		if (nodep -> z >= bbip -> best_z) {
			nodep -> z = bbip -> best_z;
			return (LB_CUTOFF);
		}

		/* Try to fix some variables using reduced costs... */
		fix_status = reduced_cost_var_fixing (bbip);
		if (fix_status EQ VFIX_INFEASIBLE) {
			nodep -> z = bbip -> best_z;
			return (LB_INFEASIBLE);
		}
		if (fix_status EQ VFIX_FIXED_FRACTIONAL) {
			continue;
		}

		if (bbip -> force_branch_flag) {
			/* User kicked us!  Stop separating and branch! */
			bbip -> force_branch_flag = FALSE;
			break;
		}

		/* Apply all separation algorithms to solution... */
		cp = do_separations (bbip, &Tp);

		if (cp EQ NULL) {
			/* No more violated constraints found! */
			break;
		}

#ifdef LPSOLVE
		/* Now get rid of any rows that have become	*/
		/* slack.  (We don't lose these constraints:	*/
		/* they're still sitting around in the		*/
		/* constraint pool.)				*/
		_gst_delete_slack_rows_from_LP (bbip);
#endif

		/* Add new contraints to the constraint pool. */
		num_const = _gst_add_constraints (bbip, cp);

		if (num_const <= 0) {
			/* Separation routines found violations, but	*/
			/* the constraint pool disagrees...		*/
			FATAL_ERROR;
		}

		while (cp NE NULL) {
			tmp = cp;
			cp = tmp -> next;
			free ((char *) (tmp -> mask));
			free ((char *) tmp);
		}
		++iteration;
	}

#if 1
	/* Print execution times of final iteration... */
	tsp = &time_str [0];
	tsp_endp = tsp + sizeof (time_str);
	n = Tp - &Tn [0];
	Tn [n] = Tn [n - 1];
	_gst_convert_cpu_time (Tn [1] - Tn [0], tsp);
	tsp = strchr (tsp, '\0');
	for (i = 2; i <= n; i++) {
		*tsp++ = '/';
		_gst_convert_cpu_time (Tn [i] - Tn [i - 1], tsp);
		tsp = strchr (tsp, '\0');
		FATAL_ERROR_IF (tsp >= tsp_endp);
	}
	(void) gst_channel_printf (param_print_solve_trace,
		"  Final iteration: %s\n", time_str);
#endif

	/* Only get here with fractional solution and	*/
	/* no more violated constraints were found.	*/

	return (LB_FRACTIONAL);
}

/*
 * Routine to print out an updated lower bound value.
 */

	static
	void
new_lower_bound (

double			lb,		/* IN - new lower bound value */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
struct gst_hypergraph *	cip;
double			prev;
double			old_gap;
double			new_gap;
cpu_time_t		t1;
char			buf1 [32];
gst_param_ptr		params;

	params = bbip -> params;

	cip = bbip -> cip;

	prev = bbip -> prevlb;
	if (lb <= prev) {
		/* There has been no improvement - get out. */
		return;
	}

	if (prev <= -DBL_MAX) {
		/* Don't make lower bound jump from initial value... */
		prev = lb;
	}

	/* Print out the old and new lower bounds, with timestamp. */
	t1 = _gst_get_cpu_time ();
	_gst_convert_cpu_time (t1 - bbip -> t0, buf1);

	if ((bbip -> best_z >= DBL_MAX) OR (bbip -> best_z EQ 0.0)) {
		old_gap = 99.9;
		new_gap = 99.9;
	}
	else {
		old_gap = 100.0 * (bbip -> best_z - prev) / bbip -> best_z;
		new_gap = 100.0 * (bbip -> best_z - lb) / bbip -> best_z;
	}

	if ((params -> gap_target NE 1.0) AND (lb > -DBL_MAX) AND
	     (bbip -> best_z < params -> gap_target * lb)) {
		PREEMPT_SOLVER (bbip -> solver, GST_SOLVE_GAP_TARGET);
	}

	if (lb >= params -> lower_bound_target) {
		PREEMPT_SOLVER (bbip -> solver, GST_SOLVE_LOWER_BOUND_TARGET);
	}

	gst_channel_printf (bbip -> params -> print_solve_trace,
		"@LO %s %24.20f %2.10f\n",
		buf1, UNSCALE (prev, cip -> scale), old_gap);
	gst_channel_printf (bbip -> params -> print_solve_trace,
		"@LN %s %24.20f %2.10f\n",
		buf1, UNSCALE (lb, cip -> scale), new_gap);

	bbip -> prevlb = lb;
}

/*
 * Routine to print out an updated upper bound value.
 */

	void
_gst_new_upper_bound (

double			ub,		/* IN - new upper bound value */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
struct gst_hypergraph *	cip;
double			prev;
cpu_time_t		t1;
double			old_gap;
double			new_gap;
char			buf1 [64];
gst_param_ptr		params;

	params = bbip -> params;

	cip = bbip -> cip;

	_gst_invoke_bb_callback (GST_CALLBACK_BB_NEW_UPPER_BOUND,
				 bbip -> node);

	prev = bbip -> best_z;
	if (ub >= prev) {
		/* Supposed to be an improvement! */
		FATAL_ERROR;
	}

	/* We HAVE a new best solution! */
	bbip -> best_z = ub;

#if CPLEX
	{ double toobig, toosmall, ulim;
	  /* Set new cutoff value for future LPs... */
	  ulim = ldexp (ub, -(bbip -> lpmem -> obj_scale));
	  if (_MYCPX_setobjulim (ulim, &toosmall, &toobig) NE 0) {
		FATAL_ERROR;
	  }
	}
#endif

#if LPSOLVE
	/* Set new cutoff value for future LPs... */
	/* (This may not really work in lp_solve.) */
	bbip -> lp -> obj_bound = ub;
#endif

	cut_off_existing_nodes (ub, bbip);

	/* Might want to do this if all other nodes were cut off. */
	update_node_preempt_value (bbip);

	/* Now print out the trace messages. */
	if (prev >= DBL_MAX) {
		/* Don't make upper bound jump from infinity... */
		prev = ub;
	}

	/* Print out the old and new lower and upper bounds, with timestamp. */
	t1 = _gst_get_cpu_time ();
	_gst_convert_cpu_time (t1 - bbip -> t0, buf1);

	if (bbip -> prevlb <= -DBL_MAX) {
		old_gap = 99.9;
		new_gap = 99.9;
	}
	else {
		old_gap = 100.0 * (prev - bbip -> prevlb) / prev;
		new_gap = 100.0 * (ub - bbip -> prevlb) / ub;
	}

	if ((params -> gap_target NE 1.0) AND (bbip -> prevlb > -DBL_MAX) AND
	    (ub < params -> gap_target * bbip -> prevlb)) {
		PREEMPT_SOLVER (bbip -> solver, GST_SOLVE_GAP_TARGET);
	}

	if (ub <= params -> upper_bound_target) {
		PREEMPT_SOLVER (bbip -> solver, GST_SOLVE_UPPER_BOUND_TARGET);
	}

	gst_channel_printf (params -> print_solve_trace,
		"@UO %s %24.20f %2.10f\n",
		buf1, UNSCALE (prev, cip -> scale), old_gap);
	gst_channel_printf (params -> print_solve_trace,
		"@UN %s %24.20f %2.10f\n",
		buf1, UNSCALE (ub, cip -> scale), new_gap);

	/* Write out checkpoint file for the new upper bound. */
	_gst_write_upper_bound_checkpoint (bbip);
}

/*
 * This routine deletes any existing node whose objective value is
 * cut off by the given latest feasible integer solution.
 */

	static
	void
cut_off_existing_nodes (

double		best_z,		/* IN - new best objective value */
struct bbinfo *	bbip		/* IN - branch-and-bound info */
)
{
int		num_cut;
struct bbtree *	tp;
struct bbheap *	hp;
struct bbnode *	p;

	tp = bbip -> bbtree;

	num_cut = 0;

	/* We process the nodes from WORST to best... */
	hp = &(tp -> heap [WORST_NODE_HEAP]);

	while (hp -> nheap > 0) {
		/* Get node with highest objective value... */
		p = hp -> array [0];
		FATAL_ERROR_IF (p -> index [WORST_NODE_HEAP] NE 0);
		if (p -> z < best_z) {
			/* All remaining nodes are < best_z... */
			break;
		}

		/* This node has been cut off! */
		_gst_delete_node_from_bbtree (p, tp);
		/* Free up saved basis info and decrement	*/
		/* constraint reference counts before freeing.	*/
		_gst_destroy_node_basis (p, bbip);
		p -> next = tp -> free;
		tp -> free = p;
		++num_cut;
	}

	if (num_cut > 0) {
		gst_channel_printf (bbip -> params -> print_solve_trace,
				    " 	=== %d nodes cut off ===\n",
				    num_cut);
	}
}

/*
 * This routine updates the node preemption value.  The node to be
 * preempted must be active when this routine is called (i.e., it must
 * be removed from the heap so that the "next best" node is at the top
 * of the heap).
 */

	static
	void
update_node_preempt_value (

struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
struct bbtree *		tp;
struct bbheap *		hp;
struct bbnode *		node2;

	tp = bbip -> bbtree;
	hp = &(tp -> heap [BEST_NODE_HEAP]);

	if (hp -> nheap <= 0) {
		/* No other nodes.  Preempt only at cutoff. */
		bbip -> preempt_z = bbip -> best_z;
	}
	else {
		/* Preempt current node when next-best is exceeded. */
		node2 = hp -> array [0];
		bbip -> preempt_z = node2 -> z;
	}
}

/*
 * This routine performs most of the separations -- in the proper order.
 */

	static
	struct constraint *
do_separations (

struct bbinfo *		bbip,		/* IN - branch-and-bound info */
cpu_time_t **		Tpp		/* IN/OUT - CPU time vector */
)
{
double *		x;
struct comp *		comp;
struct comp *		p;
struct comp **		hookp;
struct comp *		p2;
cpu_time_t *		Tp;
struct constraint *	cp;
struct constraint *	cp2;
struct constraint *	tmp;
bool			print_flag;
bool			optimal;
bool			try_localcuts;

	x		= bbip -> node -> x;

	Tp = *Tpp;

	/* Find all zero-weight cutsets... */
	cp = _gst_find_zero_weight_cutsets (x, bbip);
	*Tp++ = _gst_get_cpu_time ();

	/* Find solid integer cycles... */
	cp = _gst_find_integer_cycles (x, cp, bbip);
	*Tp++ = _gst_get_cpu_time ();

#if 0
	cp = _gst_find_weak_connectivity (x, cp, bbip);
#endif

	/* Break problem up into congested components... */
	print_flag = TRUE;
	comp = _gst_find_congested_components (x,
					       bbip -> vert_mask,
					       bbip -> edge_mask,
					       print_flag,
					       bbip);

	try_localcuts = FALSE;

	/* Exhaustively enumerate all components that are sufficiently	*/
	/* small...  Delete them from the list when done.		*/
	hookp = &comp;
	while ((p = *hookp) NE NULL) {
		if (p -> num_verts <= bbip -> params -> sec_enum_limit) {
			cp2 = _gst_enumerate_all_subtours (p, NULL, bbip);
			if (cp2 EQ NULL) {
#if 0
				/* Try finding a local cut for component. */
				cp2 = _gst_find_local_cuts_in_component (
								p,
								x,
								bbip,
								NULL);
#elif 1
				if ((bbip -> params -> local_cuts_mode EQ
				     GST_PVAL_LOCAL_CUTS_MODE_SUBTOUR_COMPONENTS) OR
				    (bbip -> params -> local_cuts_mode EQ
				     GST_PVAL_LOCAL_CUTS_MODE_BOTH)) {
					try_localcuts = TRUE;
					/* Keep this component. */
					hookp = &(p -> next);
					continue;
				}
#endif
			}
			*hookp = p -> next;
			p -> next = NULL;
			_gst_free_congested_component (p);
			while (cp2 NE NULL) {
				tmp = cp2 -> next;
				cp2 -> next = cp;
				cp = cp2;
				cp2 = tmp;
			}
		}
		else {
			hookp = &(p -> next);
		}
	}
	*Tp++ = _gst_get_cpu_time ();

	/* Find violated SEC's using a heuristic flow	*/
	/* formulation.					*/
	for (p = comp; p NE NULL; p = p -> next) {
		p -> cp = _gst_sec_flow_heuristic (p,
						   x,
						   bbip,
						   p -> cp);
	}
	*Tp++ = _gst_get_cpu_time ();

	/* Find small-cardinality subtour violations	*/
	/* by partial enumeration...			*/
	for (p = comp; p NE NULL; p = p -> next) {
		p -> cp = _gst_find_small_subtours (p, p -> cp, bbip);
	}
	*Tp++ = _gst_get_cpu_time ();

	/* Discard each component for which we have found at least one	*/
	/* violation.  Gather all constraints onto the main list...	*/
	hookp = &comp;
	for (;;) {
		p = *hookp;
		if (p EQ NULL) break;
		cp2 = p -> cp;
		if (cp2 NE NULL) {
			/* Gather these constraints onto main list... */
			p -> cp = NULL;
			while (cp2 NE NULL) {
				tmp = cp2 -> next;
				cp2 -> next = cp;
				cp = cp2;
				cp2 = tmp;
			}
			*hookp = p -> next;
			p -> next = NULL;
			_gst_free_congested_component (p);

		}
		else {
			/* No constraints yet for this component.  We	*/
			/* may want to try the more expensive method...	*/
			/* Retain this component.			*/
			hookp = &(p -> next);
		}
	}

#if 0
	/* Time to use the new-fangled SEC separator... */
	p2 = comp;
	comp = NULL;
	cp = _gst_sec_flow_separator (&p2, x, bbip, cp);
	*Tp++ = _gst_get_cpu_time ();
#else
	/* Time to use the new-fangled SEC separator... */
	/* Do it one component at a time, so that we can see if */
	/* there are any components for which no violations were found. */
	while (comp NE NULL) {
		p2 = comp;
		comp = comp -> next;
		p2 -> next = NULL;
		cp2 = _gst_sec_flow_separator (&p2, x, bbip, NULL);
		if (cp2 EQ NULL) {
			if ((bbip -> params -> local_cuts_mode EQ
			     GST_PVAL_LOCAL_CUTS_MODE_SUBTOUR_COMPONENTS) OR
			    (bbip -> params -> local_cuts_mode EQ
			     GST_PVAL_LOCAL_CUTS_MODE_BOTH)) {
				try_localcuts = TRUE;
			}

		}
		while (cp2 NE NULL) {
			tmp = cp2 -> next;
			cp2 -> next = cp;
			cp = cp2;
			cp2 = tmp;
		}
	}
	*Tp++ = _gst_get_cpu_time ();

	if (((cp EQ NULL) AND
	     ((bbip -> params -> local_cuts_mode EQ
	      GST_PVAL_LOCAL_CUTS_MODE_SUBTOUR_RELAXATION) OR
	      (bbip -> params -> local_cuts_mode EQ
	      GST_PVAL_LOCAL_CUTS_MODE_BOTH))) OR
	    try_localcuts) {
		/* Try new-fangled separator that crushes	*/
		/* small patches of fractional stuff.		*/

		cp = _gst_find_local_cuts (x, bbip, cp);
		*Tp++ = _gst_get_cpu_time ();
	}
#endif

	/* If this separation routine does not find any SEC violations,	*/
	/* it means that none exist!					*/
	optimal = TRUE;

	/* Note: congested components are all freed now... */

#if 0
	if (cp EQ NULL) {
		/* Nothing else found -- look for fractional cutsets... */
		cp = _gst_find_fractional_cutsets (x, bbip);
		*Tp++ = _gst_get_cpu_time ();
	}
#endif

	if ((cp EQ NULL) AND optimal) {
		/* We KNOW that we have NO violations!  The LP	*/
		/* relaxation is now OPTIMAL!			*/
		bbip -> node -> optimal = TRUE;
	}

	*Tpp = Tp;

	return (cp);
}

/*
 * This routine attempts to use LP reduced costs to fix variables.  Any
 * variable whose reduced cost exceeds the current LP/IP gap can be
 * permanently fixed to its current value (either 0 or 1) for the duration
 * of the current bb-node (and all of its children).
 */

	static
	int
reduced_cost_var_fixing (

struct bbinfo *		bbip	/* IN - branch-and-bound info */
)
{
int			i;
int			nedges;
int			status;
int			nfix0;
int			nfix1;
struct gst_hypergraph *	cip;
double *		zlb;
double			gap;
double			threshold;
int *			newfix0;
int *			newfix1;
struct bbnode *		nodep;

	if (bbip -> best_z >= DBL_MAX) {
		/* Can't reasonably attempt this until we have	*/
		/* a valid upper bound...			*/
		return (VFIX_NOTHING_FIXED);
	}

	nodep = bbip -> node;
	gap = bbip -> best_z - nodep -> z;

	/* Only fix if we significantly exceed the gap... */
	gap *= (1.0 + FUZZ);

	threshold = nodep -> z + gap;

	cip	= bbip -> cip;
	nedges	= cip -> num_edges;

	nodep	= bbip -> node;
	zlb	= nodep -> zlb;

	newfix0 = NEWA (nedges, int);
	newfix1 = NEWA (nedges, int);

	nfix0	= 0;
	nfix1	= 0;

	for (i = 0; i < nedges; i++) {
		if (BITON (bbip -> fixed, i)) continue;

		if (zlb [2 * i] > threshold) {
			newfix1 [nfix1++] = i;
		}
		if (zlb [2 * i + 1] > threshold) {
			newfix0 [nfix0++] = i;
		}
	}

	status = VFIX_NOTHING_FIXED;

	if ((nfix0 > 0) OR (nfix1 > 0)) {
		status = fix_variables (bbip, newfix0, nfix0, newfix1, nfix1);
	}

	free ((char *) newfix1);
	free ((char *) newfix0);

	return (status);
}

/*
 * This routine fixes variables to zero and/or one.  We are given two
 * lists of variables to fixed, those to be fixed to zero, and those
 * to be fixed to one.  This routine then iteratively applies a series
 * of deductive steps that can cause additional variables to be fixed
 * based upon connectivity and compatibility criteria.
 */

	static
	int
fix_variables (

struct bbinfo *		bbip,		/* IN - branch-and-bound info */
int *			fix_to_0,	/* IN - vars to fix to 0 */
int			nfix0,		/* IN - number of vars fixed to 0 */
int *			fix_to_1,	/* IN - vars to fix to 1 */
int			nfix1		/* IN - number of vars fixed to 1 */
)
{
int			i;
int			j;
int			k;
int			t;
int			e;
int			nedges;
int			nmasks;
int			kmasks;
int			status;
int			last_edge;
int			fix0_count;
int			fix1_count;
int			numinc;
struct gst_hypergraph *	cip;
struct bbnode *		nodep;
bitmap_t *		fixmask0;
bitmap_t *		fixmask1;
bitmap_t *		verts_checked;
int *			ep1;
int *			ep2;
int *			ep3;
int *			ep4;
int *			vp1;
int *			vp2;
int *			inclist;
int			vars_fixed;
int			fix_frac;
struct inc_info		inc_info;

#undef	PRINT_FIXED_VARIABLES

	cip	= bbip -> cip;
	nodep	= bbip -> node;

	nedges	= cip -> num_edges;
	nmasks	= cip -> num_edge_masks;
	kmasks	= cip -> num_vert_masks;

	fixmask0	= NEWA (2 * nmasks + kmasks, bitmap_t);
	fixmask1	= fixmask0 + nmasks;
	verts_checked	= fixmask1 + nmasks;

	/* Initialize masks of vars in fix-to-0 and fix-to-1 lists. */
	/* We use these to prevent adding duplicate entries later on. */
	for (i = 0; i < nmasks; i++) {
		fixmask0 [i] = 0;
		fixmask1 [i] = 0;
	}
	for (i = 0; i < nfix0; i++) {
		SETBIT (fixmask0, fix_to_0 [i]);
	}
	for (i = 0; i < nfix1; i++) {
		SETBIT (fixmask1, fix_to_1 [i]);
	}

	inclist = NULL;

	status = VFIX_NOTHING_FIXED;

	fix_frac = 0;

	fix0_count = 0;
	fix1_count = 0;

	/* Iteratively fix variables until no more fixing can be done. */
	do {
		vars_fixed = FALSE;

		/* =============== Handle fixing vars to 0 =============== */

		ep1 = fix_to_0;
		ep2 = ep1;
		ep3 = ep1 + nfix0;
		while (ep1 < ep3) {
			e = *ep1++;
			CLRBIT (fixmask0, e);
			if (NOT BITON (bbip -> edge_mask, e)) continue;
			if (BITON (bbip -> fixed, e)) {
				if (NOT BITON (bbip -> value, e)) {
					/* Already fixed to zero!  Ignore. */
					continue;
				}
				/* Already fixed to one! */
				status = VFIX_INFEASIBLE;
				goto alldone;
			}
			/* Fix it to zero now! */
			SETBIT (bbip -> fixed, e);
			CLRBIT (bbip -> value, e);
			SETBIT (nodep -> fixed, e);
			CLRBIT (nodep -> value, e);
			if ((FUZZ < nodep -> x [e]) AND
			    (nodep -> x [e] + FUZZ < 1.0)) {
				++fix_frac;
			}
			change_var_bounds (bbip -> lp, e, 0.0, 0.0);
			++fix0_count;
#ifdef PRINT_FIXED_VARIABLES
			gst_channel_printf (bbip -> params -> print_solve_trace,
				" 	Fixed x%-3d = 0\n", e);
#endif
			/* Save this edge for later check. */
			*ep2++ = e;
			/* We have fixed at least 1 variable! */
			status = VFIX_VARIABLES_FIXED;
			vars_fixed = TRUE;
		}
		ep1 = fix_to_0;
		if (ep1 < ep2) {
			/* Check if any of the vars we just set to 0	*/
			/* permit us to deduce vars that must be 1...	*/
			for (i = 0; i < kmasks; i++) {
				verts_checked [i] = 0;
			}
			while (ep1 < ep2) {
				i = *ep1++;
				vp1 = cip -> edge [i];
				vp2 = cip -> edge [i + 1];
				while (vp1 < vp2) {
					t = *vp1++;
					if (BITON (verts_checked, t)) continue;
					SETBIT (verts_checked, t);
					ep3 = cip -> term_trees [t];
					ep4 = cip -> term_trees [t + 1];
					k = 0;
					last_edge = -1;
					while (ep3 < ep4) {
						e = *ep3++;
						if (NOT BITON (bbip -> edge_mask, e)) continue;
						if (BITON (bbip -> fixed, e) AND
						    NOT BITON (bbip -> value, e)) continue;
						/* Edge e has vertex t */
						/* and is NOT fixed to zero. */
						last_edge = e;
						++k;
						if (k > 1) break;
					}
					if (k <= 0) {
						/* disconnected vertex! */
						status = VFIX_INFEASIBLE;
						goto alldone;
					}
					if ((k EQ 1) AND
					    NOT BITON (fixmask1, last_edge)) {
						/* one edge left, it */
						/* must be taken! */
						SETBIT (fixmask1, last_edge);
						fix_to_1 [nfix1++] = last_edge;
					}
				}
			}
		}
		/* Set the Fix-to-0 list to empty.  Fixmask0 should now	*/
		/* have all bits turned off.				*/
		nfix0 = 0;


		/* =============== Handle fixing vars to 1 =============== */

		ep1 = fix_to_1;
		ep2 = ep1 + nfix1;
		while (ep1 < ep2) {
			e = *ep1++;
			CLRBIT (fixmask1, e);
			if (NOT BITON (bbip -> edge_mask, e)) continue;
			if (BITON (bbip -> fixed, e)) {
				if (BITON (bbip -> value, e)) {
					/* Already fixed to one!  Ignore. */
					continue;
				}
				/* Already fixed to zero! */
				status = VFIX_INFEASIBLE;
				goto alldone;
			}
			/* Fix it to one now! */
			SETBIT (bbip -> fixed, e);
			SETBIT (bbip -> value, e);
			SETBIT (nodep -> fixed, e);
			SETBIT (nodep -> value, e);
			if ((FUZZ < nodep -> x [e]) AND
			    (nodep -> x [e] + FUZZ < 1.0)) {
				++fix_frac;
			}
			change_var_bounds (bbip -> lp, e, 1.0, 1.0);
			++fix1_count;
#ifdef PRINT_FIXED_VARIABLES
			gst_channel_printf (bbip -> params -> print_solve_trace,
				" 	Fixed x%-3d = 1\n", e);
#endif
			/* We have fixed at least 1 variable! */
			status = VFIX_VARIABLES_FIXED;
			vars_fixed = TRUE;

			/* Fix every *other* incompatible edge to zero! */
			if (inclist EQ NULL) {
				_gst_startup_incompat_edges (&inc_info, cip);
				inclist = NEWA (nedges, int);
			}
			/* Retrieve list of edges incompatible to this one */
			numinc = _gst_get_incompat_edges (inclist, e, &inc_info);
			for (k = 0; k < numinc; k++) {
				j = inclist [k];
				if (j EQ e) continue;
				if (NOT BITON (fixmask0, j)) {
					SETBIT (fixmask0, j);
					fix_to_0 [nfix0++] = j;
				}
			}
		}
		/* Set the Fix-to-1 list to empty.  Fixmask1 should now	*/
		/* have all bits turned off.				*/
		nfix1 = 0;
	} while (vars_fixed);

alldone:

	if ((fix0_count | fix1_count) NE 0) {
		/* Problem has changed -- force re-solve of LP. */
		nodep -> cpiter = -1;
	}

	switch (status) {
	case VFIX_NOTHING_FIXED:
		break;

	case VFIX_VARIABLES_FIXED:
		if (fix_frac > 0) {
			gst_channel_printf (bbip -> params -> print_solve_trace,
				"Fixed %d vars to 0 and %d vars to 1 (%d were fractional).\n",
				fix0_count, fix1_count, fix_frac);
			status = VFIX_FIXED_FRACTIONAL;
		}
		else {
			gst_channel_printf (bbip -> params -> print_solve_trace,
				"Fixed %d vars to 0 and %d vars to 1.\n",
				fix0_count, fix1_count);
		}
		break;

	case VFIX_INFEASIBLE:
		gst_channel_printf (bbip -> params -> print_solve_trace,
			"Variable fixing detected infeasibility!\n");
		break;

	default:
		FATAL_ERROR;
		break;
	}

	if (inclist NE NULL) {
		_gst_shutdown_incompat_edges (&inc_info);
		free ((char *) inclist);
	}
	free ((char *) fixmask0);

	return (status);
}

/*
 * This routine changes the bounds on the given LP variable...
 */

	static
	void
change_var_bounds (

LP_t *			lp,		/* IN - LP to changes bounds of */
int			var,		/* IN - variable to fix */
double			lower,		/* IN - lower bound */
double			upper		/* IN - upper bound */
)
{
#if CPLEX
int			b_index [2];
char			b_lu [2];
double			b_bd [2];

	b_index [0] = var;	b_lu [0] = 'L';		b_bd [0] = lower;
	b_index [1] = var;	b_lu [1] = 'U';		b_bd [1] = upper;
	if (_MYCPX_chgbds (lp, 2, b_index, b_lu, b_bd) NE 0) {
		FATAL_ERROR;
	}
#endif

#if LPSOLVE
	set_bounds (lp, var + 1, lower, upper);
#endif
}

/*
 * Return the number of fractional variables in the given LP solution.
 */

	static
	int
num_fractional_vars (

double *		x,		/* IN - LP solution to check */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i, nedges, num_frac;
struct gst_hypergraph *	cip;

	cip	= bbip -> cip;
	nedges	= cip -> num_edges;

	/* Note that we specifically check edges here that are	*/
	/* disabled in the bbip->edge_mask, because the primal	*/
	/* heuristic can produce such edges.			*/

	num_frac	= 0;
	for (i = 0; i < nedges; i++) {
		if (x [i] <= FUZZ) continue;
		if (x [i] + FUZZ >= 1.0) continue;

		/* Variable has a fractional value... */
		++num_frac;
	}

	return (num_frac);
}

/*
 * This routine checks to see if we have an integer feasible solution.
 * First we check for integrality, then we check connectedness.
 */

	static
	bool
integer_feasible_solution (

double *		x,		/* IN - LP solution to check. */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			j;
int			t;
int			fs;
int			fs2;
int			kmasks;
int			nedges;
int			nmasks;
int			num_int;
int			num_frac;
int			starting_edge;
struct gst_hypergraph *	cip;
bitmap_t *		vert_mask;
bitmap_t *		integral_edges;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			sp;
int *			stack;
bitmap_t		mask;
bitmap_t *		verts_left;

	cip		= bbip -> cip;
	vert_mask	= bbip -> vert_mask;

	nedges = cip -> num_edges;
	nmasks = cip -> num_edge_masks;
	kmasks = cip -> num_vert_masks;

	/* First do a quick check of the integrality of the solution... */
	num_frac	= 0;
	for (i = 0; i < nedges; i++) {
		if (x [i] <= FUZZ) continue;
		if (x [i] + FUZZ >= 1.0) continue;

		/* Variable has a fractional value... */
		++num_frac;
	}

	if (num_frac > 0) {
		/* We have fractional variables -- solution is	*/
		/* definitely NOT integer feasible...		*/
		return (FALSE);
	}

	/* All solution variables are either 0 or 1 -- integral!  This	*/
	/* case is much less common.  Loop back over the solution,	*/
	/* making a note of all edges present in the solution.		*/

	integral_edges = NEWA (nmasks, bitmap_t);
	for (i = 0; i < nmasks; i++) {
		integral_edges [i] = 0;
	}

	num_int		= 0;
	starting_edge	= -1;
	j = 0;
	for (i = 0; i < nedges; i++) {
		if (x [i] >= 0.5) {
			SETBIT (integral_edges, i);
			starting_edge = i;
			++num_int;
			j += (cip -> edge_size [i] - 1);
		}
	}

	if (j NE cip -> num_verts - 1) {
		/* Wrong cardinality of edges -- cannot be a tree. */
		free ((char *) integral_edges);
		return (FALSE);
	}

	if (starting_edge < 0) {
		/* No edges in solution -- problem must have one or	*/
		/* fewer vertices!  This is connected by default.	*/
		free ((char *) integral_edges);
		return (TRUE);
	}

	/* Create temporary mask of vertices we have not yet seen... */
	verts_left = NEWA (kmasks, bitmap_t);
	for (i = 0; i < kmasks; i++) {
		verts_left [i] = vert_mask [i];
	}

	stack = NEWA (num_int, int);
	sp = stack;

	/* Find connected component containing the starting_edge... */
	CLRBIT (integral_edges, starting_edge);
	--num_int;
	*sp++ = starting_edge;

	while (sp > stack) {
		fs = *--sp;
		vp1 = cip -> edge [fs];
		vp2 = cip -> edge [fs + 1];
		while (vp1 < vp2) {
			t = *vp1++;
			if (NOT BITON (verts_left, t)) continue;
			CLRBIT (verts_left, t);
			ep1 = cip -> term_trees [t];
			ep2 = cip -> term_trees [t + 1];
			while (ep1 < ep2) {
				fs2 = *ep1++;
				if (NOT BITON (integral_edges, fs2)) continue;
				CLRBIT (integral_edges, fs2);
				--num_int;
				*sp++ = fs2;
			}
		}
	}

	/* See if any vertices were not reached... */
	mask = 0;
	for (i = 0; i < kmasks; i++) {
		mask |= verts_left [i];
	}

	free ((char *) stack);
	free ((char *) verts_left);
	free ((char *) integral_edges);

	if (mask NE 0) {
		/* At least one more connected component -- solution	*/
		/* is not connected, and therefore infeasible!  (We	*/
		/* also know we have at least one integer cycle!)	*/
		return (FALSE);
	}

	/* Solution is a Steiner tree!  (Not necessarily minimal.) */

	return (TRUE);
}

/*
 * Return the API-defined version of the given node's "lower bound status."
 */

	int
_gst_get_lb_status (

struct bbnode *	node		/* IN: branch-and-bound node */
)
{
int		retval;

	FATAL_ERROR_IF (node EQ NULL);

	retval = -1;
	switch (node -> lb_status) {
	case LB_INFEASIBLE:
		retval = GST_LB_STATUS_INFEASIBLE;
		break;

	case LB_CUTOFF:
		retval = GST_LB_STATUS_CUTOFF;
		break;

	case LB_INTEGRAL:
		retval = GST_LB_STATUS_INTEGRAL;
		break;

	case LB_FRACTIONAL:
		retval = GST_LB_STATUS_FRACTIONAL;
		break;

	case LB_PREEMPTED:
		retval = GST_LB_STATUS_PREEMPTED;
		break;

	default:
		FATAL_ERROR;
		break;
	}

	return (retval);
}

/*
 * This routine goes through each of the constraints in the LP tableaux
 * for the root node.  Each constraint is checked against each LP
 * solution (recorded in the rcfile) to find the earliest iteration in
 * which we could have had root LP optimality -- if our separation
 * algorithm magically generated the *right* constraints.
 */

	static
	void
check_root_constraints (

struct bbinfo *		bbip	/* IN - branch-and-bound info */
)
{
int			i;
int			j;
int			nedges;
int			nbytes;
int			iter;
int *			list;
struct gst_hypergraph *	cip;
struct cpool *		pool;
int *			ip1;
int *			ip2;
int *			ip3;
double *		x;
FILE *			fh;

	cip	= bbip -> cip;
	pool	= bbip -> cpool;

	/* Develop the list of all (non-initial) constraints... */
	list = NEWA (pool -> nlprows, int);
	ip3 = list;
	for (i = 0; i < pool -> nlprows; i++) {
		j = pool -> lprows [i];
		if (j < pool -> initrows) continue;
		*ip3++ = j;
	}


	fclose (bbip -> rcfile);
	bbip -> rcfile = NULL;

	fh = fopen ("/tmp/lp.x", "r");

	nedges = cip -> num_edges;
	nbytes = nedges * sizeof (double);

	x = NEWA (nedges, double);

	iter = 0;
	while (ip3 > list) {
		i = fread (x, 1, nbytes, fh);
		FATAL_ERROR_IF (i NE nbytes);

		/* Delete all remaining constraints that violate x. */
		ip1 = ip2 = list;
		while (ip2 < ip3) {
			j = *ip2++;
			if (NOT _gst_is_violation (pool -> rows [j].coefs, x)) {
				/* No violation -- keep constraint around. */
				*ip1++ = j;
			}
		}
		ip3 = ip1;
		gst_channel_printf (bbip -> params -> print_solve_trace,
			"@r iter %d, %ld constraints left\n",
			iter, ip3 - list);
		if (ip3 <= list) break;
		++iter;
	}

	gst_channel_printf (bbip -> params -> print_solve_trace,
		"@RC Could have gotten root constraints in %d iterations!\n", iter);

	fclose (fh);

	free ((char *) x);
	free ((char *) list);
}
