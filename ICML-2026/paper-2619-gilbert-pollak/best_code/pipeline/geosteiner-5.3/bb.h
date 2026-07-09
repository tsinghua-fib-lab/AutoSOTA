/***********************************************************************

	$Id: bb.h,v 1.28 2022/11/19 13:45:49 warme Exp $

	File:	bb.h
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Data structure for branch-and-bound (branch-and-cut)
	procedure.

************************************************************************

	Modification Log:

	a-1:	07/17/96	warme
		: Created.
	a-2:	02/28/2001	warme
		: Changes for 3.1 release, CPLEX 7.0, etc.
	b-1:	08/05/2002	benny
		: Numerous changes for library release.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
	e-4:	11/18/2022	warme
		: Added owner, delrow_z and lb_status to bbnode.
		: Added _gst_get_lb_status().

************************************************************************/

#ifndef BB_H
#define	BB_H

#define _GNU_SOURCE

#include "bitmaskmacros.h"
#include "lpsolver.h"
#include "polltime.h"

struct gst_hypergraph;
struct gst_param;
struct gst_solver;

/*
 * Constants
 */

			/* floating point "fuzz" for comparisons. */
#define	FUZZ		0.000001

	/* Constants for the heaps... */

#define	INIT_HSIZE	128
#define	NUM_BB_HEAPS	2

#define	BEST_NODE_HEAP	0
#define	WORST_NODE_HEAP	1


/*
 * LP result status codes that are independent of the particular
 * solver used.
 */

#define	BBLP_OPTIMAL		0
#define	BBLP_CUTOFF		1
#define	BBLP_INFEASIBLE		2

/*
 * The following structure is used to represent a single node of the
 * branch-and-bound tree.
 */

struct bbnode {
	struct bbinfo *	owner;	/* The bbinfo that owns this node */
	double		z;	/* node's current objective value -- this */
				/* is initially either the parent's or an */
				/* estimate from try_branch. */
	double		delrow_z;
				/* Objective value at which slack rows	*/
				/* were last deleted from this node's LP */
	bool		optimal; /* optimal LP relaxation found? */
	int		lb_status; /* Lower bound status of this node */
	int		num;	/* node number -- assigned in order of */
				/* NODE CREATION! */
	int		iter;	/* number of LP iterations for this node */
	int		parent;	/* node number of parent node */
	int		index [NUM_BB_HEAPS]; /* index in corresponding */
				/* bbtree heap */
	int		var;	/* var that was branched to make this node */
	int		dir;	/* direction branch var was restricted */
	int		depth;	/* depth in tree -- 0 EQ root node */
	int		br1cnt;	/* count of var=1 branch decisions */
	double *	x;	/* most recent LP solution */
	int		cpiter;	/* constraint pool time stamp, used to see */
				/* if x value is up-to-date w.r.t pool */
	double *	zlb;	/* lower bounds on Xi=0 and Xi=1 branches */
	bitmap_t *	fixed;	/* variables fixed to some value */
	bitmap_t *	value;	/* value variables are fixed at */
	int		n_uids;	/* Number of UIDs in bc_uids list */
	int *		bc_uids; /* Unique IDs of all constraints that are */
				/* binding for this node.  Only non-null */
				/* when node is suspended. */
	int *		bc_row;	/* position of bc_uid row in LP tableaux */
	int *		rstat;	/* basis info for corresponding bc_uids row */
	int *		cstat;	/* basis info for each column */
	double *	bheur;	/* Branch heuristic values */
	struct bbnode *	next;	/* next unprocessed node in LIFO order */
	struct bbnode *	prev;	/* previous unprocessed node in LIFO order */
};

/*
 * The following typedef defines the type of function that is used by
 * a heap to determine if the first node belongs higher up (closer to
 * the root) than the second node.  Different heaps use different
 * comparison functions to achieve different sorting orders.
 */

typedef int	bbheap_func_t (struct bbnode *, struct bbnode *);


/*
 * The following structure is used to represent a single heap of
 * branch-and-bound nodes.
 */

struct bbheap {
	struct bbnode ** array;	/* the heap array */
	int		hsize;	/* current heap array allocation */
	int		nheap;	/* number of nodes in the heap array */
				/* comparision function to determine if */
				/* first node belongs above second in */
				/* this heap... */
	bbheap_func_t *	is_parent_funcp;
};

/*
 * The following structure is used to represent the branch-and-bound
 * tree.  It contains the following ways of accessing the nodes:
 *
 *	- a linked-list for processing the nodes in depth-first order
 *	- a heap to access the current best node (lowest objective)
 *	- a heap to access the WORST node -- used to find those nodes
 *	  that must be cut-off whenever a better feasible integer
 *	  solution is found.
 */

struct bbtree {
	struct bbnode *	first;	/* first node in LIFO (depth first) order */
	struct bbnode *	free;	/* node freelist */
	int		snum;	/* node creation serial number counter */
	int		nmasks;	/* Size of "fixed" and "value" bit masks */
	int		node_policy;	/* Next node policy */
	struct bbheap	heap [NUM_BB_HEAPS]; /* heaps used to access nodes */
				/* in various orders */
};

/*
 * Next node policies...
 */

#define	NN_DEPTH_FIRST		0
#define	NN_BEST_NODE		1

/*
 * The following structure contains all of the global information that
 * we pass around between the various components of the branch-and-cut
 * procedure:
 *
 *	- the phase-1 data (original points, full sets, and
 *	    compatibility info
 *	- the main LP problem instance
 *	- the branch-and-bound tree
 *	- info used by the various separation procedures
 *	- the best feasible integer solution seen so far
 *	- the current branch-and-bound node, including:
 *		- its LP objective value and solution
 *		- its local fixed variables and values
 */

struct bbinfo {
	struct gst_hypergraph *	/* original point set, full sets, and */
			cip;	/* compatibility info */
	struct gst_solver *
			solver; /* solution information and more... */
	bitmap_t *	vert_mask; /* Set of valid vertices in problem */
	bitmap_t *	edge_mask; /* Set of valid edges in problem */
	LP_t *		lp;	/* the main LP problem instance */
	struct lpmem *	lpmem;	/* memory allocated for LP problem instance */
	struct cpool *	cpool;	/* the global pool of constraints */
	struct bbtree *	bbtree;	/* the branch-and-bound tree */
	struct cs_info * csip;	/* cutset separation info */
	double		preempt_z; /* objective value above which to preempt */
				   /* the current node */
	double		best_z;	/* best feasible integer objective so far */
	bitmap_t *	_smt;	/* current best feas int soln (obsolete) */
	struct bbnode *	node;	/* current branch-and-bound node */
	double		_z;	/*   its objective value (obsolete) */
	double *	_x;	/*   its LP solution vector (obsolete) */
	int		slack_size; /* size of slack vector */
	double *	slack;	/*   its LP slack variables */
	double *	dj;	/*   its LP reduced costs */
	bitmap_t *	fixed;	/*   set of fixed variables */
	bitmap_t *	value;	/*   values of fixed variables */
	struct bbstats * statp;	/* statistics structure */
	cpu_time_t	t0;	/* CPU time at start of branch and cut */
	double		prevlb;	/* previous best lower bound */
	struct ubinfo *	ubip;	/* info used by upper bound heuristic */

	struct gst_param *
			params; /* the parameter set for the problem */
	FILE *		rcfile; /* pointer to root constraints file */

	struct comp *	failed_fcomps; /* components that have been tried as a
					  local cut before, with no success */
	cpu_time_t	next_ckpt_time; /* next checkpoint time */
	volatile bool	force_branch_flag;
	struct cpu_poll	mainpoll;
	struct cpu_poll	cglbpoll;
};

/*
 * This structure is used to hold statistics for the branch-and-cut.
 */

struct cstats {			/* Constraint statistics... */
	int	num_prows;	/* Number of rows in constraint pool */
	int	num_lprows;	/* Number of rows in LP */
	int	num_pnz;	/* Number of non-zeros in constraint pool */
	int	num_lpnz;	/* Number of non-zeros in LP */
};

struct bbstats {
	int		num_nodes;	/* Number of b&b nodes */
	int		num_lps;	/* Number of LP's solved */
	struct cstats	cs_init;	/* Constraint stats initially */
	struct cstats	cs_root;	/* Constraint stats for root node */
	struct cstats	cs_final;	/* Final constraint stats */
	/* Root node statistics */
	double		root_z;		/* Objective value for root node */
	bool		root_opt;	/* Is root_z optimal? */
	int		root_lps;	/* Number of LP's solved at root */
	cpu_time_t	root_time;	/* CPU time to finish root node */
};

/*
 * Extern declarations
 */

extern void		_gst_branch_and_cut (struct gst_solver * solver);
extern bool		_gst_check_for_better_IFS (double *		x,
						   struct bbinfo *	bbip,
						   double *		true_z);
extern struct bbinfo *	_gst_create_bbinfo (struct gst_solver *	solver);
extern int		_gst_get_lb_status (struct bbnode * node);
extern void		_gst_new_upper_bound (double ub, struct bbinfo * bbip);

#endif
