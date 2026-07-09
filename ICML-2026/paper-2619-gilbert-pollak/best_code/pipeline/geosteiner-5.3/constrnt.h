/***********************************************************************

	$Id: constrnt.h,v 1.14 2022/11/19 13:45:50 warme Exp $

	File:	constrnt.h
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Data structures for describing constraints.

************************************************************************

	Modification Log:

	a-1:	07/17/96	warme
		: Created.
	b-1:	02/28/2001	warme
		: Changes for 3.1 release.
		: Made add_constraint_to_pool and expand_constraint
		:  be global.
	c-1:	08/05/2002	benny
		: Some changes for library release.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
	e-4:	11/18/2022	warme
		: Simplify calling convention.

************************************************************************/

#ifndef CONSTRNT_H
#define	CONSTRNT_H

#include "bitmaskmacros.h"
#include "lpsolver.h"


/*
 * The following structures represent "raw" (or physical) constraints.
 * These constraints are expressed using the actual row of coefficients,
 * operator and right-hand-side.
 *
 * Negative values of var represent the constraint's OPERATOR and rhs,
 * as well as marking the end.
 */

#define RC_OP_LE	0
#define RC_OP_EQ	1
#define RC_OP_GE	2
#define RC_VAR_BASE	3

struct rcoef {
	int		var;	/* variable (var>=0) or operator (var<0) */
	int		val;	/* coefficient value */
};

struct rcon {
	int		len;	/* length of constraint LHS */
	struct rcoef *	coefs;	/* the actual coefficients of the row */
	int		next;	/* next rcon number in hash bucket chain */
	int		lprow;	/* not in LP if <0, current row if >=0 */
	int		biter;	/* most recent iteration during which this */
				/* constraint was binding */
	short		hval;	/* hash value for this entry */
	short		flags;	/* various flags for entry */
	int		uid;	/* unique ID */
	int		refc;	/* reference count: number of *suspended* */
				/* nodes for which this constraint is */
				/* binding */
};

/* flags */

#define	RCON_FLAG_DISCARD	0x0001	/* Discard at next opportunity. */


/*
 * Structures used to store "logical" constraints, which are expressed
 * in terms of their significance to the problem, not their coefficients.
 */

enum ctype {
	CT_CUTSET,
	CT_SUBTOUR,
	CT_RAW
};

struct constraint {
	struct constraint *	next;
	int			iteration;
	enum ctype		type;
	bitmap_t *		mask;
};


/*
 * The constraint pool.  We maintain all constraints that have ever
 * been generated in the pool, along with the hash table, freelists
 * and scratch buffers.
 */

#define	CPOOL_HASH_SIZE	1009

struct cpool {
	int		uid;		/* Bumped when pool changes */
	struct rcon *	rows;		/* All constraints, in sequence */
	int		nrows;		/* Number of constraints in the pool */
	int		maxrows;	/* Allocated size of pool */
	int		num_nz;		/* Number of non-zeros in the pool */
	int *		lprows;		/* maps LP row # to constraint # */
	int		nlprows;	/* number of LP rows */
	int		npend;		/* # rows pending addition to LP */
	struct rblk *	blocks;		/* List of rcon allocation blocks */
	struct rcoef *	cbuf;		/* Scratch constraint buffer */
	int		iter;		/* LP iteration count - a timestamp */
					/* used to find constraints that */
					/* haven't been useful in a while */
	int		initrows;	/* number of initial rows */
	int		nvars;		/* Number of variables - LP columns */
	int		hwmrow;		/* High water mark for LP rows */
	int		hwmnz;		/* High water mark for LP non-zeros */
	int		hash [CPOOL_HASH_SIZE];
};


/*
 * This structure maintains a single block of free rcoef's, from which
 * we allocate sequentially.  To maintain better cache behavior when
 * scanning all constraints for violations, we always begin a new block
 * whenever the constraint we are adding does not fit in the current
 * block.
 */

struct rblk {
	struct rblk *		next;	/* next rblk in pool */
	struct rcoef *		base;	/* base of array of rcons */
	struct rcoef *		ptr;	/* allocation pointer */
	int			nfree;	/* number of free rcons remaining */
};


/*
 * Function Prototypes
 */

struct bbinfo;
struct bbnode;
struct gst_channel;
struct gst_hypergraph;
struct gst_param;

extern bool	_gst_add_constraint_to_pool (struct cpool *	pool,
					     struct rcoef *	rp,
					     bool		add_to_lp);
extern int	_gst_add_constraints (struct bbinfo *		bbip,
				      struct constraint *	lcp);
extern void	_gst_add_pending_rows_to_LP (struct bbinfo * bbip);
extern LP_t *	_gst_build_initial_formulation (
					struct cpool *		pool,
					bitmap_t *		vert_mask,
					bitmap_t *		edge_mask,
					struct gst_hypergraph *	cip,
					struct lpmem *		lpmem,
					struct gst_param *	params);
extern void	_gst_debug_print_constraint (
					char *		  	msg1,
					char *		  	msg2,
					struct constraint *	lcp,
					double *		x,
					struct bbinfo *		bbip,
					struct gst_channel *	chan);
extern void	_gst_delete_slack_rows_from_LP (struct bbinfo * bbip);
extern void	_gst_destroy_initial_formulation (struct bbinfo * bbip);
extern void	_gst_destroy_node_basis (struct bbnode *	nodep,
					 struct bbinfo *	bbip);
extern void	_gst_free_constraint_pool (struct cpool * pool);
extern void	_gst_initialize_constraint_pool (
					struct cpool *		pool,
					bitmap_t *		vert_mask,
					bitmap_t *		edge_mask,
					struct gst_hypergraph *	cip,
					struct gst_param *	params);
extern bool	_gst_is_violation (struct rcoef * cp, double * x);
extern void	_gst_mark_row_pending_to_LP (struct cpool * pool, int row);
extern void	_gst_restore_node_basis (struct bbnode *	nodep,
					 struct bbinfo *	bbip);
extern void	_gst_save_node_basis (struct bbnode *		nodep,
				      struct bbinfo *		bbip);
extern int	_gst_solve_LP_over_constraint_pool (struct bbinfo * bbip);


#endif
