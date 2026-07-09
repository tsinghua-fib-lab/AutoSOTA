/***********************************************************************

	$Id: constrnt.c,v 1.25 2023/04/03 17:47:41 warme Exp $

	File:	constrnt.c
	Rev:	e-6
	Date:	04/03/2023

	Copyright (c) 1996, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines for handling constraints.

************************************************************************

	Modification Log:

	a-1:	11/18/96	warme
		: Created.  Split off from main file.
	a-2:	02/28/2001	warme
		: Numerous changes for 3.1 release.
	c-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Uses parameters.
		: Uses channels for trace output.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	12/12/2015	warme
		: Use memmove() instead of memcpy() when
		:  garbage collecting coefficients.
	e-3:	09/05/2016	warme
		: Fix printf format mismatches.
		: Change notices for 5.1 release.
	e-4:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.  Upgrade fatals.
	e-5:	11/18/2022	warme
		: Only delete slack rows if objective has improved.
		: Delete slack rows during solve over pool.
		: Simplify calling convention of various functions.
		: Remove unused code.
		: Fix -Wall issues.
	e-6:	04/03/2023	warme
		: Assure that solve-over-constraint-pool has a valid
		:  basis upon return.
		: Fix mis-spelled function name.
		: Added comment.

************************************************************************/

#include "constrnt.h"

#include "bb.h"
#include "channels.h"
#include "config.h"
#include "expand.h"
#include "fatal.h"
#include <float.h>
#include <limits.h>
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "parmblk.h"
#include "steiner.h"
#include <string.h>


/*
 * Global Routines
 */

bool		_gst_add_constraint_to_pool (struct cpool *	pool,
					     struct rcoef *	rp,
					     bool		add_to_lp);
int		_gst_add_constraints (struct bbinfo *		bbip,
				      struct constraint *	lcp);
void		_gst_add_pending_rows_to_LP (struct bbinfo * bbip);
LP_t *		_gst_build_initial_formulation (
					struct cpool *		pool,
					bitmap_t *		vert_mask,
					bitmap_t *		edge_mask,
					struct gst_hypergraph *	cip,
					struct lpmem *		lpmem,
					gst_param_ptr		params);
void		_gst_debug_print_constraint (char *		msg1,
					char *			msg2,
					struct constraint *	lcp,
					double *		x,
					struct bbinfo *		bbip,
					gst_channel_ptr		chan);
void		_gst_delete_slack_rows_from_LP (struct bbinfo *	bbip);
void		_gst_destroy_initial_formulation (struct bbinfo * bbip);
void		_gst_destroy_node_basis (struct bbnode *	nodep,
					 struct bbinfo *	bbip);
void		_gst_initialize_constraint_pool (
					struct cpool *		pool,
					bitmap_t *		vert_mask,
					bitmap_t *		edge_mask,
					struct gst_hypergraph *	cip,
					gst_param_ptr		params);
bool		_gst_is_violation (struct rcoef * cp, double * x);
void		_gst_mark_row_pending_to_LP (struct cpool * pool, int row);
void		_gst_restore_node_basis (struct bbnode *	nodep,
					 struct bbinfo *	bbip);
void		_gst_save_node_basis (struct bbnode *		nodep,
				      struct bbinfo *		bbip);
int		_gst_solve_LP_over_constraint_pool (struct bbinfo * bbip);

/*
 * Local Routines
 */

static double		compute_slack_value (struct rcoef *, double *);
static void		garbage_collect_pool (struct cpool *, int, int, gst_param_ptr);
static void		print_pool_memory_usage (struct cpool *,
						 gst_channel_ptr);
static void		prune_pending_rows (struct bbinfo *, bool);
static void		reduce_constraint (struct rcoef *);
static struct rblk *	reverse_rblks (struct rblk *);
static int		solve_single_LP (struct bbinfo *,
					 double *,
					 double *,
					 int);
static void		sort_gc_candidates (int *, int32u *, int);
static bool		sprint_term (char *, bool, int, int);
static void		update_lp_solution_history (double *,
						    double *,
						    struct bbinfo *);
static void		verify_pool (struct cpool *);

#if CPLEX
static void		reload_cplex_problem (struct bbinfo *);
#endif

#if LPSOLVE
static void		get_current_basis (LP_t *, int *, int *);
static void		set_current_basis (LP_t *, int *, int *);
#endif

/*
 * This routine initializes the given constraint pool and fills it with
 * the initial set of constraints:
 *
 *	- The total degree constraint.
 *	- One cutset constraint per terminal.
 *	- All two-vertex SECs.
 *	- All incompatibility constraints that aren't shadowed
 *	  by a two-vertex SEC.
 */

	void
_gst_initialize_constraint_pool (

struct cpool *		pool,		/* OUT - the pool to initialize */
bitmap_t *		vert_mask,	/* IN - set of valid vertices */
bitmap_t *		edge_mask,	/* IN - set of valid hyperedges */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
gst_param_ptr		params
)
{
int			i, j, k;
int			nterms;
int			nedges;
int			nmasks;
int			kmasks;
int			nvt;
int			nrows;
int			ncoeff;
int			num_total_degree_rows;
int			num_total_degree_coeffs;
int			num_cutset_rows;
int			num_cutset_coeffs;
int			num_incompat_rows;
int			num_incompat_coeffs;
int			num_2sec_rows;
int			num_2sec_coeffs;
int			rowsize;
int			nzsize;
int			fs;
int *			vp1;
int *			vp2;
int *			vp3;
int *			vp4;
int *			ep1;
int *			ep2;
int *			counts;
int *			tlist;
bitmap_t *		tmask;
bitmap_t *		fsmask;
struct rcoef *		rp;
struct rblk *		blkp;
cpu_time_t		T0;
cpu_time_t		T1;
char			tbuf [32];
gst_channel_ptr		param_print_solve_trace;

	param_print_solve_trace = params -> print_solve_trace;

	T0 = _gst_get_cpu_time ();

	nterms = cip -> num_verts;
	nedges = cip -> num_edges;
	kmasks = cip -> num_vert_masks;
	nmasks = cip -> num_edge_masks;

#if 0
	if (nedges + RC_VAR_BASE > USHRT_MAX) {
		gst_channel_printf (param_print_solve_trace, "Too many FSTs or hyperedges!  Max is %d.\n",
			USHRT_MAX - RC_VAR_BASE);
		exit (1);
	}
#endif

	num_2sec_rows	= 0;
	num_2sec_coeffs	= 0;

	if (params -> seed_pool_with_2secs) {
		/* Count the number of non-trivial 2SEC constraints,	*/
		/* and their non-zero coefficients.			*/
		tlist	= NEWA (nterms, int);
		counts	= NEWA (nterms, int);
		tmask	= NEWA (kmasks, bitmap_t);
		for (i = 0; i < nterms; i++) {
			counts [i] = 0;
		}

		for (i = 0; i < kmasks; i++) {
			tmask [i] = 0;
		}

		for (i = 0; i < nterms; i++) {
			if (NOT BITON (vert_mask, i)) continue;
			vp1 = tlist;
			ep1 = cip -> term_trees [i];
			ep2 = cip -> term_trees [i + 1];
			while (ep1 < ep2) {
				fs = *ep1++;
				if (NOT BITON (edge_mask, fs)) continue;
				vp3 = cip -> edge [fs];
				vp4 = cip -> edge [fs + 1];
				while (vp3 < vp4) {
					j = *vp3++;
					if (j <= i) continue;
					if (NOT BITON (vert_mask, j)) continue;
					++(counts [j]);
					if (BITON (tmask, j)) continue;
					SETBIT (tmask, j);
					*vp1++ = j;
				}
			}
			vp2 = vp1;
			vp1 = tlist;
			while (vp1 < vp2) {
				j = *vp1++;
				if (counts [j] >= 2) {
					/* S={i,j} is a non-trivial SEC... */
					++num_2sec_rows;
					num_2sec_coeffs += counts [j];
				}
				counts [j] = 0;
				CLRBIT (tmask, j);
			}
		}
		free ((char *) tmask);
		free ((char *) counts);
		free ((char *) tlist);
	}

	/* Compute the number of coefficients in the total	*/
	/* degree constraint.					*/
	num_total_degree_rows	= 1;
	num_total_degree_coeffs	= 0;
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (edge_mask, i)) continue;
		++num_total_degree_coeffs;
	}

	/* Compute the number of incompatibility constraints. */
	num_incompat_rows	= 0;
	if (cip -> inc_edges NE NULL) {
		/* Note:  inc_edges does NOT list edges that are	*/
		/* "basic" incompatibilities (edges that share 2 or	*/
		/* more vertices).  It lists ONLY incompatible edges	*/
		/* sharing 1 or fewer vertices.				*/
		for (i = 0; i < nedges; i++) {
			if (NOT BITON (edge_mask, i)) continue;

			ep1 = cip -> inc_edges [i];
			ep2 = cip -> inc_edges [i + 1];
			while (ep1 < ep2) {
				j = *ep1++;
				if (j >= i) break;
				if (NOT BITON (edge_mask, j)) continue;

				++num_incompat_rows;
			}
		}
	}
	num_incompat_coeffs = 2 * num_incompat_rows;

	/* Compute the total number of valid terminals, and the	*/
	/* number of coefficients in the 1-terminal cutsets.	*/
	nvt = 0;
	num_cutset_rows		= 0;
	num_cutset_coeffs	= 0;
	for (i = 0; i < cip -> num_verts; i++) {
		if (NOT BITON (vert_mask, i)) continue;
		/* This is a valid terminal.  There will be one	*/
		/* cutset row for it.				*/
		++nvt;
		++num_cutset_rows;
		ep1 = cip -> term_trees [i];
		ep2 = cip -> term_trees [i + 1];
		while (ep1 < ep2) {
			k = *ep1++;
			if (BITON (edge_mask, k)) {
				++num_cutset_coeffs;
			}
		}
	}

	/* Tally the total number of rows and coefficients... */
	nrows	=   num_total_degree_rows	+ num_cutset_rows
		  + num_incompat_rows		+ num_2sec_rows;

	ncoeff	=   num_total_degree_coeffs	+ num_cutset_coeffs
		  + num_incompat_coeffs		+ num_2sec_coeffs;

	rowsize = 2 * nrows;		/* extra space for more rows... */
	nzsize = 4 * ncoeff;		/* extra space for more coefficients */

	blkp		= NEW (struct rblk);
	blkp -> next	= NULL;
	blkp -> base	= NEWA (nzsize, struct rcoef);
	blkp -> ptr	= blkp -> base;
	blkp -> nfree	= nzsize;

	pool -> uid	= 0;
	pool -> rows	= NEWA (rowsize, struct rcon);
	pool -> nrows	= 0;
	pool -> maxrows	= rowsize;
	pool -> num_nz	= 0;
	pool -> lprows	= NEWA (rowsize, int);
	pool -> nlprows	= 0;
	pool -> npend	= 0;
	pool -> blocks	= blkp;
	pool -> cbuf	= NEWA (nedges + 1, struct rcoef);
	pool -> iter	= 0;
	pool -> initrows = 0;
	pool -> nvars	= nedges;
	pool -> hwmrow	= 0;
	pool -> hwmnz	= 0;

	/* Empty all of the hash table buckets... */
	for (i = 0; i < CPOOL_HASH_SIZE; i++) {
		pool -> hash [i] = -1;
	}

	/* Now generate the row for the spanning constraint... */
	rp = pool -> cbuf;
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (edge_mask, i)) continue;
		rp -> var = i + RC_VAR_BASE;
		rp -> val = (cip -> edge_size [i] - 1);
		++rp;
	}
	rp -> var = RC_OP_EQ;
	rp -> val = nvt - 1;
	_gst_add_constraint_to_pool (pool, pool -> cbuf, TRUE);

	/* Now generate one cutset constraint per terminal... */
	for (i = 0; i < cip -> num_verts; i++) {
		if (NOT BITON (vert_mask, i)) continue;
		rp = pool -> cbuf;
		ep1 = cip -> term_trees [i];
		ep2 = cip -> term_trees [i + 1];
		while (ep1 < ep2) {
			k = *ep1++;
			if (NOT BITON (edge_mask, k)) continue;
			rp -> var = k + RC_VAR_BASE;
			rp -> val = 1;
			++rp;
		}
		rp -> var = RC_OP_GE;
		rp -> val = 1;
		_gst_add_constraint_to_pool (pool, pool -> cbuf, TRUE);
	}

	/* Now generate one constraint per incompatible pair... */
	if (cip -> inc_edges NE NULL) {
		/* Note:  inc_edges does NOT list edges that are	*/
		/* "basic" incompatibilities (edges that share 2 or	*/
		/* more vertices).  It lists ONLY incompatible edges	*/
		/* sharing 1 or fewer vertices.				*/
		for (i = 0; i < nedges; i++) {
			if (NOT BITON (edge_mask, i)) continue;

			ep1 = cip -> inc_edges [i];
			ep2 = cip -> inc_edges [i + 1];
			while (ep1 < ep2) {
				j = *ep1++;
				if (j >= i) break;
				if (NOT BITON (edge_mask, j)) continue;

				rp = pool -> cbuf;
				rp [0].var = j + RC_VAR_BASE;
				rp [0].val = 1;
				rp [1].var = i + RC_VAR_BASE;
				rp [1].val = 1;
				rp [2].var = RC_OP_LE;
				rp [2].val = 1;
				_gst_add_constraint_to_pool (pool,
							     pool -> cbuf,
							     FALSE);
			}
		}
	}

	if (params -> seed_pool_with_2secs) {
		/* Now generate one constraint for each 2-SEC... */

		tlist	= NEWA (nterms, int);
		counts	= NEWA (nterms, int);
		fsmask	= NEWA (nmasks, bitmap_t);
		tmask	= NEWA (kmasks, bitmap_t);
		memset (counts, 0, nterms * sizeof (*counts));
		memset (fsmask, 0, nmasks * sizeof (*fsmask));
		memset (tmask, 0, kmasks * sizeof (*tmask));

		for (i = 0; i < nterms; i++) {
			if (NOT BITON (vert_mask, i)) continue;
			vp1 = tlist;
			ep1 = cip -> term_trees [i];
			ep2 = cip -> term_trees [i + 1];
			while (ep1 < ep2) {
				fs = *ep1++;
				if (NOT BITON (edge_mask, fs)) continue;
				SETBIT (fsmask, fs);
				vp3 = cip -> edge [fs];
				vp4 = cip -> edge [fs + 1];
				while (vp3 < vp4) {
					j = *vp3++;
					if (j <= i) continue;
					if (NOT BITON (vert_mask, j)) continue;
					++(counts [j]);
					if (BITON (tmask, j)) continue;
					SETBIT (tmask, j);
					*vp1++ = j;
				}
			}
			vp2 = vp1;
			vp1 = tlist;
			while (vp1 < vp2) {
				j = *vp1++;
				if (counts [j] < 2) continue;
				/* Generate 2SEC {i,j} */
				rp = pool -> cbuf;
				ep1 = cip -> term_trees [j];
				ep2 = cip -> term_trees [j + 1];
				while (ep1 < ep2) {
					fs = *ep1++;
					if (NOT BITON (fsmask, fs)) continue;

					rp -> var = fs + RC_VAR_BASE;
					rp -> val = 1;
					++rp;
				}
				FATAL_ERROR_IF (rp < &(pool -> cbuf [2]));
				rp -> var = RC_OP_LE;
				rp -> val = 1;
				_gst_add_constraint_to_pool (pool, pool -> cbuf, FALSE);
			}
			vp1 = tlist;
			while (vp1 < vp2) {
				j = *vp1++;
				counts [j] = 0;
				CLRBIT (tmask, j);
			}
			ep1 = cip -> term_trees [i];
			ep2 = cip -> term_trees [i + 1];
			while (ep1 < ep2) {
				fs = *ep1++;
				CLRBIT (fsmask, fs);
			}
		}
		free ((char *) tmask);
		free ((char *) fsmask);
		free ((char *) counts);
		free ((char *) tlist);
	}

	/* Note those rows that are initially present.  Note that we do	*/
	/* not necessarily separate these, so we had better not delete	*/
	/* them from the pool...					*/

	pool -> initrows = pool -> nrows;

	T1 = _gst_get_cpu_time ();
	_gst_convert_cpu_time (T1 - T0, tbuf);
	gst_channel_printf (param_print_solve_trace, "_gst_initialize_constraint_pool: %s seconds.\n", tbuf);

	/* Note: Due to the fact that duplicate rows can be produced	*/
	/* (for example by two nearby terminals having the same cutset	*/
	/* constraints), the final number of rows in the pool may be	*/
	/* smaller than the number of logical constraints seeded...	*/

	gst_channel_printf (param_print_solve_trace, "Constraint pool initialized with:\n");
	gst_channel_printf (param_print_solve_trace, "	%d	Total degree rows	%d	coeffs.\n",
		num_total_degree_rows, num_total_degree_coeffs);
	gst_channel_printf (param_print_solve_trace, "	%d	Cutset rows		%d	coeffs.\n",
		num_cutset_rows, num_cutset_coeffs);
	gst_channel_printf (param_print_solve_trace, "	%d	Incompatibility rows	%d	coeffs.\n",
		num_incompat_rows, num_incompat_coeffs);
	gst_channel_printf (param_print_solve_trace, "	%d	2-terminal SEC rows	%d	coeffs.\n",
		num_2sec_rows, num_2sec_coeffs);
	gst_channel_printf (param_print_solve_trace, "	%d	Total rows in pool	%d	in LP\n",
		pool -> nrows, pool -> npend);

	print_pool_memory_usage (pool, param_print_solve_trace);
}

/*
 * This routine frees up the constraint pool.
 */

	void
_gst_free_constraint_pool (

struct cpool *		pool		/* IN - pool to add constraint to */
)
{
struct rblk *		blkp;
struct rblk *		tmp;

	free ((char *) (pool -> cbuf));
	free ((char *) (pool -> lprows));
	free ((char *) (pool -> rows));

	blkp = pool -> blocks;
	while (blkp NE NULL) {
		tmp = blkp -> next;
		free ((char *) (blkp -> base));
		free ((char *) blkp);
		blkp = tmp;
	}

	free ((char *) pool);
}

/*
 * This routine adds a single constraint to the pool, unless it is
 * already present.  We use the hash table to determine this.
 */

	bool
_gst_add_constraint_to_pool (

struct cpool *		pool,		/* IN - pool to add constraint to */
struct rcoef *		rp,		/* IN - raw constraint to add */
bool			add_to_lp	/* IN - add it to LP tableaux also? */
)
{
int		hval;
int		len;
int		var;
int		row;
int		n;
struct rcoef *	p;
struct rcon *	rcp;
int *		hookp;
struct rblk *	blkp;
struct rblk *	blkp2;
int *		ip;
size_t		nbytes;

#define	_HASH(reg,value) \
	(reg) ^= (value); \
	(reg) = ((reg) < 0) ? ((reg) << 1) + 1 : ((reg) << 1);

	verify_pool (pool);

	/* Factor out the GCD of the row... */
	reduce_constraint (rp);

	/* Compute hash value and length of LHS... */
	hval = 0;
	len = 0;
	for (p = rp;; p++) {
		var = p -> var;
		if (var < RC_VAR_BASE) break;
		_HASH (hval, var);
		_HASH (hval, p -> val);
		++len;
	}
	hval %= CPOOL_HASH_SIZE;
	if (hval < 0) {
		hval += CPOOL_HASH_SIZE;
	}

	FATAL_ERROR_IF ((hval < 0) OR (hval >= CPOOL_HASH_SIZE));

	nbytes = (len + 1) * sizeof (*rp);

	hookp = &(pool -> hash [hval]);

	for (row = *hookp; row >= 0;) {
		rcp = &(pool -> rows [row]);
		if ((rcp -> len EQ len) AND
		    (memcmp (rcp -> coefs, rp, nbytes) EQ 0)) {
			/* Constraint already here! */
			return (FALSE);
		}
		row = rcp -> next;
	}

	/* Constraint is not present -- add it.  Start by copying the	*/
	/* coefficients...  If no room, grab another block.		*/
	blkp = pool -> blocks;
	FATAL_ERROR_IF (blkp EQ NULL);
	pool -> num_nz += len;
	++len;		/* include op/rhs in length now... */
	if (blkp -> nfree < len) {
		/* Note: the free space (if any) at the end of the	*/
		/* current block NEVER gets used.  This is by design,	*/
		/* since this results in better cache behavior while	*/
		/* scanning the pool for violations (hitting all of	*/
		/* the rcoefs in each rblk in sequence).		*/

		/* Grab same number as last time... */
		n = (blkp -> ptr - blkp -> base) + blkp -> nfree;
		if (n < len) {
			n = len;
		}
		blkp2 = NEW (struct rblk);
		blkp2 -> next	= blkp;
		blkp2 -> base	= NEWA (n, struct rcoef);
		blkp2 -> ptr	= blkp2 -> base;
		blkp2 -> nfree	= n;
		pool -> blocks = blkp2;
		blkp = blkp2;
	}
	p = blkp -> ptr;
	blkp -> ptr	+= len;
	blkp -> nfree	-= len;
	memcpy (p, rp, len * sizeof (*rp));

	/* Now grab a new row header... */
	row = (pool -> nrows)++;
	if (row >= pool -> maxrows) {
		/* Must grab more rows.  Double it...	*/
		n = 2 * pool -> maxrows;
		rcp = pool -> rows;
		pool -> rows = NEWA (n, struct rcon);
		pool -> maxrows = n;
		memcpy (pool -> rows, rcp, row * sizeof (*rcp));
		free ((char *) rcp);

		/* Increase size of the lprows array also... */
		ip = NEWA (n, int);
		memcpy (ip, pool -> lprows, row * sizeof (*ip));
		free ((char *) (pool -> lprows));
		pool -> lprows = ip;
	}
	rcp = &(pool -> rows [row]);
	rcp -> len	= len - 1;	/* op/rhs not part of length here... */
	rcp -> coefs	= p;
	rcp -> next	= *hookp;
	rcp -> lprow	= -1;
	rcp -> biter	= pool -> iter;	/* assume binding (or violated) now */
	rcp -> hval	= hval;
	rcp -> flags	= 0;
	rcp -> uid	= (pool -> uid)++;
	rcp -> refc	= 0;		/* no OTHER node references it! */
	*hookp = row;

	if (add_to_lp) {
		/* This row is pending addition to the LP tableaux. */
		_gst_mark_row_pending_to_LP (pool, row);
	}

	verify_pool (pool);

	return (TRUE);
}

/*
 * This routine reduces the given constraint row to lowest terms by
 * dividing by the GCD.
 */

	static
	void
reduce_constraint (

struct rcoef *		rp	/* IN - coefficient row */
)
{
int			j;
int			k;
int			rem;
int			com_factor;
struct rcoef *		p;

	/* Initial common factor is first coefficient. */
	com_factor = rp -> val;
	if (com_factor <= 0) {
		FATAL_ERROR_IF (com_factor EQ 0);
		com_factor = - com_factor;
	}

	if (com_factor EQ 1) return;

	for (p = rp + 1; ; p++) {
		k = p -> val;
		if (k <= 0) {
			FATAL_ERROR_IF (k EQ 0);
			k = -k;
		}
		/* Euclid's algorithm: computes GCD... */
		j = com_factor;
		while (j > 0) {
			rem = k % j;
			k = j;
			j = rem;
		}
		com_factor = k;
		if (com_factor EQ 1) return;
		if (p -> var < RC_VAR_BASE) break;
	}

	/* We have a row to reduce! */
	for (p = rp; ; p++) {
		FATAL_ERROR_IF ((p -> val % com_factor) NE 0);
		p -> val /= com_factor;
		if (p -> var < RC_VAR_BASE) break;
	}
}

/*
 * This routine sets up the LP problem instance for the initial
 * constraints of the LP relaxation.
 */

#if CPLEX

	LP_t *
_gst_build_initial_formulation (

struct cpool *		pool,		/* IN - initial constraint pool */
bitmap_t *		vert_mask,	/* IN - set of valid vertices */
bitmap_t *		edge_mask,	/* IN - set of valid hyperedges */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
struct lpmem *		lpmem,		/* OUT - dynamically allocated mem */
gst_param_ptr		params
)
{
int			i, j, k;
int			nedges;
int			nrows;
int			ncoeff;
int			row;
int			var;
int *			tmp;
struct rcon *		rcp;
struct rcoef *		cp;
LP_t *			lp;
int			macsz, marsz, matsz;
int			mac, mar;
int			objsen;
double *		objx;
double *		rhsx;
char *			senx;
double *		bdl;
double *		bdu;
int *			matbeg;
int *			matcnt;
int *			matind;
double *		matval;
cpu_time_t		T0;
cpu_time_t		T1;
double			min_c, max_c, ci;
int			min_exp, max_exp;
int			obj_scale;
char			tbuf [32];

	T0 = _gst_get_cpu_time ();

	nedges = cip -> num_edges;

	/* We know exactly how many columns (variables) we will */
	/* ever need.  We never add additional variables. */
	macsz = nedges;
	mac = macsz;

	/* Build the objective function... */
	objx = NEWA (macsz, double);
	for (i = 0; i < macsz; i++) {
		objx [i] = 0.0;
	}
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (edge_mask, i)) continue;
		objx [i] = (double) (cip -> cost [i]);
	}

	/* CPLEX does not behave well if the objective coefficients	*/
	/* have very large magnitudes.  (If so, we often get "unscaled	*/
	/* infeasibility" error codes.)  Therefore, we scale objx here	*/
	/* (by an exact power of two so that the mantissas remain	*/
	/* unchanged).  Determine a power of two that brings the objx	*/
	/* magnitudes into a reasonable range.				*/

	min_c	= DBL_MAX;
	max_c	= 0.0;
	for (i = 0; i < macsz; i++) {
		ci = fabs (objx [i]);
		if (ci EQ 0.0) continue;
		if (ci < min_c) {
			min_c = ci;
		}
		if (ci > max_c) {
			max_c = ci;
		}
	}

	(void) frexp (min_c, &min_exp);
	(void) frexp (max_c, &max_exp);
	obj_scale = (min_exp + max_exp) / 2;

	/* Remember scale factor so we can unscale results. */
	lpmem -> obj_scale = obj_scale;

	obj_scale = - obj_scale;

	for (i = 0; i < macsz; i++) {
		objx [i] = ldexp (objx [i], obj_scale);
	}

	objsen = _MYCPX_MIN;	/* Minimize */

	/* Build variable bound arrays... */
	bdl = NEWA (macsz, double);
	bdu = NEWA (macsz, double);
	for (i = 0; i < macsz; i++) {
		bdl [i] = 0.0;
		bdu [i] = 1.0;
	}

	mar	= pool -> npend;
	if (pool -> hwmrow EQ 0) {
		/* Initial allocation.  Allocate space sufficiently	*/
		/* large that we are unlikely to need to reallocate the	*/
		/* CPLEX problem buffers...				*/

		/* Start with the total number of non-zeros in the	*/
		/* entire constraint pool...				*/
		ncoeff = 0;
		nrows = pool -> nrows;
		for (i = 0; i < nrows; i++) {
			rcp = &(pool -> rows [i]);
			ncoeff += rcp -> len;
		}

		marsz	= 2 * nrows;
		matsz	= 4 * ncoeff;
	}
	else {
		/* Reallocating CPLEX problem.  We want a moderate rate	*/
		/* of growth, but must trade this off against the	*/
		/* frequency of reallocation.  We expand both the rows	*/
		/* and the non-zeros by 25% over the largest need seen	*/
		/* now or previously.					*/
		ncoeff = 0;
		for (i = 0; i < pool -> npend; i++) {
			row = pool -> lprows [i];
			rcp = &(pool -> rows [row]);
			ncoeff += rcp -> len;
		}
		if ((mar > pool -> hwmrow) OR (ncoeff > pool -> hwmnz)) {
			/* high-water marks should be updated before! */
			FATAL_ERROR;
		}
		marsz = 5 * pool -> hwmrow / 4;
		matsz = 5 * pool -> hwmnz / 4;
	}

	if (marsz < params -> cplex_min_rows) {
		marsz = params -> cplex_min_rows;
	}
	if (matsz < params -> cplex_min_nzs) {
		matsz = params -> cplex_min_nzs;
	}

	gst_channel_printf (params -> print_solve_trace, "cpx allocation: %d rows, %d cols, %d nz\n",
		marsz, macsz, matsz);

	/* Allocate arrays for constraint matrix... */
	rhsx = NEWA (marsz, double);
	senx = NEWA (marsz, char);
	matbeg = NEWA (macsz, int);
	matcnt = NEWA (macsz, int);
	matind = NEWA (matsz, int);
	matval = NEWA (matsz, double);

	for (i = 0; i < marsz; i++) {
		rhsx [i] = 0.0;
	}
	for (i = 0; i < macsz; i++) {
		matbeg [i] = 0;
		matcnt [i] = 0;
	}
	for (i = 0; i < matsz; i++) {
		matind [i] = 0;
		matval [i] = 0.0;
	}

	/* Now go through each row k and compute the number of	*/
	/* non-zero coefficients for each variable used...	*/
	tmp = NEWA (macsz, int);
	for (i = 0; i < macsz; i++) {
		tmp [i] = 0;
	}
	for (i = 0; i < pool -> npend; i++) {
		row = pool -> lprows [i];
		rcp = &(pool -> rows [row]);
		for (cp = rcp -> coefs; ; cp++) {
			var = cp -> var;
			if (var < RC_VAR_BASE) break;
			++(tmp [var - RC_VAR_BASE]);
		}
	}

	/* CPLEX wants columns, not rows... */
	j = 0;
	for (i = 0; i < mac; i++) {
		k = tmp [i];
		matbeg [i] = j;
		tmp [i] = j;
		matcnt [i] = k;
		j += k;
	}
	if (j > pool -> hwmnz) {
		pool -> hwmnz = j;
	}
	if (mar > pool -> hwmrow) {
		pool -> hwmrow = mar;
	}
	for (i = 0; i < pool -> npend; i++) {
		row = pool -> lprows [i];
		rcp = &(pool -> rows [row]);
		for (cp = rcp -> coefs; ; cp++) {
			var = cp -> var;
			if (var < RC_VAR_BASE) break;
			j = tmp [var - RC_VAR_BASE];
			matind [j] = i;
			matval [j] = cp -> val;
			++(tmp [var - RC_VAR_BASE]);
		}
		switch (var) {
		case RC_OP_LE:	senx [i] = 'L';		break;
		case RC_OP_EQ:	senx [i] = 'E';		break;
		case RC_OP_GE:	senx [i] = 'G';		break;
		default:
			FATAL_ERROR;
		}
		rhsx [i] = cp -> val;
		rcp -> lprow = i;
	}

	/* Verify consistency of what we generated... */
	for (i = 0; i < mac; i++) {
		if (tmp [i] NE matbeg [i] + matcnt [i]) {
			fprintf (stderr,
				 "i = %d, tmp = %d, matbeg = %d, matcnt = %d\n",
				 i, tmp [i], matbeg [i], matcnt [i]);
			FATAL_ERROR;
		}
	}

	free ((char *) tmp);

	pool -> nlprows	= pool -> npend;
	pool -> npend	= 0;

#if 0
	_MYCPX_setadvind (1);		/* continue from previous basis. */
#endif

	lp = _MYCPX_loadlp ("root",
			    mac,
			    mar,
			    objsen,
			    objx,
			    rhsx,
			    senx,
			    matbeg,
			    matcnt,
			    matind,
			    matval,
			    bdl,
			    bdu,
			    NULL,
			    macsz,
			    marsz,
			    matsz);

	FATAL_ERROR_IF (lp EQ NULL);

	/* Remember addresses of each buffer for when we need to free them. */
	lpmem -> objx		= objx;
	lpmem -> rhsx		= rhsx;
	lpmem -> senx		= senx;
	lpmem -> matbeg		= matbeg;
	lpmem -> matcnt		= matcnt;
	lpmem -> matind		= matind;
	lpmem -> matval		= matval;
	lpmem -> bdl		= bdl;
	lpmem -> bdu		= bdu;

	T1 = _gst_get_cpu_time ();
	_gst_convert_cpu_time (T1 - T0, tbuf);
	gst_channel_printf (params -> print_solve_trace, "_gst_build_initial_formulation: %s seconds.\n", tbuf);

	return (lp);
}

#endif

/*
 * This routine sets up the LP problem instance for the initial
 * constraints of the LP relaxation.
 */

#if LPSOLVE

	LP_t *
_gst_build_initial_formulation (

struct cpool *		pool,		/* IN - initial constraint pool */
bitmap_t *		vert_mask,	/* IN - set of valid vertices */
bitmap_t *		edge_mask,	/* IN - set of valid hyperedges */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
struct lpmem *		lpmem,		/* OUT - dynamically allocated mem */
gst_param_ptr		params
)
{
int			i;
int			nedges;
int			nrows;
int			ncols;
int			ncoeff;
int			nzi;
int			row;
int			var;
struct rcon *		rcp;
struct rcoef *		cp;
LP_t *			lp;
double *		rowvec;
double *		rhs;
short *			ctype;
int *			matbeg;
int *			matind;
double *		matval;
cpu_time_t		T0;
cpu_time_t		T1;
char			tbuf [32];

	T0 = _gst_get_cpu_time ();

	nedges = cip -> num_edges;

	/* We know exactly how many columns (variables) we will */
	/* ever need.  We never add additional variables. */
	ncols = nedges;

	/* Compute the total number of non-zeros in the INITIAL	*/
	/* constraints of the constraint pool...		*/
	ncoeff = 0;
	nrows = pool -> npend;
	for (i = 0; i < nrows; i++) {
		row = pool -> lprows [i];
		rcp = &(pool -> rows [row]);
		ncoeff += rcp -> len;
	}

	/* Make the initial LP... */
	lp = make_lp (0, ncols);

	/* All variables are 0-1 variables... */
	for (i = 1; i <= nedges; i++) {
		set_bounds (lp, i, 0.0, 1.0);
	}

	/* Minimize */
	set_minim (lp);

	/* Build the objective function... */
	rowvec = NEWA (ncols + 1, double);
	for (i = 0; i <= ncols; i++) {
		rowvec [i] = 0.0;
	}
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (edge_mask, i)) continue;
		rowvec [i + 1] = (double) (cip -> cost [i]);
	}
#if 1
	inc_mat_space (lp, ncols + 1);
#endif
	set_obj_fn (lp, rowvec);

	free ((char *) rowvec);

	/* Allocate arrays for setting the rows... */
	rhs	= NEWA (nrows, double);
	ctype	= NEWA (nrows, short);
	matbeg	= NEWA (nrows + 1, int);
	matind	= NEWA (ncoeff, int);
	matval	= NEWA (ncoeff, double);

	/* Put the rows into the format that LP-solve wants them in...	*/
	nzi = 0;
	for (i = 0; i < nrows; i++) {
		row = pool -> lprows [i];
		rcp = &(pool -> rows [row]);
		matbeg [i] = nzi;
		for (cp = rcp -> coefs; ; cp++) {
			var = cp -> var;
			if (var < RC_VAR_BASE) break;
			matind [nzi] = var - RC_VAR_BASE;
			matval [nzi] = cp -> val;
			++nzi;
		}
		rhs [i] = cp -> val;
		switch (var) {
		case RC_OP_LE:	ctype [i] = REL_LE;	break;
		case RC_OP_EQ:	ctype [i] = REL_EQ;	break;
		case RC_OP_GE:	ctype [i] = REL_GE;	break;
		default:
			FATAL_ERROR;
			break;
		}
		rcp -> lprow = i;
	}
	matbeg [i] = nzi;
	FATAL_ERROR_IF (nzi NE ncoeff);

	if (nrows > pool -> hwmrow) {
		pool -> hwmrow = nrows;
	}
	if (nzi > pool -> hwmnz) {
		pool -> hwmnz = nzi;
	}

	add_rows (lp, 0, nrows, rhs, ctype, matbeg, matind, matval);

	free ((char *) matval);
	free ((char *) matind);
	free ((char *) matbeg);
	free ((char *) ctype);
	free ((char *) rhs);

	pool -> nlprows	= nrows;
	pool -> npend	= 0;

	verify_pool (pool);

#if 0
	{ FILE *	fp;
		fp = fopen ("main.lp", "w");
		FATAL_ERROR_IF (fp EQ NULL);
		write_LP (lp, fp);
		fclose (fp);
	}
#endif

	if (params -> lp_solve_perturb) {
		/* Turn on perturbations to deal with degeneracy... */
		lp -> anti_degen = TRUE;
	}
	if (params -> lp_solve_scale) {
		/* Turn on auto-scaling of the matrix... */
		auto_scale (lp);
	}

	T1 = _gst_get_cpu_time ();
	_gst_convert_cpu_time (T1 - T0, tbuf);
	gst_channel_printf (params -> print_solve_trace, "_gst_build_initial_formulation: %s seconds.\n", tbuf);

	return (lp);
}

#endif

/*
 * This routine solves the current LP relaxation over all constraints
 * currently residing in the constraint pool, regardless of how many
 * are actually in the current LP tableaux.  Each time it solves the
 * LP tableaux, it scans the entire constraint pool for violations.
 * Every violation that is found is appended to the tableaux and we
 * loop back to re-optimize the tableaux.  This procedure terminates
 * only when all constraints in the pool have been satisfied, or a
 * cutoff or infeasibility is encountered.
 *
 * Note also that this procedure NEVER deletes any constraints from
 * the tableaux, slack or otherwise.  Other code must do this.
 */

	int
_gst_solve_LP_over_constraint_pool (

struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			status;
int			ncols;
int			nrows;
struct rcon *		rcp;
LP_t *			lp;
struct bbnode *		nodep;
double *		x;
double *		dj;
struct cpool *		pool;
bool			any_violations;
bool			can_delete_slack;
int			pool_iteration;
double			slack;
double			prev_z;

	INDENT (bbip -> params -> print_solve_trace);

	lp	= bbip -> lp;
	nodep	= bbip -> node;
	pool	= bbip -> cpool;

	ncols	= GET_LP_NUM_COLS (lp);
	nrows	= GET_LP_NUM_ROWS (lp);

	if (nodep -> cpiter EQ pool -> uid) {
		/* nodep -> x is already the optimal solution	*/
		/* over this constraint pool.			*/
#if 1
		gst_channel_printf (bbip -> params -> print_solve_trace, "	Constraint pool unchanged, skip LP solve.\n");
#endif
		/* Global solution info valid for this node... */

		/* Reallocate slack variables vector, if necessary... */
		if (bbip -> slack_size < pool -> nlprows) {
			if (bbip -> slack NE NULL) {
				free ((char *) (bbip -> slack));
			}
			bbip -> slack = NEWA (pool -> nlprows, double);
			bbip -> slack_size = pool -> nlprows;
		}

		for (i = 0; i < nrows; i++) {
			bbip -> slack [i] = 0.0;
		}
		UNINDENT (bbip -> params -> print_solve_trace);
		return (BBLP_OPTIMAL);
	}

	x  = NEWA (2 * ncols, double);
	dj = x + ncols;

	pool_iteration = 0;

	for (;;) {
		prev_z = nodep -> z;

		verify_pool (bbip -> cpool);

		/* Reallocate slack variables vector, if necessary... */
		if (bbip -> slack_size < pool -> nlprows) {
			if (bbip -> slack NE NULL) {
				free ((char *) (bbip -> slack));
			}
			bbip -> slack = NEWA (pool -> nlprows, double);
			bbip -> slack_size = pool -> nlprows;
		}

		status = solve_single_LP (bbip, x, dj, pool_iteration);

		/* Note: The previous call changes bbip->lp when	*/
		/* handling the CPLEX "unscaled infeasibility" issue.	*/

		/* Record another "real" LP solved... */
		do {
			++(pool -> iter);
		} while (pool -> iter EQ -1);

		++pool_iteration;

		if (status NE BBLP_OPTIMAL) break;

		update_lp_solution_history (x, dj, bbip);

		_gst_delete_slack_rows_from_LP (bbip);

#if 0
		if (nodep -> z >= 1.0001 * prev_z) {
			/* Objective rose enough to go ahead	*/
			/* and delete slack rows.  This helps	*/
			/* keep memory usage down on VERY LARGE	*/
			/* problems...				*/
			_gst_delete_slack_rows_from_LP (bbip);
		}
#endif

		verify_pool (bbip -> cpool);

		/* Scan entire pool for violations... */
		rcp = &(pool -> rows [0]);
		any_violations = FALSE;
		for (i = 0; i < pool -> nrows; i++, rcp++) {
			slack = compute_slack_value (rcp -> coefs, x);
			if (slack > FUZZ) {
				/* Row is not binding, much less violated. */
				continue;
			}
			/* Consider this row to be binding now! */
			rcp -> biter = pool -> iter;
			if (rcp -> lprow >= 0) {
				/* Skip this row -- it is already in	*/
				/* the LP tableaux.			*/
				continue;
			}
			if (slack < -FUZZ) {
				/* Constraint "i" is not currently in	*/
				/* the LP tableaux, and is violated.	*/
				/* Add it.				*/
				_gst_mark_row_pending_to_LP (pool, i);
				any_violations = TRUE;
			}
		}

		/* Done if no violations were appended... */
		if (NOT any_violations) {
			/* There are no violated constraints. */
#if LPSOLVE
			/* FIXME -- the LPSOLVE implementation of	*/
			/* _gst_delete_slack_rows_from_LP() always	*/
			/* sets lp->basis_valid = FALSE.  We must not	*/
			/* leave this routine without a valid basis or	*/
			/* it can cause crashes downstream in		*/
			/* try_branch() -- if no more cuts are found at	*/
			/* this node.  The better fix, of course, is to	*/
			/* not trash the basis when deleting rows.	*/
			if (NOT (bbip -> lp -> basis_valid)) {
				/* Solving the LP once again will give	*/
				/* us a valid basis again.		*/
				continue;
			}
#endif
			break;
		}

		can_delete_slack = (nodep -> z >= prev_z + 0.0001 * fabs (prev_z));

		prune_pending_rows (bbip, can_delete_slack);

		/* Time to append these pool constraints to the	*/
		/* current LP tableaux...			*/
		_gst_add_pending_rows_to_LP (bbip);
	}

	free ((char *) x);

	if (status EQ BBLP_OPTIMAL) {
		/* Nodep -> x is optimal for the current version of the	*/
		/* constraint pool.  Skip re-solve if we re-enter this	*/
		/* routine with the same constraint pool.		*/
		nodep -> cpiter = pool -> uid;
	}
	else {
		/* Not optimal -- force re-solve next time so that	*/
		/* correct status is seen next time we're called.  Yes,	*/
		/* it would have been better if we also saved the LP	*/
		/* status in the node...				*/
		nodep -> cpiter = -1;
	}

	verify_pool (bbip -> cpool);
	UNINDENT (bbip -> params -> print_solve_trace);

	return (status);
}

/*
 * This routine copies the current LP solution into the node's buffer,
 * and updates the branch heuristic values.
 */

	static
	void
update_lp_solution_history (

double *		srcx,		/* IN - source LP solution */
double *		dj,		/* IN - source reduced costs */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			j;
int			nedges;
int			dir;
int			dir2;
struct bbnode *		nodep;
double *		dstx;
double *		bheur;
double *		zlb;
double			lb;
double			z;

	nodep	= bbip -> node;
	nedges	= bbip -> cip -> num_edges;

	dstx	= nodep -> x;
	bheur	= nodep -> bheur;

	/* Update the branch heuristic value for each variable.		*/
	/* Variables that have been "stuck" at one value for a long	*/
	/* time receive low bheur values.  If fractional, these tend	*/
	/* to be good branch variables.					*/

	if ((nodep -> num EQ 0) AND (nodep -> iter EQ 0)) {
		/* First iteration: set initial values. */
		for (i = 0; i < nedges; i++) {
			dstx [i] = srcx [i];
			bheur [i] = 0;
		}
	}
	else {
		/* Susequent iterations: use time-decayed average. */
		for (i = 0; i < nedges; i++) {
			bheur [i] = 0.75 * bheur [i] + fabs (srcx [i] - dstx [i]);
			dstx [i] = srcx [i];
		}
	}

	/* Now update the Z lower bounds for each variable	*/
	/* using reduced costs.					*/

	zlb = nodep -> zlb;
	z   = nodep -> z;

	for (j = 0; j < nedges; j++) {
		lb = z + fabs (dj [j]);
		dir = (srcx [j] < 0.5);
		dir2 = 1 - dir;
		i = 2 * j;
		if (lb > zlb [i + dir]) {
			zlb [i + dir] = lb;
		}
		if (z > zlb [i + dir2]) {
			zlb [i + dir2] = z;
		}
	}
}

/*
 * This routine solves a single LP tableaux using LP_SOLVE.
 */

#ifdef LPSOLVE

	static
	int
solve_single_LP (

struct bbinfo *		bbip,		/* IN - branch and bound info */
double *		x,		/* OUT - LP solution variables */
double *		dj,		/* OUT - LP reduced costs */
int			pool_iteration	/* IN - pool iteration number */
)
{
int			i;
int			status;
double			z;
LP_t *			lp;
double *		slack;
int			nslack;
double *		djbuf;
struct gst_hypergraph *	cip;

	(void) pool_iteration;

	verify_pool (bbip -> cpool);

	cip	= bbip -> cip;
	lp	= bbip -> lp;

#if 0
	/* Debug code to dump each LP instance attempted... */
	{ char		buf [64];
		sprintf (buf,
			 "Node.%03d.%03d.%03d.lp",
			 bbip -> node -> num,
			 bbip -> node -> iter + 1,
			 pool_iteration);
		dump_lp (lp, buf);
	}
#endif

	/* Solve the current LP instance... */
	status = solve (lp);

	/* Get current LP solution... */
	z = lp -> best_solution [0];
	for (i = 0; i < cip -> num_edges; i++) {
		x [i] = lp -> best_solution [lp -> rows + i + 1];
	}

	bbip -> node -> z	= z;

	/* Get solution status into solver-independent form... */
	switch (status) {
	case OPTIMAL:		status = BBLP_OPTIMAL;		break;
	case MILP_FAIL:		status = BBLP_CUTOFF;		break;
	case INFEASIBLE:	status = BBLP_INFEASIBLE;	break;
	default:
		gst_channel_printf (bbip -> params -> print_solve_trace, "solve status = %d\n", status);
		FATAL_ERROR;
	}

	/* Grab the reduced costs, for future reference. */
	djbuf = NEWA (lp -> sum + 1, double);
	get_reduced_costs (lp, djbuf);
	memcpy (dj,
		&djbuf [lp -> rows + 1],
		lp -> columns * sizeof (double));
	free ((char *) djbuf);

	/* Grab the values of the slack variables, for future reference. */
	slack = NEWA (lp -> rows + 1, double);
	get_slack_vars (lp, slack);
	memcpy (bbip -> slack, &slack [1], lp -> rows * sizeof (double));
	free ((char *) slack);

	/* Print info about the LP tableaux we just solved... */
	slack = bbip -> slack;
	nslack = 0;
	for (i = 0; i < lp -> rows; i++) {
		if (slack [i] > FUZZ) {
			++nslack;
		}
	}
	(void) gst_channel_printf (bbip -> params -> print_solve_trace, "@PL %d rows, %d cols, %d nonzeros,"
		       " %d slack, %d tight.\n",
		       lp -> rows, lp -> columns, lp -> non_zeros,
		       nslack, lp -> rows - nslack);

	return (status);
}

#endif

/*
 * This routine appends "pool -> npend" new rows onto the end of the
 * LP tableaux from the constraint pool.  The constraint numbers of
 * the actual pool constraints to add reside at the end of the
 * "pool -> lprows []" array (starting with element pool -> nlprows).
 * An additional detail is that for each pool constraint we add, we
 * must record which row of the LP tableaux it now resides in.
 */

#ifdef LPSOLVE

	void
_gst_add_pending_rows_to_LP (

struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			j;
int			i1;
int			i2;
int			newrows;
int			ncoeff;
int			nzi;
int			row;
int			var;
struct rcon *		rcp;
struct rcoef *		cp;
LP_t *			lp;
struct cpool *		pool;
double *		rhs;
short *			ctype;
int *			matbeg;
int *			matind;
double *		matval;

	verify_pool (bbip -> cpool);

	lp	= bbip -> lp;
	pool	= bbip -> cpool;

	if (lp -> rows NE pool -> nlprows) {
		/* LP is out of sync with the pool... */
		FATAL_ERROR;
	}

	/* Get number of rows and non-zeros to add to LP... */
	newrows = pool -> npend;

	FATAL_ERROR_IF (newrows < 0);

	if (newrows EQ 0) return;

	i1	= pool -> nlprows;
	i2	= i1 + newrows;

	/* Get number of rows and non-zeros to add to LP... */
	ncoeff = 0;
	for (i = i1; i < i2; i++) {
		row = pool -> lprows [i];
		rcp = &(pool -> rows [row]);
		/* Constraint not pending? */
		FATAL_ERROR_IF (rcp -> lprow NE -2);
		rcp -> lprow = i;
		ncoeff += rcp -> len;
	}

	if (i2 > pool -> hwmrow) {
		pool -> hwmrow = i2;
	}
	if (lp -> non_zeros + ncoeff > pool -> hwmnz) {
		pool -> hwmnz = lp -> non_zeros + ncoeff;
	}

	/* Allocate arrays for setting the rows... */
	rhs	= NEWA (newrows, double);
	ctype	= NEWA (newrows, short);
	matbeg	= NEWA (newrows + 1, int);
	matind	= NEWA (ncoeff, int);
	matval	= NEWA (ncoeff, double);

	/* Put the rows into the format that LP-solve wants them in... */
	nzi = 0;
	j = 0;
	for (i = i1; i < i2; i++, j++) {
		row = pool -> lprows [i];
		rcp = &(pool -> rows [row]);
		matbeg [j] = nzi;
		for (cp = rcp -> coefs; ; cp++) {
			var = cp -> var;
			if (var < RC_VAR_BASE) break;
			matind [nzi] = var - RC_VAR_BASE;
			matval [nzi] = cp -> val;
			++nzi;
		}
		rhs [j] = cp -> val;
		switch (var) {
		case RC_OP_LE:	ctype [j] = REL_LE;	break;
		case RC_OP_EQ:	ctype [j] = REL_EQ;	break;
		case RC_OP_GE:	ctype [j] = REL_GE;	break;
		default:
			FATAL_ERROR;
			break;
		}
	}
	matbeg [j] = nzi;
	FATAL_ERROR_IF (nzi NE ncoeff);

	gst_channel_printf (bbip -> params -> print_solve_trace, "@PAP adding %d rows, %d nz to LP\n", newrows, ncoeff);

	add_rows (lp, 0, newrows, rhs, ctype, matbeg, matind, matval);

	pool -> nlprows = i2;
	pool -> npend	= 0;

	verify_pool (bbip -> cpool);

	free ((char *) matval);
	free ((char *) matind);
	free ((char *) matbeg);
	free ((char *) ctype);
	free ((char *) rhs);
}

#endif

/*
 * This routine solves a single LP tableaux using CPLEX.
 */

#ifdef CPLEX

	static
	int
solve_single_LP (

struct bbinfo *		bbip,		/* IN - branch and bound info */
double *		x,		/* OUT - LP solution variables */
double *		dj,		/* OUT - LP reduced costs */
int			pool_iteration	/* IN - pool iteration number */
)
{
int			i;
int			status;
double			z;
LP_t *			lp;
double *		slack;
int			nrows;
int			ncols;
int			non_zeros;
int			nslack;
bool			scaling_disabled;
int			small;
int			big;
int			obj_scale;
gst_channel_ptr		print_solve_trace;

	print_solve_trace = bbip -> params -> print_solve_trace;

	(void) pool_iteration;

	lp	= bbip -> lp;

	scaling_disabled = FALSE;

retry_lp:
	/* Solve the current LP instance... */
	status = _MYCPX_dualopt (lp);
	if (status NE 0) {
		gst_channel_printf (print_solve_trace, " WARNING dualopt: status = %d\n", status);
	}

	/* Get current LP solution... */
	i = _MYCPX_solution (lp,
			     &status,		/* solution status */
			     &z,		/* objective value */
			     x,			/* solution variables */
			     NULL,		/* IGNORE dual values */
			     bbip -> slack,	/* slack variables */
			     dj);		/* reduced costs */
	if (i NE 0) {
		fprintf (stderr, "err_code = %d\n", i);
		FATAL_ERROR;
	}

	obj_scale = bbip -> lpmem -> obj_scale;
	ncols	  = _MYCPX_getnumcols (lp);

	if (obj_scale NE 0) {
		/* Unscale CPLEX results. */
		z = ldexp (z, obj_scale);
		for (i = 0; i < ncols; i++) {
			dj [i] = ldexp (dj [i], obj_scale);
		}
	}

	bbip -> node -> z	= z;

	/* Get solution status into solver-independent form... */
	switch (status) {
	case _MYCPX_STAT_OPTIMAL:
		status = BBLP_OPTIMAL;
		break;

	case _MYCPX_STAT_INFEASIBLE:
	case _MYCPX_STAT_UNBOUNDED:	/* (CPLEX sometimes gives this for an	*/
				/* infeasible problem.)			*/
		status = BBLP_INFEASIBLE;
		break;

	case _MYCPX_STAT_ABORT_OBJ_LIM:	/* Objective limit exceeded... */
		status = BBLP_CUTOFF;
		break;

	case _MYCPX_STAT_OPTIMAL_INFEAS:
		/* This means that CPLEX scaled the problem, found an	*/
		/* optimal solution, unscaled the solution, but that	*/
		/* the unscaled solution no longer satisfied all of the	*/
		/* bound or row feasibility tolerances (i.e. the the	*/
		/* unscaled solution is no longer feasible).  We fix	*/
		/* This by turning off scaling and trying again.  Note	*/
		/* that this happens very rarely, but that CPLEX runs	*/
		/* much slower with scaling turned off, so we don't	*/
		/* want to leave scaling off if we can help it...	*/
		if (scaling_disabled) {
			/* CPLEX is never supposed to return this code	*/
			/* when scaling is diabled!			*/
			FATAL_ERROR;
		}

		gst_channel_printf (print_solve_trace, "TURNING OFF SCALING...\n");

		if (_MYCPX_setscaind (-1, &small, &big) NE 0) {
			FATAL_ERROR;
		}

		/* Must reload the entire problem for this to take effect! */
		reload_cplex_problem (bbip);
		lp = bbip -> lp;

		scaling_disabled = TRUE;

		goto retry_lp;

	default:
		fprintf (stderr, "Unexpected status = %d\n", status);
		_MYCPX_lpwrite (lp, "core.lp");
		FATAL_ERROR;
		break;
	}

	if (scaling_disabled) {
		/* Must re-enable scaling, or we'll be really slow! */
		gst_channel_printf (print_solve_trace, "TURNING ON SCALING...\n");
		if (_MYCPX_setscaind (0, &small, &big) NE 0) {
			FATAL_ERROR;
		}

		/* Must reload entire problem for this to take affect! */
		reload_cplex_problem (bbip);
		lp = bbip -> lp;
	}

	/* Print info about the LP tableaux we just solved... */
	nrows	  = _MYCPX_getnumrows (lp);
	ncols	  = _MYCPX_getnumcols (lp);
	non_zeros = _MYCPX_getnumnz (lp);
	slack = bbip -> slack;
	nslack = 0;
	for (i = 0; i < nrows; i++) {
		if (slack [i] > FUZZ) {
			++nslack;
		}
	}
	(void) gst_channel_printf (print_solve_trace, "@PL %d rows, %d cols, %d nonzeros,"
		       " %d slack, %d tight.\n",
		       nrows, ncols, non_zeros,
		       nslack, nrows - nslack);

	return (status);
}

#endif

/*
 * This routine appends "pool -> npend" new rows onto the end of the
 * LP tableaux from the constraint pool.  The constraint numbers of
 * the actual pool constraints to add reside at the end of the
 * "pool -> lprows []" array (starting with element pool -> nlprows).
 * An additional detail is that for each pool constraint we add, we
 * must record which row of the LP tableaux it now resides in.
 */

#ifdef CPLEX

	void
_gst_add_pending_rows_to_LP (

struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			j;
int			i1;
int			i2;
int			newrows;
int			ncoeff;
int			row;
int			nzi;
int			var;
int			num_nz;
struct rcon *		rcp;
struct rcoef *		cp;
LP_t *			lp;
struct cpool *		pool;
double *		rhs;
char *			sense;
int *			matbeg;
int *			matind;
double *		matval;

	verify_pool (bbip -> cpool);

	lp	= bbip -> lp;
	pool	= bbip -> cpool;

	if (_MYCPX_getnumrows (lp) NE pool -> nlprows) {
		/* LP is out of sync with the pool... */
		FATAL_ERROR;
	}

	/* Get number of rows and non-zeros to add to LP... */
	newrows = pool -> npend;

	FATAL_ERROR_IF (newrows < 0);

	if (newrows EQ 0) return;

	i1	= pool -> nlprows;
	i2	= i1 + newrows;

	/* Get number of rows and non-zeros to add to LP... */
	ncoeff = 0;
	for (i = i1; i < i2; i++) {
		row = pool -> lprows [i];
		rcp = &(pool -> rows [row]);
		if (rcp -> lprow NE -2) {
			/* Constraint not pending? */
			FATAL_ERROR;
		}
		rcp -> lprow = i;
		ncoeff += rcp -> len;
	}


	gst_channel_printf (bbip -> params -> print_solve_trace, "@PAP adding %d rows, %d nz to LP\n", newrows, ncoeff);

	num_nz	  = _MYCPX_getnumnz (lp);

	/* Update high-water marks... */
	if (i2 > pool -> hwmrow) {
		pool -> hwmrow = i2;
	}
	if (num_nz + ncoeff > pool -> hwmnz) {
		pool -> hwmnz = num_nz + ncoeff;
	}

#ifndef CPLEX_HAS_CREATEPROB
	/* Check to see if the current CPLEX allocations are	*/
	/* sufficient.  If not, we must reallocate...		*/
	{ int row_space;
	  int nz_space;
	  row_space = _MYCPX_getrowspace (lp);
	  nz_space  = _MYCPX_getnzspace (lp);
	  if ((i2 > row_space) OR (num_nz + ncoeff > nz_space)) {
		/* We must reallocate!  We do this by throwing away the */
		/* old LP completely and building it again from scratch	*/
		/* using only the info available in the constraint	*/
		/* pool.  Hopefully this way we avoid poor memory	*/
		/* utilization due to fragmentation...			*/

		reload_cplex_problem (bbip);

		return;
	  }
	}
#endif

	/* Allocate arrays for setting the rows... */
	rhs	= NEWA (newrows, double);
	sense	= NEWA (newrows, char);
	matbeg	= NEWA (newrows + 1, int);
	matind	= NEWA (ncoeff, int);
	matval	= NEWA (ncoeff, double);

	/* Put the rows into the format that CPLEX wants them in... */
	nzi = 0;
	j = 0;
	for (i = i1; i < i2; i++, j++) {
		row = pool -> lprows [i];
		rcp = &(pool -> rows [row]);
		matbeg [j] = nzi;
		for (cp = rcp -> coefs; ; cp++) {
			var = cp -> var;
			if (var < RC_VAR_BASE) break;
			matind [nzi] = var - RC_VAR_BASE;
			matval [nzi] = cp -> val;
			++nzi;
		}
		rhs [j] = cp -> val;
		switch (var) {
		case RC_OP_LE:	sense [j] = 'L';	break;
		case RC_OP_EQ:	sense [j] = 'E';	break;
		case RC_OP_GE:	sense [j] = 'G';	break;
		default:
			FATAL_ERROR;
			break;
		}
	}
	matbeg [j] = nzi;
	FATAL_ERROR_IF (nzi NE ncoeff);

	i = _MYCPX_addrows (lp,
			    0,
			    newrows,
			    ncoeff,
			    rhs,
			    sense,
			    matbeg,
			    matind,
			    matval,
			    NULL,
			    NULL);

	FATAL_ERROR_IF (i NE 0);

	pool -> nlprows = i2;
	pool -> npend	= 0;

	verify_pool (bbip -> cpool);

	free ((char *) matval);
	free ((char *) matind);
	free ((char *) matbeg);
	free ((char *) sense);
	free ((char *) rhs);
}

#endif

/*
 * This routine frees the current CPLEX problem, and reallocates/rebuilds
 * it from the current constraint pool.  This routine works even if
 * there are constraints pending addition to the LP tableaux.
 */

#if CPLEX

	static
	void
reload_cplex_problem (

struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			j;
int			i1;
int			i2;
int			newrows;
int			row;
int			nedges;
struct rcon *		rcp;
LP_t *			lp;
struct cpool *		pool;
int *			cstat;
int *			rstat;
int *			b_index;
char *			b_lu;
double *		b_bd;

	lp	= bbip -> lp;
	pool	= bbip -> cpool;

	newrows	= pool -> npend;
	i1	= pool -> nlprows;
	i2	= i1 + newrows;

	gst_channel_printf (bbip -> params -> print_solve_trace, "REALLOCATING CPLEX PROBLEM...\n");

	/* Save off the current basis, setting the new	*/
	/* rows to be basic...				*/
	cstat = NEWA (bbip -> cip -> num_edges, int);
	rstat = NEWA (i2, int);
	if (_MYCPX_getbase (lp, cstat, rstat) NE 0) {
		FATAL_ERROR;
	}
	for (i = i1; i < i2; i++) {
		/* Set slack variables for new rows to be basic... */
		rstat [i] = 1;
	}

	/* Free up the current LP... */
	_gst_destroy_initial_formulation (bbip);

	/* Make all LP rows be pending again... */
	for (i = 0; i < pool -> nlprows; i++) {
		row = pool -> lprows [i];
		rcp = &(pool -> rows [row]);
		if (rcp -> lprow < 0) {
			/* Not currently in LP? */
			FATAL_ERROR;
		}
		rcp -> lprow = -2;	/* is now pending... */
	}
	pool -> npend += pool -> nlprows;
	pool -> nlprows = 0;

	/* Build the initial formulation from scratch again... */
	lp = _gst_build_initial_formulation (pool,
					     bbip -> vert_mask,
					     bbip -> edge_mask,
					     bbip -> cip,
					     bbip -> lpmem,
					     bbip -> params);
	bbip -> lp = lp;

	/* The initial formulation bounds all variables	*/
	/* from 0 to 1.  We must restore the proper	*/
	/* bounds for all variables that have been	*/
	/* fixed to 0 or 1...				*/

	nedges = bbip -> cip -> num_edges;
	b_index = NEWA (2 * nedges, int);
	b_lu	= NEWA (2 * nedges, char);
	b_bd	= NEWA (2 * nedges, double);
	j = 0;
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (bbip -> fixed, i)) continue;
		b_index [j]	= i;
		b_lu [j]	= 'L';
		b_index [j+1]	= i;
		b_lu [j+1]	= 'U';
		if (NOT BITON (bbip -> value, i)) {
			b_bd [j]	= 0.0;
			b_bd [j+1]	= 0.0;
		}
		else {
			b_bd [j]	= 1.0;
			b_bd [j+1]	= 1.0;
		}
		j += 2;
	}

	if (j > 0) {
		if (_MYCPX_chgbds (lp, j, b_index, b_lu, b_bd) NE 0) {
			FATAL_ERROR;
		}
	}

	free ((char *) b_bd);
	free ((char *) b_lu);
	free ((char *) b_index);

	/* Restore augmented basis... */
	if (_MYCPX_copybase (lp, cstat, rstat) NE 0) {
		FATAL_ERROR;
	}
	free ((char *) rstat);
	free ((char *) cstat);
}

#endif

/*
 * This routine marks a single row as "pending addition to the LP tableaux,"
 * assuming it is not already pending or in the LP.
 */

	void
_gst_mark_row_pending_to_LP (

struct cpool *		pool,		/* IN - constraint pool */
int			row		/* IN - row to mark pending */
)
{
int			i;
struct rcon *		rcp;

	FATAL_ERROR_IF ((row < 0) OR (row >= pool -> nrows));
	rcp = &(pool -> rows [row]);
	if ((rcp -> lprow >= 0) OR (rcp -> lprow EQ -2)) {
		/* Row is already in the LP, or was previously	*/
		/* made pending...				*/
		return;
	}
	if (rcp -> lprow NE -1) {
		/* Pool constraint has bad state... */
		FATAL_ERROR;
	}

	/* row is now pending... */
	rcp -> lprow = -2;

	i = pool -> nlprows + (pool -> npend)++;
	pool -> lprows [i] = row;
}

/*
 * This routine adds the given list of LOGICAL constraints to the pool
 * as physical constraints (actual coefficient rows).  Any duplicates
 * that might exist in this process are discarded.  Those that
 * represent violations are also appended to the LP tableaux.
 */

	int
_gst_add_constraints (

struct bbinfo *		bbip,		/* IN - branch and bound info */
struct constraint *	lcp		/* IN - list of logical constraints */
)
{
int			nrows;
int			ncoeffs;
struct constraint *	p;
struct cpool *		pool;
struct rcoef *		cp;
double *		x;
bool			add_to_lp;
bool			any_violations;
bool			violation;
bool			newly_added;
int			num_con;

	verify_pool (bbip -> cpool);

	x		= bbip -> node -> x;
	pool		= bbip -> cpool;

	/* Compute the space requirements for these constraints.  Don't	*/
	/* bother trying to account for duplicates or hits on		*/
	/* constraints already in the pool -- assume they are all	*/
	/* unique.							*/
	ncoeffs = 0;
	nrows = 0;
	for (p = lcp; p NE NULL; p = p -> next) {
		cp = _gst_expand_constraint (p, pool -> cbuf, bbip);
		ncoeffs += (cp - pool -> cbuf);
		++nrows;
	}

	if (ncoeffs > pool -> blocks -> nfree) {
		/* Must delete some not recently used rows... */
		garbage_collect_pool (pool, ncoeffs, nrows, bbip -> params);
	}

	any_violations = FALSE;

	num_con = 0;

	while (lcp NE NULL) {
		_gst_expand_constraint (lcp, pool -> cbuf, bbip);

		add_to_lp = FALSE;
		violation = _gst_is_violation (pool -> cbuf, x);
		if (violation) {
			/* Add-to-lp only does anything if this constraint */
			/* is brand new...				   */
			add_to_lp = TRUE;
			any_violations = TRUE;
		}
		newly_added = _gst_add_constraint_to_pool (pool,
							   pool -> cbuf,
							   add_to_lp);
		if (newly_added AND violation) {
			++num_con;
		}

		lcp = lcp -> next;
	}

	if (any_violations) {
		/* Prune back the pending rows so that only the	*/
		/* smallest constraints are added to the LP	*/
		/* the first time around.  The hope is that	*/
		/* this gives a more sparse LP that can be	*/
		/* solved more quickly, and that many of the	*/
		/* larger constraints (that we didn't add in)	*/
		/* will no longer be violated by the new	*/
		/* solution.  In other words, why bog down the	*/
		/* LP solver with lots of dense rows that will	*/
		/* become slack with the slightest perturbation	*/
		/* of the solution?				*/

#if 1
		prune_pending_rows (bbip, FALSE);
#endif

		/* Add the new violations to the LP tableaux... */
		_gst_add_pending_rows_to_LP (bbip);
	}

	print_pool_memory_usage (pool, bbip -> params -> print_solve_trace);

	return (num_con);
}

/*
 * Prune back the pending rows so that only the smallest of these rows
 * are added to the LP tableaux the first time around.  (The larger rows
 * that are "pruned" stay in the pool, however.)  We hope that by adding
 * only the small (i.e. sparse) rows that the following happens:
 *
 *	1.	The resulting LP solves more quickly.
 *	2.	The solution perturbs sufficiently so that
 *		most of the dense rows we avoided adding are
 *		now slack.
 *
 * In other words, why bog down the LP solver with lots of dense rows
 * that will become slack with the slightest perturbation of the
 * LP solution?  If we hold off, we may NEVER have to add them!
 *
 * This pruning process actually seems to make us run SLOWER in practice,
 * probably because it usually results in at least one more scan over
 * the pool and one more LP call before "solve-LP-over-pool" terminates.
 * Therefore, we only do this pruning in cases where the pending rows have
 * an extra large number of non-zero coefficients.  When this happens we
 * trim off the largest rows until the threshold is met.
 */

	static
	void
prune_pending_rows (

struct bbinfo *		bbip,		/* IN - branch-and-bound info */
bool			can_del_slack	/* IN - OK to delete slack rows? */
)
{
int		i;
int		k;
int		n;
int		h;
int		tmp;
int		key;
struct cpool *		pool;
int *		p1;
int *		p2;
int *		p3;
int *		p4;
int *		endp;
int *		parray;
int		total;
struct rcon *	rcp;

#define	THRESHOLD (2 * 1000 * 1000)

	pool = bbip -> cpool;

	/* Get address and count of pending LP rows... */
	n = pool -> npend;
	parray = &(pool -> lprows [pool -> nlprows]);
	endp = &parray [n];

	total = 0;
	for (i = 0; i < n; i++) {
		tmp = parray [i];
		total += pool -> rows [tmp].len;
		if (total > THRESHOLD) break;
	}

	if (total <= THRESHOLD) {
		/* Don't bother pruning anything back... */
		return;
	}

	if (can_del_slack) {
		/* To help keep memory from growing needlessly,	*/
		/* get rid of any slack rows first...		*/
		_gst_delete_slack_rows_from_LP (bbip);
		parray = &(pool -> lprows [pool -> nlprows]);
		endp = &parray [n];
	}

	/* Sort rows in ascending order by number of non-zero coeffs... */

	for (h = 1; h <= n; h = 3*h+1) {
	}

	do {
		h = h / 3;
		p4 = &parray [h];
		p1 = p4;
		while (p1 < endp) {
			tmp = *p1;
			key = pool -> rows [tmp].len;
			p2 = p1;
			while (TRUE) {
				p3 = (p2 - h);
				if (pool -> rows [*p3].len <= key) break;
				*p2 = *p3;
				p2 = p3;
				if (p2 < p4) break;
			}
			*p2 = tmp;
			++p1;
		}
	} while (h > 1);

	/* Find the first I constraints having fewer than the	*/
	/* threshold number of coefficients.			*/
	total = 0;
	for (i = 0; i < n; i++) {
		tmp = parray [i];
		total += pool -> rows [tmp].len;
		if (total > THRESHOLD) break;
	}

	/* Make pending rows i through n-1 not be pending any more... */
	for (k = i; k < n; k++) {
		tmp = parray [k];
		rcp = &(pool -> rows [tmp]);
		FATAL_ERROR_IF (rcp -> lprow NE -2);
		rcp -> lprow = -1;
	}

	/* Reduce the number of pending rows to be only the small ones	*/
	/* we are going to keep...					*/
	pool -> npend = i;

#undef THRESHOLD
}

/*
 * This routine determines whether or not the given physical constraint
 * coefficients are violated by the given solution X.
 */

	bool
_gst_is_violation (

struct rcoef *		cp,		/* IN - row of coefficients */
double *		x		/* IN - LP solution */
)
{
int			var;
double			sum;

	sum = 0.0;
	for (;;) {
		var = cp -> var;
		if (var < RC_VAR_BASE) break;
		sum += (cp -> val * x [var - RC_VAR_BASE]);
		++cp;
	}

	switch (var) {
	case RC_OP_LE:
		return (sum > cp -> val + FUZZ);

	case RC_OP_EQ:
		sum -= cp -> val;
		return ((sum < -FUZZ) OR (FUZZ < sum));

	case RC_OP_GE:
		return (sum + FUZZ < cp -> val);

	default:
		FATAL_ERROR;
		break;
	}

	return (FALSE);
}

/*
 * This routine computes the amount of slack (if any) for the given
 * coefficient row with respect to the given solution X.
 */

	static
	double
compute_slack_value (

struct rcoef *		cp,		/* IN - row of coefficients */
double *		x		/* IN - LP solution */
)
{
int			var;
double			sum;

	sum = 0.0;
	for (;;) {
		var = cp -> var;
		if (var < RC_VAR_BASE) break;
		sum += (cp -> val * x [var - RC_VAR_BASE]);
		++cp;
	}

	switch (var) {
	case RC_OP_LE:
		return (cp -> val - sum);

	case RC_OP_EQ:
		sum -= cp -> val;
		if (sum > 0.0) {
			/* No such thing as slack -- only violation! */
			sum = -sum;
		}
		return (sum);

	case RC_OP_GE:
		return (sum - cp -> val);

	default:
		FATAL_ERROR;
		break;
	}

	return (0.0);
}

/*
 * This routine performs a "garbage collection" on the constraint pool, and
 * is done any time we have too many coefficients to fit into the alloted
 * pool space.
 *
 * Constraints are considered for replacement based upon the product of
 * their size and the number of iterations since they were effective
 * (i.e. binding).  We *never* remove "initial" constraints, since they
 * would never be found by the separation algorithms.  Neither do we
 * remove constraints that have been binding sometime during the most
 * recent few iterations.
 */

	static
	void
garbage_collect_pool (

struct cpool *		pool,		/* IN - constraint pool */
int			ncoeff,		/* IN - approx num coeffs needed */
int			nrows,		/* IN - approx num rows needed */
gst_param_ptr		params
)
{
int			i;
int			j;
int			k;
int			maxsize;
int			minrow;
int			count;
int			time;
int			nz;
int			target;
int			impending_size;
int			min_recover;
struct rcon *		rcp;
int *			cnum;
int32u *		cost;
bool *			delflags;
int *			renum;
int *			ihookp;
struct rblk *		blkp;
struct rcoef *		p1;
struct rcoef *		p2;
struct rcoef *		p3;
struct rblk *		tmp1;
struct rblk *		tmp2;

	gst_channel_printf (params -> print_solve_trace, "Entering garbage_collect_pool\n");
	print_pool_memory_usage (pool, params -> print_solve_trace);

	FATAL_ERROR_IF (pool -> npend > 0);

	maxsize = pool -> nrows - pool -> initrows;

	cnum	= NEWA (maxsize, int);
	cost	= NEWA (maxsize, int32u);

	/* Count non-zeros in all constraints that are binding	*/
	/* for ANY node.  This is the total number of pool	*/
	/* non-zeros that are currently "useful".		*/

	nz = 0;
	for (i = 0; i < pool -> nrows; i++) {
		rcp = &(pool -> rows [i]);
		if ((rcp -> lprow NE -1) OR
		    (rcp -> refc > 0)) {
			/* This row is in (or on its way to) the LP,	*/
			/* or is binding for some suspended node.	*/
			nz += (rcp -> len + 1);
		}
	}

	count = 0;
	for (i = pool -> initrows; i < pool -> nrows; i++) {
		rcp = &(pool -> rows [i]);
		if (rcp -> lprow NE -1) {
			/* NEVER get rid of any row currently in (or on	*/
			/* its way to) the LP tableaux!			*/
			continue;
		}
		if (rcp -> refc > 0) {
			/* NEVER get rid of a row that is binding for	*/
			/* some currently suspended node!		*/
			continue;
		}
		if ((rcp -> flags & RCON_FLAG_DISCARD) NE 0) {
			/* Always discard these! */
			cnum [count]	= i;
			cost [count]	= INT_MAX;
			++count;
			continue;
		}
		time = pool -> iter - rcp -> biter;
#if 1
#define	GRACE_TIME	10
#else
#define	GRACE_TIME	50
#endif
		if (time < GRACE_TIME) {
			/* Give this constraint more time in the pool.	*/
			continue;
		}
#undef GRACE_TIME

		/* This row is a candidate for being deleted! */
		cnum [count]	= i;
		cost [count]	= (rcp -> len + 1) * time;
		++count;
	}

	if (count <= 0) {
		free ((char *) cost);
		free ((char *) cnum);
		return;
	}

	/* Determine how many non-zeros to chomp from the pool.	*/

#if 0
	/* What we want is for the pool to remain proportionate	*/
	/* in size to the LP tableaux, so our target is a	*/
	/* multiple of the high-water mark of non-zeros in the	*/
	/* LP tableaux.						*/

	target = 4 * pool -> hwmnz;
#else
	/* What we want is for the pool to remain proportionate	*/
	/* in size to the number of non-zeros currently being	*/
	/* used by any node.					*/
	target = 16 * nz;
#endif

	if (params -> target_pool_non_zeros > 0) {
		target = params -> target_pool_non_zeros;
	}

	impending_size = pool -> num_nz + ncoeff;

	if (impending_size <= target) {
		free ((char *) cost);
		free ((char *) cnum);
		return;
	}

	min_recover = 3 * ncoeff / 2;

	if ((impending_size > target) AND
	    (impending_size - target > min_recover)) {
		min_recover = impending_size - target;
	}

	/* Sort candidate rows by cost... */
	sort_gc_candidates (cnum, cost, count);

	/* Find most-costly rows to delete that will achieve	*/
	/* the target pool size.				*/
	minrow = pool -> nrows;
	nz = 0;
	i = count - 1;
	for (;;) {
		k = cnum [i];
		nz += pool -> rows [k].len;
		if (k < minrow) {
			minrow = k;
		}
		if (nz >= min_recover) break;
		if (i EQ 0) break;
		--i;
	}

	/* We are deleting this many non-zeros from the pool... */
	pool -> num_nz -= nz;

	/* We want to delete constraints numbered cnum [i..count-1]. */
	delflags = NEWA (pool -> nrows, bool);
	memset (delflags, 0, pool -> nrows);
	for (; i < count; i++) {
		delflags [cnum [i]] = TRUE;
	}

	/* Compute a map for renumbering the constraints that remain.	*/
	renum = NEWA (pool -> nrows, int);
	j = 0;
	for (i = 0; i < pool -> nrows; i++) {
		if (delflags [i]) {
			renum [i] = -1;
		}
		else {
			renum [i] = j++;
		}
	}

	/* Renumber the constraint numbers of the LP tableaux... */
	for (i = 0; i < pool -> nlprows; i++) {
		j = renum [pool -> lprows [i]];
		FATAL_ERROR_IF (j < 0);
		pool -> lprows [i] = j;
	}

	/* Renumber all of the hash table linked lists.  Unthread all	*/
	/* entries that are being deleted.				*/
	for (i = 0; i < CPOOL_HASH_SIZE; i++) {
		ihookp = &(pool -> hash [i]);
		while ((j = *ihookp) >= 0) {
			rcp = &(pool -> rows [j]);
			k = renum [j];
			if (k < 0) {
				*ihookp = rcp -> next;
			}
			else {
				*ihookp = k;
				ihookp = &(rcp -> next);
			}
		}
	}

	/* Delete proper row headers... */
	j = minrow;
	for (i = minrow; i < pool -> nrows; i++) {
		if (delflags [i]) continue;
		pool -> rows [j] = pool -> rows [i];
		++j;
	}
	pool -> nrows = j;

	/* Temporarily reverse the order of the coefficient blocks... */
	blkp = reverse_rblks (pool -> blocks);
	pool -> blocks = blkp;

	/* Now compact the coefficient rows... */
	p1 = blkp -> base;
	p2 = blkp -> ptr + blkp -> nfree;
	for (i = 0; i < pool -> nrows; i++) {
		rcp = &(pool -> rows [i]);
		p3 = rcp -> coefs;
		j = rcp -> len + 1;
		if (p1 + j > p2) {
			/* Not enough room in current block -- on to next. */
			blkp -> ptr = p1;
			blkp -> nfree = p2 - p1;
			blkp = blkp -> next;
			p1 = blkp -> base;
			p2 = blkp -> ptr + blkp -> nfree;
		}
		if (p3 NE p1) {
			rcp -> coefs = p1;
			memmove (p1, p3, j * sizeof (*p1));
		}
		p1 += j;
	}
	blkp -> ptr = p1;
	blkp -> nfree = p2 - p1;

	/* Free up any blocks that are now unused.  Note: this	*/
	/* code assumes the first rblk survives!		*/
	tmp1 = blkp -> next;
	blkp -> next = NULL;
	while (tmp1 NE NULL) {
		tmp2 = tmp1 -> next;
		tmp1 -> next = NULL;
		free ((char *) (tmp1 -> base));
		free ((char *) tmp1);
		tmp1 = tmp2;
	}

	/* Re-reverse list of coefficient blocks so that the one we are	*/
	/* allocating from is first					*/
	pool -> blocks = reverse_rblks (pool -> blocks);

	free ((char *) renum);
	free ((char *) delflags);
	free ((char *) cost);
	free ((char *) cnum);

	print_pool_memory_usage (pool, params -> print_solve_trace);
	gst_channel_printf (params -> print_solve_trace, "Leaving garbage_collect_pool\n");
}

/*
 * This routine sorts the candidate rows in order by cost (of retaining
 * them).
 */

	static
	void
sort_gc_candidates (

int *		cnum,		/* IN - constraint numbers within pool */
int32u *	cost,		/* IN - constraint costs (mem*time products) */
int		n		/* IN - number of candidates */
)
{
int		i;
int		j;
int		h;
int		tmp_cnum;
int32u		tmp_cost;

	for (h = 1; h <= n; h = 3*h+1) {
	}

	do {
		h = h / 3;
		for (i = h; i < n; i++) {
			tmp_cnum = cnum [i];
			tmp_cost = cost [i];
			for (j = i; j >= h; j -= h) {
				if (cost [j - h] <= tmp_cost) break;
				cnum [j] = cnum [j - h];
				cost [j] = cost [j - h];
			}
			cnum [j] = tmp_cnum;
			cost [j] = tmp_cost;
		}
	} while (h > 1);
}

/*
 * This routine reverses a list of rblk structures.
 */

	static
	struct rblk *
reverse_rblks (

struct rblk *		p		/* IN - list of rblk structures */
)
{
struct rblk *		r;
struct rblk *		tmp;

	r = NULL;
	while (p NE NULL) {
		tmp = p -> next;
		p -> next = r;
		r = p;
		p = tmp;
	}
	return (r);
}

/*
 * This routine deletes all rows from the LP that are currently slack.
 * Note that these constraints remain in the pool.  This is purely an
 * efficiency hack designed to limit the number of rows that the LP
 * solver has to contend with at any one time.
 */

#ifdef LPSOLVE

	void
_gst_delete_slack_rows_from_LP (

struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			j;
int			k;
int			n;
int			row;
int			nrows;
int *			dlist;
int *			rowflags;
LP_t *			lp;
struct cpool *		pool;
double *		slack;
struct rcon *		rcp;
struct bbnode *		nodep;

	lp	= bbip -> lp;
	pool	= bbip -> cpool;
	nodep	= bbip -> node;

	if (nodep -> z <= nodep -> delrow_z) {
		/* Cannot delete any rows until the objective value at	*/
		/* this node strictly improves over the last time we	*/
		/* deleted slack rows.					*/
		return;
	}

	nrows	= GET_LP_NUM_ROWS (lp);

	if (nrows NE pool -> nlprows) {
		/* LP and constraint pool are out of sync... */
		FATAL_ERROR;
	}

	slack = bbip -> slack;

	n = pool -> nlprows;

	dlist	= NEWA (nrows, int);

	j = 0;
	k = 0;
	for (i = 0; i < n; i++) {
		row = pool -> lprows [i];
		rcp = &(pool -> rows [row]);
		FATAL_ERROR_IF (rcp -> lprow NE i);
		if (slack [i] > FUZZ) {
			/* This row is slack -- move to delete list... */
			rcp -> lprow = -1;
			dlist [k++] = i;
		}
		else if ((rcp -> flags) & RCON_FLAG_DISCARD) {
			/* This row is to be discarded because it is	*/
			/* shadowed by a new constraint...		*/
			rcp -> lprow = -1;
			dlist [k++] = i;
		}
		else {
			/* Slide constraint up to new position in LP... */
			rcp -> lprow = j;
			pool -> lprows [j++] = row;
		}
	}
	pool -> nlprows = j;

	/* Copy any pending constraints... */
	for (i = 0; i < pool -> npend; i++) {
		pool -> lprows [j++] = pool -> lprows [n + i];
	}

	if (k > 0) {
		/* Time to actually delete the constraints.	*/

		gst_channel_printf (bbip -> params -> print_solve_trace, "@D deleting %d slack rows\n", k);

		rowflags = NEWA (n + 1, int);
		for (i = 0; i <= n; i++) {
			rowflags [i] = 0;
		}
		for (i = 0; i < k; i++) {
			rowflags [1 + dlist [i]] = 1;
		}
		delete_row_set (lp, rowflags);

		free ((char *) rowflags);

		nodep -> delrow_z = nodep -> z;
	}

	free ((char *) dlist);
}

#endif

/*
 * This routine deletes all rows from the LP that are currently slack.
 * Note that these constraints remain in the pool.  This is purely an
 * efficiency hack designed to limit the number of rows that the LP
 * solver has to contend with at any one time.
 */

#ifdef CPLEX

	void
_gst_delete_slack_rows_from_LP (

struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			j;
int			k;
int			n;
int			row;
int			nrows;
int			slack_nz;
int *			dlist;
int *			rowflags;
LP_t *			lp;
struct cpool *		pool;
double *		slack;
struct rcon *		rcp;
struct bbnode *		nodep;

	lp	= bbip -> lp;
	pool	= bbip -> cpool;
	nodep	= bbip -> node;

	if (nodep -> z <= nodep -> delrow_z) {
		/* Cannot delete any rows until the objective value at	*/
		/* this node strictly improves over the last time we	*/
		/* deleted slack rows.					*/
		return;
	}

	nrows	= GET_LP_NUM_ROWS (lp);

	if (nrows NE pool -> nlprows) {
		/* LP and constraint pool are out of sync... */
		FATAL_ERROR;
	}

	slack = bbip -> slack;

	n = pool -> nlprows;

	k = 0;
	slack_nz = 0;
	for (i = 0; i < n; i++) {
		if (slack [i] > FUZZ) {
			++k;
			j = pool -> lprows [i];
			slack_nz += pool -> rows [j].len;
		}
	}

	dlist	= NEWA (nrows, int);

	j = 0;
	k = 0;
	for (i = 0; i < n; i++) {
		row = pool -> lprows [i];
		rcp = &(pool -> rows [row]);
		FATAL_ERROR_IF (rcp -> lprow NE i);
		if (slack [i] > FUZZ) {
			/* This row is slack -- move to delete list... */
			rcp -> lprow = -1;
			dlist [k++] = i;
		}
		else if ((rcp -> flags) & RCON_FLAG_DISCARD) {
			/* This row is to be discarded because it is	*/
			/* shadowed by a new constraint...		*/
			rcp -> lprow = -1;
			dlist [k++] = i;
		}
		else {
			/* Slide constraint up to new position in LP... */
			rcp -> lprow = j;
			pool -> lprows [j++] = row;
		}
	}
	pool -> nlprows = j;

	/* Copy any pending constraints... */
	for (i = 0; i < pool -> npend; i++) {
		pool -> lprows [j++] = pool -> lprows [n + i];
	}

	if (k > 0) {
		/* Time to actually delete the constraints.	*/

		gst_channel_printf (bbip -> params -> print_solve_trace, "@D deleting %d slack rows\n", k);

		rowflags = NEWA (n, int);
		for (i = 0; i < n; i++) {
			rowflags [i] = 0;
		}
		for (i = 0; i < k; i++) {
			rowflags [dlist [i]] = 1;
		}
		if (_MYCPX_delsetrows (lp, rowflags) NE 0) {
			FATAL_ERROR;
		}
		free ((char *) rowflags);

		nodep -> delrow_z = nodep -> z;
	}

	free ((char *) dlist);
}

#endif

/*
 * Free up the LP tableaux for a CPLEX problem.
 */

#ifdef CPLEX

	void
_gst_destroy_initial_formulation (

struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
LP_t *			lp;
struct lpmem *		lpmem;

	lp	= bbip -> lp;
	lpmem	= bbip -> lpmem;

	/* Free up CPLEX's memory... */
	if (_MYCPX_freeprob (&lp) NE 0) {
		FATAL_ERROR;
	}

	/* Free up our own memory... */
	free ((char *) (lpmem -> objx));
	free ((char *) (lpmem -> rhsx));
	free ((char *) (lpmem -> senx));
	free ((char *) (lpmem -> matbeg));
	free ((char *) (lpmem -> matcnt));
	free ((char *) (lpmem -> matind));
	free ((char *) (lpmem -> matval));
	free ((char *) (lpmem -> bdl));
	free ((char *) (lpmem -> bdu));
	memset ((char *) lpmem, 0, sizeof (*lpmem));

	bbip -> lp = NULL;
}

#endif

/*
 * Free up the LP tableaux for an LP_SOLVE problem.
 */

#ifdef LPSOLVE

	void
_gst_destroy_initial_formulation (

struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
	delete_lp (bbip -> lp);

	bbip -> lp = NULL;
}

#endif

/*
 * This routine records the current state of the node's LP tableaux and
 * basis.  Saving and restoring this information permits rapid switching
 * between nodes, at the price of some extra memory in each node.  It also
 * *LOCKS* this node's constraints in the pool (by bumping reference counts)
 * so that processing other nodes does not cause this node's constraints to
 * be deleted.
 *
 * For each column in the LP tableaux we record its basis status (basic,
 * non-basic at lower bound, or non-basic at upper bound).
 *
 * For each row in the LP tableaux, we record tuples containing the following
 * info:
 *
 *	1. The unique ID of the constraint in the pool.
 *	2. The row's basis status.
 *	3. The constraints position in the LP tableaux.
 *
 * These tuples are listed in order by unique ID.  Since this is the
 * order in which these constraints appear in the pool (with perhaps
 * additional constraints interleaved), it is easy to locate these pool
 * rows later on (after the pool has been modified) by scanning the list
 * and the pool in parallel.  We put the rows back in exactly the same
 * position they were in so that saving and restoring the basis works
 * correctly.  (This really seems necessary for lp_solve -- grog!)
 */

	void
_gst_save_node_basis (

struct bbnode *		nodep,		/* IN - BB node to save basis for */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			j;
int			k;
int			n;
int			nvars;
int			nrows;
struct cpool *		pool;
LP_t *			lp;
struct rcon *		rcp;

	lp	= bbip -> lp;
	pool	= bbip -> cpool;
	nrows	= pool -> nrows;
	n	= pool -> nlprows;

	nvars	= GET_LP_NUM_COLS (lp);

	FATAL_ERROR_IF (n NE GET_LP_NUM_ROWS (lp));

	nodep -> n_uids		= n;
	nodep -> bc_uids	= NEWA (n, int);
	nodep -> bc_row		= NEWA (n, int);
	nodep -> rstat		= NEWA (n, int);
	nodep -> cstat		= NEWA (nvars, int);

#ifdef CPLEX
	if (_MYCPX_getbase (lp, nodep -> cstat, nodep -> rstat) NE 0) {
		FATAL_ERROR;
	}
#endif

#ifdef LPSOLVE
	get_current_basis (lp, nodep -> cstat, nodep -> rstat);
#endif

	/* Now record the rows and bump the reference counts... */
	j = 0;
	rcp = &(pool -> rows [0]);
	for (i = 0; i < nrows; i++, rcp++) {
		k = rcp -> lprow;
		if (k < 0) continue;
		++(rcp -> refc);
		nodep -> bc_uids [j]	= rcp -> uid;
		nodep -> bc_row [j]	= k;
		++j;
	}

	FATAL_ERROR_IF (j NE nodep -> n_uids);
}

/*
 * This routine restores the LP tableaux and basis to the state it was
 * in when we did a "save-node-basis" on the given node.  We do this when
 * resuming work on the given node.  This consists of the following steps:
 *
 *	- Delete all constraints from the LP tableaux.
 *	- Make each of the necessary constraints pending, decrementing
 *	  reference counts as you go.
 *	- Add all pending rows to the LP tableaux.
 *	- Restore the LP basis.
 *
 * If the reference count on a row becomes zero, then no other node considers
 * this row to be binding and it is OK to delete the row from the pool (when
 * it becomes slack for THIS node).
 */

	void
_gst_restore_node_basis (

struct bbnode *		nodep,		/* IN - BB node to restore */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			j;
int			n;
int			row;
int			uid;
int			n_uids;
LP_t *			lp;
struct cpool *		pool;
struct rcon *		rcp;
struct rcon *		rcp_endp;
int *			rowflags;

	lp	= bbip -> lp;
	pool	= bbip -> cpool;

	FATAL_ERROR_IF (nodep -> bc_uids EQ NULL);

	/* Transition all rows back to the "not-in-LP-tableaux" state. */
	n = GET_LP_NUM_ROWS (lp);
	FATAL_ERROR_IF ((n NE pool -> nlprows) OR (pool -> npend NE 0));
	for (i = 0; i < n; i++) {
		row = pool -> lprows [i];
		FATAL_ERROR_IF ((row < 0) OR (row >= pool -> nrows));
		rcp = &(pool -> rows [row]);
		FATAL_ERROR_IF (rcp -> lprow NE i);
		rcp -> lprow = -1;
	}
	pool -> nlprows = 0;

	/* Delete all rows from the LP tableaux... */
	rowflags = NEWA (n + 1, int);
	for (i = 0; i <= n; i++) {
		rowflags [i] = 1;
	}

#ifdef CPLEX
	if (_MYCPX_delsetrows (lp, rowflags) NE 0) {
		FATAL_ERROR;
	}
#endif

#ifdef LPSOLVE
	rowflags [0] = 0;	/* keep the objective row! */
	delete_row_set (lp, rowflags);
#endif

	free ((char *) rowflags);

	n_uids	= nodep -> n_uids;

	rcp	 = pool -> rows;
	rcp_endp = rcp + pool -> nrows;

	for (i = 0; i < n_uids; i++) {
		pool -> lprows [i] = -1;
	}

	for (i = 0; i < n_uids; i++) {
		uid = nodep -> bc_uids [i];
		j   = nodep -> bc_row [i];
		for (;;) {
			if (rcp >= rcp_endp) {
				/* Row not found! */
				FATAL_ERROR;
			}
			if (rcp -> uid EQ uid) break;
			++rcp;
		}
		--(rcp -> refc);
		row = rcp - pool -> rows;
		/* Make constraint number "row" be the "j-th" pending	*/
		/* constraint. */
		if ((rcp -> lprow NE -1) OR
		    (j < 0) OR
		    (j >= n_uids) OR
		    (pool -> lprows [j] NE -1)) {
			FATAL_ERROR;
		}
		rcp -> lprow = -2;
		pool -> lprows [j] = row;
	}
	pool -> npend = n_uids;

	/* Load all pending rows into the LP tableaux! */
	_gst_add_pending_rows_to_LP (bbip);

	if ((nodep -> cstat NE NULL) AND (nodep -> rstat NE NULL)) {
		/* We have a basis to restore... */
#ifdef CPLEX
		i = _MYCPX_copybase (lp, nodep -> cstat, nodep -> rstat);
		FATAL_ERROR_IF (i NE 0);
#endif

#ifdef LPSOLVE
		set_current_basis (lp, nodep -> cstat, nodep -> rstat);
#endif
		free ((char *) (nodep -> rstat));
		free ((char *) (nodep -> cstat));
	}

	/* Free up the list of UIDs... */
	nodep -> n_uids = 0;

	free ((char *) (nodep -> bc_uids));
	free ((char *) (nodep -> bc_row));

	nodep -> bc_uids = NULL;
	nodep -> bc_row	 = NULL;
	nodep -> rstat	 = NULL;
	nodep -> cstat	 = NULL;
}

/*
 * This routine destroys basis information saved in the given node.  The
 * only really important things to do are to decrement the constraint
 * reference counts and free up the memory.
 */

	void
_gst_destroy_node_basis (

struct bbnode *		nodep,		/* IN - BB node to restore */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			uid;
int			n_uids;
struct cpool *		pool;
struct rcon *		rcp;
struct rcon *		rcp_endp;

	if (nodep -> n_uids <= 0) return;

	pool	= bbip -> cpool;

	FATAL_ERROR_IF ((nodep -> bc_uids EQ NULL) OR
			(nodep -> rstat EQ NULL) OR
			(nodep -> cstat EQ NULL));

	n_uids	= nodep -> n_uids;

	rcp	 = pool -> rows;
	rcp_endp = rcp + pool -> nrows;

	for (i = 0; i < n_uids; i++) {
		uid = nodep -> bc_uids [i];
		for (;;) {
			if (rcp >= rcp_endp) {
				/* Row not found! */
				FATAL_ERROR;
			}
			if (rcp -> uid EQ uid) break;
			++rcp;
		}
		--(rcp -> refc);
	}

	/* Free up the list of UIDs... */
	nodep -> n_uids = 0;

	free ((char *) (nodep -> bc_uids));
	free ((char *) (nodep -> bc_row));
	free ((char *) (nodep -> rstat));
	free ((char *) (nodep -> cstat));

	nodep -> bc_uids = NULL;
	nodep -> bc_row	 = NULL;
	nodep -> rstat	 = NULL;
	nodep -> cstat	 = NULL;
}

/*
 * This routine retrieves the current basis under lp_solve, which records
 * the basis differently than CPLEX.  We use the "cstat" array to hold
 * the column upper/lower bound flags of the non-basic columns, and in
 * "rstat" we indicate which column is the basic variable for that row
 * (including slack variable columns).
 */

#ifdef LPSOLVE

	static
	void
get_current_basis (

LP_t *		lp,		/* IN - LP tableaux to get basis of */
int *		cstat,		/* OUT - basis flags for each column */
int *		rstat		/* OUT - basis flags for each row */
)
{
int		i;
int		j;

	if (NOT (lp -> basis_valid)) {
		/* Scribble out the default starting basis. */
		for (i = 0; i < lp -> rows; i++) {
			rstat [i] = i + 1;
		}
		for (i = 0; i < lp -> columns; i++) {
			cstat [i] = 1;
		}
		return;
	}

	/* Set the row status flags... */
	for (i = 0; i < lp -> rows; i++) {
		rstat [i] = lp -> bas [i + 1];
	}

	/* Set the column status flags... */
	j = 0;
	for (i = 1; i <= lp -> sum; i++) {
		if (lp -> basis [i]) continue;
		cstat [j] = lp -> lower [i];
		++j;
	}
}

#endif

/*
 * This routine sets the current basis under lp_solve, which records
 * the basis differently than CPLEX.  We use the "cstat" array to hold
 * the column upper/lower bound flags, and in "rstat" we indicate which
 * column is the basic variable for that row (including slack variable
 * columns).
 */

#ifdef LPSOLVE

	static
	void
set_current_basis (

LP_t *		lp,		/* IN - LP tableaux to set basis of */
int *		cstat,		/* IN - basis flags for each column */
int *		rstat		/* IN - basis flags for each row */
)
{
int		i;
int		j;

	for (i = 1; i <= lp -> sum; i++) {
		lp -> basis [i] = 0;
		lp -> lower [i] = 1;
	}

	/* Set the row status flags... */
	for (i = 0; i < lp -> rows; i++) {
		j = rstat [i];
		lp -> bas [i + 1] = j;
		lp -> basis [j] = 1;
	}

	/* Set the column status flags... */
	j = 0;
	for (i = 1; i <= lp -> sum; i++) {
		if (lp -> basis [i]) continue;
		lp -> lower [i] = cstat [j];
		++j;
	}

	lp -> basis_valid = TRUE;
	lp -> eta_valid = FALSE;
}

#endif

/*
 * This routine prints debugging information about the amount of memory
 * currently being used by the constraint pool.
 */

	static
	void
print_pool_memory_usage (

struct cpool *		pool,		/* IN - constraint pool */
gst_channel_ptr		param_print_solve_trace
)
{
int32u			nblks;
int32u			nzfree;
int32u			nzwaste;
int32u			nztotal;
int32u			used;
struct rblk *		p;

	nblks = 0;
	nzfree = 0;
	nzwaste = 0;
	nztotal = 0;

	for (p = pool -> blocks; p NE NULL; p = p -> next) {
		if (nblks <= 0) {
			nzfree += p -> nfree;
		}
		else {
			nzwaste += p -> nfree;
		}
		used = p -> ptr - p -> base;
		nztotal += (used + p -> nfree);
		++nblks;
	}

	gst_channel_printf (param_print_solve_trace, "@PMEM %d rows,"
		" %u blocks, %u nzfree, %u nzwasted, %u nztotal\n",
		pool -> nrows, nblks, nzfree, nzwaste, nztotal);
}

/*
 * This routine verifies the consistency of the constraint pool.
 */

	static
	void
verify_pool (

struct cpool *		pool		/* IN - constraint pool */
)
{
#if 0
int			i;
int			j;
int			nvars;
int			row;
int			var;
struct rcon *		rcp;
struct rcoef *		cp;

	nvars	= pool -> nvars;

	for (i = 0; i < pool -> nrows; i++) {
		rcp = &(pool -> rows [i]);
		j = 0;
		for (cp = rcp -> coefs; ; cp++) {
			var = cp -> var;
			if (var < 0) break;
			FATAL_ERROR_IF (var >= nvars);
			++j;
		}
		FATAL_ERROR_IF ((var NE RC_OP_LE) AND
				(var NE RC_OP_EQ) AND
				(var NE RC_OP_GE));
		FATAL_ERROR_IF (rcp -> len NE j);
		j = rcp -> lprow;
		if (j >= 0) {
			FATAL_ERROR_IF (j >= pool -> nlprows);
			FATAL_ERROR_IF (pool -> lprows [j] NE i);
		}
	}

	for (i = 0; i < pool -> nlprows; i++) {
		j = pool -> lprows [i];
		FATAL_ERROR_IF ((j < 0) OR (j >= pool -> nrows));
		rcp = &(pool -> rows [j]);
		FATAL_ERROR_IF (rcp -> lprow NE i);
	}
#endif
}

/*
 * This routine prints out a constraint for debugging purposes...
 */

	void
_gst_debug_print_constraint (

char *			msg1,	/* IN - message to print (1st line). */
char *			msg2,	/* IN - message to print (2nd line). */
struct constraint *	lcp,	/* IN - logical constraint to print... */
double *		x,	/* IN - LP solution (or NULL). */
struct bbinfo *		bbip,	/* IN - branch-and-bound info. */
gst_channel_ptr		param_print_solve_trace
)
{
int			j, k;
int			nedges;
int			mcols1;
int			mcols2;
int			col;
double			z;
bool			first;
struct gst_hypergraph *	cip;
char *			sp;
struct rcoef *		cbuf;
struct rcoef *		cp;
char			buf [64];

	cip	= bbip -> cip;
	nedges	= cip -> num_edges;

	cbuf	= NEWA (nedges + 1, struct rcoef);

	_gst_expand_constraint (lcp, cbuf, bbip);

	mcols1 = strlen (msg1);
	mcols2 = strlen (msg2);
	gst_channel_printf (param_print_solve_trace, "%s", msg1);
	col = mcols1;

	first = TRUE;
	z = 0.0;
	for (cp = cbuf; ; cp++) {
		k = cp -> var;
		if (k < RC_VAR_BASE) break;
		k -= RC_VAR_BASE;
		first = sprint_term (buf, first, cp -> val, k);
		j = strlen (buf);
		if (col + j >= 72) {
			gst_channel_printf (param_print_solve_trace, "\n%s", msg2);
			col = mcols2;
		}
		gst_channel_printf (param_print_solve_trace, "%s", buf);
		col += j;
		if (x NE NULL) {
			z += (cp -> val) * x [k];
		}
	}

	switch (k) {
	case RC_OP_LE:	sp = " <=";	break;
	case RC_OP_EQ:	sp = " =";	break;
	case RC_OP_GE:	sp = " >=";	break;
	default:
		sp = "";
		FATAL_ERROR;
	}

	j = strlen (sp);
	if (col + j >= 72) {
		gst_channel_printf (param_print_solve_trace, "\n%s", msg2);
		col = mcols2;
	}
	gst_channel_printf (param_print_solve_trace, "%s", sp);
	col += j;

	sprintf (buf, " %d", cp -> val);

	j = strlen (buf);
	if (col + j >= 72) {
		gst_channel_printf (param_print_solve_trace, "\n%s", msg2);
		col = mcols2;
	}
	gst_channel_printf (param_print_solve_trace, "%s", buf);

	if (x NE NULL) {
		sprintf (buf, " (%f)", z);
		j = strlen (buf);
		if (col + j >= 72) {
			gst_channel_printf (param_print_solve_trace, "\n%s", msg2);
			col = mcols2;
		}
		gst_channel_printf (param_print_solve_trace, "%s", buf);
	}

	gst_channel_printf (param_print_solve_trace, "\n");

	free ((char *) cbuf);
}

/*
 * This routine "sprint's" a single term of a linear equation into
 * the given text buffer.
 */

	static
	bool
sprint_term (

char *		buf,		/* OUT - buffer to sprint into. */
bool		first,		/* IN - TRUE iff first term. */
int		coeff,		/* IN - coefficient of term. */
int		var		/* IN - variable. */
)
{
	if (coeff EQ 0) {
		buf [0] = '\0';
		return (first);
	}

	if (NOT first) {
		*buf++ = ' ';
	}
	if (coeff < 0) {
		*buf++ = '-';
		*buf++ = ' ';
		coeff = - coeff;
	}
	else if (NOT first) {
		*buf++ = '+';
		*buf++ = ' ';
	}

	if (coeff NE 1) {
		(void) sprintf (buf, "%u ", (int32u) coeff);
		buf = strchr (buf, '\0');
	}

	(void) sprintf (buf, "x%u", (int32u) var);

	return (FALSE);
}

/*
 * Special debugging routine to display the constraint pool.  An
 * option flag says whether to display only constraints that are
 * currently in the LP tableaux.
 */

	void
_gst_print_constraint_pool (

struct bbinfo *	bbip,		/* IN - branch-and-bound info */
bool		only_LP		/* IN - display only rows in the LP? */
)
{
int			i;
int			k;
int			nedges;
int			row;
int			var;
struct cpool *		pool;
struct gst_hypergraph *	cip;
double *		C;
char			ch;
struct rcon *		rcp;
struct rcoef *		cp;
double			coeff;
gst_channel_ptr		trace;

	trace = bbip -> params -> print_solve_trace;

	cip	= bbip -> cip;
	pool	= bbip -> cpool;

	nedges = cip -> num_edges;

	gst_channel_printf (trace, "Minimize\n");


#ifdef CPLEX
	C = NEWA (nedges, double);

	if (_MYCPX_getobj (bbip -> lp, C, 0, nedges - 1) NE 0) {
		FATAL_ERROR;
	}
	for (i = 0; i < nedges; i++) {
		coeff = C [i];
		if (coeff EQ 0.0) continue;
		ch = '+';
		if (coeff < 0.0) {
			coeff = - coeff;
			ch = '-';
		}
		gst_channel_printf (trace, "\t%c %f x%d\n", ch, coeff, i);
	}
	free ((char *) C);
#endif

#ifdef LPSOLVE
	C = NEWA (nedges + 1, double);
	get_row (bbip -> lp, 0, C);
	for (i = 0; i < nedges; i++) {
		coeff = C [i + 1];
		if (coeff EQ 0.0) continue;
		ch = '+';
		if (coeff < 0.0) {
			coeff = - coeff;
			ch = '-';
		}
		gst_channel_printf (trace, "\t%c %f x%d\n", ch, coeff, i);
	}
	free ((char *) C);
#endif

	gst_channel_printf (trace, "\nSubject To\n");

	for (row = 0; row < pool -> nrows; row++) {
		rcp = &(pool -> rows [row]);
		if (only_LP AND (rcp -> lprow < 0)) continue;
		gst_channel_printf (trace, "\nc%d:\n", row);
		for (cp = rcp -> coefs; ; cp++) {
			var = cp -> var;
			if (var < RC_VAR_BASE) break;
			var -= RC_VAR_BASE;
			k = cp -> val;
			ch = '+';
			if (k < 0) {
				k = -k;
				ch = '-';
			}
			gst_channel_printf (trace, "\t%c %d x%d\n",
					    ch, k, var);
		}
		switch (var) {
		case RC_OP_LE:
			gst_channel_printf (trace, "\t<= %d\n", cp -> val);
			break;

		case RC_OP_GE:
			gst_channel_printf (trace, "\t>= %d\n", cp -> val);
			break;

		case RC_OP_EQ:
			gst_channel_printf (trace, "\t= %d\n", cp -> val);
			break;

		default:
			FATAL_ERROR;
			break;
		}
	}

	gst_channel_printf (trace, "\nBounds\n\n");

	for (i = 0; i < nedges; i++) {
		if (BITON (bbip -> edge_mask, i)) {
			gst_channel_printf (trace, "	0 <= x%d <= 1\n", i);
		}
	}
}
