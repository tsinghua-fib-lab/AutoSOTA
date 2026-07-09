/***********************************************************************

	$Id: expand.c,v 1.12 2022/11/19 13:45:51 warme Exp $

	File:	expand.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Expand logical constraints into physical constraints.

************************************************************************

	Modification Log:

	a-1:	01/16/2001	warme
		: Split off from constrnt.c.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.
	e-4:	11/18/2022	warme
		: Convert sparse subtours from compile-time option
		:  to an algorithm parameter.
		: Simplify calling conventions.

************************************************************************/

#include "expand.h"

#include "bb.h"
#include "constrnt.h"
#include "fatal.h"
#include "logic.h"
#include "parmblk.h"
#include "steiner.h"


/*
 * Global Routines
 */

struct rcoef *	_gst_expand_constraint (struct constraint *	lcp,
					struct rcoef *		cp,
					struct bbinfo *		bbip);

/*
 * Local Functions
 */

static struct rcoef *	expand_normal_subtour (
					struct constraint *	lcp,
					struct rcoef *		cp,
					bitmap_t *		edge_mask,
					struct gst_hypergraph *	cip,
					struct bbinfo *		bbip);
static struct rcoef *	expand_sparse_subtour (
					struct constraint *	lcp,
					struct rcoef *		cp,
					bitmap_t *		edge_mask,
					struct gst_hypergraph *	cip,
					struct bbinfo *		bbip);

/*
 * Expand the given subtour constraint into its physical coefficient,
 * op, rhs form.  This version chooses whether to do the "sparse subtour"
 * optimization or not, depending upon the parameter setting.
 */

	struct rcoef *
_gst_expand_constraint (

struct constraint *	lcp,		/* IN - logical constraint */
struct rcoef *		cp,		/* OUT - coefficient row */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
struct gst_hypergraph *	cip;
struct gst_param *	params;
bitmap_t *		edge_mask;

	cip		= bbip -> cip;
	params		= bbip -> params;
	edge_mask	= cip -> initial_edge_mask;

	if (params -> sparse_subtours) {
		cp = expand_sparse_subtour (lcp, cp, edge_mask, cip, bbip);
	}
	else {
		cp = expand_normal_subtour (lcp, cp, edge_mask, cip, bbip);
	}

	return (cp);
}

/*
 * This routine expands a "logical" constraint into its coefficient/OP/RHS
 * form.
 *
 * This is the smart version that knows about the sparse representation
 * of large subtours.
 */

	static
	struct rcoef *
expand_sparse_subtour (

struct constraint *	lcp,		/* IN - logical constraint */
struct rcoef *		cp,		/* OUT - coefficient row */
bitmap_t *		edge_mask,	/* IN - set of valid hyperedges */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			j;
int			t;
int			ssize;
int			isize;
int			ccount;
int			nedges;
int			kmasks;
struct rcoef *		orig_cp;
bitmap_t *		bp1;
bitmap_t		mask;
int *			vp1;
int *			vp2;

	nedges = cip -> num_edges;

	orig_cp = cp;

	switch (lcp -> type) {
	case CT_CUTSET:
		/* We are given a set F of full sets... */
		for (i = 0; i < nedges; i++) {
			if (NOT BITON (lcp -> mask, i)) continue;
			if (NOT BITON (edge_mask, i)) continue;
			cp -> var = i + RC_VAR_BASE;
			cp -> val = 1.0;
			++cp;
		}
		if (cp <= orig_cp) {
			/* Empty cutset! */
			FATAL_ERROR;
		}
		cp -> var = RC_OP_GE;
		cp -> val = 1.0;
		break;

	case CT_SUBTOUR:
		/* We are given a set S of terminals...  Get size */
		/* of subtour - 1... */
		ssize = -1;
		bp1 = &(lcp -> mask [0]);
		kmasks = cip -> num_vert_masks;
		for (i = 0; i < kmasks; i++) {
			mask = *bp1++;
			ssize += NBITSON (mask);
		}
		/* Compute coefficients of the subtour constraint.  At	*/
		/* the same time, count the number of non-zeros in the	*/
		/* complementary representation of the constraint	*/
		/* (obtained by subtracting the subtour row from the	*/
		/* total degree equation).  We actually emit the one	*/
		/* that is more sparse.					*/
		ccount = 0;
		for (j = 0; j < nedges; j++) {
			if (NOT BITON (edge_mask, j)) continue;
			vp1 = cip -> edge [j];
			vp2 = cip -> edge [j + 1];
			isize = -1;
			while (vp1 < vp2) {
				t = *vp1++;
				if (BITON (lcp -> mask, t)) {
					++isize;
				}
			}
			if (isize <= 0) {
				/* edge is entirely outside subtour but	*/
				/* entirely inside complement.		*/
				++ccount;
			}
			else {
				if (isize < cip -> edge_size [i] - 1) {
					/* Edge is partly in subtour and */
					/* partly in the complement. */
					++ccount;
				}
				cp -> var = j + RC_VAR_BASE;
				cp -> val = isize;
				++cp;
			}
		}
		if (ccount >= (cp - orig_cp)) {
			/* Emit the standard subtour constraint. */
			cp -> var = RC_OP_LE;
			cp -> val = ssize;
			break;
		}
		/* Start over and emit the complementary constraint. */
		cp = orig_cp;
		for (j = 0; j < nedges; j++) {
			if (NOT BITON (edge_mask, j)) continue;
			vp1 = cip -> edge [j];
			vp2 = cip -> edge [j + 1];
			isize = 0;
			while (vp1 < vp2) {
				t = *vp1++;
				if (BITON (lcp -> mask, t)) {
					++isize;
				}
			}
			if (isize <= 0) {
				++isize;
			}
			isize = cip -> edge_size [j] - isize;
			if (isize <= 0) continue;
			cp -> var = j + RC_VAR_BASE;
			cp -> val = isize;
			++cp;
		}
		cp -> var = RC_OP_GE;
		cp -> val = cip -> num_verts - (ssize + 1);
		break;

	case CT_RAW:
		/* Already in sparse row form.  Just copy it out. */
		orig_cp = ((struct rcoef *) (lcp -> mask));
		for (;;) {
			*cp++ = *orig_cp;
			if (orig_cp -> var < RC_VAR_BASE) break;
			++orig_cp;
		}
		break;

	default:
		FATAL_ERROR;
		break;
	}

	return (cp);
}

/*
 * This routine expands a "logical" constraint into its coefficient/OP/RHS
 * form.
 */

	static
	struct rcoef *
expand_normal_subtour (

struct constraint *	lcp,		/* IN - logical constraint */
struct rcoef *		cp,		/* OUT - coefficient row */
bitmap_t *		edge_mask,	/* IN - set of valid hyperedges */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
struct bbinfo *		bbip		/* IN - branch-and-bound info */
)
{
int			i;
int			j;
int			t;
int			ssize;
int			isize;
int			nedges;
int			kmasks;
struct rcoef *		orig_cp;
bitmap_t *		bp1;
bitmap_t		mask;
int *			vp1;
int *			vp2;

	nedges = cip -> num_edges;

	switch (lcp -> type) {
	case CT_CUTSET:
		orig_cp = cp;
		/* We are given a set F of full sets... */
		for (i = 0; i < nedges; i++) {
			if (NOT BITON (lcp -> mask, i)) continue;
			if (NOT BITON (edge_mask, i)) continue;
			cp -> var = i + RC_VAR_BASE;
			cp -> val = 1.0;
			++cp;
		}
		if (cp <= orig_cp) {
			/* Empty cutset! */
			FATAL_ERROR;
		}
		cp -> var = RC_OP_GE;
		cp -> val = 1.0;
		break;

	case CT_SUBTOUR:
		/* We are given a set S of terminals...  Get size */
		/* of subtour - 1... */
		ssize = -1;
		bp1 = &(lcp -> mask [0]);
		kmasks = cip -> num_vert_masks;
		for (i = 0; i < kmasks; i++) {
			mask = *bp1++;
			ssize += NBITSON (mask);
		}
		for (j = 0; j < nedges; j++) {
			if (NOT BITON (edge_mask, j)) continue;
			vp1 = cip -> edge [j];
			vp2 = cip -> edge [j + 1];
			isize = -1;
			while (vp1 < vp2) {
				t = *vp1++;
				if (BITON (lcp -> mask, t)) {
					++isize;
				}
			}
			if (isize <= 0) continue;
			cp -> var = j + RC_VAR_BASE;
			cp -> val = isize;
			++cp;
		}
		cp -> var = RC_OP_LE;
		cp -> val = ssize;
		break;

	case CT_RAW:
		/* Already in sparse row form.  Just copy it out. */
		orig_cp = ((struct rcoef *) (lcp -> mask));
		for (;;) {
			*cp++ = *orig_cp;
			if (orig_cp -> var < RC_VAR_BASE) break;
			++orig_cp;
		}
		break;

	default:
		FATAL_ERROR;
		break;
	}

	return (cp);
}
