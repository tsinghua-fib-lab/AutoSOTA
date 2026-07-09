/***********************************************************************

	$Id: analyze.c,v 1.12 2022/11/19 13:45:49 warme Exp $

	File:	analyze.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1995, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	A program that determines the subtour S associated with
	each constraint in the constraint pool.

************************************************************************

	Modification Log:

	a-1:	02/25/2000	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Upgrade fatals.
	e-4:	11/18/2022	warme
		: Change calling convention.
		: Fix -Wall issue.

************************************************************************/

#include "analyze.h"

#include "bb.h"
#include "constrnt.h"
#include "expand.h"
#include "fatal.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "steiner.h"
#include <string.h>


/*
 * Global Routines
 */

void		_gst_analyze_constraints (struct bbinfo * bbip, bool lp_only);


/*
 * Local Types
 */

struct ainfo {
	struct bbinfo *		bbip;	/* Branch and bound info */
	struct rcoef *		cp1;	/* Constraint to process */
	int			len;	/* Length of constrain LHS */
	int			rhs;	/* Constraint RHS */
	struct rcoef *		cp2;	/* Constraint to match */
	bitmap_t *		Sset;	/* TRUE if vertex value known */
	bitmap_t *		Sval;	/* vertex value, 0 or 1 */
	bitmap_t *		enz;	/* TRUE if edge has coef NE 0 */
	bitmap_t *		edone;	/* TRUE if edge fully processed */
	int *			vstack;	/* Temporary vertex stack */
	int *			vmap;	/* Map vertices to LP columns */
	int *			vlist;	/* List of undetermined vertices */
	int *			emap;	/* Map edges to index within elist */
	int *			elist;	/* Temporary edge list */
	int			num_le;	/* Number <= constraints processed */
	int			dec_le;	/* Number <= constraints decoded */
	int			dec_le_lp; /* num decoded via LP */
	int			num_ge;	/* Number >= constraints processed */
	int			dec_ge;	/* Number >= constraints decoded */
	int			dec_ge_lp; /* num decoded via LP */
	int			num_eq;	/* Number EQ constraints seen */
};

struct clist {
	int		nrows;		/* Number of LP rows */
	int		ncols;		/* Number of LP columns */
	double *	obj;		/* Objective coefficients */
	int **		rows;		/* Row data */
	char *		op;		/* Operator for each row */
	int *		rhs;		/* RHS for each row */
};


/*
 * Local Routines
 */

static void		build_constraints (struct ainfo *	aip,
					   int			numS,
					   struct clist *	clist);
static LP_t *		build_lp (struct ainfo *	aip,
				  struct clist *	clist,
				  struct lpmem *	lpmem);
static void		destroy_lp (LP_t *, struct lpmem *);
/* static void		process_ge (struct ainfo *); */
static void		process_le (struct ainfo *, int *, int *);
static void		use_lp (struct ainfo *, int *, int, int *);
static bool		verify_subtour (struct ainfo *, int *);

/*
 * This routine attempts to determine the subset S associated with each
 * subtour constraint in the constraint pool.
 */

	void
_gst_analyze_constraints (

struct bbinfo *		bbip,		/* IN - branch and bound info */
bool			lp_only		/* IN - LP constraints, or all? */
)
{
int			i;
int			j;
int			k;
int			nverts;
int			nedges;
int			kmasks;
int			nmasks;
int			len;
int			nrows;
int			var;
struct gst_hypergraph *	cip;
struct cpool *		pool;
struct rcon *		rcp;
struct rcoef *		cp;
struct rcoef *		rbuf;
int *			A;
struct ainfo		ainfo;

	cip	= bbip -> cip;
	pool	= bbip -> cpool;

	nverts	= cip -> num_verts;
	nedges	= cip -> num_edges;
	kmasks	= cip -> num_vert_masks;
	nmasks	= cip -> num_edge_masks;

	rbuf = NEWA (nedges + 1, struct rcoef);
	A = NEWA (nedges, int);

	ainfo.bbip	= bbip;
	ainfo.cp1	= NULL;
	ainfo.len	= 0;
	ainfo.rhs	= 0;
	ainfo.cp2	= NULL;
	ainfo.Sset	= NEWA (kmasks, bitmap_t);
	ainfo.Sval	= NEWA (kmasks, bitmap_t);
	ainfo.enz	= NEWA (nmasks, bitmap_t);
	ainfo.edone	= NEWA (nmasks, bitmap_t);
	ainfo.vstack	= NEWA (nverts, int);
	ainfo.vmap	= NEWA (nverts, int);
	ainfo.vlist	= NEWA (nverts, int);
	ainfo.emap	= NEWA (nedges, int);
	ainfo.elist	= NEWA (nedges, int);
	ainfo.num_le	= 0;
	ainfo.dec_le	= 0;
	ainfo.dec_le_lp	= 0;
	ainfo.num_ge	= 0;
	ainfo.dec_ge	= 0;
	ainfo.dec_ge_lp	= 0;
	ainfo.num_eq	= 0;

	for (i = 0; i < kmasks; i++) {
		ainfo.Sset [i] = 0;
		ainfo.Sval [i] = 0;
	}
	for (i = 0; i < nmasks; i++) {
		ainfo.enz [i] = 0;
		ainfo.edone [i] = 0;
	}
	for (i = 0; i < nverts; i++) {
		ainfo.vmap [i] = -1;
	}
	for (i = 0; i < nedges; i++) {
		ainfo.emap [i] = -1;
	}

	nrows = pool -> nrows;
	i = pool -> initrows;
	if (lp_only) {
		/* Do ALL subtours in the LP, even those with */
		/* only 2 vertices or with N-1 vertices... */
		i = 1;
	}
	for ( ; i < nrows; i++) {
		rcp = &(pool -> rows [i]);
		len = rcp -> len;
		cp = rcp -> coefs;

		if (lp_only AND (rcp -> lprow < 0)) continue;

		switch (cp [len].var) {
		case RC_OP_LE:
			++ainfo.num_le;
			for (j = 0; j < len; j++) {
				var = cp [j].var;
				FATAL_ERROR_IF (var < RC_VAR_BASE);
				var -= RC_VAR_BASE;
				rbuf [j].var	= var;
				rbuf [j].val	= cp [j].val;
			}
			rbuf [len] = cp [len];	/* copy operator and RHS. */
			ainfo.cp1 = rbuf;
			ainfo.len = len;
			ainfo.rhs = cp [len].val;
			ainfo.cp2 = cp;
			process_le (&ainfo, &ainfo.dec_le, &ainfo.dec_le_lp);
			break;

		case RC_OP_GE:
			++ainfo.num_ge;
#if 1
			/* Subtract constraint from total degree row. */
			for (j = 0; j < nedges; j++) {
				if (NOT BITON (cip -> initial_edge_mask, j)) {
					A [j] = 0;
				}
				else {
					A [j] = cip -> edge_size [j] - 1;
				}
			}
			for (j = 0; j < len; j++) {
				k = cp [j].var;
				FATAL_ERROR_IF (k < RC_VAR_BASE);
				k -= RC_VAR_BASE;
				A [k] -= cp [j].val;
			}
			k = 0;
			for (j = 0; j < nedges; j++) {
				if (A [j] EQ 0) continue;
				rbuf [k].var = j;
				rbuf [k].val = A [j];
				++k;
			}
			rbuf [k].var = RC_OP_LE;
			rbuf [k].val = nverts - cp [len].val - 1;
			len = k;
			ainfo.cp1 = rbuf;
			ainfo.len = len;
			ainfo.rhs = rbuf [len].val;
			ainfo.cp2 = cp;
			process_le (&ainfo, &ainfo.dec_ge, &ainfo.dec_ge_lp);
#endif
			break;

		case RC_OP_EQ:
			++ainfo.num_eq;
			break;

		default:
			FATAL_ERROR;
			break;
		}
	}

	free ((char *) ainfo.vstack);
	free ((char *) ainfo.edone);
	free ((char *) ainfo.enz);
	free ((char *) ainfo.Sval);
	free ((char *) ainfo.Sset);
	free ((char *) A);
	free ((char *) rbuf);

	printf ("<= Constraints: processed = %d, decoded = %d + %d with LP\n",
		ainfo.num_le, ainfo.dec_le, ainfo.dec_le_lp);
	printf (">= Constraints: processed = %d, decoded = %d + %d with LP\n",
		ainfo.num_ge, ainfo.dec_ge, ainfo.dec_ge_lp);
	printf ("EQ Constraints: seen = %d\n",
		ainfo.num_eq);
}

/*
 * Process a less-than-or-equal constraint.
 */

	static
	void
process_le (

struct ainfo *		aip,
int *			counter,
int *			lp_counter
)
{
int			i;
int			j;
int			k;
int			e;
int			ae;
int			numS;
int			num0;
int			num1;
int			len;
int			rhs;
struct bbinfo *		bbip;
struct gst_hypergraph *	cip;
struct rcoef *		cp;
struct rcoef *		cp1;
struct rcoef *		cp2;
bitmap_t *		Sset;
bitmap_t *		Sval;
bitmap_t *		enz;
bitmap_t *		edone;
bitmap_t *		edge_mask;
int *			vp1;
int *			vp2;
int *			ep1;
int *			ep2;
int *			vstack;
int *			vsp;
int *			vsp_process;
int *			vsp_temp;
int *			vsp_temp_end;
bool			changed;

	bbip	= aip -> bbip;
	cip	= bbip -> cip;

	cp	= aip -> cp1;
	len	= aip -> len;
	rhs	= aip -> rhs;
	Sset	= aip -> Sset;
	Sval	= aip -> Sval;
	enz	= aip -> enz;
	edone	= aip -> edone;
	vstack	= aip -> vstack;

	edge_mask = cip -> initial_edge_mask;

	cp2 = cp + len;

	cp1 = cp;
	while (cp1 < cp2) {
		e = cp1 -> var;
		++cp1;
		SETBIT (enz, e);
	}

	numS = 0;	/* No vertices yet known to be in S. */

	vsp = vstack;	/* Keep track of all vertices fixed, whether in	*/
			/* S or not in S. */

	/* First, process all edges lying entirely within the subtour. */
	cp1 = cp;
	while (cp1 < cp2) {
		e = cp1 -> var;
		ae = cp1 -> val;
		++cp1;
		if (BITON (edone, e)) continue;
		if (ae NE cip -> edge_size [e] - 1)  continue;
		vp1 = cip -> edge [e];
		vp2 = cip -> edge [e + 1];
		while (vp1 < vp2) {
			j = *vp1++;
			if (BITON (Sset, j)) continue;
			SETBIT (Sset, j);
			SETBIT (Sval, j);
			*vsp++ = j;
			++numS;
		}
		SETBIT (edone, e);
	}
	if (numS <= 0) {
		/* Nowhere easy to start on this one! */
		goto skip_iteration;
	}

	vsp_process = vstack;

	/* Now iteratively discover the subtour set S. */
	while (numS - 1 < rhs) {
		changed = FALSE;

		/* Process all members of S discovered since last time. */
		while (vsp_process < vsp) {
			j = *vsp_process++;
			FATAL_ERROR_IF (NOT BITON (Sset, j));
			if (NOT BITON (Sval, j)) continue;
			/* Vertex j was recently discovered to be in S. */
			ep1 = cip -> term_trees [j];
			ep2 = cip -> term_trees [j + 1];
			while (ep1 < ep2) {
				e = *ep1++;
				if (NOT BITON (edge_mask, e)) continue;
				if (BITON (enz, e)) continue;
				/* Edge e has a zero coefficient, and	*/
				/* Contains vertex j \in S.  All other	*/
				/* vertices in edge e must NOT be in S.	*/
				vp1 = cip -> edge [e];
				vp2 = cip -> edge [e + 1];
				while (vp1 < vp2) {
					k = *vp1++;
					if (k EQ j) continue;
					if (BITON (Sset, k)) {
						if (BITON (Sval, k)) {
#if 1
							printf ("Inconsistent A\n");
#endif
							goto fail;
						}
						continue;
					}
					SETBIT (Sset, k);
					CLRBIT (Sval, k);
					*vsp++ = k;
				}
			}
		}

		cp1 = cp;
		while (cp1 < cp2) {
			e = cp1 -> var;
			ae = cp1 -> val;
			++cp1;
			if (BITON (edone, e)) continue;

			/* Count vertices of edge e known to be in S	*/
			/* and those known to be NOT in S.		*/
			num0 = 0;
			num1 = 0;
			vp1 = cip -> edge [e];
			vp2 = cip -> edge [e + 1];
			while (vp1 < vp2) {
				j = *vp1++;
				if (NOT BITON (Sset, j)) continue;
				if (BITON (Sval, j)) {
					++num1;
				}
				else {
					++num0;
				}
			}
			if (num1 - 1 EQ ae) {
				/* All subtour vertices of edge e are known! */
				/* All other vertices must be in V \ S. */
				vp1 = cip -> edge [e];
				vp2 = cip -> edge [e + 1];
				while (vp1 < vp2) {
					j = *vp1++;
					if (BITON (Sset, j)) continue;
					SETBIT (Sset, j);
					CLRBIT (Sval, j);
					*vsp++ = j;
					changed = TRUE;
				}
				SETBIT (edone, e);
			}
			else if (num0 + ae EQ cip -> edge_size [e] - 1) {
				/* All non-subtour verts of edge e known! */
				/* The rest must be in V. */
				vp1 = cip -> edge [e];
				vp2 = cip -> edge [e + 1];
				while (vp1 < vp2) {
					j = *vp1++;
					if (BITON (Sset, j)) continue;
					SETBIT (Sset, j);
					SETBIT (Sval, j);
					*vsp++ = j;
					++numS;
					changed = TRUE;
				}
				SETBIT (edone, e);
			}
		}
		/* Make a *temporary* list of all unresolved vertices */
		/* referenced by the constraint. */
		vsp_temp = vsp;
		vsp_temp_end = vsp;
		cp1 = cp;
		while (cp1 < cp2) {
			e = cp1 -> var;
			++cp1;
			if (BITON (edone, e)) continue;
			vp1 = cip -> edge [e];
			vp2 = cip -> edge [e + 1];
			while (vp1 < vp2) {
				j = *vp1++;
				if (BITON (Sset, j)) continue;
				SETBIT (Sset, j);
				*vsp_temp_end++ = j;
			}
		}
		for (vp1 = vsp_temp; vp1 < vsp_temp_end; vp1++) {
			j = *vp1;
			CLRBIT (Sset, j);
		}
		/* Look for unresolved vertices that CAN'T be in subtour. */
		while (vsp_temp < vsp_temp_end) {
			j = *vsp_temp++;
			ep1 = cip -> term_trees [j];
			ep2 = cip -> term_trees [j + 1];
			while (ep1 < ep2) {
				e = *ep1++;
				if (NOT BITON (edge_mask, e)) continue;
				if (BITON (enz, e)) continue;
				vp1 = cip -> edge [e];
				vp2 = cip -> edge [e + 1];
				while (vp1 < vp2) {
					k = *vp1++;
					if (NOT BITON (Sset, k)) continue;
					if (NOT BITON (Sval, k)) continue;
					/* Edge e has a zero coefficient. */
					/* Vertex j CANNOT be in S, or else */
					/* vertices j,k \in S and j,k \in e */
					/* so that edge e would have a */
					/* coefficient of at least 1. */
					SETBIT (Sset, j);
					CLRBIT (Sval, j);
					*vsp++ = j;
					changed = TRUE;
					goto advance;
				}
			}
advance:		;
		}
		if (NOT changed) break;
	}

	/* Special hack for the case when S = V-{1}, and the only  */
	/* edges adjacent to vertices 0 and 1 are {0,1} and {1,2}. */
	/* In this case, we have discovered 2..N-1 \in S and	   */
	/* 1 \notin S, but the status of vertex 0 is still not	   */
	/* known.						   */

	if ((numS EQ rhs) AND
	    ((rhs + 2) EQ cip -> num_verts)) {
		for (i = 0; i < cip -> num_verts; i++) {
			if (NOT BITON (Sset, i)) {
				SETBIT (Sset, i);
				SETBIT (Sval, i);
				++numS;
				*vsp++ = i;
				break;
			}
		}
	}

skip_iteration:

	if (numS EQ rhs + 1) {
		if (verify_subtour (aip, vsp)) {
			/* We have found the subtour S!  Hooray! */
			++(*counter);
		}
		else {
			printf ("Verify failed: |S| = %d, |lhs| = %d\n",
				numS, len);
		}
	}
	else if (numS < rhs + 1) {
		/* Formulate the rest as an LP... */
		use_lp (aip, vsp, rhs + 1 - numS, lp_counter);
	}

fail:	;

	/* Clean up. */
	while (vstack < vsp) {
		j = *--vsp;
		CLRBIT (Sset, j);
		CLRBIT (Sval, j);
	}
	cp1 = cp;
	while (cp1 < cp2) {
		e = cp1 -> var;
		++cp1;
		CLRBIT (edone, e);
		CLRBIT (enz, e);
	}
}

/*
 * Verify that we have found the subtour.
 */

	static
	bool
verify_subtour (

struct ainfo *		aip,
int *			vsp
)
{
int			i;
int			j;
int			k;
int			n;
int			numS;
int			nedges;
int			kmasks;
struct bbinfo *		bbip;
struct gst_hypergraph *	cip;
bitmap_t *		stour;
bitmap_t *		Sset;
bitmap_t *		Sval;
int *			ip;
int *			S;
struct rcoef *		rbuf;
struct rcoef *		cp1;
struct rcoef *		cp2;
struct rcoef *		cp3;
struct constraint	con;
bool			match;
bool			swapped;

	bbip	= aip -> bbip;
	cip	= bbip -> cip;
	nedges	= cip -> num_edges;
	kmasks	= cip -> num_vert_masks;

	Sset	= aip -> Sset;
	Sval	= aip -> Sval;

	stour = NEWA (kmasks, bitmap_t);
	for (i = 0; i < kmasks; i++) {
		stour [i] = 0;
	}

	ip = aip -> vstack;
	numS = 0;
	while (ip < vsp) {
		j = *ip++;
		FATAL_ERROR_IF (NOT BITON (Sset, j));
		if (NOT BITON (Sval, j)) continue;
		SETBIT (stour, j);
		++numS;
	}
	con.next	= NULL;
	con.iteration	= 0;
	con.type	= CT_SUBTOUR;
	con.mask	= stour;

	rbuf = NEWA (nedges + 1, struct rcoef);

	cp3 = _gst_expand_constraint (&con, rbuf, bbip);

	match = FALSE;

	cp1 = aip -> cp2;
	cp2 = rbuf;
	for (;;) {
		if (cp2 >= cp3) {
			if (cp1 -> var < RC_VAR_BASE) break;
			printf ("LHS too short!\n");
			goto getout;
		}
		if (cp1 -> var < RC_VAR_BASE) {
			printf ("LHS too long!\n");
			goto getout;
		}
		if ((cp1 -> var NE cp2 -> var) OR
		    (cp1 -> val NE cp2 -> val)) {
			printf ("LHS mismatch!\n");
			goto getout;
		}
		++cp1;
		++cp2;
	}
	if (cp3 -> var NE cp1 -> var) {
		printf ("Operator mismatch!\n");
		goto getout;
	}
	if (cp3 -> val NE cp1 -> val) {
		printf ("RHS mismatch!\n");
		goto getout;
	}

	match = TRUE;

#if 1
	/* Sort and print out the decoded constraint. */
	ip = aip -> vstack;
	if (2 * numS > cip -> num_verts) {
		/* Print out V-S form. */
		printf ("%d	", numS - cip -> num_verts);
		for (i = 0; i < kmasks; i++) {
			stour [i] = ~stour [i];
		}
	}
	else {
		/* Print out normal S form. */
		printf ("%d	", numS);
	}
	S = NEWA (numS, int);
	n = 0;
	for (i = 0; i < cip -> num_verts; i++) {
		if (NOT BITON (stour, i)) continue;
		S [n] = i;
		++n;
	}
	do {
		swapped = FALSE;
		for (i = 1; i < n; i++) {
			j = S [i - 1];
			k = S [i];
			if (j > k) {
				S [i - 1] = k;
				S [i]	  = j;
				swapped = TRUE;
			}
		}
	} while (swapped);
	k = 0;
	for (i = 0; i < n; i++) {
		if (k >= 10) {
			printf ("\n\t");
			k = 0;
		}
		printf (" %d", S [i]);
		++k;
	}
	printf ("\n");

	free ((char *) S);
#endif

getout:

	free ((char *) rbuf);
	free ((char *) stour);

	return (match);
}

/*
 * Now attempt to solve the problem by setting up an LP.
 */

	static
	void
use_lp (

struct ainfo *		aip,
int *			vsp,
int			shortfall,	/* IN - num elts of S left to find */
int *			lp_counter
)
{
int			i;
int			j;
int			status;
LP_t *			lp;
double *		x;
struct clist		clist;
struct lpmem		lpmem;


	build_constraints (aip, shortfall, &clist);

	lp = build_lp (aip, &clist, &lpmem);

	x = NEWA (clist.ncols, double);

#if CPLEX
	status = _MYCPX_primopt (lp);
	if (status NE 0) {
		printf ("CPXprimopt: status = %d\n", status);
	}

	i = _MYCPX_solution (lp, &status, NULL, x, NULL, NULL, NULL);
	if (i NE 0) {
		printf ("CPXsolution: return code = %d\n", i);
	}
	if (status NE _MYCPX_STAT_OPTIMAL) {
		printf ("CPXsolution: status = %d\n", status);
	}
#endif

#if LPSOLVE
	status = solve (lp);
	if (status NE OPTIMAL) {
		printf ("solve: status = %d\n", status);
	}
	for (i = 0; i < clist.ncols; i++) {
		x [i] = lp -> best_solution [lp -> rows + i + 1];
	}
#endif

	for (i = 0; i < clist.ncols; i++) {
		if (fabs (x [i]) < 0.0001) continue;
		if (fabs (x [i] - 1.0) >= 0.0001) {
			goto non_integral;
		}
	}

	/* Apparently we have an integral solution!  Add the vertices	*/
	/* found to those already accumulated and see if we get the	*/
	/* correct constraint...					*/

	for (i = 0; i < clist.ncols; i++) {
		if (x [i] <= 0.75) continue;
		j = aip -> vlist [i];
		SETBIT (aip -> Sset, j);
		SETBIT (aip -> Sval, j);
		*vsp++ = j;
	}
	if (verify_subtour (aip, vsp)) {
		/* We have found the subtour S!  Hooray! */
		++(*lp_counter);
	}
	else {
		printf ("Verify of LP solution failed.\n");
	}

	/* Restore the "set" and "value" bitmaps... */
	for (i = 0; i < clist.ncols; i++) {
		if (x [i] <= 0.75) continue;
		j = aip -> vlist [i];
		CLRBIT (aip -> Sset, j);
		CLRBIT (aip -> Sval, j);
	}

non_integral:

	destroy_lp (lp, &lpmem);

	free ((char *) clist.rhs);
	free ((char *) clist.op);
	free ((char *) clist.rows [0]);
	free ((char *) clist.rows);
	free ((char *) clist.obj);
}

/*
 * Build the list of constraints in an LP-solver independent format.
 */

	static
	void
build_constraints (

struct ainfo *		aip,
int			shortfall,
struct clist *		clist
)
{
int			i;
int			j;
int			k;
int			e;
int			num1;
int			len;
int			numv;
int			nrows;
int			ncols;
int			nzsize;
int			row;
int			v;
struct bbinfo *		bbip;
struct gst_hypergraph *	cip;
struct rcoef *		cp;
struct rcoef *		cp1;
struct rcoef *		cp2;
bitmap_t *		Sset;
bitmap_t *		Sval;
bitmap_t *		enz;
bitmap_t *		edone;
bitmap_t *		edge_mask;
int *			vp1;
int *			vp2;
int *			ep1;
int *			ep2;
int *			vmap;
int *			vlist;
int *			emap;
int *			elist;
int *			ip;
double *		objx;
int			cedge_rows;
int			cedge_nz;
int			zedge_rows;
int			zedge_nz;

	bbip	= aip -> bbip;
	cip	= bbip -> cip;

	cp	= aip -> cp1;
	len	= aip -> len;
	Sset	= aip -> Sset;
	Sval	= aip -> Sval;
	enz	= aip -> enz;
	edone	= aip -> edone;
	vmap	= aip -> vmap;
	vlist	= aip -> vlist;
	emap	= aip -> emap;
	elist	= aip -> elist;

	edge_mask = cip -> initial_edge_mask;

	cp2 = cp + len;

	/* Make a list of all unresolved vertices in the constraint. */
	cp1 = cp;
	k = 0;
	while (cp1 < cp2) {
		e = cp1 -> var;
		++cp1;
		if (BITON (edone, e)) continue;
		vp1 = cip -> edge [e];
		vp2 = cip -> edge [e + 1];
		while (vp1 < vp2) {
			j = *vp1++;
			if (BITON (Sset, j)) continue;
			if (vmap [j] >= 0) continue;
			vlist [k] = j;
			vmap [j] = k;
			++k;
		}
	}
	numv	= k;
	ncols	= k;

	/* Build an essentially random objective function.	*/
	/* All we really want is a feasible solution.		*/
	objx = NEWA (ncols, double);
	k = 123456789;
	for (i = 0; i < ncols; i++) {
		objx [i] = fabs ((double) (k % 20011));
		k = k * 10007 + 5523;
	}

	/* Count the number of "unfinished edge" rows we need. */
	/* (Each unfinished row has at least 2 unresolved vertices.) */

	cedge_rows = 0;
	cedge_nz = 0;
	for (cp1 = cp; cp1 < cp2; ++cp1) {
		e = cp1 -> var;
		if (BITON (edone, e)) continue;

		k = 0;
		vp1 = cip -> edge [e];
		vp2 = cip -> edge [e + 1];
		while (vp1 < vp2) {
			j = *vp1++;
			if (vmap [j] < 0) continue;
			++k;
		}
		FATAL_ERROR_IF (k < 2);
		++cedge_rows;
		cedge_nz += k;
	}

	/* Find all edges NOT in the constraint (i.e., having a		*/
	/* coefficient of zero) that have AT LEAST two unresolved	*/
	/* vertices in them.  Add these edges to the list.		*/

	zedge_rows = 0;
	zedge_nz = 0;
	for (i = 0; i < numv; i++) {
		j = vlist [i];
		ep1 = cip -> term_trees [j];
		ep2 = cip -> term_trees [j + 1];
		while (ep1 < ep2) {
			e = *ep1++;
			if (NOT BITON (edge_mask, e)) continue;
			if (BITON (enz, e)) continue;
			if (emap [e] >= 0) continue;
			vp1 = cip -> edge [e];
			vp2 = cip -> edge [e + 1];
			k = 0;
			while (vp1 < vp2) {
				v = *vp1++;
				if (vmap [v] < 0) continue;
				++k;
			}
			if (k < 2) continue;
			elist [zedge_rows] = e;
			emap [e] = zedge_rows;
			++zedge_rows;
			zedge_nz += k;
		}
	}

	nrows	= cedge_rows + zedge_rows + 1;
	nzsize	= cedge_nz   + zedge_nz   + numv;

	clist -> nrows		= nrows;
	clist -> ncols		= ncols;
	clist -> obj		= objx;
	clist -> rows		= NEWA (nrows + 1, int *);
	clist -> op		= NEWA (nrows, char);
	clist -> rhs		= NEWA (nrows, int);

	row = 0;
	ip = NEWA (nzsize, int);

	/* Generate one row per unfinished edge in the constraint. */
	for (cp1 = cp; cp1 < cp2; cp1++) {
		e = cp1 -> var;
		if (BITON (edone, e)) continue;

		clist -> rows [row] = ip;
		vp1 = cip -> edge [e];
		vp2 = cip -> edge [e + 1];
		num1 = 0;
		while (vp1 < vp2) {
			j = *vp1++;
			if (BITON (Sset, j)) {
				if (BITON (Sval, j)) {
					++num1;
				}
				continue;
			}
			k = vmap [j];
			if (k < 0) continue;
			*ip++ = k;
		}
		clist -> op [row]	= '=';
		clist -> rhs [row]	= cp1 -> val + 1 - num1;
		++row;
	}

	/* Generate one row per edge in the "NOT in the constraint" list. */
	for (i = 0; i < zedge_rows; i++) {
		clist -> rows [row] = ip;
		e = elist [i];
		vp1 = cip -> edge [e];
		vp2 = cip -> edge [e + 1];
		while (vp1 < vp2) {
			j = *vp1++;
			k = vmap [j];
			if (k < 0) continue;
			*ip++ = k;
		}
		clist -> op [row]	= '<';	/* <= */
		clist -> rhs [row]	= 1;
		++row;
	}

	/* Generate the final row:  Sum of all columns must equal the	*/
	/* number of vertices of S that remain to be discovered.	*/

	clist -> rows [row] = ip;
	for (i = 0; i < numv; i++) {
		*ip++ = i;
	}
	clist -> op [row]	= '=';
	clist -> rhs [row]	= shortfall;
	++row;
	clist -> rows [row] = ip;

	FATAL_ERROR_IF (ip - clist -> rows [0] NE nzsize);

	/* Restore the edge map. */
	for (i = 0; i < zedge_rows; i++) {
		emap [elist [i]] = -1;
	}

	/* Restore the vertex map. */
	for (i = 0; i < numv; i++) {
		vmap [vlist [i]] = -1;
	}
}

/*
 * The CPLEX version of build_lp.
 */

#if CPLEX

	static
	LP_t *
build_lp (

struct ainfo *		aip,
struct clist *		clist,
struct lpmem *		lpmem
)
{
int			i, j, k;
int			nrows;
int			ncols;
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
int *			ip1;
int *			ip2;
int *			tmp;
LP_t *			lp;

	nrows	= clist -> nrows;
	ncols	= clist -> ncols;

	macsz	= ncols;
	mac	= ncols;

	/* Build the objective function... */
	objx = NEWA (macsz, double);
	for (i = 0; i < ncols; i++) {
		objx [i] = clist -> obj [i];
	}
	objsen = _MYCPX_MIN;	/* Minimize */

	/* Build variable bound arrays... */
	bdl = NEWA (macsz, double);
	bdu = NEWA (macsz, double);
	for (i = 0; i < macsz; i++) {
		bdl [i] = 0.0;
		bdu [i] = 1.0;
	}

	/* Allocate enough space to add some additional constraints. */
	marsz	= nrows + 20;
	matsz	= (clist -> rows [nrows] - clist -> rows [0]) + 20 * ncols;

	mar = nrows;

	/* Allocate arrays for constraint matrix... */
	rhsx = NEWA (marsz, double);
	senx = NEWA (marsz, char);
	matbeg = NEWA (macsz, int);
	matcnt = NEWA (macsz, int);
	matind = NEWA (matsz, int);
	matval = NEWA (matsz, double);

	/* Now go through each row k and compute the number of	*/
	/* non-zero coefficients for each variable used...	*/
	tmp = NEWA (macsz, int);
	for (i = 0; i < macsz; i++) {
		tmp [i] = 0;
	}
	ip1 = clist -> rows [0];
	ip2 = clist -> rows [nrows];
	while (ip1 < ip2) {
		j = *ip1++;
		FATAL_ERROR_IF ((j < 0) OR (j >= ncols));
		++(tmp [j]);
	}

	/* CPLEX wants columns, not rows... */
	j = 0;
	for (i = 0; i < ncols; i++) {
		k = tmp [i];
		matbeg [i] = j;
		tmp [i] = j;
		matcnt [i] = k;
		j += k;
	}
	for (i = 0; i < nrows; i++) {
		ip1 = clist -> rows [i];
		ip2 = clist -> rows [i + 1];
		while (ip1 < ip2) {
			j = *ip1++;
			k = tmp [j];
			matind [k] = i;
			matval [k] = 1.0;
			++(tmp [j]);
		}
		switch (clist -> op [i]) {
		case '<':	senx [i] = 'L';		break;
		case '=':	senx [i] = 'E';		break;
		case '>':	senx [i] = 'G';		break;
		default:
			FATAL_ERROR;
		}
		rhsx [i] = clist -> rhs [i];
	}

	/* Verify consistency of what we generated... */
	for (i = 0; i < ncols; i++) {
		FATAL_ERROR_IF (tmp [i] NE matbeg [i] + matcnt [i]);
	}

	free ((char *) tmp);

	lp = _MYCPX_loadlp ("analyz",
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

	return (lp);
}

#endif

/*
 * The CPLEX version of the routine to free up the LP.
 */

#if CPLEX

	static
	void
destroy_lp (

LP_t *			lp,
struct lpmem *		lpmem
)
{
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
}

#endif

/*
 * The lp_solve version of build_lp.
 */

#if LPSOLVE

	static
	LP_t *
build_lp (

struct ainfo *		aip,
struct clist *		clist,
struct lpmem *		lpmem
)
{
int			i, j;
int			nrows;
int			ncols;
int			ncoeff;
int			nzi;
double *		rhs;
short *			ctype;
int *			matbeg;
int *			matind;
double *		matval;
int *			ip1;
int *			ip2;
LP_t *			lp;

	nrows	= clist -> nrows;
	ncols	= clist -> ncols;
	ncoeff	= clist -> rows [nrows] - clist -> rows [0];

	/* Make the initial LP... */
	lp = make_lp (0, ncols);

	/* All variables are 0-1 variables... */
	for (i = 1; i <= ncols; i++) {
		set_bounds (lp, i, 0.0, 1.0);
	}

	/* Minimize */
	set_minim (lp);

	/* Set the objective function... */
#if 1
	inc_mat_space (lp, ncols + 1);
#endif
	set_obj_fn (lp, clist -> obj - 1);

	/* Allocate arrays for setting the rows... */
	rhs	= NEWA (nrows, double);
	ctype	= NEWA (nrows, short);
	matbeg	= NEWA (nrows + 1, int);
	matind	= NEWA (ncoeff, int);
	matval	= NEWA (ncoeff, double);

	/* Put the rows into the format that LP-solve wants them in...	*/
	nzi = 0;
	for (i = 0; i < nrows; i++) {
		matbeg [i] = nzi;
		ip1 = clist -> rows [i];
		ip2 = clist -> rows [i + 1];
		while (ip1 < ip2) {
			j = *ip1++;
			matind [nzi] = j;
			matval [nzi] = 1.0;
			++nzi;
		}
		rhs [i] = clist -> rhs [i];
		switch (clist -> op [i]) {
		case '<':	ctype [i] = REL_LE;	break;
		case '=':	ctype [i] = REL_EQ;	break;
		case '>':	ctype [i] = REL_GE;	break;
		default:
			FATAL_ERROR;
			break;
		}
	}
	matbeg [i] = nzi;
	FATAL_ERROR_IF (nzi NE ncoeff);

	add_rows (lp, 0, nrows, rhs, ctype, matbeg, matind, matval);

	free ((char *) matval);
	free ((char *) matind);
	free ((char *) matbeg);
	free ((char *) ctype);
	free ((char *) rhs);

	return (lp);
}

#endif

/*
 * The lp_solve version of the routine to free up the LP.
 */

#if LPSOLVE

	static
	void
destroy_lp (

LP_t *			lp,
struct lpmem *		lpmem
)
{
	(void) lpmem;

	delete_lp (lp);
}

#endif
