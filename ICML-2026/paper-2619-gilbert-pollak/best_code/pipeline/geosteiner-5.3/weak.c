/***********************************************************************

	$Id: weak.c,v 1.13 2022/11/19 13:45:55 warme Exp $

	File:	weak.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	A subtour separator that finds weak connectivity.

************************************************************************

	Modification Log:

	a-1:	07/05/2000	warme
		: Created.
	c-1:	08/05/2002	benny
		: Some changes for library release.
		: Uses parameters.
		: Uses channels for trace output.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes,
		:  upgrade fatals.

************************************************************************/

#include "weak.h"

#include "bb.h"
#include "constrnt.h"
#include "fatal.h"
#include "geosteiner.h"
#include "logic.h"
#include "memory.h"
#include "parmblk.h"
#include "sec_heur.h"
#include "steiner.h"


/*
 * Global Routines
 */

struct constraint * _gst_find_weak_connectivity (
					double *		x,
					struct constraint *	cp,
					struct bbinfo *		bbip);


/*
 * External References
 */

	/* None */

/*
 * Local Types
 */

struct bc2 {
	struct bbinfo *	bbip;		/* branch and bound info */
	double *	x;		/* LP solution to separate */
	bitmap_t *	vert_mask;	/* valid problem vertices */
	bitmap_t *	edge_mask;	/* valid problem edges */
	int		kmasks;		/* size of vert_mask */
	int		nmasks;		/* size of edge_mask */
	int *		dfs;		/* DFS number of each vertex */
	int *		low;		/* lowest DFS num in component */
	int *		parent;		/* parents of vertices in DFS tree */
	int		max_stack;	/* size of stack */
	int *		stack;		/* base-address of edge stack */
	int *		sp;		/* current stack pointer */
	int		counter;	/* DFS number generator */
	bitmap_t *	cc_vmask;	/* vertices in connected component */
	bitmap_t *	cc_emask;	/* edges in connected component */
	int *		cc_vlist;	/* vertices in connected component */
	int *		cc_vlp;		/* end of cc_vlist */
	int *		cc_elist;	/* edges in connected component */
	int *		cc_elp;		/* end of cc_elist */
	bitmap_t *	bcc_vmask;	/* scratch buffer for new BCCs */
	bitmap_t *	bcc_emask;	/* scratch buffer for new BCCs */
	bitmap_t *	edges_seen;	/* edges already pushed */
	int *		bcc_vlist;	/* scratch vertex list buffer */
	int *		degree;		/* temp vertex degree counter */
	double *	weight;		/* temp vertex weight */
	struct cslist *	cslist;		/* list of cutset edge sets */
	bitmap_t *	cut_mask;	/* scratch buffer for cuts */
	bitmap_t *	cut_vseen;	/* mark flags for cuts */
};

struct cslist {
	int *		elist;		/* list of cut edges */
	int		count;		/* number of edges in list */
	struct cslist *	next;		/* next node in linked list */
};


/*
 * Local Routines
 */

static void		bcc2 (struct bc2 *, int);
static struct constraint * check_unique_cut (char *,
					     bitmap_t *,
					     int,
					     int,
					     struct constraint *,
					     struct bbinfo *,
					     double);
static struct constraint * identify_cuts (struct bc2 *,
					  struct constraint *,
					  int *,
					  int);
static bool		int_lists_match (int *, int *, int);
static void		process_bcc2 (struct bc2 *, int *, int *);

/*
 * Examine the bi-connected components of the solution, hoping to find
 * violated cuts.  If any vertex in a BCC has a congestion level less
 * than 1 (measured using only edges within the BCC), then this represents
 * a violated cutset -- for which we can generate two SEC's.
 */

	struct constraint *
_gst_find_weak_connectivity (

double *		x,		/* IN - LP solution to check */
struct constraint *	cp,		/* IN - previous constraints */
struct bbinfo *		bbip		/* IN - branch and bound info */
)
{
int			i;
int			j;
int			nverts;
int			nedges;
int			kmasks;
int			nmasks;
int			num_vactive;
struct gst_hypergraph *	cip;
bitmap_t *		vert_mask;
bitmap_t *		edge_mask;
struct cslist *		tmp;
int *			ip1;
int *			ip2;
gst_channel_ptr		trace;
struct constraint *	cp2;
struct constraint *	cp3;
struct bc2		bc;

	cip	  = bbip -> cip;
	vert_mask = bbip -> vert_mask;
	edge_mask = bbip -> edge_mask;
	trace	  = bbip -> params -> print_solve_trace;

	nverts	= cip -> num_verts;
	nedges	= cip -> num_edges;

	if (nverts < 2) {
		/* No BCC's. */
		return (cp);
	}

	kmasks	= BMAP_ELTS (nverts);
	nmasks	= BMAP_ELTS (nedges);

	bc.bbip		= bbip;
	bc.x		= x;
	bc.vert_mask	= vert_mask;
	bc.edge_mask	= edge_mask;
	bc.kmasks	= kmasks;
	bc.nmasks	= nmasks;
	bc.dfs		= NEWA (nverts, int);
	bc.low		= NEWA (nverts, int);
	bc.parent	= NEWA (nverts, int);

	for (i = 0; i < nverts; i++) {
		bc.dfs [i] = 0;
		bc.low [i] = 0;
		bc.parent [i] = -1;
	}

	j = (nedges > nverts) ? nedges : nverts;
	bc.max_stack	= j;
	bc.stack	= NEWA (j, int);
	bc.sp		= bc.stack;
	bc.counter	= 0;

	bc.cc_vmask	= NEWA (kmasks, bitmap_t);
	bc.cc_emask	= NEWA (nmasks, bitmap_t);
	bc.cc_vlist	= NEWA (nverts, int);
	bc.cc_vlp	= bc.cc_vlist;
	bc.cc_elist	= NEWA (nedges, int);
	bc.cc_elp	= bc.cc_elist;
	bc.bcc_vmask	= NEWA (kmasks, bitmap_t);
	bc.bcc_emask	= NEWA (nmasks, bitmap_t);
	bc.edges_seen	= NEWA (nmasks, bitmap_t);
	bc.bcc_vlist	= NEWA (nverts, int);
	bc.degree	= NEWA (nverts, int);
	bc.weight	= NEWA (nverts, double);
	bc.cslist	= NULL;

	bc.cut_mask	= NEWA (kmasks, bitmap_t);
	bc.cut_vseen	= NEWA (kmasks, bitmap_t);

	for (i = 0; i < bc.kmasks; i++) {
		bc.cc_vmask [i] = 0;
		bc.bcc_vmask [i] = 0;
		bc.cut_mask [i] = 0;
		bc.cut_vseen [i] = 0;
	}
	for (i = 0; i < bc.nmasks; i++) {
		bc.cc_emask [i] = 0;
		bc.bcc_emask [i] = 0;
		bc.edges_seen [i] = 0;
	}

	num_vactive = 0;
	for (i = 0; i < nverts; i++) {
		if (NOT BITON (vert_mask, i)) continue;
		++num_vactive;
	}

	cp2 = NULL;

	/* Traverse each connected component, identifying its BCC's as	*/
	/* we go.							*/
	for (i = 0; i < nverts; i++) {
		if (NOT BITON (vert_mask, i)) continue;
		if (bc.dfs [i] > 0) continue;

		/* Prepare to traverse one connected component, finding	*/
		/* each of its BCCs as we go.				*/
		bc.cc_vlp = bc.cc_vlist;
		bc.cc_elp = bc.cc_elist;

		bcc2 (&bc, i);

		for (;;) {
			tmp = bc.cslist;
			if (tmp EQ NULL) break;
			bc.cslist = tmp -> next;

			/* tmp -> elist/count is a list of edges of	*/
			/* total weight < 1 whose removal disconnects	*/
			/* this connected component.  Identify the	*/
			/* subtour violation(s) implied by this cut.	*/

			cp2 = identify_cuts (&bc,
					     cp2,
					     tmp -> elist,
					     tmp -> count);

#if 0
			if (trace NE NULL) {
				gst_channel_printf (trace, "Weak cut edges:");
				for (j = 0; j < tmp -> count; j++) {
					gst_channel_printf (trace,
						" %d", tmp -> elist [j]);
				}
				gst_channel_printf (trace, "\n");
			}
#endif
			free ((char *) (tmp -> elist));
			free ((char *) tmp);
		}
		ip1 = bc.cc_vlist;
		ip2 = bc.cc_vlp;
		while (ip1 < ip2) {
			j = *ip1++;
			CLRBIT (bc.cc_vmask, j);
		}
		ip1 = bc.cc_elist;
		ip2 = bc.cc_elp;
		while (ip1 < ip2) {
			j = *ip1++;
			CLRBIT (bc.cc_emask, j);
		}
	}

	i = 0;
	while (cp2 NE NULL) {
		cp3 = cp2 -> next;
		cp2 -> next = cp;
		cp = cp2;
		cp2 = cp3;
		++i;
	}

#if 1
	if (i > 0) {
		gst_channel_printf (trace,
			"_gst_find_weak_connectivity found %d subtours.\n", i);
	}
#endif

	free ((char *) bc.cut_vseen);
	free ((char *) bc.cut_mask);
	free ((char *) bc.weight);
	free ((char *) bc.degree);
	free ((char *) bc.bcc_vlist);
	free ((char *) bc.edges_seen);
	free ((char *) bc.bcc_emask);
	free ((char *) bc.bcc_vmask);
	free ((char *) bc.cc_elist);
	free ((char *) bc.cc_vlist);
	free ((char *) bc.cc_emask);
	free ((char *) bc.cc_vmask);
	free ((char *) bc.stack);
	free ((char *) bc.parent);
	free ((char *) bc.low);
	free ((char *) bc.dfs);

	return (cp);
}

/*
 * This is the recursive part of the bi-connected-components algorithm.  It
 * is the standard method, with a few tweaks to work on hypergraphs instead.
 * We process each bi-connected component individually.
 */

	static
	void
bcc2 (

struct bc2 *		bcp,		/* IN - global BCC data */
int			v		/* IN - current DFS vertex */
)
{
int			e;
int			e2;
int			w;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			stack_endp;
int *			sp;
int *			stack;
struct bbinfo *		bbip;
struct gst_hypergraph *	cip;

	bbip = bcp -> bbip;
	cip = bbip -> cip;

	FATAL_ERROR_IF ((v < 0) OR (v >= cip -> num_verts));

	*(bcp -> cc_vlp)++ = v;
	SETBIT (bcp -> cc_vmask, v);

	++(bcp -> counter);
	bcp -> dfs [v] = bcp -> counter;
	bcp -> low [v] = bcp -> counter;
	ep1 = cip -> term_trees [v];
	ep2 = cip -> term_trees [v + 1];
	while (ep1 < ep2) {
		e = *ep1++;
		FATAL_ERROR_IF ((e < 0) OR (e >= cip -> num_edges));
		if (bcp -> x [e] <= FUZZ) continue;
		if (NOT BITON (bcp -> edges_seen, e)) {
			/* We haven't seen this edge before.  Push	*/
			/* it onto the stack...				*/
			stack_endp = &(bcp -> stack [bcp -> max_stack]);
			FATAL_ERROR_IF ((bcp -> sp < bcp -> stack) OR
					(bcp -> sp >= stack_endp));
			*(bcp -> sp)++ = e;
			SETBIT (bcp -> edges_seen, e);

			*(bcp -> cc_elp)++ = e;
			SETBIT (bcp -> cc_emask, e);
		}
		/* Scan the vertices and process them... */
		vp1 = cip -> edge [e];
		vp2 = cip -> edge [e + 1];
		while (vp1 < vp2) {
			w = *vp1++;
			FATAL_ERROR_IF ((w < 0) OR (w >= cip -> num_verts));
			if (bcp -> dfs [w] EQ 0) {
				bcp -> parent [w] = v;
				bcc2 (bcp, w);
				if (bcp -> low [w] >= bcp -> dfs [v]) {
					/* We have a new BCC! */
					stack	= bcp -> stack;
					sp	= bcp -> sp;
					do {
						FATAL_ERROR_IF (sp <= stack);
						e2 = *--sp;
					} while (e2 NE e);

					/* Process the bi-connected comp. */
					process_bcc2 (bcp, sp, bcp -> sp);

					/* Pop BCC edges from stack */
					bcp -> sp = sp;
				}
				if (bcp -> low [w] < bcp -> low [v]) {
					bcp -> low [v] = bcp -> low [w];
				}
			}
			else if ((w NE bcp -> parent [v]) AND
				 (bcp -> dfs [w] < bcp -> low [v])) {
				bcp -> low [v] = bcp -> dfs [w];
			}
		}
	}
}

/*
 * Process a single bi-connected component, specified as a list of edges.
 * We look for vertices that are too weakly connected to the component.
 * Such vertices are on the opposite side of a violated cutset.
 */

	static
	void
process_bcc2 (

struct bc2 *	bcp,		/* IN - global BCC data */
int *		edge_ptr,	/* IN - list of BCC edges */
int *		endp		/* IN - end of BCC edge list */
)
{
int			e;
int			v;
int			n;
int *			ep1;
int *			ep2;
int *			ep3;
int *			vp1;
int *			vp2;
int *			vlp;
int *			elist;
struct cslist *		cslp;
struct bbinfo *		bbip;
struct gst_hypergraph *	cip;
double			xe;

	bbip = bcp -> bbip;
	cip = bbip -> cip;

	/* Gather a list of all vertices in this BCC.  Compute their	*/
	/* weights and degrees (with respect to the BCC).		*/
	vlp = bcp -> bcc_vlist;

	ep1 = edge_ptr;
	ep2 = endp;
	while (ep1 < ep2) {
		e = *ep1++;
		xe = bcp -> x [e];
		SETBIT (bcp -> bcc_emask, e);
		vp1 = cip -> edge [e];
		vp2 = cip -> edge [e + 1];
		while (vp1 < vp2) {
			v = *vp1++;
			if (NOT BITON (bcp -> vert_mask, v)) continue;
			if (NOT BITON (bcp -> bcc_vmask, v)) {
				*vlp++ = v;
				SETBIT (bcp -> bcc_vmask, v);
				bcp -> degree [v] = 0;
				bcp -> weight [v] = 0.0;
			}
			++(bcp -> degree [v]);
			bcp -> weight [v] += xe;
		}
	}

	/* All of the vertices of this BCC are now known, as are their	*/
	/* degrees and weights (relative to the component).		*/
	/* Time to look for any vertices that are too weakly connected.	*/

	vp1 = bcp -> bcc_vlist;
	vp2 = vlp;
	while (vp1 < vp2) {
		v = *vp1++;
		CLRBIT (bcp -> bcc_vmask, v);		/* clean up as we go */
		if (bcp -> weight [v] >= 1.0 - FUZZ) continue;

		/* Vertex v is too weakly connected to the BCC!	*/
		/* This represents at least 1 subtour violation! */

		n = bcp -> degree [v];
		elist = NEWA (n, int);
		ep1 = cip -> term_trees [v];
		ep2 = cip -> term_trees [v + 1];
		ep3 = elist;
		while (ep1 < ep2) {
			e = *ep1++;
			if (NOT BITON (bcp -> bcc_emask, e)) continue;
			*ep3++ = e;
		}
		FATAL_ERROR_IF (elist + n NE ep3);
		for (cslp = bcp -> cslist; cslp NE NULL; cslp = cslp -> next) {
			if (cslp -> count NE n) continue;
			if (int_lists_match (cslp -> elist, elist, n)) break;
		}
		if (cslp NE NULL) {
			/* This set of cut edges previously located. */
			free ((char *) elist);
		}
		else {
			/* A new set of cut edges.  Save it off. */
			cslp = NEW (struct cslist);
			cslp -> elist	= elist;
			cslp -> count	= n;
			cslp -> next	= bcp -> cslist;
			bcp -> cslist = cslp;
		}
	}

	/* Clean up edge mark flags. */
	ep1 = edge_ptr;
	ep2 = endp;
	while (ep1 < ep2) {
		e = *ep1++;
		CLRBIT (bcp -> bcc_emask, e);
	}
}

/*
 * See if two integer arrays match.
 */

	static
	bool
int_lists_match (

int *		p1,		/* IN - first int array */
int *		p2,		/* IN - second int array */
int		n		/* IN - length of each array */
)
{
	for (; n > 0; n--, p1++, p2++) {
		if (*p1 NE *p2) return (FALSE);
	}
	return (TRUE);
}

/*
 * Process a set of edges having the following properties:
 *
 *	1. Their total LP weight is less than 1.
 *	2. Removal of these edges disconnects the current
 *	   connected component.
 *
 * This implies there are subtour violations.  We identify various
 * subtours, at least one of which must be violated.  The methodology
 * we use here is simple, but usually generates some duplicates.
 * Therefore we take the time to check for and remove duplicates.
 */

	static
	struct constraint *
identify_cuts (

struct bc2 *		bcp,	/* IN - BCC/CC info */
struct constraint *	cp,	/* IN - existing constraints */
int *			elist,	/* IN - list of disconnecting edges */
int			count	/* IN - number of disconnecting edges */
)
{
int			i;
int			e;
int			v;
int			u;
int			kmasks;
int			ncuts;
int			nverts;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			vp3;
int *			vp4;
int *			sp;
struct bbinfo *		bbip;
struct gst_hypergraph *	cip;
bitmap_t *		cut_mask;
double			sum;

	bbip		= bcp -> bbip;
	kmasks		= bcp -> kmasks;
	cut_mask	= bcp -> cut_mask;
	cip		= bbip -> cip;
	nverts		= cip -> num_verts;

	/* Remove the edges from the current connected component. */
	sum = 0.0;
	ep1 = elist;
	ep2 = ep1 + count;
	while (ep1 < ep2) {
		e = *ep1++;
		FATAL_ERROR_IF (NOT BITON (bcp -> cc_emask, e));
		CLRBIT (bcp -> cc_emask, e);
		sum += bcp -> x [e];
	}
	FATAL_ERROR_IF (sum > 1.0 - FUZZ);

	ncuts = 0;

	/* Re-scan the (no longer) connected component to identify the	*/
	/* connected components generated by removal of these edges.	*/
	vp1 = bcp -> cc_vlist;
	vp2 = bcp -> cc_vlp;
	while (vp1 < vp2) {
		v = *vp1++;
		if (BITON (bcp -> cut_vseen, v)) continue;
		/* v is the first vertex of a new cut (connected component). */

		for (i = 0; i < kmasks; i++) {
			cut_mask [i] = 0;
		}

		sp = bcp -> stack;
		*sp++ = v;
		SETBIT (bcp -> cut_vseen, v);
		while (sp > bcp -> stack) {
			v = *--sp;
			SETBIT (cut_mask, v);
			ep1 = cip -> term_trees [v];
			ep2 = cip -> term_trees [v + 1];
			while (ep1 < ep2) {
				e = *ep1++;
				if (NOT BITON (bcp -> cc_emask, e)) continue;
				vp3 = cip -> edge [e];
				vp4 = cip -> edge [e + 1];
				while (vp3 < vp4) {
					u = *vp3++;
					if (BITON (bcp -> cut_vseen, u)) continue;
					*sp++ = u;
					SETBIT (bcp -> cut_vseen, u);
				}
			}
		}

		/* We have identified one cut.  Process it. */
		++ncuts;

		/* The cut S. */
		cp = check_unique_cut ("S", cut_mask, nverts, kmasks, cp, bcp -> bbip, sum);

		/* The cut V - S. */
		for (i = 0; i < kmasks; i++) {
			cut_mask [i] ^= bcp -> vert_mask [i];
		}
		cp = check_unique_cut ("V-S", cut_mask, nverts, kmasks, cp, bcp -> bbip, sum);

		/* The cut V - (CC - S). */
		for (i = 0; i < kmasks; i++) {
			cut_mask [i] ^= bcp -> cc_vmask [i];
		}
		cp = check_unique_cut ("V-(CC-S)", cut_mask, nverts, kmasks, cp, bcp -> bbip, sum);

		/* The cut CC - S. */
		for (i = 0; i < kmasks; i++) {
			cut_mask [i] ^= bcp -> vert_mask [i];
		}
		cp = check_unique_cut ("CC-S", cut_mask, nverts, kmasks, cp, bcp -> bbip, sum);
	}

	FATAL_ERROR_IF (ncuts < 2);

	/* Reset cut_vseen. */
	vp1 = bcp -> cc_vlist;
	vp2 = bcp -> cc_vlp;
	while (vp1 < vp2) {
		v = *vp1++;
		CLRBIT (bcp -> cut_vseen, v);
	}

	/* Put the edges back into the connected component. */
	ep1 = elist;
	ep2 = ep1 + count;
	while (ep1 < ep2) {
		e = *ep1++;
		SETBIT (bcp -> cc_emask, e);
	}

	return (cp);
}

/*
 * This routine checks the given subtour to see if it is unique.  If so
 * a new constraint is added to the given list.  Otherwise, the given
 * constraint list is returned unchanged.
 */

	static
	struct constraint *
check_unique_cut (

char *			msg,		/* IN - debug message */
bitmap_t *		stour,		/* IN - subtour to check */
int			nverts,		/* IN - number of vertices */
int			kmasks,		/* IN - number of vertex masks */
struct constraint *	cp,		/* IN - current constraint list */
struct bbinfo *		bbip,		/* IN - branch and bound info */
double			weight		/* IN - weight of the cut */
)
{
int			i;
int			n;
struct constraint *	p;
bitmap_t *		bp1;
bitmap_t *		bp2;
bitmap_t *		bp3;
bitmap_t		mask;

	n = 0;
	for (i = 0; i < kmasks; i++) {
		mask = stour [i];
		n += NBITSON (mask);
	}

	if ((n > 25) AND (n < (nverts - 25))) {
		return (cp);
	}

	for (p = cp; p NE NULL; p = p -> next) {
		if (_gst_is_equal (stour, p -> mask, kmasks)) {
			/* Already seen this subtour... */
			return (cp);
		}
	}

#if 0
	gst_channel_printf (bbip -> params -> print_solve_trace,
		"\tWeak cutset, %s = %d vertices, x = %g\n",
		msg, n, weight);
#endif

	/* Not seen before -- record it. */
	p = NEW (struct constraint);
	bp1 = NEWA (kmasks, bitmap_t);

	p -> next	= cp;
	p -> iteration	= 0;
	p -> type	= CT_SUBTOUR;
	p -> mask	= bp1;
	bp2 = stour;
	bp3 = bp2 + kmasks;
	while (bp2 < bp3) {
		*bp1++ = *bp2++;
	}
	return (p);
}
