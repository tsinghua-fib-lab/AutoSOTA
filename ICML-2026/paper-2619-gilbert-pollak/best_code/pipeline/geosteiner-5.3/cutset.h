/***********************************************************************

	$Id: cutset.h,v 1.12 2022/11/19 13:45:50 warme Exp $

	File:	cutset.h
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Data structures for separating cutset constraints.

************************************************************************

	Modification Log:

	b-1:	11/14/96	warme
		: Created.
	b-2:	02/28/2001	warme
		: Changes for 3.1 release.
	c-1:	08/05/2002	benny
		: Some changes for library release.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
	e-4:	11/18/2022	warme
		: Rename one function.
		: Simplify calling convention of various functions.

************************************************************************/

#ifndef CUTSET_H
#define	CUTSET_H

#include "bitmaskmacros.h"
#include "flow.h"

struct bbinfo;
struct constraint;

/*
 * The following data structure defines the flow graph that we use
 * to separate cutset constraints that are fractionally violated.
 */

struct cs_info {

	/* Data used by the flow solver... */
	struct flow_prob	prob;	/* The network flow formulation */
	struct flow_soln	soln;	/* The network flow solution */
	struct flow_temp	temp;	/* Temporary data structures */

	/* Data used to set the arc capacities and modify the	*/
	/* flow network during cutset separation. */
	int *		arc_to_fset;	/* arc # -> full set #. */
};


extern struct constraint * _gst_add_cutset_to_list (
					bitmap_t *		verts,
					struct constraint *	clist,
					double *		x,
					struct bbinfo *		bbip);
extern void		_gst_build_cutset_separation_formulation (
						bitmap_t *	vert_mask,
						bitmap_t *	edge_mask,
						struct bbinfo *	bbip);
extern struct constraint * _gst_find_fractional_cutsets (
						double *	x,
						struct bbinfo *	bbip);
extern struct constraint * _gst_find_zero_weight_cutsets (
						double *	x,
						struct bbinfo *	bbip);
extern void		_gst_free_cutset_separation_formulation (
						struct cs_info * csip);

#endif
