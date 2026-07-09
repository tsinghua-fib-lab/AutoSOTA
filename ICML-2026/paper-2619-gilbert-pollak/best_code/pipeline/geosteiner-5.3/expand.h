/***********************************************************************

	$Id: expand.h,v 1.3 2022/11/19 13:45:51 warme Exp $

	File:	expand.h
	Rev:	e-2
	Date:	11/18/2022

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Expand logical constraints into the proper physical encoding
	as used in the LP.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from constrnt.h.
	e-2:	11/18/2022	warme
		: Simplify calling convention.

************************************************************************/

#ifndef EXPAND_H
#define	EXPAND_H


/*
 * Function Prototypes
 */

struct bbinfo;
struct constraint;
struct rcoef;

extern struct rcoef *	_gst_expand_constraint (
					struct constraint *	lcp,
					struct rcoef *		cp,
					struct bbinfo *		bbip);

#endif
