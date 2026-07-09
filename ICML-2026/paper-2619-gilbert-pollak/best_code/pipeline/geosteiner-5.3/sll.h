/***********************************************************************

	$Id: sll.h,v 1.2 2022/11/19 13:45:54 warme Exp $

	File:	sll.h
	Rev:	e-1
	Date:	09/24/2016

	Copyright (c) 1997, 2022 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	The Smith-Lee-Liebman heuristic for Euclidean Steiner tree.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from efst.h.

************************************************************************/

#ifndef SLL_H_INCLUDED
#define	SLL_H_INCLUDED

#include "geomtypes.h"

struct bsd;
struct pset;

extern dist_t		_gst_smith_lee_liebman (struct pset *	pts,
						int *		termidx,
						struct bsd *	bsdp);

#endif
