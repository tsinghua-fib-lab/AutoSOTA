/***********************************************************************

	$Id: bmst.h,v 1.6 2022/11/19 13:45:49 warme Exp $

	File:	bmst.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1997, 2022 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Routines to compute Minimum Spanning Trees using the
	Bottleneck Steiner Distance.

************************************************************************

	Modification Log:

	b-1:	06/27/2003	warme
		: Created header file for bmst stuff.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef BMST_H
#define BMST_H

#include "geomtypes.h"

struct bsd;
struct edge;

extern dist_t	_gst_bmst_terms_length (int * terms, int n, struct bsd * bsdp);
extern int	_gst_bmst_terms (int *		terms,
				 int		n,
				 struct bsd *	bsdp,
				 struct edge *	edges);

#endif
