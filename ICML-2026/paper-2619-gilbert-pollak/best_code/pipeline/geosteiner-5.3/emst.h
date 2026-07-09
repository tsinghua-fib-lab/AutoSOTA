/***********************************************************************

	$Id: emst.h,v 1.2 2022/11/19 13:45:51 warme Exp $

	File:	emst.h
	Rev:	e-1
	Date:	09/24/2016

	Copyright (c) 1997, 2022 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Euclidean Minimum Spanning Tree.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from steiner.h.

************************************************************************/

#ifndef EMST_H
#define	EMST_H

#include "geomtypes.h"

struct edge;
struct pset;

extern int	_gst_euclidean_mst (struct pset * pts, struct edge * edges);
extern dist_t	_gst_euclidean_mst_length (struct pset * pts);

#endif
