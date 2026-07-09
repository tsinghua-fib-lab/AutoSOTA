/***********************************************************************

	$Id: rmst.h,v 1.2 2022/11/19 13:45:54 warme Exp $

	File:	rmst.h
	Rev:	e-1
	Date:	09/24/2016

	Copyright (c) 1997, 2022 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Rectilinear Minimum Spanning Tree and Kahng-Robins heuristic.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from steiner.h.

************************************************************************/

#ifndef RMST_H
#define	RMST_H

#include "bitmaskmacros.h"
#include "geomtypes.h"

struct edge;
struct pset;

extern int	_gst_kahng_robins (struct pset *	pts,
				   dist_t		limit,
				   struct edge *	edges);
extern dist_t	_gst_kahng_robins_length (struct pset * pts, dist_t limit);
extern int	_gst_rect_mst (struct pset *	pts,
			       struct edge *	edges,
			       bitmap_t *	empty_rect);
extern dist_t	_gst_rect_mst_length (struct pset * pts);

#endif
