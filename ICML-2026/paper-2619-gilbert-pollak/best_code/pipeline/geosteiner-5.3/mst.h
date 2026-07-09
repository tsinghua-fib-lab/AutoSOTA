/***********************************************************************

	$Id: mst.h,v 1.2 2022/11/19 13:45:53 warme Exp $

	File:	mst.h
	Rev:	e-1
	Date:	09/24/2016

	Copyright (c) 1997, 2022 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	Generic Minimum Spanning Tree (Kruskal's Algorithm).

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from steiner.h.

************************************************************************/

#ifndef MST_H_INCLUDED
#define	MST_H_INCLUDED

struct edge;

extern int	_gst_mst_edge_list (int			n,
				    int			nedges,
				    struct edge *	edge_list,
				    struct edge *	edges);
extern void	_gst_sort_edge_list (struct edge * a, int n);

#endif
