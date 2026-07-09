/***********************************************************************

	$Id: fstfuncs.h,v 1.10 2022/11/19 13:45:52 warme Exp $

	File:	fstfuncs.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	General functions used by the fst generators.

************************************************************************

	Modification Log:

	a-1:	06/17/02	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef	FSTFUNCS_H
#define	FSTFUNCS_H

struct full_set;
struct gst_hypergraph;
struct pset;

extern struct pset *	_gst_create_pset (int, double *);
extern int		_gst_generate_duplicate_terminal_groups (struct pset *,
						     int *,
						     int ***);
extern void		_gst_initialize_hypergraph (struct gst_hypergraph *);

extern struct full_set ** _gst_put_trees_in_array (struct full_set *, int *);
extern struct pset *	_gst_remove_duplicates (struct pset * pts,
					   int		 ndg,
					   int **	 list,
					   int **	 fwd_map,
					   int **	 rev_map);

#endif
