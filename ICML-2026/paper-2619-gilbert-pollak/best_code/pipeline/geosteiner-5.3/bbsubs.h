/***********************************************************************

	$Id: bbsubs.h,v 1.9 2022/11/19 13:45:49 warme Exp $

	File:	bbsubs.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1998, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Declarations of low-level branch-and-cut routines.

************************************************************************

	Modification Log:

	a-1:	02/28/2001	warme
		: Split off from bb.h.  Changes for 3.1 release.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef	BBSUBS_H
#define	BBSUBS_H

struct bbnode;
struct bbtree;
struct bbinfo;

extern void		_gst_add_bbnode (struct bbinfo *	bbip,
					 int			var,
					 int			dir,
					 double			z);
extern void		_gst_append_node_to_tree (struct bbnode *	p,
						  struct bbtree *	tp);
extern void		_gst_bbheap_insert (struct bbnode *	p,
					    struct bbtree *	tp,
					    int			heap_no);
extern struct bbtree *	_gst_create_bbtree (int nmasks);
extern void		_gst_delete_node_from_bbtree (struct bbnode *	p,
						      struct bbtree *	tp);
extern void		_gst_destroy_bbinfo (struct bbinfo * bbip);

#endif
