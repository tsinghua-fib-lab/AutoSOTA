/***********************************************************************

	$Id: ddsuf.h,v 1.8 2022/11/19 13:45:50 warme Exp $

	File:	ddsuf.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1998, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Declarations for the "dynamic disjoint set union-find"
	data structure.  This is identical to the normal DSUF,
	except that it keeps a log of all its modifications so
	that previous states can be restored.

************************************************************************

	Modification Log:

	a-1:	11/08/99	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef	DDSUF_H
#define	DDSUF_H

/*
 * This is the so-called "Dynamic Disjoint Set Union-Find" data structure.
 * The normal DSUF operations are "makeset", "find", and "union".
 * The dynamic operations are "get_state" and "restore_state".
 * Note that a "restore_state" operation CANNOT be reversed.
 *
 * See chapter 2 of "Data Structures and Network Algorithms", by
 * Robert Endre Tarjan, SIAM, 1983 for complete details.
 */


struct ddsuf {
	int *			parent;
	int *			rank;
	int			set_size;

	struct ddsuf_restore *	stack;
	int			sp;
	int			stack_size;
};


/*
 * This structure records a single modification made to the DSUF
 * data structure.  It contains a pointer to the cell modified,
 * and the value that was previously stored there.
 */

struct ddsuf_restore {
	int *		ptr;
	int		val;
};


/*
 * Global Routines
 */

extern void	_gst_ddsuf_create (struct ddsuf * dsp, int n);
extern void	_gst_ddsuf_destroy (struct ddsuf * dsp);
extern int	_gst_ddsuf_find (struct ddsuf * dsp, int i);
extern void	_gst_ddsuf_makeset (struct ddsuf * dsp, int i);
extern void	_gst_ddsuf_restore (struct ddsuf * dsp, int old_state);
extern void	_gst_ddsuf_unite (struct ddsuf * dsp, int i, int j);

#define	_gst_ddsuf_get_state(p)	((p) -> sp)

#endif
