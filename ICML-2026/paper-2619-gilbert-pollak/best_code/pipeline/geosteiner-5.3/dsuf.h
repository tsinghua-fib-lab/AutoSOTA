/***********************************************************************

	$Id: dsuf.h,v 1.8 2022/11/19 13:45:51 warme Exp $

	File:	dsuf.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1998, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Declarations for the "disjoint set union-find" data structure.

************************************************************************

	Modification Log:

	a-1:	11/03/98	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef	DSUF_H
#define	DSUF_H

/*
 * This is the so-called "Disjoint Set Union-Find" data structure.
 * The operations are "makeset", "find", and "union".
 *
 * See chapter 2 of "Data Structures and Network Algorithms", by
 * Robert Endre Tarjan, SIAM, 1983 for complete details.
 */


struct dsuf {
	int *	parent;
	int *	rank;
	int	set_size;
};


/*
 * Global Routines
 */

extern void		_gst_dsuf_create (struct dsuf * dsp, int n);
extern void		_gst_dsuf_destroy (struct dsuf * dsp);
extern int		_gst_dsuf_find (struct dsuf * dsp, int i);
extern void		_gst_dsuf_makeset (struct dsuf * dsp, int i);
extern void		_gst_dsuf_unite (struct dsuf * dsp, int i, int j);

#endif
