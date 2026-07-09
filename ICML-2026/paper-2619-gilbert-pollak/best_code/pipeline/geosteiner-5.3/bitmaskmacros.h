/***********************************************************************

	$Id: bitmaskmacros.h,v 1.2 2022/11/19 13:45:49 warme Exp $

	File:	bitmaskmacros.h
	Rev:	e-1
	Date:	09/24/2016

	Copyright (c) 1996, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Bit masks implemented as arrays of unsigned integers.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from general.h.

************************************************************************/

#ifndef BITMASKMACROS_H
#define	BITMASKMACROS_H

#include "gsttypes.h"

/*
 * Bit Maps
 */

typedef int32u		bitmap_t;	/* Element of Bit-map vector. */
#define	BPW		32		/* Bits per word of a bit-map. */
#define	BMAP_ELTS(n)	(((n) + (BPW-1)) / BPW)

/*
 * Useful macros.
 */

extern const int8u _gst_nbits [];	/* A pre-initialized lookup table. */

#define	SETBIT(bm, n)	((bm) [(n) / BPW] |= (1ul << ((n) % BPW)))
#define	CLRBIT(bm, n)	((bm) [(n) / BPW] &= ~(1ul << ((n) % BPW)))
#define	BITON(bm, n)	(((bm) [(n) / BPW] & (1ul << ((n) % BPW))) NE 0)
#define	NBITSON(m)	(  _gst_nbits [ (m)        & 0xFFlu]	\
			 + _gst_nbits [((m) >>  8) & 0xFFlu]	\
			 + _gst_nbits [((m) >> 16) & 0xFFlu]	\
			 + _gst_nbits [((m) >> 24) & 0xFFlu])

#endif
