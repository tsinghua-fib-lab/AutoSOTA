/***********************************************************************

	$Id: io.h,v 1.6 2022/11/19 13:45:52 warme Exp $

	File:	io.h
	Rev:	e-1
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	General declarations for the Steiner Tree program.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from steiner.h.

************************************************************************/

#ifndef	IO_H_INCLUDED
#define	IO_H_INCLUDED

#include "geomtypes.h"
#include <stdio.h>


/*
 * A linked list structure, each node of which holds a single real
 * number in special "scaled" form.  We use these during input so that we
 * can examine an entire set of numbers before we commit to an internal
 * representation for them.
 */

struct numlist {
	double		mantissa;       /* number scaled to an integer */
					/* value */
	int		expon;		/* val = mantissa * 10^expon */
	int		nsig;		/* # signif digits in mant */
	struct numlist * next;		/* next in linked list */
};


/*
 * A structure that describes how external coordinates, lengths, etc.
 * are scaled internally.  We do this to eliminate some of the bad
 * behavior of floating point representation.
 *
 * The internal values are actually 10**scale times BIGGER than the
 * external values read in.  Since scale can be negative, we have two
 * distinct conversion factors -- one to multiply by, and one to divide
 * by.
 */

struct gst_scale_info {
	int		scale;		/* internal vals mpy'd by 10**scale */
	int		min_precision;	/* min decimal places to output */
	double		scale_mul;	/* 1 or 10**-scale, if scale < 0 */
	double		scale_div;	/* 1 or 10**scale, if scale >= 0 */
};


/*
 * Macro to unscale an internal value back to external form.
 */

#define UNSCALE(val,p)  ((val) * (p) -> scale_mul / (p) -> scale_div)

/*
 * Function Prototypes.
 */

extern int	_gst_compute_scaling_factor (struct numlist *	list);
extern void	_gst_coord_to_string (char *			buf,
				      coord_t			coord,
				      struct gst_scale_info *	sip);
extern void	_gst_dist_to_string (char *			buf,
				     dist_t			dist,
				     struct gst_scale_info *	sip);
extern struct numlist *
		_gst_parse_line_of_numbers (FILE * fp);
extern coord_t	_gst_read_numlist (struct numlist *		nlp,
				   struct gst_scale_info *	sip);
extern void	_gst_set_scale_info (struct gst_scale_info *	sip,
				     int			scale_factor);

#endif
