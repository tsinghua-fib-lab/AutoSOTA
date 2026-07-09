/***********************************************************************

	$Id: genps.h,v 1.14 2022/11/19 13:45:52 warme Exp $

	File:	genps.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Stuff for generating plots in postscript.

************************************************************************

	Modification Log:

	a-1:	11/05/98	warme
		: Created.  Split off from steiner.h.
	b-1:	02/28/2001	warme
		: Pass scale_info explicitly to these routines.
		: Added plot_subtour.
	c-1:	08/05/2002	benny
		: Some prototype changes for library release.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef	GENPS_H
#define	GENPS_H

#include "bitmaskmacros.h"

struct gst_channel;
struct gst_hypergraph;
struct gst_scale_info;

/*
 * Enumerated type to identify a plot size.  Big is a single plot per
 * page.  Small is 12 plots per page.
 */

enum plot_size {
	BIG_PLOT,
	SMALL_PLOT
};


/*
 * Function Prototypes.
 */

extern void	_gst_begin_plot (struct gst_channel *	chan,
				 enum plot_size		size);
extern void	_gst_end_plot (struct gst_channel *	chan,
			       const char *		title);
extern void	_gst_define_Plot_Terminals (
				struct gst_channel *chan,
				int			nterms,
				const double *		terms,
				struct gst_scale_info *	sip);
extern void	_gst_overlay_plot_subset (
				struct gst_channel *	chan,
				struct gst_hypergraph *	H,
				char *			title,
				int			nhedges,
				int *			hedges,
				enum plot_size		plot_size);
extern void	_gst_plot_full_sets (
				struct gst_channel *	chan,
				struct gst_hypergraph *	H,
				int			nhedges,
				int *			hedges,
				enum plot_size		plot_size);
extern void	_gst_plot_full_sets_grouped (
				struct gst_channel *	chan,
				struct gst_hypergraph *	H,
				int			nhedges,
				int *			hedges,
				enum plot_size		plot_size);
extern void	_gst_plot_lp_solution (
				struct gst_channel *	chan,
				struct gst_hypergraph *	H,
				const char *		title,
				const double *		weights,
				enum plot_size		plot_size);
extern void	_gst_plot_subtour (
				struct gst_channel *	chan,
				const char *		title,
				const double *		weights,
				struct gst_hypergraph *	H,
				bitmap_t *		S,
				enum plot_size		plot_size);

#endif
