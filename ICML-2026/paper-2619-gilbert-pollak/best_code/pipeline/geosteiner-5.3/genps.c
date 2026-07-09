/***********************************************************************

	$Id: genps.c,v 1.31 2022/11/19 13:45:52 warme Exp $

	File:	genps.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines for outputting stuff.

************************************************************************

	Modification Log:

	a-1:	04/18/93	warme
		: Created.  Moved most output stuff into this file
		:  from other places.
	b-1:	02/28/2001	warme
		: Changes for 3.1 release.  Pass scaling info
		:  explicitly to these routines.  Added plot_subtour.
		:  Always define X and Y scales to be equal.
		:  Fix format in non-geometric case of plot_lp_solution.
	c-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Only uses library functions.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.
	e-4:	11/18/2022	warme
		: Fix -Wall issue.

************************************************************************/

#include "genps.h"

#include "geosteiner.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include <stdlib.h>

/*
 * Global Routines
 */

void		_gst_begin_plot (gst_channel_ptr	chan,
				enum plot_size		size);
void		_gst_define_Plot_Terminals (
				gst_channel_ptr		chan,
				int			nterms,
				const double *		terms,
				gst_scale_info_ptr	sip);
void		_gst_end_plot (gst_channel_ptr		chan,
			       const char *		title);
void		_gst_overlay_plot_subset (
				gst_channel_ptr		chan,
				gst_hg_ptr		H,
				char *			title,
				int			nhedges,
				int *			hedges,
				enum plot_size		plot_size);
void		_gst_plot_full_sets (gst_channel_ptr	chan,
				     gst_hg_ptr		H,
				     int		nhedges,
				     int *		hedges,
				     enum plot_size	plot_size);
void		_gst_plot_full_sets_grouped (gst_channel_ptr	chan,
					     gst_hg_ptr		H,
					     int		nhedges,
					     int *		hedges,
					     enum plot_size	plot_size);
void		_gst_plot_lp_solution (gst_channel_ptr	chan,
				       gst_hg_ptr	H,
				       const char *	title,
				       const double *	weights,
				       enum plot_size	plot_size);
void		_gst_plot_subtour (gst_channel_ptr	chan,
				   const char *		title,
				   const double *	weights,
				   gst_hg_ptr		H,
				   bitmap_t *		S,
				   enum plot_size	plot_size);

/*
 * External References
 */

	/* none */


/*
 * Local Equates
 */

#define	SMALL_PLOTS_PER_PAGE	12

/*
 * Local Routines
 */

static void		announce_new_page (gst_channel_ptr);
static void		define_coordinate_axes (gst_channel_ptr,
						int,
						const double *,
						gst_scale_info_ptr);
static void		draw_fst (gst_channel_ptr,
				   gst_hg_ptr,
				   int);
static void		draw_fractional_fst (gst_channel_ptr,
					     gst_hg_ptr,
					     double *,
					     double,
					     int);
static void		fst_comment (gst_channel_ptr, gst_hg_ptr, int);
static double		get_limit (double);
static void		page_break (gst_channel_ptr);


/*
 * Local Variables
 */

static int		current_page = 0;
static enum plot_size	current_plot_size;
static int		small_plots_on_page = 0;

/*
 * This routine emits Postscript that defines the coordinates of
 * all the terminals.  The "N DefineTerminals" procedure causes space
 * to be allocated for N terminals.  We then emit the terminals, one
 * per line with "X Y DT" procedure calls.  The "DT" procedure simply
 * stuffs the given X,Y coordinates into the next slot in the TermX and
 * TermY arrays.
 *
 * Once the terminals are defined, the X,Y coordinates of a terminal (e.g.,
 * terminal 57) can be pushed onto the Postscript stack using "57 T".
 */

	void
_gst_define_Plot_Terminals (

gst_channel_ptr		chan,		/* IN - output channel */
int			nterms,		/* IN - number of terminals */
const double *		terms,		/* IN - terminals to plot */
gst_scale_info_ptr	sip		/* IN - scaling info */
)
{
int			i;
char			buf1 [32];
char			buf2 [32];
 
	if (terms EQ NULL) return;

	gst_channel_printf (chan, "\n%%%%BeginSetup\n");

	define_coordinate_axes (chan, nterms, terms, sip);

	gst_channel_printf (chan, "\n%d DefineTerminals\n", nterms);

	for (i = 0; i < nterms; i++) {
		gst_unscale_to_string (buf1, terms[2*i], sip);
		gst_unscale_to_string (buf2, terms[2*i+1], sip);
		gst_channel_printf (chan, "\t%s\t%s\tDT\n", buf1, buf2);
	}
	gst_channel_printf (chan, "\n%%%%EndSetup\n\n");
}

/*
 * This routine determines appropriate ranges for the X and Y coordinate
 * axes, and emits the corresponding Postscript definitions for the
 * MinX, MaxX, MinY and MaxY variables.
 */

	static
	void
define_coordinate_axes (

gst_channel_ptr		chan,		/* IN - output channel */
int			nterms,		/* IN - number of terminals */
const double *		terms,		/* IN - terminals to plot */
gst_scale_info_ptr	sip		/* IN - problem scaling info */
)
{
int			i;
double			x, y;
double			minxcoord, maxxcoord;
double			minycoord, maxycoord;
double			xspan, yspan, span;
double			axmin, axmax;
double			aymin, aymax;

	if (nterms < 1) {
		gst_channel_printf (chan, "\n0 1 0 1 SetAxes\n");
		return;
	}

	minxcoord = maxxcoord = *terms++;
	minycoord = maxycoord = *terms++;

	for (i = 1; i < nterms; i++) {
		x = *terms++;
		y = *terms++;
		if (x < minxcoord) {
			minxcoord = x;
		}
		else if (x > maxxcoord) {
			maxxcoord = x;
		}
		if (y < minycoord) {
			minycoord = y;
		}
		else if (y > maxycoord) {
			maxycoord = y;
		}
	}

	minxcoord = gst_unscale_to_double (minxcoord, sip);
	maxxcoord = gst_unscale_to_double (maxxcoord, sip);
	minycoord = gst_unscale_to_double (minycoord, sip);
	maxycoord = gst_unscale_to_double (maxycoord, sip);

	/* We only generate square plots having equal scales on both	*/
	/* axes.  Determine the "span" of the plot, i.e., the length of	*/
	/* each axis in the plot.					*/

	xspan = maxxcoord - minxcoord;
	yspan = maxycoord - minycoord;

	if (xspan EQ 0.0) {
		if (yspan EQ 0.0) {
			/* Single point. */
			if (maxxcoord NE 0.0) {
				if (fabs (maxxcoord) >= fabs (maxycoord)) {
					span = 2.0 * fabs (maxxcoord);
				}
				else {
					span = 2.0 * fabs (maxycoord);
				}
			}
			else if (maxycoord NE 0.0) {
				span = 2.0 * fabs (maxycoord);
			}
			else {
				/* Single point at the origin. */
				span = 2.0;
			}
		}
		else {
			span = get_limit (yspan);
		}
	}
	else if (yspan EQ 0.0) {
		span = get_limit (xspan);
	}
	else if (xspan >= yspan) {
		span = get_limit (xspan);
	}
	else {
		span = get_limit (yspan);
	}

	/* Determine the minimum x axis value. */

	if (xspan EQ 0.0) {
		goto center_x;
	}
	else if ((0.0 <= minxcoord) AND (maxxcoord <= span)) {
		axmin = 0.0;
	}
	else if ((-span <= minxcoord) AND (maxxcoord <= 0.0)) {
		axmin = -span;
	}
	else if ((-0.5 * span <= minxcoord) AND (maxxcoord <= 0.5 * span)) {
		axmin = -0.5 * span;
	}
	else {
center_x:
		/* Center the x coordinates. */
		axmin = 0.5 * (minxcoord + maxxcoord - span);
	}
	axmax = axmin + span;

	/* Determine the minimum y axis value. */

	if (yspan EQ 0.0) {
		goto center_y;
	}
	else if ((0.0 <= minycoord) AND (maxycoord <= span)) {
		aymin = 0.0;
	}
	else if ((-span <= minycoord) AND (maxycoord <= 0.0)) {
		aymin = -span;
	}
	else if ((-0.5 * span <= minycoord) AND (maxycoord <= 0.5 * span)) {
		aymin = -0.5 * span;
	}
	else {
center_y:
		/* Center the y coordinates */
		aymin = 0.5 * (minycoord + maxycoord - span);
	}
	aymax = aymin + span;

	/* Good enough for now... */
	gst_channel_printf (chan, "\n%g %g %g %g SetAxes\n", axmin, axmax, aymin, aymax);
}

/*
 * This routine returns a reasonable scale limit for the given quantity.
 * These are always 1, 2 or 5 times a power of 10.
 */

	static
	double
get_limit (

double		value		/* IN - value to get scale limit for */
)
{
int		i;
double		limit;

	if (value >= 1.0) {
		limit = 1.0;
		for (i = 0; i <= 20; i++) {
			if (limit >= value) return (limit);
			if (2.0 * limit >= value) return (2.0 * limit);
			if (5.0 * limit >= value) return (5.0 * limit);
			limit *= 10.0;
		}
		return (value);
	}

	limit = 1.0;
	for (i = 0; i <= 20; i++) {
		if (0.5 < value * limit) return (1.0 / limit);
		limit *= 10.0;
		if (2.0 < value * limit) return (5.0 / limit);
		if (1.0 < value * limit) return (2.0 / limit);
	}
	return (value);
}

/*
 * This routine generates some Postscript commands to plot each full-set
 * in the given mask.
 */

	void
_gst_plot_full_sets (

gst_channel_ptr		chan,		/* IN - output channel */
gst_hg_ptr		H,		/* IN - hypergraph */
int			nhedges,	/* IN - number of hyperedges to plot */
int *			hedges,		/* IN - list of hyperedge indices */
enum plot_size		plot_size	/* IN - size of plot to produce */
)
{
int		i;
int		hedge_num;
int		nedges;
char		buf [32];
char		title [256];
double *	weights;
gst_scale_info_ptr	sip;

	gst_get_hg_edges (H, &nedges, NULL, NULL, NULL);
	weights = NEWA (nedges, double);
	gst_get_hg_edges (H, NULL, NULL, NULL, weights);

	gst_get_hg_scale_info (H, &sip);

	/* Draw the FSTs with postscript. */
	for (i = 0; i < nhedges; i++) {
		hedge_num = *hedges++;
		_gst_begin_plot (chan, plot_size);
		gst_channel_printf (chan, "\tPlot_Terminals\n");
		fst_comment (chan, H, hedge_num);
		draw_fst (chan, H, hedge_num);
		gst_unscale_to_string (buf, weights[i], sip);
		(void) sprintf (title, "FST %u,  Length = %s",
				(int32u) i, buf);
		_gst_end_plot (chan, title);
	}
	page_break (chan);
	free (weights);
}

/*
 * This routine generates some Postscript commands to plot each full-set
 * in the given mask.  Note that we "group" as many mutually-disjoint
 * full sets as possible on each plot so as to minimize the amount of
 * paper we spew...
 */

	void
_gst_plot_full_sets_grouped (

gst_channel_ptr		chan,		/* IN - output channel */
gst_hg_ptr		H,		/* IN - hypergraph */
int			nhedges,	/* IN - number of hyperedges to plot */
int *			hedges,		/* IN - list of hyperedge indices */
enum plot_size		plot_size	/* IN - size of plot to produce */
)
{
int			i;
int			j;
int			n;
int			nleft;
int			nterms;
int			kmasks;
int			nmasks;
int			nplot;
bitmap_t *		left;
bitmap_t *		tmask;
int *			plist;
int *			term_index;
char *			cp1;
char *			cp2;
char			fsnums [110];
char			fsnum [16];
char			title [120];

	gst_get_hg_terminals (H, &nterms, NULL);
	gst_get_hg_edges (H, &n, NULL, NULL, NULL);

	nmasks = BMAP_ELTS (n);
	kmasks = BMAP_ELTS (nterms);

	left = NEWA (nmasks, bitmap_t);
	tmask = NEWA (kmasks, bitmap_t);
	plist = NEWA (n, int);
	term_index = NEWA (nterms, int);

	/* Make a local copy of all full sets left to plot. */
	for (i = 0; i < nmasks; i++) {
		left [i] = 0;
	}
	for (i = 0; i < nhedges; i++, hedges++) {
		SETBIT (left,  *hedges);
	}
	nleft = n;

	while (nleft > 0) {
		_gst_begin_plot (chan, plot_size);

		for (i = 0; i < kmasks; i++) {
			tmask [i] = 0;
		}

		nplot = 0;
		cp1 = &fsnums [sizeof (fsnums)];
		*--cp1 = '\0';

		for (i = n - 1; i >= 0; i--) {
			if (NOT BITON (left, i)) continue;

			/* Skip full set "i" if not disjoint	*/
			/* with all others in this plot...	*/
			gst_get_hg_one_edge (H, i, NULL, &nterms, term_index);
 
			for (j = 0; j < nterms; j++) {
				if (BITON (tmask, term_index [j])) break;
			}
			if (j < nterms) continue;

			(void) sprintf (fsnum, "%u", (int32u) i);
			for (cp2 = fsnum; *cp2 NE '\0'; cp2++) {
			}

			/* Stop if label does not fit! */
			if ((cp2 - fsnum) + 2 > (cp1 - fsnums)) break;

			while (cp2 > fsnum) {
				*--cp1 = *--cp2;
			}
			*--cp1 = ' ';
			*--cp1 = ',';

			plist [nplot++] = i;
			CLRBIT (left, i);
			--nleft;

			fst_comment (chan, H, i);
			draw_fst (chan, H, i);

			for (j = 0; j < nterms; j++) {
				SETBIT (tmask, term_index [j]);
			}
		}

		gst_channel_printf (chan, "\tPlot_Terminals\n");
		(void) sprintf (title,
				"FST%s %s.",
				(nplot > 1) ? "s" : "",
				cp1 + 2);
		_gst_end_plot (chan, title);
	}

	page_break (chan);

	free ((char *) term_index);
	free ((char *) plist);
	free ((char *) tmask);
	free ((char *) left);
}

/*
 * This routine generates some Postscript commands to plot a SUBSET of
 * the given full-sets in overlaid fashion.
 */

	void
_gst_overlay_plot_subset (

gst_channel_ptr		chan,		/* IN - output channel */
gst_hg_ptr		H,		/* IN - hypergraph */
char *			title,		/* IN - title to display with. */
int			nhedges,	/* IN - number of hyperedges to plot */
int *			hedges,		/* IN - list of hyperedge indices */
enum plot_size		plot_size	/* IN - size of plot to produce */
)
{
int		i;
int		hedge_num;

	/* Draw the FSTs with postscript. */
	_gst_begin_plot (chan, plot_size);
	gst_channel_printf (chan, "\tPlot_Terminals\n");

	for (i = 0; i < nhedges; i++) {
		hedge_num = *hedges++;
		fst_comment (chan, H, hedge_num);
		draw_fst (chan, H, hedge_num);
	}

	_gst_end_plot (chan, title);
}

/*
 * This routine draws a single full Steiner tree.
 */

	static
	void
draw_fst (

gst_channel_ptr		chan,		/* IN - output channel */
gst_hg_ptr		H,		/* IN - hypergraph */
int			edge_number	/* IN - hyperedge number */
)
{
int		i;
int		j;
int		nterms;
int		nsps;
int		nedges;
double *	sps;
int *		edges;
int *		term_index;
char		buf1 [32];
char		buf2 [32];
gst_scale_info_ptr	sip;

	/* Retrieve the necessary information */
	gst_get_hg_one_edge_embedding (H, edge_number, &nsps, NULL, &nedges, NULL);
	sps		= NEWA (2*nsps, double);
	edges		= NEWA (2*nedges, int);
	gst_get_hg_one_edge_embedding (H, edge_number, NULL, sps, NULL, edges);

	gst_get_hg_one_edge (H, edge_number, NULL, &nterms, NULL);
	term_index	= NEWA (2*nterms, int);
	gst_get_hg_one_edge (H, edge_number, NULL, NULL, term_index);

	gst_get_hg_scale_info (H, &sip);

	/* Plot the FST */
	for (i = 0; i < nedges; i++) {
		j = edges[2*i];
		if (j < nterms) {
			gst_channel_printf (chan, "\t%d T", term_index[j]);
		}
		else {
			j -= nterms;
			gst_unscale_to_string (buf1, sps[2*j], sip);
			gst_unscale_to_string (buf2, sps[2*j+1], sip);
			gst_channel_printf (chan, "\t%s\t%s",
					    buf1, buf2);
		}
		j = edges[2*i+1];
		if (j < nterms) {
			gst_channel_printf (chan, "\t%d T\tS\n", term_index[j]);
		}
		else {
			j -= nterms;
			gst_unscale_to_string (buf1, sps[2*j], sip);
			gst_unscale_to_string (buf2, sps[2*j+1], sip);
			gst_channel_printf (chan, "\t%s\t%s\tS\n",
					    buf1, buf2);
		}
	}

	free (sps);
	free (edges);
	free (term_index);
}

/*
 * This routine plots an LP solution.  This is a set of full sets in which
 * full set i has weight Wi, where 0 <= Wi <= 1.  The full sets with
 * weight of 1 are drawn normally.  The fractional ones are drawn as
 * gray-scale "stars" emanating from the center of mass of the terminals.
 */

	void
_gst_plot_lp_solution (

gst_channel_ptr	chan,		/* IN - output channel */
gst_hg_ptr	H,		/* IN - hypergraph */
const char *	title,		/* IN - title for plot. */
const double *	weights,	/* IN - weight of each full set. */
enum plot_size	plot_size	/* IN - size of plot to produce. */
)
{
int			i;
int			j;
int			nedges;
int			nterms;
int			fst_nterms;
int *			fst_terms;
double *		terms;

	gst_get_hg_edges (H, &nedges, NULL, NULL, NULL);

	/* Try to get all terminal coordinates */
	gst_get_hg_terminals (H, &nterms, NULL);
	terms = NEWA (2*nterms, double);

	if (gst_get_hg_vertex_embedding (H, NULL, terms) EQ 0) {
		/* Draw the FSTs with postscript. */
		_gst_begin_plot (chan, plot_size);

		for (i = 0; i < nedges; i++) {
			if (weights [i] <= 0.000001) continue;
			draw_fractional_fst (chan, H, terms, weights [i], i);
		}

		gst_channel_printf (chan, "\tPlot_Terminals\n");

		_gst_end_plot (chan, title);
	}
	else {
		/* Just output the weighted hyperedges. */
		gst_channel_printf (chan, "\n");
		for (i = 0; i < nedges; i++) {
			if (weights [i] <= 0.000001) continue;
			gst_channel_printf (chan, "x^%d = %10.8g\t", i, weights [i]);
			gst_get_hg_one_edge (H, i, NULL, &fst_nterms, NULL);
			fst_terms = NEWA (fst_nterms, int);
			gst_get_hg_one_edge (H, i, NULL, NULL, fst_terms);

			for (j = 0; j < fst_nterms; j++) {
				gst_channel_printf (chan, " %d", fst_terms[j]);
			}
			gst_channel_printf (chan, "\n");
			free (fst_terms);
		}
	}

	free (terms);
}

/*
 * This routine plots a particular subtour violation S within an LP solution.
 * Only FSTs having at least 2 vertices in the subtour S are displayed.
 * The full sets with weight of 1 are drawn normally.  The fractional ones
 * are drawn as gray-scale "stars" emanating from the center of mass of the
 * terminals.
 */

	void
_gst_plot_subtour (

gst_channel_ptr	chan,		/* IN - output channel */
const char *	title,		/* IN - title for plot */
const double *	weights,	/* IN - weight of each full set */
gst_hg_ptr	H,		/* IN - hypergraph */
bitmap_t *	S,		/* IN - subtour to plot */
enum plot_size	plot_size	/* IN - size of plot to produce */
)
{
int			i;
int			j;
int			k;
int			nverts;
int			nedges;
int			esize;
double *		terms;
int *			vlist;

	gst_get_hg_edges (H, &nedges, NULL, NULL, NULL);

	/* Try to get all terminal coordinates */
	gst_get_hg_terminals (H, &nverts, NULL);
	terms = NEWA (2*nverts, double);

	vlist = NEWA (nverts, int);

	if (gst_get_hg_vertex_embedding (H, NULL, terms) EQ 0) {
		/* Draw the FSTs with postscript. */
		_gst_begin_plot (chan, plot_size);

		for (i = 0; i < nedges; i++) {
			if (weights [i] <= 0.000001) continue;
			gst_get_hg_one_edge (H, i, NULL, &esize, vlist);
			k = 0;
			for (j = 0; j < esize; j++) {
				if (BITON (S, vlist [j])) {
					++k;
				}
			}
			if (k < 2) continue;
			draw_fractional_fst (chan, H, terms, weights [i], i);
		}

		gst_channel_printf (chan, "\t0.75 setgray\n");
		gst_channel_printf (chan, "\tPlot_Terminals\n");
		gst_channel_printf (chan, "\t0 setgray\n");

		for (j = 0; j < nverts; j++) {
			if (NOT BITON (S, j)) continue;
			gst_channel_printf (chan, "\t%d\tPT\n", j);
		}

		_gst_end_plot (chan, title);
	}
	else {
		/* Just output the weighted hyperedges. */
		gst_channel_printf (chan, "\n");
		for (i = 0; i < nedges; i++) {
			if (weights [i] <= 0.000001) continue;
			gst_get_hg_one_edge (H, i, NULL, &esize, vlist);
			gst_channel_printf (chan, "x^%d = %10.8g\t", i, weights [i]);
			for (j = 0; j < esize; j++) {
				gst_channel_printf (chan, " %d", vlist [j]);
			}
			gst_channel_printf (chan, "\n");
		}
	}

	free (vlist);
	free (terms);
}

/*
 * This routine draws a single fractional-weight full set.  We draw these
 * as a "star" using the center-of-mass of the terminals as the hub.  The
 * weight is used to determine the darkness of the lines.
 */

	static
	void
draw_fractional_fst (

gst_channel_ptr		chan,		/* IN - output channel */
gst_hg_ptr		H,		/* IN - hypergraph */
double *		terms,		/* IN - terminal coordinates*/
double			weight,		/* IN - weight for full set */
int			edge_number	/* IN - hyperedge number */
)
{
int			i;
int			nterms;
int *			term_index;
double			cx;
double			cy;
char			buf1 [32];
char			buf2 [32];
gst_scale_info_ptr	sip;

	if (weight + 0.000001 >= 1.0) {
		/* Draw integral full sets "normally"... */
		fst_comment (chan, H, edge_number);
		gst_channel_printf (chan, "\t0 setgray\n");
		draw_fst (chan, H, edge_number);
		return;
	}

	fst_comment (chan, H, edge_number);

	/* Retrieve the necessary information */
	gst_get_hg_one_edge (H, edge_number, NULL, &nterms, NULL);
	term_index	= NEWA (2*nterms, int);
	gst_get_hg_one_edge (H, edge_number, NULL, NULL, term_index);

	gst_get_hg_scale_info (H, &sip);

	/* Determine the coordinates of the "hub". */
	cx = 0.0;
	cy = 0.0;
	for (i = 0; i < nterms; i++) {
		cx += terms[2*term_index[i]];
		cy += terms[2*term_index[i] + 1];
	}
	cx /= nterms;
	cy /= nterms;
	gst_unscale_to_string (buf1, cx, sip);
	gst_unscale_to_string (buf2, cy, sip);

	gst_channel_printf (chan, "\t%f setgray\n", 1.0 - weight);

	for (i = 0; i < nterms; i++) {
		gst_channel_printf (chan, "\t%d T\t%s\t%s\tS\n",
			       term_index[i], buf1, buf2);
	}

	free (term_index);
}

/*
 * This routine emits a Postscript comment describing a given hyperedge (FST).
 */

	static
	void
fst_comment (

gst_channel_ptr		chan,		/* IN - output channel */
gst_hg_ptr		H,		/* IN - hypergraph */
int			number		/* IN - hyperedge number */
)
{
int		i;
int		nterms;
int *		terms;

	gst_get_hg_one_edge (H, number, NULL, &nterms, NULL);
	terms = NEWA (nterms, int);
	gst_get_hg_one_edge (H, number, NULL, NULL, terms);

	gst_channel_printf (chan, " %% fs%d:", number);
	for (i = 0; i < nterms; i++) {
		gst_channel_printf (chan, " %u", terms[i]);
	}
	gst_channel_printf (chan, "\n");

	free (terms);
}

/*
 * This routine emits appropriate Postscript code to begin a plot of
 * the given size.  If this is the first plot on a page, then we also
 * emit DSC comments for the page number.  This helps out ghostview and
 * other Postscript previewers.
 */

	void
_gst_begin_plot (

gst_channel_ptr		chan,		/* IN - output channel */
enum plot_size		size		/* IN - size of plot to begin */
)
{
	current_plot_size = size;
	switch (size) {
	case BIG_PLOT:
		page_break (chan);
		announce_new_page (chan);
		gst_channel_printf (chan, "BeginPlot\n");
		break;

	case SMALL_PLOT:
		if (small_plots_on_page EQ 0) {
			announce_new_page (chan);
		}
		gst_channel_printf (chan, "BeginSmallPlot\n");
		break;

	default:
		fprintf (stderr, "_gst_begin_plot: Bug 1.");
		abort();
		break;
	}
}

/*
 * This routine emits appropriate Postscript code to end the plot that
 * is currently in progress.  We also track the number of finished
 * plots per page here.
 */

	void
_gst_end_plot (

gst_channel_ptr		chan,		/* IN - output channel */
const char *		title		/* IN - title for plot */
)
{
	if (title EQ NULL) {
		return;
	}

	gst_channel_printf (chan, "  (%s)\n", title);
	switch (current_plot_size) {
	case BIG_PLOT:
		gst_channel_printf (chan, "EndPlot\n\n");
		break;

	case SMALL_PLOT:
		gst_channel_printf (chan, "EndSmallPlot2\n\n");
		++small_plots_on_page;
		if (small_plots_on_page >= SMALL_PLOTS_PER_PAGE) {
			small_plots_on_page = 0;
		}
		break;

	default:
		fprintf (stderr, "_gst_end_plot: Bug 1.");
		abort();
		break;
	}
}

/*
 * This routine puts a %%Page: comment into the output to mark a page
 * boundary.  This is for the benefit of Ghostview and other Postscript
 * previewers.
 */

	static
	void
announce_new_page (

gst_channel_ptr		chan		/* IN - output channel */
)
{
	++current_page;
	gst_channel_printf (chan, "%%%%Page: %u %u\n",
		       (int32u) current_page,
		       (int32u) current_page);
}

/*
 * This routine introduces a page break.  This is needed when doing
 * small plots, and the page has not yet filled up.
 */

	static
	void
page_break (

gst_channel_ptr		chan		/* IN - output channel */
)
{
	if (small_plots_on_page NE 0) {
		gst_channel_printf (chan, "FlushSmallPlot\n");
		small_plots_on_page = 0;
	}
}
