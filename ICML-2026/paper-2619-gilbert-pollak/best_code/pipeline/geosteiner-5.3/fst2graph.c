/***********************************************************************

	$Id: fst2graph.c,v 1.25 2023/04/03 17:31:43 warme Exp $

	File:	fst2graph.c
	Rev:	e-5
	Date:	04/03/2023

	Copyright (c) 1995, 2023 by David M. Warme & Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	The main routine to generate ordinary graphs from FSTs.
	There are two possibilities:
	1. Reduced grid-graph for rectilinear problem,
	2. Graph with all terminals and Steiner points as nodes
	   and all line segments as edges (Euclidean and rectilinear
	   problem).

************************************************************************

	Modification Log:

	a-1:	05/18/95	warme
		: Created.
	b-1:	01/19/2001	martinz
		: Changed int16u to int32u in order to be able to handle
		:  bigger instances (also changed macros).
		: Changed name from redg.c to fst2graph.c
		: Support for OR-Library format and SteinLib format
		:  (made old format obsolete).
		: Added non-grid graph generation.
	c-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Uses parameters.
		: Split off main function to fst2graphmain.c.
		: Changed functions to create a new hypergraph instead
		:  of just writing to stdout.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
		: Add cost_extension for SteinLib "integer" format.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.  Upgrade fatals.
		: Conditionalize some GMP-specific code.
	e-4:	11/18/2022	warme
		: Fix -Wall issues.
	e-5:	04/03/2023	warme
		: Fix -Wall issues.

************************************************************************/

#include "egmp.h"
#include "fatal.h"
#include <float.h>
#include "geosteiner.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "metric.h"
#include "parmblk.h"
#include "point.h"
#include "prepostlude.h"
#include "sortfuncs.h"
#include "steiner.h"


/*
 * Global Routines
 */

gst_hg_ptr gst_hg_to_graph (gst_hg_ptr, gst_param_ptr, int *);

/*
 * External References
 */

	/* none */

/*
 * Local Equates
 */

#define UP_EDGE		0x80000000	/* edge exits point and goes up */
#define RIGHT_EDGE	0x40000000	/* edge exits point and goes right */
#define GMASK		0x3FFFFFFF	/* mask of other bits */


/*
 * Local Types
 */

struct grid {
	int		nt;
	int		ns;
	coord_t *	x_coord;
	coord_t *	y_coord;
	int *		xindex;
	int *		yindex;
	int32u *	gridp;
	bool		count_edges;
	int		number_of_edges;
	int *		edges;
	int *		edge_sizes;
	double *	weights;
};


/*
 * Local Routines
 */

static gst_hg_ptr	create_edge_graph (gst_hg_ptr, gst_param_ptr, int *);
static gst_hg_ptr	create_grid_graph (gst_hg_ptr, gst_param_ptr, int *);
static void		draw_bb_grid (int, int, int, int, struct grid *);
static void		draw_bb_grid_horizontal (int, int, int, struct grid *);
static void		draw_bb_grid_vertical (int, int, int, struct grid *);
static void		draw_full_sets_on_grid (bitmap_t *,
						struct gst_hypergraph *,
						struct grid *);
static void		edge_down (int32u *, int, int, int, dist_t,
				   struct gst_hypergraph *, struct grid *);
static void		edge_left (int32u *, int, int, int, dist_t,
				   struct gst_hypergraph *, struct grid *);
static void		edge_right (int32u *, int, int, int, dist_t,
				    struct gst_hypergraph *, struct grid *);
static void		edge_up (int32u *, int, int, int, dist_t,
				 struct gst_hypergraph *, struct grid *);
static void		identify_steiner_points (struct grid *);
static int		map_x (coord_t, struct grid *);
static int		map_y (coord_t, struct grid *);
static void		output_edge (int, int, dist_t,
				     struct gst_hypergraph *, struct grid *);

/*
 * This is the main routine for converting hypergraphs to graphs.
 */

	gst_hg_ptr
gst_hg_to_graph (

gst_hg_ptr	H,		/* IN - hypergraph */
gst_param_ptr	params,		/* IN - parameters */ 
int *		status		/* OUT - */
)
{
gst_hg_ptr	H2;

	GST_PRELUDE

	if (H -> full_trees EQ NULL) {
		/* This can only be done for embedded hypergraphs. */
		FATAL_ERROR;
	}

	if (params EQ NULL) {
		params = (gst_param_ptr) &_gst_default_parmblk;
	}

	if (params -> grid_overlay AND _gst_is_rectilinear (H)) {
		H2 = create_grid_graph (H, params, status);
	}
	else {
		H2 = create_edge_graph (H, params, status);
	}

	GST_POSTLUDE
	return H2;
}

/*
 * This routine computes the edge-graph obtained by taking the union
 * of all of all line segments in all FSTs
 */
	gst_hg_ptr
create_edge_graph (

gst_hg_ptr	cip,		/* IN - hypergraph */
gst_param_ptr	params,		/* IN - parameters */
int *		status		/* IN - print format */
)
{
int			i, j, i1, i2;
int			sp_total;
int			n2edges;
int			nedges;
int			nverts;
int *			edges;
int *			ep;
int *			edge_sizes;
int *			esp;
int			sp_index;
bool			euclidean;
double *		weights;
double *		wp;
double *		coords;
double *		cp;
struct full_set *	fsp;
int *			tlist;
struct pset *		terms;
struct point *		p1;
struct point *		p2;
dist_t			d;
bitmap_t *		fset_mask;
gst_hg_ptr		H2;

	euclidean = FALSE;
	if (_gst_is_euclidean (cip)) {
		euclidean = TRUE;
	}
	else if (_gst_is_rectilinear (cip)){
		euclidean = FALSE;
	}
	else {
		/* Can only handle Euclidean/rectilinear. */
		FATAL_ERROR;
	}

	if (status NE NULL) {
		*status = 0;
	}

	fset_mask	= cip -> initial_edge_mask;
	/* We copy to get scale-info, properties, metric and more, but it is
	   an inefficient approach. */
	H2 = gst_create_hg (NULL);
	gst_copy_hg (H2, cip);

	/* Total number of Steiner points and ordinary edges */
	nedges = cip -> num_edges;
	sp_total = 0;
	n2edges	 = 0;
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (fset_mask, i)) continue;
		sp_total += cip -> full_trees [i] -> steiners -> n;
		n2edges	 += cip -> full_trees [i] -> nedges;
	}
	nverts = cip -> num_verts + sp_total;

	/* Set number of vertices and their status as terminal/Steiner points */
	gst_set_hg_number_of_vertices (H2, nverts);
	for (i = 0; i < cip -> num_verts; i++) {
		H2 -> tflag[i] = TRUE;
	}
	for (i = cip -> num_verts; i < nverts; i++) {
		H2 -> tflag[i] = FALSE;
	}

	wp	= weights	= NEWA (n2edges, double);
	esp	= edge_sizes	= NEWA (n2edges, int);
	ep	= edges		= NEWA (2*n2edges, int);

	/* Write (ordinary) graph edges */
	sp_index = cip -> num_verts;
	for (i = 0; i < nedges; i++) {
		if (NOT BITON (fset_mask, i)) continue;
		fsp = cip -> full_trees [i];
		tlist = fsp -> tlist;
		terms = fsp -> terminals;
		for (j = 0; j < fsp -> nedges; j++) {

			/* Compute length of this edge */
			p1 = (fsp -> edges[j].p1 < terms -> n)
				? &(terms -> a[ fsp -> edges[j].p1 ])
				: &(fsp -> steiners -> a [fsp -> edges[j].p1 - terms -> n]);
			p2 = (fsp -> edges[j].p2 < terms -> n)
				? &(terms -> a[ fsp -> edges[j].p2 ])
				: &(fsp -> steiners -> a [fsp -> edges[j].p2 - terms -> n]);

			i1 = (fsp -> edges[j].p1 < terms -> n)
				? tlist [ fsp -> edges[j].p1 ]
				: fsp -> edges[j].p1 - terms -> n + sp_index;
			i2 = (fsp -> edges[j].p2 < terms -> n)
				? tlist [ fsp -> edges[j].p2 ]
				: fsp -> edges[j].p2 - terms -> n + sp_index;

			if (euclidean) {
				d = EDIST(p1, p2);
			}
			else {
				d = RDIST(p1, p2);
			}

			*ep++	= i1;
			*ep++	= i2;
			*esp++	= 2;
			*wp++	= d;
		}
		if (fsp -> steiners NE NULL) {
			sp_index += fsp -> steiners -> n;
		}
	}

	gst_set_hg_edges (H2, n2edges, edge_sizes, edges, weights);
	free (edges);
	free (edge_sizes);
	free (weights);

#if HAVE_GMP
	if (euclidean AND
	    (params -> save_format EQ GST_PVAL_SAVE_FORMAT_STEINLIB_INT)) {
		/* This is a hack.  We would like to implement this	*/
		/* purely within gst_save_hg(), but the graph that we	*/
		/* generate here has (cip->full_trees EQ NULL), so	*/
		/* there is no way to reconstruct the recursive		*/
		/* structure of each EFST in order to compute high-	*/
		/* precision edge lengths.  Instead, we compute them	*/
		/* here, and add them to the generated hypergraph as a	*/
		/* black box "cost_extention" object.  If present,	*/
		/* gst_save_hg() uses this extension to write out the	*/
		/* resulting edge cost values.				*/
		H2 -> cost_extension = _gst_egmp_graph_cost_extension (
							cip,
							params);
	}
#endif

	/* Coordinates */
	cp = coords = NEWA (2*nverts, double);
	for (i = 0; i < cip -> num_verts; i++) { /* terminals */
		*cp++ = cip -> pts -> a[i].x;
		*cp++ = cip -> pts -> a[i].y;
	}
	for (i = 0; i < nedges; i++) { /* Steiner points */
		fsp = cip -> full_trees [i];
		for (j = 0; j < fsp -> steiners -> n; j++) {
			*cp++ = fsp -> steiners -> a[j].x;
			*cp++ = fsp -> steiners -> a[j].y;
		}
	}
	gst_set_hg_vertex_embedding (H2, 2, coords);
	free (coords);

	return H2;
}

/*
 * This routine computes the grid-graph obtained by taking the union
 * of all FSTs
 */
	gst_hg_ptr
create_grid_graph (

gst_hg_ptr	cip,		/* IN - hypergraph */
gst_param_ptr	params,		/* IN - parameters */
int *		status		/* IN - print format */
)
{
int		i;
int		j;
int		k;
int		nverts;
int		nverts2;
int		n2edges;
int		v1;
int *		edges;
int *		edge_sizes;
double *	weights;
double *	coords;
double *	cp;
int32u *	gridp;
coord_t		coord;
dist_t		prev_coord;
int		prev_index;
int *		tmp;
struct grid *	grid;
bitmap_t *	tmap;
bitmap_t *	fset_mask;
gst_hg_ptr	H2;

	if (status NE NULL) {
		*status = 0;
	}

	tmap		= cip -> initial_vert_mask;
	fset_mask	= cip -> initial_edge_mask;

	nverts = cip -> num_verts;
	grid = NEW (struct grid);
	grid -> nt		= nverts;
	grid -> ns		= 0;
	grid -> x_coord	= NEWA (nverts, coord_t);
	grid -> y_coord	= NEWA (nverts, coord_t);
	grid -> xindex	= NEWA (nverts, int);
	grid -> yindex	= NEWA (nverts, int);

	/* Compute map giving index of each terminal in sequence when	*/
	/* sorted by increasing X coordinate.				*/
	tmp = _gst_heapsort_x (cip -> pts);

	prev_coord = INF_DISTANCE;
	prev_index = -1;
	for (i = 0; i < nverts; i++) {
		j = tmp [i];
		coord = cip -> pts -> a [j].x;
		grid -> x_coord [i] = coord;
		if (coord NE prev_coord) {
			prev_index = i;
		}
		grid -> xindex [j] = prev_index;
		prev_coord = coord;
	}
	free (tmp);

	/* Compute map giving index of each terminal in sequence when	*/
	/* sorted by increasing Y coordinate.				*/
	tmp = _gst_heapsort_y (cip -> pts);

	prev_coord = INF_DISTANCE;
	prev_index = -1;
	for (i = 0; i < nverts; i++) {
		j = tmp [i];
		coord = cip -> pts -> a [j].y;
		grid -> y_coord [i] = coord;
		if (coord NE prev_coord) {
			prev_index = i;
		}
		grid -> yindex [j] = prev_index;
		prev_coord = coord;
	}
	free (tmp);

	/* Allocate and zero matrix to hold the grid... */
	nverts2 = (nverts + 1) * nverts;
	grid -> gridp = NEWA (nverts2, int32u);
	for (i = 0; i < nverts2; i++) {
		grid -> gridp [i] = 0;
	}
	grid -> gridp += nverts;

	/* Set vertex number for each terminal in the grid... */
	for (i = 0; i < nverts; i++) {
		if (NOT BITON (tmap, i)) {
			/* Should not have any duplicate terminals! */
			FATAL_ERROR;
		}
		j = grid -> xindex [i];
		k = grid -> yindex [i];
		grid -> gridp [k * nverts + j] = i + 1;
	}

	draw_full_sets_on_grid (fset_mask, cip, grid);

	identify_steiner_points (grid);

	/* Count every edge... */
	grid -> count_edges = TRUE;
	grid -> number_of_edges = 0;
	gridp = grid -> gridp;
	for (i = 0; i < nverts; i++) {
		for (j = 0; j < nverts; j++, gridp++) {
			v1 = (*gridp & GMASK);
			if (v1 EQ 0) continue;
			if ((*gridp & RIGHT_EDGE) NE 0) {
				edge_right (gridp, v1, i, j, 0, cip, grid);
			}
			if ((*gridp & UP_EDGE) NE 0) {
				edge_up (gridp, v1, i, j, 0, cip, grid);
			}
			if ((j > 0) AND ((gridp [-1] & RIGHT_EDGE) NE 0)) {
				edge_left (gridp, v1, i, j, 0, cip, grid);
			}
			if ((i > 0) AND ((gridp [-nverts] & UP_EDGE) NE 0)) {
				edge_down (gridp, v1, i, j, 0, cip, grid);
			}
		}
	}

	/* We copy to get scale-info, properties, metric and more, but it is
	   an inefficient approach because we also copy edges/terminals. */
	H2 = gst_create_hg (NULL);
	gst_copy_hg (H2, cip);

	/* Set number of vertices and edges. */
	gst_set_hg_number_of_vertices (H2, nverts + grid -> ns);
	for (i = 0; i < nverts; i++) {
		H2 -> tflag[i] = TRUE;
	}
	for (i = nverts; i < nverts + grid -> ns; i++) {
		H2 -> tflag[i] = FALSE;
	}

	n2edges = grid -> number_of_edges;
	grid -> weights		= weights	= NEWA (n2edges, double);
	grid -> edge_sizes	= edge_sizes	= NEWA (n2edges, int);
	grid -> edges		= edges		= NEWA (2*n2edges, int);

	/* Output every edge... */
	grid -> count_edges = FALSE;
	gridp = grid -> gridp;
	for (i = 0; i < nverts; i++) {
		for (j = 0; j < nverts; j++, gridp++) {
			v1 = (*gridp & GMASK);
			if (v1 EQ 0) continue;
			if ((*gridp & RIGHT_EDGE) NE 0) {
				edge_right (gridp, v1, i, j, 0, cip, grid);
			}
			if ((*gridp & UP_EDGE) NE 0) {
				edge_up (gridp, v1, i, j, 0, cip, grid);
			}
			if ((j > 0) AND ((gridp [-1] & RIGHT_EDGE) NE 0)) {
				edge_left (gridp, v1, i, j, 0, cip, grid);
			}
			if ((i > 0) AND ((gridp [-nverts] & UP_EDGE) NE 0)) {
				edge_down (gridp, v1, i, j, 0, cip, grid);
			}
		}
	}
#if 0
	for (i = 0; i < n2edges; i++) {
		printf ("Edge %d Size %d - (%d, %d) - %f\n",
			i, edge_sizes[i], edges[2*i], edges[2*i+1], weights[i]);
	}
#endif

	gst_set_hg_edges (H2, n2edges, edge_sizes, edges, weights);
	free (edges);
	free (edge_sizes);
	free (weights);

	/* Coordinates */
	cp = coords = NEWA (2*(nverts + grid -> ns), double);
	for (i = 0; i < nverts; i++) { /* terminals */
		*cp++ = cip -> pts -> a[i].x;
		*cp++ = cip -> pts -> a[i].y;
	}
	gridp = grid -> gridp;
	for (i = 0; i < nverts; i++) { /* Steiner points */
		for (j = 0; j < nverts; j++, gridp++) {
			if ((*gridp & GMASK) > nverts) {
				 *cp++ = grid -> x_coord [j];
				 *cp++ = grid -> y_coord [i];
			}
		}
	}

	gst_set_hg_vertex_embedding (H2, 2, coords);
	free (coords);

	free ((char *) (grid -> gridp - nverts));
	free ((char *) grid -> yindex);
	free ((char *) grid -> xindex);
	free ((char *) grid -> y_coord);
	free ((char *) grid -> x_coord);
	free ((char *) grid);

	return H2;
}

/*
 * This routine draws all valid FSTs onto the given grid in
 * an overlaid fashion.
 */

	static
	void
draw_full_sets_on_grid (

bitmap_t *		fset_mask,	/* IN - mask of valid FSTs */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
struct grid *		grid
)
{
int			i;
int			j;
int			u, v, ux, uy, vx, vy;
int			nt;
int			ns;
struct full_set *	fsp;
int *			tlist;
struct pset *		terms;
struct pset *		steins;
struct edge *		ep;
int *			xi;
int *			yi;

	xi = grid -> xindex;
	yi = grid -> yindex;

	for (i = 0; i < cip -> num_edges; i++) {
		if (NOT BITON (fset_mask, i)) continue;
		fsp = cip -> full_trees [i];

		ep = fsp -> edges;
		tlist = fsp -> tlist;
		terms = fsp -> terminals;
		steins = fsp -> steiners;
		nt = terms -> n;
		ns = steins -> n;
		for (j = 0; j < fsp -> nedges; j++, ep++) {
			u = ep -> p1;
			FATAL_ERROR_IF (u < 0);
			if (u < nt) {
				u = tlist [u];
				ux = xi [u];
				uy = yi [u];
			}
			else {
				u -= nt;
				FATAL_ERROR_IF (u >= ns);
				ux = map_x (steins -> a [u].x, grid);
				uy = map_y (steins -> a [u].y, grid);
			}

			v = ep -> p2;
			FATAL_ERROR_IF (v < 0);
			if (v < nt) {
				v = tlist [v];
				vx = xi [v];
				vy = yi [v];
			}
			else {
				v -= nt;
				FATAL_ERROR_IF (v >= ns);
				vx = map_x (steins -> a [v].x, grid);
				vy = map_y (steins -> a [v].y, grid);
			}

			draw_bb_grid (ux, uy, vx, vy, grid);
		}
	}
}

/*
 * This routine determines the grid index of the given X coordinate.
 */

	static
	int
map_x (

coord_t		x,
struct grid *	grid
)
{
int		i;

	for (i = 0; i < grid -> nt; i++) {
		if (x EQ grid -> x_coord [i]) return (i);
		FATAL_ERROR_IF (x < grid -> x_coord [i]);
	}

	FATAL_ERROR;
	return (0);
}

/*
 * This routine determines the grid index of the given Y coordinate.
 */

	static
	int
map_y (

coord_t		y,
struct grid *	grid
)
{
int		i;

	for (i = 0; i < grid -> nt; i++) {
		if (y EQ grid -> y_coord [i]) return (i);
		FATAL_ERROR_IF (y < grid -> y_coord [i]);
	}

	FATAL_ERROR;
	return (0);
}

/*
 * This routine is used to draw a backbone onto the grid.  A backbone
 * consists of a vertical line segment from the first point, and a
 * horizontal line from the second point.  Each segment consists only
 * of the points between the given points and the intersections of the
 * horizontal and vertical lines.
 */

	static
	void
draw_bb_grid (

int		x0,
int		y0,
int		x1,
int		y1,
struct grid *	grid
)
{
int		nt;

	nt = grid -> nt;
	FATAL_ERROR_IF ((x0 < 0) OR (x0 >= nt));
	FATAL_ERROR_IF ((y0 < 0) OR (y0 >= nt));
	FATAL_ERROR_IF ((x1 < 0) OR (x1 >= nt));
	FATAL_ERROR_IF ((y1 < 0) OR (y1 >= nt));

	/* Assume RFSTs are left-most topologies.  Plot corner	*/
	/* segments so that the vertical segment is to the left	*/
	/* of the horizontal segment.				*/

	if (x0 < x1) {
		draw_bb_grid_vertical (x0, y0, y1, grid);
		draw_bb_grid_horizontal (x1, y1, x0, grid);
	}
	else {
		draw_bb_grid_vertical (x1, y1, y0, grid);
		draw_bb_grid_horizontal (x0, y0, x1, grid);
	}
}

/*
 * This routine draws a vertical line onto the grid.
 */

	static
	void
draw_bb_grid_vertical (

int		x0,
int		y0,
int		y1,
struct grid *	grid
)
{
int		i;
int		n;
int32u *	gridp;

	if (y0 > y1) {
		i = y0;
		y0 = y1;
		y1 = i;
	}

	n = grid -> nt;

	FATAL_ERROR_IF ((x0 < 0) OR (x0 >= n) OR (y0 < 0) OR (y1 > n));

	gridp = grid -> gridp;
	gridp += (n * y0 + x0);
	while (y0 < y1) {
		*gridp |= UP_EDGE;
		gridp += n;
		++y0;
	}
}

/*
 * This routine draws a horizontal line onto the grid.
 */

	static
	void
draw_bb_grid_horizontal (

int		x0,
int		y0,
int		x1,
struct grid *	grid
)
{
int		i;
int		n;
int32u *	gridp;

	if (x0 > x1) {
		i = x0;
		x0 = x1;
		x1 = i;
	}

	n = grid -> nt;

	FATAL_ERROR_IF ((y0 < 0) OR (y0 >= n) OR (x0 < 0) OR (x1 > n));

	gridp = grid -> gridp;
	gridp += (n * y0 + x0);
	while (x0 < x1) {
		*gridp |= RIGHT_EDGE;
		++gridp;
		++x0;
	}
}

/*
 * This routine is used to identify the steiner points on the grid.
 */

	static
	void
identify_steiner_points (

struct grid *	grid
)

{
int		i;
int		j;
int		nt;
int		ns;
int		code;
int32u *	gridp;

	nt = grid -> nt;
	ns = 0;
	gridp = grid -> gridp;
	for (i = 0; i < nt; i++) {
		for (j = 0; j < nt; j++, gridp++) {
			code = 0;
			if ((*gridp & RIGHT_EDGE) NE 0) {
				code |= 1;
			}
			if ((*gridp & UP_EDGE) NE 0) {
				code |= 0x02;
			}
			if ((j > 0) AND ((gridp [-1] & RIGHT_EDGE) NE 0)) {
				code |= 0x04;
			}
			if ((i > 0) AND ((gridp [-nt] & UP_EDGE) NE 0)) {
				code |= 0x08;
			}
			switch (code) {
			case 0x00:
				if ((*gridp & GMASK) NE 0) {
					/* Terminal with no connections! */
					FATAL_ERROR;
				}
				break;

			case 0x01:
			case 0x02:
			case 0x04:
			case 0x08:
				if ((*gridp & GMASK) EQ 0) {
					/* degree 1 vertex must be terminal! */
					FATAL_ERROR;
				}
				break;

			case 0x03:
			case 0x05:
			case 0x06:
			case 0x09:
			case 0x0A:
			case 0x0C:
				break;

			default:
				/* vertex has degree 3 or 4... */
				if ((*gridp & GMASK) EQ 0) {
					/* This is a new steiner point! */
					++ns;
					*gridp = (*gridp & ~GMASK) |
						 ((nt + ns) & GMASK);
				}
				break;
			}
		}
	}

	grid -> ns = ns;
}

/*
 * This routine chases the grid-graph edge going RIGHT from position
 * I, J.  We traverse at least one grid element, but stop once we
 * discover a terminal or steiner node.
 */

	static
	void
edge_right (

int32u *		gridp,		/* IN - current grid element */
int			v1,		/* IN - first vertex of edge */
int			i,		/* IN - current Y index */
int			j,		/* IN - current X index */
dist_t			total,		/* IN - total length of edge so far */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
struct grid *		grid
)
{
int		nt;
int		v2;
int		code;

	nt = grid -> nt;
	for (;;) {
		/* Step once to the right. */
		if ((j + 1) >= nt) {
			/* Ran off edge of grid! */
			FATAL_ERROR;
		}
		total += (grid -> x_coord [j+1] - grid -> x_coord [j]);
		++j;
		++gridp;

		code = 0;
		if ((*gridp & RIGHT_EDGE) NE 0) {
			code |= 1;
		}
		if ((*gridp & UP_EDGE) NE 0) {
			code |= 0x02;
		}
		if ((j > 0) AND ((gridp [-1] & RIGHT_EDGE) NE 0)) {
			code |= 0x04;
		}
		if ((i > 0) AND ((gridp [-nt] & UP_EDGE) NE 0)) {
			code |= 0x08;
		}
		if ((code & 0x04) EQ 0) {
			/* No edge going out the way we came in! */
			FATAL_ERROR;
		}
		v2 = (*gridp & GMASK);

		/* If we have stepped to a vertex, get out! */
		if (v2 > 0) break;

		/* Determine which direction to go... */
		switch (code) {
		case 0x05:	break;		/* keep on going... */
		case 0x06:
			edge_up (gridp, v1, i, j, total, cip, grid);
			return;

		case 0x0C:
			edge_down (gridp, v1, i, j, total, cip, grid);
			return;

		default:
			/* Bad type of intersection... */
			FATAL_ERROR;
		}
	}

	if (v1 < v2) {
		/* We have an edge of the grid-graph! */
		output_edge (v1, v2, total, cip, grid);
	}
}

/*
 * This routine chases the grid-graph edge going UP from position
 * I, J.  We traverse at least one grid element, but stop once we
 * discover a terminal or steiner node.
 */

	static
	void
edge_up (

int32u *		gridp,		/* IN - current grid element */
int			v1,		/* IN - first vertex of edge */
int			i,		/* IN - current Y index */
int			j,		/* IN - current X index */
dist_t			total,		/* IN - total length of edge so far */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
struct grid *		grid
)
{
int		nt;
int		v2;
int		code;

	nt = grid -> nt;
	for (;;) {
		/* Step once upward. */
		if ((i + 1) >= nt) {
			/* Ran off edge of grid! */
			FATAL_ERROR;
		}
		total += (grid -> y_coord [i+1] - grid -> y_coord [i]);
		++i;
		gridp += nt;

		code = 0;
		if ((*gridp & RIGHT_EDGE) NE 0) {
			code |= 1;
		}
		if ((*gridp & UP_EDGE) NE 0) {
			code |= 0x02;
		}
		if ((j > 0) AND ((gridp [-1] & RIGHT_EDGE) NE 0)) {
			code |= 0x04;
		}
		if ((i > 0) AND ((gridp [-nt] & UP_EDGE) NE 0)) {
			code |= 0x08;
		}
		if ((code & 0x08) EQ 0) {
			/* No edge going out the way we came in! */
			FATAL_ERROR;
		}
		v2 = (*gridp & GMASK);

		/* If we have stepped to a vertex, get out! */
		if (v2 > 0) break;

		/* Determine which direction to go... */
		switch (code) {
		case 0x0A:	break;		/* keep on going... */
		case 0x09:
			edge_right (gridp, v1, i, j, total, cip, grid);
			return;

		case 0x0C:
			edge_left (gridp, v1, i, j, total, cip, grid);
			return;

		default:
			/* Bad type of intersection... */
			FATAL_ERROR;
		}
	}

	if (v1 < v2) {
		/* We have an edge of the grid-graph! */
		output_edge (v1, v2, total, cip, grid);
	}
}

/*
 * This routine chases the grid-graph edge going LEFT from position
 * I, J.  We traverse at least one grid element, but stop once we
 * discover a terminal or steiner node.
 */

	static
	void
edge_left (

int32u *		gridp,		/* IN - current grid element */
int			v1,		/* IN - first vertex of edge */
int			i,		/* IN - current Y index */
int			j,		/* IN - current X index */
dist_t			total,		/* IN - total length of edge so far */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
struct grid *		grid
)
{
int		nt;
int		v2;
int		code;

	nt = grid -> nt;
	for (;;) {
		/* Step once to the left. */
		if (j <= 0) {
			/* Ran off edge of grid! */
			FATAL_ERROR;
		}
		total += (grid -> x_coord [j] - grid -> x_coord [j-1]);
		--j;
		--gridp;

		code = 0;
		if ((*gridp & RIGHT_EDGE) NE 0) {
			code |= 1;
		}
		if ((*gridp & UP_EDGE) NE 0) {
			code |= 0x02;
		}
		if ((j > 0) AND ((gridp [-1] & RIGHT_EDGE) NE 0)) {
			code |= 0x04;
		}
		if ((i > 0) AND ((gridp [-nt] & UP_EDGE) NE 0)) {
			code |= 0x08;
		}
		if ((code & 0x01) EQ 0) {
			/* No edge going out the way we came in! */
			FATAL_ERROR;
		}
		v2 = (*gridp & GMASK);

		/* If we have stepped to a vertex, get out! */
		if (v2 > 0) break;

		/* Determine which direction to go... */
		switch (code) {
		case 0x05:	break;		/* keep on going... */
		case 0x03:
			edge_up (gridp, v1, i, j, total, cip, grid);
			return;

		case 0x09:
			edge_down (gridp, v1, i, j, total, cip, grid);
			return;

		default:
			/* Bad type of intersection... */
			FATAL_ERROR;
		}
	}

	if (v1 < v2) {
		/* We have an edge of the grid-graph! */
		output_edge (v1, v2, total, cip, grid);
	}
}

/*
 * This routine chases the grid-graph edge going DOWN from position
 * I, J.  We traverse at least one grid element, but stop once we
 * discover a terminal or steiner node.
 */

	static
	void
edge_down (

int32u *		gridp,		/* IN - current grid element */
int			v1,		/* IN - first vertex of edge */
int			i,		/* IN - current Y index */
int			j,		/* IN - current X index */
dist_t			total,		/* IN - total length of edge so far */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
struct grid *		grid
)
{
int		nt;
int		v2;
int		code;

	nt = grid -> nt;
	for (;;) {
		/* Step once downward. */
		if (i <= 0) {
			/* Ran off edge of grid! */
			FATAL_ERROR;
		}
		total += (grid -> y_coord [i] - grid -> y_coord [i-1]);
		--i;
		gridp -= nt;

		code = 0;
		if ((*gridp & RIGHT_EDGE) NE 0) {
			code |= 1;
		}
		if ((*gridp & UP_EDGE) NE 0) {
			code |= 0x02;
		}
		if ((j > 0) AND ((gridp [-1] & RIGHT_EDGE) NE 0)) {
			code |= 0x04;
		}
		if ((i > 0) AND ((gridp [-nt] & UP_EDGE) NE 0)) {
			code |= 0x08;
		}
		if ((code & 0x02) EQ 0) {
			/* No edge going out the way we came in! */
			FATAL_ERROR;
		}
		v2 = (*gridp & GMASK);

		/* If we have stepped to a vertex, get out! */
		if (v2 > 0) break;

		/* Determine which direction to go... */
		switch (code) {
		case 0x0A:	break;		/* keep on going... */
		case 0x03:
			edge_right (gridp, v1, i, j, total, cip, grid);
			return;

		case 0x06:
			edge_left (gridp, v1, i, j, total, cip, grid);
			return;

		default:
			/* Bad type of intersection... */
			FATAL_ERROR;
		}
	}

	if (v1 < v2) {
		/* We have an edge of the grid-graph! */
		output_edge (v1, v2, total, cip, grid);
	}
}

/*
 * This routine either outputs the edges, or counts them, as appropriate.
 */

	static
	void
output_edge (

int			v1,		/* IN - first vertex of edge */
int			v2,		/* IN - second vertex of edge */
dist_t			total,		/* IN - total length of edge */
struct gst_hypergraph *	cip,		/* IN - compatibility info */
struct grid *		grid
)
{
	if (grid -> count_edges) {
		++grid -> number_of_edges;
	}
	else {
		*(grid -> edges++)	= v1 - 1;
		*(grid -> edges++)	= v2 - 1;
		*(grid -> edge_sizes++) = 2;
		*(grid -> weights++)	= total;
	}
}
