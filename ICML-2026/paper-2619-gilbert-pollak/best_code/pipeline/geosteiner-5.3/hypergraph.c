/***********************************************************************

	$Id: hypergraph.c,v 1.60 2022/11/19 13:45:52 warme Exp $

	File:	hypergraph.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	04/30/02	benny
		: Created.
	b-1:	02/02/2014	warme
		: Added gst_{get|set}_hg_vertex_types().
	b-2:	04/24/2014	warme
		: Provide correct implementation of terminal flags
		:  in gst_get_hg_terminals().
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
		: Added cost_extension.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix several issues in gst_get_hg_edge_embedding().
		: Fix -Wall issues.  Upgrade fatals.
	e-4:	11/18/2022	warme
		: Fix comment style.
		: Fix -Wall issues.

************************************************************************/

#include "costextension.h"
#include "geosteiner.h"
#include "fatal.h"
#include <float.h>
#include "io.h"
#include "logic.h"
#include "memory.h"
#include "metric.h"
#include "p1read.h"
#include "parmblk.h"
#include "point.h"
#include "prepostlude.h"
#include "steiner.h"
#include <string.h>

/*
 * Global functions
 */

gst_proplist_ptr	gst_get_hg_properties (gst_hg_ptr);

struct full_set *	_gst_remove_degree_two_steiner_points (struct full_set *);

/*
 * Local functions
 */

static void		free_edges (gst_hg_ptr);
static void		free_vertices (gst_hg_ptr);

/* Only bump the version number if it is really necessary */
#define bump_version(hg) \
	{ if ((hg) -> version EQ (hg) -> requested_version) {	\
		++((hg) -> version);				\
	  }							\
	}

/*
 * Create an empty hypergraph object.
 */

	gst_hg_ptr
gst_create_hg (

int *	status
)
{
gst_hg_ptr	H;

	GST_PRELUDE

	if (status NE NULL) {
		*status = 0;
	}

	H = NEW (struct gst_hypergraph);
	(void) memset (H, 0, sizeof (struct gst_hypergraph));

	H -> metric		= gst_create_metric (GST_METRIC_NONE, 0, NULL);

	H -> scale			= NEW (struct gst_scale_info);
	H -> scale -> scale		= 0;
	H -> scale -> scale_mul		= 1.0;
	H -> scale -> scale_div		= 1.0;
	H -> scale -> min_precision	= DBL_DIG + 1;

	H -> proplist = gst_create_proplist (NULL);

	H -> version = 0;
	H -> requested_version = 0;

	GST_POSTLUDE
	return H;
}

/*
 * Free all of the vertices of the given hypergraph.
 */

	static
	void
free_vertices (

gst_hg_ptr		H
)
{
	H -> num_verts		= 0;
	H -> num_vert_masks	= 0;

	if (H -> tflag NE NULL) {
		free ((char *) (H -> tflag));
		H -> tflag = NULL;
	}
	if (H -> pts NE NULL) {
		free ((char *) (H -> pts));
		H -> pts = NULL;
	}
	if (H -> initial_vert_mask NE NULL) {
		free ((char *) (H -> initial_vert_mask));
		H -> initial_vert_mask = NULL;
	}
}

/*
 * Free a single FST.
 */

	void
_gst_free_full_set (

struct full_set *	fsp	/* IN - full tree to be freed */
)
{
	if (fsp -> tlist NE NULL) {
		free ((char *) (fsp -> tlist));
		fsp -> tlist = NULL;
	}
	if (fsp -> terminals NE NULL) {
		free ((char *) (fsp -> terminals));
		fsp -> terminals = NULL;
	}
	if (fsp -> steiners NE NULL) {
		free ((char *) (fsp -> steiners));
		fsp -> steiners = NULL;
	}
	if (fsp -> edges NE NULL) {
		free ((char *) (fsp -> edges));
		fsp -> edges = NULL;
	}
	free ((char *) fsp);
}

/*
 * Free all of the edges of the given hypergraph.
 */

	static
	void
free_edges (

gst_hg_ptr		H
)
{
int			i;
int			nedges;

	nedges	= H -> num_edges;

	H -> num_edges		= 0;
	H -> num_edge_masks	= 0;

	if (H -> full_trees NE NULL) {
		for (i = 0; i < nedges; i++) {
			_gst_free_full_set (H -> full_trees [i]);
			H -> full_trees [i] = NULL;
		}
		free ((char *) (H -> full_trees));
		H -> full_trees = NULL;
	}
	if (H -> edge NE NULL) {
		if (H -> edge [0] NE NULL) {
			free ((char *) (H -> edge [0]));
		}
		free ((char *) (H -> edge));
		H -> edge = NULL;
	}
	if (H -> edge_size NE NULL) {
		free ((char *) (H -> edge_size));
		H -> edge_size = NULL;
	}
	if (H -> initial_edge_mask NE NULL) {
		free ((char *) (H -> initial_edge_mask));
		H -> initial_edge_mask = NULL;
	}
	if (H -> required_edges NE NULL) {
		free ((char *) (H -> required_edges));
		H -> required_edges = NULL;
	}
	if (H -> cost NE NULL) {
		free ((char *) (H -> cost));
		H -> cost = NULL;
	}
	if (H -> cost_extension NE NULL) {
		/* Use "virtual destructor" to free these. */
		_gst_hg_cost_extension_free (H -> cost_extension);
		H -> cost_extension = NULL;
	}
	if (H -> inc_edges NE NULL) {
		if (H -> inc_edges [0] NE NULL) {
			free ((char *) (H -> inc_edges [0]));
		}
		free ((char *) (H -> inc_edges));
		H -> inc_edges = NULL;
	}

	if (H -> term_trees NE NULL) {
		if (H -> term_trees [0] NE NULL) {
			free ((char *) (H -> term_trees [0]));
		}
		free ((char *) (H -> term_trees));
		H -> term_trees = NULL;
	}
}

/*
 * Freeing a hypergraph and all of its content.
 */

	int
gst_free_hg (

gst_hg_ptr		H
)
{
	GST_PRELUDE

	if (H NE NULL) {
		if (H -> metric NE NULL) {
			gst_free_metric (H -> metric);
			H -> metric = NULL;
		}

		if (H -> scale NE NULL) {
			free (H -> scale);
			H -> scale = NULL;
		}

		if (H -> proplist NE NULL) {
			gst_free_proplist (H -> proplist);
			H -> proplist = NULL;
		}

		free_vertices (H);
		free_edges (H);

		free (H);
	}

	GST_POSTLUDE
	return 0;
}

/*
 * Copy a hypergraph. Note that the destination must already exist.
 */

	int
gst_copy_hg (

gst_hg_ptr		dst,	/* IN/OUT - Destination hypergraph */
gst_hg_ptr		H	/* IN - Source hypergraph */
)
{
int			i;
int			nverts;
int			nedges;
int			kmasks;
int			nmasks;
int			status;
struct full_set *	fst;
struct full_set *	dstfst;

	GST_PRELUDE

	status = 0;

	do {	/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		if (dst EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}
		if (dst EQ H) {
			/* Don't destroy source if copying to self! */
			break;
		}

		nverts = H -> num_verts;
		nedges = H -> num_edges;
		kmasks = H -> num_vert_masks;
		nmasks = H -> num_edge_masks;

		i = gst_set_hg_number_of_vertices (dst, nverts);
		if ((i NE 0) AND (status EQ 0)) {
			status = i;
		}
		i = gst_set_hg_edges (dst,
				      nedges,
				      H -> edge_size,
				      H -> edge [0],
				      H -> cost);
		if ((i NE 0) AND (status EQ 0)) {
			status = i;
		}

		i = gst_copy_metric (dst -> metric, H -> metric);
		if ((i NE 0) AND (status EQ 0)) {
			status = i;
		}
		dst -> scale -> scale		= H -> scale -> scale;
		dst -> scale -> scale_mul	= H -> scale -> scale_mul;
		dst -> scale -> scale_div	= H -> scale -> scale_div;
		dst -> scale -> min_precision	= H -> scale -> min_precision;

		if (nverts > 0) {
			memcpy (dst -> tflag,
				H -> tflag,
				nverts * sizeof (bool));
			memcpy (dst -> initial_vert_mask,
				H -> initial_vert_mask,
				kmasks * sizeof (bitmap_t));
		}
		if (nmasks > 0) {
			memcpy (dst -> initial_edge_mask,
				H -> initial_edge_mask,
				nmasks * sizeof (bitmap_t));
			memcpy (dst -> required_edges,
				H -> required_edges,
				nmasks * sizeof (bitmap_t));
		}

		/* inc_edges is not copied. */

		/* FIXME (DMW) - Why are the inc_edges not copied?  Is	*/
		/* there some good reason why they should *NOT* be	*/
		/* copied?						*/

		/* Copy the property list */
		i = gst_copy_proplist (dst -> proplist, H -> proplist);
		if ((i NE 0) AND (status EQ 0)) {
			status = i;
		}

		/* Geometric information */
		if (H -> pts NE NULL) {
			dst -> pts = NEW_PSET (nverts);
			COPY_PSET(dst -> pts, H -> pts);
		}

		if (H -> full_trees NE NULL) {
			dst -> full_trees = NEWA (nedges, struct full_set *);

			for (i = 0; i < nedges; i++) {
				/* Copy structure */
				fst	= H -> full_trees[i];

				dstfst	= NEW (struct full_set);

				dstfst -> next = NULL;
				dstfst -> tree_num = fst -> tree_num;
				dstfst -> tree_len = fst -> tree_len;
				dstfst -> tlist = NEWA (fst -> terminals -> n,
							int);
				memcpy (dstfst -> tlist,
					fst -> tlist,
					fst -> terminals -> n * sizeof (int));
				dstfst -> terminals = NEW_PSET (fst -> terminals -> n);
				COPY_PSET(dstfst -> terminals,
					  fst -> terminals);
				if (fst -> steiners NE NULL) {
					dstfst -> steiners = NEW_PSET (fst -> steiners -> n);
					COPY_PSET (dstfst -> steiners,
						   fst -> steiners);
				}
				dstfst -> nedges = fst -> nedges;
				dstfst -> edges = NEWA (fst -> nedges,
							struct edge);
				memcpy (dstfst -> edges,
					fst -> edges,
					fst -> nedges * sizeof(struct edge));

				dst -> full_trees[i] = dstfst;
			}
		}
	} while (FALSE);

	GST_POSTLUDE
	return (status);
}

/*
 * Make a copy of a hypergraph with a subset of the original edges.
 */

	int
gst_copy_hg_edges (

gst_hg_ptr		dst,	/* IN/OUT - destination hypergraph */
gst_hg_ptr		H,	/* IN - source hypergraph */
int			nedges, /* IN - number of edges to copy */
int *			edges	/* IN - index numbers of edges to copy */
)
{
int			i;
int			j;
int			k;
int			count;
int			status;
int *			dst_edge_sizes;
int *			dst_edges;
int *			ep;
int *			ep2;
double *		dst_weights;
struct full_set **	fsts;
struct full_set *	fst;
struct full_set *	dstfst;

	GST_PRELUDE

	status = 0;

	dst_weights	= NULL;
	dst_edge_sizes	= NULL;
	dst_edges	= NULL;

	do {	/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		if (dst EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		if ((nedges < 0) OR (nedges > H -> num_edges)) {
			status = GST_ERR_INVALID_NUMBER_OF_EDGES;
			break;
		}

		if (dst EQ H) {
			/* FIXME - This should truncate and	*/
			/* renumber the edges in place!		*/
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		/* Verify that the user isn't asking for bogus edges... */
		for (i = 0; i < nedges; i++) {
			if ((edges [i] < 0) OR (edges [i] >= H -> num_edges)) {
				status = GST_ERR_INVALID_EDGE;
				break;
			}
		}
		if (status NE 0) break;

		/* An easy, if inefficient approach -- copy everything	*/
		/* except the FSTs (edge embeddings)...			*/

		fsts = H -> full_trees;
		H -> full_trees = NULL;
		i = gst_copy_hg (dst, H);
		H -> full_trees = fsts;

		if (i NE 0) {
			status = i;
			/* Since we have already clobbered dst, we	*/
			/* might as well continue on...			*/
		}

		count = 0;
		for (i = 0; i < nedges; i++) {
			j = edges [i];
			count += H -> edge_size [j];
		}

		dst_weights	= NEWA (nedges, double);
		dst_edge_sizes	= NEWA (nedges, int);
		dst_edges	= NEWA (count, int);

		ep = dst_edges;
		for (i = 0; i < nedges; i++) {
			j = edges [i];
			k = H -> edge_size [j];
			dst_weights [i]	   = H -> cost [j];
			dst_edge_sizes [i] = k;
			ep2 = H -> edge [j];
			for (j = 0; j < k; j++) {
				*ep++	= *ep2++;
			}
		}

		i = gst_set_hg_edges (dst,
				      nedges,
				      dst_edge_sizes,
				      dst_edges,
				      dst_weights);
		if ((i NE 0) AND (status EQ 0)) {
			status = i;
		}

		if (H -> full_trees NE NULL) {
			dst -> full_trees	= NEWA (nedges,
							struct full_set *);

			for (i = 0; i < nedges; i++) {
				/* Copy structure */
				j = edges [i];
				fst	= H -> full_trees [j];
				dstfst	= NEW (struct full_set);

				dstfst -> next = NULL;
				dstfst -> tree_num = i;
				dstfst -> tree_len = fst -> tree_len;
				dstfst -> tlist = NEWA (fst -> terminals -> n,
							int);
				memcpy (dstfst -> tlist,
					fst -> tlist,
					fst -> terminals -> n * sizeof (int));
				dstfst -> terminals = NEW_PSET (fst -> terminals -> n);
				COPY_PSET (dstfst -> terminals,
					   fst -> terminals);
				dstfst -> steiners = NULL;
				if (fst -> steiners NE NULL) {
					dstfst -> steiners = NEW_PSET (fst -> steiners -> n);
					COPY_PSET (dstfst -> steiners,
						   fst -> steiners);
				}
				dstfst -> nedges = fst -> nedges;
				dstfst -> edges	 = NEWA (fst -> nedges,
							 struct edge);
				memcpy (dstfst -> edges,
					fst -> edges,
					fst -> nedges * sizeof (struct edge));

				dst -> full_trees [i] = dstfst;
			}
		}
	} while (FALSE);

	if (dst_weights NE NULL) {
		free (dst_weights);
	}
	if (dst_edge_sizes NE NULL) {
		free (dst_edge_sizes);
	}
	if (dst_edges NE NULL) {
		free (dst_edges);
	}

	GST_POSTLUDE
	return (status);
}

/*
 * Copy a full set while removing any degree 2 Steiner points
 * (corner points..).
 *
 * This function might seem a bit complicated but it is necessary to keep the
 * running time linear. It can handle edges with an arbitrary number of
 * corner points.
 */

struct stpinfo {
	int degree;	/* Steiner point degree */
	int number;	/* New Steiner point index */
	int alias;	/* Reference to another Steiner point or terminal to
			   which this Steiner point is connected through
			   degree 2 Steiner points */
	double length;	/* The distance to the alias */
};

	struct full_set *
_gst_remove_degree_two_steiner_points (

struct full_set *	ofsp	/* IN - old full set */
)
{
int			i;
int			p1;
int			p2;
int			tmp;
int			nterms;
int			old_nedges;
int			old_nsteins;
int			new_nedges;
int			new_nsteins;
struct edge *		old_ep;
struct edge *		new_ep;
struct point *		old_pt;
struct point *		new_pt;
struct stpinfo *	info;
struct full_set *	nfsp;

	/* Initialize a new full set and copy the basic information */
	nfsp	= NEW (struct full_set);
	memset (nfsp, 0, sizeof (struct full_set));
	nfsp -> next = NULL; /* !!!!!!!! */
	nfsp -> tree_num = ofsp -> tree_num;
	nfsp -> tree_len = ofsp -> tree_len;

	nfsp -> tlist = NEWA (ofsp -> terminals -> n, int);
	memcpy (nfsp -> tlist,
		ofsp -> tlist,
		ofsp -> terminals -> n * sizeof (int));

	nfsp -> terminals = NEW_PSET (ofsp -> terminals -> n);
	COPY_PSET(nfsp -> terminals, ofsp -> terminals);

	if (ofsp -> steiners EQ NULL) {
		/* No Steiner points */
		nfsp -> steiners = NULL;
		nfsp -> nedges	= ofsp -> nedges;
		nfsp -> edges	= NEWA (ofsp -> nedges, struct edge);
		memcpy (nfsp -> edges,
			ofsp -> edges,
			ofsp -> nedges * sizeof(struct edge));
		return (nfsp);
	}

	nterms = ofsp -> terminals -> n;
	old_nsteins = ofsp -> steiners -> n;
	old_nedges = ofsp -> nedges;

	info = NEWA (nterms + old_nsteins, struct stpinfo);
	memset (info, 0, (nterms + old_nsteins) * sizeof (info [0]));

	/* Find Steiner point degrees */
	old_ep = ofsp -> edges;
	for (i = 0; i < old_nedges; i++, old_ep++) {
		if (old_ep -> p1 >= nterms) { /* Steiner point */
			info [old_ep -> p1].degree++;
		}
		if (old_ep -> p2 >= nterms) { /* Steiner point */
			info [old_ep -> p2].degree++;
		}
	}

	new_nsteins = 0;
	for (i = nterms; i < nterms + old_nsteins; i++) {
		info [i].alias = -1;
		if (info [i].degree > 2) {
			++new_nsteins;
		}
	}
	new_nedges = new_nsteins + nterms - 1;

	/* Allocate space for Steiner points and edges */
	nfsp -> steiners	= NEW_PSET (new_nsteins);
	nfsp -> steiners -> n	= new_nsteins;
	nfsp -> edges		= NEWA (new_nedges, struct edge);
	nfsp -> nedges		= new_nedges;

	/* Copy Steiner points of degree higher than 2 */
	new_nsteins = 0;
	new_pt = &(nfsp -> steiners -> a [0]);
	old_pt = &(ofsp -> steiners -> a [0]);
	for (i = nterms; i < nterms + old_nsteins; i++) {
		if (info [i].degree > 2) {
			info [i].number = nterms + new_nsteins;
			new_pt -> x = old_pt -> x;
			new_pt -> y = old_pt -> y ;

			++new_nsteins;
			++new_pt;
		}
		++old_pt;
	}

	/* Copy edges -- this is the hard part */
	old_ep = ofsp -> edges;
	new_ep = nfsp -> edges;
	for (i = 0; i < old_nedges; i++, old_ep++) {
		p1 = old_ep -> p1;
		p2 = old_ep -> p2;
		new_ep -> len = 0;

		if (p1 >= nterms) { /* p1 is a Steiner point */
			if ((info [p1].degree EQ 2) AND
			    (info [p1].alias NE -1)) {
				new_ep -> len += info [p1].length;
				p1 = info [p1].alias;
			}
		}

		if (p2 >= nterms) { /* p2 is a Steiner point */
			if ((info [p2].degree EQ 2) AND
			    (info [p2].alias NE -1)) {
				new_ep -> len += info [p2].length;
				p2 = info [p2].alias;
			}
		}

		if (p1 > p2) { /* Swap the points to simplify the following */
			tmp = p1; p1 = p2; p2 = tmp;
		}

		if (p1 < nterms) { /* p1 is a terminal */
			if (p2 < nterms) { /* p2 is a terminal */
				new_ep -> p1   = p1;
				new_ep -> p2   = p2;
				new_ep -> len += old_ep -> len;
				++new_ep;
			}
			else { /* p2 is a Steiner point */
				if (info [p2].degree > 2) {
					new_ep -> p1   = p1;
					new_ep -> p2   = info [p2].number;
					new_ep -> len += old_ep -> len;
					++new_ep;
				}
				else { /* p2 is a degree 2 Steiner point */
					info [p2].alias = p1;
					info [p2].length = new_ep -> len + old_ep -> len;
				}
			}
		}
		else { /* p1 AND p2 are Steiner points */
			if ((info [p1].degree > 2) AND (info [p2].degree > 2)) {
				new_ep -> p1   = info [p1].number;
				new_ep -> p2   = info [p2].number;
				new_ep -> len += old_ep -> len;
				++new_ep;
			}

			if (info [p1].degree EQ 2) {
					info [p1].alias = p2;
					info [p1].length = new_ep -> len + old_ep -> len;
			}

			if (info [p2].degree EQ 2) {
					info [p2].alias = p1;
					info [p2].length = new_ep -> len + old_ep -> len;
			}
		}
	}

#if 0
	printf ("FST dump:\n");
	printf ("Terminals:\n");
	for (i = 0; i < nterms; i++) {
		struct point *new_pt = &(ofsp -> terminals -> a [i]);
		printf ("(%f, %f) - %d\n",
			new_pt -> x, new_pt -> y, ofsp -> tlist [i]);
	}
	printf ("Steiner points:\n");
	for (i = 0; i < old_nsteins; i++) {
		struct point *new_pt = &(ofsp -> steiners -> a [i]);
		printf ("(%f, %f)\n", new_pt -> x, new_pt -> y);
	}
	printf ("-->\n");
	printf ("Terminals:\n");
	for (i = 0; i < nterms; i++) {
		struct point *new_pt = &(nfsp -> terminals -> a [i]);
		printf ("(%f, %f) - %d\n",
			new_pt -> x, new_pt->y, nfsp -> tlist [i]);
	}
	printf ("Steiner points:\n");
	for (i = 0; i < new_nsteins; i++) {
		struct point *new_pt = &(nfsp -> steiners -> a [i]);
		printf ("(%f, %f)\n", new_pt -> x, new_pt -> y);
	}
	printf ("----------------\n");
	printf ("Edges:\n");
	for (i = 0; i < old_nedges; i++) {
		struct edge *ep = &(ofsp -> edges [i]);
		printf ("(%d, %d) - %f\n", ep -> p1, ep -> p2, ep -> len);
	}
	printf ("-->\n");
	for (i = 0; i < new_nedges; i++) {
		struct edge *ep = &(nfsp -> edges [i]);
		printf ("(%d, %d) - %f\n", ep -> p1, ep -> p2, ep -> len);
	}
	printf ("Length: %f\n", nfsp -> tree_len);
#endif

	free (info);

	return (nfsp);
}

/*
 * Set the number of vertices in a hypergraph.
 */

	int
gst_set_hg_number_of_vertices (

gst_hg_ptr	H,		/* IN - hypergraph */
int		nvertices	/* IN - number of vertices */
)
{
int		i;
int		kmasks;
int		status;

	GST_PRELUDE

	status = 0;

	do {		/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		if (nvertices < 0) {
			status = GST_ERR_INVALID_NUMBER_OF_VERTICES;
			break;
		}

		/* Always free the edges too.  This avoids invalid	*/
		/* edges if the number of vertices decreased.		*/

		free_vertices (H);
		free_edges (H);

		bump_version (H);

		kmasks = BMAP_ELTS (nvertices);
		H -> num_verts		= nvertices;
		H -> num_vert_masks	= kmasks;

		if (nvertices EQ 0) break;

		H -> tflag	= NEWA (nvertices, bool);
		for (i = 0; i < nvertices; i++) {
			H -> tflag [i] = TRUE;
		}

		H -> initial_vert_mask = NEWA (kmasks, bitmap_t);
		memset (H -> initial_vert_mask, 0, kmasks * sizeof (bitmap_t));
		for (i = 0; i < nvertices; i++) {
			SETBIT (H -> initial_vert_mask, i);
		}

	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * This function can set all edge information. If no weights are provided
 * then all weights will be set to 1.
 */
	int
gst_set_hg_edges (

gst_hg_ptr	H,		/* IN/OUT - hypergraph */
int		nedges,		/* IN - number of edges */
int *		edge_sizes,	/* IN - number of vertices for each edge */
int *		edges,		/* IN - vertex indices of each edge */
double *	weights		/* IN - edge weights */
)
{
int		i;
int		j;
int		k;
int		nverts;
int		count;
int		nmasks;
int		status;
int		esize;
int *		ip1;
dist_t *	cost;
bool *		vflags;

	GST_PRELUDE

	status = 0;

	do {		/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		/* Verify the edges given. */

		nverts = H -> num_verts;

		vflags = NEWA (nverts, bool);
		memset (vflags, 0, nverts * sizeof (vflags [0]));

		ip1 = edges;
		count = 0;
		for (i = 0; i < nedges; i++) {
			esize = edge_sizes [i];
			if ((esize < 2) OR (esize > nverts)) {
				status = GST_ERR_INVALID_EDGE;
				break;
			}
			for (j = 0; j < esize; j++) {
				k = ip1 [j];
				if ((k < 0) OR (k >= nverts)) {
					status = GST_ERR_INVALID_EDGE;
					break;
				}
				if (vflags [k]) {
					/* Vertex listed more than once	*/
					/* in a single edge!		*/
					status = GST_ERR_INVALID_EDGE;
					break;
				}
				vflags [k] = TRUE;
			}
			if (status NE 0) break;
			for (j = 0; j < esize; j++) {
				k = ip1 [j];
				vflags [k] = FALSE;
			}
			count += esize;
			ip1 += esize;
		}
		free (vflags);
		if (status NE 0) {
			/* User botched the edges somehow. */
			break;
		}

		free_edges (H);

		bump_version (H);

		nmasks = BMAP_ELTS (nedges);

		H -> num_edges		= nedges;
		H -> num_edge_masks	= nmasks;
		H -> edge		= NEWA (nedges + 1, int *);
		H -> edge_size		= NEWA (nedges, int);

		ip1 = NEWA (count, int);
		memcpy (ip1, edges, count * sizeof(int));
		for (i = 0; i < nedges; i++) {
			k = edge_sizes [i];
			H -> edge [i]	   = ip1;
			H -> edge_size [i] = k;
			ip1 += k;
		}
		H -> edge [i] = ip1;

		H -> initial_edge_mask = NEWA (nmasks, bitmap_t);
		memset (H -> initial_edge_mask, 0, nmasks * sizeof (bitmap_t));
		for (i = 0; i < nedges; i++) {
			SETBIT (H -> initial_edge_mask, i);
		}

		H -> required_edges = NEWA (nmasks, bitmap_t);
		memset (H -> required_edges, 0, nmasks * sizeof (bitmap_t));

		_gst_init_term_trees (H);

		/* H -> inc_edges = NULL; */

		cost = NEWA (nedges, dist_t);

		if (weights EQ NULL) {
			for (i = 0; i < nedges; i++) {
				*cost++ = 1;
			}
		}
		else {
			memcpy (cost, weights, nedges * sizeof (dist_t));
		}
		H -> cost = cost;

	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * This function sets all edge weights.	 If weights is a NULL pointer
 * then all weights will be set to 1.
 */

	int
gst_set_hg_edge_weights (

gst_hg_ptr	H,		/* IN/OUT - hypergraph */
double *	weights		/* IN - edge weights */
)
{
int		i;
int		nedges;
int		status;
dist_t *	cost;

	GST_PRELUDE

	status = 0;

	do {
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		nedges = H -> num_edges;

		cost = H -> cost;
		if ((cost EQ NULL) AND (nedges > 0)) {
			cost = NEWA (nedges, dist_t);
			H -> cost = cost;
		}
		if (weights EQ NULL) {
			for (i = 0; i < nedges; i++) {
				*cost++ = 1.0;
			}
		}
		else {
			memcpy (cost, weights, nedges * sizeof (dist_t));
		}

		bump_version (H);

	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * The embedding of all vertices can be set with this function. Currently
 * it can only deal with 2 dimensions.
 */

	int
gst_set_hg_vertex_embedding (

gst_hg_ptr		H,
int			dim,
double *		coords
)
{
int		i;
int		n;
int		status;
struct point *	pt;

	GST_PRELUDE

	status = 0;

	do {		/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		if (dim NE 2) {
			status = GST_ERR_INVALID_DIMENSION;
			break;
		}

		n = H -> num_verts;

		if (H -> pts NE NULL) {
			free ((char *) (H -> pts));
			H -> pts = NULL;
		}

		bump_version (H);

		if (coords EQ NULL) {
			/* Passing NULL simply removes the current	*/
			/* vertex embedding.				*/
			break;
		}

		if (n > 0) {
			H -> pts = NEW_PSET (n);
			pt = &(H -> pts -> a [0]);
			for (i = 0; i < n; i++) {
				pt -> x = *coords++;
				pt -> y = *coords++;
				++pt;
			}
		}
	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * Get the number of vertices in a hypergraph.
 */

	int
gst_get_hg_number_of_vertices (

gst_hg_ptr	H		/* IN - hypergraph */
)
{
int	nverts;

	GST_PRELUDE

	nverts = -1;
	if (H NE NULL) {
		nverts = H -> num_verts;
		if (nverts < 0) {
			nverts = -1;
		}
	}

	GST_POSTLUDE

	return (nverts);
}

/*
 * Get information about a single hyperedge. Weight and vertices.
 */

	int
gst_get_hg_one_edge (

gst_hg_ptr	H,		/* IN - hypergraph */
int		number,		/* IN - hyperedge number */
double *	weight,		/* OUT - hyperedge weight */
int *		nverts,		/* OUT - number of vertices in hyperedge */
int *		verts		/* OUT - vertex indices */
)
{
int		nedges;
int		status;

	GST_PRELUDE

	status = 0;

	do {		/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		nedges = H -> num_edges;

		if ((number < 0) OR (number >= nedges)) {
			status = GST_ERR_INVALID_EDGE;
			break;
		}

		if (weight NE NULL) {
			*weight = H -> cost [number];
		}

		if (nverts NE NULL) {
			*nverts = H -> edge_size [number];
		}

		if (verts NE NULL) {
			memcpy (verts,
				H -> edge [number],
				H -> edge_size [number] * sizeof (int));
		}
	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * This function returns the basic edge information.  If any of the three
 * pointers is NULL then the corresponding information is not returned.
 */

	int
gst_get_hg_edges (

gst_hg_ptr	H,		/* IN - hypergraph */
int *		num_edges,	/* OUT - number of edges */
int *		edge_sizes,	/* OUT - number of vertices for each edge */
int *		edges,		/* OUT - vertex indices of each edge */
double *	weights		/* OUT - edge weights */
)
{
int		n;
int		nedges;
int		status;

	GST_PRELUDE

	status = 0;

	do {		/* Used only for "break". */

		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		nedges = H -> num_edges;

		if (num_edges NE NULL) {
			*num_edges = nedges;
		}

		if (edge_sizes NE NULL) {
			if ((H -> edge_size EQ NULL) AND (nedges > 0)) {
				FATAL_ERROR;
			}

			memcpy (edge_sizes,
				H -> edge_size,
				nedges * sizeof (int));
		}

		if (edges NE NULL) {
			if ((H -> edge EQ NULL) AND (nedges > 0)) {
				FATAL_ERROR;
			}

			n = H -> edge [nedges] - H -> edge [0];
			memcpy (edges, H -> edge [0], n * sizeof (int));
		}

		if (weights NE NULL) {
			if ((H -> cost EQ NULL) AND (nedges > 0)) {
				FATAL_ERROR;
			}

			memcpy (weights, H -> cost, nedges * sizeof (dist_t));
		}
	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * The embedding of all vertices can be obtained with this function. Currently
 * it can only deal with 2 dimensions.
 */

	int
gst_get_hg_vertex_embedding (

gst_hg_ptr		H,
int *			dim,
double *		coords
)
{
int		i;
int		n;
int		status;
struct point *	pt;

	GST_PRELUDE

	status = 0;

	do {		/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		if (H -> pts EQ NULL) {
			status = GST_ERR_NO_EMBEDDING;
			if (dim NE NULL) {
				*dim = 0;
			}
			break;
		}

		if (dim NE NULL) {
			*dim = 2;
		}

		if (coords NE NULL) {
			n = H -> pts -> n;
			pt = &H -> pts -> a[0];
			for (i = 0; i < n; i++) {
				*coords++ = pt -> x;
				*coords++ = pt -> y;
				++pt;
			}
		}
	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * Get the coordinates for a single vertex.
 */

	int
gst_get_hg_one_vertex_embedding (

gst_hg_ptr		H,		/* IN - hypergraph */
int			num_vert,	/* IN - vertex number */
double *		coords		/* OUT - coordinates */
)
{
int			status;
struct point *		pt;

	GST_PRELUDE

	status = 0;

	do {		/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		if ((num_vert < 0) OR (num_vert > H -> num_verts)) {
			status = GST_ERR_INVALID_VERTEX;
			break;
		}

		if (H -> pts EQ NULL) {
			status = GST_ERR_NO_EMBEDDING;
			break;
		}

		if (coords NE NULL) {
			pt = &(H -> pts -> a [num_vert]);
			*coords++ = pt[num_vert].x;
			*coords++ = pt[num_vert].y;
		}
	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * The embedding of a single hyperedge can be obtained with this function.
 * This is especially useful when wanting to draw a given hyperedge (fst).
 * Currently it can only deal with 2 dimensions.
 */

	int
gst_get_hg_one_edge_embedding (

gst_hg_ptr		H,		/* IN - hypergraph */
int			edge_number,	/* IN - hyperedge number */
int *			nsps,		/* OUT - number of Steiner points */
double *		sps,		/* OUT - Steiner point coordinates */
int *			nedges_out,	/* OUT - number of embedded edges */
int *			edges		/* OUT - indices for edge endpoints */
)
{
int			i;
int			nedges;
int			status;
struct point *		p;
struct edge *		ep;
struct full_set *	fst;
struct pset *		steiners;

	GST_PRELUDE

	status = 0;

	do {		/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		nedges = H -> num_edges;

		if ((edge_number < 0) OR (edge_number >= nedges)) {
			status = GST_ERR_INVALID_EDGE;
			break;
		}

		if (H -> full_trees EQ NULL) {
			status = GST_ERR_NO_EMBEDDING;
			break;
		}

		fst	  = H -> full_trees [edge_number];
		steiners  = fst -> steiners;

		if (steiners EQ NULL) {
			if (nsps NE NULL) {
				*nsps = 0;
			}
		}
		else {
			if (nsps NE NULL) {
				*nsps = steiners -> n;
			}
			if (sps NE NULL) {
				p = &(steiners -> a [0]);
				for (i = 0; i < steiners -> n; i++) {
					*sps++ = p -> x;
					*sps++ = p -> y;
					++p;
				}
			}
		}

		if (nedges_out NE NULL) {
			*nedges_out = fst -> nedges;
		}

		if (edges NE NULL) {
			ep = fst -> edges;
			for (i = 0; i < fst -> nedges; i++) {
				*edges++ = ep -> p1;
				*edges++ = ep -> p2;
				++ep;
			}
		}

	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * The embedding of a list of hyperedges can be obtained with this function.
 * Currently it can only deal with 2 dimensions.
 */

	int
gst_get_hg_edge_embedding (

gst_hg_ptr		H,		/* IN - hypergraph */
int			nfsts,		/* IN - number of wanted fsts */
int *			fsts,		/* IN - list of indices of wanted fsts */
int *			nsps_out,	/* OUT - number of Steiner points */
double *		sps_out,	/* OUT - Steiner point coordinates */
int *			nedges_out,	/* OUT - number of embedded edges */
int *			edges_out	/* OUT - indices for edge endpoints */
)
{
int			i;
int			j;
int			e;
int			ep1;
int			ep2;
int			sp_index;
int			nedges;
int			ntterms;
int			sp_count;
int			status;
struct point *		p;
int *			tlist;
int *			fstlist;
struct pset *		stmp;
struct pset *		terms;
struct full_set *	fst;

	GST_PRELUDE

	status = 0;

	fstlist = NULL;
	sp_count = 0;

	if (nsps_out NE NULL) {
		*nsps_out = 0;
	}
	if (nedges_out NE NULL) {
		*nedges_out = 0;
	}

	do {		/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		if (H -> full_trees EQ NULL) {
			status = GST_ERR_NO_EMBEDDING;
			break;
		}

		if ((H -> metric -> type NE GST_METRIC_L) AND
		    (H -> metric -> type NE GST_METRIC_UNIFORM)) {
			status = GST_ERR_NO_EMBEDDING;
			break;
		}

		nedges = H -> num_edges;

		if (fsts NE NULL) {
			if (nfsts <= 0) {
				status = GST_ERR_NO_EMBEDDING;
				break;
			}
			/* Verify edges are in range. */
			for (i = 0; i < nfsts; i++) {
				j = fsts [i];
				if ((j < 0) OR (j >= nedges)) {
					status = GST_ERR_INVALID_EDGE;
					break;
				}
			}
			if (status NE 0) break;
			fstlist = NEWA (nfsts, int);
			memcpy (fstlist, fsts, nfsts * sizeof (int));
		}
		else {
			if ((nfsts < 0) OR (nfsts > nedges)) {
				status = GST_ERR_NO_EMBEDDING;
				break;
			}
			if (nfsts EQ 0) {
				nfsts = nedges;
			}
			fstlist = NEWA (nfsts, int);
			for (i = 0; i < nfsts; i++) {
				fstlist [i] = i;
			}
		}

		for (i = 0; i < nfsts; i++) {
			e = fstlist [i];
			FATAL_ERROR_IF ((e < 0) OR (e >= nedges));

			fst = H -> full_trees [e];

			tlist	= fst -> tlist;
			stmp	= fst -> steiners;
			terms	= fst -> terminals;
			ntterms = terms -> n;

			/* Any Steiner points in the FST? */
			sp_index = H -> num_verts + sp_count - ntterms;
			if (stmp NE NULL) {
				sp_count += stmp -> n;

				if (sps_out NE NULL) {
					p = &(stmp -> a [0]);
					for (j = 0; j < stmp -> n; j++) {
						*sps_out++ = p -> x;
						*sps_out++ = p -> y;
						++p;
					}
				}
			}

			if (nedges_out NE NULL) {
				*nedges_out += fst -> nedges;
			}

			if (edges_out NE NULL) {
				for (j = 0; j < fst -> nedges; j++) {
					ep1 = fst -> edges [j].p1;
					ep2 = fst -> edges [j].p2;

					if (ep1 < ntterms) {
						/* terminal */
						ep1 = tlist [ep1];
					}
					else {
						/* Steiner point */
						ep1 = sp_index + ep1;
					}

					if (ep2 < ntterms) {
						/* terminal */
						ep2 = tlist [ep2];
					}
					else {
						/* Steiner point */
						ep2 = sp_index + ep2;
					}

					*edges_out++ = ep1;
					*edges_out++ = ep2;
				}
			}
		}
		if (nsps_out NE NULL) {
			*nsps_out = sp_count;
		}
	} while (FALSE);

	if (fstlist NE NULL) {
		free (fstlist);
	}

	GST_POSTLUDE

	return (status);
}

/*
 * Get the terminals in a hypergraph.
 */

	int
gst_get_hg_terminals (

gst_hg_ptr		H,		/* IN - hypergraph */
int *			nterms,		/* OUT - number of terminals */
int *			terms		/* OUT - terminal indices... */
)
{
int		i, nverts, tcount;
int		status;
bool *		tflag;

	GST_PRELUDE

	do {		/* Used only for "break". */

		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		nverts = H -> num_verts;
		tflag  = H -> tflag;
		if (tflag EQ NULL) {
			/* All vertices are terminals. */
			tcount = H -> num_verts;
		}
		else {
			tcount = 0;
			for (i = 0; i < nverts; i++) {
				if (tflag [i]) {
					++tcount;
				}
			}
		}

		if (nterms NE NULL) {
			*nterms = tcount;
		}

		if (terms NE NULL) {
			if (tflag EQ NULL) {
				/* All vertices are terminals. */
				for (i = 0; i < nverts; i++) {
					*terms++ = i;
				}
			}
			else {
				for (i = 0; i < nverts; i++) {
					if (tflag [i]) {
						*terms++ = i;
					}
				}
			}
		}

		status = 0;

	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * Get the metric for a hypergraph if available.
 */

	int
gst_get_hg_metric (

gst_hg_ptr		H,		/* IN - hypergraph */
gst_metric_ptr *	metric		/* OUT - metric */
)
{
int		status;

	GST_PRELUDE

	status = 0;

	do {		/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		if (metric NE NULL) {
			*metric = H -> metric;
		}
	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * Set a metric for a hypergraph. A NULL pointer sets the default (NONE, 0).
 */

	int
gst_set_hg_metric (

gst_hg_ptr		H,		/* IN - hypergraph */
gst_metric_ptr		metric		/* IN - metric */
)
{
int		status;

	GST_PRELUDE

	status = 0;

	do {		/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		status = gst_copy_metric (H -> metric, metric);

		bump_version (H);

	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * This routine returns a pointer to the hypergraph's property list.
 */

	gst_proplist_ptr
gst_get_hg_properties (

gst_hg_ptr		H	/* IN - hypergraph */
)
{
gst_proplist_ptr	plist;

	GST_PRELUDE

	plist = NULL;
	if (H NE NULL) {
		plist = H -> proplist;
	}

	GST_POSTLUDE

	return (plist);
}

/*
 * Get the edge status for single hyperedge.
 */
	int
gst_get_hg_edge_status (

gst_hg_ptr		H,		/* IN - hypergraph */
int			edge_number,	/* IN - edge number */
int *			unneeded,	/* OUT - is the edge pruned/deleted */
int *			required	/* OUT - is the edge required */
)
{
int		status;

	GST_PRELUDE

	status = 0;

	do {		/* Used only for "break". */
		if (H EQ NULL) {
			status = GST_ERR_INVALID_HYPERGRAPH;
			break;
		}

		if (unneeded NE NULL) {
			*unneeded = NOT BITON (H -> initial_edge_mask,
					       edge_number);
		}

		if (required NE NULL) {
			*required = BITON (H -> required_edges, edge_number);
		}
	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * Set the type of each vertex in the hypergraph.  A type of 'T'
 * indicates that the vertex is a terminal.  A type of 'S' indicates
 * that the vertex is a Steiner vertex.  All other types are illegal.
 */

	int
gst_set_hg_vertex_types (

gst_hg_ptr		H,	/* IN/OUT - hypergraph to set types for */
char *			types	/* IN - vertex types to set */
)
{
int		i, nverts, status;

	GST_PRELUDE

	status = 0;
	nverts = H -> num_verts;

	do {		/* Used only for "break". */
		if ((H EQ NULL) OR (types EQ NULL)) {
			status = GST_ERR_PARAMETER_VALUE_OUT_OF_RANGE;
			break;
		}

		/* Pass 1 -- verify the types. */
		for (i = 0; i < nverts; i++) {
			switch (types [i]) {
			case 'T':
			case 'S':
				break;

			default:
				status = GST_ERR_PARAMETER_VALUE_OUT_OF_RANGE;
				break;
			}
			if (status NE 0) break;
		}
		if (status NE 0) {
			/* One of the types[] values was bad. */
			break;
		}

		/* Pass 2 -- set the types. */
		for (i = 0; i < nverts; i++) {
			switch (types [i]) {
			case 'T':
				H -> tflag [i] = TRUE;
				break;

			case 'S':
				H -> tflag [i] = FALSE;
				break;

			default:
				FATAL_ERROR;
				break;
			}
		}
	} while (FALSE);

	GST_POSTLUDE

	return (status);
}

/*
 * Get the type of each vertex in the hypergraph.  A type of 'T'
 * indicates that the vertex is a terminal.  A type of 'S' indicates
 * that the vertex is a Steiner vertex.  All other types are illegal.
 */

	int
gst_get_hg_vertex_types (

gst_hg_ptr		H,	/* IN - hypergraph to get vertex types of */
char *			types	/* IN/OUT - vertex types */
)
{
int		i, nverts, status;

	GST_PRELUDE

	status = 0;

	if (H EQ NULL) {
		status = GST_ERR_PARAMETER_VALUE_OUT_OF_RANGE;
	}
	else if (types NE NULL) {
		nverts = H -> num_verts;
		for (i = 0; i < nverts; i++) {
			types [i] = (H -> tflag [i] ? 'T' : 'S');
		}
	}

	GST_POSTLUDE

	return (status);
}

/*
 * Temporary debug routine
 */

#if 0

	void
dump_hg (

gst_hg_ptr		H		/* IN - hypergraph */
)
{
int i;
int j;
int k;

	printf ("num_verts:         %d\n", H->num_verts);
	printf ("num_edges:         %d\n", H->num_edges);
	printf ("num_vert_masks:    %d\n", H->num_vert_masks);
	printf ("num_edge_masks:    %d\n", H->num_edge_masks);
	printf ("edges:             %p\n", H->edge);
	printf ("edge_size:         %p\n", H->edge_size);

	if ((H->num_edges < 50) AND (H -> edge NE NULL)) {
		for (i = 0; i < H->num_edges; i++) {
			k = H -> edge_size [i];
			for (j = 0; j < k; j++) {
				printf (" %d", H->edge[i][j]);
			}
			printf (" (cost: %f)\n", H->cost[i]);
		}
	}

	printf ("cost:              %p\n", H->cost);
	printf ("tflag:             %p\n", H->tflag);
	printf ("initial_vert_mask: %p\n", H->initial_vert_mask);
	printf ("initial_edge_mask: %p\n", H->initial_edge_mask);
	printf ("metric:            %d\n", H->metric);
	printf ("required_edges:    %p\n", H->required_edges);
	printf ("inc_edges:         %p\n", H->inc_edges);
	printf ("term_trees:        %p\n", H->term_trees);

	if ((H->num_verts < 50) AND (H -> term_trees)) {
		for (i = 0; i < H->num_verts; i++) {
			int *ep1 = H -> term_trees [i];
			int *ep2 = H -> term_trees [i + 1];
			while (ep1 < ep2) {
				printf (" %d", *ep1++);
			}
			printf (" (tflag: %d)\n", H->tflag[i]);
		}
	}
	printf ("pts:               %p\n", H->pts);
	printf ("full_trees:        %p\n", H->full_trees);
	if (H -> full_trees NE NULL) {
		for (i = 0; i < H -> num_edges; i++) {
			struct full_set *fst = H -> full_trees[i];
			printf (" tree_num: %d, tree_len: %f\n", fst->tree_num, fst->tree_len);
			printf (" nedges: %d\n", fst->nedges);
			printf (" terminals: %p (%d)\n", fst->terminals, fst->terminals->n);
			printf (" steiners: %p (%d)\n", fst->steiners, fst->steiners ? fst->steiners->n : 0);
			printf (" edges: %p (%d)\n", fst->edges, fst->nedges);
		}
	}

	/* FIXME: Dump properties */
}

#endif
