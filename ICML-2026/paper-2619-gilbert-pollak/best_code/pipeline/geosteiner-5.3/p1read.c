/***********************************************************************

	$Id: p1read.c,v 1.36 2023/04/03 17:23:42 warme Exp $

	File:	p1read.c
	Rev:	e-6
	Date:	04/03/2023

	Copyright (c) 1993, 2023 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines for reading the data that is output from phase 1.

************************************************************************

	Modification Log:

	b-1:	01/12/97	warme
		: Split p1io.c into reading and writing parts.
	b-2:	02/28/2001	warme
		: Numerous changes for 3.1 release.
		: New input scaling stuff.
		: Fixed uninitialized incompat array bug.
		: Use common sort_ints routine.
		: Free *all* gst_hypergraph data structures, and be more
		:  careful about freeing term_trees.
	c-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Uses parameters.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
		: Allow arbitrary line length for description
		:  (instance name) and machine description fields.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Fix -Wall issues.  Upgrade fatals.
	e-4:	09/30/2016	warme
		: Fix unsigned type where signed is needed.
	e-5:	11/18/2022	warme
		: Fixed comment style.
	e-6:	04/03/2023	warme
		: Always set status (if requested).
o		: Fix -Wall issue.

************************************************************************/

#include "p1read.h"

#include "fatal.h"
#include <float.h>
#include "geosteiner.h"
#include "io.h"
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "metric.h"
#include "parmblk.h"
#include "point.h"
#include "prepostlude.h"
#include "sortfuncs.h"
#include "steiner.h"
#include <string.h>

/*
 * Global Routines
 */

gst_hg_ptr	gst_load_hg (FILE *, gst_param_ptr, int *);

int **		_gst_compute_basic_incompat (struct gst_hypergraph *);
void		_gst_init_term_trees (struct gst_hypergraph *);


/*
 * External References
 */

	/* none */


/*
 * Local Routines
 */

static int		get_d (FILE *);
static double		get_dec_double (FILE *);
static double		get_hex_double (FILE *);
static char *		get_line (FILE * fp);
static int		get_version (FILE *, int, int);
/* static bitmap_t		get_x (FILE *); */
static double		hex_to_double (char *);
static int		hexdig (char);
static void		init_inc_edges (struct gst_hypergraph *,
					int **,
					int *);
static void		init_output_conversion_old (gst_hg_ptr);
static void		read_duplicate_terminal_groups (
					FILE *,
					struct gst_hypergraph *,
					int);
static gst_hg_ptr	read_version_0 (FILE *, int);
static gst_hg_ptr	read_version_2 (FILE *, int);
static void		remove_duplicates (int, int **, bitmap_t *);
static void		skip (FILE *);
static void		verify_symmetric (int **, int);

/*
 * This routine reads in the data as printed by the print-phase-1-data
 * routine.
 */


	gst_hg_ptr
gst_load_hg (

FILE *		fp,		/* IN - input file pointer. */
gst_param_ptr	params,		/* IN - parameter set (currently not used) */
int *		status		/* OUT - status */
)
{
int		version;
int		min;
int		max;
gst_hg_ptr	H;

	GST_PRELUDE

	if (params EQ NULL) {
		params = (gst_param_ptr) &_gst_default_parmblk;
	}

	gst_query_int_param ((gst_param_ptr) &_gst_default_parmblk,
			     GST_PARAM_SAVE_FORMAT,
			     NULL, NULL, &min, &max);
	version = get_version (fp, min, max);

	switch (version) {
	case GST_PVAL_SAVE_FORMAT_ORLIBRARY:
		H = read_version_0 (fp, version);
		break;

	case GST_PVAL_SAVE_FORMAT_VERSION2:
	case GST_PVAL_SAVE_FORMAT_VERSION3:
		H = read_version_2 (fp, version);
		break;

	default:
		H = NULL;
		break;
	}

	/* Set up the output conversion stuff. */
	if (H) {
		init_output_conversion_old (H);
		if (status NE NULL) {
			*status = 0;
		}
	}
	else {
		if (status NE NULL) {
			*status = GST_ERR_LOAD_ERROR;
		}
	}

	GST_POSTLUDE
	return (H);
}

/*
 * This routine reads in the extended OR-library format.
 */

	static
	gst_hg_ptr
read_version_0 (

FILE *		fp,		/* IN - input file pointer. */
int		version		/* IN - version number. */
)
{
int			i;
int			j;
int			k;
int			nverts;
int			nedges;
int			nmasks;
int			scale_factor;
int			total_edge_cardinality;
int			nt;
struct numlist *	line;
struct numlist *	p;
struct numlist **	hookp;
struct numlist **	prev_hookp;
struct numlist *	costs;
struct numlist **	cost_hookp;
int *			ip1;
int *			ip2;
struct gst_hypergraph *	cip;

	nverts = get_d (fp);
	nedges = get_d (fp);

	/* Allocate hypergraph... */
	cip = gst_create_hg (NULL);
	gst_set_hg_number_of_vertices (cip, nverts);

	nmasks = BMAP_ELTS (nedges);

	cip -> num_edges	= nedges;
	cip -> num_edge_masks	= nmasks;

	cip -> edge		= NEWA (nedges + 1, int *);
	cip -> edge_size	= NEWA (nedges, int);
	cip -> cost		= NEWA (nedges, dist_t);

	cip -> initial_edge_mask = NEWA (nmasks, bitmap_t);
	cip -> required_edges	 = NEWA (nmasks, bitmap_t);

	memset (cip -> initial_edge_mask, 0, nmasks * sizeof (bitmap_t));
	memset (cip -> required_edges,	  0, nmasks * sizeof (bitmap_t));

	for (i = 0; i < nedges; i++) {
		SETBIT (cip -> initial_edge_mask, i);
	}

	costs = NULL;
	cost_hookp = &costs;

	total_edge_cardinality = 0;

	for (i = 0; i < nedges; i++) {
		line = _gst_parse_line_of_numbers (fp);
		if (line EQ NULL) {
			fprintf (stderr, "Unexpected EOF!\n");
			exit (1);
		}

		/* Count the numbers and detach the last one */
		j = 0;
		prev_hookp = NULL;
		hookp = &line;
		for (;;) {
			p = *hookp;
			if (p EQ NULL) break;
			prev_hookp = hookp;
			hookp = &(p -> next);
			++j;
		}
		if (j < 3) {
			fprintf (stderr,
				 "Hyperedge with less than 2 vertices!\n");
			exit (1);
		}

		/* Transfer last number onto cost list. */
		p = *prev_hookp;
		*prev_hookp = NULL;
		--j;
		cip -> edge_size [i] = j;
		total_edge_cardinality += j;

		*cost_hookp = p;
		cost_hookp = &(p -> next);

		ip1 = NEWA (j, int);
		cip -> edge [i] = ip1;

		/* Verify remaining numbers are integers and convert. */
		j = 0;
		for (p = line; p NE NULL; p = p -> next) {
			if (p -> expon < 0) goto nonint;
			if (p -> expon > 0) {
				do {
					p -> mantissa *= 10.0;
				} while (--(p -> expon) > 0);
			}
			if ((p -> mantissa < 1) OR
			    (nverts < p -> mantissa) OR
			    (p -> mantissa NE floor (p -> mantissa))) {
				fprintf (stderr,
					 "Vertex number %g out of range.\n",
					 p -> mantissa);
				exit (1);
			}
			*ip1++ = p -> mantissa - 1;
		}
		while (line NE NULL) {
			 p = line;
			 line = line -> next;
			 free (p);
		}
		FATAL_ERROR_IF (cip -> edge [i] + cip -> edge_size [i] NE ip1);
	}

	/* Now convert the hyperedge costs. */
	scale_factor = _gst_compute_scaling_factor (costs);

	_gst_set_scale_info (cip -> scale, scale_factor);

	i = 0;
	while (costs NE NULL) {
		p = costs;
		costs = p -> next;
		p -> next = NULL;
		cip -> cost [i++] = _gst_read_numlist (p, cip -> scale);
	}
	FATAL_ERROR_IF (i NE nedges);

	/* Read in the list of terminal vertices... */
	if (fscanf (fp, " %d", &nt) NE 1) {
		/* EOF -- assume all vertices are terminals. */
		for (i = 0; i < nverts; i++) {
			cip -> tflag [i] = TRUE;
		}
	}
	else {
		/* All are Steiner vertices unless listed! */
		for (i = 0; i < nverts; i++) {
			cip -> tflag [i] = FALSE;
		}
		for (i = 0; i < nt; i++) {
			j = get_d (fp);
			FATAL_ERROR_IF ((j < 1) OR (nverts < j));
			cip -> tflag [j - 1] = TRUE;
		}
	}

	/* Copy hyperedges into contiguous form. */
	ip1 = NEWA (total_edge_cardinality, int);
	for (i = 0; i < nedges; i++) {
		ip2 = cip -> edge [i];
		cip -> edge [i] = ip1;
		k = cip -> edge_size [i];
		for (j = 0; j < k; j++) {
			*ip1++ = ip2 [j];
		}
		free ((char *) ip2);
	}
	cip -> edge [i] = ip1;

	_gst_init_term_trees (cip);
	cip -> inc_edges = NULL;
	return (cip);

	/* Error exit. */
nonint:
	fprintf (stderr, "Expected integer!\n");
	exit (1);
}

/*
 * This routine reads in the new data formats -- versions 2 and 3.
 */

	static
	gst_hg_ptr
read_version_2 (

FILE *		fp,		/* IN - input file pointer. */
int		version		/* IN - version number. */
)
{
int			i;
int			j;
int			k;
int			nverts;
int			nedges;
int			nmasks;
int			nt;
int			ns;
int			scale_factor;
int			fst_edge_count;
int			num_incompat;
int			ndg;
int			total_edge_card;
int			metric;
int			idelta;
char *			line;
struct pset *		terms;
struct pset *		steins;
struct point *		p1;
struct full_set *	fsp;
struct edge *		ep;
int *			ip1;
int *			ip2;
int *			icounts;
int **			incompat;
bool			geometric;
dist_t			len;
struct gst_hypergraph *	cip;

	skip (fp);	/* Skip rest of version line */

	line = get_line (fp);

#define RECTILINEAR		1
#define EUCLIDEAN		2
#define PURE_GRAPH		3
/* Other values should be interpreted as follows (to stay backwards compatible)
  #define UNIFORM_LAMBDA_2	1002
  #define UNIFORM_LAMBDA_3	1003
   ...
*/

	metric = get_d (fp);
#if 0
	if ((metric NE RECTILINEAR) AND
	    (metric NE EUCLIDEAN) AND
	    (metric NE PURE_GRAPH)) {
		fprintf (stderr, "Bad metric: %d\n", metric);
		exit (1);
	}
#else
	if (metric < 1) {
		fprintf (stderr, "Bad metric: %d\n", metric);
		exit (1);
	}
#endif
	geometric = (metric NE PURE_GRAPH);

	nverts = (int) get_d (fp);

	/* Allocate hypergraph... */
	cip = gst_create_hg (NULL);
	gst_set_hg_number_of_vertices (cip, nverts);
	gst_set_str_property (cip -> proplist, GST_PROP_HG_NAME, line);
	if (line NE NULL) {
		free (line);
	}

	gst_free_metric (cip -> metric);
	switch (metric) {
	case RECTILINEAR:
		cip -> metric	= gst_create_metric (GST_METRIC_L, 1, NULL);
		break;
	case EUCLIDEAN:
		cip -> metric	= gst_create_metric (GST_METRIC_L, 2, NULL);
		break;
	case PURE_GRAPH:
		cip -> metric	= gst_create_metric (GST_METRIC_NONE, 0, NULL);
		break;
	default:
		cip -> metric	= gst_create_metric (GST_METRIC_UNIFORM,
						     metric - 1000, NULL);
		break;
	}

#undef RECTILINEAR
#undef EUCLIDEAN
#undef PURE_GRAPH

	if (geometric) {
		(void) get_dec_double (fp);
		gst_set_dbl_property (cip -> proplist,
				      GST_PROP_HG_MST_LENGTH,
				      get_hex_double (fp));
	}

	ndg = 0;
	if ((version <= GST_PVAL_SAVE_FORMAT_VERSION2) AND geometric) {
		/* Version 2 has duplicate terminal groups... */
		ndg = (int) get_d (fp);
	}

	scale_factor = (int) get_d (fp);

	_gst_set_scale_info (cip -> scale, scale_factor);

	idelta = 0;
	if (version >= GST_PVAL_SAVE_FORMAT_VERSION3) {
		/* Version 3 has integrality delta... */
		(void) get_dec_double (fp);
		idelta = get_hex_double (fp);
	}
	gst_set_dbl_property (cip -> proplist, GST_PROP_HG_INTEGRALITY_DELTA, idelta);

	skip (fp);	/* skip rest of line before machine description */
	/* Read and discard machine description line. */
	line = get_line (fp);
	if (line NE NULL) {
		free (line);
	}

	gst_set_dbl_property (cip -> proplist, GST_PROP_HG_GENERATION_TIME, (get_d (fp))/100.0);

	nedges = (int) get_d (fp);

	nmasks = BMAP_ELTS (nedges);

	cip -> initial_edge_mask	= NEWA (nmasks, bitmap_t);
	cip -> required_edges		= NEWA (nmasks, bitmap_t);
	for (i = 0; i < nmasks; i++) {
		cip -> initial_edge_mask [i]	= 0;
		cip -> required_edges [i]	= 0;
	}

	cip -> num_edges	= nedges;
	cip -> num_edge_masks	= nmasks;
	cip -> edge		= NEWA (nedges + 1, int *);
	cip -> edge_size	= NEWA (nedges, int);
	cip -> cost		= NEWA (nedges, dist_t);
	if (geometric) {
		cip -> pts	= NEW_PSET (nverts);
		ZERO_PSET (cip -> pts, nverts);
		cip -> pts -> n = nverts;
		cip -> full_trees = NEWA (nedges, struct full_set *);
	}

	incompat		= NEWA (nedges, int *);
	icounts			= NEWA (nedges, int);

	if (geometric) {
		/* Read in the terminals... */
		for (i = 0; i < nverts; i++) {
			p1 = &(cip -> pts -> a [i]);
			(void) get_dec_double (fp);
			(void) get_dec_double (fp);
			p1 -> x		 = get_hex_double (fp);
			p1 -> y		 = get_hex_double (fp);
		}
	}

	if (version >= GST_PVAL_SAVE_FORMAT_VERSION3) {
		/* Version 3 has terminal/Steiner flag for each vertex. */
		for (i = 0; i < nverts; i++) {
			cip -> tflag [i] = get_d (fp);
		}
	}
	else {
		/* Version 2: assume all vertices are terminals. */
		for (i = 0; i < nverts; i++) {
			cip -> tflag [i] = TRUE;
		}
	}

	if ((version <= GST_PVAL_SAVE_FORMAT_VERSION2) AND geometric) {
		read_duplicate_terminal_groups (fp, cip, ndg);
	}

	/* hyperedges... */
	total_edge_card = 0;
	fsp = NULL;
	terms = NULL;
	for (i = 0; i < nedges; i++) {
		nt = get_d (fp);
		total_edge_card += nt;
		cip -> edge_size [i] = nt;
		ip1 = NEWA (nt, int);
		cip -> edge [i] = ip1;
		if (geometric) {
			fsp = NEW (struct full_set);
			(void) memset (fsp, 0, sizeof (*fsp));
			cip -> full_trees [i] = fsp;
			fsp -> tree_num = i;
			terms = NEW_PSET (nt);
			ZERO_PSET (terms, nt);
			terms -> n = nt;
		}
		for (j = 0; j < nt; j++) {
			k = get_d (fp);
			if ((k < 1) OR (k > nverts)) {
				fprintf (stderr,
					 "Terminal index out of range.\n");
				exit (1);
			}
			--k;
			*ip1++ = k;
			if (geometric) {
				terms -> a [j] = cip -> pts -> a [k];
			}
		}
		(void) get_dec_double (fp);
		len = get_hex_double (fp);
		cip -> cost [i] = len;

		if (geometric) {
			fsp -> tree_len = len;
			ns = get_d (fp);
			steins = NEW_PSET (ns);
			ZERO_PSET (steins, ns);
			steins -> n = ns;
			for (j = 0; j < ns; j++) {
				p1 = &(steins -> a [j]);
				(void) get_dec_double (fp);
				(void) get_dec_double (fp);
				p1 -> x = get_hex_double (fp);
				p1 -> y = get_hex_double (fp);
			}
			fsp -> terminals = terms;
			fsp -> steiners = steins;
			fst_edge_count = get_d (fp);
			fsp -> nedges = fst_edge_count;
			ep = NEWA (fst_edge_count, struct edge);
			fsp -> edges = ep;
			for (j = 0; j < fst_edge_count; j++, ep++) {
				ep -> len = 0;	/* should be unused... */
				k = get_d (fp);
				if ((-ns <= k) AND (k < 0)) {
					k = nt - k - 1;
				}
				else if ((0 < k) AND (k <= nt)) {
					--k;
				}
				else {
					fprintf (stderr, "Invalid edge endpoint!\n");
					exit (1);
				}
				ep -> p1 = k;
				k = get_d (fp);
				if ((-ns <= k) AND (k < 0)) {
					k = nt - k - 1;
				}
				else if ((0 < k) AND (k <= nt)) {
					--k;
				}
				else {
					fprintf (stderr, "Invalid edge endpoint!\n");
					exit (1);
				}
				ep -> p2 = k;
			}
		}
		k = get_d (fp);		/* full set status... */
		switch (k) {
		case 0:
			break;

		case 1:
			SETBIT (cip -> initial_edge_mask, i);
			break;

		case 2:
			SETBIT (cip -> required_edges, i);
			SETBIT (cip -> initial_edge_mask, i);
			break;

		default:
			fprintf (stderr, "Invalid full set status: %d\n", k);
			exit (1);
		}

		num_incompat = get_d (fp);
		icounts [i] = num_incompat;
		ip1 = NULL;
		if (num_incompat > 0) {
			ip1 = NEWA (num_incompat, int);
			for (j = 0; j < num_incompat; j++) {
				k = get_d (fp);
				if ((k <= 0) OR (k > nedges)) {
					fprintf (stderr, "Bad incompatible index.\n");
					exit (1);
				}
				ip1 [j] = k - 1;
			}
		}
		incompat [i] = ip1;

		if (version <= GST_PVAL_SAVE_FORMAT_VERSION2) {
			/* Version 2 has strongly compatible full sets... */
			k = get_d (fp);
			for (j = 0; j < k; j++) {
				(void) get_d (fp);
			}
		}
	}

	/* Copy hyperedges into contiguous form. */
	ip1 = NEWA (total_edge_card, int);
	for (i = 0; i < nedges; i++) {
		ip2 = cip -> edge [i];
		cip -> edge [i] = ip1;
		k = cip -> edge_size [i];
		for (j = 0; j < k; j++) {
			*ip1++ = ip2 [j];
		}
		if (geometric) {
			cip -> full_trees [i] -> tlist = ip2;
		}
		else {
			free ((char *) ip2);
		}
	}
	cip -> edge [i] = ip1;

	_gst_init_term_trees (cip);
	init_inc_edges (cip, incompat, icounts);

	free ((char *) icounts);
	for (i = 0; i < nedges; i++)
		free ((char *)incompat [i]);
	free ((char *) incompat);

	return (cip);
}

/*
 * This routine reads in the duplicate terminal groups.  We do not know
 * how big this is going to be.  Therefore we must plan for the worst
 * and reallocate when we are done...
 */

	static
	void
read_duplicate_terminal_groups (

FILE *			fp,		/* IN - input file pointer. */
struct gst_hypergraph *	cip,		/* IN/OUT - compatibility info */
int			ndg		/* IN - number of duplicate groups */
)
{
int		i;
int		j;
int		k;
int		n;
int		t;
int		nverts;
int		kmasks;
int *		real_terms;
bitmap_t *	mark;
int *		index;
int *		terms;
int **		dup_grps;

	if (ndg <= 0) {
		return;				/* nothing to read */
	}

	nverts = cip -> num_verts;

	if (2 * ndg > nverts) {
		/* Too many duplicate terminal groups! */
		FATAL_ERROR;
	}

	kmasks = cip -> num_vert_masks;

	mark = NEWA (kmasks, bitmap_t);
	index = NEWA (ndg + 1, int);
	terms = NEWA (nverts, int);

	for (i = 0; i < kmasks; i++) {
		mark [i] = 0;
	}

	k = 0;
	for (i = 0; i < ndg; i++) {
		index [i] = k;
		n = (int) get_d (fp);
		for (j = 0; j < n; j++) {
			t = get_d (fp);
			FATAL_ERROR_IF ((t < 1) OR (cip -> num_verts < t));
			--t;
			FATAL_ERROR_IF (BITON (mark, t));
			SETBIT (mark, t);
			terms [k++] = t;
		}
	}
	index [i] = k;

	real_terms = NEWA (k, int);
	(void) memcpy ((char *) real_terms,
		       (char *) terms,
		       k * sizeof (int));

	dup_grps = NEWA (ndg + 1, int *);
	for (i = 0; i <= ndg; i++) {
		dup_grps [i] = real_terms + index [i];
	}

	free ((char *) dup_grps);
	free ((char *) real_terms);
	free ((char *) terms);
	free ((char *) index);
	free ((char *) mark);

	/* Remove duplicate terminals from the problem... */
	remove_duplicates (ndg, dup_grps, cip -> initial_vert_mask);
}

/*
 * This routine creates the "term_trees" array that is indexed by point
 * number and gives a list of all tree-numbers involving that point.
 */

	void
_gst_init_term_trees (

struct gst_hypergraph *	cip		/* IN - compatibility info */
)
{
int			i;
int			j;
int			nverts;
int			nedges;
int			total;
int *			ip;
int *			counts;
int **			ptrs;
int *			vp1;
int *			vp2;

	nverts = cip -> num_verts;
	nedges = cip -> num_edges;

	counts	= NEWA (nverts, int);

	total = 0;
	for (i = 0; i < nverts; i++) {
		counts [i] = 0;
	}

	for (i = 0; i < nedges; i++) {
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			++counts [*vp1++];
			++total;
		}
	}

	ip			= NEWA (total, int);
	cip -> term_trees	= NEWA (nverts + 1, int *);
	ptrs			= NEWA (nverts, int *);

	for (i = 0; i < nverts; i++) {
		cip -> term_trees [i]	= ip;
		ptrs [i]		= ip;
		ip += counts [i];
	}
	cip -> term_trees [i] = ip;

	for (i = 0; i < nedges; i++) {
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			j = *vp1++;
			*(ptrs [j])++ = i;
		}
	}

	free ((char *) ptrs);
	free ((char *) counts);
}

/*
 * We don't want to store all of the "basic" incompatibilities.
 * Although our own code does not put any of these into the phase 1
 * data files, this routine filters out any that might be present.
 * We take this precaution in case somebody wants to import phase 1
 * data files from some "foreign" program.
 * The final result is saved as "cip -> inc_edges".
 */

	static
	void
init_inc_edges (

struct gst_hypergraph *	cip,		/* IN - compatibility info */
int **			incompat,	/* IN - input incompatibility lists */
int *			icounts		/* IN - lengths of incompat lists */
)
{
int			i;
int			j;
int			k;
int			nedges;
int			kmasks;
int			total;
int			common;
int *			ip1;
int *			ip2;
int *			ip3;
int *			ip4;
int *			ip5;
int *			flist;
int *			newcounts;
int **			inc_edges;
bitmap_t *		edge_mask;
bitmap_t *		vmask;

	nedges	  = cip -> num_edges;
	kmasks	  = cip -> num_vert_masks;
	edge_mask = cip -> initial_edge_mask;

	inc_edges = NEWA (nedges + 1, int *);
	newcounts = NEWA (nedges, int);
	vmask	  = NEWA (kmasks, bitmap_t);

	for (i = 0; i < kmasks; i++) {
		vmask [i] = 0;
	}

	flist = NEWA (nedges, int);

	total = 0;
	for (i = 0; i < nedges; i++) {
		newcounts [i] = 0;
		inc_edges [i] = NULL;
		k = icounts [i];
		if (k <= 0) continue;
		if (NOT BITON (edge_mask, i)) continue;

		ip1 = cip -> edge [i];
		ip2 = cip -> edge [i + 1];
		while (ip1 < ip2) {
			j = *ip1++;
			SETBIT (vmask, j);
		}

		ip1 = flist;
		ip2 = incompat [i];
		ip3 = ip2 + k;
		while (ip2 < ip3) {
			j = *ip2++;
			if (NOT BITON (edge_mask, j)) continue;

			common = 0;
			ip4 = cip -> edge [j];
			ip5 = cip -> edge [j + 1];
			while (ip4 < ip5) {
				k = *ip4++;
				if (BITON (vmask, k)) {
					++common;
					if (common > 1) break;
				}
			}
			if (common > 1) {
				/* Two or more vertices in common...	*/
				/* Do not keep these "basic" in the	*/
				/* inc_edges data structure!		*/
				continue;
			}
			*ip1++ = j;
		}
		k = ip1 - flist;
		newcounts [i] = k;
		total += k;
		ip2 = NULL;
		if (k > 0) {
			_gst_sort_ints (flist, k);
			ip2 = NEWA (k, int);
			for (j = 0; j < k; j++) {
				ip2 [j] = flist [j];
			}
		}
		inc_edges [i] = ip2;

		/* Reset vmask to all zeros. */
		ip1 = cip -> edge [i];
		ip2 = cip -> edge [i + 1];
		while (ip1 < ip2) {
			j = *ip1++;
			CLRBIT (vmask, j);
		}
	}
	free ((char *) flist);
	free ((char *) vmask);

	if (total <= 0) {
		free ((char *) inc_edges);
		inc_edges = NULL;
	}
	else {
		ip1 = NEWA (total, int);
		for (i = 0; i < nedges; i++) {
			ip2 = inc_edges [i];
			inc_edges [i] = ip1;

			if (ip2 EQ NULL) continue;

			k = newcounts [i];
			for (j = 0; j < k; j++) {
				*ip1++ = ip2 [j];
			}
			free ((char *) ip2);
		}
		inc_edges [i] = ip1;

		FATAL_ERROR_IF (ip1 - inc_edges [0] NE total);
	}

	free ((char *) newcounts);

	cip -> inc_edges = inc_edges;

	verify_symmetric (cip -> inc_edges, nedges);
}

/*
 * This routine computes for each FST, a list of those FSTs having
 * at least 2 vertices in common.
 */

#if 0

	int **
_gst_compute_basic_incompat (

struct gst_hypergraph *	cip	/* IN/OUT - compatibility info. */
)
{
int			i;
int			j;
int			k;
int			t;
int			fs;
int			common;
int			nedges;
int			kmasks;
int			nmasks;
int			total;
bitmap_t *		fsmask;
bitmap_t *		edge_mask;
bitmap_t *		tmask;
int *			ep1;
int *			ep2;
int *			vp1;
int *			vp2;
int *			flist;
int **			incompat;
int *			counts;

	kmasks	  = cip -> num_vert_masks;
	nedges	  = cip -> num_edges;
	nmasks	  = cip -> num_edge_masks;
	edge_mask = cip -> initial_edge_mask;

	incompat = NEWA (nedges + 1, int *);
	counts	 = NEWA (nedges, int);
	flist	 = NEWA (nedges, int);

	for (i = 0; i < nedges; i++) {
		incompat [i] = NULL;
		counts [i]   = 0;
	}

	fsmask = NEWA (nmasks, bitmap_t);
	for (i = 0; i < nmasks; i++) {
		fsmask [i] = 0;
	}

	tmask = NEWA (kmasks, bitmap_t);
	for (i = 0; i < kmasks; i++) {
		tmask [i] = 0;
	}

	/* Compute the list of (lists of) basically incomatible FSTs... */
	total = 0;
	for (i = 0; i < nedges; i++) {
		incompat [i] = NULL;
		if (NOT BITON (edge_mask, i)) continue;
		/* Develop list of all FSTs adjacent to FST i... */
		k = 0;
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			t = *vp1++;
			SETBIT (tmask, t);
			ep1 = cip -> term_trees [t];
			ep2 = cip -> term_trees [t + 1];
			while (ep1 < ep2) {
				fs = *ep1++;
				if (BITON (fsmask, fs)) continue;
				if (NOT BITON (edge_mask, fs)) continue;
				if (fs EQ i) continue;
				SETBIT (fsmask, fs);
				flist [k++] = fs;
			}
		}
		ep1 = &flist [0];
		ep2 = &flist [k];
		k = 0;
		while (ep1 < ep2) {
			fs = *ep1++;
			CLRBIT (fsmask, fs);
			/* Count number of vertices in common. */
			common = 0;
			vp1 = cip -> edge [fs];
			vp2 = cip -> edge [fs + 1];
			while (vp1 < vp2) {
				j = *vp1++;
				if (BITON (tmask, j)) {
					++common;
				}
			}
			if (common >= 2) {
				/* Too many in common!  Retain... */
				flist [k++] = fs;
			}
		}
		counts [i] = k;
		total += k;
		if (k > 0) {
			/* Save off sorted list of incompatible FSTs. */
			_gst_sort_ints (flist, k);
			ep1 = NEWA (k, int);
			incompat [i] = ep1;
			ep2 = flist;
			for (j = 0; j < k; j++) {
				*ep1++ = *ep2++;
			}
		}
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			j = *vp1++;
			CLRBIT (tmask, j);
		}
	}
	free ((char *) tmask);
	free ((char *) fsmask);
	free ((char *) flist);

	/* Now allocate and copy into contiguous memory... */
	ep1 = NEWA (total, int);
	for (i = 0; i < nedges; i++) {
		ep2 = incompat [i];
		incompat [i] = ep1;
		if (ep2 EQ NULL) continue;
		k = counts [i];
		for (j = 0; j < k; j++) {
			*ep1++ = ep2 [j];
		}
		free ((char *) ep2);
	}
	incompat [i] = ep1;

	free ((char *) counts);

	FATAL_ERROR_IF (ep1 - incompat [0] NE total);

	return (incompat);
}

#endif

/*
 * Verify that the given "incompatibility" relation is symmetric.
 */

	static
	void
verify_symmetric (

int **		incompat,		/* IN - incompatibility lists */
int		nedges			/* IN - number of FSTs */
)
{
int		i;
int		j;
int **		begp;
int **		endp;

	if (incompat EQ NULL) return;

	begp = NEWA (nedges, int *);
	endp = NEWA (nedges, int *);

	for (i = 0; i < nedges; i++) {
		begp [i] = incompat [i];
		endp [i] = incompat [i + 1];
	}

	for (i = 0; i < nedges; i++) {
		for (;;) {
			if (begp [i] >= endp [i]) break;
			j = begp [i] [0];
			if (j > i) break;
			++(begp [i]);
			if ((begp [j] >= endp [j]) OR
			    (begp [j] [0] NE i)) {
				/* Incompatibility array not symmetric! */
				FATAL_ERROR;
			}
			++(begp [j]);
		}
	}

	for (i = 0; i < nedges; i++) {
		if (begp [i] NE endp [i]) {
			/* Incompatibility array not symmetric! */
			FATAL_ERROR;
		}
	}

	free ((char *) endp);
	free ((char *) begp);
}

	static
	void
init_output_conversion_old (

gst_hg_ptr		H
)
{
int			i;
struct point *		p;
bool			integral;
double			c;
struct pset *		pts;
struct gst_scale_info *	sip;

	pts = H -> pts;
	sip = H -> scale;

	integral = TRUE;

	if (pts NE NULL) {
		p = &(pts -> a [0]);
		for (i = 0; i < pts -> n; i++, p++) {
			c = floor ((double) (p -> x));
			if (c NE p -> x) {
				integral = FALSE;
				break;
			}
			c = floor ((double) (p -> y));
			if (c NE p -> y) {
				integral = FALSE;
				break;
			}
		}
	}

	/* If integral and not Euclidean */
	if (integral AND (_gst_is_rectilinear (H))) {
		sip -> min_precision = 0;
	}
	else {
		sip -> min_precision = DBL_DIG + 1;
	}
}

/*
 * This routine determines the version number of the input data.
 */

	static
	int
get_version (

FILE *		fp,		/* IN - input file pointer. */
int		min,		/* IN - min version number */
int		max		/* IN - max version number */
)
{
int		c;
int		n;
int		version;

	/* Strip any white space... */
	do {
		c = getc (fp);
	} while ((c EQ ' ') OR
		 (c EQ '\t') OR
		 (c EQ '\n') OR
		 (c EQ '\f'));

	if (c < 0) {
		fprintf (stderr, "Unexpected EOF!\n");
		exit (1);
	}
	if (('0' <= c) AND (c <= '9')) {
		/* No version number input -- version 0. */
		ungetc (c, fp);
		return (GST_PVAL_SAVE_FORMAT_ORLIBRARY);
	}

	if (c NE 'V') {
		fprintf (stderr, "Bad data version number!\n");
		exit (1);
	}
	version = -1;
	n = fscanf (fp, "%d", &version);
	if (n NE 1) {
		fprintf (stderr, "Data version number not found!\n");
		exit (1);
	}
	if (version < min OR version > max) {
		fprintf (stderr, "Version number out of range!\n");
		exit (1);
	}

	return (version);
}

/*
 * This routine reads a single decimal number from stdin.
 */

	static
	int
get_d (

FILE *		fp		/* IN - input file pointer. */
)
{
int		i, n;

	n = fscanf (fp, " %d", &i);
	if (n NE 1) {
		fprintf (stderr, "Unexpected EOF!\n");
		exit (1);
	}

	return (i);
}


#if 0 /* This function is currently not in use... */
/*
 * This routine reads in a 32-bit bitmap element as a hex number
 * from stdin.
 */

	static
	bitmap_t
get_x (

FILE *		fp		/* IN - input file pointer. */
)
{
bitmap_t	i;
int		n;

	n = fscanf (fp, " %lx", &i);
	if (n NE 1) {
		fprintf (stderr, "Unexpected EOF!\n");
		exit (1);
	}

	return (i);
}
#endif

/*
 * This routine reads in an entire line as a string, properly handling
 * lines of arbitrary length.  The caller must free the returned string.
 */

struct lbuf {
	struct lbuf *	next;
	int		nbytes;
	char		buf [128];
};

	static
	char *
get_line (

FILE *		fp		/* IN - input file pointer. */
)
{
int		c;
size_t		i, n;
char *		p;
struct lbuf *	bp;
struct lbuf *	new_bp;
struct lbuf	first_buf;

	bp = &first_buf;
	bp -> next	= NULL;
	bp -> nbytes	= 0;

#define LBPUT(ch)						\
	do {							\
		FATAL_ERROR_IF (bp -> next NE NULL);		\
		if (bp -> nbytes >= sizeof (bp -> buf)) {	\
			new_bp = NEW (struct lbuf);		\
			bp -> next = new_bp;			\
			bp = new_bp;				\
			bp -> next	= NULL;			\
			bp -> nbytes	= 0;			\
		}						\
		bp -> buf [bp -> nbytes++] = (ch);		\
	} while (FALSE)

	n = 0;
	for (;;) {
		c = fgetc (fp);
		if (c < 0) {
			fprintf (stderr, "Unexpected EOF!\n");
			exit (1);
		}
		if (c EQ '\n') break;
		if (c EQ '\0') {
			/* Silently strip nul bytes for now. */
			continue;
		}
		LBPUT (c);
		++n;
	}
	LBPUT ('\0');
	++n;

	p = NEWA (n, char);

	/* Copy first buffer. */
	i = 0;
	bp = &first_buf;
	memcpy (&p [i], bp -> buf, bp -> nbytes);
	i += bp -> nbytes;

	/* Copy remaining buffers, freeing as we go. */
	bp = bp -> next;
	while (bp NE NULL) {
		memcpy (&p [i], bp -> buf, bp -> nbytes);
		i += bp -> nbytes;
		new_bp = bp -> next;
		bp -> next = NULL;
		free (bp);
		bp = new_bp;
	}
	FATAL_ERROR_IF (i NE n);

	return (p);
}

/*
 * This routine reads a decimal floating point number.
 */

	static
	double
get_dec_double (

FILE *		fp		/* IN - input file pointer. */
)
{
double		x;

	if (fscanf (fp, " %lg", &x) NE 1) {
		fprintf (stderr, "Expected floating point number.\n");
		exit (1);
	}

	return (x);
}

/*
 * This routine reads a HEXIDECIMAL floating point number.
 */

	static
	double
get_hex_double (

FILE *		fp		/* IN - input file pointer. */
)
{
char		buf1 [128];

	if (fscanf (fp, " %s", buf1) NE 1) {
		fprintf (stderr, "Unexpected EOF!\n");
		exit (1);
	}

	return (hex_to_double (buf1));
}

/*
 * This routine skips the remainder of the current line.
 */

	static
	void
skip (

FILE *		fp
)

{
int		c;

	for (;;) {
		c = getc (fp);
		if (c < 0) break;
		if (c EQ '\n') break;
	}
}

/*
 * This routine converts the printable ASCII hex format back into
 * a floating point number.
 */

	static
	double
hex_to_double (

char *		s		/* IN - the hex ASCII string to decode */
)
{
char		c;
char *		p1;
char *		p2;
int		digit;
int		expon;
int		esign;
double		msign;
double		mant;
double		value;

	/* Strip leading white space */
	for (;;) {
		c = *s++;
		if (c EQ ' ') continue;
		if (c EQ '\n') continue;
		if (c EQ '\t') continue;
		if (c EQ '\f') continue;
		if (c NE '\v') break;
	}

	msign = 1.0;
	if (c EQ '-') {
		msign = -1.0;
		c = *s++;
	}

	mant = 0.0;

	if (c NE '.') return (msign * mant);

	/* find end of mantissa */
	p1 = s;
	for (;;) {
		c = *s++;
		digit = hexdig (c);
		if (digit < 0) break;
	}

	/* rescan mantissa in reverse */
	for (p2 = s - 1; p2 > p1; ) {
		digit = hexdig (*--p2);
		mant += ((double) digit);
		mant *= 0.0625;		/* divide by 16... */
	}

	expon = 0;
	if ((c EQ 'x') OR (c EQ 'X')) {
		c = *s++;
		esign = 1;
		if (c EQ '-') {
			esign = -1;
			c = *s++;
		}
		for (;;) {
			digit = hexdig (c);
			if (digit < 0) break;
			expon = (expon << 4) + digit;
			c = *s++;
		}
		expon *= esign;
	}

	value = msign * ldexp (mant, expon);

	return (value);
}

/*
 * Routine to identify and convert hexidecimal ASCII digits.
 */

	static
	int
hexdig (

char		c
)
{
	switch (c) {

	case '0':		return (0);
	case '1':		return (1);
	case '2':		return (2);
	case '3':		return (3);
	case '4':		return (4);
	case '5':		return (5);
	case '6':		return (6);
	case '7':		return (7);
	case '8':		return (8);
	case '9':		return (9);
	case 'A': case 'a':	return (10);
	case 'B': case 'b':	return (11);
	case 'C': case 'c':	return (12);
	case 'D': case 'd':	return (13);
	case 'E': case 'e':	return (14);
	case 'F': case 'f':	return (15);

	}

	return (-1);
}

/*
 * This routine turns off the "tmap" bit for all but the first
 * terminal in each duplicate terminal group.  This effectively
 * removes the duplicated terminals from the problem.
 */

	static
	void
remove_duplicates (

int		ndg,		/* IN - number of duplicate terminal groups */
int **		list,		/* IN - list of duplicate terminal groups */
bitmap_t *	tmap		/* IN/OUT - valid subset of terminals */
)
{
int		i;
int *		ip1;
int *		ip2;

	for (i = 0; i < ndg; i++) {
		ip1 = list [i];
		ip2 = list [i + 1];

		/* Retain the first in this group, exclude all	*/
		/* of the others...				*/
		while (++ip1 < ip2) {
			CLRBIT (tmap, *ip1);
		}
	}
}
