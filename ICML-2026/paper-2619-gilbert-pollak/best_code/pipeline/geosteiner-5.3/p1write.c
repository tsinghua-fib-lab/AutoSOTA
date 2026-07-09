/***********************************************************************

	$Id: p1write.c,v 1.35 2022/11/19 13:45:53 warme Exp $

	File:	p1write.c
	Rev:	e-4
	Date:	11/18/2022

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines for writing the data that is output from phase 1.

************************************************************************

	Modification Log:

	b-1:	01/12/97	warme
		: Split p1io.c into reading and writing parts.
	b-2:	02/28/2001	warme
		: Changes for 3.1 release.
		: Use new scaling stuff.
		: Terminate last tflags line.
		: Don't include pruned FSTs in incompatible lists.
		: Avoid empty line (a single tab) when incompatible
		:  list is empty.
	c-1:	08/05/2002	benny
		: Numerous changes for library release.
		: Uses parameters.
		: Changed version defines.
		: Handles SteinLib (version 1).
	d-1:	12/23/2014	warme
		: Fix two 64-bit architecture issues.
		: Fix two SteinLib formatting issues reported by
		:  Renato Werneck.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
		: Add SteinLib "integer" format.
	e-3:	09/24/2016	warme
		: Reorganize include files, upgrade fatals.
		: Use better encapsulation for time conversions.
	e-4:	11/18/2022	warme
		: Fix -Wall issue.

************************************************************************/

#include "config.h"
#include "costextension.h"
#include "cputime.h"
#include "geosteiner.h"
#include "fatal.h"
#include "io.h"
#include <limits.h>
#include "logic.h"
#include <math.h>
#include "memory.h"
#include "metric.h"
#include "parmblk.h"
#include "point.h"
#include "prepostlude.h"
#include "steiner.h"
#include <string.h>

/*
 * Global Routines
 */

int		gst_save_hg (FILE *, struct gst_hypergraph *, gst_param_ptr);


/*
 * Local Routines
 */

static void		double_to_hex (double, char *);
static void		print_version_0 (FILE *, struct gst_hypergraph *, int);
static void		print_version_2 (FILE *, struct gst_hypergraph *, int);

/*
 * This routine prints out all of the data that is output from
 * phase 1 of the algorithm.  The output is almost readable by
 * humans, but is designed more to be machine-readable.
 */

	int
gst_save_hg (

FILE *			fp,		/* IN - file pointer for output. */
struct gst_hypergraph *	cip,		/* IN - compatability info. */
gst_param_ptr		params		/* IN - parameters. */
)
{
int	version;

	GST_PRELUDE

	if (params EQ NULL) {
		params = (gst_param_ptr) &_gst_default_parmblk;
	}
	gst_get_int_param (params, GST_PARAM_SAVE_FORMAT, &version);

	switch (version) {
	case GST_PVAL_SAVE_FORMAT_ORLIBRARY:
	case GST_PVAL_SAVE_FORMAT_STEINLIB:
	case GST_PVAL_SAVE_FORMAT_STEINLIB_INT:
		print_version_0 (fp, cip, version);
		break;

	case GST_PVAL_SAVE_FORMAT_VERSION2:
	case GST_PVAL_SAVE_FORMAT_VERSION3:
		/* Versions 2 and 3 are different enough that we use	*/
		/* a completely different routine. */
		print_version_2 (fp, cip, version);
		break;

	default:
		FATAL_ERROR;
	}

	GST_POSTLUDE
	return (0);
}

/*
 * This routine prints out the extended OR-library format -- version 0.
 */

	static
	void
print_version_0 (

FILE *			fp,		/* IN - file pointer for output. */
struct gst_hypergraph *	cip,		/* IN - compatibility info. */
int			version		/* IN - version to generate. */
)
{
int			i;
int			j;
int			n;
int			m;
int			slen;
int *			vp1;
int *			vp2;
bool			orlibrary;
bool			steinlib;
bool			steinlib_int;
bool			have_extended_costs;
double			mst_weight, scale_mul, scale_div, scale_target;
char			buf1 [128];
char			buf2 [128];
char *			descr;
gst_proplist_ptr	hgprop;

	orlibrary	= (version EQ GST_PVAL_SAVE_FORMAT_ORLIBRARY);
	steinlib	= (version EQ GST_PVAL_SAVE_FORMAT_STEINLIB);
	steinlib_int	= (version EQ GST_PVAL_SAVE_FORMAT_STEINLIB_INT);

	/* See if we have extended edge cost info available. */
	have_extended_costs =
		_gst_hg_cost_extension_have_write_edge_cost (
						cip -> cost_extension);

	n = cip -> num_verts;
	m = cip -> num_edges;

	if (orlibrary) {
		fprintf (fp, " %d %d\n", n, m);
	}

	scale_mul = 1.0;
	scale_div = 1.0;
	if (steinlib_int) {
		/* compute scaling factor for edge weights */
		mst_weight = 0.0;
		for (i = 0; i < m; i++) {
			if (cip -> edge_size [i] EQ 2) {
				mst_weight += cip -> cost [i];
			}
		}

		/* Provide additional safety margin. */
		mst_weight = 9.0 * mst_weight / 8.0;

		scale_target = (double) ULONG_MAX;
		if (mst_weight < scale_target) {
			/* Scale edge weights up. */
			while (10.0 * scale_mul * mst_weight < scale_target) {
				scale_mul *= 10.0;
			}
		}
		else {
			/* Scale edge weights down. */
			while (mst_weight / scale_div > scale_target) {
				scale_div *= 10.0;
			}
		}
	}

	if (steinlib OR steinlib_int) {
		hgprop = gst_get_hg_properties (cip);
		slen = -1;
		gst_get_str_property (hgprop, GST_PROP_HG_NAME, &slen, NULL);
		if (slen > 0) {
			descr = NEWA (slen + 1, char);
			gst_get_str_property (hgprop, GST_PROP_HG_NAME, NULL, descr);
		}
		else {
			descr  = NEWA (1, char);
			*descr = '\0';
		}

		fprintf (fp, "33d32945 STP File, STP Format Version 1.00\n"
			"Section Comment\n"
			"Name    \"%s\"\n"
			"Creator \"GeoSteiner\"\n"
			"Remark  \"Reduced graph from FST generator\"\n"
			"End\n\n"
			"Section Graph\n"
			"Nodes %d\n"
			"Edges %d\n",
			descr, n, m);
		free (descr);
	}

	/* hyperedges... */
	for (i = 0; i < m; i++) {
		if (steinlib OR steinlib_int) {
			fprintf (fp, "E");
		}

		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		while (vp1 < vp2) {
			j = *vp1++;
			fprintf (fp, " %d", j + 1);
		}

		if (have_extended_costs) {
			/* Write out cost from specially encapsulated	*/
			/* extension object.				*/
			fprintf (fp, " ");
			_gst_hg_cost_extension_write_edge_cost (
						cip -> cost_extension, fp, i);
			fprintf (fp, "\n");
		}
		else if (steinlib_int) {
			double cost = cip -> cost [i];
			double scaled_cost = cost * scale_mul / scale_div;
			double rounded_cost = floor (scaled_cost + 0.5);
			fprintf (fp, " %lu\n", (unsigned long) rounded_cost);
		} 
		else {
			_gst_dist_to_string (buf1, cip -> cost [i], cip -> scale);
			fprintf (fp, " %s\n", buf1);
		}
	}

	if (orlibrary) {
		/* Info about terminals. */
		j = 0;
		for (i = 0; i < n; i++) {
			if (NOT (cip -> tflag [i])) continue;
			++j;
		}
		fprintf (fp, "\t%d\n", j);
		j = 0;
		for (i = 0; i < n; i++) {
			if (NOT (cip -> tflag [i])) continue;
			fprintf (fp, " %d", i + 1);
			if (++j >= 10) {
				fprintf (fp, "\n");
				j = 0;
			}
		}
	}

	if (steinlib OR steinlib_int) {
		/* Info about terminals. */
		j = 0;
		for (i = 0; i < n; i++) {
			if (NOT (cip -> tflag [i])) continue;
			++j;
		}
		fprintf (fp, "End\n\n"
			"Section Terminals\n"
			"Terminals %d\n",
			j);
		for (i = 0; i < n; i++) {
			if (NOT (cip -> tflag [i])) continue;
			fprintf (fp, "T %d\n", i + 1);
		}
		fprintf (fp, "End\n\n"
			"Section Coordinates\n");
		for (i = 0; i < n; i++) { /* terminals */
			_gst_dist_to_string (buf1,
					     cip -> pts -> a[i].x,
					     cip -> scale);
			_gst_dist_to_string (buf2,
					     cip -> pts -> a[i].y,
					     cip -> scale);
			fprintf (fp, "DD %d %s %s\n", i+1, buf1, buf2);
		}
		fprintf (fp, "End\n\n"
			"EOF\n");
	}
}

/*
 * This routine prints out the new data format -- versions 2 and 3.
 */

	static
	void
print_version_2 (

FILE *			fp,		/* IN - file pointer for output. */
struct gst_hypergraph *	cip,		/* IN - compatibility info. */
int			version		/* IN - version to generate. */
)
{
int			i;
int			j;
int			k;
int			n;
int			m;
int			col;
int			kmasks;
int			count;
int			slen;
dist_t			mst_len;
struct point *		p1;
struct full_set *	fsp;
struct pset *		terms;
struct pset *		steins;
int *			ep1;
int *			ep2;
int *			ep3;
int *			flist;
int *			vp1;
int *			vp2;
double			idelta;
double			gen_time;
double			prune_time;
bitmap_t *		tmask;
bool			geometric;
char *			descr;
char			buf1 [64];
char			buf2 [64];
char			buf3 [64];
char			buf4 [64];
gst_proplist_ptr	hgprop;

#define MAXCOL	78

	geometric = ((cip -> metric NE GST_METRIC_NONE) AND
		     (cip -> full_trees NE NULL) AND
		     (cip -> pts NE NULL));

	n = cip -> num_verts;
	m = cip -> num_edges;
	kmasks = cip -> num_vert_masks;

	fprintf (fp, "V%d\n", version);

	hgprop = gst_get_hg_properties (cip);

	slen = -1;
	gst_get_str_property (hgprop, GST_PROP_HG_NAME, &slen, NULL);
	if (slen > 0) {
		descr = NEWA (slen + 1, char);
		gst_get_str_property (hgprop, GST_PROP_HG_NAME, NULL, descr);
		fprintf (fp, "%s\n", descr);
		free (descr);
	}
	else {
		fprintf (fp, "\n");
	}

#define RECTILINEAR		1
#define EUCLIDEAN		2
#define PURE_GRAPH		3
/* Other values should be interpreted as follows (to stay backwards compatible)
  #define UNIFORM_LAMBDA_2	1002
  #define UNIFORM_LAMBDA_3	1003
   ...
*/
	if (geometric) {
		if (_gst_is_rectilinear (cip)) {
			fprintf (fp, "%d\n", RECTILINEAR);
		}
		else if (_gst_is_euclidean (cip)) {
			fprintf (fp, "%d\n", EUCLIDEAN);
		}
		else if (cip -> metric -> type EQ GST_METRIC_UNIFORM) {
			fprintf (fp, "%d\n", cip -> metric -> parameter + 1000);
		}
		else {
			FATAL_ERROR;
		}
	}
	else {
		/* Not enough info to put out geometric stuff -- pure graph */
		fprintf (fp, "%d\n", PURE_GRAPH);
	}
	fprintf (fp, "%d\n", n);
#undef	RECTILINEAR
#undef	EUCLIDEAN
#undef	PURE_GRAPH

	if (geometric) {
		/* Compute MST length... */
		mst_len = 0.0;
		gst_get_dbl_property (cip -> proplist,
				      GST_PROP_HG_MST_LENGTH,
				      &mst_len);
		_gst_dist_to_string (buf1, mst_len, cip -> scale);
		double_to_hex ((double) mst_len, buf2);
		fprintf (fp, "%s %s\n", buf1, buf2);
	}

	if ((version <= GST_PVAL_SAVE_FORMAT_VERSION2) AND geometric) {
		/* No duplicate terminal groups... */
		fprintf (fp, "0\n");
	}
	fprintf (fp, "%d\n", cip -> scale -> scale);
	if (version >= GST_PVAL_SAVE_FORMAT_VERSION3) {
		/* Version 3 has integrality delta... */
		idelta = 0.0;
		gst_get_dbl_property (cip -> proplist,
				      GST_PROP_HG_INTEGRALITY_DELTA,
				      &idelta);
		_gst_dist_to_string (buf1, idelta, cip -> scale);
		double_to_hex ((double) idelta, buf2);
		fprintf (fp, "%s %s\n", buf1, buf2);
	}

	fprintf (fp, "%s\n", gst_env -> machine_string);

	/* Note that generation and pruning time cannot be saved independently.
	   Instead they are added and saved as generation time (old p1time) */
	gen_time = 0.0;
	gst_get_dbl_property (cip -> proplist,
			      GST_PROP_HG_GENERATION_TIME,
			      &gen_time);
	prune_time = 0.0;
	gst_get_dbl_property (cip -> proplist,
			      GST_PROP_HG_PRUNING_TIME,
			      &prune_time);
	fprintf (fp, "%u\n",			/* CPU time */
		 _gst_double_seconds_to_cpu_time_t (gen_time + prune_time));
	fprintf (fp, "%d\n", m);		/* Number of hyperedges */

	if (geometric) {
		p1 = &(cip -> pts -> a [0]);
		for (i = 0; i < n; i++, p1++) {
			_gst_coord_to_string (buf1, p1 -> x, cip -> scale);
			_gst_coord_to_string (buf2, p1 -> y, cip -> scale);
			double_to_hex ((double) (p1 -> x), buf3);
			double_to_hex ((double) (p1 -> y), buf4);
			fprintf (fp, "\t%s\t%s\t%s\t%s\n",
				 buf1, buf2, buf3, buf4);
		}
	}

	if (version >= GST_PVAL_SAVE_FORMAT_VERSION3) {
		/* Print the terminal/Steiner flag for each vertex. */
		j = 0;
		for (i = 0; i < n; i++) {
			if (j EQ 0) {
				fprintf (fp, "\t");
			}
			fprintf (fp, " %d", cip -> tflag [i]);
			if (++j >= 10) {
				fprintf (fp, "\n");
				j = 0;
			}
		}
		if (j > 0) {
			fprintf (fp, "\n");
		}
	}

	flist = NEWA (m, int);
	tmask = NEWA (kmasks, bitmap_t);
	for (i = 0; i < kmasks; i++) {
		tmask [i] = 0;
	}

	/* hyperedges... */
	for (i = 0; i < m; i++) {
		fprintf (fp, "\t%d\n", cip -> edge_size [i]);
		fprintf (fp, "\t");
		vp1 = cip -> edge [i];
		vp2 = cip -> edge [i + 1];
		col = 8;
		while (vp1 < vp2) {
			j = *vp1++;
			sprintf (buf1, "%d", j + 1);
			k = strlen (buf1);
			if (col + 1 + k >= MAXCOL) {
				fprintf (fp, "\n\t\t%s", buf1);
				col = 16 + k;
			}
			else if (col <= 8) {
				fprintf (fp, "%s", buf1);
				col += k;
			}
			else {
				fprintf (fp, " %s", buf1);
				col += (1 + k);
			}
		}
		fprintf (fp, "\n");

		_gst_dist_to_string (buf1, cip -> cost [i], cip -> scale);
		double_to_hex ((double) (cip -> cost [i]), buf2);
		fprintf (fp, "\t%s\t%s\n", buf1, buf2);

		if (geometric) {
			fsp = cip -> full_trees [i];
			terms = fsp -> terminals;
			steins = fsp -> steiners;

			if (steins EQ NULL) {
				fprintf (fp, "\t0\n");
			}
			else {
				fprintf (fp, "\t%d\n", steins -> n);
				for (j = 0; j < steins -> n; j++) {
					p1 = &(steins -> a [j]);
					_gst_coord_to_string (buf1,
							      p1 -> x,
							      cip -> scale);
					_gst_coord_to_string (buf2,
							      p1 -> y,
							      cip -> scale);
					double_to_hex ((double) (p1 -> x), buf3);
					double_to_hex ((double) (p1 -> y), buf4);
					fprintf (fp, "\t%s\t%s\t%s\t%s\n",
						 buf1, buf2, buf3, buf4);
				}
			}

			fprintf (fp, "\t%d\n", fsp -> nedges);
			for (j = 0; j < fsp -> nedges; j++) {
				k = fsp -> edges [j].p1;
				fprintf (fp, "\t\t%d",
					(k < terms -> n)
					    ? k + 1
					    : terms -> n - k - 1);
				k = fsp -> edges [j].p2;
				fprintf (fp, "\t%d\n",
					(k < terms -> n)
					    ? k + 1
					    : terms -> n - k - 1);
			}
		}

		if ((cip -> initial_edge_mask NE NULL) AND
		    (NOT BITON (cip -> initial_edge_mask, i))) {
			fprintf (fp, "\t0\n");	/* edge never needed */
		}
		else if ((cip -> required_edges NE NULL) AND
			 (BITON (cip -> required_edges, i))) {
			fprintf (fp, "\t2\n");	/* edge always needed */
		}
		else {
			fprintf (fp, "\t1\n");	/* edge sometimes needed */
		}

		/* List the hyperedges that are incompatible with this one. */
		if (cip -> inc_edges EQ NULL) {
			/* No incompatibility info to give! */
			fprintf (fp, "\t0\n");
		}
		else if ((cip -> initial_edge_mask NE NULL) AND
			 (NOT BITON (cip -> initial_edge_mask, i))) {
			/* This hyperedge was pruned. */
			/* Do not list all others as incompatible!!! */
			fprintf (fp, "\t0\n");
		}
		else {
			/* Gather the list of incompatible edges we	*/
			/* choose to mention.  (i.e., eliminate those	*/
			/* that are "basic" incompatibilities.)		*/
			vp1 = cip -> edge [i];
			vp2 = cip -> edge [i + 1];
			while (vp1 < vp2) {
				j = *vp1++;
				SETBIT (tmask, j);
			}
			ep1 = flist;
			ep2 = cip -> inc_edges [i];
			ep3 = cip -> inc_edges [i + 1];
			while (ep2 < ep3) {
				j = *ep2++;
				if (i EQ j) continue;
				if (NOT BITON (cip -> initial_edge_mask, j)) continue;
				count = 0;
				vp1 = cip -> edge [j];
				vp2 = cip -> edge [j + 1];
				while (vp1 < vp2) {
					k = *vp1++;
					if (BITON (tmask, k)) {
						++count;
					}
				}
				if (count >= 2) {
#if 1
					/* There shouldn't be any of	*/
					/* these in inc_edges any more!	*/
					FATAL_ERROR;
#else
					/* FST's i and j have >= 2 terms */
					/* in common -- don't list this */
					/* obvious incompatibility... */
					continue;
#endif
				}
				*ep1++ = j;
			}

			fprintf (fp, "\t%ld\n", (long) (ep1 - flist));
			if (ep1 > flist) {
				fprintf (fp, "\t");
				col = 8;
				for (ep2 = flist; ep2 < ep1; ep2++) {
					j = *ep2;
					sprintf (buf1, "%d", j + 1);
					k = strlen (buf1);
					if (col + 1 + k >= MAXCOL) {
						fprintf (fp, "\n\t\t%s", buf1);
						col = 16 + k;
					}
					else if (col <= 8) {
						fprintf (fp, "%s", buf1);
						col += (1 + k);
					}
					else {
						fprintf (fp, " %s", buf1);
						col += (1 + k);
					}
				}
				fprintf (fp, "\n");
			}
			vp1 = cip -> edge [i];
			vp2 = cip -> edge [i + 1];
			while (vp1 < vp2) {
				j = *vp1++;
				CLRBIT (tmask, j);
			}
		}

		if (version <= GST_PVAL_SAVE_FORMAT_VERSION2) {
			/* Number of strongly compatible full sets... */
			fprintf (fp, "\t0\n");
		}
	}

	free ((char *) tmask);
	free ((char *) flist);

#undef MAXCOL
}

/*
 * This routine converts a double into a printable ASCII string
 * that represents the exact numeric value in hexidecimal.  The
 * printable string has the following format:
 *
 *	[-].{hexdigits}[x[-]{hexdigits}]
 *
 * The leading minus sign is optional and the exponent part will be
 * left off if it is zero.
 */

	static
	void
double_to_hex (

double		value,		/* IN - the floating point value to encode */
char *		s		/* OUT - the hex ASCII string */
)
{
double		mant;
int		expon;
int		digit;
char *		p;
char		buf [12];

static const char	hex [] = "0123456789ABCDEF";

	if (value < 0) {
		*s++ = '-';
		value = - value;
	}

	*s++ = '.';

	mant = frexp (value, &expon);

	if (mant EQ 0.0) {
		*s++ = '0';
	}
	else {
		while (mant > 0.0) {
			mant *= 16.0;
			digit = (int) mant;
			*s++ = hex [digit];
			mant -= ((double) digit);
		}
		if (expon NE 0) {
			*s++ = 'x';
			if (expon < 0) {
				*s++ = '-';
				expon = - expon;
			}
			p = &buf [0];
			while (expon > 0) {
				*p++ = hex [expon & 0x0F];
				expon >>= 4;
			}
			while (p > &buf [0]) {
				*s++ = *--p;
			}
		}
	}

	*s++ = '\0';
}
