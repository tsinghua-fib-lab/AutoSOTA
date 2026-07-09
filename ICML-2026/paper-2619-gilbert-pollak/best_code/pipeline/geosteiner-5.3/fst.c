/***********************************************************************

	$Id: fst.c,v 1.13 2022/11/19 13:45:52 warme Exp $

	File:	fst.c
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zachariasen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************

	General FST generator.

************************************************************************

	Modification Log:

	a-1:	08/05/2002	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#include "logic.h"
#include "geosteiner.h"
#include "metric.h"
#include "prepostlude.h"
#include "steiner.h"
#include "ufst.h"

/*
 * Global functions
 */

gst_hg_ptr	gst_generate_fsts (int,
				   double *,
				   gst_metric_ptr,
				   struct gst_param *,
				   int *);

/*
 * A general FST generator...
 */

	gst_hg_ptr
gst_generate_fsts (

int			nterms,
double *		terminals,
gst_metric_ptr		metric,
gst_param_ptr		params,
int *			status
)
{
gst_hg_ptr	H;

	GST_PRELUDE

	H = NULL;

	if (metric EQ NULL) {
		if (status NE NULL) {
			*status = GST_ERR_INVALID_METRIC;
		}
	}
	else {
		switch (metric -> type) {
		case GST_METRIC_L:
			if (metric -> parameter EQ 1) {
				H = gst_generate_rfsts (nterms, terminals,
							params, status);
			}
			else if (metric -> parameter EQ 2) {
				H = gst_generate_efsts (nterms, terminals,
							params, status);
			}
			else if (status NE NULL) {
				*status = GST_ERR_INVALID_METRIC;
			}
			break;

		case GST_METRIC_UNIFORM:
			H = gst_generate_ufsts (nterms, terminals, metric -> parameter,
					    params, status);
			break;

		case GST_METRIC_NONE:
		default:
			if (status NE NULL) {
				*status = GST_ERR_INVALID_METRIC;
			}
			break;
		}
	}

	GST_POSTLUDE

	return H;
}
