/***********************************************************************

	$Id: egmp.h,v 1.10 2022/11/19 13:45:51 warme Exp $

	File:	egmp.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2000, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Support routines for the EFST generator that use the
	GNU Multi-Precision arithmetic library (GMP -- if we have
	it) to compute certain items with high accuracy.

************************************************************************

	Modification Log:

	b-1:	11/25/2000	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
		: Add cost_extension for SteinLib "integer" format.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.

************************************************************************/

#ifndef EGMP_H
#define EGMP_H

#include "config.h"

#if defined(HAVE_GMP)

#include "logic.h"
#include <stdio.h>	/* Should include <stdio.h> before <gmp.h> */
#include <gmp.h>	/* to get gmp IO functions. */

struct gst_hypergraph;
struct gst_param;


struct einfo;
struct eqp_t;


typedef struct {			/* a + b * sqrt(3) */
	mpq_t		a;
	mpq_t		b;
} qr3_t;

struct qr3_point {
	qr3_t		x;
	qr3_t		y;
};


/*
 * Global Routines
 */

extern double	_gst_compute_EFST_length (struct einfo *	eip,
					  struct eqp_t *	eqpt);
extern void	_gst_qr3_clear (qr3_t * p);
extern void	_gst_qr3_init (qr3_t * p);
extern void	_gst_update_eqpoint_and_displacement (
					struct einfo *		eip,
					struct eqp_t *		eqpk);

extern struct cost_extension *
		_gst_egmp_graph_cost_extension (
					struct gst_hypergraph *	cip,
					struct gst_param *	params);

#endif

#endif
