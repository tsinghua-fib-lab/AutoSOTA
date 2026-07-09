/***********************************************************************

	$Id: costextension.h,v 1.4 2022/11/19 13:45:50 warme Exp $

	File:	costextension.h
	Rev:	e-2
	Date:	09/24/2016

	Copyright (c) 1999, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	An "abstract base class" for encapsulating specialized
	edge cost information for a hypergraph.

************************************************************************

	Modification Log:

	e-1:	09/05/2016	warme
		: Created.
	e-2:	09/24/2016	warme
		: Reorganized include files.

************************************************************************/

#ifndef COSTEXTENSION_H
#define	COSTEXTENSION_H

#include "gsttypes.h"
#include "logic.h"
#include <stdio.h>

/*
 * An "abstract base class" for encapsulating specialized cost information
 * for a hypergraph.
 *
 * The only member is a "virtual table pointer".
 */

struct cost_extension {
	const struct cost_ext_vtab *	vtab;	/* Virtual table pointer */
};

/*
 * The virtual table for "struct cost_extension" objects.
 */

struct cost_ext_vtab {
	/* Virtual destructor, but also frees the storage for the	*/
	/* cost_extension object itself, so these *MUST* be dynamically	*/
	/* allocated.							*/
	void	(*free) (struct cost_extension * extp);

	/* Write the "special" cost for edge i to the given file.  This	*/
	/* is an ASCII, human readable format.				*/
	void	(*write_edge_cost) (struct cost_extension *	extp,
				    FILE *			fp,
				    int				i);
};

	static
	inline
	void
_gst_hg_cost_extension_free (

struct cost_extension *		extp	/* IN: cost_extension to free */
)
{
	if ((extp NE NULL) AND
	    (extp -> vtab NE NULL) AND
	    (extp -> vtab -> free NE NULL)) {
		(*(extp -> vtab -> free)) (extp);
	}
}

	static
	inline
	bool
_gst_hg_cost_extension_have_write_edge_cost (

struct cost_extension *		extp	/* IN: cost_extension to acces */
)
{
	if (extp EQ NULL) return (FALSE);

	if (extp -> vtab EQ NULL) return (FALSE);

	return (extp -> vtab -> write_edge_cost NE NULL);
}

	static
	inline
	void
_gst_hg_cost_extension_write_edge_cost (

struct cost_extension *		extp,	/* IN: cost_extension to acces */
FILE *				fp,	/* IN: file to write edge cost into */
int				i	/* IN: which edge to write cost for */
)
{
	if (_gst_hg_cost_extension_have_write_edge_cost (extp)) {
		(*(extp -> vtab -> write_edge_cost)) (extp, fp, i);
	}
}

#endif
