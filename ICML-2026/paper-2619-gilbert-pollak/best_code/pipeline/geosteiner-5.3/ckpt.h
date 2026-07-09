/***********************************************************************

	$Id: ckpt.h,v 1.9 2022/11/19 13:45:50 warme Exp $

	File:	ckpt.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 1999, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	Routines for checkpointing and restarting the computation.

************************************************************************

	Modification Log:

	a-1:	03/08/99	warme
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files, apply prefixes.
		: Make features unconditional.

************************************************************************/

#ifndef CKPT_H
#define	CKPT_H

#include "gsttypes.h"

struct bbinfo;
struct gst_hypergraph;
struct gst_param;


extern bool		_gst_checkpoint_needed (struct bbinfo * bbip);
extern void		_gst_merge_constraints (struct bbinfo * bbip,
						char **		paths);
extern struct bbinfo *	_gst_restore_checkpoint (
					struct gst_hypergraph *	cip,
					struct gst_param *	params);
extern double *		_gst_restore_upper_bound_checkpoint (struct bbinfo * bbip);
extern void		_gst_write_checkpoint (struct bbinfo * bbip);
extern void		_gst_write_upper_bound_checkpoint (struct bbinfo * bbip);

#endif
