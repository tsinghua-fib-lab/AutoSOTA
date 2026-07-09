/***********************************************************************

	$Id: prepostlude.h,v 1.2 2022/11/19 13:45:54 warme Exp $

	File:	prepostlude.h
	Rev:	e-1
	Date:	09/24/2016

	Copyright (c) 1993, 2022 by David M. Warme.  This work is
	licensed under a Creative Commons Attribution-NonCommercial
	4.0 International License.

************************************************************************

	General declarations for the Steiner Tree program.

************************************************************************

	Modification Log:

	e-1:	09/24/2016	warme
		: Split off from steiner.h.

************************************************************************/

#ifndef	PREPOSTLUDE_H
#define	PREPOSTLUDE_H

#include "environment.h"
#include "fputils.h"
#include "gsttypes.h"
#include "logic.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * Macros used in the beginning and the end of all gst_-functions. They check
 * that the library is open. It also checks the FPU settings.
 */

#define GST_PRELUDE \
bool		restore_fp; \
struct fpsave	fpsave; \
	GST_PRELUDE_OPEN_CHECK \
	restore_fp = FALSE; \
	/* FIXME - this is not thread-safe. */ \
	if (NOT gst_env -> fp_saved) { \
		_gst_set_floating_point_configuration (&fpsave); \
		gst_env -> fp_saved = TRUE; \
		restore_fp = TRUE; \
	}

#define GST_PRELUDE_NO_FP \
bool		restore_fp; \
	GST_PRELUDE_OPEN_CHECK \
	restore_fp = FALSE;

#define GST_PRELUDE_OPEN_CHECK \
	if ((gst_env EQ NULL) OR (gst_env -> opencount EQ 0)) { \
		fprintf (stderr, "%s line %d: GeoSteiner is closed\n", \
			 __FILE__, __LINE__); \
		abort (); \
	}

#define GST_POSTLUDE \
	/* FIXME - this is not thread-safe. */ \
	if (restore_fp) { \
		_gst_restore_floating_point_configuration (&fpsave); \
		gst_env -> fp_saved = FALSE; \
	}

#endif
