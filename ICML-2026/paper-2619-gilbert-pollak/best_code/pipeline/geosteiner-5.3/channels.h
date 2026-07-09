/***********************************************************************

	$Id: channels.h,v 1.11 2022/11/19 13:45:50 warme Exp $

	File:	channels.h
	Rev:	e-3
	Date:	09/24/2016

	Copyright (c) 2002, 2022 by Pawel Winter, Martin Zacharisen.
	This work is licensed under a Creative Commons
	Attribution-NonCommercial 4.0 International License.

************************************************************************


************************************************************************

	Modification Log:

	a-1:	04/30/02	benny
		: Created.
	e-1:	04/14/2015	warme
		: Changes for 5.0 release.
	e-2:	09/05/2016	warme
		: Change notices for 5.1 release.
	e-3:	09/24/2016	warme
		: Reorganize include files.

************************************************************************/

#ifndef CHANNELS_H
#define CHANNELS_H

#include "geosteiner.h"

/*
 * Global Types
 */

struct gst_channel {
	gst_channel_options	options;
	gst_dest_ptr		head;
};

/*
 * States
 */

#define	GST_CHSTATE_		0x01


/*
 * Macros
 */

#define INDENT(a) \
	{ if ((a) NE NULL) { \
		++((a) -> options.indent); \
	  } \
	}

#define UNINDENT(a) \
	{ if ((a) NE NULL) { \
		--((a) -> options.indent); \
	  } \
	}

#endif
